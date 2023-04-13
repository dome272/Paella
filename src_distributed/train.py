import os
import torch
import wandb
import numpy as np
import torchvision
from tqdm import tqdm
from modules import Paella
from torch import nn, optim
import torch.multiprocessing as mp
from warmup_scheduler import GradualWarmupScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_dataloader, load_conditional_models, ddp_setup, sample

updates = 1500000
warmup_updates = 10000
batch_size = 2048
grad_accum_steps = 1
max_iters = updates * grad_accum_steps
print_every = 2000 * grad_accum_steps
lr = 1e-4

dataset_path = ""
clip_model_name = ('ViT-H-14', 'laion2b_s32b_b79k')
byt5_model_name = "google/byt5-xl"
vqmodel_path = ""
run_name = "Paella-ByT5-XL-v1"
output_path = "output"
os.makedirs(output_path, exist_ok=True)
checkpoint_path = f"{run_name}.pt"
wandv_project, wandb_run_name, wandv_entity = "", "", ""


def train(gpu_id, world_size, n_nodes):
    node_id = int(os.environ["SLURM_PROCID"])
    main_node = gpu_id == 0 and node_id == 0
    ddp_setup(gpu_id, world_size, n_nodes, node_id)
    device = torch.device(gpu_id)

    local_batch_size = batch_size//(world_size*n_nodes*grad_accum_steps)
    dataloader = get_dataloader(dataset_path, batch_size=local_batch_size)
    checkpoint = torch.load(checkpoint_path, map_location=device) if os.path.exists(checkpoint_path) else None

    if main_node:
        print("BATCH SIZE / DEVICE:", local_batch_size)
        run_id = checkpoint['wandb_run_id'] if checkpoint is not None else wandb.util.generate_id()
        wandb.init(project=wandv_project, name=wandb_run_name, entity=wandv_entity, id=run_id, resume="allow")

    model = Paella(byt5_embd=2560).to(device)
    vqgan, (clip_tokenizer, clip_model, clip_preprocess), (byt5_tokenizer, byt5) = load_conditional_models(clip_model_name, byt5_model_name, vqmodel_path, device)
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])

    model = DDP(model, device_ids=[gpu_id], output_device=device, find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_updates)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    scaler = torch.cuda.amp.GradScaler()

    start_iter = 1
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.last_epoch = checkpoint['scheduler_last_step']
        start_iter = checkpoint['scheduler_last_step']*grad_accum_steps + 1
        if 'grad_scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        del checkpoint
        torch.cuda.empty_cache()

    grad_norm = torch.tensor(0, device=device)
    dataloader_iterator = iter(dataloader)
    pbar = tqdm(range(start_iter, max_iters+1)) if main_node else range(start_iter, max_iters+1)
    model.train()
    for it in pbar:
        images, captions = next(dataloader_iterator)
        images = images.to(device)

        with torch.no_grad():
            if np.random.rand() < 0.05:
                byt5_captions = [''] * len(captions)
            else:
                byt5_captions = captions
            byt5_tokens = byt5_tokenizer(byt5_captions, padding="longest", return_tensors="pt", max_length=768, truncation=True).input_ids.to(device)
            byt_embeddings = byt5(input_ids=byt5_tokens).last_hidden_state

            with torch.cuda.amp.autocast():
                if np.random.rand() < 0.9:
                    clip_captions = [''] * len(captions)
                else:
                    clip_captions = captions
                clip_tokens = clip_tokenizer(clip_captions).to(device)
                clip_embeddings = clip_model.encode_text(clip_tokens).float()
 
                if np.random.rand() < 0.9:
                    clip_image_embeddings = None
                else:
                    clip_image_embeddings = clip_model.encode_image(clip_preprocess(images)).float()

            t = (1-torch.rand(images.size(0), device=device)).add(0.001).clamp(0.001, 1.0)
            latents = vqgan.encode(images)[2]
            noised_latents, mask = model.module.add_noise(latents, t)
            loss_weight = model.module.get_loss_weight(t, mask)

        with torch.cuda.amp.autocast():
            pred = model(noised_latents, t, byt_embeddings, clip_embeddings, clip_image_embeddings)
            loss = criterion(pred, latents)
            loss = ((loss * loss_weight).sum(dim=[1, 2]) / loss_weight.sum(dim=[1, 2])).mean()
            loss_adjusted = loss / grad_accum_steps
        
        acc = (pred.argmax(1) == latents).float()
        acc = acc.mean()

        if it % grad_accum_steps == 0 or it == max_iters:
            scaler.scale(loss_adjusted).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            with model.no_sync():
                scaler.scale(loss_adjusted).backward()

        if main_node:
            pbar.set_postfix({'bs': images.size(0), 'loss': loss_adjusted.item(), 'acc': acc.item(), 'grad_norm': grad_norm.item(), 'lr': optimizer.param_groups[0]['lr'], 'total_steps': scheduler.last_epoch})

        if main_node and (it == 1 or it % print_every == 0 or it == max_iters):
            print(f"ITER {it}/{max_iters} - loss {loss_adjusted}")

            torch.save({
                'iter': it,
                'wandb_run_id': run_id,
                'state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_last_step': scheduler.last_epoch,
                'grad_scaler_state_dict': scaler.state_dict(),
            }, checkpoint_path)

            model.eval()
            images, captions = next(dataloader_iterator)
            with torch.no_grad():
                byt5_tokens = byt5_tokenizer(captions, padding="longest", return_tensors="pt", max_length=768, truncation=True).input_ids.to(device)
                byt_embeddings = byt5(input_ids=byt5_tokens).last_hidden_state
                byt5_tokens_uncond = byt5_tokenizer([''] * len(captions), padding="longest", return_tensors="pt", max_length=768, truncation=True).input_ids.to(device)
                byt_embeddings_uncond = byt5(input_ids=byt5_tokens_uncond).last_hidden_state

                clip_tokens = clip_tokenizer(captions).to(device)
                clip_embeddings = clip_model.encode_text(clip_tokens).float()
                clip_tokens_uncond = clip_tokenizer([''] * len(captions)).to(device)
                clip_embeddings_uncond = clip_model.encode_text(clip_tokens_uncond).float()
                clip_image_embeddings = clip_model.encode_image(clip_preprocess(images)).float()

                pred = model(noised_latents, t, byt_embeddings, clip_embeddings, clip_image_embeddings)
                pred_tokens = pred.div(0.1).softmax(dim=1).permute(0, 2, 3, 1) @ vqgan.vquantizer.codebook.weight.data
                pred_tokens = vqgan.vquantizer.forward(pred_tokens, dim=-1)[-1]

                sampled = sample(model.module,
                                 {'byt5': byt_embeddings, 'clip': clip_embeddings, 'clip_image': clip_image_embeddings},
                                 {'byt5': byt_embeddings_uncond, 'clip': clip_embeddings_uncond, 'clip_image': None},
                                 (byt_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4))

                sampled_noimg = sample(model.module,
                                       {'byt5': byt_embeddings, 'clip': clip_embeddings, 'clip_image': None},
                                       {'byt5': byt_embeddings_uncond, 'clip': clip_embeddings_uncond, 'clip_image': None},
                                       (byt_embeddings.size(0), images.size(-2) // 4, images.size(-1) // 4))

                noised_images = vqgan.decode_indices(noised_latents).clamp(0, 1)
                pred_images = vqgan.decode_indices(pred_tokens).clamp(0, 1)
                sampled_images = vqgan.decode_indices(sampled).clamp(0, 1)
                sampled_images_noimg = vqgan.decode_indices(sampled_noimg).clamp(0, 1)
            model.train()

            torchvision.utils.save_image(torch.cat([
                torch.cat([i for i in images.cpu()], dim=-1),
                torch.cat([i for i in noised_images.cpu()], dim=-1),
                torch.cat([i for i in pred_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images.cpu()], dim=-1),
                torch.cat([i for i in sampled_images_noimg.cpu()], dim=-1),
            ], dim=-2), f'{output_path}{it:06d}.jpg')

            log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_noimg[i])] + [wandb.Image(images[i])] for i in range(len(images))]
            wandb.log({"Log": wandb.Table(data=log_data, columns=["Captions", "Sampled", "Sampled NoImg", "Orig"])})


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    n_node = 16
    mp.spawn(train, args=(world_size, n_node), nprocs=world_size)
    