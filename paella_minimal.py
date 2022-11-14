import math
import os
import torch
from torch import nn, optim
import torchvision
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from modules import DenoiseUNet
from utils import get_dataloader, sample, encode, decode
import open_clip
from open_clip import tokenizer
from rudalle import get_vae


def train(proc_id, args):
    parallel = len(args.devices) > 1
    device = torch.device(proc_id)

    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend="nccl", init_method="file://dist_file", world_size=args.n_nodes * len(args.devices), rank=proc_id + len(args.devices) * args.node_id)
        torch.set_num_threads(6)

    model = DenoiseUNet(num_labels=args.num_codebook_vectors, c_clip=1024).to(device)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    del clip_model.visual
    clip_model = clip_model.to(device).eval().requires_grad_(False)

    lr = 3e-4
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_stepss=args.total_steps, max_lr=lr, pct_start=0.1 if not args.finetune else 0.0, div_factor=25, final_div_factor=1 / 25, anneal_strategy='linear')

    losses, accuracies = [], []
    start_step, total_loss, total_acc = 0, 0, 0

    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    pbar = tqdm(enumerate(dataset, start=start_step), total=args.total_steps, initial=start_step) if args.node_id == 0 and proc_id == 0 else enumerate(dataset, start=start_step)
    model.train()
    for step, (images, captions) in pbar:
        images = images.to(device)
        with torch.no_grad():
            image_indices = encode(vqmodel, images)  # encode images (batch_size x 3 x 256 x 256) to tokens (batch_size x 32 x 32)
            r = torch.rand(images.size(0), device=device)  # generate random timesteps
            noised_indices, mask = model.module.add_noise(image_indices, r)  # noise the tokens according to the timesteps

            if np.random.rand() < 0.1:  # 10% of the times -> unconditional training for classifier-free-guidance
                text_embeddings = images.new_zeros(images.size(0), 1024)
            else:
                text_tokens = tokenizer.tokenize(captions)
                text_tokens = text_tokens.to(device)
                text_embeddings = clip_model.encode_text(text_tokens).float()  # text embeddings (batch_size x 1024)

        pred = model(noised_indices, text_embeddings, r)  # predict denoised tokens (batch_size x 32 x 32 x 8192
        loss = criterion(pred, image_indices)  # cross entropy loss
        loss_adjusted = loss / args.accum_grad

        loss_adjusted.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5).item()
        if (step + 1) % args.accum_grad == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        acc = (pred.argmax(1) == image_indices).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()

        if not proc_id and args.node_id == 0:
            pbar.set_postfix({"loss": total_loss / (step + 1), "acc": total_acc / (step + 1), "curr_loss": loss.item(), "curr_acc": acc.item(), "ppx": np.exp(total_loss / (step + 1)), "lr": optimizer.param_groups[0]['lr'], "grad_norm": grad_norm})

        if args.node_id == 0 and proc_id == 0 and step % args.log_period == 0:
            print(f"Step {step} - loss {total_loss / (step + 1)} - acc {total_acc / (step + 1)} - ppx {np.exp(total_loss / (step + 1))}")

            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                sampled = sample(model.module, c=text_embeddings)[-1]
                sampled = decode(vqmodel, sampled)

            model.train()
            log_images = torch.cat([torch.cat([i for i in sampled.cpu()], dim=-1)], dim=-2)
            torchvision.utils.save_image(log_images, os.path.join(f"results/{args.run_name}", f"{step:03d}.png"))

            del sampled

            torch.save(model.module.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save({'step': step, 'losses': losses, 'accuracies': accuracies}, f"results/{args.run_name}/log.pt")

        del images, image_indices, r, text_embeddings
        del noised_indices, mask, pred, loss, loss_adjusted, acc


def launch(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in args.devices])
    if len(args.devices) == 1:
        train(0, args)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "33751"
        p = mp.spawn(train, nprocs=len(args.devices), args=(args,))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Paella_f8_8192"
    args.dataset_type = "webdataset"
    args.total_steps = 501_000
    args.batch_size = 22
    args.image_size = 256
    args.num_workers = 10
    args.log_period = 5000
    args.accum_grad = 1
    args.num_codebook_vectors = 8192

    args.n_nodes = 8
    args.node_id = int(os.environ["SLURM_PROCID"])
    args.devices = [0, 1, 2, 3, 4, 5, 6, 7]

    args.dataset_path = ""
    print("Launching with args: ", args)
    launch(
        args
    )
