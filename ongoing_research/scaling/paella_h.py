import os
import torch
import wandb
from torch import nn, optim
import torchvision
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from modules import DenoiseGIC, DenoiseUNet
from utils import get_dataloader, encode, decode, sample
import open_clip
from rudalle import get_vae


def train(proc_id, args):
    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True
    else:
        resume = False

    parallel = len(args.devices) > 1
    device = torch.device(proc_id)

    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend="nccl", init_method="file:///fsx/mas/paella_unet/dist_file2",
                                world_size=args.n_nodes * len(args.devices),
                                rank=proc_id + len(args.devices) * args.node_id)
        torch.set_num_threads(6)

    if args.model == "GIC":
        print(f"Model: DenoiseGIC")
        model = DenoiseGIC(num_labels=args.num_codebook_vectors, layers=36, c_hidden=1280).to(device)
    elif args.model == "UNet":
        model = DenoiseUNet(num_labels=args.num_codebook_vectors, c_clip=1024, c_hidden=1280, down_levels=[1, 2, 8, 32], up_levels=[32, 8, 2, 1]).to(device)
    else:
        raise NotImplementedError()

    if not proc_id and args.node_id == 0:
        print(f"Starting run '{args.run_name}'....")
        print(f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}")
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir="/fsx/mas/.cache")
    clip_model = clip_model.to(device).eval().requires_grad_(False)
    clip_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    lr = 1e-4
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    grad_accum_steps = args.accum_grad
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=args.total_steps, pct_start=0.03,
                                              div_factor=25, final_div_factor=1 / 25, anneal_strategy='linear')

    if resume:
        if not proc_id and args.node_id == 0:
            print("Loading last checkpoint....")
        logs = torch.load(f"results/{args.run_name}/log.pt")
        run_id = logs["wandb_run_id"]
        start_step = logs["step"] + 1
        losses = logs["losses"]
        accuracies = logs["accuracies"]
        total_loss, total_acc = losses[-1] * start_step, accuracies[-1] * start_step
        model.load_state_dict(torch.load(f"models/{args.run_name}/model.pt", map_location=device))
        if not proc_id and args.node_id == 0:
            print("Loaded model.")
        opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        last_lr = opt_state["param_groups"][0]["lr"]
        with torch.no_grad():
            for _ in range(logs["step"]):
                scheduler.step()
        if not proc_id and args.node_id == 0:
            print(f"Initialized scheduler")
            print(f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} -> {last_lr == optimizer.param_groups[0]['lr']}")
        optimizer.load_state_dict(opt_state)
        del opt_state
    else:
        run_id = wandb.util.generate_id()
        losses = []
        accuracies = []
        start_step, total_loss, total_acc = 0, 0, 0

    if not proc_id and args.node_id == 0:
        wandb.init(project="DenoiseGIT", name=args.run_name, entity="wand-tech", config=vars(args), id=run_id,
                   resume="allow")
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)
        wandb.watch(model)

    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    pbar = tqdm(enumerate(dataset, start=start_step), total=args.total_steps, initial=start_step) if args.node_id == 0 and proc_id == 0 else enumerate(dataset, start=start_step)
    model.train()
    for step, (images, captions) in pbar:
        images = images.to(device)
        with torch.no_grad():
            image_indices = encode(vqmodel, images)
            r = torch.rand(images.size(0), device=device)
            noised_indices, mask = model.module.add_noise(image_indices, r)

            if np.random.rand() < 0.1:  # 10% of the times...
                img_embeddings = images.new_zeros(images.size(0), 1024)
            else:
                img_embeddings = clip_model.encode_image(clip_preprocess(images)).float()

        pred = model(noised_indices, img_embeddings, r)
        loss = criterion(pred, image_indices)
        loss_adjusted = loss / grad_accum_steps

        loss_adjusted.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1).item()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        acc = (pred.argmax(1) == image_indices).float()
        acc = acc.mean()

        total_loss += loss.item()
        total_acc += acc.item()

        if not proc_id and args.node_id == 0:
            pbar.set_postfix({
                'loss': total_loss / (step + 1),
                'acc': total_acc / (step + 1),
                'ppx': np.exp(total_loss / (step + 1)),
                'lr': optimizer.param_groups[0]['lr'],
                'gn': grad_norm
            })
            wandb.log({
                "loss": total_loss / (step + 1),
                "acc": total_acc / (step + 1),
                "curr_loss": loss.item(),
                "curr_acc": acc.item(),
                "ppx": np.exp(total_loss / (step + 1)),
                "lr": optimizer.param_groups[0]['lr'],
                "grad_norm": grad_norm
            })

        if args.node_id == 0 and proc_id == 0 and step % args.log_period == 0:
            print(f"Step {step} - loss {total_loss / (step + 1)} - acc {total_acc / (step + 1)} - ppx {np.exp(total_loss / (step + 1))}")

            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                sampled = sample(model.module, c=img_embeddings)
                sampled = decode(vqmodel, sampled)
                recon_images = decode(vqmodel, image_indices)

                log_images = torch.cat([
                    torch.cat([i for i in sampled.cpu()], dim=-1),
                ], dim=-2)

            model.train()

            torchvision.utils.save_image(log_images, os.path.join(f"results/{args.run_name}", f"{step:03d}.png"))

            log_data = [[captions[i]] + [wandb.Image(sampled[i])] + [wandb.Image(images[i])] + [wandb.Image(recon_images[i])] for i in range(len(captions))]
            log_table = wandb.Table(data=log_data, columns=["Caption", "Image", "Orig", "Recon"])
            wandb.log({"Log": log_table})

            del sampled, log_data

            if step % args.extra_ckpt == 0:
                torch.save(model.module.state_dict(), f"models/{args.run_name}/model_{step}.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/model_{step}_optim.pt")
                torch.save(grad_scaler.state_dict(), f"models/{args.run_name}/model_{step}_scaler.pt")
            torch.save(model.module.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save(grad_scaler.state_dict(), f"models/{args.run_name}/scaler.pt")
            torch.save({'step': step, 'losses': losses, 'accuracies': accuracies, 'wandb_run_id': run_id}, f"results/{args.run_name}/log.pt")

        del images, image_indices, r, img_embeddings
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
    args.run_name = "Paella_1B_img"
    args.model = "UNet"
    args.dataset_type = "webdataset"
    args.total_steps = 2_001_000
    args.batch_size = 16
    args.image_size = 512
    args.num_workers = 10
    args.log_period = 2000
    args.extra_ckpt = 50_000
    args.accum_grad = 1
    args.num_codebook_vectors = 8192  # 1024
    args.log_captions = False

    args.n_nodes = 16
    args.node_id = int(os.environ["SLURM_PROCID"])
    args.devices = [0, 1, 2, 3, 4, 5, 6, 7]

    # args.dataset_path = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
    # args.dataset_path = "pipe:aws s3 cp s3://deep-floyd-s3/datasets/{laion_cleaned-part1/{00000..79752}.tar,laion_cleaned-part2/{00000..94330}.tar,laion_cleaned-part3/{00000..94336}.tar,laion_cleaned-part4/{00000..94340}.tar,laion_cleaned-part5/{00000..94333}.tar,laion_cleaned-part6/{00000..77178}.tar} -"
    args.dataset_path = "pipe:aws s3 cp s3://stability-aws/laion-a-native/{part-0/{00000..18699}.tar,part-1/{00000..18699}.tar,part-2/{00000..18699}.tar,part-3/{00000..18699}.tar,part-4/{00000..18699}.tar} -"
    # args.dataset_path = "pipe:aws s3 cp s3://s-datasets/laion-high-resolution/{00000..17535}.tar -"
    print("Launching with args: ", args)
    launch(
        args
    )
