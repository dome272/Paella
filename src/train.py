import os
import torch
import numpy as np
from tqdm import tqdm
from modules import Paella
from torch import nn, optim
from warmup_scheduler import GradualWarmupScheduler
from utils import get_dataloader, load_conditional_models

steps = 100_000
warmup_updates = 10000
batch_size = 16
checkpoint_frequency = 2000
lr = 1e-4

train_device = "cuda"
dataset_path = ""
byt5_model_name = "google/byt5-xl"
vqmodel_path = ""
run_name = "Paella-ByT5-XL-v1"
output_path = "output"
checkpoint_path = f"{run_name}.pt"


def train():
    os.makedirs(output_path, exist_ok=True)
    device = torch.device(train_device)

    dataloader = get_dataloader(dataset_path, batch_size=batch_size)
    checkpoint = torch.load(checkpoint_path, map_location=device) if os.path.exists(checkpoint_path) else None

    model = Paella(byt5_embd=2560).to(device)
    vqgan, (byt5_tokenizer, byt5) = load_conditional_models(byt5_model_name, vqmodel_path, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_updates)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')

    start_iter = 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.last_epoch = checkpoint['scheduler_last_step']
        start_iter = checkpoint['scheduler_last_step'] + 1
        del checkpoint

    pbar = tqdm(range(start_iter, steps+1))
    model.train()
    for i, (images, captions) in enumerate(dataloader):
        images = images.to(device)

        with torch.no_grad():
            if np.random.rand() < 0.05:
                byt5_captions = [''] * len(captions)
            else:
                byt5_captions = captions
            byt5_tokens = byt5_tokenizer(byt5_captions, padding="longest", return_tensors="pt", max_length=768, truncation=True).input_ids.to(device)
            byt_embeddings = byt5(input_ids=byt5_tokens).last_hidden_state

            t = (1-torch.rand(images.size(0), device=device))
            latents = vqgan.encode(images)[2]
            noised_latents, _ = model.add_noise(latents, t)

        pred = model(noised_latents, t, byt_embeddings)
        loss = criterion(pred, latents)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        optimizer.zero_grad()

        acc = (pred.argmax(1) == latents).float().mean()

        pbar.set_postfix({'bs': images.size(0), 'loss': loss.item(), 'acc': acc.item(), 'grad_norm': grad_norm.item(), 'lr': optimizer.param_groups[0]['lr'], 'total_steps': scheduler.last_epoch})

        if i % checkpoint_frequency == 0:
            torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_last_step': scheduler.last_epoch, 'iter' : i}, checkpoint_path)


if __name__ == '__main__':
    train()