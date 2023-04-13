import torch
import torchvision
from vqgan import VQModel
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, AutoTokenizer

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(256),
])


class YOUR_DATASET(Dataset):
    def __init__(self, dataset_path):
        pass


def get_dataloader(dataset_path, batch_size):
    dataset = YOUR_DATASET(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)


def load_conditional_models(byt5_model_name, vqgan_path, device):
    vqgan = VQModel().to(device)
    vqgan.load_state_dict(torch.load(vqgan_path, map_location=device)['state_dict'])
    vqgan.eval().requires_grad_(False)

    byt5 = T5EncoderModel.from_pretrained(byt5_model_name).to(device).eval().requires_grad_(False)
    byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_model_name)

    return vqgan, (byt5_tokenizer, byt5)


def sample(model, model_inputs, latent_shape, unconditional_inputs=None, steps=12, renoise_steps=11, temperature=(1.0, 0.2), cfg=8.0, t_start=1.0, t_end=0.0, device="cuda"):
    with torch.inference_mode():
        sampled = torch.randint(0, model.num_labels, size=latent_shape, device=device)
        init_noise = sampled.clone()
        t_list = torch.linspace(t_start, t_end, steps+1)
        temperatures = torch.linspace(temperature[0], temperature[1], steps)
        for i, t in enumerate(t_list[:steps]):
            t = torch.ones(latent_shape[0], device=device) * t

            logits = model(sampled, t, **model_inputs)
            if cfg:
                logits = logits * cfg + model(sampled, t, **unconditional_inputs) * (1-cfg)
            scores = logits.div(temperatures[i]).softmax(dim=1)

            sampled = scores.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
            sampled = torch.multinomial(sampled, 1)[:, 0].view(logits.size(0), *logits.shape[2:])

            if i < renoise_steps:
                t_next = torch.ones(latent_shape[0], device=device) * t_list[i+1]
                sampled = model.add_noise(sampled, t_next, random_x=init_noise)[0]
    return sampled
