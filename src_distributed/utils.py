import os
import json
import torch
import open_clip
import torchvision
import webdataset as wds
from src.vqgan import VQModel
from torch.utils.data import DataLoader
from torch.distributed import init_process_group
from webdataset.handlers import warn_and_continue
from transformers import T5EncoderModel, AutoTokenizer

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(256),
])


class WebdatasetFilter():
    def __init__(self, min_size=512, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99, text_conditions=None):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.aesthetic_threshold = aesthetic_threshold
        self.unsafe_threshold = unsafe_threshold
        self.text_conditions = text_conditions

    def __call__(self, x):
        try:
            if 'json' in x:
                x_json = json.loads(x['json'])
                filter_size = (x_json.get('original_width', 0.0) or 0.0) >= self.min_size and x_json.get(
                    'original_height', 0) >= self.min_size
                filter_watermark = (x_json.get('pwatermark', 1.0) or 1.0) <= self.max_pwatermark
                filter_aesthetic_a = (x_json.get('aesthetic', 0.0) or 0.0) >= self.aesthetic_threshold
                filter_aesthetic_b = (x_json.get('AESTHETIC_SCORE', 0.0) or 0.0) >= self.aesthetic_threshold
                filter_unsafe = (x_json.get('punsafe', 1.0) or 1.0) <= self.unsafe_threshold
                if self.text_conditions is not None:
                    caption = x['txt'].decode("utf-8")
                    filter_min_words = len(caption.split(" ")) >= self.text_conditions['min_words']
                    filter_ord_128 = all([ord(c) < 128 for c in caption])
                    filter_forbidden_words = all(
                        [c not in caption.lower() for c in self.text_conditions['forbidden_words']])
                    filter_text = filter_min_words and filter_ord_128 and filter_forbidden_words
                else:
                    filter_text = True
                return filter_size and filter_watermark and (
                            filter_aesthetic_a or filter_aesthetic_b) and filter_unsafe and filter_text
            else:
                return False
        except:
            return False


def get_dataloader(dataset_path, batch_size):
    dataset = wds.WebDataset(dataset_path, resampled=True, handler=warn_and_continue) \
        .select(WebdatasetFilter(min_size=256, max_pwatermark=0.5, aesthetic_threshold=5.0, unsafe_threshold=0.99)) \
        .shuffle(690, handler=warn_and_continue) \
        .decode("pilrgb", handler=warn_and_continue) \
        .to_tuple("jpg", "txt", handler=warn_and_continue) \
        .map_tuple(transforms, lambda x: x, handler=warn_and_continue)
    return DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)


def load_conditional_models(clip_model_name, byt5_model_name, vqgan_path, device):
    vqgan = VQModel().to(device)
    vqgan.load_state_dict(torch.load(vqgan_path, map_location=device)['state_dict'])
    vqgan.eval().requires_grad_(False)

    byt5 = T5EncoderModel.from_pretrained(byt5_model_name).to(device).eval().requires_grad_(False)
    byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_model_name)

    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name[0], pretrained=clip_model_name[1], device=device)
    clip_model.to(device).eval().requires_grad_(False)
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name[0])

    clip_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    return vqgan, (clip_tokenizer, clip_model, clip_preprocess), (byt5_tokenizer, byt5)


def ddp_setup(rank, world_size, n_node, node_id):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "33751"
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rank+node_id*world_size, world_size=world_size*n_node,
        init_method="file:///dist_file",
    )
    print(f"[GPU {rank+node_id*world_size}] READY")


def sample(model, model_inputs, unconditional_inputs, latent_shape, init_x=None, steps=12, renoise_steps=None, temperature = (0.7, 0.3), cfg=(8.0, 8.0), t_start=1.0, t_end=0.0, sampling_conditional_steps=None):
    device = unconditional_inputs["byt5"].device
    if sampling_conditional_steps is None:
        sampling_conditional_steps = steps
    if renoise_steps is None:
        renoise_steps = steps-1
    with torch.inference_mode():
        init_noise = torch.randint(0, model.num_labels, size=latent_shape, device=device)
        if init_x != None:
            sampled = init_x
        else:
            sampled = init_noise.clone()
        t_list = torch.linspace(t_start, t_end, steps+1)
        temperatures = torch.linspace(temperature[0], temperature[1], steps)
        cfgs = torch.linspace(cfg[0], cfg[1], steps)
        for i, tv in enumerate(t_list[:steps]):
            t = torch.ones(latent_shape[0], device=device) * tv

            logits = model(sampled, t, **model_inputs)
            if cfg is not None and i < sampling_conditional_steps:
                logits = logits * cfgs[i] + model(sampled, t, **unconditional_inputs) * (1-cfgs[i])
            scores = logits.div(temperatures[i]).softmax(dim=1)

            sampled = scores.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
            sampled = torch.multinomial(sampled, 1)[:, 0].view(logits.size(0), *logits.shape[2:])

            if i < renoise_steps:
                t_next = torch.ones(latent_shape[0], device=device) * t_list[i+1]
                sampled = model.add_noise(sampled, t_next, random_x=init_noise)[0]
    return sampled