from math import sqrt
import typing
import torchvision
from torchvision.utils import save_image
from PIL import Image
from Replicate_demos.common import sample, decode, encode
from modules import DenoiseUNet
import open_clip
from open_clip import tokenizer
from rudalle import get_vae
import torch
from einops import rearrange

from cog import BasePredictor, Path, Input

device = torch.device("cuda:0")


class Predictor(BasePredictor):
    def setup(self):
        self.vqmodel = get_vae(cache_dir="./").to(device)
        self.vqmodel.eval().requires_grad_(False)

        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k',
                                                                 cache_dir="./")
        self.clip_model = clip_model.to(device).eval().requires_grad_(False)

        state_dict = torch.load("./model_600000.pt", map_location=device)
        self.model = DenoiseUNet(num_labels=8192).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval().requires_grad_()

    def predict(self, **kwargs):
        raise NotImplemented


class Text2ImagePredictor(Predictor):
    def predict(
            self,
            prompt: str = Input(default="Highly detailed photograph of darth vader. artstation"),
            num_outputs: int = Input(default=1),
    ) -> typing.List[Path]:
        prompt = str(prompt)
        num_outputs = int(num_outputs)
        latent_shape = (32, 32)
        tokenized_text = tokenizer.tokenize([prompt] * num_outputs).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                clip_embeddings = self.clip_model.encode_text(tokenized_text)
                sampled = sample(self.model, clip_embeddings, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                                 typical_filtering=True, typical_mass=0.2, typical_min_tokens=1,
                                 classifier_free_scale=5, renoise_steps=11,
                                 renoise_mode="start")
            sampled = decode(self.vqmodel, sampled[-1], latent_shape)

        output_paths = []
        for i in range(len(sampled)):
            out_path = f'output-{i}.png'
            save_image(sampled[i], out_path, normalize=True, nrow=int(sqrt(len(sampled))))
            output_paths.append(Path(out_path))
        return output_paths


class LatentInterpolationPredictor(Predictor):
    def predict(
            self,
            text1: str = Input(default="High quality front portrait photo of a tiger."),
            text2: str = Input(default="High quality front portrait photo of a dog."),
            n_interpolations: int = Input(default=10, description="How many interpolation steps"),
    ) -> typing.List[Path]:
        text1 = str(text1)
        text2 = str(text2)
        n_interpolations = int(n_interpolations)

        text1_encoded = tokenizer.tokenize([text1]).to(device)
        text2_encoded = tokenizer.tokenize([text2]).to(device)
        latent_shape = (32, 32)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                clip_embeddings1 = self.clip_model.encode_text(text1_encoded)
                clip_embeddings2 = self.clip_model.encode_text(text2_encoded)
                dtype = clip_embeddings2.dtype

            outputs = []
            for i in torch.linspace(0, 1, n_interpolations).to(device):
                lerp = torch.lerp(clip_embeddings1.float(), clip_embeddings2.float(), i).to(dtype)
                with torch.autocast(device_type="cuda"):
                    sampled = sample(self.model, lerp, T=12, size=latent_shape, starting_t=0,
                                     temp_range=[1.0, 1.0],
                                     typical_filtering=True, typical_mass=0.2, typical_min_tokens=1,
                                     classifier_free_scale=5, renoise_steps=11,
                                     renoise_mode="start")
                sampled = decode(self.vqmodel, sampled[-1], latent_shape)[0]
                outputs.append(sampled)
            sampled = outputs

        output_paths = []
        for i in range(len(sampled)):
            out_path = f'OutputImage-{i}.png'
            save_image(sampled[i], out_path, normalize=True, nrow=int(sqrt(len(sampled))))
            output_paths.append(Path(out_path))
        return output_paths


class ImageVariationPredictor(Predictor):
    def setup(self):
        super(ImageVariationPredictor, self).setup()

        state_dict = torch.load("./model_50000_img.pt", map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval().requires_grad_()

        self.clip_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                             std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def predict(
            self,
            input_image: Path = Input(description="Image to variate on"),
            num_outputs: int = Input(default=3),
    ) -> typing.List[Path]:
        input_image = Image.open(str(input_image))
        num_outputs = int(num_outputs)
        latent_shape = (32, 32)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                input_image = self.clip_preprocess(input_image).to(device).unsqueeze(0)
                clip_embeddings = self.clip_model.encode_image(input_image).float()
                clip_embeddings = torch.repeat_interleave(clip_embeddings, num_outputs, dim=0)
                sampled = sample(self.model, clip_embeddings, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                                 typical_filtering=True, typical_mass=0.2, typical_min_tokens=1,
                                 classifier_free_scale=5, renoise_steps=11,
                                 renoise_mode="start")

            sampled = decode(self.vqmodel, sampled[-1], latent_shape)

        output_paths = []
        for i in range(len(sampled)):
            out_path = f'output-{i}.png'
            save_image(sampled[i], out_path, normalize=True, nrow=int(sqrt(len(sampled))))
            output_paths.append(Path(out_path))
        return output_paths


def find_next_multiplicity_of_8(x):
    z = 24
    while z < x:
        z += 8
    return z

class OutpaintingPredictor(Predictor):
    def predict(
            self,
            input_image: Path = Input(description="Image to variate on"),
            output_relative_size: str = Input(default="1.5,1.5", description="define size of output relative to the input."
                                                                    " 2,1.5 means x2 hgiher and x1.5 wide image"),
            input_location: str = Input(default="0.5,0.5", description="Define the relative location of the input in "
                                                                       "the canvas 0.5,0.5 means int the middle"),
            prompt: str = Input(default="An image hanged on the wall"),
    ) -> typing.List[Path]:
        input_image = str(input_image)
        input_image = torchvision.transforms.ToTensor()(Image.open(str(input_image))).unsqueeze(0).to(device)
        prompt = str(prompt)
        output_relative_size = eval(str(output_relative_size))
        input_location = eval(str(input_location))

        tokenized_text = tokenizer.tokenize([prompt]).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                clip_embeddings = self.clip_model.encode_text(tokenized_text)
                encoded_tokens = encode(input_image, self.vqmodel)
                lh, lw = encoded_tokens.shape[1:]
                ch, cw = int(lh * output_relative_size[0]), int(lw * output_relative_size[1])
                ch = find_next_multiplicity_of_8(ch)
                cw = find_next_multiplicity_of_8(cw)
                loc_h, loc_w = int(ch * input_location[0]), int(cw * input_location[1])

                canvas = torch.zeros((input_image.shape[0], ch, cw), dtype=torch.long).to(device)
                y = min(max(loc_h - lh//2, 0), ch - lh)
                x = min(max(loc_w - lw//2, 0), cw - lw)
                print((lh, lw), (ch, cw ), (loc_h, loc_w), (y,x))
                canvas[:, y: y + lh, x: x + lw] = encoded_tokens
                mask = torch.ones_like(canvas)
                mask[:, y: y + lh, x: x + lw] = 0
                sampled = sample(self.model, clip_embeddings, x=canvas, mask=mask, T=12, size=(ch, cw), starting_t=0,
                                 temp_range=[1.0, 1.0],
                                 typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=4,
                                 renoise_steps=11)
            sampled = decode(self.vqmodel, sampled[-1], (ch, cw))

        output_paths = []
        for i in range(len(sampled)):
            out_path = f'output-{i}.png'
            save_image(sampled[i], out_path, normalize=True, nrow=int(sqrt(len(sampled))))
            output_paths.append(Path(out_path))
        return output_paths


class StructuralMorphingPredictor(Predictor):

    def predict(
            self,
            input_image: Path = Input(description="Image to variate on"),
            prompt: str = Input(default="A fox posing for a photo. stock photo. highly detailed. 4k"),
    ) -> typing.List[Path]:
        input_image = str(input_image)
        input_image = torchvision.transforms.ToTensor()(Image.open(str(input_image))).unsqueeze(0).to(device)

        prompt = str(prompt)
        max_steps = 24
        init_step = 8
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                latent_image = encode(input_image, self.vqmodel)
                latent_shape = latent_image.shape[-2:]
                r = torch.ones(latent_image.size(0), device=device) * (init_step / max_steps)
                noised_latent_image, _ = self.model.add_noise(latent_image, r)

                tokenized_text = tokenizer.tokenize([prompt]).to(device)
                clip_embeddings = self.clip_model.encode_text(tokenized_text)

                sampled = sample(self.model, clip_embeddings, x=noised_latent_image, T=max_steps, size=latent_shape,
                                 starting_t=init_step, temp_range=[1.0, 1.0],
                                 typical_filtering=True, typical_mass=0.2, typical_min_tokens=1,
                                 classifier_free_scale=6, renoise_steps=max_steps - 1,
                                 renoise_mode="prev")
            sampled = decode(self.vqmodel, sampled[-1], latent_shape)

        output_paths = []
        for i in range(len(sampled)):
            out_path = f'output-{i}.png'
            save_image(sampled[i], out_path, normalize=True, nrow=int(sqrt(len(sampled))))
            output_paths.append(Path(out_path))
        return output_paths


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def sample(model, c, x=None, mask=None, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0],
           typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=-1, renoise_steps=11,
           renoise_mode='start'):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T + 1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        preds = []
        if x is None:
            x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
        elif mask is not None:
            noise = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
            x = noise * mask + (1 - mask) * x
        init_x = x.clone()
        for i in range(starting_t, T):
            if renoise_mode == 'prev':
                prev_x = x.clone()
            r, temp = r_range[i], temperatures[i]
            logits = model(x, c, r)
            if classifier_free_scale >= 0:
                logits_uncond = model(x, torch.zeros_like(c), r)
                logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
            x = logits
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            if typical_filtering:
                x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                x_flat_norm_p = torch.exp(x_flat_norm)
                entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                if typical_min_tokens > 1:
                    sorted_indices_to_remove[..., :typical_min_tokens] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
            # x_flat = torch.multinomial(x_flat.div(temp).softmax(-1), num_samples=1)[:, 0]
            x_flat = gumbel_sample(x_flat, temperature=temp)
            x = x_flat.view(x.size(0), *x.shape[2:])
            if mask is not None:
                x = x * mask + (1 - mask) * init_x
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i + 1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i + 1], random_x=prev_x)
                else:  # 'rand'
                    x, _ = model.add_noise(x, r_range[i + 1])
            preds.append(x.detach())
    return preds


def decode(vqmodel, img_seq, shape=(32, 32)):
    img_seq = img_seq.view(img_seq.shape[0], -1)
    b, n = img_seq.shape
    one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=vqmodel.num_tokens).float()
    z = (one_hot_indices @ vqmodel.model.quantize.embed.weight)
    z = rearrange(z, 'b (h w) c -> b c h w', h=shape[0], w=shape[1])
    img = vqmodel.model.decode(z)
    img = (img.clamp(-1., 1.) + 1) * 0.5
    return img


def encode(x, vqmodel):
    return vqmodel.model.encode((2 * x - 1))[-1][-1]