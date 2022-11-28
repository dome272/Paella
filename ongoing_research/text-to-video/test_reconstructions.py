import math
import random
import torch
import torchvision.io
from vivq import VIVQ
import torchvision.utils as vutils
from utils import transforms, VideoDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


def load_video(path, clip_len=10, skip_frames=3):
    video_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(128),
        torchvision.transforms.CenterCrop(128),
    ])
    video, _, _ = torchvision.io.read_video(path)
    video = video.permute(0, 3, 1, 2) / 255.
    max_seek = video.shape[0] - (clip_len * skip_frames)
    start = math.floor(random.uniform(0., max_seek))
    video = video[start:start + (clip_len * skip_frames) + 1:skip_frames]
    if video_transform:
        video = video_transform(video)
    image, video = video[0], video[1:]
    return image.unsqueeze(0), video.unsqueeze(0)


path = r"C:\Users\d6582\Documents\ml\phenaki\data\webvid\example_videos\1066656142.mp4"
name = "vivq_2"
device = "cuda"
num_frames = 50
skip_frames = 5
c_hidden = 512

# ckpt_path = "./models/server/vivq_8192_drop_video/model_80000.pt"
ckpt_path = "./models/server/vivq_8192_5_skipframes/model_100000.pt"
model = VIVQ(c_hidden=c_hidden, codebook_size=8192).to(device)
state_dict = torch.load(ckpt_path)
model.load_state_dict(state_dict)
model.eval().requires_grad_(False)

if path is None:
    dataset = DataLoader(VideoDataset(video_transform=transforms, clip_len=num_frames, skip_frames=skip_frames), batch_size=1)
    image, video = next(iter(dataset))
else:
    image, video = load_video(path, clip_len=num_frames, skip_frames=skip_frames)

image, video = image.to(device), video.to(device)

# video = None

reconstruction, _ = model(image, video)

if video is None:
    orig = image
    # orig = orig[0]
    recon = reconstruction[0]
    print(f"results/{name}_{num_frames}.mp4")
    comp = vutils.make_grid(torch.cat([orig, recon]), nrow=len(orig)).detach().cpu()
else:
    orig = torch.cat([image.unsqueeze(1), video], dim=1)
    orig = orig[0]
    recon = reconstruction[0]
    print(f"results/{name}_{num_frames}.mp4")
    # torchvision.io.write_video(f"results/{name}_{num_frames}_.mp4", (recon * 255).cpu().permute(0, 2, 3, 1), fps=5)
    # torchvision.io.write_video(f"results/{num_frames}_orig_.mp4", (orig * 255).cpu().permute(0, 2, 3, 1), fps=5)
    comp = vutils.make_grid(torch.cat([orig, recon]), nrow=len(orig)).detach().cpu()
plt.imshow(comp.permute(1, 2, 0))
plt.show()

vutils.save_image(comp, f"results/{num_frames}.jpg")
