import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import Discriminator
import lpips


class FirstStageLoss(nn.Module):
    def __init__(self, mse_weight=1.0, vq_weight=0.1, adv_weight=0.1, perc_weight=0.1, start_disc=0, device="cpu"):
        super(FirstStageLoss, self).__init__()
        self.mse_weight = mse_weight
        self.vq_weight = vq_weight
        self.adv_weight = adv_weight
        self.perc_weight = perc_weight
        self.start_disc = start_disc
        self.discriminator = Discriminator().to(device)
        self.lpips = lpips.LPIPS(net="vgg").to(device).requires_grad_(False)

    def forward(self, images, videos, reconstructions, vq_loss, step):
        if videos is not None:
            videos = torch.cat([images.unsqueeze(1), videos], dim=1).view(-1, *videos.shape[2:])
        else:
            videos = images
        reconstructions = reconstructions.contiguous().view(-1, *reconstructions.shape[2:])
        mse_loss = F.mse_loss(videos, reconstructions)
        if step >= self.start_disc:
            d_real = self.discriminator(videos)
            d_real_loss = F.binary_cross_entropy(d_real, torch.zeros_like(d_real)+0.1)
            d_fake = self.discriminator(reconstructions.detach())
            d_fake_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake)-0.1)
            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_recon = self.discriminator(reconstructions)
            g_loss = F.binary_cross_entropy(d_recon, torch.zeros_like(d_recon)+0.1)
        else:
            g_loss = reconstructions.new_tensor(0)
            d_loss = None
        lpips_loss = self.lpips(videos, reconstructions).mean()

        loss = self.mse_weight * mse_loss + self.adv_weight * g_loss + self.perc_weight * lpips_loss + self.vq_weight * vq_loss

        return loss, d_loss


if __name__ == '__main__':
    device = "cuda"
    videos = torch.randn(1, 100, 3, 128, 128).to(device)
    l = FirstStageLoss(device=device)
    print(l(videos, videos, 1.0, 100000))
