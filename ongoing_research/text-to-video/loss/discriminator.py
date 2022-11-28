import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, c_in=3, c_cond=0, c_hidden=256, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(c_in + c_cond, c_hidden // (2 ** d), kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = c_hidden // (2 ** max((d - i), 0))
            c_out = c_hidden // (2 ** max((d - 1 - i), 0))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(c_hidden, 1, kernel_size=1))
        layers.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        x = self.discriminator(x)
        return x


if __name__ == '__main__':
    x = torch.randn(10, 3, 256, 256)
    d = Discriminator()
    print(sum([p.numel() for p in d.parameters()]))
    print(d(x).shape)