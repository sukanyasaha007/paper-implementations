import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import SelfAttention


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)  # split the channels in groups of 32 and normalize each group
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:  # if the dims don't match, a 1x1 conv maps the residual to out_channels
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x) -> torch.Tensor:
        residue = x
        x = self.conv_1(F.silu(self.groupnorm_1(x)))
        x = self.conv_2(F.silu(self.groupnorm_2(x)))
        return x + self.residual_layer(residue)


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x) -> torch.Tensor:
        residue = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).transpose(-1, -2)  # (batch, H * W, C): every pixel is a token
        x = self.attention(x)
        x = x.transpose(-1, -2).view(n, c, h, w)
        return x + residue


class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),  # (B, 4, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),  # (B, 512, H/8, W/8)
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),  # (B, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),  # (B, 512, H/2, W/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),  # (B, 256, H/2, W/2)
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),  # (B, 256, H, W)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),  # (B, 3, H, W)
        )

    def forward(self, x):
        # x - (B, 4, H/8, W/8) latent; undo the scaling the encoder applied
        x = x / 0.18215
        for module in self:
            x = module(x)
        return x
