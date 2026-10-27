import torch
import torch.nn as nn
from torch.nn import functional as F

from decoder import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # padding=0: no border, the output shrinks (64 -> 62 with a 3x3 kernel)
            # padding=1 (for 3x3): output stays the same size
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # (B, 3, H, W) -> (B, 128, H, W)
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),  # -> (B, 128, H/2, W/2)
            VAEResidualBlock(128, 256),  # -> (B, 256, H/2, W/2)
            VAEResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  # -> (B, 256, H/4, W/4)
            VAEResidualBlock(256, 512),  # -> (B, 512, H/4, W/4)
            VAEResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  # -> (B, 512, H/8, W/8)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),  # -> (B, 8, H/8, W/8): mean and log-variance
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x (batch, channel, H, W)
        # noise and output - (batch, 4, H/8, W/8)
        for module in self:  # nn.Sequential, so looping over self gives each layer in order
            if getattr(module, 'stride', None) == (2, 2):  # only the downsampling convs have stride (2, 2)
                # Asymmetric padding (left, right, top, bottom) so the stride-2 conv halves the size exactly
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (batch, 8, H/8, W/8) -> two (batch, 4, H/8, W/8) tensors
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)  # clamp too small / too large values
        variance = log_variance.exp()
        std = variance.sqrt()
        # Reparameterization: sample from N(mean, std^2) using externally supplied noise
        x = mean + std * noise
        return x * 0.18215  # constant scaling from the Stable Diffusion repo
