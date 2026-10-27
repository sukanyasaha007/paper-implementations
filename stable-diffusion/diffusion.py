import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import CrossAttention, SelfAttention


def get_time_embedding(timestep):
    # The timestep is encoded with sinusoids, like Transformer positions: a clock with many hands
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)  # (160,)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]  # (1, 160)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (1, 320)


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x):
        # (1, 320) -> (1, 1280)
        return self.linear_2(F.silu(self.linear_1(x)))


class UNETResidualBlock(nn.Module):
    """Processes image features and adds the time embedding to every pixel."""

    def __init__(self, in_channel, out_channel, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channel)
        self.conv_feature = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channel)
        self.groupnorm_merged = nn.GroupNorm(32, out_channel)
        self.conv_merged = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature - (B, in_c, H, W), time - (1, 1280)
        residue = feature
        feature = self.conv_feature(F.silu(self.groupnorm_feature(feature)))
        time = self.linear_time(F.silu(time))
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)  # broadcast the time vector over H and W
        merged = self.conv_merged(F.silu(self.groupnorm_merged(merged)))
        return merged + self.residual_layer(residue)


class UNETAttentionBlock(nn.Module):
    """Self-attention over pixels, cross-attention to the prompt, then a GeGLU feed-forward."""

    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x - (B, C, H, W), context - (B, 77, 768)
        residue_long = x
        x = self.conv_input(self.groupnorm(x))
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).transpose(-1, -2)  # (B, H * W, C)

        x = x + self.attention_1(self.layernorm_1(x))
        x = x + self.attention_2(self.layernorm_2(x), context)

        # GeGLU: one half of the projection gates the other half
        residue_short = x
        x, gate = self.linear_geglu_1(self.layernorm_3(x)).chunk(2, dim=-1)
        x = self.linear_geglu_2(x * F.gelu(gate))
        x = x + residue_short

        x = x.transpose(-1, -2).view(n, c, h, w)
        return self.conv_output(x) + residue_long


class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # (B, C, H, W) -> (B, C, 2H, 2W)
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """Sequential that passes the text context to attention blocks and the time to residual blocks."""

    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNETAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNETResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # Attention width n_head * n_embd must equal the block's channel count: 8 x 40 = 320, 8 x 80 = 640, 8 x 160 = 1280
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),                       # (B, 320, H/8, W/8)
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),           # (B, 320, H/16, W/16)
            SwitchSequential(UNETResidualBlock(320, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(UNETResidualBlock(640, 640), UNETAttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),           # (B, 640, H/32, W/32)
            SwitchSequential(UNETResidualBlock(640, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),         # (B, 1280, H/64, W/64)
            SwitchSequential(UNETResidualBlock(1280, 1280)),
            SwitchSequential(UNETResidualBlock(1280, 1280)),
        ])
        self.bottleneck = SwitchSequential(
            UNETResidualBlock(1280, 1280),
            UNETAttentionBlock(8, 160),
            UNETResidualBlock(1280, 1280),
        )
        # Input channels = previous output + the skip connection concatenated onto it
        self.decoders = nn.ModuleList([
            SwitchSequential(UNETResidualBlock(2560, 1280)),                                              # 0
            SwitchSequential(UNETResidualBlock(2560, 1280)),                                              # 1
            SwitchSequential(UNETResidualBlock(2560, 1280), UpSample(1280)),                              # 2  up to H/32
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),                 # 3
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),                 # 4
            SwitchSequential(UNETResidualBlock(1920, 1280), UNETAttentionBlock(8, 160), UpSample(1280)), # 5  up to H/16
            SwitchSequential(UNETResidualBlock(1920, 640), UNETAttentionBlock(8, 80)),                   # 6
            SwitchSequential(UNETResidualBlock(1280, 640), UNETAttentionBlock(8, 80)),                   # 7
            SwitchSequential(UNETResidualBlock(960, 640), UNETAttentionBlock(8, 80), UpSample(640)),     # 8  up to H/8
            SwitchSequential(UNETResidualBlock(960, 320), UNETAttentionBlock(8, 40)),                    # 9
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),                    # 10
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),                    # 11
        ])

    def forward(self, x, context, time):
        # x: (B, 4, H/8, W/8)    context: (B, 77, 768)    time: (1, 1280)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)                         # push: save every down-path output
        x = self.bottleneck(x, context, time)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)  # pop: newest first, glued onto channels
            x = layers(x, context, time)
        return x                                               # (B, 320, H/8, W/8)


class UNETOutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (B, 320, H/8, W/8) -> (B, 4, H/8, W/8)
        return self.conv(F.silu(self.groupnorm(x)))


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

    def forward(self, latent, context, time):
        # latent: (B, 4, H/8, W/8)    context: (B, 77, 768)    time: (1, 320)
        time = self.time_embedding(time)           # (1, 1280)
        output = self.unet(latent, context, time)  # (B, 320, H/8, W/8)
        return self.final(output)                  # (B, 4, H/8, W/8): predicted noise
