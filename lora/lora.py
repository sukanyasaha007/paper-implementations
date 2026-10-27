"""LoRA as a weight parametrization: W' = W0 + (B @ A) * alpha / rank, with W0 frozen.

W0 is (d, k) = (out_features, in_features) for nn.Linear. B is (d, r) and A is (r, k), so LoRA trains
r * (d + k) parameters per layer instead of d * k.
"""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class LoRAParameterization(nn.Module):
    def __init__(self, out_features, in_features, rank, alpha=1, device='cpu'):
        super().__init__()
        # Paper init: A random Gaussian, B zero. B @ A = 0, so the adapted model starts identical to the original.
        # If A were also zero, both gradients would be zero (dB = dW' A^T, dA = B^T dW') and LoRA could never learn.
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank), device=device))
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features), device=device))
        nn.init.normal_(self.lora_A, mean=0, std=1)

        # alpha / rank keeps the update scale comparable when the rank changes
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, og_weights):
        if self.enabled:
            return og_weights + torch.matmul(self.lora_B, self.lora_A).view(og_weights.shape) * self.scale
        return og_weights


def linear_layer_parametrization(layer, device, rank=1, lora_alpha=1):
    out_features, in_features = layer.weight.shape
    return LoRAParameterization(out_features, in_features, rank, lora_alpha, device)


def add_lora(layers, device, rank=1, lora_alpha=1):
    # After this, layer.weight is computed on every access as parametrization(layer.parametrizations.weight.original)
    for layer in layers:
        parametrize.register_parametrization(layer, 'weight', linear_layer_parametrization(layer, device, rank, lora_alpha))


def enable_disable_lora(layers, enabled=True):
    for layer in layers:
        layer.parametrizations['weight'][0].enabled = enabled


def freeze_non_lora(net):
    for name, param in net.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False


def count_parameters(layers):
    lora_params = 0
    non_lora_params = 0
    for layer in layers:
        lora = layer.parametrizations['weight'][0]
        lora_params += lora.lora_A.nelement() + lora.lora_B.nelement()
        non_lora_params += layer.weight.nelement() + layer.bias.nelement()
    return lora_params, non_lora_params
