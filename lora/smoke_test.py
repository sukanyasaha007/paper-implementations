"""Fast LoRA checks on random data; no MNIST download needed.

Usage: python smoke_test.py
"""
import torch
import torch.nn as nn

from lora import add_lora, count_parameters, enable_disable_lora, freeze_non_lora
from model import RichBoyNet


def smoke_lora():
    torch.manual_seed(0)
    net = RichBoyNet(hidden_size_1=64, hidden_size_2=32)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    with torch.no_grad():
        original_output = net(x)
    assert original_output.shape == (8, 10)
    og_weights = {name: p.clone().detach() for name, p in net.named_parameters()}

    layers = net.linear_layers()
    add_lora(layers, 'cpu', rank=2, lora_alpha=2)
    freeze_non_lora(net)

    # B starts at zero, so adding LoRA does not change the model's output
    with torch.no_grad():
        assert torch.allclose(net(x), original_output)

    # r * (d + k) parameters per layer
    lora_params, original_params = count_parameters(layers)
    assert lora_params == 2 * ((64 + 784) + (32 + 64) + (10 + 32))
    assert original_params == sum(p.nelement() for p in og_weights.values())

    trainable = [name for name, p in net.named_parameters() if p.requires_grad]
    assert trainable and all('lora' in name for name in trainable)
    print('lora init, parameter count, freezing: ok')

    # One training step: B gets a gradient on the first step because A is random (it would be zero if A started at 0)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-2)
    loss = nn.CrossEntropyLoss()(net(x), y)
    loss.backward()
    assert net.linear1.parametrizations.weight[0].lora_B.grad.abs().sum() > 0
    optimizer.step()

    # The frozen weights never change; the effective weight is W0 + B @ A * scale
    for i, layer in enumerate(layers, start=1):
        assert torch.equal(layer.parametrizations.weight.original, og_weights[f'linear{i}.weight'])
    lora = net.linear1.parametrizations.weight[0]
    assert torch.allclose(net.linear1.weight,
                          net.linear1.parametrizations.weight.original + (lora.lora_B @ lora.lora_A) * lora.scale)

    # Disabling LoRA recovers the original model exactly
    with torch.no_grad():
        assert not torch.allclose(net(x), original_output)
        enable_disable_lora(layers, enabled=False)
        assert torch.allclose(net(x), original_output)
    print('lora update, frozen weights, enable/disable: ok')


if __name__ == '__main__':
    smoke_lora()
