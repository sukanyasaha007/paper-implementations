"""Pretrain an MLP on MNIST, then use LoRA to fine-tune it on one digit while the original weights stay frozen.

Usage: python train.py
Writes results/lora_results.json.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

from lora import add_lora, count_parameters, enable_disable_lora, freeze_non_lora
from model import RichBoyNet

FINE_TUNE_DIGIT = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_loaders():
    # ToTensor: PIL image -> (1, 28, 28) float tensor in [0, 1]; then normalize with MNIST's mean and std
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=False)
    return mnist_trainset, train_loader, test_loader


def train(train_loader, net, epochs=1, total_iterations_limit=None, lr=0.01):
    cross_en = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    total_iter = 0
    for epoch in range(epochs):
        net.train()
        num_iter = 0
        sum_loss = 0
        data_iterator = tqdm.tqdm(train_loader, desc=f'Epoch: {epoch + 1}')

        for x, y in data_iterator:
            num_iter += 1
            total_iter += 1
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            loss = cross_en(logits, y)
            sum_loss += loss.item()  # .item() so the running sum doesn't keep every step's graph alive
            data_iterator.set_postfix(loss=sum_loss / num_iter)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if total_iterations_limit is not None and total_iter >= total_iterations_limit:
                return


@torch.no_grad()
def test(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    wrong = [0] * 10

    for x, y in tqdm.tqdm(test_loader, desc='Testing'):
        x = x.to(device)
        y = y.to(device)
        output = net(x.view(-1, 784))
        for idx, logits in enumerate(output):
            if torch.argmax(logits) == y[idx]:
                correct += 1
            else:
                wrong[y[idx]] += 1  # count errors by the true digit
            total += 1

    accuracy = correct / total
    print(f'Accuracy = {accuracy:.3f}')
    for digit, wrong_count in enumerate(wrong):
        print(f'Digit {digit} Wrong Count: {wrong_count}')
    return {'accuracy': accuracy, 'wrong_by_digit': wrong}


def main():
    _ = torch.manual_seed(0)
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    mnist_trainset, train_loader, test_loader = get_loaders()

    # 1. Pretrain the full network for one epoch
    net = RichBoyNet().to(device)
    train(train_loader, net, epochs=1)
    before = test(net, test_loader)

    # Keep a copy of the original weights to check LoRA never modifies them
    og_weights = {name: param.clone().detach() for name, param in net.named_parameters()}

    # 2. Add LoRA (rank 1) to every linear layer and freeze everything else
    layers = net.linear_layers()
    add_lora(layers, device, rank=1, lora_alpha=1)
    freeze_non_lora(net)
    lora_params, original_params = count_parameters(layers)
    print(f'Original parameters: {original_params:,}, LoRA parameters: {lora_params:,} '
          f'({100 * lora_params / original_params:.3f}%)')

    # 3. Fine-tune only the LoRA matrices on one digit
    digit_indices = mnist_trainset.targets == FINE_TUNE_DIGIT
    mnist_trainset.data = mnist_trainset.data[digit_indices]
    mnist_trainset.targets = mnist_trainset.targets[digit_indices]
    digit_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
    train(digit_loader, net, epochs=1, total_iterations_limit=100)

    # 4. The frozen weights are untouched, and LoRA can be switched on and off
    for i, layer in enumerate(layers, start=1):
        assert torch.equal(layer.parametrizations.weight.original, og_weights[f'linear{i}.weight'])

    enable_disable_lora(layers, enabled=True)
    lora = net.linear1.parametrizations.weight[0]
    expected = net.linear1.parametrizations.weight.original + (lora.lora_B @ lora.lora_A) * lora.scale
    assert torch.equal(net.linear1.weight, expected)

    enable_disable_lora(layers, enabled=False)
    assert torch.equal(net.linear1.weight, og_weights['linear1.weight'])

    # 5. Compare with LoRA on and off
    enable_disable_lora(layers, enabled=True)
    with_lora = test(net, test_loader)
    enable_disable_lora(layers, enabled=False)
    without_lora = test(net, test_loader)

    results = {
        'fine_tune_digit': FINE_TUNE_DIGIT,
        'original_parameters': original_params,
        'lora_parameters': lora_params,
        'before_fine_tune': before,
        'lora_enabled': with_lora,
        'lora_disabled': without_lora,
    }
    with open(results_dir / 'lora_results.json', 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
