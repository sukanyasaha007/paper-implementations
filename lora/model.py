"""The network LoRA adapts: a deliberately oversized MLP for MNIST (the "RichBoyNet" from Umar Jamil's tutorial)."""
import torch.nn as nn


class RichBoyNet(nn.Module):
    def __init__(self, hidden_size_1=1000, hidden_size_2=3000):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def linear_layers(self):
        return [self.linear1, self.linear2, self.linear3]
