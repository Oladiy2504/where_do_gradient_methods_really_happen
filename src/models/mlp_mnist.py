import torch
import torch.nn as nn


class MLP3MNIST(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, width: int = 200, num_classes: int = 10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
