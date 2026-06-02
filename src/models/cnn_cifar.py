import torch
import torch.nn as nn


class CNN3CIFAR(nn.Module):
    """3-layer CNN following Cohen et al. (2021) edge-of-stability setup.

    Three conv blocks each with MaxPool(2); on 32x32 inputs the spatial map
    shrinks 32 -> 16 -> 8 -> 4. The classifier is a single Linear on the
    flattened 4x4xwidth feature map.

    conv(3->w,3x3,p=1) -> ReLU -> MaxPool(2)
    -> conv(w->w,3x3,p=1) -> ReLU -> MaxPool(2)
    -> conv(w->w,3x3,p=1) -> ReLU -> MaxPool(2)
    -> Flatten -> Linear(w*4*4, num_classes)
    """

    def __init__(self, width: int = 32, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
