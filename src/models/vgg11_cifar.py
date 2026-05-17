import math

import torch
import torch.nn as nn


class VGG11CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: bool = False, batch_norm: bool = True):
        super().__init__()

        cfg = [
            64, "M",
            128, "M",
            256, 256, "M",
            512, 512, "M",
            512, 512, "M",
        ]

        layers = []
        in_channels = 3

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
