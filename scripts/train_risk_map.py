"""Minimal trainable risk-fusion model (Phase 2/5).

Student-friendly tiny CNN to fuse 4 cues into risk map.
"""

import torch
import torch.nn as nn


class TinyRiskNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = TinyRiskNet()
    x = torch.rand(2, 4, 128, 128)
    y = model(x)
    print("ok", y.shape)
