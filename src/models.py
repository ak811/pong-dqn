from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions: int, fc_hidden: int = 256):
        super().__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out = self._get_conv_out((c, h, w))
        self.fc = nn.Sequential(
            nn.Linear(conv_out, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_actions),
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y.view(x.size(0), -1)
        return self.fc(y)
