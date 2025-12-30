from __future__ import annotations

import math
import random
import torch


class EpsilonScheduler:
    def __init__(self, eps_start: float, eps_end: float, decay: float):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = float(decay)

    def value(self, step: int) -> float:
        # Exponential decay
        return float(self.eps_end + (self.eps_start - self.eps_end) * math.exp(-step / self.decay))


@torch.no_grad()
def select_action(policy_net: torch.nn.Module, state: torch.Tensor, num_actions: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(num_actions)
    q = policy_net(state)
    return int(q[0].argmax().item())
