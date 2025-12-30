from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np


class ReplayBuffer:
    """
    Stores tuples of:
      state:      (1,C,H,W) float32
      action:     int
      reward:     float
      next_state: (1,C,H,W) float32
      done:       bool/float
    """
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, float(reward), next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.concatenate(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.concatenate(next_states, axis=0),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
