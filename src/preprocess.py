from __future__ import annotations

from collections import deque
from typing import Deque

import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB (H,W,3) -> normalized grayscale (1,84,84) float32."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    expanded = np.expand_dims(resized, axis=0)  # (1,84,84)
    return expanded.astype(np.float32) / 255.0


class FrameStacker:
    """Stacks 4 preprocessed frames into (4,84,84)."""

    def __init__(self, stack_size: int = 4):
        self.stack_size = stack_size
        self.frames: Deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        p = preprocess_frame(frame)
        self.frames = deque([p] * self.stack_size, maxlen=self.stack_size)
        return self.get()

    def step(self, frame: np.ndarray) -> np.ndarray:
        p = preprocess_frame(frame)
        self.frames.append(p)
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)  # (4,84,84)
