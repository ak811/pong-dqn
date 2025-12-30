import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from preprocess import FrameStacker, preprocess_frame  # noqa: E402


def test_preprocess_frame_shape_and_range():
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    out = preprocess_frame(frame)
    assert out.shape == (1, 84, 84)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_frame_stacker():
    fs = FrameStacker(4)
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    s0 = fs.reset(frame)
    assert s0.shape == (4, 84, 84)
    s1 = fs.step(frame)
    assert s1.shape == (4, 84, 84)
