import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("src"))

from replay_buffer import ReplayBuffer  # noqa: E402


def test_replay_buffer_sample_shapes():
    rb = ReplayBuffer(100)
    for _ in range(50):
        s = np.zeros((1, 4, 84, 84), dtype=np.float32)
        ns = np.ones((1, 4, 84, 84), dtype=np.float32)
        rb.push(s, 1, 1.0, ns, False)

    states, actions, rewards, next_states, dones = rb.sample(16)
    assert states.shape == (16, 4, 84, 84)
    assert next_states.shape == (16, 4, 84, 84)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert dones.shape == (16,)
