import os
import sys

import torch

sys.path.insert(0, os.path.abspath("src"))

from models import DQN  # noqa: E402


def test_dqn_forward_shape():
    input_shape = (4, 84, 84)
    num_actions = 6
    model = DQN(input_shape, num_actions, fc_hidden=256)
    x = torch.zeros(2, 4, 84, 84)
    y = model(x)
    assert y.shape == (2, num_actions)
