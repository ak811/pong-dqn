from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from env import make_pong_env
from models import DQN
from preprocess import FrameStacker
from utils import load_checkpoint


def evaluate(
    env_id: str,
    checkpoint_path: str,
    episodes: int = 20,
    difficulty: int = 0,
    seed: int = 42,
    device: str = "cuda",
    record_video: bool = False,
    video_path: Optional[str] = None,
) -> Dict[str, float]:
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # If recording, gym expects a folder. We'll record into that folder and later user can move/rename.
    video_folder = "eval_videos"
    if record_video:
        os.makedirs(video_folder, exist_ok=True)

    env = make_pong_env(
        env_id=env_id,
        render_mode="rgb_array",
        difficulty=difficulty,
        seed=seed,
        record_video=record_video,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,
        name_prefix="eval",
    )

    obs, _ = env.reset()
    stacker = FrameStacker(4)
    state_np = stacker.reset(obs)
    input_shape = state_np.shape
    num_actions = env.action_space.n

    policy = DQN(input_shape, num_actions, fc_hidden=256).to(dev)
    load_checkpoint(checkpoint_path, policy)
    policy.eval()

    rewards: List[float] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            state_np = stacker.reset(obs)
            done = False
            total = 0.0

            while not done:
                state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(dev)
                with torch.no_grad():
                    action = int(policy(state)[0].argmax().item())

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                total += float(reward)
                state_np = stacker.step(next_obs)

            rewards.append(total)
    finally:
        env.close()

    avg = float(np.mean(rewards)) if rewards else 0.0
    std = float(np.std(rewards)) if rewards else 0.0

    # Note: RecordVideo writes files itself. If user asked for a specific `video_path`,
    # they'd move/rename after. Keeping it simple since you removed CI/precommit anyway.
    _ = video_path  # reserved for future post-processing if needed

    return {"avg_reward": avg, "std_reward": std, "episodes": float(episodes)}
