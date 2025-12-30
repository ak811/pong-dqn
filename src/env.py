from __future__ import annotations

import os
from typing import Callable, Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def make_pong_env(
    env_id: str,
    render_mode: Optional[str],
    difficulty: int = 0,
    seed: int = 42,
    record_video: bool = False,
    video_folder: str = "videos",
    episode_trigger: Optional[Callable[[int], bool]] = None,
    name_prefix: str = "video",
) -> gym.Env:
    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if difficulty and difficulty > 0:
        kwargs["difficulty"] = difficulty

    env = gym.make(env_id, **kwargs)

    # Keep fps sane for recordings
    try:
        env.metadata["render_fps"] = 30
        env.unwrapped.metadata["render_fps"] = 30
    except Exception:
        pass

    env.reset(seed=seed)

    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        if episode_trigger is None:
            episode_trigger = lambda ep: ep % 100 == 0  # noqa: E731
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
        )
    return env
