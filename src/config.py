from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Run control
    algorithm: str = "dqn"          # "dqn" or "double_dqn"
    env_id: str = "ALE/Pong-v5"
    difficulty: int = 0
    seed: int = 42
    device: str = "cuda"            # "cuda" or "cpu"

    # Training
    episodes: int = 10000
    gamma: float = 0.99
    learning_rate: float = 1e-4

    # Replay / batch
    replay_capacity: int = 50000
    batch_size: int = 64
    start_training_steps: int = 10000

    # Epsilon schedule (exponential decay)
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.1
    epsilon_decay: float = 300000.0

    # Target net
    target_update_freq: int = 1000  # in environment steps (frames)

    # Gradient accumulation
    accumulation_steps: int = 4

    # Checkpointing
    save_every_episodes: int = 100
    save_best: bool = True

    # Video
    record_video: bool = False
    record_every_episodes: int = 100
