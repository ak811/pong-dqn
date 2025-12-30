import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from config import TrainConfig  # noqa: E402
from train import train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--difficulty", type=int, choices=[2, 3], required=True)
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--algo", type=str, choices=["dqn", "double_dqn"], default="dqn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--record-every", type=int, default=100)
    args = p.parse_args()

    cfg = TrainConfig(
        algorithm=args.algo,
        difficulty=args.difficulty,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        record_video=args.record_video,
        record_every_episodes=args.record_every,
    )
    train(cfg, args.exp_dir)


if __name__ == "__main__":
    main()
