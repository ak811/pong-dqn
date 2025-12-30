import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from config import TrainConfig  # noqa: E402
from train import train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-exp-dir", type=str, required=True)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Small illustrative sweep. Expand as needed.
    gammas = [0.95, 0.99]
    lrs = [5e-4, 1e-4]
    decays = [100000.0, 200000.0, 300000.0]
    algos = ["dqn", "double_dqn"]

    for algo, g, lr, decay in itertools.product(algos, gammas, lrs, decays):
        exp_name = f"{algo}_g{g}_lr{lr}_decay{int(decay)}"
        exp_dir = os.path.join(args.base_exp_dir, exp_name)

        cfg = TrainConfig(
            algorithm=algo,
            episodes=args.episodes,
            device=args.device,
            seed=args.seed,
            gamma=g,
            learning_rate=lr,
            epsilon_decay=decay,
            record_video=False,
        )
        train(cfg, exp_dir)


if __name__ == "__main__":
    main()
