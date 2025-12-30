import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from evaluate import evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-path", type=str, default=None)
    args = p.parse_args()

    stats = evaluate(
        env_id="ALE/Pong-v5",
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        difficulty=args.difficulty,
        seed=args.seed,
        device=args.device,
        record_video=args.record_video,
        video_path=args.video_path,
    )
    print(stats)


if __name__ == "__main__":
    main()
