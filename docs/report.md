# Pong DQN Project Report

## 1) Overview
This project trains an RL agent to play Atari Pong using Deep Q-Networks (DQN) and Double DQN, using Gymnasium + ALE.

## 2) Environment
- Environment: `ALE/Pong-v5`
- Observation: RGB frames (210x160x3)
- Action space: discrete actions (depends on ALE version; typically 6)

We preprocess frames to grayscale 84x84, normalize to [0,1], and stack 4 frames.

## 3) Methods
### DQN
We approximate Q(s,a) with a CNN:
- Conv(32, 8x8, stride 4) + ReLU
- Conv(64, 4x4, stride 2) + ReLU
- Conv(64, 3x3, stride 1) + ReLU
- FC(256) + ReLU
- Output layer to |A|

We train with:
- Experience replay
- Target network (periodic hard update)
- Epsilon-greedy exploration with exponential decay
- Huber loss (SmoothL1)

### Double DQN
Same network, but the target uses:
- action selection from policy net
- action evaluation from target net

## 4) Results
Store plots in `docs/figures/` and reference them here. Suggested plots:
- Reward per episode + moving average
- Epsilon schedule over time
- Loss curve

## 5) Reproducibility
All runs should specify:
- seed
- config values (gamma, lr, replay capacity, etc.)
- checkpoint path(s)

## 6) Extra: Difficulty Levels
This project supports `difficulty=2` and `difficulty=3` for Pong if ALE exposes the parameter.

---

### Command Examples (run from repo root)
Training:
- `python scripts/train_dqn.py --exp-dir experiments/dqn_run --episodes 10000`
- `python scripts/train_double_dqn.py --exp-dir experiments/ddqn_run --episodes 10000`

Evaluation:
- `python scripts/eval.py --checkpoint models/best.pth --episodes 20 --record-video --video-path docs/videos/demo.mp4`
