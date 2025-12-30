from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent import EpsilonScheduler, select_action
from config import TrainConfig
from env import make_pong_env
from models import DQN
from preprocess import FrameStacker
from replay_buffer import ReplayBuffer
from utils import make_logger, save_checkpoint, save_config, set_seed


def _device_from_cfg(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(cfg: TrainConfig, exp_dir: str) -> Dict[str, List[float]]:
    os.makedirs(exp_dir, exist_ok=True)
    video_dir = os.path.join(exp_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    logger = make_logger(os.path.join(exp_dir, "train.log"))
    save_config(os.path.join(exp_dir, "config.json"), cfg)
    logger.info(f"Config: {asdict(cfg)}")

    set_seed(cfg.seed)
    device = _device_from_cfg(cfg)
    logger.info(f"Using device: {device}")

    # Env
    env = make_pong_env(
        env_id=cfg.env_id,
        render_mode="rgb_array",
        difficulty=cfg.difficulty,
        seed=cfg.seed,
        record_video=cfg.record_video,
        video_folder=video_dir,
        episode_trigger=(lambda ep: ep % cfg.record_every_episodes == 0) if cfg.record_video else None,
        name_prefix=cfg.algorithm,
    )

    num_actions = env.action_space.n

    # Init observation -> shape
    obs, _ = env.reset()
    stacker = FrameStacker(stack_size=4)
    state_np = stacker.reset(obs)  # (4,84,84)
    input_shape = state_np.shape

    policy_net = DQN(input_shape, num_actions, fc_hidden=256).to(device)
    target_net = DQN(input_shape, num_actions, fc_hidden=256).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.learning_rate)
    memory = ReplayBuffer(cfg.replay_capacity)
    eps_sched = EpsilonScheduler(cfg.initial_epsilon, cfg.final_epsilon, cfg.epsilon_decay)

    steps_done = 0
    total_updates = 0
    best_reward = -float("inf")

    # Metrics
    rewards: List[float] = []
    losses: List[float] = []
    epsilons: List[float] = []

    accumulation_steps = max(1, int(cfg.accumulation_steps))
    mini_batch_size = max(1, int(cfg.batch_size // accumulation_steps))

    start_time_all = time.time()

    try:
        for ep in range(cfg.episodes):
            ep_start = time.time()

            obs, _ = env.reset()
            state_np = stacker.reset(obs)
            state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1,4,84,84)

            ep_reward = 0.0
            ep_steps = 0
            ep_loss_accum = 0.0
            done = False

            while not done:
                ep_steps += 1

                epsilon = eps_sched.value(steps_done)
                steps_done += 1

                action = select_action(policy_net, state, num_actions, epsilon)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += float(reward)

                next_state_np = stacker.step(next_obs)
                next_state = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0).to(device)

                # Store (with batch dim for fast concat)
                memory.push(
                    state.detach().cpu().numpy(),
                    action,
                    float(reward),
                    next_state.detach().cpu().numpy(),
                    done,
                )

                state = next_state

                # Train
                if len(memory) >= cfg.start_training_steps:
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0

                    for _ in range(accumulation_steps):
                        states_np, actions_np, rewards_np, next_states_np, dones_np = memory.sample(mini_batch_size)

                        states_t = torch.tensor(states_np, dtype=torch.float32).to(device)
                        actions_t = torch.tensor(actions_np, dtype=torch.int64).unsqueeze(1).to(device)
                        rewards_t = torch.tensor(rewards_np, dtype=torch.float32).to(device)
                        next_states_t = torch.tensor(next_states_np, dtype=torch.float32).to(device)
                        dones_t = torch.tensor(dones_np, dtype=torch.float32).to(device)

                        q_sa = policy_net(states_t).gather(1, actions_t).squeeze(1)

                        with torch.no_grad():
                            if cfg.algorithm == "double_dqn":
                                # action selection: policy net
                                next_actions = policy_net(next_states_t).argmax(dim=1, keepdim=True)
                                # evaluation: target net
                                next_q = target_net(next_states_t).gather(1, next_actions).squeeze(1)
                            else:
                                # vanilla DQN
                                next_q = target_net(next_states_t).max(dim=1)[0]

                            target = rewards_t + cfg.gamma * next_q * (1.0 - dones_t)

                        loss = nn.SmoothL1Loss()(q_sa, target)
                        loss = loss / accumulation_steps
                        loss.backward()
                        total_loss += float(loss.item())

                    optimizer.step()
                    total_updates += 1
                    ep_loss_accum += total_loss

                # Target update
                if steps_done % cfg.target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            rewards.append(ep_reward)
            epsilons.append(float(eps_sched.value(steps_done)))
            avg_loss = ep_loss_accum / max(1, ep_steps)
            losses.append(float(avg_loss))

            dur = time.time() - ep_start
            elapsed = time.time() - start_time_all

            logger.info(
                f"Episode {ep} | Steps: {ep_steps} | Reward: {ep_reward:.2f} | "
                f"Epsilon: {epsilons[-1]:.4f} | AvgLoss: {avg_loss:.6f} | "
                f"Updates: {total_updates} | Dur: {dur:.2f}s | Elapsed: {elapsed:.1f}s"
            )

            # Save periodic checkpoints
            if cfg.save_every_episodes > 0 and (ep % cfg.save_every_episodes == 0):
                save_checkpoint(
                    os.path.join(exp_dir, f"checkpoint_ep{ep}.pth"),
                    policy_net,
                    optimizer=optimizer,
                    meta={"episode": ep, "reward": ep_reward, "steps_done": steps_done},
                )

            # Save best checkpoint
            if cfg.save_best and ep_reward > best_reward:
                best_reward = ep_reward
                save_checkpoint(
                    os.path.join(exp_dir, "best.pth"),
                    policy_net,
                    optimizer=optimizer,
                    meta={"episode": ep, "reward": ep_reward, "steps_done": steps_done},
                )

    finally:
        env.close()

    # Final save
    save_checkpoint(
        os.path.join(exp_dir, "final.pth"),
        policy_net,
        optimizer=optimizer,
        meta={"episode": cfg.episodes - 1, "reward": rewards[-1] if rewards else None, "steps_done": steps_done},
    )

    return {"reward": rewards, "loss": losses, "epsilon": epsilons}
