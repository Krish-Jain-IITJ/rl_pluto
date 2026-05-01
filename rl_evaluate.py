"""
rl_evaluate.py — Evaluation script for saved SAC checkpoints
Loads a trained model and runs deterministic episodes (no exploration noise).
Prints per-episode stats and average performance.

Usage:
    python rl_evaluate.py                          # uses latest checkpoint
    python rl_evaluate.py --model checkpoints/sac_obstacle_dodge_final
    python rl_evaluate.py --episodes 50
"""

import argparse
import os
import glob
import numpy as np

from stable_baselines3 import SAC
from sim import SimParams
from rl_environment import ObstacleDodgeEnv
from obstacle_dataset import training_data
from rl_config import TrainingConfig, EnvConfig


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recently saved checkpoint zip in checkpoint_dir."""
    pattern = os.path.join(checkpoint_dir, '*.zip')
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir!r}")
    return files[-1][:-4]  # strip .zip


def evaluate(model_path: str, num_episodes: int = 20, verbose: bool = True):
    """
    Run deterministic evaluation for num_episodes episodes.

    Parameters
    ----------
    model_path   : Path to saved SAC model (without .zip extension).
    num_episodes : Number of evaluation episodes.
    verbose      : Whether to print per-episode stats.

    Returns
    -------
    stats : dict with keys 'rewards', 'crashed', 'avg_reward', 'crash_rate'
    """
    print(f"\nLoading model from: {model_path}.zip")
    model = SAC.load(model_path)

    env = ObstacleDodgeEnv(training_data=training_data, sim_params=SimParams())

    rewards  = []
    crashes  = []
    d_mins   = []
    steps_ep = []

    for ep in range(num_episodes):
        env.current_episode = ep % len(training_data)
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        step_count = 0
        ep_d_min = np.inf
        crashed = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1
            d = info.get('d_min', np.inf)
            if d < ep_d_min:
                ep_d_min = d
            if info.get('crashed', False):
                crashed = True

        rewards.append(ep_reward)
        crashes.append(crashed)
        d_mins.append(ep_d_min)
        steps_ep.append(step_count)

        if verbose:
            print(
                f"  Ep {ep+1:3d}/{num_episodes}  "
                f"Reward: {ep_reward:9.2f}  Steps: {step_count:4d}  "
                f"d_min: {ep_d_min:.3f}m  Crashed: {'YES' if crashed else ' no'}"
            )

    avg_reward  = float(np.mean(rewards))
    crash_rate  = float(np.mean(crashes)) * 100
    avg_d_min   = float(np.mean(d_mins))

    print(f"\n{'='*55}")
    print(f"  Evaluation Results ({num_episodes} episodes)")
    print(f"  Avg Reward : {avg_reward:.2f}")
    print(f"  Crash Rate : {crash_rate:.1f}%")
    print(f"  Avg d_min  : {avg_d_min:.3f} m")
    print(f"{'='*55}\n")

    return {
        'rewards':     rewards,
        'crashed':     crashes,
        'd_mins':      d_mins,
        'avg_reward':  avg_reward,
        'crash_rate':  crash_rate,
        'avg_d_min':   avg_d_min,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a saved SAC checkpoint.')
    parser.add_argument('--model',    type=str,  default=None,
                        help='Path to model checkpoint (no .zip)')
    parser.add_argument('--episodes', type=int,  default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--dir',      type=str,  default=TrainingConfig.checkpoint_dir,
                        help='Checkpoint directory to search')
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    else:
        model_path = find_latest_checkpoint(args.dir)
        print(f"Auto-selected checkpoint: {model_path}.zip")

    evaluate(model_path, num_episodes=args.episodes)
