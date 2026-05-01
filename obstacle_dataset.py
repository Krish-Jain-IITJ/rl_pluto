"""
obstacle_dataset.py — Training Dataset Generator for RL Obstacle Avoidance
Fixes applied (from ideation doc):
  - Added random seed for reproducibility
  - Enforced minimum angular separation between obstacles
  - Curriculum-aware dataset generation (Stage 1/2/3)
  - Obstacles always placed exactly ON the desired circle (r_d)
"""

import numpy as np
from sim import SimParams
from rl_config import EnvConfig, CurriculumConfig


# ─── Minimum Separation Obstacle Sampling ────────────────────────────────────

def _sample_angles_with_separation(rng, n: int, min_gap_rad: float = np.pi / 4) -> np.ndarray:
    """
    Sample n angles in [0, 2π) with minimum angular separation min_gap_rad.
    Falls back to independent sampling if separation cannot be guaranteed
    (e.g., too many obstacles for the gap size).
    """
    if n * min_gap_rad > 2 * np.pi:
        # Cannot satisfy separation with this many obstacles; sample freely
        return rng.uniform(0, 2 * np.pi, n)

    angles = []
    max_attempts = 2000
    while len(angles) < n:
        attempts = 0
        angles = []
        for _ in range(n):
            ok = False
            for _ in range(max_attempts):
                candidate = rng.uniform(0, 2 * np.pi)
                if all(min((candidate - a) % (2 * np.pi),
                           (a - candidate) % (2 * np.pi)) > min_gap_rad
                       for a in angles):
                    angles.append(candidate)
                    ok = True
                    break
            if not ok:
                # Restart this episode's sampling
                break
        attempts += 1
        if attempts > 10:
            break  # give up and return what we have

    # Pad with random if we couldn't get enough
    while len(angles) < n:
        angles.append(rng.uniform(0, 2 * np.pi))

    return np.array(angles[:n])


# ─── Main Dataset Generator ───────────────────────────────────────────────────

def generate_rl_training_dataset(
    num_episodes: int = 2000,
    obs_per_episode: int = None,
    path_radius: float = None,
    seed: int = 42,
    min_separation: float = None,
) -> np.ndarray:
    """
    Generate obstacle dataset for RL training.

    Parameters
    ----------
    num_episodes     : Total number of episodes.
    obs_per_episode  : Obstacles per episode (defaults to EnvConfig.max_obstacles).
    path_radius      : Circle radius (defaults to SimParams.r_d).
    seed             : Random seed for reproducibility.
    min_separation   : Minimum angular gap between obstacles (defaults to
                       EnvConfig.obs_min_separation).

    Returns
    -------
    dataset : np.ndarray of shape (num_episodes, obs_per_episode, 2)
              Each entry [ep, obs, :] = (x, y) of obstacle obs in episode ep.
    """
    if obs_per_episode is None:
        obs_per_episode = EnvConfig.max_obstacles
    if path_radius is None:
        path_radius = SimParams.r_d
    if min_separation is None:
        min_separation = EnvConfig.obs_min_separation

    rng = np.random.default_rng(seed)
    dataset = np.zeros((num_episodes, obs_per_episode, 2))

    for ep in range(num_episodes):
        angles = _sample_angles_with_separation(rng, obs_per_episode, min_separation)
        dataset[ep, :, 0] = path_radius * np.cos(angles)
        dataset[ep, :, 1] = path_radius * np.sin(angles)

    return dataset


def generate_curriculum_dataset(
    num_episodes: int = 2000,
    path_radius: float = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate dataset with curriculum-aware obstacle placement.

    Stage 1 (ep 0 → CurriculumConfig.stage1_end):
        Single obstacle in second half of circle [π, 2π).
        Padded with a duplicate so array shape stays consistent.

    Stage 2 (stage1_end → stage2_end):
        Two obstacles with minimum angular separation enforced.

    Stage 3 (stage2_end → num_episodes):
        Two obstacles, fully random (no separation constraint).

    Returns
    -------
    dataset : np.ndarray of shape (num_episodes, 2, 2)
    """
    if path_radius is None:
        path_radius = SimParams.r_d

    rng = np.random.default_rng(seed)
    max_obs = EnvConfig.max_obstacles
    dataset = np.zeros((num_episodes, max_obs, 2))

    for ep in range(num_episodes):
        if ep < CurriculumConfig.stage1_end:
            # Stage 1: single obstacle in second half of circle
            angle = rng.uniform(np.pi, 2 * np.pi)
            x = path_radius * np.cos(angle)
            y = path_radius * np.sin(angle)
            # Duplicate so shape stays (2, 2)
            dataset[ep, :, 0] = x
            dataset[ep, :, 1] = y

        elif ep < CurriculumConfig.stage2_end:
            # Stage 2: two obstacles with minimum separation
            angles = _sample_angles_with_separation(
                rng, max_obs, EnvConfig.obs_min_separation
            )
            dataset[ep, :, 0] = path_radius * np.cos(angles)
            dataset[ep, :, 1] = path_radius * np.sin(angles)

        else:
            # Stage 3: fully random
            angles = rng.uniform(0, 2 * np.pi, max_obs)
            dataset[ep, :, 0] = path_radius * np.cos(angles)
            dataset[ep, :, 1] = path_radius * np.sin(angles)

    return dataset


# ─── Module-level dataset (used by rl_training.py and tests) ─────────────────
training_data = generate_curriculum_dataset(num_episodes=2000)

if __name__ == '__main__':
    print(f"Dataset shape:  {training_data.shape}")
    print(f"Episode 0  (Stage 1 — single obs duplicated):\n  {training_data[0]}")
    print(f"Episode 600 (Stage 2 — separated pair):\n  {training_data[600]}")
    print(f"Episode 1500 (Stage 3 — fully random):\n  {training_data[1500]}")
