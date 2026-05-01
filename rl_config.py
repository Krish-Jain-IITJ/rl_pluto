"""
rl_config.py — Centralized Configuration for Deep RL Obstacle Avoidance System
All hyperparameters, environment constants, and training settings live here.
"""

import numpy as np


class EnvConfig:
    """Environment constants."""
    max_obstacles       = 2          # number of obstacles per episode
    crash_distance      = 0.25       # meters — termination threshold
    max_error           = 2.0        # meters — tracking error termination
    obs_min_separation  = np.pi / 4  # radians — minimum angle between obstacles
    action_gain_k       = 1.0        # error-gated action gain


class RewardConfig:
    """Reward function coefficients."""
    alpha_alignment    = 0.3    # cos(alignment) reward weight
    alpha_error        = 1.0    # -e² penalty weight
    alpha_smoothness   = 0.01   # -Δω² smoothness penalty weight
    alpha_proximity    = 1.0    # obstacle proximity penalty weight
    alpha_heading_obs  = 0.5    # heading-toward-obstacle penalty weight
    crash_penalty      = -500.0 # terminal reward on crash
    circle_bonus       = 50.0   # sparse bonus for completing a circle


class TrainingConfig:
    """SAC training hyperparameters."""
    total_episodes      = 2000
    seed                = 42
    learning_rate_start = 3e-4
    learning_rate_end   = 1e-5
    lr_end_fraction     = 0.8        # fraction of training when LR reaches minimum
    buffer_size         = 500_000
    batch_size          = 256
    gradient_steps      = 2
    learning_starts     = 100
    net_arch_pi         = [64, 64]   # actor network layers
    net_arch_qf         = [128, 128] # critic network layers (larger capacity)
    target_entropy      = -0.5       # SAC entropy target
    max_grad_norm       = 10.0       # gradient clipping
    device              = 'cpu'
    checkpoint_dir      = './checkpoints'
    log_dir             = './logs'
    vis_update_interval = 5          # update UI every N steps


class CurriculumConfig:
    """Curriculum learning stage boundaries (by episode number)."""
    stage1_end = 500   # single obstacle, second-half of circle
    stage2_end = 1200  # two obstacles, min separation enforced
    # stage3: episodes 1201–2000 — fully random (full dataset)
