"""
test_sac.py — Quick smoke test for SAC model creation and prediction.
Updated to use TrainingConfig and 8D observation space.
"""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn

from rl_environment import ObstacleDodgeEnv
from obstacle_dataset import training_data
from sim import SimParams
from rl_config import TrainingConfig
import numpy as np

print("Creating environment...")
env = ObstacleDodgeEnv(training_data=training_data, sim_params=SimParams())

# VecNormalize wrapper (matches rl_training.py)
vec_env = DummyVecEnv([lambda: ObstacleDodgeEnv(
    training_data=training_data, sim_params=SimParams()
)])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# LR schedule
lr_schedule = get_linear_fn(
    start=TrainingConfig.learning_rate_start,
    end=TrainingConfig.learning_rate_end,
    end_fraction=TrainingConfig.lr_end_fraction,
)

print("Creating SAC model...")
model = SAC(
    'MlpPolicy',
    vec_env,
    learning_rate=lr_schedule,
    policy_kwargs=dict(
        net_arch=dict(
            pi=TrainingConfig.net_arch_pi,
            qf=TrainingConfig.net_arch_qf,
        )
    ),
    buffer_size=TrainingConfig.buffer_size,
    batch_size=TrainingConfig.batch_size,
    gradient_steps=TrainingConfig.gradient_steps,
    learning_starts=TrainingConfig.learning_starts,
    ent_coef='auto',
    target_entropy=TrainingConfig.target_entropy,
    verbose=0,
    device=TrainingConfig.device,
)

print(f"SAC model created  ✓")
print(f"  Observation space : {model.observation_space}")
print(f"  Action space      : {model.action_space}")
print(f"  Policy network    : pi={TrainingConfig.net_arch_pi}  "
      f"qf={TrainingConfig.net_arch_qf}")

# Test prediction on raw env (8D obs)
obs, _ = env.reset()
assert obs.shape == (8,), f"Expected (8,), got {obs.shape}"
action, _ = model.predict(obs, deterministic=False)
print(f"  Sample action     : {action}  (shape {action.shape})")
assert action.shape == (1,), f"Expected action shape (1,), got {action.shape}"

print("\nSAC test PASSED ✓")
