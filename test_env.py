"""
test_env.py — Quick smoke test for ObstacleDodgeEnv.
Updated for 8D observation space.
"""

from rl_environment import ObstacleDodgeEnv
from obstacle_dataset import training_data
from sim import SimParams
import numpy as np

print("Creating environment...")
env = ObstacleDodgeEnv(training_data=training_data, sim_params=SimParams())

# ── Reset ────────────────────────────────────────────────────────────────────
obs, info = env.reset()
assert obs.shape == (8,), f"Expected obs shape (8,), got {obs.shape}"
print(f"Observation shape : {obs.shape}  ✓")
print(f"Initial obs       : {np.round(obs, 4)}")

# Labels for readability
labels = ['e', 'ė', 'cos(align)', 'dx₁', 'dy₁', 'dx₂', 'dy₂', 'ω_prev']
for name, val in zip(labels, obs):
    print(f"  {name:12s} = {val:+.4f}")

# ── Steps ────────────────────────────────────────────────────────────────────
print("\nRunning 10 steps with zero action...")
for i in range(10):
    action = np.array([0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (8,), f"Obs shape wrong at step {i}"
    print(
        f"  Step {i+1:2d}: reward={reward:+8.4f}  "
        f"e={info['e']:+.4f}  d_min={info['d_min']:.4f}  "
        f"ω_total={info['omega_total']:+.4f}  "
        f"{'DONE' if terminated else 'TRUNC' if truncated else '    '}"
    )
    if terminated or truncated:
        break

# ── Action extremes ──────────────────────────────────────────────────────────
print("\nTesting action space bounds...")
env.current_episode = 1
obs, _ = env.reset()
for a_val in [-1.0, 0.0, 1.0]:
    action = np.array([a_val])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  a={a_val:+.1f}  ω_total={info['omega_total']:+.4f}  reward={reward:+8.4f}")

print("\nEnvironment test PASSED ✓")
