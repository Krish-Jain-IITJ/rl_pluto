# Multiple Obstacle Avoidance and Trajectory Tracking via Residual Reinforcement Learning on Arbitrary Control

This repository contains the implementation accompanying the paper **"Multiple Obstacle Avoidance and Trajectory Tracking via Residual Reinforcement Learning on Arbitrary Control"** by Krish Jain, Suhaib Md., and Anoop Jain (IIT Jodhpur).

A hybrid control architecture for unicycle robots that combines a classical Barrier Lyapunov Function (BLF) controller with an online Soft Actor-Critic (SAC) reinforcement learning agent, enabling safe and efficient multi-obstacle avoidance while tracking a circular reference trajectory.

---
<img width="1918" height="1020" alt="Screenshot 2026-05-06 194231" src="https://github.com/user-attachments/assets/9ba5aff1-0576-436c-abde-e5b9875b3054" />
## Overview

Traditional safety controllers such as Control Barrier Functions (CBF) and Barrier Lyapunov Functions (BLF) guarantee safety but tend to be overly conservative in cluttered environments, often forcing large, energy-wasting detours. Pure end-to-end RL controllers, on the other hand, lack verifiable safety guarantees and can be computationally expensive.

This project bridges the two approaches with a **residual reinforcement learning framework**:

- A **nominal BLF/vector-field controller** handles baseline trajectory tracking under obstacle-free conditions.
- A **SAC-based RL agent** is invoked only when an obstacle enters a predefined trigger zone, contributing a smooth corrective residual to the steering command.
- A **soft-gating function** blends the two control signals continuously, eliminating abrupt mode switches.
- The RL agent trains **online**, continuously refining its policy in real time as it interacts with novel obstacle configurations.

### Key Results

| Metric | Value |
|---|---|
| Path fidelity | > 98% |
| Collision-avoidance rate | 99.96% |
| Pre-training obstacle-avoidance success rate | 100% |
| Multi-obstacle training reward improvement | ~28% over 100 episodes |

---

## Architecture

### Total Control Law

The final angular velocity command applied to the unicycle agent is:

```
ω(t) = ω_nom(t) + σ(d_min(t)) · Δω_res(t)
```

- **ω_nom(t)** — nominal steering command from the BLF/guidance vector-field controller, guaranteeing convergence to the desired circular manifold M_d.
- **Δω_res(t)** — residual correction produced by the SAC actor network.
- **σ(d_min(t))** — soft-gating function (linear ramp) that activates the RL contribution only as the agent approaches an obstacle:
  - **Nominal Mode (σ = 0):** clearance ≥ `d_trigger` — RL is fully suppressed; the BLF controller alone governs motion.
  - **Reactive Mode (0 < σ ≤ 1):** clearance < `d_trigger` — the RL agent provides a smooth corrective nudge proportional to proximity.

### Safety Set Formulation

Obstacles are modeled as points surrounded by a circular **safety set** `S` of radius `r_safe`. Any encroachment of the agent within `r_safe` is treated as a collision during training, which lets the framework generalize to arbitrarily shaped real-world obstacles bounded by the safety envelope. The buffer between the safety boundary and the physical obstacle absorbs learning errors, model uncertainty, and unexpected disturbances.
<img width="528" height="878" alt="obs_avoid_end" src="https://github.com/user-attachments/assets/81c1f361-a4df-41c6-96d9-9bd38473df6d" />

### SAC Agent

| Component | Detail |
|---|---|
| Algorithm | Soft Actor-Critic (twin-critic, maximum-entropy) |
| Actor network | [64, 64] hidden layers |
| Critic networks | [128, 128] hidden layers (twin) |
| Control frequency | 1 ms (Δt) |
| Replay buffer | 500,000 transitions (off-policy) |
| Gradient updates per step | G = 2 |
| Gradient clipping | max norm 10.0 |
| Learning rate schedule | linear decay, 3×10⁻⁴ → 1×10⁻⁵ over first 80% of training |
| Target network update | Polyak averaging, τ = 0.005 |
| Hyperparameter tuning | Optuna, 100-trial multi-objective campaign |

#### State Space (7-D, continuous)

```
s_k = [e_k, ė_k, cos(α_k), Δx_k, Δy_k, ω_{k-1}, g_k]
```

- `e_k` — radial tracking error
- `ė_k` — derivative of tracking error
- `cos(α_k)` — heading alignment indicator
- `(Δx_k, Δy_k)` — obstacle position in agent body-frame Cartesian coordinates
- `ω_{k-1}` — previous control effort (rate feedback)
- `g_k` — binary gating flag (0 = nominal tracking, 1 = active obstacle evasion)
<img width="547" height="547" alt="2obstacles" src="https://github.com/user-attachments/assets/8c8bf1b0-24ab-48eb-927c-7672f5884c3e" />
#### Action Space

The actor outputs a bounded residual steering correction, shaped by:
- A **dynamic scaling gain** `Γ(ω_nom) = k·|ω_nom| + ε` (k ≈ 4.88, tuned via Optuna) ensuring sufficient RL authority near obstacles.
- **Vector-based half-plane partitioning** to determine left/right avoidance direction without expensive trigonometric calls.
- **Low-pass filtering** (β = 0.3) of the raw residual to suppress high-frequency chatter.
- **Amplitude saturation** to the actuator's physical limits (±ω_max).

### Reward Function

A multi-objective reward combines nine terms balancing tracking precision, control smoothness, obstacle avoidance, and recovery:

```
r_n = r_align + r_track + r_smooth + p_prox + p_head + r_conv + r_rec + p_crash + r_circle
```

| Term | Purpose |
|---|---|
| `r_align` | Rewards tangential heading alignment with the path |
| `r_track` | Penalizes squared radial tracking error |
| `r_smooth` | Penalizes jerky control (squared Δω) |
| `p_prox` | Exponential penalty for proximity to an obstacle (active only in reactive mode) |
| `p_head` | Penalizes heading directly toward an obstacle, scaled by inverse clearance |
| `r_conv` | Tiered bonus for tight steady-state tracking within the trigger region |
| `r_rec` | Rewards rapid error reduction after an evasion maneuver |
| `p_crash` | Severe non-terminal penalty (−500) for breaching the safety set |
| `r_circle` | Sparse completion bonus (+50) for surviving a full episode without divergence |

---

## Training Curriculum

Training follows a staged curriculum designed to prevent policy divergence in dense environments:

1. **Single-Obstacle Pre-training** (iterations 0–593) — establishes a stable coupling between the BLF controller and the residual policy in a simplified setting.
2. **Phase I — Dual-Obstacle Mastery** (episodes 0–99) — the agent learns to balance tracking with immediate avoidance.
3. **Phase II — Tri-Obstacle Adaptation** (episodes 100–199) — introduces overlapping trigger zones.
4. **Phase III — High-Density Navigation** (episodes 200–299) — four obstacles; agent shifts from wide detours to precise, surgical maneuvers.
5. **Phase IV — Generalized Autonomy** (episodes 300+) — evaluation on fully randomized obstacle configurations; crash count converges to zero.

---

## Repository Structure

> Update this section to match the actual repository layout.

```
.
├── envs/                 # Unicycle kinematics, obstacle/safety-set models, BLF controller
├── agents/                # SAC actor-critic implementation
├── training/              # Online training loop, curriculum scheduler, Optuna tuning scripts
├── configs/                # Hyperparameters and environment configuration files
├── results/                # Training curves, trajectory plots, logged metrics
└── README.md
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{jain2026residual,
  title   = {Multiple Obstacle Avoidance and Trajectory Tracking via Residual Reinforcement Learning on Arbitrary Control},
  author  = {Jain, Krish and Md., Suhaib and Jain, Anoop},
  journal = {IEEE},
  year    = {2026}
}
```

---

## Authors

- **Krish Jain** — Department of Electrical Engineering, IIT Jodhpur (b24ee1097@iitj.ac.in)
- **Suhaib Md.** — Department of Electrical Engineering, IIT Jodhpur (p23ee0011@iitj.ac.in)
- **Anoop Jain**, Senior Member, IEEE — Department of Electrical Engineering, IIT Jodhpur (anoopj@iitj.ac.in)

---

## License

Specify your license here (e.g., MIT, Apache 2.0).
