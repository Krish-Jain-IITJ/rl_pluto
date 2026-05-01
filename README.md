# RL Pluto: Deep Reinforcement Learning for Obstacle Avoidance

A PyQt5-based deep reinforcement learning system for autonomous obstacle avoidance in circular path tracking using Soft Actor-Critic (SAC) algorithm.

## Project Overview

This project implements a residual reinforcement learning controller that augments a Barrier Lyapunov Function (BLF) base controller with learned corrective actions to avoid obstacles while maintaining circular trajectory tracking.

**Key Features:**
- SAC (Soft Actor-Critic) algorithm for continuous control
- Real-time PyQt5 visualization with matplotlib integration
- Curriculum learning with 3 training stages
- VecNormalize wrapper for observation normalization
- CSV logging of training metrics
- Checkpoint saving during training

---

## Project Structure

```
pluto/
├── sim.py                    # Simulation environment (BLF control + unicycle dynamics)
├── rl_environment.py         # Gymnasium environment wrapper for RL training
├── rl_training.py            # SAC training loop with PyQt5 UI
├── rl_config.py              # Centralized configuration for all hyperparameters
├── rl_evaluate.py            # Evaluation script for trained models
├── obstacle_dataset.py        # Obstacle position dataset for training
├── test_env.py               # Unit tests for environment
├── test_sac.py               # SAC model testing
├── requirements.txt          # Python dependencies
├── logs/                     # Training logs (CSV format)
├── checkpoints/              # Saved model checkpoints
└── README.md                 # This file
```

---

## Installation

### Prerequisites
- Python 3.9+
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Krish-Jain-IITJ/rl_pluto.git
   cd pluto
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### Run Training with GUI

```bash
python rl_training.py
```

This launches the training window with:
- Real-time simulation visualization
- Statistics overlay (reward plots)
- Training controls (start/stop)
- Episode counter and progress bar

### Run Evaluation

```bash
python rl_evaluate.py
```

Tests a trained model checkpoint against the test dataset.

---

## Configuration

All hyperparameters are centralized in **`rl_config.py`**:

### Environment Config
```python
EnvConfig:
  max_obstacles       = 2          # Obstacles per episode
  crash_distance      = 0.25       # Crash threshold (meters)
  max_error           = 2.0        # Max tracking error (meters)
  action_gain_k       = 1.0        # Action gain
```

### Reward Config
```python
RewardConfig:
  alpha_alignment     = 0.3        # Heading alignment reward weight
  alpha_error         = 1.0        # Tracking error penalty
  alpha_smoothness    = 0.01       # Control smoothness penalty
  alpha_proximity     = 1.0        # Obstacle proximity penalty
  alpha_heading_obs   = 0.5        # Heading-toward-obstacle penalty
  crash_penalty       = -500.0     # Crash terminal penalty
  circle_bonus        = 50.0       # Circle completion bonus
```

### Training Config
```python
TrainingConfig:
  total_episodes      = 2000       # Total training episodes
  learning_rate_start = 3e-4       # Initial learning rate
  learning_rate_end   = 1e-5       # Final learning rate
  buffer_size         = 500_000    # Replay buffer size
  batch_size          = 256        # Training batch size
  gradient_steps      = 2          # Learning updates per step
```

### Curriculum Learning
```python
CurriculumConfig:
  stage1_end = 500    # Single obstacle (episodes 1-500)
  stage2_end = 1200   # Two obstacles (episodes 501-1200)
  # Stage 3: Full random complexity (episodes 1201-2000)
```

---

## Training Metrics

The training logs (saved in `logs/training_log_YYYYMMDD_HHMMSS.csv`) include:

| Metric | Description |
|--------|-------------|
| **episode** | Episode number |
| **reward** | Total reward for this episode |
| **avg_reward_50** | Rolling 50-episode average reward |
| **steps** | Number of simulation steps in episode |
| **crashed** | Whether agent hit an obstacle (True/False) |
| **min_d** | Minimum distance to any obstacle (meters) |

### Interpreting Results

- **Avg Reward Trend**: Should decrease initially (learning), then stabilize or improve
- **min_d Trend**: Should increase over time (agent learns to stay farther from obstacles)
- **Crash Rate**: Should decrease as training progresses
- **Steps per Episode**: Indicator of path efficiency; should stabilize around 800-900

### Example Output
```
[S1] Ep   10/500  Reward:    123.45  Avg50:    110.32  Steps:  150  Crashed:  no  d_min: 0.542m
```

- **[S1]** = Curriculum stage 1 (easy)
- **Ep 10/500** = Episode 10 out of 500 total
- **Avg50** = 50-episode rolling average
- **d_min** = Minimum obstacle distance

---

## Algorithm Details

### SAC (Soft Actor-Critic)

A state-of-the-art off-policy algorithm with:
- **Actor Network** (Policy): Learns what actions to take
- **Critic Networks** (Q-functions): Learn value estimates
- **Entropy Regularization**: Encourages exploration and prevents early convergence

### Network Architecture

```
Actor (π):
  Input: 8D observation [e, ė, cos(align), dx₁, dy₁, dx₂, dy₂, ω_prev]
  Hidden: [64, 64] neurons per layer
  Output: 1D continuous action [-1, 1]

Critic (Q):
  Input: 8D observation + 1D action
  Hidden: [128, 128] neurons per layer
  Output: Scalar Q-value
```

### Observation Space

8-dimensional vector per step:
```
[e, ė, cos(alignment), dx₁, dy₁, dx₂, dy₂, ω_prev]
```
- **e**: Current tracking error (radius - desired radius)
- **ė**: Error derivative
- **cos(alignment)**: Heading alignment to target direction
- **dx₁, dy₁**: Closest obstacle relative position
- **dx₂, dy₂**: Second closest obstacle relative position
- **ω_prev**: Previous angular velocity

### Action Space

1-dimensional continuous action:
```
a ∈ [-1, 1]  →  Δω = a × (k|ω_BLF| + 0.01)
```
Controls corrective angular velocity augmentation to base controller.

---

## Key Implementation Features

1. **State Machine Control**: Non-blocking training via Qt timers
2. **Off-Policy Learning**: Experience replay buffer (500k capacity)
3. **Observation Normalization**: VecNormalize wrapper for stability
4. **Gradient Clipping**: Max norm 10.0 to prevent exploding gradients
5. **Learning Rate Schedule**: Linear annealing from 3e-4 to 1e-5
6. **Checkpoint Saving**: Model snapshots during training

---

## Troubleshooting

### Issue: Training window doesn't appear
- Ensure PyQt5 is installed: `pip install PyQt5`
- Check matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`

### Issue: GPU out of memory
- Reduce `batch_size` in `rl_config.py`
- Reduce `buffer_size` for smaller replay buffer
- Note: Default is CPU-only (`device='cpu'`)

### Issue: Agent crashes frequently
- Increase `alpha_proximity` to penalize obstacle proximity more
- Decrease `crash_penalty` severity (less catastrophic)
- Add `alpha_safety_bonus` for safe distance rewards

---

## Future Improvements

- [ ] Multi-obstacle curriculum with higher densities
- [ ] Recurrent policy (LSTM) for temporal dependencies
- [ ] Distributed training (multiple environments)
- [ ] Transfer learning across different circle radii
- [ ] Real-world deployment on autonomous platforms

---

## References

- Christmann et al. (2022): Soft Actor-Critic (SAC) algorithm
- Gymnasium Documentation: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

---

## License

This project is part of academic research. For usage, contact the authors.

## Authors

- Krish Jain (IITJ)

---

**Last Updated:** May 2026  
**Training Status:** In Progress (Episode 17+)
