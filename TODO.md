# Deep RL Training System — Task Status

## Phase 1: Environment Setup
- [x] 1. Create custom Gymnasium environment (ObstacleDodgeEnv)
- [x] 2. Define 8D observation space: [e, ė, cos(align), dx₁, dy₁, dx₂, dy₂, ω_prev]
       ↳ UPGRADED from 6D: both obstacles always visible to agent (closest-first sorted)
- [x] 3. Define reward formula — exponential proximity barrier + heading-toward-obs penalty
       ↳ FIXED: old linear d_min reward replaced; circle-completion sparse bonus added
- [x] 4. Define termination conditions (crash at 0.25m & |e| > 2.0m)

## Phase 2: SAC Agent Configuration
- [x] 5. Configure SAC with MlpPolicy, pi=[64,64], qf=[128,128] (separate actor/critic)
- [x] 6. Implement error-gated action: Δω = a × (k|ω_blf| + 0.01)
- [x] 7. Implement final control: ω_total = ω_blf + Δω (clamped to ±w_max)
- [x] 8. VecNormalize wrapper for observation/reward normalisation
- [x] 9. Linear learning-rate schedule (3e-4 → 1e-5)
- [x] 10. gradient_steps=2, buffer_size=500k, learning_starts=100

## Phase 3: UI Integration
- [x] 11. TrainingOrchestrator with Qt signals — decouples training from UI
- [x] 12. Step-by-step timer state machine — FIXED: no more UI freeze
- [x] 13. StatsOverlay with per-episode and rolling-50 avg reward plots
- [x] 14. Policy loss plot in StatsOverlay
- [x] 15. Obstacle proximity color-coding (red/orange/yellow/green)
- [x] 16. d_min progress bar with dynamic color in StatsOverlay
- [x] 17. TRAIN / STOP TRAIN buttons added to panel via stored self.panel ref

## Phase 4: Training Execution
- [x] 18. Curriculum dataset: Stage1 (single obs) → Stage2 (separated) → Stage3 (random)
- [x] 19. 2000-episode training loop via TrainingOrchestrator
- [x] 20. Terminal output per episode (stage, reward, avg50, steps, crash, d_min)
- [x] 21. CSV log to logs/training_log_TIMESTAMP.csv
- [x] 22. Final model saved to checkpoints/sac_obstacle_dodge_final.zip

## Phase 5: Evaluation
- [x] 23. rl_evaluate.py — load checkpoint, run deterministic eval, print stats

## Bug Fixes Applied
- [x] BUG-1 CRITICAL: IndentationError in rl_training.py (misplaced comment)
- [x] BUG-2 HIGH:     Double state integration — now explicitly uses `_` for discarded BLF state
- [x] BUG-3 HIGH:     TrainingCallback used wrong locals key — fixed to use ep_info_buffer
- [x] BUG-4 HIGH:     Blocking episode loop in timer — replaced with step-by-step state machine
- [x] BUG-5 MEDIUM:   _get_observation() stale prev_error on reset — initialised from actual e0
- [x] BUG-6 MEDIUM:   Panel layout accessed by fragile positional index — uses self.panel ref
- [x] BUG-7 LOW:      Obstacle visualization fallback placed obs off-circle — now on-circle
- [x] BUG-8 LOW:      No minimum obstacle separation — enforced in obstacle_dataset.py
- [x] BUG-9 LOW:      No random seed in dataset — seed=42 default added

## New Files Added
- [x] rl_config.py     — Centralized EnvConfig, RewardConfig, TrainingConfig, CurriculumConfig
- [x] rl_evaluate.py   — Deterministic evaluation of saved checkpoints
- [x] checkpoints/     — Directory for model checkpoints (auto-created)
- [x] logs/            — Directory for CSV training logs (auto-created)
