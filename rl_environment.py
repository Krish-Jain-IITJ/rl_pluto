"""
rl_environment.py — Custom Gymnasium Environment for Obstacle Avoidance
Fixes applied (from ideation doc):
  - Bug #1: Double-integration clarified — BLF state discarded intentionally
  - Bug #2: Obstacle reward replaced with exponential proximity penalty
  - Bug #3: _get_observation() no longer uses stale prev_error (reset correctly)
  - Bug #4: Episode counter managed externally; reset() no longer auto-increments
  - Observation expanded to 8D: both obstacles always visible to agent
  - Improved reward: exponential proximity barrier + heading-toward-obstacle penalty
  - Sparse circle-completion bonus added
  - Heading-toward-obstacle penalty added
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from sim import SimParams, step
from rl_config import EnvConfig, RewardConfig


class ObstacleDodgeEnv(gym.Env):
    """
    Gymnasium environment for residual-RL obstacle dodging on a BLF base controller.

    Observation (8D):
        [e, ė, cos(alignment), dx₁, dy₁, dx₂, dy₂, ω_prev]
        Obstacles sorted by distance — closest first.
        If only one unique obstacle (Stage 1 curriculum), both slots hold the same values.

    Action (1D):
        a ∈ [-1, 1]  — normalized corrective angular velocity scaling

    Control law:
        Δω = a × (k|ω_blf| + 0.01)
        ω_total = ω_blf + Δω  (clamped to ±w_max)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        training_data: np.ndarray,
        sim_params: Optional[SimParams] = None,
        render_mode: str = 'human',
    ):
        super().__init__()

        self.training_data = training_data
        self.sim_params = sim_params if sim_params is not None else SimParams()
        self.render_mode = render_mode

        # Episode indexing — managed externally (training loop) or cycles here
        self.current_episode = 0
        self.max_episodes = len(training_data)

        # ── Spaces ────────────────────────────────────────────────────────────
        # 8D observation: [e, ė, cos(alignment), dx₁, dy₁, dx₂, dy₂, ω_prev]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        # Action: scalar ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ── Internal state ────────────────────────────────────────────────────
        self.state        = None   # (x, y, theta)
        self.prev_omega   = 0.0
        self.prev_error   = 0.0
        self.t_elapsed    = 0.0
        self.episode_reward = 0.0
        self.obstacles    = None   # np.ndarray (N, 2)
        self.min_distance = np.inf
        self.extras       = None   # last BLF extras dict

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset for a new episode. Episode index must be set before calling."""
        super().reset(seed=seed)

        # Load obstacle positions for this episode
        ep_idx = self.current_episode % self.max_episodes
        self.obstacles = self.training_data[ep_idx].copy()  # shape (N, 2)

        # Reset simulation state
        p = self.sim_params
        self.state = (p.x0, p.y0, p.theta0)

        # FIXED Bug #3: reset prev_error to actual initial error, not 0
        r0 = np.linalg.norm(np.array([p.x0, p.y0]))
        self.prev_error = r0 - p.r_d

        self.prev_omega     = 0.0
        self.t_elapsed      = 0.0
        self.episode_reward = 0.0
        self.min_distance   = np.inf
        self.extras         = None

        obs = self._get_observation()
        return obs, {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one simulation step.

        Residual control law:
            1. Call BLF step() to get ω_blf and extras — DISCARD integrated state.
            2. Compute Δω = a × (k|ω_blf| + 0.01).
            3. ω_total = ω_blf + Δω, clamped to ±w_max.
            4. Apply our own Euler integration with ω_total.
        """
        a = float(action[0])
        p = self.sim_params
        x, y, theta = self.state

        # FIXED Bug #1: call BLF only for extras; explicitly discard its integrated state
        _, extras = step(self.state, p)
        self.extras = extras

        e      = extras['e']
        w_blf  = extras['w']

        # Error-gated corrective action
        delta_omega = a * (EnvConfig.action_gain_k * abs(w_blf) + 0.01)
        omega_total = w_blf + delta_omega
        omega_total = float(np.clip(omega_total, -p.w_max, p.w_max))

        # Euler integration with the TOTAL angular velocity (our authority)
        v    = p.v
        xn   = x     + v * np.cos(theta) * p.dt
        yn   = y     + v * np.sin(theta) * p.dt
        thetan = theta + omega_total * p.dt
        self.state = (xn, yn, thetan)

        # Error derivative
        error_derivative = (e - self.prev_error) / p.dt
        self.prev_error  = e

        # Alignment angle (heading vs tangent to circle)
        alignment_cos = self._compute_alignment_cos(x, y, theta)

        # Closest obstacle vectors (sorted by distance → closest first)
        sorted_obs = self._sort_obstacles_by_distance(x, y)
        dx1, dy1 = sorted_obs[0]
        dx2, dy2 = sorted_obs[1] if len(sorted_obs) > 1 else sorted_obs[0]
        d_min = float(np.hypot(dx1, dy1))
        self.min_distance = d_min

        # Heading-toward-obstacle penalty
        heading_vec = np.array([np.cos(theta), np.sin(theta)])
        obs1_vec    = np.array([dx1, dy1])
        obs1_norm   = np.linalg.norm(obs1_vec)
        if obs1_norm > 1e-9:
            cos_toward = float(np.dot(heading_vec, obs1_vec / obs1_norm))
        else:
            cos_toward = 0.0

        # ── Reward ────────────────────────────────────────────────────────────
        omega_change = omega_total - self.prev_omega

        # Exponential proximity penalty: 0 far away, approaches -1 near obstacle
        proximity_penalty = RewardConfig.alpha_proximity * (
            np.exp(-3.0 * d_min) - 1.0
        )

        # Heading-toward-obstacle penalty (only when pointing toward it)
        heading_penalty = -RewardConfig.alpha_heading_obs * max(0.0, cos_toward) / (d_min + 0.1)

        reward = (
            RewardConfig.alpha_alignment  * alignment_cos
            + RewardConfig.alpha_error    * (-e ** 2)
            + RewardConfig.alpha_smoothness * (-omega_change ** 2)
            + proximity_penalty
            + heading_penalty
        )

        # ── Termination ───────────────────────────────────────────────────────
        terminated = False
        truncated  = False

        if d_min < EnvConfig.crash_distance:
            reward     = RewardConfig.crash_penalty
            terminated = True

        if abs(e) > EnvConfig.max_error:
            terminated = True

        # Time limit: one complete circle
        self.t_elapsed += p.dt
        circle_period = 2 * np.pi * p.r_d / p.v
        if self.t_elapsed >= circle_period:
            if not terminated:
                reward += RewardConfig.circle_bonus   # sparse success bonus
            truncated = True

        # Update previous omega
        self.prev_omega = omega_total

        # Accumulate
        self.episode_reward += reward

        obs = self._get_observation()

        info = {
            'e':              e,
            'error_derivative': error_derivative,
            'omega_blf':      w_blf,
            'omega_total':    omega_total,
            'd_min':          d_min,
            'episode_reward': self.episode_reward,
            'obstacles':      self.obstacles,
            'x':              xn,
            'y':              yn,
            'theta':          thetan,
            'crashed':        (d_min < EnvConfig.crash_distance and terminated),
        }

        return obs, reward, terminated, truncated, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Compute 8D observation:
            [e, ė, cos(alignment), dx₁, dy₁, dx₂, dy₂, ω_prev]
        Obstacles sorted closest-first; second slot duplicated if only one obstacle.
        """
        x, y, theta = self.state
        r    = np.linalg.norm(np.array([x, y]))
        e    = r - self.sim_params.r_d
        edot = (e - self.prev_error) / self.sim_params.dt  # uses CURRENT prev_error

        alignment_cos = self._compute_alignment_cos(x, y, theta)

        sorted_obs = self._sort_obstacles_by_distance(x, y)
        dx1, dy1 = sorted_obs[0]
        dx2, dy2 = sorted_obs[1] if len(sorted_obs) > 1 else sorted_obs[0]

        return np.array(
            [e, edot, alignment_cos, dx1, dy1, dx2, dy2, self.prev_omega],
            dtype=np.float32,
        )

    def _compute_alignment_cos(self, x: float, y: float, theta: float) -> float:
        """Cosine of alignment between heading and tangent direction."""
        norm_r = np.hypot(x, y)
        if norm_r < 1e-9:
            return 0.0
        phi = np.arctan2(y, x)
        alignment = theta - phi - np.pi / 2
        return float(np.cos(alignment))

    def _sort_obstacles_by_distance(self, x: float, y: float):
        """
        Return list of (dx, dy) vectors to obstacles, sorted closest-first.
        Each entry is the vector FROM agent TO obstacle.
        """
        if self.obstacles is None or len(self.obstacles) == 0:
            return [(0.0, 0.0), (0.0, 0.0)]

        pos = np.array([x, y])
        diffs = self.obstacles - pos   # (N, 2)
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)
        return [(float(diffs[i, 0]), float(diffs[i, 1])) for i in order]

    # ── Render / Close ────────────────────────────────────────────────────────

    def render(self):
        pass  # Rendering handled by PyQt5 in rl_training.py

    def close(self):
        pass
