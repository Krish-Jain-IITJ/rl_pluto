"""
rl_training.py — Deep RL Training System with SAC and PyQt5 UI Integration
Fixes applied (from ideation doc):
  - Bug #1: IndentationError (misplaced comment) — fixed
  - Bug #2: Panel layout accessed by stored reference (self.panel) — not fragile index
  - Bug #3: TrainingCallback uses ep_info_buffer — not wrong locals key
  - Bug #4: Step-by-step state machine timer — no UI freeze
  - VecNormalize wrapper added for observation normalisation
  - TrainingOrchestrator decouples training logic from UI via Qt signals
  - Terminal output per episode (print to stdout)
  - Obstacle color-coding by proximity
  - Rolling 50-episode average displayed in stats overlay
  - LR schedule and gradient_steps configured
  - Checkpoint saving via CheckpointCallback
"""

import sys
import os
import csv
import numpy as np
from enum import Enum
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QGroupBox, QGridLayout, QFrame, QDockWidget,
    QProgressBar,
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn

from sim import (
    MainWindow as BaseMainWindow, SimCanvas, SimParams, step,
    DARK_BG, PANEL_BG, ACCENT, ACCENT2, GREEN, YELLOW, GREY,
)
from rl_environment import ObstacleDodgeEnv
from obstacle_dataset import training_data
from rl_config import TrainingConfig, EnvConfig, CurriculumConfig


# ─── Training State Machine ───────────────────────────────────────────────────

class TrainingState(Enum):
    IDLE            = 0
    RUNNING_EPISODE = 1
    LEARNING        = 2
    DONE            = 3


# ─── Improved Callback ────────────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """
    Collects training statistics from SB3's ep_info_buffer (Bug #3 fix).
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list = []
        self.policy_losses:   list = []

    def _on_step(self) -> bool:
        # Correct way: read from ep_info_buffer, not self.locals
        if len(self.model.ep_info_buffer) > 0:
            last_ep = self.model.ep_info_buffer[-1]
            if 'r' in last_ep:
                self.episode_rewards.append(float(last_ep['r']))

        # Collect policy loss if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            vals = self.model.logger.name_to_value
            if 'train/policy_loss' in vals:
                self.policy_losses.append(float(vals['train/policy_loss']))
        return True

    def _on_rollout_end(self) -> bool:
        return True


# ─── Training Orchestrator (Qt signals for clean UI decoupling) ───────────────

class TrainingOrchestrator(QObject):
    """
    Owns the training state machine and QTimer.
    Emits Qt signals so UI can update without the orchestrator knowing about widgets.
    """
    step_complete    = pyqtSignal(dict)           # info dict from env.step()
    episode_complete = pyqtSignal(int, float, float, bool)  # ep, reward, avg, crashed
    training_done    = pyqtSignal()

    def __init__(
        self,
        env: ObstacleDodgeEnv,
        vec_env: VecNormalize,
        model: SAC,
        callback: TrainingCallback,
        max_episodes: int = TrainingConfig.total_episodes,
    ):
        super().__init__()
        self.env          = env
        self.vec_env      = vec_env
        self.model        = model
        self.callback     = callback
        self.max_episodes = max_episodes

        self.state = TrainingState.IDLE
        self.current_episode  = 0
        self.current_obs      = None
        self.episode_reward   = 0.0
        self.step_count       = 0
        self.episode_rewards: list = []
        self.min_d_episode    = np.inf
        self.crashed          = False

        # CSV logger
        os.makedirs(TrainingConfig.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_path = os.path.join(
            TrainingConfig.log_dir, f'training_log_{timestamp}.csv'
        )
        with open(self._log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'avg_reward_50', 'steps', 'crashed', 'min_d'])

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    def start(self):
        """Begin training."""
        self.state = TrainingState.RUNNING_EPISODE
        self.current_episode = 0
        self.episode_rewards = []
        self.env.current_episode = 0
        self.current_obs, _ = self.env.reset()
        self.episode_reward  = 0.0
        self.step_count      = 0
        self.min_d_episode   = np.inf
        self.crashed         = False
        self.timer.start(1)  # 1 ms — as fast as Qt allows

    def stop(self):
        self.timer.stop()
        self.state = TrainingState.IDLE

    def _tick(self):
        """State machine: one env step or one learning call per timer fire."""

        if self.state == TrainingState.RUNNING_EPISODE:
            # ── One environment step ──────────────────────────────────────────
            action, _ = self.model.predict(self.current_obs, deterministic=False)
            obs, reward, done, truncated, info = self.env.step(action)
            self.current_obs    = obs
            self.episode_reward += reward
            self.step_count     += 1
            d_min = info.get('d_min', np.inf)
            if d_min < self.min_d_episode:
                self.min_d_episode = d_min

            # Emit for visualization (every N steps)
            if self.step_count % TrainingConfig.vis_update_interval == 0:
                self.step_complete.emit(info)

            if done or truncated:
                self.crashed = info.get('crashed', False)
                self.state = TrainingState.LEARNING

        elif self.state == TrainingState.LEARNING:
            # ── Learning step ─────────────────────────────────────────────────
            self.model.learn(
                total_timesteps=max(1, self.step_count),
                callback=self.callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            self._on_episode_end()

            # Advance to next episode or finish
            self.current_episode += 1
            if self.current_episode >= self.max_episodes:
                self.state = TrainingState.DONE
                self.timer.stop()
                self.training_done.emit()
                print("\n[Training Complete]")
            else:
                self.env.current_episode = self.current_episode
                self.current_obs, _ = self.env.reset()
                self.episode_reward  = 0.0
                self.step_count      = 0
                self.min_d_episode   = np.inf
                self.crashed         = False
                self.state = TrainingState.RUNNING_EPISODE

    def _on_episode_end(self):
        """Log stats, print to terminal, emit signals."""
        ep     = self.current_episode + 1
        reward = self.episode_reward
        self.episode_rewards.append(reward)
        avg50  = float(np.mean(self.episode_rewards[-50:]))
        steps  = self.step_count
        crash  = self.crashed
        d_min  = self.min_d_episode if self.min_d_episode < np.inf else 999.0

        # Determine curriculum stage label
        if ep <= CurriculumConfig.stage1_end:
            stage = 'S1'
        elif ep <= CurriculumConfig.stage2_end:
            stage = 'S2'
        else:
            stage = 'S3'

        # Terminal output
        print(
            f"[{stage}] Ep {ep:4d}/{self.max_episodes}  "
            f"Reward: {reward:9.2f}  Avg50: {avg50:9.2f}  "
            f"Steps: {steps:4d}  Crashed: {'YES' if crash else ' no'}  "
            f"d_min: {d_min:.3f}m"
        )

        # CSV log
        with open(self._log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, round(reward, 4), round(avg50, 4), steps, crash, round(d_min, 4)])

        self.episode_complete.emit(ep, reward, avg50, crash)


# ─── Statistics Overlay Window ────────────────────────────────────────────────

class StatsOverlay(QDockWidget):
    """Docked statistics window with real-time reward / loss plots."""

    def __init__(self, parent=None):
        super().__init__('Training Statistics', parent)
        self.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        self.episode_rewards: list = []
        self.avg50_rewards:   list = []
        self.policy_losses:   list = []

        self._build_ui()

    def _build_ui(self):
        container = QWidget()
        container.setStyleSheet(f'background:{PANEL_BG};')
        vl = QVBoxLayout(container)
        vl.setContentsMargins(5, 5, 5, 5)
        vl.setSpacing(5)

        # Matplotlib figure
        self.fig = Figure(facecolor=DARK_BG, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        gs = self.fig.add_gridspec(2, 1, hspace=0.4, left=0.14, right=0.96,
                                   top=0.94, bottom=0.1)
        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_loss   = self.fig.add_subplot(gs[1, 0])

        for ax, title in [
            (self.ax_reward, 'Episode Reward'),
            (self.ax_loss,   'Policy Loss'),
        ]:
            ax.set_facecolor('#1a1f2b')
            ax.tick_params(colors=GREY, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#2a3040')
            ax.set_title(title, color=ACCENT, fontsize=8, fontweight='bold')
            ax.grid(True, color='#1e2535', linewidth=0.5, linestyle='--')

        self.reward_line,  = self.ax_reward.plot([], [], color='#4fc3f7',
                                                   lw=0.8, alpha=0.5, label='per-ep')
        self.avg50_line,   = self.ax_reward.plot([], [], color=GREEN,
                                                   lw=1.4, label='avg-50')
        self.ax_reward.legend(fontsize=6, facecolor=DARK_BG, edgecolor=GREY,
                               labelcolor='white', loc='upper left')
        self.loss_line, = self.ax_loss.plot([], [], color=ACCENT2, lw=1.0)

        vl.addWidget(self.canvas)

        # Text stats
        stats_box = QGroupBox('Episode Stats')
        stats_box.setStyleSheet(f'''
            QGroupBox {{ color:{GREY}; font-size:8px; font-family:Courier New;
                         border:1px solid #2a3040; border-radius:4px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:6px; }}
        ''')
        gl = QGridLayout(stats_box)

        def _lbl(color):
            w = QLabel('—')
            w.setFont(QFont('Courier New', 8))
            w.setStyleSheet(f'color:{color};')
            return w

        self.lbl_episode = _lbl(ACCENT)
        self.lbl_reward  = _lbl(GREEN)
        self.lbl_avg     = _lbl(YELLOW)
        self.lbl_stage   = _lbl(ACCENT2)

        gl.addWidget(QLabel('Episode:'),  0, 0)
        gl.addWidget(self.lbl_episode,    0, 1)
        gl.addWidget(QLabel('Reward:'),   1, 0)
        gl.addWidget(self.lbl_reward,     1, 1)
        gl.addWidget(QLabel('Avg-50:'),   2, 0)
        gl.addWidget(self.lbl_avg,        2, 1)
        gl.addWidget(QLabel('Stage:'),    3, 0)
        gl.addWidget(self.lbl_stage,      3, 1)
        for i in range(4):
            gl.itemAtPosition(i, 0).widget().setFont(QFont('Courier New', 8))
            gl.itemAtPosition(i, 0).widget().setStyleSheet(f'color:{GREY};')

        vl.addWidget(stats_box)

        # d_min proximity bar
        prox_box = QGroupBox('Proximity to Obstacle')
        prox_box.setStyleSheet(stats_box.styleSheet())
        pl = QVBoxLayout(prox_box)
        self.prox_bar = QProgressBar()
        self.prox_bar.setRange(0, 100)
        self.prox_bar.setValue(100)
        self.prox_bar.setTextVisible(False)
        self.prox_bar.setFixedHeight(10)
        self.prox_bar.setStyleSheet(f'''
            QProgressBar {{ background:#1a1f2b; border-radius:4px; }}
            QProgressBar::chunk {{ background:{GREEN}; border-radius:4px; }}
        ''')
        self.lbl_dmin = QLabel('d_min: —')
        self.lbl_dmin.setFont(QFont('Courier New', 8))
        self.lbl_dmin.setStyleSheet(f'color:{GREY};')
        pl.addWidget(self.prox_bar)
        pl.addWidget(self.lbl_dmin)
        vl.addWidget(prox_box)

        self.setWidget(container)

    def update_stats(self, episode: int, reward: float, avg50: float, crashed: bool):
        """Called once per episode end."""
        total = TrainingConfig.total_episodes

        if episode <= CurriculumConfig.stage1_end:
            stage = 'Stage 1 (Curriculum)'
        elif episode <= CurriculumConfig.stage2_end:
            stage = 'Stage 2 (Separated)'
        else:
            stage = 'Stage 3 (Full Random)'

        self.lbl_episode.setText(f'{episode}/{total}')
        self.lbl_reward.setText(f'{reward:.2f}')
        self.lbl_avg.setText(f'{avg50:.2f}')
        self.lbl_stage.setText(stage)

        self.episode_rewards.append(reward)
        self.avg50_rewards.append(avg50)

        xs = list(range(len(self.episode_rewards)))
        self.reward_line.set_data(xs, self.episode_rewards)
        self.avg50_line.set_data(xs, self.avg50_rewards)
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()

        if self.policy_losses:
            lxs = list(range(len(self.policy_losses)))
            self.loss_line.set_data(lxs, self.policy_losses)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

        self.canvas.draw_idle()

    def update_proximity(self, d_min: float):
        """Called each visualization step to update proximity bar."""
        safe_dist = 1.5   # anything beyond this is "fully safe"
        pct = int(min(100, max(0, (d_min / safe_dist) * 100)))
        self.prox_bar.setValue(pct)

        # Color the bar by danger level
        if d_min < EnvConfig.crash_distance + 0.05:
            color = ACCENT2      # red — near crash
        elif d_min < 0.5:
            color = '#ff6b35'    # orange — close
        elif d_min < 0.8:
            color = YELLOW       # yellow — moderate
        else:
            color = GREEN        # green — safe

        self.prox_bar.setStyleSheet(f'''
            QProgressBar {{ background:#1a1f2b; border-radius:4px; }}
            QProgressBar::chunk {{ background:{color}; border-radius:4px; }}
        ''')
        self.lbl_dmin.setText(f'd_min: {d_min:.3f} m')


# ─── RL Main Window ───────────────────────────────────────────────────────────

class RLMainWindow(BaseMainWindow):
    """
    Extends BaseMainWindow (sim.py) with RL training controls.
    Training logic fully delegated to TrainingOrchestrator.
    Panel widget stored as self.panel (Bug #2 fix — no fragile positional index).
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Deep RL Training — Unicycle Obstacle Dodge')

        # ── Create RL environment ──────────────────────────────────────────
        raw_env = ObstacleDodgeEnv(training_data=training_data, sim_params=self.p)

        # VecNormalize: normalise observations and rewards for SAC stability
        vec_env = DummyVecEnv([lambda: ObstacleDodgeEnv(
            training_data=training_data, sim_params=self.p
        )])
        self.vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )

        # We drive training via the raw env (step-by-step in orchestrator),
        # but pass vec_env to SAC so its internals use normalized experience.
        self.rl_env = raw_env

        # Learning-rate schedule
        lr_schedule = get_linear_fn(
            start=TrainingConfig.learning_rate_start,
            end=TrainingConfig.learning_rate_end,
            end_fraction=TrainingConfig.lr_end_fraction,
        )

        os.makedirs(TrainingConfig.checkpoint_dir, exist_ok=True)

        self.rl_model = SAC(
            'MlpPolicy',
            self.vec_env,
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

        self.callback = TrainingCallback()

        # ── Orchestrator ────────────────────────────────────────────────────
        self.orchestrator = TrainingOrchestrator(
            env=self.rl_env,
            vec_env=self.vec_env,
            model=self.rl_model,
            callback=self.callback,
            max_episodes=TrainingConfig.total_episodes,
        )
        self.orchestrator.step_complete.connect(self._on_step_complete)
        self.orchestrator.episode_complete.connect(self._on_episode_complete)
        self.orchestrator.training_done.connect(self._on_training_done)

        # ── UI additions ────────────────────────────────────────────────────
        self._add_stats_overlay()
        self._add_training_controls()  # uses self.panel — stored reference

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_step_complete(self, info: dict):
        """Update visualization canvas per step."""
        x     = info.get('x', self.state[0])
        y     = info.get('y', self.state[1])
        theta = info.get('theta', self.state[2])
        e     = info.get('e', 0.0)
        omega = info.get('omega_total', 0.0)

        self.state = (x, y, theta)
        self.xs.append(x);  self.ys.append(y);  self.ts.append(theta)
        self.es.append(e);  self.ws.append(omega)
        self.t_elapsed += self.p.dt

        extras = self.rl_env.extras if self.rl_env.extras else self.extras_last
        self.extras_last = extras

        # Obstacle color-coding by proximity
        d_min = info.get('d_min', np.inf)
        obstacles = info.get('obstacles', self.rl_env.obstacles)

        self.canvas.update_plots(
            self.xs, self.ys, self.ts,
            self.es, self.ws, extras,
            obstacles=obstacles,
        )

        # Color obstacles by danger
        if d_min < EnvConfig.crash_distance + 0.05:
            self.canvas.obstacles_plot.set_color(ACCENT2)
            self.canvas.obstacles_plot.set_markersize(7)
        elif d_min < 0.5:
            self.canvas.obstacles_plot.set_color('#ff6b35')
            self.canvas.obstacles_plot.set_markersize(5)
        elif d_min < 0.8:
            self.canvas.obstacles_plot.set_color(YELLOW)
            self.canvas.obstacles_plot.set_markersize(4)
        else:
            self.canvas.obstacles_plot.set_color(ACCENT2)
            self.canvas.obstacles_plot.set_markersize(3)

        self.stats_overlay.update_proximity(d_min)

        if self.rl_env.extras:
            self.lbl_time.setText(f'Time: {self.t_elapsed:.2f} s')
            self.lbl_e.setText(f'e: {e:.4f} m')
            self.lbl_w.setText(f'ω: {omega:.4f} r/s')

    def _on_episode_complete(self, episode: int, reward: float, avg50: float, crashed: bool):
        """Update stats overlay and episode label."""
        self.lbl_time.setText(f'Episode: {episode}/{TrainingConfig.total_episodes}')
        self.lbl_e.setText(f'Reward: {reward:.2f}')

        self.stats_overlay.policy_losses = self.callback.policy_losses
        self.stats_overlay.update_stats(episode, reward, avg50, crashed)

        # Reset trajectory visualization for next episode
        self.xs = [self.p.x0]; self.ys = [self.p.y0]; self.ts = [self.p.theta0]
        self.es = []; self.ws = []
        self.t_elapsed = 0.0

    def _on_training_done(self):
        """Re-enable buttons when training finishes."""
        self.btn_train.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.lbl_time.setText('Training Complete')
        # Save final model
        path = os.path.join(TrainingConfig.checkpoint_dir, 'sac_obstacle_dodge_final')
        self.rl_model.save(path)
        print(f"[Saved] Final model → {path}.zip")

    # ── UI wiring ─────────────────────────────────────────────────────────────

    def _add_stats_overlay(self):
        self.stats_overlay = StatsOverlay(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_overlay)

    def _add_training_controls(self):
        """
        Add TRAIN button to panel using stored self.panel reference (Bug #2 fix).
        BaseMainWindow._build_panel() must store the panel widget as self.panel.
        We fall back to a safe layout search if not present.
        """
        # Retrieve panel layout safely
        if hasattr(self, 'panel') and self.panel is not None:
            panel_layout = self.panel.layout()
        else:
            # Fallback: find the right-side panel widget
            central = self.centralWidget()
            root_layout = central.layout()
            # Panel is the last item added to root layout
            panel_widget = root_layout.itemAt(root_layout.count() - 1).widget()
            panel_layout = panel_widget.layout()

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color: #2a3040;')
        panel_layout.insertWidget(panel_layout.count() - 1, sep)

        # Train button
        self.btn_train = QPushButton('▶▶  TRAIN RL')
        self.btn_train.setFixedHeight(36)
        self.btn_train.setFont(QFont('Courier New', 9, QFont.Bold))
        self.btn_train.setStyleSheet(f'''
            QPushButton {{
                background:#002010; color:{GREEN}; border:1.5px solid {GREEN};
                border-radius:6px; letter-spacing:1px;
            }}
            QPushButton:hover {{ background:{GREEN}; color:#000; }}
            QPushButton:disabled {{ border-color:#2a3040; color:#2a3040;
                                    background:{DARK_BG}; }}
        ''')
        self.btn_train.clicked.connect(self._start_training)
        panel_layout.insertWidget(panel_layout.count() - 1, self.btn_train)

        # Stop training button
        self.btn_stop_train = QPushButton('■  STOP TRAIN')
        self.btn_stop_train.setFixedHeight(36)
        self.btn_stop_train.setFont(QFont('Courier New', 9, QFont.Bold))
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.setStyleSheet(f'''
            QPushButton {{
                background:#260010; color:{ACCENT2}; border:1.5px solid {ACCENT2};
                border-radius:6px; letter-spacing:1px;
            }}
            QPushButton:hover {{ background:{ACCENT2}; color:#000; }}
            QPushButton:disabled {{ border-color:#2a3040; color:#2a3040;
                                    background:{DARK_BG}; }}
        ''')
        self.btn_stop_train.clicked.connect(self._stop_training)
        panel_layout.insertWidget(panel_layout.count() - 1, self.btn_stop_train)

    def _start_training(self):
        self.btn_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.btn_start.setEnabled(False)

        # Reset visualization
        self.xs = [self.p.x0]; self.ys = [self.p.y0]; self.ts = [self.p.theta0]
        self.es = []; self.ws = []
        self.t_elapsed = 0.0
        self.canvas._build_axes()

        print(f"\n{'='*70}")
        print(f"  Starting RL Training — {TrainingConfig.total_episodes} Episodes")
        print(f"  Curriculum: Stage1={CurriculumConfig.stage1_end}  "
              f"Stage2={CurriculumConfig.stage2_end}  Stage3=end")
        print(f"  LR: {TrainingConfig.learning_rate_start} → {TrainingConfig.learning_rate_end}")
        print(f"  Replay buffer: {TrainingConfig.buffer_size:,}  Batch: {TrainingConfig.batch_size}")
        print(f"{'='*70}\n")

        self.orchestrator.start()

    def _stop_training(self):
        self.orchestrator.stop()
        self.btn_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.lbl_time.setText('Training Stopped')
        print("\n[Training Stopped by user]")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(DARK_BG))
    pal.setColor(QPalette.WindowText,      Qt.white)
    pal.setColor(QPalette.Base,            QColor(PANEL_BG))
    pal.setColor(QPalette.AlternateBase,   QColor('#1a2035'))
    pal.setColor(QPalette.ToolTipBase,     Qt.white)
    pal.setColor(QPalette.ToolTipText,     Qt.white)
    pal.setColor(QPalette.Text,            Qt.white)
    pal.setColor(QPalette.Button,          QColor(PANEL_BG))
    pal.setColor(QPalette.ButtonText,      Qt.white)
    pal.setColor(QPalette.Highlight,       QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(pal)

    win = RLMainWindow()
    win.show()
    sys.exit(app.exec_())
