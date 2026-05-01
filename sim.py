"""
Unicycle Curve-Following Simulation
BLF (Barrier Lyapunov Function) Control Law
Real-time PyQt5 + Matplotlib UI
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QGroupBox, QGridLayout, QSplitter,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.patheffects as pe


# ─── Simulation Parameters (from MATLAB code) ───────────────────────────────
class SimParams:
    x0     = -0.5
    y0     = -1.0
    theta0 = -135 * np.pi / 180
    v      = 0.5        # constant speed
    r_d    = 0.7        # desired radius (circle)
    a      = 1.5        # ellipse semi-major
    b      = 1.25       # ellipse semi-minor
    kn     = 1.0        # normal gain
    kd     = 1.0        # heading gain
    dt     = 0.01       # time step
    w_max  = 2.0        # angular velocity saturation


# ─── Control Law (translated from MATLAB) ────────────────────────────────────
def step(state, p: SimParams):
    x, y, theta = state
    v = p.v
    r   = np.array([x, y])
    m   = np.array([np.cos(theta), np.sin(theta)])
    E   = np.array([[0, 1], [-1, 0]])  # 90° rotation

    xdot = v * np.cos(theta)
    ydot = v * np.sin(theta)

    psi    = np.arctan2(y, x)
    norm_r = np.linalg.norm(r)
    psidot = (x * ydot - y * xdot) / (norm_r ** 2) if norm_r > 1e-9 else 0.0

    # Gradient / Hessian for circle: f = x^2 + y^2
    dx, dy = 2 * x, 2 * y
    dxx, dyy, dxy = 2.0, 2.0, 0.0
    H  = np.array([[dxx, dxy], [dxy, dyy]])
    n  = np.array([dx, dy])       # normal vector (unnormalized)
    tau = E @ n                    # tangent vector

    # Elliptic radius at angle psi
    M   = (p.b * np.cos(psi)) ** 2 + (p.a * np.sin(psi)) ** 2
    R   = p.a * p.b / np.sqrt(M)
    e   = norm_r - p.r_d
    del_  = R - p.r_d             # "del" (boundary slack)

    Rdot  = (-0.5 * p.a * p.b * np.sin(2 * psi) * (p.a ** 2 - p.b ** 2)
             / (M ** 1.5))
    Rddot = ((-p.a * p.b * (p.a ** 2 - p.b ** 2) * np.cos(2 * psi) / (M ** 1.5))
             + (3 * p.a * p.b * (p.a ** 2 - p.b ** 2) ** 2
                * np.sin(2 * psi) ** 2 / (4 * M ** 2.5)))

    edot  = v * (r @ m) / norm_r if norm_r > 1e-9 else 0.0

    # BLF control terms
    if abs(del_) < 1e-9:
        del_ = 1e-9

    a_blf     = -p.kn * e / del_
    a_blfdot  = -p.kn * (edot * del_ - e * Rdot * psidot) / (del_ ** 2)

    a_comp = Rdot * e / (norm_r * del_) if norm_r > 1e-9 else 0.0
    if norm_r > 1e-9:
        num   = norm_r * del_ * (Rddot * psidot * e + Rdot * edot)
        denom_part = Rdot * e * (Rdot * psidot * norm_r
                                  + del_ * (r @ (v * m)) / norm_r)
        a_compdot = (num - denom_part) / (norm_r ** 2 * del_ ** 2)
    else:
        a_compdot = 0.0

    A    = a_blf + a_comp
    Adot = a_blfdot + a_compdot

    # Desired heading
    I       = np.eye(2)
    eta     = tau + A * n
    etadot  = E @ H @ (v * m) + Adot * n + v * A * H @ m
    norm_eta = np.linalg.norm(eta)
    if norm_eta < 1e-9:
        norm_eta = 1e-9
    md     = eta / norm_eta
    mddot  = (I / norm_eta - np.outer(eta, eta) / norm_eta ** 3) @ etadot

    w_d = -mddot @ (E @ md)

    # Gamma (heading error)
    m3  = np.array([m[0], m[1], 0.0])
    md3 = np.array([md[0], md[1], 0.0])
    Z   = np.cross(m3, md3)
    cross_mag = np.linalg.norm(np.cross(m3, md3))
    dot_val   = np.dot(m3, md3)
    gamma     = np.arctan2((Z[2] / (np.linalg.norm(Z) + 1e-12)) * cross_mag,
                            dot_val)

    w = w_d + p.kd * gamma
    if abs(w) >= p.w_max:
        w = np.sign(w) * p.w_max

    # Integrate
    xn     = x + xdot * p.dt
    yn     = y + ydot * p.dt
    thetan = theta + w * p.dt

    extras = dict(
        e=e, gamma=gamma, w=w,
        tau=tau / (np.linalg.norm(tau) + 1e-12),
        n_unit=n / (np.linalg.norm(n) + 1e-12),
        m=m, del_=del_, R=R
    )
    return (xn, yn, thetan), extras


# ─── Matplotlib Canvas ────────────────────────────────────────────────────────
DARK_BG  = '#0f1117'
PANEL_BG = '#161b27'
ACCENT   = '#00e5ff'
ACCENT2  = '#ff4d6d'
GREEN    = '#39ff14'
YELLOW   = '#ffd60a'
GREY     = '#8892a4'

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=GREY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3040')
    ax.title.set_color(ACCENT)
    ax.title.set_fontsize(9)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    if title:   ax.set_title(title)
    if xlabel:  ax.set_xlabel(xlabel)
    if ylabel:  ax.set_ylabel(ylabel)
    ax.grid(True, color='#1e2535', linewidth=0.5, linestyle='--')


class SimCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor=DARK_BG, tight_layout=True)
        super().__init__(self.fig)
        self.setMinimumSize(400, 300)
        self._build_axes()

    def _build_axes(self):
        gs = self.fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                                    left=0.08, right=0.97, top=0.95, bottom=0.08)
        self.ax_traj = self.fig.add_subplot(gs[:, 0])   # left: trajectory
        self.ax_e    = self.fig.add_subplot(gs[0, 1])   # top-right: error
        self.ax_w    = self.fig.add_subplot(gs[1, 1])   # bot-right: omega

        style_ax(self.ax_traj, 'TRAJECTORY', 'x (m)', 'y (m)')
        style_ax(self.ax_e,    'TRACKING ERROR  e(t)', 'Time (s)', 'e (m)')
        style_ax(self.ax_w,    'ANGULAR VELOCITY  ω(t)', 'Time (s)', 'ω (rad/s)')

        p = SimParams()
        rho = np.linspace(0, 2 * np.pi, 500)
        # Desired circle
        self.ax_traj.plot(p.r_d * np.cos(rho), p.r_d * np.sin(rho),
                          color=ACCENT, lw=1.5, linestyle='--', label='Desired')
        # Ellipse boundary
        self.ax_traj.plot(p.a * np.cos(rho), p.b * np.sin(rho),
                          color=ACCENT2, lw=1, linestyle=':', alpha=0.5, label='Ellipse bound')
        self.ax_traj.legend(fontsize=7, facecolor=DARK_BG, edgecolor=GREY,
                             labelcolor='white', loc='upper right')
        self.ax_traj.set_aspect('equal')
        self.ax_traj.set_xlim(-5.0, 5.0)
        self.ax_traj.set_ylim(-4.5, 4.5)

        # Trajectory line
        self.traj_line, = self.ax_traj.plot([], [], color='#4fc3f7', lw=1.2, alpha=0.8)

        # Agent dot
        self.agent_dot, = self.ax_traj.plot([], [], 'o', color=GREEN,
                                              markersize=9, zorder=10)
        # Action text display
        self.action_text = self.ax_traj.text(0.02, 0.98, '', transform=self.ax_traj.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                            color='black', weight='bold')
        # Obstacles (red dots - small point-sized)
        self.obstacles_plot, = self.ax_traj.plot([], [], 'o', color=ACCENT2,
                                                  markersize=3, zorder=5, alpha=0.9)
        # Store fixed obstacle positions
        self.fixed_obstacles = None
        # Vectors (quiver)
        self.q_heading = self.ax_traj.quiver([], [], [], [],
            color=GREEN, scale=5, width=0.006, headwidth=4, label='Heading m')
        self.q_normal  = self.ax_traj.quiver([], [], [], [],
            color=YELLOW, scale=5, width=0.005, headwidth=4, label='Normal n')
        self.q_tangent = self.ax_traj.quiver([], [], [], [],
            color=ACCENT2, scale=5, width=0.005, headwidth=4, label='Tangent τ')

        vec_legend = self.ax_traj.legend(
            handles=[self.q_heading, self.q_normal, self.q_tangent],
            fontsize=7, facecolor=DARK_BG, edgecolor=GREY, labelcolor='white',
            loc='lower right'
        )
        self.ax_traj.add_artist(vec_legend)

        # Time series
        self.e_line, = self.ax_e.plot([], [], color=ACCENT, lw=1.2)
        self.ax_e.axhline(0, color=GREY, lw=0.5, linestyle='--')
        self.w_line, = self.ax_w.plot([], [], color=ACCENT2, lw=1.2)
        self.ax_w.axhline(SimParams.w_max, color=GREY, lw=0.5, linestyle=':', alpha=0.5)
        self.ax_w.axhline(-SimParams.w_max, color=GREY, lw=0.5, linestyle=':', alpha=0.5)

        self.draw()

    def update_plots(self, xs, ys, ts, es, ws, extras, obstacles=None):
        # Trajectory
        self.traj_line.set_data(xs, ys)
        x, y = xs[-1], ys[-1]
        self.agent_dot.set_data([x], [y])

        # Update obstacle markers at start of line
        if obstacles is not None and len(obstacles) > 0:
            # Use provided obstacles from RL environment
            self.obstacles_plot.set_data(obstacles[:, 0], obstacles[:, 1])
        else:
            # Generate FIXED sample obstacles at start of line
            sample_angles = np.linspace(0, 2*np.pi, 2, endpoint=False)
            sample_radius = 0.7  # same as r_d
            sample_offsets = np.array([-0.1, 0.1])
            obstacles = np.column_stack((
                (sample_radius + sample_offsets) * np.cos(sample_angles),
                (sample_radius + sample_offsets) * np.sin(sample_angles)
            ))
            self.obstacles_plot.set_data(obstacles[:, 0], obstacles[:, 1])

        # Vectors
        sc = 0.35
        m   = extras['m']
        n_u = extras['n_unit']
        tau = extras['tau']
        self.q_heading.set_offsets([[x, y]])
        self.q_heading.set_UVC([m[0] * sc], [m[1] * sc])
        self.q_normal.set_offsets([[x, y]])
        self.q_normal.set_UVC([n_u[0] * sc], [n_u[1] * sc])
        self.q_tangent.set_offsets([[x, y]])
        self.q_tangent.set_UVC([tau[0] * sc], [tau[1] * sc])

        # Auto-scroll time series (show last 30 s)
        win = 30.0
        dt  = SimParams.dt
        t_arr = np.arange(len(es)) * dt
        mask  = t_arr >= (t_arr[-1] - win) if len(t_arr) > 1 else slice(None)

        self.e_line.set_data(t_arr[mask], np.array(es)[mask])
        self.w_line.set_data(t_arr[mask], np.array(ws)[mask])

        for ax in (self.ax_e, self.ax_w):
            ax.relim()
            ax.autoscale_view()

        # Show the action that was taken in the simulation
        action = extras.get('w', 0)
        self.action_text.set_text(f'Action (ω): {action:.3f} rad/s')

        self.draw_idle()


# ─── Main Window ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Unicycle BLF Simulation')
        self.setMinimumSize(1100, 700)
        self._apply_dark_theme()

        self.p = SimParams()
        self._reset_state()

        self._build_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # ── State ──────────────────────────────────────────────────────────────
    def _reset_state(self):
        p = self.p
        self.state = (p.x0, p.y0, p.theta0)
        self.xs, self.ys, self.ts = [p.x0], [p.y0], [p.theta0]
        self.es, self.ws = [], []
        self.extras_last = dict(
            m=np.array([np.cos(p.theta0), np.sin(p.theta0)]),
            n_unit=np.array([1.0, 0.0]),
            tau=np.array([0.0, 1.0]),
        )
        self.t_elapsed = 0.0
        self.running = False
        self.episode_count = 0  # Track episodes for obstacle regeneration

    # ── UI Build ────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left: canvas
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self.canvas = SimCanvas()
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setStyleSheet(f'background:{DARK_BG}; color:{GREY};')
        lv.addWidget(toolbar)
        lv.addWidget(self.canvas)
        root.addWidget(left, stretch=5)

        # Right: control panel
        panel = self._build_panel()
        root.addWidget(panel, stretch=1)

    def _build_panel(self):
        panel = QFrame()
        panel.setFixedWidth(230)
        panel.setStyleSheet(f'background:{PANEL_BG}; border-radius:8px;')
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(12, 14, 12, 14)
        vl.setSpacing(10)

        # Title
        title = QLabel('⬡ UNICYCLE\nCONTROL SIM')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Courier New', 12, QFont.Bold))
        title.setStyleSheet(f'color:{ACCENT}; letter-spacing:2px;')
        vl.addWidget(title)
        vl.addWidget(self._hline())

        # Status
        self.lbl_time  = self._stat_label('Time', '0.00 s')
        self.lbl_e     = self._stat_label('Error e', '—')
        self.lbl_w     = self._stat_label('ω', '—')
        self.lbl_r     = self._stat_label('R(ψ)', '—')
        for w in (self.lbl_time, self.lbl_e, self.lbl_w, self.lbl_r):
            vl.addWidget(w)

        vl.addWidget(self._hline())

        # Sliders
        vl.addWidget(self._section('PARAMETERS'))
        self.sl_kn = self._slider('kn', 0.1, 5.0, SimParams.kn)
        self.sl_kd = self._slider('kd', 0.1, 5.0, SimParams.kd)
        self.sl_v  = self._slider('v  (m/s)', 0.1, 2.0, SimParams.v)
        for grp in (self.sl_kn, self.sl_kd, self.sl_v):
            vl.addWidget(grp)

        vl.addWidget(self._hline())

        # Buttons
        self.btn_start = self._button('▶  START', ACCENT, '#001f26')
        self.btn_stop  = self._button('■  STOP', ACCENT2, '#26001a')
        self.btn_reset = self._button('↺  RESET', YELLOW, '#1a1500')
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset.clicked.connect(self._reset)

        for b in (self.btn_start, self.btn_stop, self.btn_reset):
            vl.addWidget(b)

        vl.addStretch()

        # Legend
        vl.addWidget(self._section('VECTOR LEGEND'))
        for color, label in [(GREEN, 'Heading  m'),
                              (YELLOW, 'Normal   n'),
                              (ACCENT2, 'Tangent  τ'),
                              (ACCENT, '— Desired circle'),
                              ('#4fc3f7', '— Trajectory')]:
            vl.addWidget(self._legend_item(color, label))

        return panel

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f'color: #2a3040;')
        return line

    def _section(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont('Courier New', 7, QFont.Bold))
        lbl.setStyleSheet(f'color:{GREY}; letter-spacing:2px;')
        return lbl

    def _stat_label(self, name, val):
        w = QLabel(f'{name}:  {val}')
        w.setFont(QFont('Courier New', 9))
        w.setStyleSheet(f'color:{ACCENT};')
        return w

    def _slider(self, name, lo, hi, init):
        grp = QGroupBox(name)
        grp.setStyleSheet(f'''
            QGroupBox {{ color:{GREY}; font-size:8px; font-family:Courier New;
                         border:1px solid #2a3040; border-radius:4px; margin-top:6px;}}
            QGroupBox::title {{ subcontrol-origin:margin; left:6px;}}
        ''')
        vl = QVBoxLayout(grp)
        vl.setContentsMargins(4, 8, 4, 4)
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(0)
        sl.setMaximum(100)
        sl.setValue(int((init - lo) / (hi - lo) * 100))
        sl.setStyleSheet(f'''
            QSlider::groove:horizontal {{height:4px; background:#2a3040; border-radius:2px;}}
            QSlider::handle:horizontal {{width:12px; height:12px; margin:-4px 0;
                background:{ACCENT}; border-radius:6px;}}
            QSlider::sub-page:horizontal {{background:{ACCENT}; border-radius:2px;}}
        ''')
        lbl = QLabel(f'{init:.2f}')
        lbl.setFont(QFont('Courier New', 8))
        lbl.setStyleSheet(f'color:{ACCENT};')
        lbl.setAlignment(Qt.AlignRight)

        def _update(v):
            val = lo + (v / 100) * (hi - lo)
            lbl.setText(f'{val:.2f}')

        sl.valueChanged.connect(_update)
        vl.addWidget(sl)
        vl.addWidget(lbl)
        grp._slider = sl
        grp._lo, grp._hi = lo, hi
        grp._lbl = lbl
        return grp

    def _get_slider_val(self, grp):
        v = grp._slider.value()
        return grp._lo + (v / 100) * (grp._hi - grp._lo)

    def _button(self, text, fg, bg):
        b = QPushButton(text)
        b.setFixedHeight(36)
        b.setFont(QFont('Courier New', 9, QFont.Bold))
        b.setStyleSheet(f'''
            QPushButton {{
                background:{bg}; color:{fg}; border:1.5px solid {fg};
                border-radius:6px; letter-spacing:1px;
            }}
            QPushButton:hover {{ background:{fg}; color:#000; }}
            QPushButton:disabled {{ border-color:#2a3040; color:#2a3040; background:{DARK_BG}; }}
        ''')
        return b

    def _legend_item(self, color, label):
        w = QLabel(f'<span style="color:{color}; font-size:14px;">■</span>'
                   f' <span style="color:{GREY}; font-size:8px;">{label}</span>')
        w.setTextFormat(Qt.RichText)
        return w

    # ── Control ──────────────────────────────────────────────────────────────
    def _start(self):
        # Sync params from sliders
        self.p.kn = self._get_slider_val(self.sl_kn)
        self.p.kd = self._get_slider_val(self.sl_kd)
        self.p.v  = self._get_slider_val(self.sl_v)
        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start(int(self.p.dt * 1000))   # real-time

    def _stop(self):
        self.timer.stop()
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _reset(self):
        self._stop()
        self.episode_count += 1
        self._reset_state()
        self.canvas._build_axes()  # redraw fresh axes
        self.lbl_time.setText('Time:  0.00 s')
        self.lbl_e.setText('Error e:  —')
        self.lbl_w.setText('ω:  —')
        self.lbl_r.setText('R(ψ):  —')

    # ── Simulation tick ──────────────────────────────────────────────────────
    def _tick(self):
        # Run multiple sub-steps per timer fire for speed options (currently 1:1)
        state, extras = step(self.state, self.p)
        self.state = state
        x, y, theta = state

        self.xs.append(x)
        self.ys.append(y)
        self.ts.append(theta)
        self.es.append(extras['e'])
        self.ws.append(extras['w'])
        self.extras_last = extras
        self.t_elapsed += self.p.dt

        # Update stats every 10 ticks to reduce overhead
        if len(self.es) % 10 == 0:
            self.lbl_time.setText(f'Time:  {self.t_elapsed:.2f} s')
            self.lbl_e.setText(f'Error e:  {extras["e"]:.4f} m')
            self.lbl_w.setText(f'ω:  {extras["w"]:.4f} r/s')
            self.lbl_r.setText(f'R(ψ):  {extras["R"]:.4f} m')

        # Redraw every tick
        self.canvas.update_plots(self.xs, self.ys, self.ts,
                                  self.es, self.ws, extras)

    # ── Dark theme ───────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet(f'''
            QMainWindow, QWidget {{ background: {DARK_BG}; color: white; }}
            QScrollBar {{ background: {PANEL_BG}; }}
        ''')


# ─── Entry ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(DARK_BG))
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, QColor(PANEL_BG))
    pal.setColor(QPalette.AlternateBase, QColor('#1a2035'))
    pal.setColor(QPalette.ToolTipBase, Qt.white)
    pal.setColor(QPalette.ToolTipText, Qt.white)
    pal.setColor(QPalette.Text, Qt.white)
    pal.setColor(QPalette.Button, QColor(PANEL_BG))
    pal.setColor(QPalette.ButtonText, Qt.white)
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())