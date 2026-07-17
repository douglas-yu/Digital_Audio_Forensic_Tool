"""Waveform visualization tab."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QGroupBox, QSlider, QSplitter,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from utils.constants import COLORS
from ui.audio_player import AudioPlayerWidget


class WaveformTab(QWidget):
    """Tab for waveform display and basic time-domain visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio = None
        self._analyzer = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Controls ───────────────────────────────────────────────
        ctrl_group = QGroupBox("波形控制")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.cb_show_rms = QCheckBox("显示 RMS 包络")
        self.cb_show_zcr = QCheckBox("显示过零率")
        self.cb_show_centroid = QCheckBox("显示频谱质心")
        self.btn_zoom_fit = QPushButton("适应窗口")
        self.btn_zoom_fit.clicked.connect(self._zoom_fit)

        ctrl_layout.addWidget(self.cb_show_rms)
        ctrl_layout.addWidget(self.cb_show_zcr)
        ctrl_layout.addWidget(self.cb_show_centroid)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_zoom_fit)

        self.cb_show_rms.stateChanged.connect(self._update_plot)
        self.cb_show_zcr.stateChanged.connect(self._update_plot)
        self.cb_show_centroid.stateChanged.connect(self._update_plot)

        layout.addWidget(ctrl_group)

        # ── Matplotlib canvas ──────────────────────────────────────
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.figure.set_facecolor("#FAFAFA")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # ── Info bar ───────────────────────────────────────────────
        info_layout = QHBoxLayout()
        self.lbl_info = QLabel("未加载音频文件")
        info_layout.addWidget(self.lbl_info)
        layout.addLayout(info_layout)

        # ── Audio Player ──────────────────────────────────────────
        self.player = AudioPlayerWidget()
        layout.addWidget(self.player)

    def set_audio(self, audio_data, analyzer):
        """Set audio data for display."""
        self._audio = audio_data
        self._analyzer = analyzer
        if audio_data and audio_data.is_loaded:
            self.player.set_file(audio_data.file_path)
        self._update_plot()

    def _update_plot(self):
        """Redraw the waveform plot."""
        self.figure.clear()
        if self._audio is None or not self._audio.is_loaded:
            self.canvas.draw()
            return

        n_plots = 1 + sum([
            self.cb_show_rms.isChecked(),
            self.cb_show_zcr.isChecked(),
            self.cb_show_centroid.isChecked(),
        ])

        plot_idx = 1

        # Main waveform
        ax_wave = self.figure.add_subplot(n_plots, 1, plot_idx)
        times = np.arange(len(self._audio.y)) / self._audio.sr
        ax_wave.plot(times, self._audio.y, color=COLORS["waveform"], linewidth=0.5)
        ax_wave.set_ylabel("振幅")
        ax_wave.set_title(f"波形 - {self._audio.file_name}", fontsize=10)
        ax_wave.grid(True, alpha=0.3)
        ax_wave.set_xlim(0, self._audio.duration)
        plot_idx += 1

        # RMS envelope
        if self.cb_show_rms.isChecked() and self._analyzer:
            ax_rms = self.figure.add_subplot(n_plots, 1, plot_idx, sharex=ax_wave)
            rms, rms_times = self._analyzer.compute_rms()
            ax_rms.plot(rms_times, rms, color=COLORS["accent"], linewidth=1)
            ax_rms.set_ylabel("RMS")
            ax_rms.set_title("RMS 能量包络", fontsize=10)
            ax_rms.grid(True, alpha=0.3)
            plot_idx += 1

        # Zero-crossing rate
        if self.cb_show_zcr.isChecked() and self._analyzer:
            ax_zcr = self.figure.add_subplot(n_plots, 1, plot_idx, sharex=ax_wave)
            zcr, zcr_times = self._analyzer.compute_zero_crossing_rate()
            ax_zcr.plot(zcr_times, zcr, color=COLORS["success"], linewidth=1)
            ax_zcr.set_ylabel("过零率")
            ax_zcr.set_title("过零率", fontsize=10)
            ax_zcr.grid(True, alpha=0.3)
            plot_idx += 1

        # Spectral centroid
        if self.cb_show_centroid.isChecked() and self._analyzer:
            ax_sc = self.figure.add_subplot(n_plots, 1, plot_idx, sharex=ax_wave)
            centroid, sc_times = self._analyzer.compute_spectral_centroid()
            ax_sc.plot(sc_times, centroid, color=COLORS["warning"], linewidth=1)
            ax_sc.set_ylabel("频率 (Hz)")
            ax_sc.set_title("频谱质心", fontsize=10)
            ax_sc.grid(True, alpha=0.3)
            plot_idx += 1

        self.figure.tight_layout()
        self.canvas.draw()

        self.lbl_info.setText(
            f"文件: {self._audio.file_name} | "
            f"采样率: {self._audio.sr} Hz | "
            f"时长: {self._audio.duration:.2f}s | "
            f"采样点: {self._audio.n_samples:,}"
        )

    def _zoom_fit(self):
        """Reset zoom to show full waveform."""
        if self._audio and self._audio.is_loaded:
            for ax in self.figure.axes:
                ax.set_xlim(0, self._audio.duration)
            self.canvas.draw()

    def clear(self):
        self._audio = None
        self._analyzer = None
        self.figure.clear()
        self.canvas.draw()
        self.lbl_info.setText("未加载音频文件")
        self.player.clear()
