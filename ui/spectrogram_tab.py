"""Spectrogram analysis tab."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import librosa.display

from utils.constants import SPECTROGRAM_CMAPS, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_N_MELS
from ui.audio_player import AudioPlayerWidget


class SpectrogramTab(QWidget):
    """Tab for spectrogram and frequency-domain visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio = None
        self._analyzer = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Controls ───────────────────────────────────────────────
        ctrl_group = QGroupBox("频谱图设置")
        ctrl_layout = QHBoxLayout(ctrl_group)

        ctrl_layout.addWidget(QLabel("类型:"))
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(["STFT 频谱图", "Mel 频谱图", "MFCC", "FFT 频谱", "功率谱密度"])
        ctrl_layout.addWidget(self.cmb_type)

        ctrl_layout.addWidget(QLabel("颜色:"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(SPECTROGRAM_CMAPS)
        ctrl_layout.addWidget(self.cmb_cmap)

        ctrl_layout.addWidget(QLabel("FFT:"))
        self.spn_nfft = QSpinBox()
        self.spn_nfft.setRange(256, 16384)
        self.spn_nfft.setSingleStep(256)
        self.spn_nfft.setValue(DEFAULT_N_FFT)
        ctrl_layout.addWidget(self.spn_nfft)

        ctrl_layout.addWidget(QLabel("Hop:"))
        self.spn_hop = QSpinBox()
        self.spn_hop.setRange(64, 4096)
        self.spn_hop.setSingleStep(64)
        self.spn_hop.setValue(DEFAULT_HOP_LENGTH)
        ctrl_layout.addWidget(self.spn_hop)

        self.btn_update = QPushButton("更新")
        self.btn_update.clicked.connect(self._update_plot)
        ctrl_layout.addWidget(self.btn_update)

        layout.addWidget(ctrl_group)

        # ── Matplotlib canvas ──────────────────────────────────────
        self.figure = Figure(figsize=(12, 5), dpi=100)
        self.figure.set_facecolor("#FAFAFA")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # ── Audio Player ──────────────────────────────────────────
        self.player = AudioPlayerWidget()
        layout.addWidget(self.player)

    def set_audio(self, audio_data, analyzer):
        self._audio = audio_data
        self._analyzer = analyzer
        if audio_data and audio_data.is_loaded:
            self.player.set_file(audio_data.file_path)
        self._update_plot()

    def _update_plot(self):
        self.figure.clear()
        if self._audio is None or not self._audio.is_loaded or self._analyzer is None:
            self.canvas.draw()
            return

        spec_type = self.cmb_type.currentText()
        cmap = self.cmb_cmap.currentText()
        n_fft = self.spn_nfft.value()
        hop_length = self.spn_hop.value()

        ax = self.figure.add_subplot(111)

        if spec_type == "STFT 频谱图":
            S_db, freqs, times = self._analyzer.compute_spectrogram(n_fft, hop_length)
            img = ax.pcolormesh(times, freqs, S_db, cmap=cmap, shading="gouraud")
            ax.set_ylabel("频率 (Hz)")
            ax.set_xlabel("时间 (秒)")
            ax.set_title("STFT 频谱图")
            self.figure.colorbar(img, ax=ax, format="%+2.0f dB")

        elif spec_type == "Mel 频谱图":
            mel_db, times = self._analyzer.compute_mel_spectrogram(n_fft, hop_length)
            img = librosa.display.specshow(
                mel_db, sr=self._audio.sr, hop_length=hop_length,
                x_axis="time", y_axis="mel", ax=ax, cmap=cmap,
            )
            ax.set_title("Mel 频谱图")
            self.figure.colorbar(img, ax=ax, format="%+2.0f dB")

        elif spec_type == "MFCC":
            mfccs, times = self._analyzer.compute_mfcc(n_fft=n_fft, hop_length=hop_length)
            img = librosa.display.specshow(
                mfccs, sr=self._audio.sr, hop_length=hop_length,
                x_axis="time", ax=ax, cmap=cmap,
            )
            ax.set_ylabel("MFCC 系数")
            ax.set_title("MFCC")
            self.figure.colorbar(img, ax=ax)

        elif spec_type == "FFT 频谱":
            freqs, magnitude_db = self._analyzer.compute_fft()
            ax.plot(freqs, magnitude_db, color="#1976D2", linewidth=0.5)
            ax.set_xlabel("频率 (Hz)")
            ax.set_ylabel("幅度 (dB)")
            ax.set_title("FFT 频谱")
            ax.grid(True, alpha=0.3)

        elif spec_type == "功率谱密度":
            freqs, psd = self._analyzer.compute_power_spectral_density(nperseg=n_fft)
            ax.semilogy(freqs, psd, color="#388E3C", linewidth=0.8)
            ax.set_xlabel("频率 (Hz)")
            ax.set_ylabel("PSD (V²/Hz)")
            ax.set_title("功率谱密度 (Welch)")
            ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        self._audio = None
        self._analyzer = None
        self.figure.clear()
        self.canvas.draw()
        self.player.clear()
