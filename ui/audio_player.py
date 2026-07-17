"""Audio playback widget using QMediaPlayer."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QStyle,
)
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import os


class AudioPlayerWidget(QWidget):
    """Compact audio playback control bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path = None
        self._setup_ui()
        self._setup_player()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        # Play/Pause button
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setFixedSize(32, 32)
        self.btn_play.setToolTip("播放/暂停")
        self.btn_play.clicked.connect(self._toggle_play)
        layout.addWidget(self.btn_play)

        # Stop button
        self.btn_stop = QPushButton()
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.btn_stop.setFixedSize(32, 32)
        self.btn_stop.setToolTip("停止")
        self.btn_stop.clicked.connect(self._stop)
        layout.addWidget(self.btn_stop)

        # Current time label
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setFixedWidth(45)
        layout.addWidget(self.lbl_time)

        # Position slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._seek)
        layout.addWidget(self.slider, stretch=1)

        # Duration label
        self.lbl_duration = QLabel("00:00")
        self.lbl_duration.setFixedWidth(45)
        layout.addWidget(self.lbl_duration)

        # Volume
        self.btn_vol = QPushButton()
        self.btn_vol.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.btn_vol.setFixedSize(32, 32)
        self.btn_vol.setToolTip("静音")
        self.btn_vol.clicked.connect(self._toggle_mute)
        layout.addWidget(self.btn_vol)

        self.slider_vol = QSlider(Qt.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(80)
        self.slider_vol.setFixedWidth(80)
        self.slider_vol.valueChanged.connect(self._set_volume)
        layout.addWidget(self.slider_vol)

    def _setup_player(self):
        self.player = QMediaPlayer()
        self.player.setVolume(80)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.stateChanged.connect(self._on_state_changed)

    def set_file(self, file_path: str):
        """Set audio file for playback."""
        self._file_path = file_path
        if file_path and os.path.exists(file_path):
            url = QUrl.fromLocalFile(os.path.abspath(file_path))
            self.player.setMedia(QMediaContent(url))
            self.btn_play.setEnabled(True)
        else:
            self.player.setMedia(QMediaContent())
            self.btn_play.setEnabled(False)

    def _toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _stop(self):
        self.player.stop()

    def _seek(self, position):
        self.player.setPosition(position)

    def _set_volume(self, value):
        self.player.setVolume(value)

    def _toggle_mute(self):
        self.player.setMuted(not self.player.isMuted())
        if self.player.isMuted():
            self.btn_vol.setIcon(self.style().standardIcon(QStyle.SP_MediaVolumeMuted))
        else:
            self.btn_vol.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))

    def _on_position_changed(self, position):
        self.slider.setValue(position)
        self.lbl_time.setText(self._format_time(position))

    def _on_duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.lbl_duration.setText(self._format_time(duration))

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def _format_time(self, ms):
        s = ms // 1000
        m = s // 60
        s = s % 60
        return f"{m:02d}:{s:02d}"

    def clear(self):
        """Stop and clear the player."""
        self.player.stop()
        self.player.setMedia(QMediaContent())
        self._file_path = None
        self.slider.setValue(0)
        self.lbl_time.setText("00:00")
        self.lbl_duration.setText("00:00")
