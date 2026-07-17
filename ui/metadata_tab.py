"""Audio metadata viewer tab."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QTextEdit,
)
from PyQt5.QtCore import Qt
import datetime

from utils.helpers import format_duration, format_file_size


class MetadataTab(QWidget):
    """Tab for viewing audio file metadata and properties."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Unified metadata table ─────────────────────────────────
        info_group = QGroupBox("元数据信息")
        info_layout = QVBoxLayout(info_group)

        self.tbl_metadata = QTableWidget()
        self.tbl_metadata.setColumnCount(2)
        self.tbl_metadata.setHorizontalHeaderLabels(["属性", "值"])
        self.tbl_metadata.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_metadata.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_metadata.setAlternatingRowColors(True)
        self.tbl_metadata.setEditTriggers(QTableWidget.NoEditTriggers)
        info_layout.addWidget(self.tbl_metadata)
        layout.addWidget(info_group, stretch=1)

        # ── Raw metadata text ──────────────────────────────────────
        raw_group = QGroupBox("原始信息")
        raw_layout = QVBoxLayout(raw_group)
        self.txt_raw = QTextEdit()
        self.txt_raw.setReadOnly(True)
        self.txt_raw.setMinimumHeight(300)
        #self.txt_raw.setMaximumHeight(300)
        raw_layout.addWidget(self.txt_raw)
        layout.addWidget(raw_group)

    def set_audio(self, audio_data):
        """Populate metadata table from AudioData."""
        self._audio = audio_data
        self._update_table()

    def _update_table(self):
        if self._audio is None or not self._audio.is_loaded:
            self.tbl_metadata.setRowCount(0)
            self.txt_raw.clear()
            return

        a = self._audio

        # Build unified list: basic info first, then extended metadata
        items = [
            ("文件名", a.file_name),
            ("文件路径", a.file_path),
            ("格式", a.file_format),
            ("文件大小", format_file_size(a.file_size)),
            ("采样率", f"{a.sr} Hz"),
            ("声道数", str(a.n_channels)),
            ("位深度", f"{a.bit_depth} bit"),
            ("时长", format_duration(a.duration)),
            ("采样点数", f"{a.n_samples:,}"),
            ("修改时间", datetime.datetime.fromtimestamp(
                a.metadata.get("修改时间", 0)
            ).strftime("%Y-%m-%d %H:%M:%S") if "修改时间" in a.metadata else "N/A"),
        ]

        # Append extended metadata (skip duplicates already shown above)
        skip_keys = {"修改时间", "格式", "采样率", "声道数", "文件大小", "时长(秒)", "帧数", "额外信息"}
        for k, v in a.metadata.items():
            if k not in skip_keys:
                items.append((str(k), str(v)))

        self.tbl_metadata.setRowCount(len(items))
        for row, (key, val) in enumerate(items):
            key_item = QTableWidgetItem(key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            val_item = QTableWidgetItem(str(val))
            val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
            self.tbl_metadata.setItem(row, 0, key_item)
            self.tbl_metadata.setItem(row, 1, val_item)

        # Raw text
        raw_text = "\n".join(f"{k}: {v}" for k, v in a.metadata.items())
        if "额外信息" in a.metadata:
            raw_text += f"\n\n=== 额外信息 ===\n{a.metadata['额外信息']}"
        self.txt_raw.setPlainText(raw_text)

    def clear(self):
        self._audio = None
        self.tbl_metadata.setRowCount(0)
        self.txt_raw.clear()
