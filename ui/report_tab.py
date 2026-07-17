"""Report generation tab."""

import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QLineEdit, QTextEdit, QFileDialog,
    QMessageBox, QCheckBox,
)
from PyQt5.QtCore import Qt

from utils.constants import REPORT_FORMATS
from utils.helpers import format_duration, format_file_size, get_timestamp
from core.report_generator import ReportGenerator


class ReportTab(QWidget):
    """Tab for generating forensic analysis reports."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio = None
        self._statistics = None
        self._enf_results = None
        self._edit_results = None
        self._auto_conclusions = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Report settings ────────────────────────────────────────
        settings_group = QGroupBox("报告设置")
        settings_layout = QVBoxLayout(settings_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("报告标题:"))
        self.txt_title = QLineEdit("数字音频取证分析报告")
        row1.addWidget(self.txt_title)
        settings_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("输出格式:"))
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(REPORT_FORMATS)
        row2.addWidget(self.cmb_format)

        row2.addWidget(QLabel("输出路径:"))
        self.txt_output = QLineEdit()
        row2.addWidget(self.txt_output, stretch=1)
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._browse_output)
        row2.addWidget(self.btn_browse)
        settings_layout.addLayout(row2)

        layout.addWidget(settings_group)

        # ── Include sections ───────────────────────────────────────
        sections_group = QGroupBox("包含内容")
        sections_layout = QHBoxLayout(sections_group)

        self.cb_file_info = QCheckBox("文件信息")
        self.cb_file_info.setChecked(True)
        self.cb_statistics = QCheckBox("音频统计")
        self.cb_statistics.setChecked(True)
        self.cb_enf = QCheckBox("ENF 分析")
        self.cb_enf.setChecked(True)
        self.cb_edit = QCheckBox("编辑检测")
        self.cb_edit.setChecked(True)

        sections_layout.addWidget(self.cb_file_info)
        sections_layout.addWidget(self.cb_statistics)
        sections_layout.addWidget(self.cb_enf)
        sections_layout.addWidget(self.cb_edit)
        layout.addWidget(sections_group)

        # ── Conclusions ────────────────────────────────────────────
        conclusions_group = QGroupBox("分析结论 (可选，每行一条)")
        conclusions_layout = QVBoxLayout(conclusions_group)
        self.txt_conclusions = QTextEdit()
        self.txt_conclusions.setMaximumHeight(120)
        self.txt_conclusions.setPlaceholderText("在此输入分析结论，每行一条...")
        conclusions_layout.addWidget(self.txt_conclusions)
        layout.addWidget(conclusions_group)

        # ── Generate button ────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_generate = QPushButton("生成报告")
        self.btn_generate.setMinimumWidth(200)
        self.btn_generate.setMinimumHeight(40)
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #1976D2; color: white; "
            "font-size: 14px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1565C0; }"
        )
        self.btn_generate.clicked.connect(self._generate_report)
        btn_layout.addWidget(self.btn_generate)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # ── Preview ────────────────────────────────────────────────
        preview_group = QGroupBox("报告预览")
        preview_layout = QVBoxLayout(preview_group)
        self.txt_preview = QTextEdit()
        self.txt_preview.setReadOnly(True)
        preview_layout.addWidget(self.txt_preview)
        layout.addWidget(preview_group, stretch=1)

    def set_data(self, audio_data=None, statistics=None, enf_results=None,
                 edit_results=None, auto_conclusions=None):
        """Set analysis data for report generation."""
        if audio_data is not None:
            self._audio = audio_data
        if statistics is not None:
            self._statistics = statistics
        if enf_results is not None:
            self._enf_results = enf_results
        if edit_results is not None:
            self._edit_results = edit_results
        if auto_conclusions is not None:
            self._auto_conclusions = auto_conclusions

    def _browse_output(self):
        fmt = self.cmb_format.currentText().lower()
        filter_str = f"{fmt.upper()} Files (*.{fmt})"
        path, _ = QFileDialog.getSaveFileName(self, "选择保存位置", "", filter_str)
        if path:
            self.txt_output.setText(path)

    def _generate_report(self):
        output_path = self.txt_output.text().strip()
        if not output_path:
            QMessageBox.warning(self, "提示", "请指定输出文件路径")
            return

        fmt = self.cmb_format.currentText()
        title = self.txt_title.text().strip() or "数字音频取证分析报告"

        try:
            gen = ReportGenerator()

            # File info
            if self.cb_file_info.isChecked() and self._audio:
                gen.set_file_info({
                    "文件名": self._audio.file_name,
                    "格式": self._audio.file_format,
                    "文件大小": format_file_size(self._audio.file_size),
                    "采样率": f"{self._audio.sr} Hz",
                    "声道数": str(self._audio.n_channels),
                    "位深度": f"{self._audio.bit_depth} bit",
                    "时长": format_duration(self._audio.duration),
                })

            # Statistics
            if self.cb_statistics.isChecked() and self._statistics:
                display_stats = {}
                for k, v in self._statistics.items():
                    if isinstance(v, float):
                        display_stats[k] = f"{v:.6f}"
                    else:
                        display_stats[k] = str(v)
                gen.set_statistics(display_stats)

            # ENF
            if self.cb_enf.isChecked() and self._enf_results:
                gen.set_enf_results(self._enf_results)

            # Edit detection
            if self.cb_edit.isChecked() and self._edit_results:
                gen.set_edit_results(self._edit_results)

            # Conclusions - combine auto-generated and manual
            all_conclusions = []
            if self._auto_conclusions:
                all_conclusions.extend(self._auto_conclusions)
            conclusions_text = self.txt_conclusions.toPlainText().strip()
            if conclusions_text:
                manual = [l.strip() for l in conclusions_text.split("\n") if l.strip()]
                all_conclusions.extend(manual)
            if all_conclusions:
                gen.set_conclusions(all_conclusions)

            # Generate
            if fmt == "HTML":
                gen.generate_html(output_path, title)
            elif fmt == "PDF":
                gen.generate_pdf(output_path, title)

            self.txt_preview.setPlainText(
                f"报告已成功生成!\n\n"
                f"文件: {output_path}\n"
                f"格式: {fmt}\n"
                f"时间: {get_timestamp()}"
            )

            QMessageBox.information(self, "成功", f"报告已生成: {output_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成报告失败:\n{e}")

    def clear(self):
        self._audio = None
        self._statistics = None
        self._enf_results = None
        self._edit_results = None
        self._auto_conclusions = None
        self.txt_preview.clear()
