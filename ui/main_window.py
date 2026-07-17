"""Main application window for Audio Forensics Tool."""

import os
import sys

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QAction, QFileDialog, QStatusBar,
    QMessageBox, QApplication, QToolBar, QLabel, QProgressBar,
    QWidget, QVBoxLayout, QTextEdit, QGroupBox, QHBoxLayout, QPushButton,
    QSplitter, QScrollArea,
)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer

from utils.constants import APP_NAME, APP_VERSION, SUPPORTED_FORMATS, COLORS
from core.audio_loader import AudioLoader
from core.analysis import AudioAnalyzer
from core.conclusion_engine import ConclusionEngine
from ui.waveform_tab import WaveformTab
from ui.spectrogram_tab import SpectrogramTab
from ui.metadata_tab import MetadataTab
from ui.noise_tab import NoiseTab
from ui.edit_detection_tab import EditDetectionTab
from ui.report_tab import ReportTab
from ui.advanced_tab import AdvancedAnalysisTab


STYLESHEET = """
QMainWindow {
    background-color: #FAFAFA;
}
QTabWidget::pane {
    border: 1px solid #E0E0E0;
    background: white;
}
QTabBar::tab {
    background: #E0E0E0;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: white;
    border-bottom: 2px solid #1976D2;
    font-weight: bold;
}
QTabBar::tab:hover {
    background: #BBDEFB;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #E0E0E0;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #1976D2;
    color: white;
    border: none;
    padding: 6px 16px;
    border-radius: 4px;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #1565C0;
}
QPushButton:pressed {
    background-color: #0D47A1;
}
QPushButton:disabled {
    background-color: #BDBDBD;
}
QStatusBar {
    background: #F5F5F5;
    border-top: 1px solid #E0E0E0;
}
QTableWidget {
    gridline-color: #E0E0E0;
    selection-background-color: #BBDEFB;
}
QHeaderView::section {
    background-color: #1976D2;
    color: white;
    padding: 5px;
    border: none;
}
"""


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()
        self._loader = AudioLoader()
        self._analyzer = None
        self._conclusion_engine = ConclusionEngine()

        self._setup_window()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_tabs()
        self._setup_statusbar()

        self.setStyleSheet(STYLESHEET)

    def _setup_window(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Set window/taskbar icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "resources", "icons", "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("文件(&F)")

        self.act_open = QAction("打开音频文件(&O)", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.triggered.connect(self._open_file)
        file_menu.addAction(self.act_open)

        file_menu.addSeparator()

        self.act_close = QAction("关闭文件(&C)", self)
        self.act_close.setShortcut("Ctrl+W")
        self.act_close.triggered.connect(self._close_file)
        self.act_close.setEnabled(False)
        file_menu.addAction(self.act_close)

        file_menu.addSeparator()

        act_exit = QAction("退出(&X)", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Analysis menu
        analysis_menu = menubar.addMenu("分析(&A)")

        self.act_stats = QAction("计算统计信息", self)
        self.act_stats.triggered.connect(self._compute_statistics)
        self.act_stats.setEnabled(False)
        analysis_menu.addAction(self.act_stats)

        # Help menu
        help_menu = menubar.addMenu("帮助(&H)")

        act_about = QAction("关于(&A)", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _setup_toolbar(self):
        toolbar = QToolBar("工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        btn_open = QAction("📂 打开", self)
        btn_open.triggered.connect(self._open_file)
        toolbar.addAction(btn_open)

        btn_close = QAction("❌ 关闭", self)
        btn_close.triggered.connect(self._close_file)
        toolbar.addAction(btn_close)

        toolbar.addSeparator()

        btn_stats = QAction("📊 统计", self)
        btn_stats.triggered.connect(self._compute_statistics)
        toolbar.addAction(btn_stats)

        btn_summary = QAction("📋 综合结论", self)
        btn_summary.triggered.connect(self._generate_summary)
        toolbar.addAction(btn_summary)

    def _setup_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setDocumentMode(True)

        self.tab_waveform = WaveformTab()
        self.tab_spectrogram = SpectrogramTab()
        self.tab_metadata = MetadataTab()
        self.tab_noise = NoiseTab()
        self.tab_edit = EditDetectionTab()
        self.tab_advanced = AdvancedAnalysisTab()
        self.tab_summary = self._create_summary_tab()
        self.tab_report = ReportTab()

        self.tabs.addTab(self.tab_waveform, "📈 波形")
        self.tabs.addTab(self.tab_spectrogram, "🎵 频谱图")
        self.tabs.addTab(self.tab_metadata, "📋 元数据")
        self.tabs.addTab(self.tab_noise, "🔌 ENF分析")
        self.tabs.addTab(self.tab_edit, "✂️ 编辑检测")
        self.tabs.addTab(self.tab_advanced, "🧠 高级分析")
        self.tabs.addTab(self.tab_summary, "📊 综合结论")
        self.tabs.addTab(self.tab_report, "📄 报告生成")

        self.setCentralWidget(self.tabs)

    def _create_summary_tab(self) -> QWidget:
        """Create the comprehensive summary conclusion tab.

        Layout:
        ┌──────────────────────────────────────────────┐
        │  Header + Generate button                    │
        ├─────────────────────┬────────────────────────┤
        │  音频统计分析        │  编辑检测分析结论       │
        ├─────────────────────┤                        │
        │  ENF 分析结论        │                        │
        ├─────────────────────┴────────────────────────┤
        │  最终综合判定                                  │
        └──────────────────────────────────────────────┘
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Header
        header = QLabel("综合分析结论")
        header.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: bold; color: #1976D2; "
            "padding: 10px 0; }"
        )
        layout.addWidget(header)

        desc = QLabel(
            "点击「生成综合结论」汇总所有分析结果，生成各模块的检测分析描述和最终综合判定。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("QLabel { color: #666; margin-bottom: 10px; }")
        layout.addWidget(desc)

        # Generate button
        btn_layout = QHBoxLayout()
        self.btn_generate_summary = QPushButton("🔍 生成综合结论")
        self.btn_generate_summary.setMinimumHeight(40)
        self.btn_generate_summary.setMinimumWidth(200)
        self.btn_generate_summary.clicked.connect(self._generate_summary)
        btn_layout.addWidget(self.btn_generate_summary)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # ── Top section: left (stats + ENF stacked) | right (edit detection) ──
        top_splitter = QSplitter(Qt.Horizontal)

        # Left column: stats on top, ENF below
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        stats_group = QGroupBox("音频统计分析")
        stats_layout = QVBoxLayout(stats_group)
        self.txt_stats_conclusion = QTextEdit()
        self.txt_stats_conclusion.setReadOnly(True)
        self.txt_stats_conclusion.setPlaceholderText("请先计算统计信息...")
        self.txt_stats_conclusion.setStyleSheet(
            "QTextEdit { background: #E3F2FD; border: 1px solid #90CAF9; border-radius: 5px; }"
        )
        stats_layout.addWidget(self.txt_stats_conclusion)
        left_layout.addWidget(stats_group, stretch=1)

        enf_group = QGroupBox("ENF 分析结论")
        enf_layout = QVBoxLayout(enf_group)
        self.txt_enf_conclusion = QTextEdit()
        self.txt_enf_conclusion.setReadOnly(True)
        self.txt_enf_conclusion.setPlaceholderText("请先运行 ENF 分析...")
        self.txt_enf_conclusion.setStyleSheet(
            "QTextEdit { background: #E8F5E9; border: 1px solid #A5D6A7; border-radius: 5px; }"
        )
        enf_layout.addWidget(self.txt_enf_conclusion)
        left_layout.addWidget(enf_group, stretch=1)

        top_splitter.addWidget(left_widget)

        # Right column: edit detection (same height as left two combined)
        edit_group = QGroupBox("编辑检测分析结论")
        edit_layout = QVBoxLayout(edit_group)
        self.txt_edit_conclusion = QTextEdit()
        self.txt_edit_conclusion.setReadOnly(True)
        self.txt_edit_conclusion.setPlaceholderText("请先运行编辑检测...")
        self.txt_edit_conclusion.setStyleSheet(
            "QTextEdit { background: #FFF3E0; border: 1px solid #FFCC80; border-radius: 5px; }"
        )
        edit_layout.addWidget(self.txt_edit_conclusion)
        top_splitter.addWidget(edit_group)

        top_splitter.setSizes([500, 500])

        # ── Bottom section: final summary ──────────────────────────
        summary_group = QGroupBox("最终综合判定")
        summary_layout = QVBoxLayout(summary_group)
        self.txt_final_summary = QTextEdit()
        self.txt_final_summary.setReadOnly(True)
        self.txt_final_summary.setPlaceholderText("请先生成综合结论...")
        self.txt_final_summary.setStyleSheet(
            "QTextEdit { background: #FCE4EC; border: 2px solid #E91E63; "
            "border-radius: 5px; font-size: 13px; }"
        )
        summary_layout.addWidget(self.txt_final_summary)

        # Vertical splitter: top (stats+ENF | edit) and bottom (summary)
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(summary_group)
        main_splitter.setSizes([600, 400])

        layout.addWidget(main_splitter, stretch=1)

        return widget

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.lbl_status = QLabel("就绪 - 请打开音频文件")
        self.statusbar.addWidget(self.lbl_status, stretch=1)

    # ── File Operations ────────────────────────────────────────────

    def _open_file(self):
        filter_str = ";;".join(SUPPORTED_FORMATS)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开音频文件", "", filter_str
        )
        if not file_path:
            return

        self.lbl_status.setText(f"正在加载: {file_path}...")
        QApplication.processEvents()

        try:
            audio = self._loader.load(file_path)
            self._analyzer = AudioAnalyzer(audio.y, audio.sr)

            # Update all tabs
            self.tab_waveform.set_audio(audio, self._analyzer)
            self.tab_spectrogram.set_audio(audio, self._analyzer)
            self.tab_metadata.set_audio(audio)
            self.tab_noise.set_audio(audio)
            self.tab_edit.set_audio(audio)
            self.tab_advanced.set_audio(audio)
            self.tab_report.set_data(audio_data=audio)

            self.act_close.setEnabled(True)
            self.act_stats.setEnabled(True)

            self.lbl_status.setText(
                f"已加载: {audio.file_name} | "
                f"{audio.sr}Hz | {audio.n_channels}ch | "
                f"{audio.duration:.2f}s"
            )

        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            self.lbl_status.setText("加载失败")

    def _close_file(self):
        self._loader.unload()
        self._analyzer = None
        self._conclusion_engine = ConclusionEngine()

        self.tab_waveform.clear()
        self.tab_spectrogram.clear()
        self.tab_metadata.clear()
        self.tab_noise.clear()
        self.tab_edit.clear()
        self.tab_advanced.clear()
        self.tab_report.clear()

        # Clear summary tab
        self.txt_stats_conclusion.clear()
        self.txt_enf_conclusion.clear()
        self.txt_edit_conclusion.clear()
        self.txt_final_summary.clear()

        self.act_close.setEnabled(False)
        self.act_stats.setEnabled(False)
        self.lbl_status.setText("就绪 - 请打开音频文件")

    def _compute_statistics(self):
        if self._analyzer is None:
            return
        stats = self._analyzer.compute_statistics()
        self._conclusion_engine.set_statistics(stats)
        self.tab_report.set_data(statistics=stats)

        # Also pass ENF and edit results if available
        enf = self.tab_noise.get_results()
        if enf:
            self._conclusion_engine.set_enf_results(enf)
            self.tab_report.set_data(enf_results=enf)

        edit = self.tab_edit.get_results()
        if edit:
            self._conclusion_engine.set_edit_results(edit)
            self.tab_report.set_data(edit_results=edit)

        # Show statistics conclusion
        stats_text = self._conclusion_engine.analyze_statistics()
        self.txt_stats_conclusion.setPlainText(stats_text)

        self.lbl_status.setText("统计计算完成 - 数据已同步到报告和综合结论标签页")
        QMessageBox.information(
            self, "完成",
            "统计信息已计算完成。\n"
            "数据已同步到报告标签页。\n"
            "请前往「综合结论」标签页查看分析描述。"
        )

    def _generate_summary(self):
        """Generate comprehensive analysis summary from all available results."""
        if self._analyzer is None:
            QMessageBox.warning(self, "提示", "请先加载音频文件")
            return

        # Compute statistics if not already done
        stats = self._analyzer.compute_statistics()
        self._conclusion_engine.set_statistics(stats)
        self.tab_report.set_data(statistics=stats)

        # Collect audio info
        audio = self._loader.current
        if audio:
            self._conclusion_engine.set_audio_info({
                "file_name": audio.file_name,
                "duration": audio.duration,
                "sr": audio.sr,
            })

        # Collect ENF results
        enf = self.tab_noise.get_results()
        if enf:
            self._conclusion_engine.set_enf_results(enf)
            self.tab_report.set_data(enf_results=enf)

        # Collect edit results
        edit = self.tab_edit.get_results()
        if edit:
            self._conclusion_engine.set_edit_results(edit)
            self.tab_report.set_data(edit_results=edit)

        # Generate all conclusions
        conclusions = self._conclusion_engine.generate_all_conclusions()

        self.txt_stats_conclusion.setPlainText(conclusions["statistics"])
        self.txt_enf_conclusion.setPlainText(conclusions["enf"])
        self.txt_edit_conclusion.setPlainText(conclusions["edit_detection"])
        self.txt_final_summary.setPlainText(conclusions["summary"])

        # Sync conclusions to report tab
        all_text = [
            conclusions["statistics"],
            conclusions["enf"],
            conclusions["edit_detection"],
            conclusions["summary"],
        ]
        self.tab_report.set_data(
            auto_conclusions=all_text,
            statistics=stats,
        )

        # Switch to summary tab
        for i in range(self.tabs.count()):
            if self.tabs.widget(i) == self.tab_summary:
                self.tabs.setCurrentIndex(i)
                break

        self.lbl_status.setText("综合分析结论已生成")

    def _show_about(self):
        QMessageBox.about(
            self,
            f"关于 {APP_NAME}",
            f"<h2>{APP_NAME}</h2>"
            f"<p>版本: {APP_VERSION}</p>"
            f"<p>数字音频取证分析工具</p>"
            f"<p>功能:</p>"
            f"<ul>"
            f"<li>波形可视化与分析</li>"
            f"<li>频谱图 (STFT, Mel, MFCC)</li>"
            f"<li>音频元数据查看</li>"
            f"<li>ENF (电网频率) 分析</li>"
            f"<li>编辑/拼接检测</li>"
            f"<li>取证报告生成 (PDF/HTML)</li>"
            f"</ul>"
        )
