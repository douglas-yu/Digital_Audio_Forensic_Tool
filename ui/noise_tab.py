"""ENF (Electric Network Frequency) analysis tab."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from utils.constants import ENF_FREQUENCIES, COLORS
from core.enf_analysis import ENFAnalyzer
from core.conclusion_engine import ConclusionEngine


class ENFWorker(QThread):
    """Worker thread for ENF analysis."""
    finished = pyqtSignal(dict)
    harmonics_finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr, nominal_freq, bandwidth):
        super().__init__()
        self.y = y
        self.sr = sr
        self.nominal_freq = nominal_freq
        self.bandwidth = bandwidth

    def run(self):
        try:
            analyzer = ENFAnalyzer(self.y, self.sr, self.nominal_freq)
            results = analyzer.extract_enf(band_width=self.bandwidth)
            self.finished.emit(results)
            harmonics = analyzer.detect_enf_harmonics()
            self.harmonics_finished.emit(harmonics)
        except Exception as e:
            self.error.emit(str(e))


class NoiseTab(QWidget):
    """Tab for noise / ENF analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio = None
        self._enf_results = None
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Controls ───────────────────────────────────────────────
        ctrl_group = QGroupBox("ENF 分析设置")
        ctrl_layout = QHBoxLayout(ctrl_group)

        ctrl_layout.addWidget(QLabel("地区/频率:"))
        self.cmb_region = QComboBox()
        self.cmb_region.addItems(ENF_FREQUENCIES.keys())
        ctrl_layout.addWidget(self.cmb_region)

        ctrl_layout.addWidget(QLabel("带宽 (Hz):"))
        self.spn_bandwidth = QDoubleSpinBox()
        self.spn_bandwidth.setRange(0.1, 5.0)
        self.spn_bandwidth.setValue(0.5)
        self.spn_bandwidth.setSingleStep(0.1)
        ctrl_layout.addWidget(self.spn_bandwidth)

        self.btn_analyze = QPushButton("开始分析")
        self.btn_analyze.clicked.connect(self._run_analysis)
        ctrl_layout.addWidget(self.btn_analyze)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl_layout.addWidget(self.progress)

        layout.addWidget(ctrl_group)

        # ── ENF trace plot ─────────────────────────────────────────
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.figure.set_facecolor("#FAFAFA")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # ── Results ────────────────────────────────────────────────
        results_layout = QHBoxLayout()

        stats_group = QGroupBox("ENF 统计")
        stats_layout = QVBoxLayout(stats_group)
        self.lbl_stats = QLabel("请先运行 ENF 分析")
        stats_layout.addWidget(self.lbl_stats)
        results_layout.addWidget(stats_group)

        harmonics_group = QGroupBox("ENF 谐波")
        harmonics_layout = QVBoxLayout(harmonics_group)
        self.tbl_harmonics = QTableWidget()
        self.tbl_harmonics.setColumnCount(3)
        self.tbl_harmonics.setHorizontalHeaderLabels(["谐波", "检测频率 (Hz)", "幅度 (dB)"])
        self.tbl_harmonics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_harmonics.setEditTriggers(QTableWidget.NoEditTriggers)
        harmonics_layout.addWidget(self.tbl_harmonics)
        results_layout.addWidget(harmonics_group)

        layout.addLayout(results_layout)

        # ── Analysis conclusion ────────────────────────────────────
        conclusion_group = QGroupBox("分析结论")
        conclusion_layout = QVBoxLayout(conclusion_group)
        self.txt_conclusion = QLabel("请先运行 ENF 分析以生成分析结论")
        self.txt_conclusion.setWordWrap(True)
        self.txt_conclusion.setStyleSheet(
            "QLabel { background: #E8F5E9; padding: 12px; border-radius: 5px; "
            "font-size: 12px; line-height: 1.6; }"
        )
        self.txt_conclusion.setTextFormat(Qt.PlainText)
        conclusion_layout.addWidget(self.txt_conclusion)
        layout.addWidget(conclusion_group)

    def set_audio(self, audio_data):
        self._audio = audio_data

    def _run_analysis(self):
        if self._audio is None or not self._audio.is_loaded:
            return

        self.btn_analyze.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate

        region = self.cmb_region.currentText()
        nominal_freq = ENF_FREQUENCIES[region]
        bandwidth = self.spn_bandwidth.value()

        self._worker = ENFWorker(
            self._audio.y, self._audio.sr, nominal_freq, bandwidth
        )
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.harmonics_finished.connect(self._on_harmonics_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_done(self, results):
        self._enf_results = results
        self.progress.setVisible(False)
        self.btn_analyze.setEnabled(True)

        # Update stats
        self.lbl_stats.setText(
            f"标称频率: {results['nominal_freq']:.1f} Hz\n"
            f"平均频率: {results['mean_freq']:.4f} Hz\n"
            f"标准差: {results['std_freq']:.4f} Hz\n"
            f"最小值: {results['min_freq']:.4f} Hz\n"
            f"最大值: {results['max_freq']:.4f} Hz\n"
            f"信噪比: {results['snr_db']:.2f} dB"
        )

        # Plot ENF trace
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        times = results["times"]
        trace = results["enf_trace"]
        valid = results["valid_mask"]

        ax.plot(times[valid], trace[valid], color=COLORS["enf_line"],
                linewidth=1, label="ENF 轨迹")
        ax.axhline(y=results["nominal_freq"], color="gray",
                   linestyle="--", alpha=0.5, label=f"标称 {results['nominal_freq']} Hz")
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("频率 (Hz)")
        ax.set_title("ENF 频率轨迹")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

        # Generate analysis conclusion
        engine = ConclusionEngine()
        engine.set_enf_results(results)
        conclusion = engine.analyze_enf()
        self.txt_conclusion.setText(conclusion)

    def _on_harmonics_done(self, harmonics):
        self.tbl_harmonics.setRowCount(len(harmonics))
        for row, (name, data) in enumerate(harmonics.items()):
            self.tbl_harmonics.setItem(row, 0, QTableWidgetItem(name))
            self.tbl_harmonics.setItem(
                row, 1, QTableWidgetItem(f"{data['detected_freq']:.2f}")
            )
            self.tbl_harmonics.setItem(
                row, 2, QTableWidgetItem(f"{data['magnitude_db']:.1f}")
            )

    def _on_analysis_error(self, error_msg):
        self.progress.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.lbl_stats.setText(f"分析出错: {error_msg}")

    def get_results(self):
        return self._enf_results

    def clear(self):
        self._audio = None
        self._enf_results = None
        self.figure.clear()
        self.canvas.draw()
        self.lbl_stats.setText("请先运行 ENF 分析")
        self.tbl_harmonics.setRowCount(0)
        self.txt_conclusion.setText("请先运行 ENF 分析以生成分析结论")
