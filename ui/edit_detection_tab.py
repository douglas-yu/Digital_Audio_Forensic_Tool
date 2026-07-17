"""Edit/splice detection tab."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QSplitter, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from utils.constants import EDIT_DETECTION_SENSITIVITY, COLORS
from core.edit_detector import EditDetector
from core.conclusion_engine import ConclusionEngine


class EditWorker(QThread):
    """Worker thread for edit detection."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr, sensitivity):
        super().__init__()
        self.y = y
        self.sr = sr
        self.sensitivity = sensitivity

    def run(self):
        try:
            detector = EditDetector(self.y, self.sr)
            results = detector.detect_all(sensitivity=self.sensitivity)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class EditDetectionTab(QWidget):
    """Tab for audio edit/splice detection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio = None
        self._results = None
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Controls ───────────────────────────────────────────────
        ctrl_group = QGroupBox("编辑检测设置")
        ctrl_layout = QHBoxLayout(ctrl_group)

        ctrl_layout.addWidget(QLabel("灵敏度:"))
        self.cmb_sensitivity = QComboBox()
        self.cmb_sensitivity.addItems(EDIT_DETECTION_SENSITIVITY.keys())
        self.cmb_sensitivity.setCurrentIndex(1)  # default to 中
        ctrl_layout.addWidget(self.cmb_sensitivity)

        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.clicked.connect(self._run_detection)
        ctrl_layout.addWidget(self.btn_detect)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl_layout.addWidget(self.progress)

        ctrl_layout.addStretch()

        self.lbl_summary = QLabel("")
        ctrl_layout.addWidget(self.lbl_summary)

        layout.addWidget(ctrl_group)

        # ── Three-way splitter for equal 1/3 heights ───────────────
        splitter = QSplitter(Qt.Vertical)

        # Panel 1: Waveform plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(12, 3), dpi=100)
        self.figure.set_facecolor("#FAFAFA")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)

        # Panel 2: Results table
        results_group = QGroupBox("检测结果")
        results_layout = QVBoxLayout(results_group)
        self.tbl_results = QTableWidget()
        self.tbl_results.setColumnCount(4)
        self.tbl_results.setHorizontalHeaderLabels(["时间 (秒)", "置信度", "类型", "检测方法"])
        self.tbl_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_results.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_results.setAlternatingRowColors(True)
        self.tbl_results.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_results.cellClicked.connect(self._on_row_clicked)
        results_layout.addWidget(self.tbl_results)
        splitter.addWidget(results_group)

        # Panel 3: Analysis conclusion
        conclusion_group = QGroupBox("检测分析结论")
        conclusion_layout = QVBoxLayout(conclusion_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.txt_conclusion = QLabel("请先运行编辑检测以生成分析结论")
        self.txt_conclusion.setWordWrap(True)
        self.txt_conclusion.setAlignment(Qt.AlignTop)
        self.txt_conclusion.setStyleSheet(
            "QLabel { background: #FFF3E0; padding: 12px; border-radius: 5px; "
            "font-size: 12px; line-height: 1.6; }"
        )
        self.txt_conclusion.setTextFormat(Qt.PlainText)
        scroll.setWidget(self.txt_conclusion)
        conclusion_layout.addWidget(scroll)
        splitter.addWidget(conclusion_group)

        # Set equal sizes for all three panels
        splitter.setSizes([1000, 1000, 1000])
        layout.addWidget(splitter, stretch=1)

    def set_audio(self, audio_data):
        self._audio = audio_data

    def _run_detection(self):
        if self._audio is None or not self._audio.is_loaded:
            return

        self.btn_detect.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        sens_label = self.cmb_sensitivity.currentText()
        sensitivity = EDIT_DETECTION_SENSITIVITY[sens_label]

        self._worker = EditWorker(
            self._audio.y, self._audio.sr, sensitivity
        )
        self._worker.finished.connect(self._on_detection_done)
        self._worker.error.connect(self._on_detection_error)
        self._worker.start()

    def _on_detection_done(self, results):
        self._results = results
        self.progress.setVisible(False)
        self.btn_detect.setEnabled(True)

        n = results["total_detections"]
        by_method = results["by_method"]
        self.lbl_summary.setText(
            f"检测到 {n} 个可疑编辑点 | "
            f"频谱: {by_method.get('spectral', 0)} | "
            f"相位: {by_method.get('phase', 0)} | "
            f"能量: {by_method.get('energy', 0)} | "
            f"统计: {by_method.get('statistical', 0)}"
        )

        # Fill table
        edit_points = results["edit_points"]
        self.tbl_results.setRowCount(len(edit_points))
        for row, pt in enumerate(edit_points):
            self.tbl_results.setItem(row, 0, QTableWidgetItem(f"{pt['time']:.3f}"))
            self.tbl_results.setItem(
                row, 1, QTableWidgetItem(f"{pt.get('confidence', 0) * 100:.1f}%")
            )
            self.tbl_results.setItem(
                row, 2, QTableWidgetItem(pt.get("type", "unknown"))
            )
            methods = pt.get("methods", [pt.get("method", "unknown")])
            self.tbl_results.setItem(row, 3, QTableWidgetItem(", ".join(methods)))

        # Plot waveform with edit markers
        self._plot_results(edit_points)

        # Generate analysis conclusion
        engine = ConclusionEngine()
        engine.set_edit_results(results)
        conclusion = engine.analyze_edit_detection()
        self.txt_conclusion.setText(conclusion)

    def _plot_results(self, edit_points):
        self.figure.clear()
        if self._audio is None:
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)
        times = np.arange(len(self._audio.y)) / self._audio.sr
        ax.plot(times, self._audio.y, color=COLORS["waveform"], linewidth=0.5, alpha=0.7)

        for pt in edit_points:
            t = pt["time"]
            conf = pt.get("confidence", 0)
            alpha = 0.3 + conf * 0.7
            ax.axvline(x=t, color=COLORS["edit_marker"], alpha=alpha,
                       linewidth=1.5, linestyle="--")

        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("振幅")
        ax.set_title(f"编辑检测结果 - 共 {len(edit_points)} 个可疑点")
        ax.set_xlim(0, self._audio.duration)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def _on_row_clicked(self, row, col):
        """Zoom to the selected edit point."""
        if self._results is None or self._audio is None:
            return
        points = self._results["edit_points"]
        if row < len(points):
            t = points[row]["time"]
            window = 2.0  # seconds around the edit point
            for ax in self.figure.axes:
                ax.set_xlim(max(0, t - window), min(self._audio.duration, t + window))
            self.canvas.draw()

    def _on_detection_error(self, error_msg):
        self.progress.setVisible(False)
        self.btn_detect.setEnabled(True)
        self.lbl_summary.setText(f"检测出错: {error_msg}")

    def get_results(self):
        return self._results

    def clear(self):
        self._audio = None
        self._results = None
        self.figure.clear()
        self.canvas.draw()
        self.tbl_results.setRowCount(0)
        self.lbl_summary.setText("")
        self.txt_conclusion.setText("请先运行编辑检测以生成分析结论")
