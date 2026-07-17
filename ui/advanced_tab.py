"""Advanced analysis tab - Voiceprint, Content, Deepfake, AI Detection."""

from typing import Dict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QScrollArea,
    QSpinBox, QSplitter, QFileDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from utils.constants import COLORS


# ── Worker threads ──────────────────────────────────────────────────

class VoiceprintWorker(QThread):
    finished = pyqtSignal(dict)
    diarization_finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr):
        super().__init__()
        self.y = y
        self.sr = sr

    def run(self):
        try:
            # Try neural model first, fallback to heuristic
            try:
                from core.model_integration import SpeakerEmbedder
                embedder = SpeakerEmbedder()
                embedding = embedder.extract_embedding(self.y, self.sr)
                result = {
                    'embedding_dim': len(embedding),
                    'embedding_norm': float(np.linalg.norm(embedding)),
                    'model_used': 'Wespeaker ResNet34 (Neural)',
                    'embedding': embedding.tolist()[:10],  # first 10 dims for display
                }
                self.finished.emit(result)
                diar = embedder.diarize(self.y, self.sr)
                self.diarization_finished.emit(diar)
            except Exception:
                from core.voiceprint import VoiceprintAnalyzer
                analyzer = VoiceprintAnalyzer(self.y, self.sr)
                result = analyzer.extract_voiceprint()
                self.finished.emit(result)
                diar = analyzer.segment_speakers()
                self.diarization_finished.emit(diar)
        except Exception as e:
            self.error.emit(str(e))


class ContentWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr):
        super().__init__()
        self.y = y
        self.sr = sr

    def run(self):
        try:
            # Try Silero VAD model first, fallback to heuristic
            try:
                from core.model_integration import SileroVAD
                vad = SileroVAD()
                vad_result = vad.detect(self.y, self.sr)
                # Combine with content analyzer for full analysis
                from core.content_analysis import ContentAnalyzer
                analyzer = ContentAnalyzer(self.y, self.sr)
                result = analyzer.full_analysis()
                # Override VAD with neural result
                result['vad'] = vad_result
                result['vad']['model_used'] = 'Silero VAD v5.1 (Neural)'
                self.finished.emit(result)
            except Exception:
                from core.content_analysis import ContentAnalyzer
                analyzer = ContentAnalyzer(self.y, self.sr)
                result = analyzer.full_analysis()
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class DeepfakeWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr):
        super().__init__()
        self.y = y
        self.sr = sr

    def run(self):
        try:
            from core.deepfake_detector import DeepfakeDetector
            detector = DeepfakeDetector(self.y, self.sr)
            result = detector.detect()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class AIDetectionWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr):
        super().__init__()
        self.y = y
        self.sr = sr

    def run(self):
        try:
            from core.ai_detection import AIGenerationDetector
            detector = AIGenerationDetector(self.y, self.sr)
            result = detector.detect()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ── Main Tab ────────────────────────────────────────────────────────

class AdvancedAnalysisTab(QWidget):
    """Advanced analysis tab with sub-tabs for voiceprint, content, deepfake, AI detection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio = None
        self._results = {}
        self._workers = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Sub-tab widget
        self.sub_tabs = QTabWidget()

        self.sub_tabs.addTab(self._create_voiceprint_panel(), "🎤 声纹分析")
        self.sub_tabs.addTab(self._create_content_panel(), "📝 内容分析")
        self.sub_tabs.addTab(self._create_deepfake_panel(), "🎭 Deepfake检测")
        self.sub_tabs.addTab(self._create_ai_panel(), "🤖 AI生成检测")
        self.sub_tabs.addTab(self._create_model_panel(), "⚙️ 模型管理")

        layout.addWidget(self.sub_tabs)

    # ── Voiceprint Panel ──────────────────────────────────────────

    def _create_voiceprint_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        ctrl = QHBoxLayout()
        self.btn_voiceprint = QPushButton("提取声纹特征")
        self.btn_voiceprint.clicked.connect(self._run_voiceprint)
        ctrl.addWidget(self.btn_voiceprint)
        self.progress_vp = QProgressBar()
        self.progress_vp.setVisible(False)
        ctrl.addWidget(self.progress_vp)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Horizontal)

        # Features table
        feat_group = QGroupBox("声纹特征")
        feat_layout = QVBoxLayout(feat_group)
        self.tbl_voiceprint = QTableWidget()
        self.tbl_voiceprint.setColumnCount(2)
        self.tbl_voiceprint.setHorizontalHeaderLabels(["特征", "值"])
        self.tbl_voiceprint.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_voiceprint.setEditTriggers(QTableWidget.NoEditTriggers)
        feat_layout.addWidget(self.tbl_voiceprint)
        splitter.addWidget(feat_group)

        # Diarization results
        diar_group = QGroupBox("说话人分离")
        diar_layout = QVBoxLayout(diar_group)
        self.txt_diarization = QTextEdit()
        self.txt_diarization.setReadOnly(True)
        self.txt_diarization.setPlaceholderText("运行声纹分析后显示说话人分离结果...")
        diar_layout.addWidget(self.txt_diarization)
        splitter.addWidget(diar_group)

        layout.addWidget(splitter, stretch=1)
        return widget

    # ── Content Panel ─────────────────────────────────────────────

    def _create_content_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        ctrl = QHBoxLayout()
        self.btn_content = QPushButton("分析音频内容")
        self.btn_content.clicked.connect(self._run_content)
        ctrl.addWidget(self.btn_content)
        self.progress_ct = QProgressBar()
        self.progress_ct.setVisible(False)
        ctrl.addWidget(self.progress_ct)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Vertical)

        # VAD plot
        self.fig_content = Figure(figsize=(12, 3), dpi=100)
        self.fig_content.set_facecolor("#FAFAFA")
        self.canvas_content = FigureCanvas(self.fig_content)
        splitter.addWidget(self.canvas_content)

        # Results
        result_widget = QWidget()
        result_layout = QHBoxLayout(result_widget)

        vad_group = QGroupBox("语音活动检测")
        vad_layout = QVBoxLayout(vad_group)
        self.txt_vad = QTextEdit()
        self.txt_vad.setReadOnly(True)
        self.txt_vad.setPlaceholderText("等待分析...")
        vad_layout.addWidget(self.txt_vad)
        result_layout.addWidget(vad_group)

        noise_group = QGroupBox("噪声分析")
        noise_layout = QVBoxLayout(noise_group)
        self.txt_noise = QTextEdit()
        self.txt_noise.setReadOnly(True)
        self.txt_noise.setPlaceholderText("等待分析...")
        noise_layout.addWidget(self.txt_noise)
        result_layout.addWidget(noise_group)

        quality_group = QGroupBox("录音质量")
        quality_layout = QVBoxLayout(quality_group)
        self.txt_quality = QTextEdit()
        self.txt_quality.setReadOnly(True)
        self.txt_quality.setPlaceholderText("等待分析...")
        quality_layout.addWidget(self.txt_quality)
        result_layout.addWidget(quality_group)

        splitter.addWidget(result_widget)
        splitter.setSizes([400, 600])
        layout.addWidget(splitter, stretch=1)
        return widget

    # ── Deepfake Panel ────────────────────────────────────────────

    def _create_deepfake_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        ctrl = QHBoxLayout()
        self.btn_deepfake = QPushButton("运行 Deepfake 检测")
        self.btn_deepfake.clicked.connect(self._run_deepfake)
        ctrl.addWidget(self.btn_deepfake)
        self.progress_df = QProgressBar()
        self.progress_df.setVisible(False)
        ctrl.addWidget(self.progress_df)
        ctrl.addStretch()
        self.lbl_df_verdict = QLabel("")
        self.lbl_df_verdict.setStyleSheet("font-size: 14px; font-weight: bold;")
        ctrl.addWidget(self.lbl_df_verdict)
        layout.addLayout(ctrl)

        # Results
        self.txt_deepfake = QTextEdit()
        self.txt_deepfake.setReadOnly(True)
        self.txt_deepfake.setPlaceholderText("点击「运行 Deepfake 检测」开始分析...")
        self.txt_deepfake.setStyleSheet(
            "QTextEdit { background: #FBE9E7; border: 1px solid #FFAB91; border-radius: 5px; "
            "font-size: 12px; }"
        )
        layout.addWidget(self.txt_deepfake, stretch=1)
        return widget

    # ── AI Detection Panel ────────────────────────────────────────

    def _create_ai_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        ctrl = QHBoxLayout()
        self.btn_ai = QPushButton("运行 AI 生成检测")
        self.btn_ai.clicked.connect(self._run_ai_detection)
        ctrl.addWidget(self.btn_ai)
        self.progress_ai = QProgressBar()
        self.progress_ai.setVisible(False)
        ctrl.addWidget(self.progress_ai)
        ctrl.addStretch()
        self.lbl_ai_verdict = QLabel("")
        self.lbl_ai_verdict.setStyleSheet("font-size: 14px; font-weight: bold;")
        ctrl.addWidget(self.lbl_ai_verdict)
        layout.addLayout(ctrl)

        self.txt_ai = QTextEdit()
        self.txt_ai.setReadOnly(True)
        self.txt_ai.setPlaceholderText("点击「运行 AI 生成检测」开始分析...")
        self.txt_ai.setStyleSheet(
            "QTextEdit { background: #E8EAF6; border: 1px solid #9FA8DA; border-radius: 5px; "
            "font-size: 12px; }"
        )
        layout.addWidget(self.txt_ai, stretch=1)
        return widget

    # ── Model Management Panel ────────────────────────────────────

    def _create_model_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_group = QGroupBox("AI 模型状态")
        info_layout = QVBoxLayout(info_group)

        self.txt_model_status = QTextEdit()
        self.txt_model_status.setReadOnly(True)
        self.txt_model_status.setMaximumHeight(200)
        info_layout.addWidget(self.txt_model_status)

        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("刷新模型状态")
        btn_refresh.clicked.connect(self._refresh_model_status)
        btn_layout.addWidget(btn_refresh)

        btn_template = QPushButton("创建模型模板")
        btn_template.clicked.connect(self._create_model_templates)
        btn_layout.addWidget(btn_template)

        btn_layout.addStretch()
        info_layout.addLayout(btn_layout)
        layout.addWidget(info_group)

        # Instructions
        guide_group = QGroupBox("使用指南")
        guide_layout = QVBoxLayout(guide_group)
        guide_text = QLabel(
            "📌 如何导入本地 AI 模型:\n\n"
            "1. 将 ONNX 格式模型文件放入 models/ 目录\n"
            "2. 创建 config.json 描述模型输入输出规格\n"
            "3. 点击「刷新模型状态」加载模型\n\n"
            "支持的模型类型:\n"
            "• speaker_embedding - 说话人嵌入模型\n"
            "• deepfake_detector - 深度伪造检测模型\n"
            "• ai_generation_detector - AI生成检测模型\n\n"
            "模型格式要求: ONNX (.onnx)\n"
            "推荐框架: PyTorch → ONNX export, TensorFlow → tf2onnx\n\n"
            "⚠ 无模型时使用内置启发式算法分析（精度较低）\n"
            "✅ 导入专业模型后可显著提高检测准确性"
        )
        guide_text.setWordWrap(True)
        guide_text.setStyleSheet("QLabel { padding: 10px; line-height: 1.8; }")
        guide_layout.addWidget(guide_text)
        layout.addWidget(guide_group, stretch=1)

        # Initial status refresh
        self._refresh_model_status()

        return widget

    # ── Set Audio ─────────────────────────────────────────────────

    def set_audio(self, audio_data):
        self._audio = audio_data

    # ── Run Analysis Methods ──────────────────────────────────────

    def _run_voiceprint(self):
        if not self._audio or not self._audio.is_loaded:
            return
        self.btn_voiceprint.setEnabled(False)
        self.progress_vp.setVisible(True)
        self.progress_vp.setRange(0, 0)

        worker = VoiceprintWorker(self._audio.y, self._audio.sr)
        worker.finished.connect(self._on_voiceprint_done)
        worker.diarization_finished.connect(self._on_diarization_done)
        worker.error.connect(lambda e: self._on_error("voiceprint", e))
        self._workers["voiceprint"] = worker
        worker.start()

    def _on_voiceprint_done(self, result):
        self._results["voiceprint"] = result
        self.progress_vp.setVisible(False)
        self.btn_voiceprint.setEnabled(True)

        model_used = result.get("model_used", result.get("method", "N/A"))
        embedding_dim = result.get("embedding_dim", 0)

        rows = [("使用模型", model_used), ("嵌入维度", str(embedding_dim))]

        # Neural model specific fields
        if "embedding_norm" in result:
            rows.append(("嵌入向量L2范数", f"{result['embedding_norm']:.4f}"))
        if "embedding" in result:
            rows.append(("嵌入前10维", str([f"{v:.3f}" for v in result['embedding'][:10]])))

        # Heuristic model features
        features = result.get("features", {})
        for key, val in features.items():
            rows.append((key, f"{val:.4f}" if isinstance(val, float) else str(val)))

        self.tbl_voiceprint.setRowCount(len(rows))
        for row_idx, (k, v) in enumerate(rows):
            self.tbl_voiceprint.setItem(row_idx, 0, QTableWidgetItem(k))
            self.tbl_voiceprint.setItem(row_idx, 1, QTableWidgetItem(v))

    def _on_diarization_done(self, result):
        # Handle both neural and heuristic results
        n_speakers = result.get('num_speakers', result.get('n_speakers', 0))
        model = result.get('model', result.get('method', 'N/A'))
        lines = [f"检测到 {n_speakers} 个说话人"]
        lines.append(f"方法: {model}\n")

        if 'silhouette_score' in result:
            lines.append(f"聚类质量(轮廓系数): {result['silhouette_score']:.3f}")

        if 'speaker_durations' in result:
            lines.append("\n说话人时长分布:")
            for spk, dur in result['speaker_durations'].items():
                lines.append(f"  说话人 {spk}: {dur:.1f} 秒")

        if 'segments' in result and isinstance(result['segments'], list):
            lines.append(f"\n时间线片段 (共 {result.get('total_segments', len(result['segments']))} 段):")
            for seg in result['segments'][:20]:  # Show first 20
                if len(seg) == 3:
                    start, end, spk = seg
                    lines.append(f"  [{start:.1f}s - {end:.1f}s] → 说话人 {spk}")

        # Heuristic model stats
        for spk, stats in result.get("speaker_stats", {}).items():
            lines.append(f"\n【{spk}】")
            for k, v in stats.items():
                lines.append(f"  {k}: {v}")

        self.txt_diarization.setPlainText("\n".join(lines))

    def _run_content(self):
        if not self._audio or not self._audio.is_loaded:
            return
        self.btn_content.setEnabled(False)
        self.progress_ct.setVisible(True)
        self.progress_ct.setRange(0, 0)

        worker = ContentWorker(self._audio.y, self._audio.sr)
        worker.finished.connect(self._on_content_done)
        worker.error.connect(lambda e: self._on_error("content", e))
        self._workers["content"] = worker
        worker.start()

    def _on_content_done(self, result):
        self._results["content"] = result
        self.progress_ct.setVisible(False)
        self.btn_content.setEnabled(True)

        # VAD statistics
        vad = result.get("vad", result.get("voice_activity", {}))
        vad_lines = ["【语音活动检测结果】\n"]

        # Neural VAD (Silero) results
        if "model_used" in vad:
            vad_lines.append(f"  模型: {vad['model_used']}")
        if "speech_ratio" in vad:
            vad_lines.append(f"  语音占比: {vad['speech_ratio'] * 100:.1f}%")
            vad_lines.append(f"  语音时长: {vad.get('speech_duration', 0):.1f} 秒")
            vad_lines.append(f"  总时长: {vad.get('total_duration', 0):.1f} 秒")
            vad_lines.append(f"  语音段数: {vad.get('num_segments', 0)}")
            if vad.get('speech_segments'):
                vad_lines.append(f"\n  语音片段:")
                for i, (start, end) in enumerate(vad['speech_segments'][:15]):
                    vad_lines.append(f"    [{start:.1f}s - {end:.1f}s] ({end - start:.1f}s)")
        else:
            # Heuristic VAD
            stats = vad.get("statistics", {})
            for k, v in stats.items():
                vad_lines.append(f"  {k}: {v}")
        self.txt_vad.setPlainText("\n".join(vad_lines))

        # Noise profile
        noise = result.get("noise_profile", {})
        noise_lines = [
            "【噪声分析结果】\n",
            f"  噪声类型: {noise.get('noise_type', 'N/A')}",
            f"  描述: {noise.get('noise_description', '')}",
            f"  噪声电平: {noise.get('noise_level_db', 0):.1f} dB",
            f"  信号电平: {noise.get('signal_level_db', 0):.1f} dB",
            f"  信噪比: {noise.get('snr_db', 0):.1f} dB",
            f"\n  质量评估: {noise.get('assessment', 'N/A')}",
        ]
        hum = noise.get("hum_detected", {})
        for freq, data in hum.items():
            if data.get("detected"):
                noise_lines.append(f"\n  ⚠ 检测到 {freq} 工频干扰 (强度比: {data['strength_ratio']:.1f})")
        self.txt_noise.setPlainText("\n".join(noise_lines))

        # Quality
        quality = result.get("speech_quality", {})
        quality_lines = [
            "【录音质量评估】\n",
            f"  质量评分: {quality.get('quality_score', 0)}/100",
            f"  等级: {quality.get('grade', 'N/A')}",
            f"  削波比例: {quality.get('clipping_ratio', 0) * 100:.3f}%",
            f"  动态范围: {quality.get('dynamic_range_db', 0):.1f} dB",
            f"  频谱带宽: {quality.get('spectral_bandwidth_hz', 0):.0f} Hz",
            "\n  发现的问题:",
        ]
        for issue in quality.get("issues", []):
            quality_lines.append(f"    • {issue}")
        self.txt_quality.setPlainText("\n".join(quality_lines))

        # Plot VAD
        self._plot_vad(vad)

    def _plot_vad(self, vad):
        self.fig_content.clear()
        if not vad:
            self.canvas_content.draw()
            return

        ax = self.fig_content.add_subplot(111)

        # Plot waveform lightly
        audio_times = np.arange(len(self._audio.y)) / self._audio.sr
        ax.plot(audio_times, self._audio.y, color=COLORS["waveform"], alpha=0.3, linewidth=0.3)

        # Handle neural VAD (Silero) with probabilities
        if "probabilities" in vad:
            probs = vad["probabilities"]
            window_dur = vad.get("window_duration", 0.032)
            prob_times = np.arange(len(probs)) * window_dur
            # Plot probability curve
            ax2 = ax.twinx()
            ax2.plot(prob_times, probs, color='orange', alpha=0.7, linewidth=0.8, label="语音概率")
            ax2.set_ylabel("语音概率")
            ax2.set_ylim(0, 1)
            ax2.axhline(0.5, color='red', linestyle='--', alpha=0.3)
            # Highlight speech segments
            for start, end in vad.get("speech_segments", []):
                ax.axvspan(start, end, alpha=0.15, color=COLORS["success"])
        elif "times" in vad:
            # Heuristic VAD
            times = vad["times"]
            is_speech = vad["is_speech"].astype(float)
            ax.fill_between(times, -1, 1, where=is_speech > 0.5,
                            alpha=0.2, color=COLORS["success"], label="语音")
            ax.fill_between(times, -1, 1, where=is_speech < 0.5,
                            alpha=0.1, color=COLORS["warning"], label="静音/噪声")

        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("振幅")
        model_name = vad.get("model", "VAD")
        ax.set_title(f"语音活动检测 ({model_name})")
        ax.set_xlim(0, len(self._audio.y) / self._audio.sr)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        self.fig_content.tight_layout()
        self.canvas_content.draw()

    def _run_deepfake(self):
        if not self._audio or not self._audio.is_loaded:
            return
        self.btn_deepfake.setEnabled(False)
        self.progress_df.setVisible(True)
        self.progress_df.setRange(0, 0)

        worker = DeepfakeWorker(self._audio.y, self._audio.sr)
        worker.finished.connect(self._on_deepfake_done)
        worker.error.connect(lambda e: self._on_error("deepfake", e))
        self._workers["deepfake"] = worker
        worker.start()

    def _on_deepfake_done(self, result):
        self._results["deepfake"] = result
        self.progress_df.setVisible(False)
        self.btn_deepfake.setEnabled(True)

        verdict = result.get("verdict", "未知")
        risk = result.get("risk_level", "")
        score = result.get("final_score", 0)

        color_map = {"高": "#D32F2F", "中": "#FF8F00", "低": "#388E3C", "极低": "#1976D2"}
        color = color_map.get(risk, "#333")
        self.lbl_df_verdict.setText(f"判定: {verdict} (风险: {risk})")
        self.lbl_df_verdict.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color};")

        lines = [
            f"{'=' * 50}",
            f"  Deepfake 检测结果",
            f"{'=' * 50}",
            f"\n综合评分: {score:.3f} / 1.0",
            f"判定结果: {verdict}",
            f"风险等级: {risk}",
            f"使用AI模型: {'是' if result.get('model_used') else '否 (使用启发式算法)'}",
            f"\n{result.get('verdict_description', '')}",
            f"\n{'─' * 50}",
            "各维度评分:",
        ]
        for k, v in result.get("scores", {}).items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            lines.append(f"  {k:25s} [{bar}] {v:.3f}")

        lines.append(f"\n{'─' * 50}")
        lines.append("详细分析:")

        details = result.get("details", {})
        for category, detail in details.items():
            if detail is None:
                continue
            lines.append(f"\n【{category}】")
            for indicator in detail.get("indicators", []):
                lines.append(f"  • {indicator}")

        self.txt_deepfake.setPlainText("\n".join(lines))

    def _run_ai_detection(self):
        if not self._audio or not self._audio.is_loaded:
            return
        self.btn_ai.setEnabled(False)
        self.progress_ai.setVisible(True)
        self.progress_ai.setRange(0, 0)

        worker = AIDetectionWorker(self._audio.y, self._audio.sr)
        worker.finished.connect(self._on_ai_done)
        worker.error.connect(lambda e: self._on_error("ai", e))
        self._workers["ai"] = worker
        worker.start()

    def _on_ai_done(self, result):
        self._results["ai_detection"] = result
        self.progress_ai.setVisible(False)
        self.btn_ai.setEnabled(True)

        verdict = result.get("verdict", "未知")
        confidence = result.get("confidence", "")
        score = result.get("final_score", 0)

        color_map = {"AI生成": "#D32F2F", "疑似AI生成": "#FF8F00",
                     "不确定": "#FFA000", "自然录音": "#388E3C"}
        color = color_map.get(verdict, "#333")
        self.lbl_ai_verdict.setText(f"判定: {verdict} (置信度: {confidence})")
        self.lbl_ai_verdict.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color};")

        lines = [
            f"{'=' * 50}",
            f"  AI 生成音频检测结果",
            f"{'=' * 50}",
            f"\n综合评分: {score:.3f} / 1.0",
            f"判定结果: {verdict}",
            f"置信度: {confidence}",
            f"使用AI模型: {'是' if result.get('model_used') else '否 (使用启发式算法)'}",
            f"\n{result.get('verdict_description', '')}",
            f"\n{'─' * 50}",
            "各维度评分:",
        ]
        for k, v in result.get("scores", {}).items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            lines.append(f"  {k:25s} [{bar}] {v:.3f}")

        lines.append(f"\n{'─' * 50}")
        lines.append("检测证据:")
        for indicator in result.get("indicators", []):
            lines.append(f"  • {indicator}")

        self.txt_ai.setPlainText("\n".join(lines))

    def _on_error(self, module, error_msg):
        for key in ["voiceprint", "content", "deepfake", "ai"]:
            getattr(self, f"progress_{key[:2]}", None)
        # Reset relevant buttons
        if module == "voiceprint":
            self.progress_vp.setVisible(False)
            self.btn_voiceprint.setEnabled(True)
        elif module == "content":
            self.progress_ct.setVisible(False)
            self.btn_content.setEnabled(True)
        elif module == "deepfake":
            self.progress_df.setVisible(False)
            self.btn_deepfake.setEnabled(True)
        elif module == "ai":
            self.progress_ai.setVisible(False)
            self.btn_ai.setEnabled(True)

    def _refresh_model_status(self):
        from pathlib import Path
        import json
        import sys

        # Direct check for onnxruntime availability
        onnx_available = False
        onnx_version = "N/A"
        onnx_error = ""
        try:
            import onnxruntime as _ort
            onnx_available = True
            onnx_version = _ort.__version__
        except BaseException as e:
            onnx_error = f"{type(e).__name__}: {e}"
        
        # Also check if it's already in sys.modules (may have been imported elsewhere)
        if not onnx_available and 'onnxruntime' in sys.modules:
            try:
                _ort = sys.modules['onnxruntime']
                onnx_available = True
                onnx_version = _ort.__version__
            except BaseException:
                pass

        lines = [
            "【AI 模型运行时状态】\n",
            f"  ONNX Runtime: {'✅ 已安装 (v' + onnx_version + ')' if onnx_available else '❌ 未安装'}",
        ]
        if not onnx_available and onnx_error:
            lines.append(f"  错误信息: {onnx_error}")
            lines.append(f"  Python: {sys.executable}")

        # Resolve model directory robustly
        from core.model_manager import get_model_manager
        mgr = get_model_manager()
        models_dir = Path(mgr.model_dir)
        lines.append(f"  模型目录: {models_dir}")

        # Scan actual model directories
        installed_models = []
        if models_dir.exists():
            for sub in models_dir.iterdir():
                if sub.is_dir():
                    config_file = sub / "config.json"
                    if config_file.exists():
                        try:
                            cfg = json.loads(config_file.read_text(encoding='utf-8'))
                            model_file = sub / cfg.get("model_file", "")
                            size_mb = model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0
                            installed_models.append({
                                'name': cfg.get('name', sub.name),
                                'type': cfg.get('type', 'unknown'),
                                'size_mb': size_mb,
                                'source': cfg.get('source', ''),
                            })
                        except Exception:
                            pass

        lines.append(f"  已安装模型: {len(installed_models)} 个\n")

        if installed_models:
            lines.append("  ┌─────────────────────────────────────────────────┐")
            for m in installed_models:
                lines.append(f"  │ ✅ {m['name']}")
                lines.append(f"  │    类型: {m['type']} | 大小: {m['size_mb']:.1f} MB")
                if m['source']:
                    lines.append(f"  │    来源: {m['source']}")
                lines.append(f"  │")
            lines.append("  └─────────────────────────────────────────────────┘")
        else:
            lines.append("  ⚠ 未找到本地模型")
            lines.append("  将使用内置启发式算法进行分析")

        self.txt_model_status.setPlainText("\n".join(lines))

    def _create_model_templates(self):
        from core.model_manager import get_model_manager
        mgr = get_model_manager()
        for task in ["speaker_embedding", "deepfake_detector", "ai_generation_detector"]:
            mgr.create_model_template(task)
        self._refresh_model_status()

    def get_results(self) -> Dict:
        return self._results

    def clear(self):
        self._audio = None
        self._results = {}
        self.tbl_voiceprint.setRowCount(0)
        self.txt_diarization.clear()
        self.txt_vad.clear()
        self.txt_noise.clear()
        self.txt_quality.clear()
        self.txt_deepfake.clear()
        self.txt_ai.clear()
        self.lbl_df_verdict.setText("")
        self.lbl_ai_verdict.setText("")
        self.fig_content.clear()
        self.canvas_content.draw()
