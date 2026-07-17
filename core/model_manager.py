"""Local AI Model Manager for Audio Forensics Tool.

Manages loading, caching, and inference of local AI models.
Supports ONNX Runtime for cross-platform inference.
Falls back to heuristic methods when models are unavailable.
"""

import os
import json
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except (ImportError, Exception):
    HAS_ONNX = False
    ort = None


# Default model directory - check multiple locations
_file_based_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_cwd_based_dir = os.path.join(os.getcwd(), "models")

# Prefer the one that actually exists and has models
if os.path.isdir(_file_based_dir):
    MODEL_DIR = _file_based_dir
elif os.path.isdir(_cwd_based_dir):
    MODEL_DIR = _cwd_based_dir
else:
    MODEL_DIR = _file_based_dir  # fallback


class ModelManager:
    """Manage local AI models for audio forensic analysis.

    Supports ONNX models for:
    - Speaker embedding (voiceprint)
    - Deepfake detection
    - AI-generated audio detection

    Falls back to heuristic/statistical methods when models are unavailable.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or MODEL_DIR
        self._sessions: Dict[str, Any] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        os.makedirs(self.model_dir, exist_ok=True)
        self._scan_models()

    def _scan_models(self):
        """Scan model directory for available models."""
        self._model_info = {}
        if not os.path.exists(self.model_dir):
            return

        for item in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, item)
            # Check for model config
            config_path = os.path.join(model_path, "config.json")
            if os.path.isdir(model_path) and os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self._model_info[item] = config
            elif item.endswith(".onnx"):
                self._model_info[item] = {
                    "name": item,
                    "type": "onnx",
                    "path": model_path,
                }

    @property
    def has_onnx_runtime(self) -> bool:
        return HAS_ONNX

    @property
    def available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._model_info.copy()

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get path to a model file."""
        if model_name in self._model_info:
            info = self._model_info[model_name]
            return info.get("path", os.path.join(self.model_dir, model_name))
        # Direct path check
        direct = os.path.join(self.model_dir, model_name)
        if os.path.exists(direct):
            return direct
        return None

    def load_onnx_model(self, model_name: str) -> Optional[Any]:
        """Load an ONNX model session."""
        if not HAS_ONNX:
            return None

        if model_name in self._sessions:
            return self._sessions[model_name]

        model_path = self.get_model_path(model_name)
        if model_path is None:
            return None

        # Find the .onnx file
        if os.path.isdir(model_path):
            onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")]
            if not onnx_files:
                return None
            model_file = os.path.join(model_path, onnx_files[0])
        else:
            model_file = model_path

        if not os.path.exists(model_file):
            return None

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(model_file, sess_options)
            self._sessions[model_name] = session
            return session
        except Exception:
            return None

    def run_inference(
        self, model_name: str, inputs: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Run inference on a loaded model."""
        session = self.load_onnx_model(model_name)
        if session is None:
            return None

        try:
            results = session.run(None, inputs)
            output_names = [o.name for o in session.get_outputs()]
            return dict(zip(output_names, results))
        except Exception:
            return None

    def get_model_status(self) -> Dict[str, Any]:
        """Get status report of available models and runtime."""
        return {
            "onnx_runtime_available": HAS_ONNX,
            "onnx_version": ort.__version__ if HAS_ONNX else None,
            "model_directory": self.model_dir,
            "models_found": len(self._model_info),
            "models": list(self._model_info.keys()),
            "loaded_sessions": list(self._sessions.keys()),
        }

    def create_model_template(self, model_type: str) -> str:
        """Create a config template for users to add their own models.

        Returns path to the created template directory.
        """
        template_dir = os.path.join(self.model_dir, f"{model_type}_template")
        os.makedirs(template_dir, exist_ok=True)

        config = {
            "name": f"{model_type}_model",
            "type": "onnx",
            "task": model_type,
            "description": f"Place your {model_type} ONNX model file here",
            "input_spec": {
                "input_name": "audio_features",
                "input_shape": "dynamic",
                "sample_rate": 16000,
            },
            "output_spec": {
                "output_name": "predictions",
            },
        }

        config_path = os.path.join(template_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        readme_path = os.path.join(template_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {model_type} Model\n\n")
            f.write(f"将你的 ONNX 模型文件放入此目录。\n\n")
            f.write("## 支持的模型格式\n")
            f.write("- ONNX (.onnx)\n\n")
            f.write("## 使用方法\n")
            f.write("1. 将模型文件 (.onnx) 放入此目录\n")
            f.write("2. 修改 config.json 中的输入输出规格\n")
            f.write("3. 重启应用程序\n")

        return template_dir


# Global model manager instance
_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create the global ModelManager instance."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
