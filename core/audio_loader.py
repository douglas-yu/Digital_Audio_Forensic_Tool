"""Audio file loader module for Audio Forensics Tool."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import librosa
import soundfile as sf

from utils.constants import DEFAULT_SR, SUPPORTED_EXTENSIONS


@dataclass
class AudioData:
    """Container for loaded audio data and metadata."""
    file_path: str = ""
    file_name: str = ""
    y: Optional[np.ndarray] = None          # audio time series (mono)
    y_stereo: Optional[np.ndarray] = None   # original stereo if applicable
    sr: int = DEFAULT_SR
    duration: float = 0.0
    n_samples: int = 0
    n_channels: int = 1
    bit_depth: int = 16
    file_size: int = 0
    file_format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_loaded(self) -> bool:
        return self.y is not None and len(self.y) > 0


class AudioLoader:
    """Handles loading and basic validation of audio files."""

    def __init__(self):
        self._current_audio: Optional[AudioData] = None

    @property
    def current(self) -> Optional[AudioData]:
        return self._current_audio

    def load(self, file_path: str, target_sr: Optional[int] = None) -> AudioData:
        """Load an audio file and return AudioData.

        Args:
            file_path: Path to the audio file.
            target_sr: Target sample rate. None keeps original.

        Returns:
            AudioData with loaded audio information.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not supported.
            RuntimeError: If loading fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的音频格式: {ext}")

        audio = AudioData()
        audio.file_path = file_path
        audio.file_name = os.path.basename(file_path)
        audio.file_size = os.path.getsize(file_path)
        audio.file_format = ext[1:].upper()

        try:
            # Get file info via soundfile
            info = sf.info(file_path)
            audio.n_channels = info.channels
            audio.bit_depth = _get_bit_depth(info.subtype)
            audio.metadata = self._extract_metadata(file_path, info)

            # Load with librosa (always returns mono float32)
            sr_to_use = target_sr if target_sr else info.samplerate
            audio.y, audio.sr = librosa.load(file_path, sr=sr_to_use, mono=True)

            # Also load stereo if multi-channel
            if info.channels > 1:
                audio.y_stereo, _ = librosa.load(file_path, sr=sr_to_use, mono=False)

            audio.n_samples = len(audio.y)
            audio.duration = audio.n_samples / audio.sr

        except Exception as e:
            raise RuntimeError(f"加载音频文件失败: {e}") from e

        self._current_audio = audio
        return audio

    def _extract_metadata(self, file_path: str, info: sf.SoundFile) -> Dict[str, Any]:
        """Extract metadata from audio file."""
        meta = {
            "格式": info.format,
            "子类型": info.subtype,
            "采样率": info.samplerate,
            "声道数": info.channels,
            "帧数": info.frames,
            "时长(秒)": info.frames / info.samplerate if info.samplerate > 0 else 0,
            "文件大小": os.path.getsize(file_path),
            "修改时间": os.path.getmtime(file_path),
        }

        # Try to read extra sections for WAV files
        try:
            with sf.SoundFile(file_path) as f:
                if hasattr(f, "extra_info") and f.extra_info:
                    meta["额外信息"] = f.extra_info
        except Exception:
            pass

        return meta

    def unload(self):
        """Unload current audio data."""
        self._current_audio = None


def _get_bit_depth(subtype: str) -> int:
    """Map soundfile subtype to bit depth."""
    mapping = {
        "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
        "PCM_S8": 8, "PCM_U8": 8,
        "FLOAT": 32, "DOUBLE": 64,
    }
    return mapping.get(subtype, 16)
