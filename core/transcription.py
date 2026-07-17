"""Speech transcription module using Whisper model.

Provides high-accuracy speech-to-text transcription with timestamps,
language detection, and exportable results.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path


class SpeechTranscriber:
    """Speech transcription using faster-whisper (CTranslate2 backend).
    
    Supports multiple model sizes for accuracy/speed tradeoff:
    - tiny: ~75MB, fastest, lower accuracy
    - base: ~145MB, good balance
    - small: ~484MB, better accuracy  
    - medium: ~1.5GB, high accuracy
    - large-v3: ~3GB, highest accuracy
    """
    
    _instance = None
    _model = None
    _model_size = None
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None
    
    def _ensure_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None and self._model_size == self.model_size:
            return
        
        from faster_whisper import WhisperModel
        
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type="int8" if self.device == "cpu" else "float16",
        )
        self._model_size = self.model_size
    
    def transcribe(self, audio_path: str = None,
                   audio_array: np.ndarray = None,
                   sr: int = 16000,
                   language: str = None,
                   task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            audio_array: Audio numpy array (alternative to path)
            sr: Sample rate (used with audio_array)
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate" (to English)
            
        Returns:
            Dict with text, segments, language, and metadata
        """
        self._ensure_model()
        
        # Prepare audio input
        if audio_array is not None:
            import librosa
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            audio_input = audio_array.astype(np.float32)
        elif audio_path:
            audio_input = audio_path
        else:
            raise ValueError("Must provide audio_path or audio_array")
        
        # Run transcription
        segments_gen, info = self._model.transcribe(
            audio_input,
            language=language,
            task=task,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )
        
        # Collect segments
        segments = []
        full_text_parts = []
        
        for seg in segments_gen:
            segment_data = {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            }
            
            # Word-level timestamps
            if seg.words:
                segment_data["words"] = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ]
            
            segments.append(segment_data)
            full_text_parts.append(seg.text.strip())
        
        full_text = " ".join(full_text_parts)
        
        return {
            "text": full_text,
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "model_size": self.model_size,
            "num_segments": len(segments),
        }
    
    def save_transcript(self, result: Dict[str, Any], output_path: str,
                        format: str = "txt") -> str:
        """
        Save transcription result to file.
        
        Args:
            result: Transcription result dict
            output_path: Output file path (without extension)
            format: "txt", "srt", "vtt", or "json"
            
        Returns:
            Path to saved file
        """
        import json
        
        if format == "txt":
            path = f"{output_path}.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"语言: {result.get('language', 'unknown')} "
                        f"(置信度: {result.get('language_probability', 0):.1%})\n")
                f.write(f"模型: Whisper {result.get('model_size', '')}\n")
                f.write(f"时长: {result.get('duration', 0):.1f}秒\n")
                f.write("=" * 60 + "\n\n")
                
                for seg in result.get("segments", []):
                    start = self._format_time(seg["start"])
                    end = self._format_time(seg["end"])
                    f.write(f"[{start} → {end}] {seg['text']}\n")
                
                f.write("\n" + "=" * 60 + "\n完整文本:\n\n")
                f.write(result.get("text", ""))
        
        elif format == "srt":
            path = f"{output_path}.srt"
            with open(path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(result.get("segments", []), 1):
                    start = self._format_srt_time(seg["start"])
                    end = self._format_srt_time(seg["end"])
                    f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")
        
        elif format == "vtt":
            path = f"{output_path}.vtt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for seg in result.get("segments", []):
                    start = self._format_vtt_time(seg["start"])
                    end = self._format_vtt_time(seg["end"])
                    f.write(f"{start} --> {end}\n{seg['text']}\n\n")
        
        elif format == "json":
            path = f"{output_path}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return path
    
    def _format_time(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"
    
    def _format_srt_time(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        ms = int((s % 1) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        ms = int((s % 1) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"
