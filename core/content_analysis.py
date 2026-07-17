"""Audio content analysis module.

Provides speech/silence/noise segmentation, voice activity detection (VAD),
noise profiling, and audio event classification.
"""

from typing import Dict, Any, List, Tuple

import numpy as np
import librosa
from scipy import signal as scipy_signal

from utils.constants import DEFAULT_N_FFT, DEFAULT_HOP_LENGTH


class ContentAnalyzer:
    """Analyze audio content: speech segments, silence, noise, and events."""

    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr

    def full_analysis(self) -> Dict[str, Any]:
        """Run complete content analysis pipeline."""
        vad = self.detect_voice_activity()
        noise = self.analyze_noise_profile()
        segments = self.segment_content()
        quality = self.assess_speech_quality()

        return {
            "voice_activity": vad,
            "noise_profile": noise,
            "segments": segments,
            "speech_quality": quality,
        }

    def detect_voice_activity(
        self, frame_duration: float = 0.025, energy_threshold_db: float = -40
    ) -> Dict[str, Any]:
        """Detect voice activity using energy and spectral features.

        Returns:
            Dictionary with VAD results, speech/silence segments.
        """
        frame_len = int(frame_duration * self.sr)
        hop = frame_len // 2

        # Compute frame energy
        rms = librosa.feature.rms(y=self.y, frame_length=frame_len, hop_length=hop)[0]
        rms_db = 20 * np.log10(rms + 1e-10)

        # Compute spectral flatness (speech has lower flatness than noise)
        flatness = librosa.feature.spectral_flatness(y=self.y, n_fft=frame_len, hop_length=hop)[0]

        # Adaptive threshold
        sorted_rms = np.sort(rms_db)
        noise_floor = np.mean(sorted_rms[:len(sorted_rms) // 4])
        adaptive_threshold = max(energy_threshold_db, noise_floor + 10)

        # VAD decision
        is_speech = (rms_db > adaptive_threshold) & (flatness < 0.5)

        # Smooth: remove very short segments
        min_speech_frames = int(0.1 * self.sr / hop)  # 100ms minimum
        is_speech = self._smooth_vad(is_speech, min_speech_frames)

        times = librosa.frames_to_time(np.arange(len(is_speech)), sr=self.sr, hop_length=hop)

        # Extract speech and silence segments
        speech_segments = self._extract_segments(times, is_speech, True)
        silence_segments = self._extract_segments(times, is_speech, False)

        total_duration = len(self.y) / self.sr
        speech_duration = sum(s["end"] - s["start"] for s in speech_segments)
        silence_duration = total_duration - speech_duration

        return {
            "is_speech": is_speech,
            "times": times,
            "speech_segments": speech_segments,
            "silence_segments": silence_segments,
            "statistics": {
                "总时长 (秒)": round(total_duration, 2),
                "语音时长 (秒)": round(speech_duration, 2),
                "静音时长 (秒)": round(silence_duration, 2),
                "语音占比": f"{speech_duration / total_duration * 100:.1f}%",
                "语音段落数": len(speech_segments),
                "静音段落数": len(silence_segments),
                "自适应阈值 (dB)": round(adaptive_threshold, 1),
                "噪声底部 (dB)": round(noise_floor, 1),
            },
        }

    def analyze_noise_profile(self) -> Dict[str, Any]:
        """Analyze noise characteristics of the recording.

        Returns:
            Noise profile with spectral characteristics and type classification.
        """
        # Use first/last 0.5s and detected silence for noise estimation
        noise_len = min(int(0.5 * self.sr), len(self.y) // 10)
        noise_samples = np.concatenate([self.y[:noise_len], self.y[-noise_len:]])

        # Noise spectrum
        freqs, psd = scipy_signal.welch(noise_samples, fs=self.sr, nperseg=min(2048, len(noise_samples)))

        # Noise level
        noise_rms = np.sqrt(np.mean(noise_samples ** 2))
        noise_db = 20 * np.log10(noise_rms + 1e-10)

        # Signal level (from loudest portions)
        rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        signal_rms = np.percentile(rms, 90)
        signal_db = 20 * np.log10(signal_rms + 1e-10)
        snr = signal_db - noise_db

        # Spectral shape analysis for noise type classification
        spectral_slope = np.polyfit(np.log10(freqs[1:] + 1), 10 * np.log10(psd[1:] + 1e-10), 1)[0]

        # Classify noise type based on spectral characteristics
        if spectral_slope < -5:
            noise_type = "粉红噪声/环境噪声"
            noise_desc = "频谱随频率下降，典型的环境背景噪声"
        elif spectral_slope < -2:
            noise_type = "布朗噪声/低频噪声"
            noise_desc = "低频能量占主导，可能来自空调、交通等"
        elif abs(spectral_slope) < 2:
            noise_type = "白噪声/电子噪声"
            noise_desc = "频谱平坦，可能来自电子设备自身噪声"
        else:
            noise_type = "高频噪声"
            noise_desc = "高频能量较强，可能来自电子干扰"

        # Check for specific noise patterns
        hum_detected = self._detect_hum(freqs, psd)

        return {
            "noise_level_db": float(noise_db),
            "signal_level_db": float(signal_db),
            "snr_db": float(snr),
            "noise_type": noise_type,
            "noise_description": noise_desc,
            "spectral_slope": float(spectral_slope),
            "hum_detected": hum_detected,
            "freqs": freqs,
            "psd": psd,
            "assessment": self._assess_noise_quality(snr),
        }

    def _detect_hum(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, Any]:
        """Detect power line hum (50/60 Hz)."""
        results = {}
        for freq, name in [(50, "50Hz"), (60, "60Hz")]:
            mask = (freqs >= freq - 2) & (freqs <= freq + 2)
            if np.any(mask):
                peak_power = np.max(psd[mask])
                avg_power = np.mean(psd)
                ratio = peak_power / (avg_power + 1e-10)
                results[name] = {
                    "detected": ratio > 10,
                    "strength_ratio": float(ratio),
                }
        return results

    def _assess_noise_quality(self, snr: float) -> str:
        """Assess recording quality based on SNR."""
        if snr > 30:
            return "优秀 - 录音质量极佳，噪声几乎不可察觉"
        elif snr > 20:
            return "良好 - 录音质量好，轻微背景噪声"
        elif snr > 10:
            return "一般 - 有明显背景噪声，但语音可辨识"
        elif snr > 5:
            return "较差 - 噪声较大，可能影响语音辨识"
        else:
            return "极差 - 噪声严重，语音难以辨识"

    def segment_content(self) -> Dict[str, Any]:
        """Segment audio into content types (speech, music, noise, silence)."""
        hop_length = DEFAULT_HOP_LENGTH
        frame_length = DEFAULT_N_FFT

        # Features per frame
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(self.y, frame_length=frame_length, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=self.y, n_fft=frame_length, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr, n_fft=frame_length, hop_length=hop_length)[0]

        n_frames = min(len(rms), len(zcr), len(flatness), len(centroid))
        rms = rms[:n_frames]
        zcr = zcr[:n_frames]
        flatness = flatness[:n_frames]
        centroid = centroid[:n_frames]

        rms_db = 20 * np.log10(rms + 1e-10)
        silence_threshold = np.percentile(rms_db, 10) + 10

        labels = np.full(n_frames, "noise", dtype=object)
        for i in range(n_frames):
            if rms_db[i] < silence_threshold:
                labels[i] = "silence"
            elif flatness[i] < 0.3 and zcr[i] < 0.15:
                labels[i] = "speech"
            elif flatness[i] > 0.6:
                labels[i] = "noise"
            elif centroid[i] > 3000 and flatness[i] > 0.4:
                labels[i] = "music"
            else:
                labels[i] = "speech"

        times = librosa.frames_to_time(np.arange(n_frames), sr=self.sr, hop_length=hop_length)

        # Summarize
        content_types = {"speech": 0, "silence": 0, "noise": 0, "music": 0}
        frame_dur = hop_length / self.sr
        for label in labels:
            content_types[label] += frame_dur

        return {
            "labels": labels,
            "times": times,
            "duration_by_type": {k: round(v, 2) for k, v in content_types.items()},
            "dominant_content": max(content_types, key=lambda k: content_types[k] if k != "silence" else 0),
        }

    def assess_speech_quality(self) -> Dict[str, Any]:
        """Assess overall speech/recording quality metrics."""
        # Compute various quality indicators
        rms = librosa.feature.rms(y=self.y)[0]
        rms_db = 20 * np.log10(rms + 1e-10)

        # Clipping detection
        clip_threshold = 0.99
        n_clipped = np.sum(np.abs(self.y) > clip_threshold)
        clip_ratio = n_clipped / len(self.y)

        # Dynamic range
        dr = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)

        # Spectral quality
        spectral_bw = np.mean(librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr))

        # Overall score
        score = 100
        issues = []

        if clip_ratio > 0.01:
            score -= 30
            issues.append(f"严重削波 ({clip_ratio * 100:.2f}% 采样点)")
        elif clip_ratio > 0.001:
            score -= 15
            issues.append(f"轻微削波 ({clip_ratio * 100:.3f}% 采样点)")

        if dr < 10:
            score -= 20
            issues.append(f"动态范围不足 ({dr:.1f} dB)")
        elif dr < 20:
            score -= 10

        if spectral_bw < 1000:
            score -= 15
            issues.append("频带较窄，可能为电话质量")

        if not issues:
            issues.append("未发现明显质量问题")

        return {
            "quality_score": max(0, score),
            "clipping_ratio": float(clip_ratio),
            "dynamic_range_db": float(dr),
            "spectral_bandwidth_hz": float(spectral_bw),
            "issues": issues,
            "grade": "优秀" if score >= 90 else "良好" if score >= 70 else "一般" if score >= 50 else "较差",
        }

    def _smooth_vad(self, vad: np.ndarray, min_frames: int) -> np.ndarray:
        """Smooth VAD output by removing very short segments."""
        result = vad.copy()
        in_segment = False
        seg_start = 0

        for i in range(len(result)):
            if result[i] and not in_segment:
                seg_start = i
                in_segment = True
            elif not result[i] and in_segment:
                if i - seg_start < min_frames:
                    result[seg_start:i] = False
                in_segment = False

        return result

    def _extract_segments(
        self, times: np.ndarray, is_speech: np.ndarray, target: bool
    ) -> List[Dict[str, float]]:
        """Extract continuous segments from binary VAD."""
        segments = []
        in_seg = False
        start = 0

        for i in range(len(is_speech)):
            if is_speech[i] == target and not in_seg:
                start = times[i]
                in_seg = True
            elif is_speech[i] != target and in_seg:
                segments.append({"start": float(start), "end": float(times[i])})
                in_seg = False

        if in_seg:
            segments.append({"start": float(start), "end": float(times[-1])})

        return segments
