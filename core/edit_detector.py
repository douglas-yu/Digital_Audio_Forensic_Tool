"""Audio edit/splice detection module."""

from typing import List, Dict, Any, Tuple

import numpy as np
import librosa
from scipy import signal as scipy_signal

from utils.constants import DEFAULT_N_FFT, DEFAULT_HOP_LENGTH


class EditDetector:
    """Detect potential edits, splices, and discontinuities in audio recordings.

    Techniques:
    - Spectral discontinuity detection
    - Phase discontinuity analysis
    - Energy envelope anomalies
    - Statistical change-point detection (CUSUM)
    """

    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr

    def detect_all(self, sensitivity: float = 0.5) -> Dict[str, Any]:
        """Run all detection methods and merge results.

        Args:
            sensitivity: Detection sensitivity 0.0-1.0 (higher = more detections).

        Returns:
            Dictionary with detected edit points and details per method.
        """
        results = {
            "spectral": self.detect_spectral_discontinuities(sensitivity),
            "phase": self.detect_phase_discontinuities(sensitivity),
            "energy": self.detect_energy_anomalies(sensitivity),
            "statistical": self.detect_statistical_changepoints(sensitivity),
        }

        # Merge all detected points
        all_points: List[Dict[str, Any]] = []
        for method, detections in results.items():
            for det in detections:
                det["method"] = method
                all_points.append(det)

        # Sort by time
        all_points.sort(key=lambda x: x["time"])

        # Merge nearby detections (within 0.05s)
        merged = self._merge_nearby(all_points, threshold=0.05)

        return {
            "edit_points": merged,
            "total_detections": len(merged),
            "by_method": {k: len(v) for k, v in results.items()},
            "details": results,
        }

    def detect_spectral_discontinuities(
        self, sensitivity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect abrupt spectral changes using spectral flux."""
        n_fft = DEFAULT_N_FFT
        hop_length = DEFAULT_HOP_LENGTH

        S = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length))
        # Spectral flux: sum of positive differences between consecutive frames
        flux = np.sum(np.maximum(0, np.diff(S, axis=1)), axis=0)

        threshold = np.mean(flux) + (1.5 - sensitivity) * np.std(flux)
        peaks, properties = scipy_signal.find_peaks(
            flux, height=threshold, distance=int(0.1 * self.sr / hop_length)
        )

        times = librosa.frames_to_time(peaks + 1, sr=self.sr, hop_length=hop_length)

        detections = []
        for i, (peak, t) in enumerate(zip(peaks, times)):
            detections.append({
                "time": float(t),
                "sample": int(peak * hop_length),
                "confidence": float(
                    min(1.0, (flux[peak] - threshold) / (np.std(flux) + 1e-10))
                ),
                "type": "spectral_discontinuity",
            })
        return detections

    def detect_phase_discontinuities(
        self, sensitivity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect phase jumps in the STFT."""
        n_fft = DEFAULT_N_FFT
        hop_length = DEFAULT_HOP_LENGTH

        S = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length)
        phase = np.angle(S)

        # Instantaneous frequency deviation
        phase_diff = np.diff(phase, axis=1)
        # Wrap to [-pi, pi]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        # Expected phase advance
        expected = 2 * np.pi * np.arange(n_fft // 2 + 1)[:, None] * hop_length / n_fft
        deviation = np.abs(phase_diff - expected[:, :phase_diff.shape[1]])
        mean_deviation = np.mean(deviation, axis=0)

        threshold = np.mean(mean_deviation) + (1.5 - sensitivity) * np.std(mean_deviation)
        peaks, _ = scipy_signal.find_peaks(
            mean_deviation, height=threshold,
            distance=int(0.1 * self.sr / hop_length),
        )
        times = librosa.frames_to_time(peaks + 1, sr=self.sr, hop_length=hop_length)

        detections = []
        for peak, t in zip(peaks, times):
            detections.append({
                "time": float(t),
                "sample": int(peak * hop_length),
                "confidence": float(
                    min(1.0, (mean_deviation[peak] - threshold) /
                        (np.std(mean_deviation) + 1e-10))
                ),
                "type": "phase_discontinuity",
            })
        return detections

    def detect_energy_anomalies(
        self, sensitivity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect sudden energy changes in the signal."""
        hop_length = DEFAULT_HOP_LENGTH
        rms = librosa.feature.rms(
            y=self.y, frame_length=DEFAULT_N_FFT, hop_length=hop_length
        )[0]

        # Compute the derivative of RMS
        rms_diff = np.abs(np.diff(rms))
        threshold = np.mean(rms_diff) + (2.0 - sensitivity * 1.5) * np.std(rms_diff)

        peaks, _ = scipy_signal.find_peaks(
            rms_diff, height=threshold,
            distance=int(0.1 * self.sr / hop_length),
        )
        times = librosa.frames_to_time(peaks + 1, sr=self.sr, hop_length=hop_length)

        detections = []
        for peak, t in zip(peaks, times):
            detections.append({
                "time": float(t),
                "sample": int(peak * hop_length),
                "confidence": float(
                    min(1.0, (rms_diff[peak] - threshold) / (np.std(rms_diff) + 1e-10))
                ),
                "type": "energy_anomaly",
            })
        return detections

    def detect_statistical_changepoints(
        self, sensitivity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect change-points using CUSUM algorithm on short-time features."""
        hop_length = DEFAULT_HOP_LENGTH
        frame_length = DEFAULT_N_FFT

        # Use RMS as the feature for CUSUM
        rms = librosa.feature.rms(
            y=self.y, frame_length=frame_length, hop_length=hop_length
        )[0]

        # CUSUM algorithm
        mean_rms = np.mean(rms)
        cusum_pos = np.zeros(len(rms))
        cusum_neg = np.zeros(len(rms))
        threshold_h = (1.5 - sensitivity) * np.std(rms) * 5

        for i in range(1, len(rms)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + rms[i] - mean_rms - np.std(rms) * 0.5)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - rms[i] + mean_rms - np.std(rms) * 0.5)

        # Find where CUSUM exceeds threshold
        change_points = []
        in_alarm = False
        for i in range(len(rms)):
            if (cusum_pos[i] > threshold_h or cusum_neg[i] > threshold_h) and not in_alarm:
                t = librosa.frames_to_time(i, sr=self.sr, hop_length=hop_length)
                change_points.append({
                    "time": float(t),
                    "sample": int(i * hop_length),
                    "confidence": float(
                        min(1.0, max(cusum_pos[i], cusum_neg[i]) / (threshold_h + 1e-10) - 1)
                    ),
                    "type": "statistical_changepoint",
                })
                in_alarm = True
            elif cusum_pos[i] <= threshold_h and cusum_neg[i] <= threshold_h:
                in_alarm = False

        return change_points

    def _merge_nearby(
        self, points: List[Dict[str, Any]], threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Merge detection points that are within threshold seconds of each other."""
        if not points:
            return []

        merged = [points[0].copy()]
        merged[0]["methods"] = [merged[0].get("method", "unknown")]

        for p in points[1:]:
            if abs(p["time"] - merged[-1]["time"]) <= threshold:
                # Merge: keep higher confidence, combine methods
                if p["confidence"] > merged[-1]["confidence"]:
                    t = merged[-1]["time"]
                    methods = merged[-1]["methods"]
                    merged[-1] = p.copy()
                    merged[-1]["time"] = t
                    merged[-1]["methods"] = methods
                if p.get("method") not in merged[-1]["methods"]:
                    merged[-1]["methods"].append(p.get("method", "unknown"))
            else:
                new_point = p.copy()
                new_point["methods"] = [p.get("method", "unknown")]
                merged.append(new_point)

        # Update confidence based on number of methods that detected the point
        for p in merged:
            method_boost = len(p["methods"]) / 4.0
            p["confidence"] = min(1.0, p["confidence"] + method_boost * 0.2)

        return merged
