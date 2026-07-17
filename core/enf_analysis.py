"""Electric Network Frequency (ENF) analysis module."""

from typing import Tuple, Optional, Dict, Any

import numpy as np
from scipy import signal as scipy_signal

from utils.constants import ENF_FREQUENCIES


class ENFAnalyzer:
    """Detect and analyze Electric Network Frequency (ENF) in audio recordings.

    ENF analysis exploits the fact that mains hum (50/60 Hz) captured by
    microphones varies slightly over time.  Comparing these fluctuations
    against a reference grid-frequency database can authenticate recordings
    and establish time-of-recording.
    """

    def __init__(self, y: np.ndarray, sr: int, nominal_freq: float = 50.0):
        self.y = y
        self.sr = sr
        self.nominal_freq = nominal_freq

    def extract_enf(
        self,
        band_width: float = 0.5,
        frame_duration: float = 1.0,
        overlap: float = 0.5,
    ) -> Dict[str, Any]:
        """Extract ENF trace from the audio signal.

        Args:
            band_width: Bandwidth around nominal frequency for bandpass filter (Hz).
            frame_duration: Analysis frame duration in seconds.
            overlap: Frame overlap ratio.

        Returns:
            Dictionary with ENF trace, times, and statistics.
        """
        # Design bandpass filter around nominal frequency
        low = self.nominal_freq - band_width
        high = self.nominal_freq + band_width
        filtered = self._bandpass_filter(low, high)

        # Frame the filtered signal
        frame_len = int(frame_duration * self.sr)
        hop = int(frame_len * (1 - overlap))
        n_frames = max(1, (len(filtered) - frame_len) // hop + 1)

        enf_trace = np.zeros(n_frames)
        times = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop
            end = start + frame_len
            frame = filtered[start:end]
            times[i] = (start + frame_len / 2) / self.sr

            # Estimate instantaneous frequency via zero-crossing
            enf_trace[i] = self._estimate_frequency(frame)

        # Compute statistics
        valid_mask = (enf_trace > self.nominal_freq - 2) & (enf_trace < self.nominal_freq + 2)
        valid_enf = enf_trace[valid_mask] if np.any(valid_mask) else enf_trace

        return {
            "enf_trace": enf_trace,
            "times": times,
            "valid_mask": valid_mask,
            "mean_freq": float(np.mean(valid_enf)),
            "std_freq": float(np.std(valid_enf)),
            "min_freq": float(np.min(valid_enf)),
            "max_freq": float(np.max(valid_enf)),
            "nominal_freq": self.nominal_freq,
            "snr_db": self._estimate_enf_snr(filtered),
        }

    def _bandpass_filter(
        self, low: float, high: float, order: int = 8
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyq = self.sr / 2.0
        b, a = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band")
        return scipy_signal.filtfilt(b, a, self.y)

    def _estimate_frequency(self, frame: np.ndarray) -> float:
        """Estimate frequency of a quasi-sinusoidal frame via zero-crossing."""
        # Use high-resolution FFT for frequency estimation
        n_fft = max(len(frame), 8192)
        windowed = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        # Search around nominal frequency
        search_mask = (freqs >= self.nominal_freq - 2) & (freqs <= self.nominal_freq + 2)
        if not np.any(search_mask):
            return self.nominal_freq

        search_freqs = freqs[search_mask]
        search_spectrum = spectrum[search_mask]
        peak_idx = np.argmax(search_spectrum)

        return float(search_freqs[peak_idx])

    def _estimate_enf_snr(self, filtered: np.ndarray) -> float:
        """Estimate SNR of the ENF component."""
        n_fft = min(len(filtered), 65536)
        spectrum = np.abs(np.fft.rfft(filtered, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        # Signal power: around nominal frequency
        sig_mask = (freqs >= self.nominal_freq - 0.5) & (freqs <= self.nominal_freq + 0.5)
        noise_mask = ~sig_mask & (freqs > 0)

        if not np.any(sig_mask) or not np.any(noise_mask):
            return 0.0

        sig_power = np.mean(spectrum[sig_mask] ** 2)
        noise_power = np.mean(spectrum[noise_mask] ** 2)

        if noise_power < 1e-20:
            return 60.0
        return float(10 * np.log10(sig_power / noise_power))

    def detect_enf_harmonics(
        self, n_harmonics: int = 5
    ) -> Dict[str, Any]:
        """Detect ENF harmonics in the audio spectrum.

        Returns:
            Dictionary with harmonic frequencies and their strengths.
        """
        n_fft = min(len(self.y), 65536)
        spectrum = np.abs(np.fft.rfft(self.y, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        harmonics = {}
        for h in range(1, n_harmonics + 1):
            target = self.nominal_freq * h
            if target >= self.sr / 2:
                break
            mask = (freqs >= target - 1) & (freqs <= target + 1)
            if np.any(mask):
                peak_idx = np.argmax(spectrum[mask])
                harmonics[f"H{h} ({target:.0f}Hz)"] = {
                    "detected_freq": float(freqs[mask][peak_idx]),
                    "magnitude_db": float(20 * np.log10(spectrum[mask][peak_idx] + 1e-10)),
                }

        return harmonics
