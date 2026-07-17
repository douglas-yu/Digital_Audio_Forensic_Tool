"""Audio analysis algorithms for Audio Forensics Tool."""

from typing import Dict, Any, Tuple, Optional

import numpy as np
import librosa
from scipy import signal as scipy_signal

from utils.constants import DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_N_MELS


class AudioAnalyzer:
    """Core audio analysis engine providing spectral, temporal, and statistical analysis."""

    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr

    # ── Spectral Analysis ──────────────────────────────────────────

    def compute_spectrogram(
        self, n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute STFT magnitude spectrogram.

        Returns:
            (S_db, freqs, times) - Spectrogram in dB, frequency bins, time frames.
        """
        S = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length
        )
        return S_db, freqs, times

    def compute_mel_spectrogram(
        self,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mel spectrogram.

        Returns:
            (mel_db, times) - Mel spectrogram in dB, time frames.
        """
        mel = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        times = librosa.frames_to_time(
            np.arange(mel.shape[1]), sr=self.sr, hop_length=hop_length
        )
        return mel_db, times

    def compute_mfcc(
        self, n_mfcc: int = 13, n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MFCCs.

        Returns:
            (mfccs, times) - MFCC coefficients, time frames.
        """
        mfccs = librosa.feature.mfcc(
            y=self.y, sr=self.sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length,
        )
        times = librosa.frames_to_time(
            np.arange(mfccs.shape[1]), sr=self.sr, hop_length=hop_length
        )
        return mfccs, times

    # ── Temporal Analysis ──────────────────────────────────────────

    def compute_rms(
        self, frame_length: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RMS energy over time.

        Returns:
            (rms, times) - RMS values, time frames.
        """
        rms = librosa.feature.rms(
            y=self.y, frame_length=frame_length, hop_length=hop_length
        )[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=self.sr, hop_length=hop_length
        )
        return rms, times

    def compute_zero_crossing_rate(
        self, frame_length: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute zero-crossing rate over time.

        Returns:
            (zcr, times) - Zero-crossing rate values, time frames.
        """
        zcr = librosa.feature.zero_crossing_rate(
            self.y, frame_length=frame_length, hop_length=hop_length
        )[0]
        times = librosa.frames_to_time(
            np.arange(len(zcr)), sr=self.sr, hop_length=hop_length
        )
        return zcr, times

    def compute_spectral_centroid(
        self, n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spectral centroid over time.

        Returns:
            (centroid, times) - Spectral centroid values in Hz, time frames.
        """
        centroid = librosa.feature.spectral_centroid(
            y=self.y, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        times = librosa.frames_to_time(
            np.arange(len(centroid)), sr=self.sr, hop_length=hop_length
        )
        return centroid, times

    def compute_spectral_rolloff(
        self, n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH,
        roll_percent: float = 0.85,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spectral rolloff frequency over time."""
        rolloff = librosa.feature.spectral_rolloff(
            y=self.y, sr=self.sr, n_fft=n_fft,
            hop_length=hop_length, roll_percent=roll_percent,
        )[0]
        times = librosa.frames_to_time(
            np.arange(len(rolloff)), sr=self.sr, hop_length=hop_length
        )
        return rolloff, times

    # ── Statistical Analysis ───────────────────────────────────────

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute global audio statistics."""
        return {
            "peak_amplitude": float(np.max(np.abs(self.y))),
            "rms_level": float(np.sqrt(np.mean(self.y ** 2))),
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(self.y ** 2)) + 1e-10)),
            "dynamic_range_db": float(
                20 * np.log10((np.max(np.abs(self.y)) + 1e-10) /
                              (np.min(np.abs(self.y[self.y != 0])) + 1e-10))
            ) if np.any(self.y != 0) else 0.0,
            "mean": float(np.mean(self.y)),
            "std": float(np.std(self.y)),
            "crest_factor": float(
                np.max(np.abs(self.y)) / (np.sqrt(np.mean(self.y ** 2)) + 1e-10)
            ),
            "zero_crossings": int(np.sum(librosa.zero_crossings(self.y))),
            "dc_offset": float(np.mean(self.y)),
        }

    # ── Frequency Domain ───────────────────────────────────────────

    def compute_fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT magnitude spectrum.

        Returns:
            (freqs, magnitude) - Frequency bins, magnitude in dB.
        """
        n = len(self.y)
        fft_vals = np.fft.rfft(self.y)
        magnitude = np.abs(fft_vals)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr)
        return freqs, magnitude_db

    def compute_power_spectral_density(
        self, nperseg: int = DEFAULT_N_FFT
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density using Welch's method.

        Returns:
            (freqs, psd) - Frequency bins, PSD values.
        """
        freqs, psd = scipy_signal.welch(self.y, fs=self.sr, nperseg=nperseg)
        return freqs, psd

    # ── Pitch Detection ────────────────────────────────────────────

    def compute_pitch(
        self, fmin: float = 50.0, fmax: float = 2000.0,
        hop_length: int = DEFAULT_HOP_LENGTH,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate fundamental frequency (pitch) over time.

        Returns:
            (times, f0, voiced_flag) - Time stamps, F0 estimates, voiced/unvoiced flag.
        """
        f0, voiced_flag, _ = librosa.pyin(
            self.y, fmin=fmin, fmax=fmax,
            sr=self.sr, hop_length=hop_length,
        )
        times = librosa.frames_to_time(
            np.arange(len(f0)), sr=self.sr, hop_length=hop_length
        )
        return times, f0, voiced_flag
