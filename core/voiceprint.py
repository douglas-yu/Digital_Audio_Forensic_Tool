"""Voiceprint (speaker) analysis module.

Performs speaker embedding extraction, comparison, and clustering
for speaker identification and verification in forensic contexts.
"""

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import librosa
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from utils.constants import DEFAULT_SR, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH
from core.model_manager import get_model_manager


class VoiceprintAnalyzer:
    """Analyze speaker characteristics and voiceprint features.

    Methods:
    - MFCC-based speaker embeddings (no model required)
    - ONNX model-based embeddings (if model available)
    - Speaker similarity comparison
    - Multi-speaker clustering/diarization
    """

    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr
        self._manager = get_model_manager()

    def extract_voiceprint(self, use_model: bool = True) -> Dict[str, Any]:
        """Extract voiceprint embedding from audio.

        Args:
            use_model: Try to use ONNX model first, fallback to MFCC.

        Returns:
            Dictionary with embedding vector and feature statistics.
        """
        # Try ONNX model first
        if use_model and self._manager.has_onnx_runtime:
            model_result = self._extract_with_model()
            if model_result is not None:
                return model_result

        # Fallback: MFCC-based voiceprint
        return self._extract_mfcc_voiceprint()

    def _extract_mfcc_voiceprint(self) -> Dict[str, Any]:
        """Extract MFCC-based voiceprint embedding."""
        # Compute extended MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=20,
                                      n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH)
        # Delta and delta-delta
        delta_mfcc = librosa.feature.delta(mfccs)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)

        # Statistics over time for each coefficient
        features = []
        for feat_set in [mfccs, delta_mfcc, delta2_mfcc]:
            features.extend([
                np.mean(feat_set, axis=1),
                np.std(feat_set, axis=1),
                np.min(feat_set, axis=1),
                np.max(feat_set, axis=1),
            ])

        embedding = np.concatenate(features)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        # Spectral features for voice quality
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=self.y, sr=self.sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=self.y, sr=self.sr))

        # Pitch statistics
        f0, voiced, _ = librosa.pyin(self.y, fmin=50, fmax=500, sr=self.sr)
        f0_valid = f0[voiced] if np.any(voiced) else np.array([0])

        return {
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "method": "MFCC统计特征",
            "model_used": False,
            "features": {
                "基频均值 (Hz)": float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0,
                "基频标准差 (Hz)": float(np.std(f0_valid)) if len(f0_valid) > 0 else 0,
                "基频范围 (Hz)": f"{float(np.min(f0_valid)):.1f} - {float(np.max(f0_valid)):.1f}",
                "频谱质心 (Hz)": float(spectral_centroid),
                "频谱带宽 (Hz)": float(spectral_bandwidth),
                "频谱滚降点 (Hz)": float(spectral_rolloff),
                "发声比例": float(np.sum(voiced) / len(voiced)) if len(voiced) > 0 else 0,
            },
        }

    def _extract_with_model(self) -> Optional[Dict[str, Any]]:
        """Try to extract embedding using ONNX model."""
        session = self._manager.load_onnx_model("speaker_embedding")
        if session is None:
            return None

        try:
            # Resample to 16kHz if needed (common for speaker models)
            if self.sr != 16000:
                y_16k = librosa.resample(self.y, orig_sr=self.sr, target_sr=16000)
            else:
                y_16k = self.y

            inputs = {"input": y_16k.reshape(1, -1).astype(np.float32)}
            results = self._manager.run_inference("speaker_embedding", inputs)
            if results:
                embedding = list(results.values())[0].flatten()
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                return {
                    "embedding": embedding,
                    "embedding_dim": len(embedding),
                    "method": "AI模型嵌入",
                    "model_used": True,
                    "features": {},
                }
        except Exception:
            pass
        return None

    def compare_voiceprints(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> Dict[str, float]:
        """Compare two voiceprint embeddings.

        Returns:
            Dictionary with similarity scores.
        """
        cos_sim = 1.0 - cosine(embedding1, embedding2)
        euc_dist = euclidean(embedding1, embedding2)
        # Normalize euclidean to 0-1 similarity
        euc_sim = 1.0 / (1.0 + euc_dist)

        # Combined score
        combined = 0.7 * cos_sim + 0.3 * euc_sim

        return {
            "cosine_similarity": float(cos_sim),
            "euclidean_similarity": float(euc_sim),
            "combined_score": float(combined),
            "same_speaker_likely": combined > 0.75,
            "confidence": "高" if combined > 0.85 else "中" if combined > 0.7 else "低",
        }

    def segment_speakers(
        self, segment_duration: float = 3.0, n_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Segment audio by speaker (simple diarization).

        Args:
            segment_duration: Duration of each analysis segment in seconds.
            n_speakers: Expected number of speakers (None for auto-detect).

        Returns:
            Dictionary with speaker segments and statistics.
        """
        seg_samples = int(segment_duration * self.sr)
        n_segments = max(1, len(self.y) // seg_samples)

        # Extract embedding for each segment
        embeddings = []
        times = []
        for i in range(n_segments):
            start = i * seg_samples
            end = min(start + seg_samples, len(self.y))
            segment = self.y[start:end]

            if len(segment) < self.sr:  # skip very short segments
                continue

            mfccs = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=20)
            emb = np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
            embeddings.append(emb)
            times.append((start / self.sr, end / self.sr))

        if len(embeddings) < 2:
            return {
                "n_speakers": 1,
                "segments": [{"start": 0, "end": len(self.y) / self.sr, "speaker": 0}],
                "method": "单段落分析",
            }

        embeddings = np.array(embeddings)
        embeddings = normalize(embeddings)

        # Determine number of speakers
        if n_speakers is None:
            # Use silhouette score to find optimal clusters
            best_n = 2
            best_score = -1
            from sklearn.metrics import silhouette_score
            for n in range(2, min(6, len(embeddings))):
                try:
                    clustering = AgglomerativeClustering(n_clusters=n)
                    labels = clustering.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                except Exception:
                    break
            n_speakers = best_n

        # Final clustering
        clustering = AgglomerativeClustering(n_clusters=min(n_speakers, len(embeddings)))
        labels = clustering.fit_predict(embeddings)

        segments = []
        for i, (time_range, label) in enumerate(zip(times, labels)):
            segments.append({
                "start": time_range[0],
                "end": time_range[1],
                "speaker": int(label),
            })

        # Speaker statistics
        speaker_stats = {}
        for spk in range(n_speakers):
            spk_segs = [s for s in segments if s["speaker"] == spk]
            total_dur = sum(s["end"] - s["start"] for s in spk_segs)
            speaker_stats[f"说话人 {spk + 1}"] = {
                "段落数": len(spk_segs),
                "总时长 (秒)": round(total_dur, 2),
                "占比": f"{total_dur / (len(self.y) / self.sr) * 100:.1f}%",
            }

        return {
            "n_speakers": n_speakers,
            "segments": segments,
            "speaker_stats": speaker_stats,
            "method": "层次聚类 (MFCC)",
        }
