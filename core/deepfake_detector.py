"""Deepfake audio detection module.

Detects synthetic/manipulated speech using spectral artifacts,
temporal consistency analysis, and optional AI model inference.
"""

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import librosa
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from utils.constants import DEFAULT_N_FFT, DEFAULT_HOP_LENGTH
from core.model_manager import get_model_manager


class DeepfakeDetector:
    """Detect deepfake/synthetic audio using multi-modal analysis.

    Detection techniques:
    1. Spectral artifact detection (vocoder fingerprints)
    2. Temporal consistency analysis
    3. Naturalness metrics (prosody, formant transitions)
    4. Statistical distribution analysis
    5. AI model-based detection (when available)
    """

    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr
        self._manager = get_model_manager()

    def detect(self) -> Dict[str, Any]:
        """Run full deepfake detection pipeline.

        Returns:
            Dictionary with detection results, scores, and explanations.
        """
        # Run all detection methods
        spectral = self._analyze_spectral_artifacts()
        temporal = self._analyze_temporal_consistency()
        naturalness = self._analyze_naturalness()
        statistical = self._analyze_statistical_distribution()

        # Speaker embedding consistency (uses Wespeaker model)
        embedding_consistency = self._analyze_embedding_consistency()

        # Try AI model
        model_result = self._run_model_detection()

        # Combine scores
        scores = {
            "频谱伪影": spectral["score"],
            "时序一致性": temporal["score"],
            "自然度": naturalness["score"],
            "统计分布": statistical["score"],
        }

        if embedding_consistency is not None:
            scores["嵌入一致性"] = embedding_consistency["score"]

        if model_result is not None:
            scores["AI模型"] = model_result["score"]

        # Weighted average
        if model_result is not None:
            weights = {"频谱伪影": 0.12, "时序一致性": 0.12,
                       "自然度": 0.12, "统计分布": 0.12,
                       "嵌入一致性": 0.12, "AI模型": 0.4}
        elif embedding_consistency is not None:
            weights = {"频谱伪影": 0.22, "时序一致性": 0.22,
                       "自然度": 0.22, "统计分布": 0.14,
                       "嵌入一致性": 0.20}
        else:
            weights = {"频谱伪影": 0.3, "时序一致性": 0.25,
                       "自然度": 0.25, "统计分布": 0.2}

        final_score = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights if k in scores)

        # Determine verdict
        if final_score > 0.7:
            verdict = "高度可疑"
            verdict_desc = "检测到多项深度伪造特征，该音频极有可能为合成或篡改音频。"
            risk_level = "高"
        elif final_score > 0.5:
            verdict = "可疑"
            verdict_desc = "检测到部分异常特征，该音频可能经过合成或编辑处理。建议进一步分析。"
            risk_level = "中"
        elif final_score > 0.3:
            verdict = "低风险"
            verdict_desc = "检测到少量异常，但不足以判定为深度伪造。可能为自然语音变化。"
            risk_level = "低"
        else:
            verdict = "正常"
            verdict_desc = "未检测到明显的深度伪造特征，音频特征符合自然语音规律。"
            risk_level = "极低"

        return {
            "final_score": float(final_score),
            "verdict": verdict,
            "verdict_description": verdict_desc,
            "risk_level": risk_level,
            "scores": {k: float(v) for k, v in scores.items()},
            "details": {
                "spectral_artifacts": spectral,
                "temporal_consistency": temporal,
                "naturalness": naturalness,
                "statistical": statistical,
                "embedding_consistency": embedding_consistency,
                "ai_model": model_result,
            },
            "model_used": model_result is not None,
            "embedding_model_used": embedding_consistency is not None,
        }

    def _analyze_spectral_artifacts(self) -> Dict[str, Any]:
        """Detect spectral artifacts typical of vocoders and synthesis."""
        n_fft = DEFAULT_N_FFT
        hop = DEFAULT_HOP_LENGTH

        S = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=hop))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        indicators = []
        score = 0.0

        # 1. Check for unnaturally sharp frequency cutoff (vocoder artifact)
        freq_profile = np.mean(S_db, axis=1)
        high_freq_idx = len(freq_profile) * 3 // 4
        high_freq_energy = np.mean(freq_profile[high_freq_idx:])
        mid_freq_energy = np.mean(freq_profile[len(freq_profile) // 4:high_freq_idx])

        cutoff_sharpness = mid_freq_energy - high_freq_energy
        if cutoff_sharpness > 40:
            score += 0.3
            indicators.append("检测到异常尖锐的高频截断（可能为声码器处理痕迹）")
        elif cutoff_sharpness > 30:
            score += 0.15
            indicators.append("高频衰减较陡峭")

        # 2. Check for periodic spectral patterns (synthesis fingerprint)
        spectral_var = np.var(S_db, axis=1)
        periodicity = self._detect_spectral_periodicity(spectral_var)
        if periodicity > 0.7:
            score += 0.3
            indicators.append("频谱中检测到周期性模式（合成语音特征）")
        elif periodicity > 0.4:
            score += 0.15
            indicators.append("频谱有轻微周期性倾向")

        # 3. Check for missing or artificial harmonics
        harmonic_regularity = self._check_harmonic_structure(S)
        if harmonic_regularity > 0.8:
            score += 0.2
            indicators.append("谐波结构过于规则（自然语音通常有更多变化）")

        # 4. Spectral flatness variance (synthetic speech tends to be more uniform)
        flatness = librosa.feature.spectral_flatness(y=self.y, n_fft=n_fft, hop_length=hop)[0]
        flatness_var = np.var(flatness)
        if flatness_var < 0.005:
            score += 0.2
            indicators.append("频谱平坦度变化过小（可能为合成信号）")

        if not indicators:
            indicators.append("未检测到明显的频谱伪造痕迹")

        return {
            "score": min(1.0, score),
            "indicators": indicators,
            "cutoff_sharpness": float(cutoff_sharpness),
            "spectral_periodicity": float(periodicity),
        }

    def _analyze_temporal_consistency(self) -> Dict[str, Any]:
        """Analyze temporal consistency of speech features."""
        hop = DEFAULT_HOP_LENGTH
        indicators = []
        score = 0.0

        # 1. Pitch continuity
        f0, voiced, _ = librosa.pyin(self.y, fmin=50, fmax=500, sr=self.sr, hop_length=hop)
        if np.any(voiced):
            f0_valid = f0[voiced]
            f0_diff = np.abs(np.diff(f0_valid))

            # Unnatural pitch jumps
            large_jumps = np.sum(f0_diff > 50) / (len(f0_diff) + 1)
            if large_jumps > 0.1:
                score += 0.25
                indicators.append(f"检测到异常频繁的基频跳变 ({large_jumps * 100:.1f}%)")

            # Pitch too stable (synthetic often lacks micro-variations)
            f0_micro_var = np.std(f0_diff[f0_diff < 10])
            if f0_micro_var < 0.5 and len(f0_valid) > 20:
                score += 0.2
                indicators.append("基频微变化过小（缺乏自然语音的微抖动）")

        # 2. Energy envelope smoothness
        rms = librosa.feature.rms(y=self.y, hop_length=hop)[0]
        rms_diff = np.abs(np.diff(rms))
        envelope_smoothness = np.mean(rms_diff) / (np.std(rms) + 1e-10)

        if envelope_smoothness < 0.05:
            score += 0.2
            indicators.append("能量包络过于平滑（可能经过合成处理）")
        elif envelope_smoothness > 0.5:
            score += 0.15
            indicators.append("能量包络存在异常波动")

        # 3. Formant transition naturalness
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13, hop_length=hop)
        mfcc_delta = librosa.feature.delta(mfcc)
        transition_energy = np.mean(np.abs(mfcc_delta))

        if transition_energy < 5:
            score += 0.2
            indicators.append("共振峰过渡过于平缓（合成语音特征）")

        # 4. Phase consistency
        S_complex = librosa.stft(self.y, n_fft=DEFAULT_N_FFT, hop_length=hop)
        phase = np.angle(S_complex)
        phase_consistency = np.mean(np.abs(np.diff(phase, axis=1)))
        if phase_consistency < 0.5:
            score += 0.15
            indicators.append("相位连贯性异常高（可能为合成信号）")

        if not indicators:
            indicators.append("时间特征连贯性正常")

        return {
            "score": min(1.0, score),
            "indicators": indicators,
        }

    def _analyze_naturalness(self) -> Dict[str, Any]:
        """Analyze speech naturalness metrics."""
        indicators = []
        score = 0.0

        # 1. Breathing pattern detection
        rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        rms_db = 20 * np.log10(rms + 1e-10)

        # Natural speech has breath pauses
        silence_threshold = np.percentile(rms_db, 15)
        silence_frames = rms_db < silence_threshold
        total_dur = len(self.y) / self.sr

        # Check for breath-like patterns in silence regions
        silence_ratio = np.sum(silence_frames) / len(silence_frames)
        if silence_ratio < 0.05 and total_dur > 5:
            score += 0.25
            indicators.append("缺乏自然呼吸停顿（合成语音常见特征）")

        # 2. Lip smack / filler detection (natural speech markers)
        zcr = librosa.feature.zero_crossing_rate(self.y, frame_length=2048, hop_length=512)[0]
        high_zcr_ratio = np.sum(zcr > 0.3) / len(zcr)
        if high_zcr_ratio < 0.01 and total_dur > 5:
            score += 0.15
            indicators.append("缺少唇齿摩擦音等自然语音标记")

        # 3. Prosody variability
        f0, voiced, _ = librosa.pyin(self.y, fmin=50, fmax=500, sr=self.sr)
        if np.any(voiced):
            f0_valid = f0[voiced]
            if len(f0_valid) > 10:
                f0_range = np.percentile(f0_valid, 95) - np.percentile(f0_valid, 5)
                if f0_range < 20:
                    score += 0.2
                    indicators.append(f"基频变化范围过窄 ({f0_range:.1f} Hz)，语调过于单一")

        # 4. Spectral tilt naturalness
        freqs, psd = scipy_signal.welch(self.y, fs=self.sr, nperseg=2048)
        valid = freqs > 100
        if np.any(valid):
            slope = np.polyfit(np.log10(freqs[valid]), 10 * np.log10(psd[valid] + 1e-10), 1)[0]
            if slope > -2:
                score += 0.2
                indicators.append("频谱倾斜度异常（缺乏自然语音的频谱衰减特征）")

        if not indicators:
            indicators.append("语音自然度指标正常")

        return {
            "score": min(1.0, score),
            "indicators": indicators,
        }

    def _analyze_statistical_distribution(self) -> Dict[str, Any]:
        """Analyze statistical distribution of audio features."""
        indicators = []
        score = 0.0

        # 1. Sample distribution
        kurt = kurtosis(self.y)
        skewness = skew(self.y)

        if abs(kurt) < 1:
            score += 0.2
            indicators.append(f"信号峰度偏低 ({kurt:.2f})，分布过于均匀")

        if abs(skewness) > 0.5:
            score += 0.1
            indicators.append(f"信号偏度异常 ({skewness:.3f})")

        # 2. MFCC distribution
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        mfcc_kurt = np.mean([kurtosis(mfcc[i]) for i in range(13)])

        if mfcc_kurt < 0:
            score += 0.2
            indicators.append("MFCC 分布呈轻尾特征（可能为合成语音）")

        # 3. Correlation between adjacent frames
        frame_corr = []
        mfcc_t = mfcc.T
        for i in range(min(100, len(mfcc_t) - 1)):
            c = np.corrcoef(mfcc_t[i], mfcc_t[i + 1])[0, 1]
            if not np.isnan(c):
                frame_corr.append(c)

        if frame_corr:
            mean_corr = np.mean(frame_corr)
            if mean_corr > 0.98:
                score += 0.25
                indicators.append(f"帧间相关性过高 ({mean_corr:.4f})，可能为合成产物")
            elif mean_corr < 0.7:
                score += 0.15
                indicators.append(f"帧间相关性异常低 ({mean_corr:.4f})")

        # 4. Bit-depth consistency
        unique_levels = len(np.unique(np.round(self.y * 32768).astype(int)))
        expected = min(65536, len(self.y))
        level_ratio = unique_levels / expected

        if level_ratio < 0.01:
            score += 0.2
            indicators.append("量化级别不足，可能为低精度合成")

        if not indicators:
            indicators.append("统计分布特征正常")

        return {
            "score": min(1.0, score),
            "indicators": indicators,
            "kurtosis": float(kurt),
            "skewness": float(skewness),
        }

    def _analyze_embedding_consistency(self) -> Optional[Dict[str, Any]]:
        """Use Wespeaker embedding model to detect inconsistencies.
        
        Real speech has consistent speaker characteristics across segments.
        Deepfakes often show unusual embedding variance or unnatural patterns.
        """
        try:
            from core.model_integration import SpeakerEmbedder
            embedder = SpeakerEmbedder()
        except Exception:
            return None
        
        indicators = []
        score = 0.0
        
        # Split audio into segments and compare embeddings
        segment_dur = 2.0  # seconds
        segment_samples = int(segment_dur * self.sr)
        
        if len(self.y) < segment_samples * 2:
            return None
        
        embeddings = []
        for start in range(0, len(self.y) - segment_samples, segment_samples):
            segment = self.y[start:start + segment_samples]
            if np.max(np.abs(segment)) < 0.01:
                continue
            try:
                emb = embedder.extract_embedding(segment, self.sr)
                embeddings.append(emb)
            except Exception:
                continue
        
        if len(embeddings) < 3:
            return None
        
        embeddings = np.array(embeddings)
        
        # 1. Embedding variance analysis
        # Real speech: consistent but with natural variation
        # Deepfake: either too consistent (copy-paste) or inconsistent (model artifacts)
        pairwise_sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                pairwise_sims.append(sim)
        
        mean_sim = np.mean(pairwise_sims)
        std_sim = np.std(pairwise_sims)
        
        # Too consistent (all segments nearly identical) → suspicious
        if mean_sim > 0.98 and std_sim < 0.005:
            score += 0.4
            indicators.append(f"嵌入向量过于一致 (均值:{mean_sim:.3f}, 标准差:{std_sim:.4f})，可能为合成拼接")
        elif mean_sim > 0.95 and std_sim < 0.01:
            score += 0.2
            indicators.append(f"嵌入向量一致性偏高 (均值:{mean_sim:.3f})")
        
        # Too inconsistent → suspicious (unless multiple speakers)
        if std_sim > 0.15 and mean_sim < 0.6:
            score += 0.3
            indicators.append(f"嵌入向量不一致 (均值:{mean_sim:.3f}, 标准差:{std_sim:.3f})，可能有拼接痕迹")
        
        # 2. Embedding dimension distribution
        # Deepfakes often have unnatural distribution in embedding space
        dim_kurtosis = np.mean([float(kurtosis(embeddings[:, d])) for d in range(min(32, embeddings.shape[1]))])
        if abs(dim_kurtosis) > 5:
            score += 0.2
            indicators.append(f"嵌入分布异常 (峰度:{dim_kurtosis:.2f})")
        
        # 3. Temporal embedding drift
        # Real speech drifts naturally; deepfakes may have abrupt changes
        if len(embeddings) >= 4:
            diffs = [float(np.linalg.norm(embeddings[i+1] - embeddings[i])) 
                     for i in range(len(embeddings) - 1)]
            diff_std = np.std(diffs)
            diff_max = np.max(diffs)
            diff_mean = np.mean(diffs)
            
            if diff_max > 3 * diff_mean and diff_std > 0.1:
                score += 0.2
                indicators.append(f"嵌入时序跳变异常 (最大跳变: {diff_max:.3f}, 均值: {diff_mean:.3f})")
        
        score = min(1.0, score)
        
        if not indicators:
            indicators.append(f"嵌入一致性正常 (相似度均值:{mean_sim:.3f}, 标准差:{std_sim:.3f})")
        
        return {
            "score": score,
            "indicators": indicators,
            "mean_similarity": float(mean_sim),
            "std_similarity": float(std_sim),
            "num_segments": len(embeddings),
            "model": "Wespeaker ResNet34",
        }

    def _run_model_detection(self) -> Optional[Dict[str, Any]]:
        """Run AI model-based deepfake detection if available."""
        # Try AASIST anti-spoofing model first
        try:
            import onnxruntime as ort
            from pathlib import Path
            import os
            
            candidates = [
                Path(__file__).parent.parent / "models" / "aasist" / "aasist_antispoof.onnx",
                Path(os.getcwd()) / "models" / "aasist" / "aasist_antispoof.onnx",
            ]
            
            model_path = None
            for p in candidates:
                if p.exists():
                    model_path = str(p)
                    break
            
            if model_path:
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                
                if self.sr != 16000:
                    y_16k = librosa.resample(self.y, orig_sr=self.sr, target_sr=16000)
                else:
                    y_16k = self.y.copy()
                
                if np.max(np.abs(y_16k)) > 0:
                    y_16k = y_16k / np.max(np.abs(y_16k))
                
                target_len = 64000  # 4 seconds
                if len(y_16k) > target_len:
                    y_16k = y_16k[:target_len]
                else:
                    y_16k = np.pad(y_16k, (0, target_len - len(y_16k)))
                
                audio_input = y_16k.reshape(1, 1, -1).astype(np.float32)
                result = session.run(None, {"audio": audio_input})
                spoof_prob = float(result[0].flatten()[0])
                
                return {
                    "score": spoof_prob,
                    "model_name": "AASIST Anti-Spoofing",
                    "indicators": [f"AASIST 模型伪造概率: {spoof_prob:.3f}"],
                }
        except Exception:
            pass
        
        # Fallback: try model_manager generic approach
        if not self._manager.has_onnx_runtime:
            return None

        session = self._manager.load_onnx_model("deepfake_detector")
        if session is None:
            return None

        try:
            if self.sr != 16000:
                y_16k = librosa.resample(self.y, orig_sr=self.sr, target_sr=16000)
            else:
                y_16k = self.y

            max_len = 16000 * 10
            if len(y_16k) > max_len:
                y_16k = y_16k[:max_len]
            else:
                y_16k = np.pad(y_16k, (0, max_len - len(y_16k)))

            inputs = {"input": y_16k.reshape(1, -1).astype(np.float32)}
            results = self._manager.run_inference("deepfake_detector", inputs)

            if results:
                output = list(results.values())[0].flatten()
                deepfake_prob = float(output[0]) if len(output) > 0 else 0.5
                return {
                    "score": deepfake_prob,
                    "model_name": "deepfake_detector",
                    "indicators": [f"AI 模型检测评分: {deepfake_prob:.3f}"],
                }
        except Exception:
            pass
        return None

    def _detect_spectral_periodicity(self, spectral_var: np.ndarray) -> float:
        """Detect periodic patterns in spectral variance."""
        if len(spectral_var) < 20:
            return 0.0
        # Autocorrelation
        norm = spectral_var - np.mean(spectral_var)
        autocorr = np.correlate(norm, norm, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Find peaks in autocorrelation (exclude lag 0)
        if len(autocorr) > 5:
            peaks = autocorr[2:len(autocorr) // 2]
            if len(peaks) > 0:
                return float(np.max(peaks))
        return 0.0

    def _check_harmonic_structure(self, S: np.ndarray) -> float:
        """Check if harmonic structure is unnaturally regular."""
        # Average spectrum across time
        avg_spectrum = np.mean(S, axis=1)
        if len(avg_spectrum) < 20:
            return 0.0

        # Find peaks (harmonics)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(avg_spectrum, height=np.mean(avg_spectrum))

        if len(peaks) < 3:
            return 0.0

        # Check regularity of peak spacing
        spacings = np.diff(peaks)
        if len(spacings) < 2:
            return 0.0

        regularity = 1.0 - (np.std(spacings) / (np.mean(spacings) + 1e-10))
        return float(max(0, regularity))
