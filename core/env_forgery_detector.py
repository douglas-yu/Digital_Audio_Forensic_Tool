"""Environmental sound forgery detection module.

Detects forged or synthesized non-speech audio content:
- Background noise consistency analysis
- Environmental acoustic fingerprint verification
- Ambient sound naturalness assessment
- Reverb/room impulse response consistency
- Inserted/spliced environmental sound detection
"""

import numpy as np
import librosa
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew, entropy
from typing import Dict, Any, List, Optional

from utils.constants import DEFAULT_N_FFT, DEFAULT_HOP_LENGTH


class EnvironmentalForgeryDetector:
    """Detect forged or synthesized environmental/non-speech audio.
    
    Analysis dimensions:
    1. Background noise consistency - checks for artificial noise floors
    2. Room acoustics analysis - reverb consistency, RT60 estimation
    3. Environmental fingerprint - spectral texture continuity
    4. Splice detection in ambient sounds - discontinuities in background
    5. Synthesized sound detection - patterns typical of sound generators
    """
    
    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr
    
    def detect(self) -> Dict[str, Any]:
        """Run full environmental forgery detection."""
        noise_consistency = self._analyze_noise_consistency()
        room_acoustics = self._analyze_room_acoustics()
        env_fingerprint = self._analyze_environmental_fingerprint()
        splice_detection = self._detect_ambient_splices()
        synth_detection = self._detect_synthesized_sounds()
        
        scores = {
            "噪底一致性": noise_consistency["score"],
            "室内声学": room_acoustics["score"],
            "环境指纹": env_fingerprint["score"],
            "拼接检测": splice_detection["score"],
            "合成声检测": synth_detection["score"],
        }
        
        weights = {
            "噪底一致性": 0.25,
            "室内声学": 0.20,
            "环境指纹": 0.20,
            "拼接检测": 0.20,
            "合成声检测": 0.15,
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        
        if final_score > 0.7:
            verdict = "环境声伪造"
            desc = "检测到多项环境声异常，音频中的环境声/背景声极可能为人工合成或后期添加。"
            risk = "高"
        elif final_score > 0.5:
            verdict = "疑似伪造"
            desc = "环境声特征存在部分异常，可能经过后期处理或混入合成环境声。"
            risk = "中"
        elif final_score > 0.3:
            verdict = "轻度异常"
            desc = "检测到少量环境声异常，可能为录音设备特性或轻度处理。"
            risk = "低"
        else:
            verdict = "环境声自然"
            desc = "环境声特征符合自然录音规律，未检测到明显的合成或伪造痕迹。"
            risk = "极低"
        
        all_indicators = []
        for detail in [noise_consistency, room_acoustics, env_fingerprint,
                       splice_detection, synth_detection]:
            all_indicators.extend(detail.get("indicators", []))
        
        return {
            "final_score": float(final_score),
            "verdict": verdict,
            "verdict_description": desc,
            "risk_level": risk,
            "scores": {k: float(v) for k, v in scores.items()},
            "indicators": all_indicators,
            "details": {
                "noise_consistency": noise_consistency,
                "room_acoustics": room_acoustics,
                "environmental_fingerprint": env_fingerprint,
                "splice_detection": splice_detection,
                "synthesized_detection": synth_detection,
            },
        }
    
    def _analyze_noise_consistency(self) -> Dict[str, Any]:
        """Analyze background noise floor consistency across the recording.
        
        Natural recordings have varying but consistent noise floors.
        Forged audio often has:
        - Perfectly uniform noise (synthesized)
        - Abrupt noise level changes (spliced from different sources)
        - Missing or artificial low-level noise texture
        """
        indicators = []
        score = 0.0
        
        # Segment audio and analyze noise floor per segment
        segment_dur = 1.0  # seconds
        segment_samples = int(segment_dur * self.sr)
        hop = segment_samples // 2
        
        noise_levels = []
        spectral_flatness_values = []
        
        for start in range(0, len(self.y) - segment_samples, hop):
            segment = self.y[start:start + segment_samples]
            
            # RMS level
            rms = float(np.sqrt(np.mean(segment ** 2)))
            noise_levels.append(rms)
            
            # Spectral flatness (Wiener entropy) - flat = noise-like
            S = np.abs(np.fft.rfft(segment))
            S = S[1:]  # exclude DC
            if np.sum(S) > 0:
                geo_mean = np.exp(np.mean(np.log(S + 1e-10)))
                arith_mean = np.mean(S)
                flatness = geo_mean / (arith_mean + 1e-10)
                spectral_flatness_values.append(float(flatness))
        
        if len(noise_levels) < 4:
            return {"score": 0, "indicators": ["音频太短，无法分析噪底"]}
        
        noise_levels = np.array(noise_levels)
        flatness_arr = np.array(spectral_flatness_values)
        
        # 1. Noise level variance (too uniform = suspicious)
        noise_cv = np.std(noise_levels) / (np.mean(noise_levels) + 1e-10)
        if noise_cv < 0.02:
            score += 0.35
            indicators.append(f"噪底电平异常均匀 (变异系数={noise_cv:.4f}) — 可能为合成噪声")
        elif noise_cv < 0.05:
            score += 0.15
            indicators.append(f"噪底电平变化偏小 (变异系数={noise_cv:.4f})")
        
        # 2. Abrupt noise level changes
        level_diffs = np.abs(np.diff(noise_levels))
        mean_diff = np.mean(level_diffs)
        max_diff = np.max(level_diffs)
        
        if max_diff > 5 * mean_diff and mean_diff > 0.001:
            score += 0.25
            jump_idx = np.argmax(level_diffs)
            jump_time = jump_idx * segment_dur / 2
            indicators.append(f"检测到噪底突变 (时间≈{jump_time:.1f}s, 跳变={max_diff:.4f}) — 可能为拼接点")
        
        # 3. Spectral flatness consistency
        if len(flatness_arr) > 3:
            flatness_std = np.std(flatness_arr)
            if flatness_std < 0.01:
                score += 0.2
                indicators.append(f"频谱平坦度过于一致 (标准差={flatness_std:.4f}) — 合成噪声特征")
        
        # 4. Check for perfectly white/pink noise (artificial)
        S_full = np.abs(librosa.stft(self.y, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH))
        avg_spectrum = np.mean(S_full, axis=1)
        
        if len(avg_spectrum) > 10:
            # Fit log-log slope (white noise = 0, pink = -1)
            freqs = np.arange(1, len(avg_spectrum) + 1)
            log_freqs = np.log(freqs[1:])
            log_power = np.log(avg_spectrum[1:] + 1e-10)
            
            # Linear regression in log domain
            slope = np.polyfit(log_freqs, log_power, 1)[0]
            
            # Check residual for perfect fit (too perfect = synthesized)
            fit = np.poly1d(np.polyfit(log_freqs, log_power, 1))
            residual = np.std(log_power - fit(log_freqs))
            
            if residual < 0.1 and abs(slope + 1) < 0.1:
                score += 0.2
                indicators.append(f"频谱完美符合1/f分布 (残差={residual:.3f}) — 可能为合成粉噪声")
            elif residual < 0.15 and abs(slope) < 0.1:
                score += 0.15
                indicators.append(f"频谱接近白噪声特征 (斜率={slope:.3f})")
        
        if not indicators:
            indicators.append(f"噪底特征正常 (变异系数={noise_cv:.4f})")
        
        return {"score": min(1.0, score), "indicators": indicators}
    
    def _analyze_room_acoustics(self) -> Dict[str, Any]:
        """Analyze room acoustic consistency (reverb, RT60).
        
        Natural recordings have consistent room acoustics throughout.
        Forged audio may show:
        - Changing reverb characteristics
        - Mismatched room signatures between segments
        - Artificial reverb patterns
        """
        indicators = []
        score = 0.0
        
        # Estimate reverb characteristics per segment
        segment_dur = 2.0
        segment_samples = int(segment_dur * self.sr)
        
        reverb_profiles = []
        
        for start in range(0, len(self.y) - segment_samples, segment_samples):
            segment = self.y[start:start + segment_samples]
            
            # Autocorrelation-based reverb estimation
            autocorr = np.correlate(segment[:len(segment)//2], segment[:len(segment)//2], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            
            # Decay rate (approximate RT60 indicator)
            decay_samples = np.where(autocorr < 0.1)[0]
            if len(decay_samples) > 0:
                decay_time = float(decay_samples[0]) / self.sr
            else:
                decay_time = float(len(autocorr)) / self.sr
            
            # Spectral centroid (room coloring indicator)
            S = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1.0 / self.sr)
            centroid = float(np.sum(freqs * S) / (np.sum(S) + 1e-10))
            
            reverb_profiles.append({
                'decay_time': decay_time,
                'centroid': centroid,
            })
        
        if len(reverb_profiles) < 3:
            return {"score": 0, "indicators": ["音频太短，无法分析室内声学"]}
        
        # 1. Reverb decay consistency
        decay_times = np.array([p['decay_time'] for p in reverb_profiles])
        decay_cv = np.std(decay_times) / (np.mean(decay_times) + 1e-10)
        
        if decay_cv > 0.5:
            score += 0.3
            indicators.append(f"混响衰减时间不一致 (变异系数={decay_cv:.3f}) — 可能混合了不同录音环境")
        
        # 2. Spectral centroid consistency (room coloring)
        centroids = np.array([p['centroid'] for p in reverb_profiles])
        centroid_cv = np.std(centroids) / (np.mean(centroids) + 1e-10)
        
        if centroid_cv > 0.3:
            score += 0.25
            indicators.append(f"频谱重心不一致 (变异系数={centroid_cv:.3f}) — 房间声学特征变化")
        
        # 3. Check for artificially perfect reverb (too consistent)
        if decay_cv < 0.01 and len(reverb_profiles) > 5:
            score += 0.2
            indicators.append(f"混响特征过度一致 — 可能为人工混响")
        
        if not indicators:
            indicators.append(f"室内声学特征一致 (衰减变异={decay_cv:.3f}, 频谱变异={centroid_cv:.3f})")
        
        return {"score": min(1.0, score), "indicators": indicators}
    
    def _analyze_environmental_fingerprint(self) -> Dict[str, Any]:
        """Analyze environmental spectral texture continuity.
        
        Real environments have characteristic spectral textures that evolve
        slowly and naturally. Forged environments may show:
        - Repeating texture patterns (looped background)
        - Abrupt texture changes (different environment spliced)
        - Unnaturally static texture (synthesized)
        """
        indicators = []
        score = 0.0
        
        # Extract spectral texture features per frame
        S = np.abs(librosa.stft(self.y, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        if S_db.shape[1] < 20:
            return {"score": 0, "indicators": ["音频太短"]}
        
        # 1. Texture repetition detection (looped background)
        # Compare spectral frames at different time offsets
        n_frames = S_db.shape[1]
        frame_features = []
        for i in range(0, n_frames, 5):
            frame_features.append(S_db[:, i])
        
        if len(frame_features) > 10:
            features = np.array(frame_features)
            # Self-similarity matrix
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            normalized = features / (norms + 1e-10)
            sim_matrix = normalized @ normalized.T
            
            # Check for periodic high similarity (indicates looping)
            upper_tri = sim_matrix[np.triu_indices(len(sim_matrix), k=3)]
            high_sim_ratio = np.mean(upper_tri > 0.95)
            
            if high_sim_ratio > 0.5:
                score += 0.4
                indicators.append(f"检测到高度重复的频谱纹理 ({high_sim_ratio:.1%}) — 可能为循环播放的背景声")
            elif high_sim_ratio > 0.3:
                score += 0.2
                indicators.append(f"频谱纹理重复率偏高 ({high_sim_ratio:.1%})")
        
        # 2. Texture stationarity (too static = suspicious)
        spectral_flux = np.sum(np.diff(S_db, axis=1) ** 2, axis=0)
        flux_cv = np.std(spectral_flux) / (np.mean(spectral_flux) + 1e-10)
        
        if np.mean(spectral_flux) < 0.5 and flux_cv < 0.3:
            score += 0.25
            indicators.append(f"环境声频谱过于静态 (通量均值={np.mean(spectral_flux):.2f}) — 可能为合成环境")
        
        # 3. Abrupt texture changes
        flux_diff = np.abs(np.diff(spectral_flux))
        if len(flux_diff) > 10:
            flux_threshold = np.mean(flux_diff) + 4 * np.std(flux_diff)
            jumps = np.where(flux_diff > flux_threshold)[0]
            
            if len(jumps) > 0:
                hop_dur = DEFAULT_HOP_LENGTH / self.sr
                jump_times = [f"{j * hop_dur:.1f}s" for j in jumps[:5]]
                score += min(0.3, len(jumps) * 0.1)
                indicators.append(f"检测到 {len(jumps)} 处环境声突变 (时间: {', '.join(jump_times)})")
        
        if not indicators:
            indicators.append("环境声纹理连续性正常")
        
        return {"score": min(1.0, score), "indicators": indicators}
    
    def _detect_ambient_splices(self) -> Dict[str, Any]:
        """Detect splices in ambient/background audio.
        
        Analyzes low-energy (non-speech) segments for discontinuities
        that indicate splicing from different recordings.
        """
        indicators = []
        score = 0.0
        
        # Focus on low-energy frames (ambient/background)
        frame_len = int(0.05 * self.sr)  # 50ms frames
        hop = frame_len // 2
        
        rms_values = []
        for start in range(0, len(self.y) - frame_len, hop):
            rms = np.sqrt(np.mean(self.y[start:start + frame_len] ** 2))
            rms_values.append(rms)
        
        rms_arr = np.array(rms_values)
        threshold = np.percentile(rms_arr, 30)  # Bottom 30% = ambient
        
        # Extract spectral features of ambient segments
        ambient_features = []
        ambient_times = []
        
        for i, (start_idx) in enumerate(range(0, len(self.y) - frame_len, hop)):
            if rms_arr[i] < threshold:
                segment = self.y[start_idx:start_idx + frame_len]
                S = np.abs(np.fft.rfft(segment))
                # Use spectral shape as feature
                if np.sum(S) > 0:
                    S_norm = S / (np.sum(S) + 1e-10)
                    ambient_features.append(S_norm[:100])  # First 100 bins
                    ambient_times.append(start_idx / self.sr)
        
        if len(ambient_features) < 10:
            return {"score": 0, "indicators": ["环境声段落不足，无法分析"]}
        
        features = np.array(ambient_features)
        
        # Compare consecutive ambient segments
        diffs = []
        for i in range(len(features) - 1):
            diff = np.sum((features[i] - features[i + 1]) ** 2)
            diffs.append(diff)
        
        diffs = np.array(diffs)
        mean_diff = np.mean(diffs)
        
        # Find outlier discontinuities
        if len(diffs) > 5:
            threshold_disc = mean_diff + 3 * np.std(diffs)
            disc_indices = np.where(diffs > threshold_disc)[0]
            
            if len(disc_indices) > 0:
                score += min(0.5, len(disc_indices) * 0.15)
                disc_times = [f"{ambient_times[i]:.1f}s" for i in disc_indices[:5]]
                indicators.append(f"检测到 {len(disc_indices)} 处环境声不连续 (时间: {', '.join(disc_times)})")
        
        # Check ambient spectral entropy distribution
        entropies = []
        for feat in features:
            ent = float(entropy(feat + 1e-10))
            entropies.append(ent)
        
        ent_arr = np.array(entropies)
        ent_bimodal = abs(skew(ent_arr))
        
        if ent_bimodal > 1.5:
            score += 0.2
            indicators.append(f"环境声熵分布异常 (偏度={ent_bimodal:.2f}) — 可能混合了不同来源")
        
        if not indicators:
            indicators.append("环境声连续性正常，未检测到拼接痕迹")
        
        return {"score": min(1.0, score), "indicators": indicators}
    
    def _detect_synthesized_sounds(self) -> Dict[str, Any]:
        """Detect synthesized or artificially generated environmental sounds.
        
        Synthesized sounds often have:
        - Unnaturally perfect harmonic relationships
        - Missing micro-variations present in natural sounds
        - Regular temporal patterns (machine-generated)
        - Lack of natural onset/offset characteristics
        """
        indicators = []
        score = 0.0
        
        S = np.abs(librosa.stft(self.y, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH))
        
        # 1. Harmonic regularity (synthesizers produce perfect harmonics)
        avg_spectrum = np.mean(S, axis=1)
        if len(avg_spectrum) > 20:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(avg_spectrum, height=np.mean(avg_spectrum) * 2)
            
            if len(peaks) >= 4:
                spacings = np.diff(peaks)
                spacing_regularity = 1.0 - (np.std(spacings) / (np.mean(spacings) + 1e-10))
                
                if spacing_regularity > 0.95:
                    score += 0.3
                    indicators.append(f"谐波间距完美规则 (规律性={spacing_regularity:.3f}) — 合成声特征")
                elif spacing_regularity > 0.85:
                    score += 0.15
                    indicators.append(f"谐波间距较规则 (规律性={spacing_regularity:.3f})")
        
        # 2. Temporal regularity (natural sounds have irregular timing)
        onset_frames = librosa.onset.onset_detect(y=self.y, sr=self.sr, units='frames')
        
        if len(onset_frames) >= 5:
            onset_intervals = np.diff(onset_frames)
            interval_cv = np.std(onset_intervals) / (np.mean(onset_intervals) + 1e-10)
            
            if interval_cv < 0.05:
                score += 0.25
                indicators.append(f"声音事件间隔过于规则 (变异系数={interval_cv:.3f}) — 机器生成特征")
            elif interval_cv < 0.1:
                score += 0.1
                indicators.append(f"声音事件间隔规律性偏高 (变异系数={interval_cv:.3f})")
        
        # 3. Amplitude envelope naturalness
        # Natural sounds have smooth, varied envelopes
        rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        
        if len(rms) > 20:
            # Check for step-like amplitude (unnatural)
            rms_diff = np.abs(np.diff(rms))
            rms_diff2 = np.abs(np.diff(rms_diff))
            
            # Many zero second derivatives = step function
            near_zero_ratio = np.mean(rms_diff2 < 1e-6)
            if near_zero_ratio > 0.5:
                score += 0.2
                indicators.append(f"振幅包络呈阶梯状 ({near_zero_ratio:.1%}近零二阶导) — 非自然特征")
        
        # 4. Phase coherence (synthesized audio often has abnormal phase)
        S_complex = librosa.stft(self.y, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH)
        phase = np.angle(S_complex)
        
        if phase.shape[1] > 10:
            phase_diff = np.diff(phase, axis=1)
            # Group delay deviation
            gd_std = np.mean(np.std(phase_diff, axis=1))
            
            if gd_std < 0.1:
                score += 0.15
                indicators.append(f"相位一致性异常 (群延迟标准差={gd_std:.3f}) — 合成音频特征")
        
        if not indicators:
            indicators.append("未检测到明显的合成声特征")
        
        return {"score": min(1.0, score), "indicators": indicators}
