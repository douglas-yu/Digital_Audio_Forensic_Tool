"""Silero VAD integration for precise voice activity detection."""

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional


def _find_model(subdir: str, filename: str) -> str:
    """Find model file checking multiple locations."""
    candidates = [
        Path(__file__).parent.parent / "models" / subdir / filename,
        Path(os.getcwd()) / "models" / subdir / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Model not found: {subdir}/{filename}. "
        f"Run 'python download_models.py' to download models."
    )


class SileroVAD:
    """Silero Voice Activity Detection using ONNX model."""
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = _find_model("silero-vad", "silero_vad.onnx")
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.sample_rate = 16000
        self.window_size = 512  # for 16kHz
        self._reset_state()
    
    def _reset_state(self):
        """Reset internal RNN state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(self.sample_rate, dtype=np.int64)
    
    def _predict_chunk(self, audio_chunk: np.ndarray) -> float:
        """Run inference on a single audio chunk."""
        input_data = audio_chunk.astype(np.float32).reshape(1, -1)
        
        ort_inputs = {
            'input': input_data,
            'state': self._state,
            'sr': self._sr
        }
        
        ort_outputs = self.session.run(None, ort_inputs)
        probability = ort_outputs[0][0, 0]
        self._state = ort_outputs[1]
        
        return float(probability)
    
    def detect(self, audio: np.ndarray, sr: int,
               threshold: float = 0.5,
               min_speech_duration: float = 0.25,
               min_silence_duration: float = 0.1) -> dict:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            threshold: Speech probability threshold
            min_speech_duration: Minimum speech segment duration (seconds)
            min_silence_duration: Minimum silence between segments (seconds)
            
        Returns:
            Dict with speech_segments, speech_ratio, probabilities
        """
        import librosa
        
        # Resample to 16kHz if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Reset state for new audio
        self._reset_state()
        
        # Process in windows
        probabilities = []
        num_windows = len(audio) // self.window_size
        
        for i in range(num_windows):
            chunk = audio[i * self.window_size: (i + 1) * self.window_size]
            prob = self._predict_chunk(chunk)
            probabilities.append(prob)
        
        probabilities = np.array(probabilities)
        
        # Convert to speech segments
        speech_flags = probabilities >= threshold
        segments = self._flags_to_segments(
            speech_flags, 
            self.window_size / self.sample_rate,
            min_speech_duration,
            min_silence_duration
        )
        
        # Calculate speech ratio
        total_duration = len(audio) / self.sample_rate
        speech_duration = sum(end - start for start, end in segments)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
        
        return {
            'speech_segments': segments,
            'speech_ratio': speech_ratio,
            'speech_duration': speech_duration,
            'total_duration': total_duration,
            'num_segments': len(segments),
            'probabilities': probabilities,
            'window_duration': self.window_size / self.sample_rate,
            'model': 'Silero VAD v5.1'
        }
    
    def _flags_to_segments(self, flags: np.ndarray, window_duration: float,
                           min_speech: float, min_silence: float) -> List[Tuple[float, float]]:
        """Convert boolean flags to time segments with smoothing."""
        segments = []
        in_speech = False
        start_time = 0.0
        silence_start = 0.0
        
        for i, is_speech in enumerate(flags):
            t = i * window_duration
            
            if is_speech and not in_speech:
                start_time = t
                in_speech = True
            elif not is_speech and in_speech:
                if silence_start == 0:
                    silence_start = t
                if t - silence_start >= min_silence:
                    # End segment
                    duration = silence_start - start_time
                    if duration >= min_speech:
                        segments.append((start_time, silence_start))
                    in_speech = False
                    silence_start = 0.0
            elif is_speech and in_speech:
                silence_start = 0.0
        
        # Handle last segment
        if in_speech:
            end_time = len(flags) * window_duration
            duration = end_time - start_time
            if duration >= min_speech:
                segments.append((start_time, end_time))
        
        return segments


class SpeakerEmbedder:
    """Speaker embedding extraction using Wespeaker ResNet34 ONNX model."""
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = _find_model("ecapa-tdnn", "speaker_model.onnx")
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.sample_rate = 16000
        self.n_mels = 80
    
    def extract_fbank(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract 80-dim filterbank features."""
        import librosa
        
        # Resample to 16kHz
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Extract mel spectrogram (80 bins)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_fft=512, hop_length=160, win_length=400,
            n_mels=self.n_mels, fmin=20, fmax=7600
        )
        
        # Log mel
        log_mel = np.log(mel_spec + 1e-6)
        
        # Transpose to (T, 80)
        log_mel = log_mel.T
        
        # Normalize
        log_mel = (log_mel - np.mean(log_mel, axis=0)) / (np.std(log_mel, axis=0) + 1e-8)
        
        return log_mel.astype(np.float32)
    
    def extract_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract speaker embedding vector from audio.
        
        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            
        Returns:
            256-dim embedding vector (normalized)
        """
        # Extract features
        feats = self.extract_fbank(audio, sr)
        
        # Add batch dimension: (1, T, 80)
        feats_input = feats[np.newaxis, :, :]
        
        # Run inference
        ort_inputs = {'feats': feats_input}
        ort_outputs = self.session.run(None, ort_inputs)
        
        # Get embedding and normalize
        embedding = ort_outputs[0][0]  # (256,)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def compare_speakers(self, audio1: np.ndarray, sr1: int,
                         audio2: np.ndarray, sr2: int) -> dict:
        """
        Compare two audio segments for speaker similarity.
        
        Returns:
            Dict with similarity score and verdict
        """
        emb1 = self.extract_embedding(audio1, sr1)
        emb2 = self.extract_embedding(audio2, sr2)
        
        # Cosine similarity
        cosine_sim = float(np.dot(emb1, emb2))
        
        # Euclidean distance
        euclidean_dist = float(np.linalg.norm(emb1 - emb2))
        
        # Verdict based on cosine similarity thresholds
        if cosine_sim >= 0.75:
            verdict = "极可能同一说话人"
            confidence = "高"
        elif cosine_sim >= 0.55:
            verdict = "可能同一说话人"
            confidence = "中"
        elif cosine_sim >= 0.35:
            verdict = "不确定"
            confidence = "低"
        else:
            verdict = "极可能不同说话人"
            confidence = "高"
        
        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'verdict': verdict,
            'confidence': confidence,
            'embedding_dim': len(emb1),
            'model': 'Wespeaker ResNet34 (VoxCeleb)'
        }
    
    def diarize(self, audio: np.ndarray, sr: int,
                segment_duration: float = 1.5,
                overlap: float = 0.5,
                max_speakers: int = 10) -> dict:
        """
        Perform speaker diarization using neural embeddings + clustering.
        
        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            segment_duration: Duration of each analysis segment (seconds)
            overlap: Overlap between segments (seconds)
            max_speakers: Maximum number of speakers to detect
            
        Returns:
            Dict with segments labeled by speaker ID
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Segment audio
        segment_samples = int(segment_duration * sr)
        hop_samples = int((segment_duration - overlap) * sr)
        
        embeddings = []
        segment_times = []
        
        for start in range(0, len(audio) - segment_samples, hop_samples):
            segment = audio[start:start + segment_samples]
            
            # Skip very quiet segments
            if np.max(np.abs(segment)) < 0.01:
                continue
            
            emb = self.extract_embedding(segment, sr)
            embeddings.append(emb)
            segment_times.append((start / sr, (start + segment_samples) / sr))
        
        if len(embeddings) < 2:
            return {
                'num_speakers': 1,
                'segments': [(0, len(audio) / sr, 0)],
                'model': 'Wespeaker ResNet34 + AgglomerativeClustering'
            }
        
        embeddings = np.array(embeddings)
        
        # Find optimal number of speakers
        best_n = 2
        best_score = -1
        
        for n in range(2, min(max_speakers + 1, len(embeddings))):
            clustering = AgglomerativeClustering(
                n_clusters=n, metric='cosine', linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            if len(set(labels)) < 2:
                continue
            
            score = silhouette_score(embeddings, labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_n = n
        
        # Final clustering
        clustering = AgglomerativeClustering(
            n_clusters=best_n, metric='cosine', linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Build labeled segments
        labeled_segments = []
        for (start, end), label in zip(segment_times, labels):
            labeled_segments.append((start, end, int(label)))
        
        # Merge adjacent same-speaker segments
        merged = self._merge_segments(labeled_segments)
        
        # Speaker statistics
        speaker_durations = {}
        for start, end, spk in merged:
            speaker_durations[spk] = speaker_durations.get(spk, 0) + (end - start)
        
        return {
            'num_speakers': best_n,
            'silhouette_score': float(best_score),
            'segments': merged,
            'speaker_durations': speaker_durations,
            'total_segments': len(merged),
            'model': 'Wespeaker ResNet34 + AgglomerativeClustering'
        }
    
    def _merge_segments(self, segments: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        """Merge adjacent segments with same speaker label."""
        if not segments:
            return []
        
        merged = [segments[0]]
        for start, end, label in segments[1:]:
            prev_start, prev_end, prev_label = merged[-1]
            if label == prev_label and start - prev_end < 0.3:
                merged[-1] = (prev_start, end, label)
            else:
                merged.append((start, end, label))
        
        return merged
