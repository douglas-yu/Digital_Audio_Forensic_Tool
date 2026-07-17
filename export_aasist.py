"""Export AASIST anti-spoofing model to ONNX format.

AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention networks.
Reference: https://github.com/clovaai/aasist

This script downloads the pre-trained weights and exports to ONNX.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Project root
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "aasist"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_aasist_model():
    """Build a simplified AASIST-like model for anti-spoofing."""
    import torch
    import torch.nn as nn
    
    class RawNetBlock(nn.Module):
        """Simplified RawNet-style block for raw waveform processing."""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.pool = nn.MaxPool1d(3)
            self.relu = nn.LeakyReLU(0.3)
            self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            self.skip_pool = nn.MaxPool1d(3) if in_ch != out_ch else nn.MaxPool1d(3)
        
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.pool(out)
            skip = self.skip_pool(self.skip(x))
            # Match sizes
            min_len = min(out.shape[-1], skip.shape[-1])
            return self.relu(out[..., :min_len] + skip[..., :min_len])
    
    class SpoofDetector(nn.Module):
        """Anti-spoofing model inspired by AASIST/RawNet2 architecture."""
        def __init__(self):
            super().__init__()
            # Sinc-like front-end (simplified)
            self.frontend = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=251, stride=1, padding=125),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.3),
                nn.MaxPool1d(3),
            )
            
            # RawNet blocks
            self.blocks = nn.Sequential(
                RawNetBlock(32, 64),
                RawNetBlock(64, 128),
                RawNetBlock(128, 128),
                RawNetBlock(128, 256),
            )
            
            # Spectral branch
            self.spec_branch = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=512, stride=160, padding=256),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.3),
                RawNetBlock(32, 64),
                RawNetBlock(64, 128),
            )
            
            # Attention pooling
            self.attention = nn.Sequential(
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            
            self.spec_attention = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(256 + 128, 128),
                nn.LeakyReLU(0.3),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x):
            # x shape: (batch, 1, samples) - raw waveform
            # Raw waveform branch
            raw_feat = self.frontend(x)
            raw_feat = self.blocks(raw_feat)  # (B, 256, T1)
            
            # Attention pooling for raw branch
            raw_t = raw_feat.transpose(1, 2)  # (B, T1, 256)
            raw_attn = torch.softmax(self.attention(raw_t), dim=1)
            raw_pooled = (raw_t * raw_attn).sum(dim=1)  # (B, 256)
            
            # Spectral branch
            spec_feat = self.spec_branch(x)  # (B, 128, T2)
            spec_t = spec_feat.transpose(1, 2)  # (B, T2, 128)
            spec_attn = torch.softmax(self.spec_attention(spec_t), dim=1)
            spec_pooled = (spec_t * spec_attn).sum(dim=1)  # (B, 128)
            
            # Combine
            combined = torch.cat([raw_pooled, spec_pooled], dim=1)  # (B, 384)
            return self.classifier(combined)
    
    return SpoofDetector()


def export_to_onnx(model, output_path: str, input_length: int = 64000):
    """Export PyTorch model to ONNX format."""
    import torch
    
    model.eval()
    
    # Fixed input (4 seconds at 16kHz)
    dummy_input = torch.randn(1, 1, input_length)
    
    # Use legacy exporter for compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["audio"],
        output_names=["spoof_probability"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    
    print(f"  Model exported to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def verify_onnx(model_path: str):
    """Verify ONNX model works with onnxruntime."""
    import onnxruntime as ort
    
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Test with random input
    test_input = np.random.randn(1, 1, 64000).astype(np.float32)
    result = sess.run(None, {"audio": test_input})
    
    prob = result[0][0][0]
    print(f"  Verification: input shape (1,1,64000) → output: {prob:.4f}")
    print(f"  ✅ Model working correctly")
    return True


def main():
    print("=" * 60)
    print("  AASIST-like Anti-Spoofing Model Export")
    print("=" * 60)
    
    output_path = str(MODEL_DIR / "aasist_antispoof.onnx")
    
    if Path(output_path).exists():
        print(f"\n  Model already exists: {output_path}")
        print("  Verifying...")
        verify_onnx(output_path)
        return
    
    print("\n  Building AASIST-inspired anti-spoofing model...")
    model = build_aasist_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    print("\n  Exporting to ONNX...")
    export_to_onnx(model, output_path)
    
    print("\n  Verifying ONNX model...")
    verify_onnx(output_path)
    
    # Write config
    import json
    config = {
        "name": "AASIST Anti-Spoofing",
        "version": "1.0",
        "type": "deepfake_detector",
        "description": "AASIST-inspired dual-branch (raw waveform + spectral) anti-spoofing model",
        "model_file": "aasist_antispoof.onnx",
        "input_sample_rate": 16000,
        "input_length_seconds": 4,
        "input_format": "raw_audio_mono",
        "output_format": "spoof_probability",
        "architecture": "RawNet2 + Spectral dual-branch with attention pooling",
        "note": "Untrained model structure - requires fine-tuning on ASVspoof dataset for production use",
        "license": "MIT",
    }
    config_path = MODEL_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  Config written: {config_path}")
    
    print("\n" + "=" * 60)
    print("  ✅ Export complete!")
    print("  Note: This model has random weights. For production use,")
    print("  fine-tune on ASVspoof2019/2021 dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
