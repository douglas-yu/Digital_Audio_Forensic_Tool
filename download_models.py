"""Download AI models for Audio Forensic application."""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve


MODELS = {
    "silero-vad": {
        "url": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
        "file": "silero_vad.onnx",
        "description": "Silero VAD v5.1 - Voice Activity Detection (~2.2MB)",
    },
    "ecapa-tdnn": {
        "url": "https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.onnx",
        "file": "speaker_model.onnx",
        "description": "Wespeaker ResNet34 - Speaker Embedding (~25MB)",
    },
}


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        bar = "█" * int(percent // 2) + "░" * (50 - int(percent // 2))
        sys.stdout.write(f"\r  [{bar}] {percent:.0f}% ({downloaded / 1024 / 1024:.1f}MB)")
        sys.stdout.flush()


def main():
    base_dir = Path(__file__).parent / "models"
    base_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  数字音频取证 - AI 模型下载工具")
    print("=" * 60)

    for name, info in MODELS.items():
        model_dir = base_dir / name
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / info["file"]

        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ {info['description']}")
            print(f"   已存在 ({size_mb:.1f}MB): {model_path}")
            continue

        print(f"\n⬇ 下载: {info['description']}")
        print(f"  来源: {info['url']}")

        try:
            urlretrieve(info["url"], str(model_path), download_progress)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"\n  ✅ 完成 ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"\n  ❌ 下载失败: {e}")
            if model_path.exists():
                model_path.unlink()

    print("\n" + "=" * 60)
    print("  下载完成! 模型将在下次分析时自动加载。")
    print("=" * 60)


if __name__ == "__main__":
    main()
