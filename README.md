# 数字音频取证分析工具 (Digital Audio Forensics Tool)

基于 PyQt5 的数字音频取证分析应用，用于音频文件的专业分析和取证。

## 功能特性

- **波形可视化**: 时域波形显示，支持 RMS 包络、过零率、频谱质心叠加
- **频谱分析**: STFT 频谱图、Mel 频谱图、MFCC、FFT 频谱、功率谱密度
- **元数据查看**: 音频文件属性、格式信息、扩展元数据
- **ENF 分析**: 电网频率 (50/60Hz) 检测与追踪，谐波分析
- **编辑检测**: 频谱不连续、相位跳变、能量异常、统计变点检测
- **报告生成**: 支持 PDF 和 HTML 格式的取证分析报告

## 支持格式

WAV, MP3, FLAC, OGG, AAC/M4A

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

## 项目结构

```
├── main.py              # 应用入口
├── requirements.txt     # Python 依赖
├── ui/                  # 界面模块
│   ├── main_window.py   # 主窗口
│   ├── waveform_tab.py  # 波形标签页
│   ├── spectrogram_tab.py # 频谱图标签页
│   ├── metadata_tab.py  # 元数据标签页
│   ├── noise_tab.py     # ENF 分析标签页
│   ├── edit_detection_tab.py # 编辑检测标签页
│   └── report_tab.py    # 报告生成标签页
├── core/                # 核心分析模块
│   ├── audio_loader.py  # 音频加载
│   ├── analysis.py      # 音频分析引擎
│   ├── enf_analysis.py  # ENF 分析
│   ├── edit_detector.py # 编辑检测
│   └── report_generator.py # 报告生成
└── utils/               # 工具模块
    ├── constants.py     # 常量定义
    └── helpers.py       # 辅助函数
```

## 技术栈

- **GUI**: PyQt5
- **音频处理**: librosa, soundfile, pydub
- **信号处理**: numpy, scipy
- **可视化**: matplotlib
- **报告生成**: reportlab (PDF), Jinja2 (HTML)
