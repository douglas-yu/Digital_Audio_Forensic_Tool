"""Application constants for Audio Forensics Tool."""

APP_NAME = "数字音频取证分析工具"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Audio Forensics Team"

# Supported audio formats
SUPPORTED_FORMATS = [
    "WAV Files (*.wav)",
    "MP3 Files (*.mp3)",
    "FLAC Files (*.flac)",
    "OGG Files (*.ogg)",
    "AAC Files (*.aac *.m4a)",
    "All Audio Files (*.wav *.mp3 *.flac *.ogg *.aac *.m4a)",
    "All Files (*.*)",
]

SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]

# Analysis defaults
DEFAULT_SR = 22050
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 128

# ENF reference frequencies (Hz) by region
ENF_FREQUENCIES = {
    "中国 (50Hz)": 50.0,
    "美国 (60Hz)": 60.0,
    "欧洲 (50Hz)": 50.0,
    "日本东部 (50Hz)": 50.0,
    "日本西部 (60Hz)": 60.0,
}

# Edit detection thresholds
EDIT_DETECTION_SENSITIVITY = {
    "低": 0.3,
    "中": 0.5,
    "高": 0.7,
}

# Spectrogram color maps
SPECTROGRAM_CMAPS = [
    "viridis", "magma", "inferno", "plasma",
    "coolwarm", "hot", "jet", "gray",
]

# Report templates
REPORT_FORMATS = ["PDF", "HTML"]

# UI Colors
COLORS = {
    "primary": "#1976D2",
    "secondary": "#424242",
    "accent": "#FF5722",
    "background": "#FAFAFA",
    "surface": "#FFFFFF",
    "error": "#D32F2F",
    "warning": "#FFA000",
    "success": "#388E3C",
    "waveform": "#1976D2",
    "spectrogram_bg": "#000000",
    "edit_marker": "#FF5722",
    "enf_line": "#4CAF50",
}
