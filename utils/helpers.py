"""Utility helper functions for Audio Forensics Tool."""

import os
import datetime
import numpy as np


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS.ms string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def db_to_amplitude(db: float) -> float:
    """Convert decibels to amplitude."""
    return 10 ** (db / 20.0)


def amplitude_to_db(amplitude: float) -> float:
    """Convert amplitude to decibels."""
    if amplitude <= 0:
        return -np.inf
    return 20.0 * np.log10(amplitude)


def get_timestamp() -> str:
    """Get current timestamp string for reports."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_filename(name: str) -> str:
    """Sanitize a string for use as filename."""
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in name if c.isalnum() or c in keepchars).rstrip()


def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if needed."""
    os.makedirs(path, exist_ok=True)
    return path


def samples_to_time(samples: int, sr: int) -> float:
    """Convert sample count to time in seconds."""
    return samples / sr


def time_to_samples(time_sec: float, sr: int) -> int:
    """Convert time in seconds to sample count."""
    return int(time_sec * sr)
