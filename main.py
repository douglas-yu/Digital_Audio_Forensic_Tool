"""Digital Audio Forensics Tool - Main Entry Point.

A PyQt5-based application for forensic analysis of audio recordings.
Features: waveform visualization, spectrogram analysis, metadata viewing,
ENF (Electric Network Frequency) analysis, edit/splice detection,
and forensic report generation (PDF/HTML).
"""

import sys
import os
import ctypes

# Ensure the project root is on the path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

# IMPORTANT: Import onnxruntime BEFORE PyQt5 to avoid DLL conflict on Windows
try:
    import onnxruntime
except ImportError:
    pass

# Configure matplotlib Chinese fonts BEFORE any matplotlib imports in UI modules
from utils.matplotlib_config import configure_matplotlib_chinese
configure_matplotlib_chinese()

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer

from ui.main_window import MainWindow
from utils.constants import APP_NAME


def main():
    # Set Windows taskbar icon (must be called before QApplication)
    if sys.platform == "win32":
        app_id = "audioforensics.tool.1.0"
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    # Set application-wide icon (taskbar + title bar)
    icon_path = os.path.join(APP_DIR, "resources", "icons", "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Set default font
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # Show splash screen
    splash_path = os.path.join(APP_DIR, "resources", "splash.png")
    splash = None
    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()

    window = MainWindow()

    # Close splash and show main window after a brief delay
    if splash:
        QTimer.singleShot(2000, lambda: _show_main(splash, window))
    else:
        window.show()

    sys.exit(app.exec_())


def _show_main(splash: QSplashScreen, window: MainWindow):
    """Close splash screen and show main window."""
    splash.close()
    window.show()


if __name__ == "__main__":
    main()
