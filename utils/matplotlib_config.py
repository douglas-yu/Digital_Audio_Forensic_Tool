"""Matplotlib Chinese font configuration.

Configures matplotlib to correctly display Chinese characters on Windows
by setting font properties to use system Chinese fonts.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import os


def _find_chinese_font() -> str:
    """Find an available Chinese font on the system."""
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
        "DengXian",
        "Source Han Sans CN",
        "Noto Sans CJK SC",
    ]

    available = {f.name for f in fontManager.ttflist}
    for font_name in preferred:
        if font_name in available:
            return font_name

    # Fallback: try to find any font with CJK/Chinese in the path
    for f in fontManager.ttflist:
        lower_path = f.fname.lower()
        if any(kw in lower_path for kw in ["msyh", "simhei", "simsun", "yahei", "dengxian"]):
            return f.name

    return "SimHei"  # last resort default


def configure_matplotlib_chinese():
    """Configure matplotlib to display Chinese text correctly."""
    font_name = _find_chinese_font()

    plt.rcParams.update({
        "font.sans-serif": [font_name, "Arial", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,  # fix minus sign display
    })

    # Rebuild font cache if needed
    matplotlib.rcParams["font.sans-serif"] = [font_name, "Arial", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    return font_name
