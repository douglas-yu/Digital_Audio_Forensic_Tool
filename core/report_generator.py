"""Forensic report generator module."""

import os
import datetime
from typing import Dict, Any, Optional, List

import numpy as np

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from jinja2 import Template

from utils.constants import APP_NAME, APP_VERSION, COLORS
from utils.helpers import format_duration, format_file_size, get_timestamp


# ── HTML report template ─────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>
  body { font-family: "Microsoft YaHei", Arial, sans-serif; margin: 40px; color: #333; }
  h1 { color: {{ primary_color }}; border-bottom: 2px solid {{ primary_color }}; padding-bottom: 10px; }
  h2 { color: {{ secondary_color }}; margin-top: 30px; }
  h3 { color: #555; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
  th { background-color: {{ primary_color }}; color: white; }
  tr:nth-child(even) { background-color: #f9f9f9; }
  .warning { color: {{ warning_color }}; font-weight: bold; }
  .success { color: {{ success_color }}; }
  .error { color: {{ error_color }}; font-weight: bold; }
  .info-box { background: #e3f2fd; border-left: 4px solid {{ primary_color }};
              padding: 12px 16px; margin: 15px 0; }
  .edit-marker { background: #fff3e0; border-left: 4px solid {{ accent_color }};
                 padding: 10px 14px; margin: 10px 0; }
  .footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd;
            font-size: 12px; color: #999; }
  img { max-width: 100%; height: auto; margin: 10px 0; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p><strong>生成时间:</strong> {{ timestamp }}</p>
<p><strong>工具版本:</strong> {{ app_name }} v{{ app_version }}</p>

<h2>1. 文件信息</h2>
<div class="info-box">
<table>
{% for key, value in file_info.items() %}
<tr><th>{{ key }}</th><td>{{ value }}</td></tr>
{% endfor %}
</table>
</div>

{% if statistics %}
<h2>2. 音频统计</h2>
<table>
{% for key, value in statistics.items() %}
<tr><th>{{ key }}</th><td>{{ value }}</td></tr>
{% endfor %}
</table>
{% endif %}

{% if enf_results %}
<h2>3. ENF 分析</h2>
<div class="info-box">
<p><strong>标称频率:</strong> {{ enf_results.nominal_freq }} Hz</p>
<p><strong>平均频率:</strong> {{ enf_results.mean_freq|round(4) }} Hz</p>
<p><strong>频率标准差:</strong> {{ enf_results.std_freq|round(4) }} Hz</p>
<p><strong>信噪比:</strong> {{ enf_results.snr_db|round(2) }} dB</p>
</div>
{% endif %}

{% if edit_results %}
<h2>4. 编辑/拼接检测</h2>
<p>共检测到 <strong>{{ edit_results.total_detections }}</strong> 个可疑编辑点</p>
{% for point in edit_results.edit_points %}
<div class="edit-marker">
  <strong>时间: {{ point.time|round(3) }}s</strong> |
  置信度: {{ (point.confidence * 100)|round(1) }}% |
  检测方法: {{ point.methods|join(', ') }}
</div>
{% endfor %}
{% endif %}

{% if conclusions %}
<h2>5. 分析结论</h2>
{% for conclusion in conclusions %}
<div class="info-box" style="white-space: pre-wrap; font-family: monospace, 'Microsoft YaHei';">{{ conclusion }}</div>
{% endfor %}
{% endif %}

<div class="footer">
  <p>本报告由 {{ app_name }} v{{ app_version }} 自动生成</p>
  <p>生成时间: {{ timestamp }}</p>
</div>
</body>
</html>
"""


class ReportGenerator:
    """Generate forensic analysis reports in HTML and PDF formats."""

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def set_file_info(self, info: Dict[str, Any]):
        self.data["file_info"] = info

    def set_statistics(self, stats: Dict[str, Any]):
        self.data["statistics"] = stats

    def set_enf_results(self, results: Dict[str, Any]):
        # Convert numpy types for template rendering
        clean = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                continue
            clean[k] = v
        self.data["enf_results"] = clean

    def set_edit_results(self, results: Dict[str, Any]):
        clean = {
            "total_detections": results.get("total_detections", 0),
            "by_method": results.get("by_method", {}),
            "edit_points": [],
        }
        for pt in results.get("edit_points", []):
            clean["edit_points"].append({
                "time": pt["time"],
                "confidence": pt.get("confidence", 0),
                "methods": pt.get("methods", [pt.get("method", "unknown")]),
            })
        self.data["edit_results"] = clean

    def set_conclusions(self, conclusions: List[str]):
        self.data["conclusions"] = conclusions

    def add_image(self, label: str, path: str):
        self.data.setdefault("images", {})[label] = path

    # ── HTML ───────────────────────────────────────────────────────

    def generate_html(self, output_path: str, title: str = "数字音频取证分析报告"):
        """Generate an HTML report."""
        template = Template(HTML_TEMPLATE)
        html = template.render(
            title=title,
            timestamp=get_timestamp(),
            app_name=APP_NAME,
            app_version=APP_VERSION,
            primary_color=COLORS["primary"],
            secondary_color=COLORS["secondary"],
            accent_color=COLORS["accent"],
            warning_color=COLORS["warning"],
            success_color=COLORS["success"],
            error_color=COLORS["error"],
            **self.data,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    # ── PDF ────────────────────────────────────────────────────────

    def generate_pdf(self, output_path: str, title: str = "数字音频取证分析报告"):
        """Generate a PDF report using ReportLab."""
        if not HAS_REPORTLAB:
            raise RuntimeError(
                "reportlab 未安装。请运行: pip install reportlab"
            )

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=20 * mm, leftMargin=20 * mm,
            topMargin=20 * mm, bottomMargin=20 * mm,
        )
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Title"],
            fontSize=20, textColor=HexColor(COLORS["primary"]),
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"生成时间: {get_timestamp()}", styles["Normal"]))
        story.append(Paragraph(f"工具: {APP_NAME} v{APP_VERSION}", styles["Normal"]))
        story.append(Spacer(1, 20))

        # File info table
        if "file_info" in self.data:
            story.append(Paragraph("1. 文件信息", styles["Heading2"]))
            table_data = [["属性", "值"]]
            for k, v in self.data["file_info"].items():
                table_data.append([str(k), str(v)])
            t = Table(table_data, colWidths=[150, 300])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor(COLORS["primary"])),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(t)
            story.append(Spacer(1, 15))

        # Statistics
        if "statistics" in self.data:
            story.append(Paragraph("2. 音频统计", styles["Heading2"]))
            table_data = [["指标", "值"]]
            for k, v in self.data["statistics"].items():
                table_data.append([str(k), str(v)])
            t = Table(table_data, colWidths=[200, 250])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor(COLORS["primary"])),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 15))

        # Edit detection results
        if "edit_results" in self.data:
            er = self.data["edit_results"]
            story.append(Paragraph("3. 编辑/拼接检测", styles["Heading2"]))
            story.append(Paragraph(
                f"共检测到 {er['total_detections']} 个可疑编辑点",
                styles["Normal"],
            ))
            if er["edit_points"]:
                table_data = [["时间(秒)", "置信度", "检测方法"]]
                for pt in er["edit_points"]:
                    table_data.append([
                        f"{pt['time']:.3f}",
                        f"{pt['confidence'] * 100:.1f}%",
                        ", ".join(pt.get("methods", [])),
                    ])
                t = Table(table_data, colWidths=[100, 100, 250])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor(COLORS["accent"])),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]))
                story.append(t)

        doc.build(story)
        return output_path
