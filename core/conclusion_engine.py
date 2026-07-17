"""Intelligent conclusion engine for audio forensic analysis.

Automatically generates per-detection analysis descriptions and
a comprehensive summary conclusion based on all analysis results.
"""

from typing import Dict, Any, List, Optional
from utils.helpers import format_duration


class ConclusionEngine:
    """Generate intelligent analysis conclusions from forensic results."""

    def __init__(self):
        self._audio_info: Optional[Dict[str, Any]] = None
        self._statistics: Optional[Dict[str, Any]] = None
        self._enf_results: Optional[Dict[str, Any]] = None
        self._edit_results: Optional[Dict[str, Any]] = None

    def set_audio_info(self, info: Dict[str, Any]):
        self._audio_info = info

    def set_statistics(self, stats: Dict[str, Any]):
        self._statistics = stats

    def set_enf_results(self, results: Dict[str, Any]):
        self._enf_results = results

    def set_edit_results(self, results: Dict[str, Any]):
        self._edit_results = results

    # ── Per-Detection Analysis ────────────────────────────────────

    def analyze_statistics(self) -> str:
        """Generate analysis description for audio statistics."""
        if not self._statistics:
            return "尚未计算音频统计信息。"

        s = self._statistics
        lines = ["【音频统计分析】\n"]

        # Peak amplitude analysis
        peak = s.get("peak_amplitude", 0)
        if peak > 0.99:
            lines.append("⚠ 峰值振幅接近满刻度 (clipping)，音频可能存在削波失真。")
        elif peak > 0.9:
            lines.append("⚡ 峰值振幅较高，动态余量较小，但未达到削波水平。")
        elif peak < 0.1:
            lines.append("📉 峰值振幅极低，音频信号非常微弱，可能存在录制问题或音量过低。")
        else:
            lines.append(f"✅ 峰值振幅正常 ({peak:.4f})，信号强度适中。")

        # RMS analysis
        rms_db = s.get("rms_db", -100)
        if rms_db > -6:
            lines.append(f"🔊 RMS 电平较高 ({rms_db:.1f} dB)，音频整体响度大。")
        elif rms_db < -40:
            lines.append(f"🔇 RMS 电平极低 ({rms_db:.1f} dB)，可能包含大量静音段或录音音量不足。")
        else:
            lines.append(f"🎵 RMS 电平 {rms_db:.1f} dB，处于正常范围。")

        # DC offset
        dc = s.get("dc_offset", 0)
        if abs(dc) > 0.01:
            lines.append(f"⚠ 检测到显著直流偏移 ({dc:.6f})，可能表示录制设备存在问题或音频经过不当处理。")
        else:
            lines.append("✅ 直流偏移可忽略，录制设备工作正常。")

        # Dynamic range
        dr = s.get("dynamic_range_db", 0)
        if dr > 60:
            lines.append(f"📊 动态范围宽广 ({dr:.1f} dB)，音频保留了较好的动态细节。")
        elif dr < 20:
            lines.append(f"⚠ 动态范围较窄 ({dr:.1f} dB)，音频可能经过压缩处理。")
        else:
            lines.append(f"📊 动态范围 {dr:.1f} dB，处于中等水平。")

        # Crest factor
        cf = s.get("crest_factor", 0)
        if cf > 10:
            lines.append(f"🎯 波峰因数高 ({cf:.2f})，表明信号具有明显的瞬态峰值特征。")
        elif cf < 2:
            lines.append(f"⚠ 波峰因数低 ({cf:.2f})，信号可能经过严重压限处理。")

        return "\n".join(lines)

    def analyze_enf(self) -> str:
        """Generate analysis description for ENF results."""
        if not self._enf_results:
            return "尚未执行 ENF 分析。"

        r = self._enf_results
        lines = ["【ENF (电网频率) 分析】\n"]

        nominal = r.get("nominal_freq", 50)
        mean_f = r.get("mean_freq", 0)
        std_f = r.get("std_freq", 0)
        snr = r.get("snr_db", 0)

        # SNR assessment
        if snr > 20:
            lines.append(f"✅ ENF 信号信噪比良好 ({snr:.1f} dB)，电网频率成分清晰可辨。")
            lines.append("   这表明录音环境靠近交流电源，录音设备可能通过电磁耦合捕获了电网频率。")
        elif snr > 10:
            lines.append(f"⚡ ENF 信号信噪比中等 ({snr:.1f} dB)，可检测到电网频率但信号较弱。")
        elif snr > 0:
            lines.append(f"⚠ ENF 信号信噪比较低 ({snr:.1f} dB)，电网频率信号微弱。")
            lines.append("   分析结果可靠性较低，建议谨慎采信。")
        else:
            lines.append(f"❌ 未检测到明显的 ENF 信号 (SNR: {snr:.1f} dB)。")
            lines.append("   可能原因：录音设备使用电池供电、远离电网、或音频经过了处理。")

        # Frequency deviation
        dev = abs(mean_f - nominal)
        if dev < 0.02:
            lines.append(f"\n📊 平均频率 {mean_f:.4f} Hz，与标称 {nominal} Hz 偏差极小 ({dev:.4f} Hz)。")
            lines.append("   频率稳定性良好，未见异常跳变。")
        elif dev < 0.1:
            lines.append(f"\n📊 平均频率 {mean_f:.4f} Hz，偏差 {dev:.4f} Hz 在正常电网波动范围内。")
        else:
            lines.append(f"\n⚠ 平均频率 {mean_f:.4f} Hz，偏离标称值 {dev:.4f} Hz，偏差较大。")
            lines.append("   可能原因：录音时电网负荷异常、或音频经过变速处理。")

        # Stability
        if std_f < 0.01:
            lines.append(f"\n🎯 频率标准差 {std_f:.4f} Hz，ENF 轨迹非常稳定。")
        elif std_f < 0.05:
            lines.append(f"\n📈 频率标准差 {std_f:.4f} Hz，波动在正常范围内。")
        else:
            lines.append(f"\n⚠ 频率标准差 {std_f:.4f} Hz，波动较大。")
            lines.append("   可能存在 ENF 不连续，需结合轨迹图进一步判断是否有编辑痕迹。")

        # Forensic implications
        lines.append("\n【取证意义】")
        if snr > 10 and std_f < 0.05:
            lines.append("• ENF 轨迹连续稳定，支持录音为连续录制（未经剪辑）的判断。")
            lines.append("• 可将 ENF 轨迹与电网频率数据库进行比对以确定录制时间。")
        elif snr > 10 and std_f >= 0.05:
            lines.append("• ENF 波动较大，需进一步检查是否存在录音中断或拼接痕迹。")
        else:
            lines.append("• ENF 信号不足以得出可靠结论，建议结合其他分析方法综合判断。")

        return "\n".join(lines)

    def analyze_edit_detection(self) -> str:
        """Generate analysis description for edit detection results."""
        if not self._edit_results:
            return "尚未执行编辑检测。"

        r = self._edit_results
        total = r.get("total_detections", 0)
        by_method = r.get("by_method", {})
        points = r.get("edit_points", [])

        lines = ["【编辑/拼接检测分析】\n"]

        if total == 0:
            lines.append("✅ 未检测到可疑编辑点。")
            lines.append("   四种检测方法（频谱不连续、相位跳变、能量异常、统计变点）均未发现异常。")
            lines.append("   这支持音频为连续录制、未经后期编辑的判断。")
            lines.append("\n   注意：此结论不能完全排除经过高水平处理的编辑可能性。")
        else:
            lines.append(f"⚠ 共检测到 {total} 个可疑编辑点：\n")

            # Per-method summary
            method_names = {
                "spectral": "频谱不连续检测",
                "phase": "相位跳变检测",
                "energy": "能量异常检测",
                "statistical": "统计变点检测 (CUSUM)",
            }
            for method, count in by_method.items():
                name = method_names.get(method, method)
                if count > 0:
                    lines.append(f"  • {name}: 检测到 {count} 个可疑点")
                else:
                    lines.append(f"  • {name}: 未检测到异常")

            # Per-point detailed analysis
            high_conf = [p for p in points if p.get("confidence", 0) > 0.7]
            medium_conf = [p for p in points if 0.3 <= p.get("confidence", 0) <= 0.7]
            low_conf = [p for p in points if p.get("confidence", 0) < 0.3]

            if high_conf:
                lines.append(f"\n🔴 高置信度可疑点 ({len(high_conf)} 个):")
                for p in high_conf[:5]:
                    methods = p.get("methods", [])
                    t = p["time"]
                    conf = p.get("confidence", 0) * 100
                    n_methods = len(methods)
                    lines.append(
                        f"   - 时间 {t:.3f}s (置信度 {conf:.0f}%, "
                        f"被 {n_methods} 种方法检测到: {', '.join(methods)})"
                    )
                    if n_methods >= 3:
                        lines.append(f"     → 多种独立方法均检测到该点，强烈提示此处存在编辑痕迹")
                    elif n_methods >= 2:
                        lines.append(f"     → 两种方法交叉验证，此处较可能存在编辑")

            if medium_conf:
                lines.append(f"\n🟡 中等置信度可疑点 ({len(medium_conf)} 个):")
                for p in medium_conf[:5]:
                    lines.append(
                        f"   - 时间 {p['time']:.3f}s (置信度 {p.get('confidence', 0) * 100:.0f}%)"
                    )

            if low_conf:
                lines.append(f"\n🟢 低置信度可疑点 ({len(low_conf)} 个): 可能为误报，建议人工复核")

            # Overall assessment
            lines.append("\n【检测评估】")
            if len(high_conf) >= 3:
                lines.append("• 检测到多个高置信度编辑点，音频极有可能经过后期编辑或拼接处理。")
            elif len(high_conf) >= 1:
                lines.append("• 存在高置信度编辑点，音频可能经过局部编辑，建议进一步人工审听。")
            elif len(medium_conf) >= 3:
                lines.append("• 多个中等置信度可疑点，不排除编辑可能性，建议结合上下文分析。")
            else:
                lines.append("• 可疑点置信度不高，可能为自然语音特征或环境噪声导致的误报。")

        return "\n".join(lines)

    # ── Summary Conclusion ────────────────────────────────────────

    def generate_summary(self) -> str:
        """Generate a comprehensive summary conclusion."""
        lines = [
            "=" * 50,
            "            综 合 分 析 结 论",
            "=" * 50,
            "",
        ]

        # Audio quality assessment
        lines.append("一、音频质量评估")
        lines.append("-" * 30)
        if self._statistics:
            s = self._statistics
            peak = s.get("peak_amplitude", 0)
            rms_db = s.get("rms_db", -100)
            dc = abs(s.get("dc_offset", 0))

            quality_issues = []
            if peak > 0.99:
                quality_issues.append("存在削波失真")
            if rms_db < -40:
                quality_issues.append("信号电平过低")
            if dc > 0.01:
                quality_issues.append("存在直流偏移")

            if not quality_issues:
                lines.append("音频文件质量良好，信号电平正常，未发现技术性缺陷。")
            else:
                lines.append(f"音频文件存在以下质量问题: {'; '.join(quality_issues)}。")
        else:
            lines.append("未进行统计分析，无法评估音频质量。")

        lines.append("")

        # Authenticity assessment
        lines.append("二、真实性评估")
        lines.append("-" * 30)

        authenticity_score = 100  # start from 100, deduct for issues
        authenticity_notes = []

        # Factor 1: Edit detection
        if self._edit_results:
            total = self._edit_results.get("total_detections", 0)
            points = self._edit_results.get("edit_points", [])
            high_conf = [p for p in points if p.get("confidence", 0) > 0.7]

            if total == 0:
                authenticity_notes.append("编辑检测未发现可疑编辑点 (+)")
            elif len(high_conf) >= 3:
                authenticity_score -= 40
                authenticity_notes.append(f"发现 {len(high_conf)} 个高置信度编辑点，高度可疑 (---)")
            elif len(high_conf) >= 1:
                authenticity_score -= 25
                authenticity_notes.append(f"发现 {len(high_conf)} 个高置信度编辑点 (--)")
            else:
                authenticity_score -= 10
                authenticity_notes.append(f"发现 {total} 个低/中置信度可疑点 (-)")
        else:
            authenticity_notes.append("未执行编辑检测")
            authenticity_score -= 5

        # Factor 2: ENF
        if self._enf_results:
            snr = self._enf_results.get("snr_db", 0)
            std_f = self._enf_results.get("std_freq", 1)

            if snr > 10 and std_f < 0.05:
                authenticity_notes.append("ENF 轨迹连续稳定，支持录音连续性 (+)")
            elif snr > 10 and std_f >= 0.05:
                authenticity_score -= 15
                authenticity_notes.append("ENF 轨迹波动较大，存在不连续风险 (-)")
            else:
                authenticity_notes.append("ENF 信号不足，无法评估")
        else:
            authenticity_notes.append("未执行 ENF 分析")

        for note in authenticity_notes:
            lines.append(f"  • {note}")

        lines.append("")

        # Overall verdict
        lines.append("三、综合判定")
        lines.append("-" * 30)

        if authenticity_score >= 90:
            verdict = "高可信度"
            desc = ("综合各项分析指标，该音频文件未发现明显的编辑、拼接或篡改痕迹。"
                    "分析结果支持该音频为原始连续录制的判断。")
            icon = "✅"
        elif authenticity_score >= 70:
            verdict = "中等可信度"
            desc = ("分析发现部分可疑指标，但尚不能确定音频经过编辑处理。"
                    "建议进一步进行人工审听和对比分析。")
            icon = "🟡"
        elif authenticity_score >= 50:
            verdict = "低可信度"
            desc = ("分析发现多项异常指标，音频可能经过编辑或处理。"
                    "建议由专业鉴定人员进行深入分析。")
            icon = "🟠"
        else:
            verdict = "高度可疑"
            desc = ("分析发现显著的编辑/篡改痕迹，音频真实性存疑。"
                    "强烈建议进行专业司法鉴定。")
            icon = "🔴"

        lines.append(f"{icon} 综合判定: {verdict}")
        lines.append(f"   真实性评分: {authenticity_score}/100")
        lines.append(f"\n{desc}")

        lines.append("\n" + "=" * 50)
        lines.append("注: 本分析结果仅供参考，不构成法律意见。")
        lines.append("如需用于司法程序，请委托具有资质的专业鉴定机构。")
        lines.append("=" * 50)

        return "\n".join(lines)

    def generate_all_conclusions(self) -> Dict[str, str]:
        """Generate all analysis conclusions as a dictionary."""
        return {
            "statistics": self.analyze_statistics(),
            "enf": self.analyze_enf(),
            "edit_detection": self.analyze_edit_detection(),
            "summary": self.generate_summary(),
        }
