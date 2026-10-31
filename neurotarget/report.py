from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_report(project: Path, results: list[dict], events=None, name="minimum", agent_mode="demo", runtime="neurotarget"):
    if not results:
        raise ValueError("Cannot create an empty report")
    report_dir = project / "outputs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for result in results:
        with np.load(Path(result["output_dir"]) / "timeseries.npz") as data:
            t, source = data["time_seconds"], data["source_signal"]
            rms = np.sqrt(np.mean((source - source[:min(100, len(source))].mean(axis=0)) ** 2, axis=1))
            axes[0].plot(t, rms, label=result["target_id"], alpha=.85)
            axes[1].plot(t, data["accumulation"].max(axis=1), label=result["target_id"], alpha=.85)
    axes[0].set(ylabel="Source RMS (model units)", title="One patient, different stimulation locations")
    axes[1].set(xlabel="Time (seconds; released VEP time scaling)", ylabel="Maximum accumulation m")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    png = report_dir / f"{name}_comparison.png"
    figure.savefig(png, dpi=150)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150)
    plt.close(figure)
    embedded = base64.b64encode(buffer.getvalue()).decode()
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in [
            r["target_id"], f'{r["summary"]["post_source_variance"]:.5g}',
            r["summary"]["threshold_crossing_regions"],
            f'{r["wall_seconds"]:.1f}', r["spec"]["seed"]]) + "</tr>"
        for r in results)
    event_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(e.get(k, '')))}</td>"
        for k in ["event", "agent", "target_id", "reason"]) + "</tr>" for e in (events or []))
    agent_note = ("本页未运行 agent。" if not events else
                  "Agent 使用 ScienceClaw 原生 Gemini 客户端提出实验，数值结果来自 VEP 模拟器。" if agent_mode == "gemini" else
                  "Multi-agent 演示模式使用确定性的专业策略，不调用大模型。")
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroTarget Swarm | {html.escape(name)}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;background:#f3f6fa;color:#182739;margin:0}}
main{{max-width:1080px;margin:auto;padding:36px 24px}}h1{{font-size:36px;margin-bottom:4px}}
.sub{{color:#526477}}section{{background:white;padding:24px;margin:20px 0;border-radius:14px;border:1px solid #dce4ee}}
.notice{{border-left:5px solid #c08324;background:#fff8e8}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #e1e7ef;vertical-align:top}}.scroll{{overflow:auto}}
code{{overflow-wrap:anywhere}}a{{color:#12629b}}@media(max-width:600px){{main{{padding:20px 12px}}section{{padding:16px}}h1{{font-size:28px}}}}
</style><main><p class="sub">ScienceClaw · VEP sub-003 · offline research prototype</p>
<h1>NeuroTarget Swarm</h1><p>公开患者来源连接组上的刺激位置比较</p>
<section class="notice"><b>研究范围</b><p>这里是诊断性刺激的现象学模拟，不是治疗建议。
作者代码的耦合索引由 [0] 显式修正为 [0,3,6]，避免三个通道的索引越界。
尚未验证与论文发布运行逐点一致。累积变量越阈值也不等于经过临床验证的发作标签。</p>
<p>{html.escape(agent_note)} 当前页面比较模拟反应；尚未完成盲测致痫区定位 benchmark。</p></section>
<section><h2>刺激位置的实际模拟输出</h2><img alt="不同刺激位置的源活动和累积变量随时间变化" src="data:image/png;base64,{embedded}">
<div class="scroll"><table><thead><tr><th>刺激电极对</th><th>刺激后源方差</th><th>累积变量越阈值脑区数</th><th>首次计算秒数</th><th>随机种子</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="sub">模型：{html.escape(results[0]['model_variant'])}。缓存命中会复用相同参数和种子的计算结果。</p></section>
<section><h2>Agent 实验记录</h2><div class="scroll"><table><thead><tr><th>事件</th><th>Agent</th><th>电极对</th><th>依据</th></tr></thead><tbody>{event_rows or '<tr><td colspan="4">此页面是最小模拟，未启用 agent。</td></tr>'}</tbody></table></div></section>
<section><h2>数据与可复现性</h2><p>数据 DOI：<a href="https://doi.org/10.25493/XWG2-S8X">10.25493/XWG2-S8X</a>，CC BY 4.0。
方法：<a href="https://doi.org/10.1371/journal.pcbi.1012911">Dollomaja et al., PLOS Computational Biology (2025)</a>。
连接矩阵和刺激投影的顺序、波形、时间缩放均取自公开数据。下载清单保存文件 SHA-256，运行目录保存参数、模型版本和数组。</p></section></main></html>'''
    target = report_dir / f"{name}.html"
    target.write_text(doc, encoding="utf-8")
    (report_dir / f"{name}.json").write_text(json.dumps({"results": results, "events": events or [],
        "execution": {"runtime": runtime, "agent_mode": agent_mode if events else None}}, indent=2), encoding="utf-8")
    return target
