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
    if not events:
        agent_note = "No agents were run for this report."
    elif agent_mode == "gemini" and runtime == "scienceclaw":
        agent_note = "Agents propose experiments through ScienceClaw's native Gemini client. Numerical results come from the VEP simulator."
    elif agent_mode in {"llm", "gemini"}:
        agent_note = "Agents propose experiments through the configured language model. Numerical results come from the VEP simulator."
    else:
        agent_note = "Multi-agent demo mode uses deterministic specialist policies without language-model calls."
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroTarget Swarm | {html.escape(name)}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;background:#f3f6fa;color:#182739;margin:0}}
main{{max-width:1080px;margin:auto;padding:36px 24px}}h1{{font-size:36px;margin-bottom:4px}}
.sub{{color:#526477}}section{{background:white;padding:24px;margin:20px 0;border-radius:14px;border:1px solid #dce4ee}}
.notice{{border-left:5px solid #c08324;background:#fff8e8}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #e1e7ef;vertical-align:top}}.scroll{{overflow:auto}}
code{{overflow-wrap:anywhere}}a{{color:#12629b}}@media(max-width:600px){{main{{padding:20px 12px}}section{{padding:16px}}h1{{font-size:28px}}}}
</style><main><p class="sub">ScienceClaw · VEP sub-003 · offline research prototype</p>
<h1>NeuroTarget Swarm</h1><p>Stimulation-site comparison using a public VEP connectome</p>
<section class="notice"><b>Research scope</b><p>This is a phenomenological simulation of diagnostic stimulation, not a treatment recommendation.
The author model's coupling indices are explicitly corrected from [0] to [0,3,6] to prevent out-of-bounds access across three channels.
Point-by-point agreement with the released paper run has not been established. Accumulation threshold crossings are not clinically validated seizure labels.</p>
<p>{html.escape(agent_note)} This report compares simulated responses; a blinded epileptogenic-zone localization benchmark has not been completed.</p></section>
<section><h2>Simulated responses by stimulation site</h2><img alt="Source activity and accumulation over time for different stimulation sites" src="data:image/png;base64,{embedded}">
<div class="scroll"><table><thead><tr><th>Stimulation contact pair</th><th>Post-stimulus source variance</th><th>Regions crossing the accumulation threshold</th><th>Original compute time (s)</th><th>Random seed</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="sub">Model: {html.escape(results[0]['model_variant'])}. Cache hits reuse results computed with the same parameters and seed.</p></section>
<section><h2>Agent experiment log</h2><div class="scroll"><table><thead><tr><th>Event</th><th>Agent</th><th>Contact pair</th><th>Rationale</th></tr></thead><tbody>{event_rows or '<tr><td colspan="4">This is the minimum simulation; no agents were enabled.</td></tr>'}</tbody></table></div></section>
<section><h2>Data and reproducibility</h2><p>Dataset DOI: <a href="https://doi.org/10.25493/XWG2-S8X">10.25493/XWG2-S8X</a>, CC BY 4.0.
Method: <a href="https://doi.org/10.1371/journal.pcbi.1012911">Dollomaja et al., PLOS Computational Biology (2025)</a>.
The connectivity and stimulation-projection ordering, waveform and time scaling follow the public release. The download manifest records file SHA-256 hashes; run directories store parameters, model versions and numerical arrays.</p></section></main></html>'''
    target = report_dir / f"{name}.html"
    target.write_text(doc, encoding="utf-8")
    (report_dir / f"{name}.json").write_text(json.dumps({"results": results, "events": events or [],
        "execution": {"runtime": runtime, "agent_mode": agent_mode if events else None}}, indent=2), encoding="utf-8")
    return target
