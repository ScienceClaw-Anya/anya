"""English, self-contained diagnostic benchmark report."""
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


def write_benchmark_report(project: Path, document):
    escape = lambda x: html.escape(str(x))
    budget = document["protocol"]["experiment_budget_per_strategy"]
    directory = project / "outputs/reports"; directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for trial in document["trials"]:
        for axis, metric, title in zip(axes, ["average_precision", "auroc", "recall_at_10"],
                                       ["Average precision (primary)", "AUROC", "Recall at 10 regions"]):
            axis.plot(range(budget+1), [v[metric] for v in trial["learning_curve"]], marker="o", label=trial["strategy"])
            axis.set(title=title, xlabel="Charged stimulation queries", ylim=(-.02, 1.02))
            axis.grid(alpha=.2)
    random = document["random_reference"]
    axes[0].fill_between(range(budget+1), random["p05_ap_curve"], random["p95_ap_curve"], color="grey", alpha=.15,
                         label="Random-policy 5th–95th percentiles")
    axes[0].plot(range(budget+1), random["mean_ap_curve"], color="grey", ls="--", label="Random-policy mean")
    axes[0].axhline(document["judge_only_reference"]["positive_count"] / document["judge_only_reference"]["region_count"],
                    color="black", ls=":", label="Positive prevalence")
    axes[0].legend(fontsize=8)
    ap_values = [v["average_precision"] for t in document["trials"] for v in t["learning_curve"]]
    axes[0].set_ylim(0, min(1.02, max(.1, max(ap_values + random["p95_ap_curve"]) * 1.12)))
    buffer = io.BytesIO(); figure.savefig(buffer, format="png", dpi=145); plt.close(figure)
    (directory / f"diagnostic_{document['protocol']['mode']}_comparison.png").write_bytes(buffer.getvalue())
    plot = base64.b64encode(buffer.getvalue()).decode()
    rows = []
    for trial in document["trials"]:
        m = trial["final_metrics"]
        values = [trial["strategy"], trial["queries_charged"], trial["llm_calls"],
                  f"{m['average_precision']:.3f}", f"{m['auroc']:.3f}", f"{m['recall_at_10']:.3f}",
                  f"{trial['mean_learning_curve_average_precision']:.3f}"]
        rows.append("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>")
    logs = []
    for trial in document["trials"]:
        cards = []
        for item in trial["experiments"]:
            p, r = item["proposal"], item["reflection"]
            observation = item["observation"]
            measured = ", ".join(f"{x['channel']}: {x['control_ratio']:.3g}× control" for x in observation["top_channels_by_control_ratio"][:4])
            ranked = ", ".join(x["region"] for x in item["ranking_after"]["ranked_regions"][:3])
            cards.append(f"<article><h3>Experiment {item['experiment']} · round {item['round']} · {escape(item['role'])} → {escape(p['target_id'])}</h3>"
                         f"<p><b>Rationale:</b> {escape(p['reason'])}</p><p><b>Prediction recorded before measurement:</b> {escape(p['expected_observation'])}</p>"
                         f"<p><b>Measured SEEG ratios:</b> {escape(measured)}. Mean power: {observation['mean_sensor_power']:.4g} model units squared.</p>"
                         f"<p><b>Updated numerical ranking, first three:</b> {escape(ranked or 'No positive response evidence')}</p>"
                         f"<p><b>Reflection ({escape(r['assessment'])}):</b> {escape(r['explanation'])}</p>"
                         f"<p><b>Revised hypothesis:</b> {escape(', '.join(r['revised_hypothesis']) or 'Unspecified')}</p>"
                         f"<p><b>Next test:</b> {escape(r['next_test'])}</p></article>")
        logs.append(f"<details><summary>{escape(trial['strategy'])}: predictions, measurements and reflections</summary>{''.join(cards)}</details>")
    reference = document["judge_only_reference"]
    real = project / "outputs/real_data/validation.json"
    real_section = "<p>Real-data adapters are available; no real-data analysis has been attached to this report.</p>"
    if real.exists() or document.get("real_data_validation"):
        checked = document.get("real_data_validation") or json.loads(real.read_text(encoding="utf-8"))
        passive, stimulation = checked["passive_localization"], checked["recorded_stimulation"]
        pm = passive["metrics"]
        stimulus_rows = "".join("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in
            [r["stimulation_site"], r["pulses"], r["eligible_response_contacts"],
             ", ".join(x["contact"] for x in r["top_responses"][:3])]) + "</tr>" for r in stimulation["responses"])
        real_section = (f"<h3>Recorded seizure: {escape(passive['dataset'])}, {escape(passive['subject'])}</h3>"
            f"<p>{escape(passive['method'])}. {passive['eligible_contacts']} eligible contacts; {passive['reference_contacts']} clinical SOZ contacts. "
            f"Average precision: <b>{pm['average_precision']:.3f}</b>; AUROC: <b>{pm['auroc']:.3f}</b>; Recall@10: <b>{pm['recall_at_10']:.3f}</b>. "
            f"This passive, contact-level result is not an active stimulation-selection comparison or a region-level VEP result.</p>"
            f"<h3>Recorded electrical stimulation: {escape(stimulation['dataset'])}</h3>"
            f"<div class='scroll'><table><thead><tr><th>Recorded pair</th><th>Pulses</th><th>Eligible response contacts</th><th>Strongest normalized responses</th></tr></thead><tbody>{stimulus_rows}</tbody></table></div>"
            f"<p>{escape(stimulation['reason_no_accuracy'])}. Stimulus artefacts, bad channels and stimulating contacts are excluded. "
            "Only two recorded anatomical pairs are sampled; no response to unperformed stimulation is inferred.</p>"
            "<details><summary>Real-data provenance, byte-range checks and quality notes</summary><pre>" + escape(json.dumps(checked, indent=2)) + "</pre></details>")
    conclusions = "These one-patient results are exploratory and do not establish a winning policy."
    if all(t["final_metrics"]["recall_at_10"] == 0 for t in document["trials"]):
        conclusions = "None of the three policies retrieved a generator-reference region in its final top ten. This run does not demonstrate successful localization or a multi-agent advantage; improving and validating the shared inference method is a priority."
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anya | Diagnostic stimulation benchmark</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#172b42;background:#f3f6fa;margin:0}}main{{max-width:1180px;margin:auto;padding:32px 20px}}
h1{{font-size:34px}}section,details{{background:white;border:1px solid #dce4ee;border-radius:12px;padding:22px;margin:18px 0}}
.notice{{background:#fff7e6;border-left:5px solid #b87b16}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dde4ed}}
.scroll{{overflow:auto}}article{{border-top:1px solid #dce4ee;margin-top:18px}}summary{{cursor:pointer;font-weight:650}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}a{{color:#17689b}}
</style><main><p>ScienceClaw · {escape(document['protocol']['patient'])} · {escape(document['protocol']['mode'])} mode</p>
<h1>Diagnostic stimulation selection</h1><p>Multi-round specialists, a single agent and a random policy under the same experiment budget.</p>
<section class="notice"><b>Observed result</b><p>{escape(conclusions)}</p><p>This is a one-patient development benchmark against a released stimulation-generator susceptibility hypothesis.
The shared localizer is a SEEG-only response-ranking heuristic, not full Bayesian VEP inversion or a clinically validated diagnostic model. No therapeutic efficacy is measured.
Single realizations of the Gemini policies cannot establish multi-agent superiority. Random repetitions describe policy variability on this same patient, not independent patient replication.</p></section>
<section><h2>Matched-budget results</h2><p>Every strategy receives the same candidate pool, waveform, simulation seed, post-stimulation window, localizer and no-stimulation calibration.
Each queried site costs one experiment even when a cached response is reused. The two nonrandom Gemini strategies use one proposal plus one reflection per experiment.
Reference accuracy is computed only after policy execution and is never returned to agents.</p>
<div class="scroll"><table><thead><tr><th>Strategy</th><th>Queries</th><th>LLM calls</th><th>Average precision</th><th>AUROC</th><th>Recall@10</th><th>Mean AP across budget</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<img alt="Localization learning curves under equal stimulation budgets" src="data:image/png;base64,{plot}">
<p>Higher localization metrics are better. Average precision summarizes precision/recall over ranking thresholds. Recall@10 is the fraction of reference regions in the ten highest-ranked regions.
Ties use threshold-grouped AP, average-rank AUROC and expected hits at the top-k boundary. Positive prevalence is the tied-score baseline, not the expected finite-sample AP of random rankings. Random reference repetitions: {random['repetitions']}.</p></section>
<section><h2>Feedback and collaboration</h2><p>Four roles act in round-robin order. Each records a prediction, observes one SEEG response, updates the common numerical ranking and writes a reflection.
Later agents receive prior proposals, predictions, measured summaries and reflections. The single agent receives the same tools and observation content.
Reflections express model-generated assessments; they do not count as independently verified scientific conclusions. The shared numerical localizer determines all scored rankings.</p>{''.join(logs)}</section>
<section><h2>Hidden reference and scientific limits</h2><p>{escape(reference['definition'])}. Reference regions: {escape(', '.join(reference['labels']))}.
The two-level stimulation parameter subgroup is checked against the released lower-threshold subgroup. It is not a clinical gold standard or the complete spontaneous-seizure EZ hypothesis.
The regularization and aggregation were frozen before strategy scoring. Gain geometry, electrode coverage, inverse ambiguity, model mismatch and one simulation-noise seed constrain interpretation.
All generator source traces, excitability values, accumulation thresholds and reference scores are excluded from agent observations.</p>
<p>The forward model retains the documented coupling-index correction [0,3,6] and author-script initial state; exact paper-run reproduction has not been established.</p></section>
<section><h2>Real-data checks</h2><p>Real recordings provide complementary validation. Passive seizure recordings cannot reveal counterfactual responses to unperformed stimulation.
Clinical SOZ annotations, surgical resections and model-generator labels are distinct references and are not pooled into a single accuracy score.</p>{real_section}</section>
<section><h2>Compute and reproducibility</h2><pre>{escape(json.dumps(document['compute'], indent=2))}</pre>
<p>Frozen protocol SHA-256: {escape(document['protocol_sha256'])}. Session: {escape(document['session'])}.
The full response bank contains {document['compute']['precomputed_conditions']} simulated conditions, including one control; policies only access selected entries.
LLM decisions run through ScienceClaw's native Gemini client using Google's API. The numerical simulator uses CPU.</p>
<p>Dataset: <a href="https://doi.org/10.25493/XWG2-S8X">VEP, CC BY 4.0</a>.
Method context: <a href="https://doi.org/10.1371/journal.pcbi.1012911">Dollomaja et al. (2025)</a>.
See the repository's diagnostic protocol and real-data registry for definitions and source access.</p></section></main></html>'''
    target = directory / f"diagnostic_{document['protocol']['mode']}.html"
    target.write_text(doc, encoding="utf-8")
    (directory / f"diagnostic_{document['protocol']['mode']}.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target
