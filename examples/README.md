# Portable run evidence

`diagnostic_benchmark_summary.json` contains the executed Gemini diagnostic
comparison, per-experiment predictions and reflections, matched-budget metrics,
random-policy reference distribution and separate real-recording checks.
`diagnostic_benchmark_report.html` is a self-contained English report.
`diagnostic_protocol.json` is the frozen comparison definition. Its canonical JSON
SHA-256 in the summary is independent of Git line-ending normalization.

The formal comparison used 16 Gemini calls for the collective and 16 for the single
agent. Two earlier setup calls failed reflection-schema validation and are recorded
separately; they do not count as valid policy experiments or improve the scored
ranking. There is no silent demo fallback. Model-generated reflection assessments
are hypotheses, not independently verified conclusions.

These are one-patient development results. Final Recall@10 is zero for all three
policies. The current experiment does not demonstrate successful localization or
a multi-agent advantage. Clinical SOZ contact scores and recorded SPES extraction
checks are separate tasks and are not pooled with virtual region-level accuracy.

Raw arrays, clinical source tables, model-call journals, secrets, machine names and
absolute local paths are not included here. Reproduce the export after a run with:

```powershell
.\.venv\Scripts\python.exe scripts/export_diagnostic_example.py --results outputs/diagnostic/SESSION/benchmark.json
```

Older example files document the earlier four-experiment native integration smoke
run. They are historical evidence and are not diagnostic-accuracy benchmarks.
