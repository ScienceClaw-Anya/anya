---
name: neurotarget-spes-v4
description: Read an observation-gated public SPES acquisition snapshot without accessing hidden labels or future responses.
metadata:
  category: neuroscience
  output_format: json
---

# Diagnostic SPES acquisition observations

Execute `scripts/observe.py` with `--project`, `--state-path` and `--format json`.
Only public snapshots below `outputs/spes_agent_v4/active/trajectories/` are
accepted. No waveform data, model checkpoint, judge or outcome file is opened.
Return the supplied geometry and the responses already observed in this replay.
This is retrospective research, not a clinical stimulation recommendation.
