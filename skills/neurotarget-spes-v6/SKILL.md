---
name: neurotarget-spes-v6
description: Return public observations for the feedback-matched diagnostic proposal and review study.
metadata:
  category: scientific
---

# Feedback-matched SPES observations

Read an already gated public snapshot with `scripts/observe.py --project ROOT
--state-path SNAPSHOT --format json`. Only snapshots under
`outputs/spes_agent_v6/trajectories/` are allowed. Clinical labels, patient IDs,
future responses and outcome files are rejected. This is retrospective research.
