---
name: neurotarget-spes-v7
description: Read public, observation-gated data for the schema-constrained diagnostic acquisition experiment.
metadata:
  category: scientific
---

# Schema-constrained acquisition observations

Use `scripts/observe.py --project ROOT --state-path PUBLIC_SNAPSHOT --format
json`. Only snapshots below `outputs/spes_agent_v7/trajectories/` are accepted.
No patient identities, clinical labels, outcomes or future responses are served.
This tool supports retrospective research only.
