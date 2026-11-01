---
name: neurotarget-spes-v16
description: Read common whitelisted physical-pair anatomy, observed responses and descriptive model scores for independent-proposal research.
metadata:
  category: scientific
---

# Physical-pair common evidence

Use `scripts/observe.py --project ROOT --state-path PUBLIC_SNAPSHOT --format json`.
Only saved public snapshots below `outputs/spes_agent_v16/trajectories/` are
accepted. No clinical contact labels, outcomes or unobserved responses are
served. Every comparison arm receives the same evidence. This retrospective
diagnostic experiment does not deliver physical stimulation.
