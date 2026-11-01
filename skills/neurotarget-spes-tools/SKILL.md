---
name: neurotarget-spes-tools
description: Execute a fixed blinded SPES signal, measured-network, validation-reliability or numerical-fusion tool. Clinical labels are excluded.
metadata:
  category: scientific
  tags: [neuroscience, spes, localization, validation]
---

# Measured SPES analysis tools

Read the frozen public tool bank generated from measured responses and validation-trained models. The bank contains no held-out labels or patient identity. Signal and network outputs are descriptive summaries, not causal or therapeutic conclusions.

## Parameters
- `--project`: repository root.
- `--public-case`: tools.json inside outputs/spes_agent_v2/cases.
- `--operation`: signal, network, reliability or fusion.
- `--format json`: structured output.

## Example
`python {baseDir}/scripts/spes_tools.py --public-case CASE_TOOLS_JSON --operation signal --format json`
