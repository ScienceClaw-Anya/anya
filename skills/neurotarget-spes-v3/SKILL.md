---
name: neurotarget-spes-v3
description: Read identical label-free baseline, OOF validation, selector and distribution evidence for the SPES v3 comparison.
metadata:
  category: scientific
  tags: [neuroscience, spes, cross-fitting, validation]
---

# Executable conservative scorer evidence

Read the fixed public evidence bank. The Transformer is the explicit default.
No clinical labels, held-out patient identities or held-out performance are available.
All numerical and agent policies use the same evidence and eight query slots.

## Parameters
- `--project`: repository root.
- `--public-case`: tools.json under outputs/spes_agent_v3/cases.
- `--operation`: baseline, selector, validation or distribution.
- `--format json`: structured output.

## Example
`python {baseDir}/scripts/evidence.py --public-case CASE_TOOLS_JSON --operation validation --format json`
