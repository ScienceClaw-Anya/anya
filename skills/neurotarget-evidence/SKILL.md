---
name: neurotarget-evidence
description: Retrieve attributed method boundaries from the team's epilepsy virtual-brain literature review.
---

# Epilepsy method evidence

## Capabilities
- Return pinned team-review evidence IDs, repository paths and method limitations.
- Distinguish virtual-patient diagnostic stimulation from therapeutic control.

## Parameters
- `--project`: NeuroTarget repository root.
- `--format json`: structured output.

## Example
`python {baseDir}/scripts/evidence.py --format json`

The tool reads curated local review cards. It does not claim a new independent
literature search or verification of every claim in the source review.
