---
name: neurotarget-simulate
description: Run one seeded VEP diagnostic stimulation experiment and return observed SEEG summaries as JSON.
---

# VEP diagnostic stimulation simulator

## Capabilities
- Execute the recorded Epileptor stimulation model variant at an existing electrode pair.
- Return SEEG power summaries, a raw-result file reference and cache provenance.

## Parameters
- `--project`: NeuroTarget repository root.
- `--patient`: downloaded virtual patient, default sub-003.
- `--target-id`: an existing adjacent pair from neurotarget-catalog.
- `--seconds`: >0 and <=55; default 12. Short runs are smoke checks.
- `--seed`: default 42.
- `--format json`: machine-readable output; progress is written to stderr.

## Example
`python {baseDir}/scripts/simulate.py --target-id B2-3 --seconds 12 --format json`

This model is diagnostic, not a validated seizure-suppression controller. The
coupling-index correction is documented in the project README. Agent observations
exclude source-level activity, x0/EZ labels and accumulation-threshold metrics.
