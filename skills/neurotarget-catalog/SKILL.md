---
name: neurotarget-catalog
description: Inspect public VEP connectome and permitted diagnostic stimulation contact pairs without latent EZ parameters.
---

# VEP stimulation catalog

## Capabilities
- Return existing adjacent electrode pairs, geometry and connectome scores.
- Select a bounded diverse candidate pool without using x0/EZ labels.

## Parameters
- `--project`: NeuroTarget repository root.
- `--patient`: downloaded virtual patient, default sub-003.
- `--limit`: 2 to 179 candidate pairs, default 32.
- `--format json`: machine-readable output.

## Example
`python {baseDir}/scripts/catalog.py --patient sub-003 --limit 32 --format json`

The catalog is a computational research input, not a clinical target recommendation.
