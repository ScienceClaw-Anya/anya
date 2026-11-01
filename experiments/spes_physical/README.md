# Verified physical stimulation actions

An action-grain audit found **35/35** processed patients have compact array
columns that mix anatomical stimulation pairs across recording contacts.
Historical v4-v10 active results remain preserved as column-ablation results;
their physical-stimulation interpretation is withdrawn. See the
[English audit report](../../docs/spes-physical-action-audit.md).

The replacement maps each recording contact's response column to a real
anatomical pair using electrode coordinates, recorded stimulation metadata and
distance matching within 0.0001 mm. SOZ, resection, outcomes and clinical notes
are excluded. Ambiguous matches are excluded, never guessed or silently pooled.
Only the first twenty trials used by the current neural models are mapped.
Extending a row-only map to forty trials can change the underlying geometry
and is rejected by the observation adapter.

`PhysicalAcquisition` reveals the different source columns needed for **one
common physical pair** across contacts. It retains eight-query limits, hides
future waveforms and labels, and uses the unchanged patient-held-out neural
models. Candidates have one recorded parameter tuple; the environment makes
no treatment, prospective-safety or physiological state-carry claim.

```powershell
python -m unittest experiments.spes_physical.test_mapping -v
python -m experiments.spes_physical.audit --project . --output outputs/spes_physical
python -m experiments.spes_physical.smoke --project . --output outputs/spes_physical
python -m experiments.spes_physical.verify_remap --project . --output outputs/spes_physical
```

The smoke is a two-position extraction/inference check, not localization-effect
evidence; it refuses to overwrite a recorded run. No model API call or cloud
resource is required. Raw waveforms, subject action maps and weights stay outside
Git. Portable audit/smoke summaries contain no clinical labels or API secrets.
