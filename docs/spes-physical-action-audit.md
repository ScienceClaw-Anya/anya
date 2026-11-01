# Physical action-grain correction and verified replacement

The processed arrays contain compact response sets for individual recording
contacts. **The same array column is not the same physical stimulation pair
across contacts.** This was an implementation error in our adaptation from
contact classification to active selection. The shared-action assumption should
have been checked before the agent experiments.

The audit confirms mixed physical pairs within columns in **35/35 patients**.
Historical v4-v10 subset-selection results remain preserved as **array-column
ablation**. Their interpretation as eight physical stimulations, anatomical
target selection or a stimulation protocol is withdrawn. They cannot establish
whether multi-agent collaboration helps real stimulation selection.

## Evidence and scope

The check matches processed recording coordinates to the public electrode TSV,
then matches each retained response distance to the midpoint of actual recorded
stimulation pairs. The source coordinates match; the provider declares
**fsaverage, millimetres**, not an interchangeable MNI152 registration. Only
geometry, anatomical labels and stimulation fields are read. SOZ, resection,
outcomes and clinical notes are excluded.

Across **114,274 recording-contact / compact-column cells**,
**114,021 (99.7786%)** have one unique anatomical-pair
match within 0.0001 mm. **253 ambiguous cells are excluded**.
There are 0 duplicate assignments.
All 35 mapped patients have at least 41 verified single-condition pairs, and the
largest pool has 93. No pair had multiple recorded parameter tuples in this
metadata check; this does not validate prospective clinical safety.

These counts concern the **first twenty trials actually used by all current
neural fits and evaluations**. An initial full-forty-trial probe found changing
geometry at fixed row/column positions; it is retained as an unresolved probe.
A row-only map must not be extended to those extra trials. The replacement
adapter rejects changing geometry within selected trials.

The full-response classification task does not require a shared physical
action axis. Its patient-separated neural results and static model comparisons
are unaffected by this action-grain finding. It does affect the physical
interpretation of partial-column acquisition and tools trained on that task.

## Stopped work and retained evidence

The v9 outer evaluation stopped after **106
completed trajectories**, leaving **209 canceled
or unfinished**. All recorded choices, observations, metrics, frozen protocols
and the stop lock remain. No partial-cohort winner is reported. Its numerical
tools are not adopted for physical selection.

The ten v10 contact-model fits and training banks are retained, but outer
evaluation was blocked before execution. They used column-ablation prefixes
and require new physical-query training states before active use. No new Gemini
request was made in either version. Historical scores are not silently rewritten.
Interpretation notices are added to the old English reports and READMEs.

## Verified replacement

`PhysicalAcquisition` queries a named anatomical pair and gathers its verified
response from the appropriate, potentially different, compact column at each
recording contact. Unresolved cells remain zero/absent. Only those actual
responses enter model inference. Hidden responses and labels stay private;
the scoring guard requires eight complete physical queries.

The minimum now runs on patient ccepAgeUMCU36: two distinct pairs,
['C01', 'C02'] and ['C02', 'C03'].
The second query needs different compact columns across recording contacts,
demonstrating why the old global-column action was incorrect. This is an
extraction/inference check, with no localization-effect claim.

An actual held-out HAT check compares physically aligned responses with the
identical retained response set in original compact layout. Maximum absolute
score difference is **1.1920929e-07**
(tolerance 1e-05). Reindexing therefore preserves this model's
full-response calculation for the tested patient. Full predictions are not
exposed to agents in the active task.

## Next valid experiment

Freeze new physical-pair numerical controls before an agent comparison. Rebuild
development-patient query-value or contact-posterior training states using the
verified maps; old column-utility models are unsuitable. The agent interface can
now share real pair identities and whitelisted anatomy, in addition to measured
waveforms and trained model scores. Independent proposals, feedback timing and
total model/measurement budgets must match single-agent controls; include direct
numerical selection with the same tools.

Model/version changes or more debate do not repair an invalid action definition.
No multi-agent superiority or treatment benefit has been established. All new
evaluation on this repeatedly inspected cohort remains exploratory; a stronger
claim needs independent confirmation.

Source context: [public processed author arrays and model code](https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT)
and [OpenNeuro ds004080 v1.2.2](https://openneuro.org/datasets/ds004080/versions/1.2.2).
Per-file metadata, array, action-map and source-code hashes are retained locally
under `outputs/spes_physical/`. Raw data and private credentials stay outside Git.
All mapping/model checks ran locally on the workstation; no cloud resource or
model API request was used.
