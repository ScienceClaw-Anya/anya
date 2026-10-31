# Diagnostic stimulation selection: method and limits

This development experiment asks whether feedback-guided selection improves a
fixed response-based localization ranking within the same stimulation budget.
It evaluates experimental selection; it does not optimize therapeutic stimulation.

## Frozen comparison

The default protocol is `config/diagnostic-benchmark.json`: one virtual patient
(`sub-003`), 16 geometry-selected candidate contact pairs, 55 seconds per condition,
simulation seed 42, eight unique stimulation queries per strategy, and one common
no-stimulation calibration condition. Candidates are chosen without generator labels.
The physical waveform is the released waveform at its original amplitude.

The complete bank is a computational cache. A policy receives only its selected
entries. Every query costs one experiment, including cache hits. The 17 precomputed
conditions are reported separately from the eight conditions revealed to each
policy. Unselected outcomes are not used to rank or preselect experiments.

The collective uses `structure`, `dynamics`, `evidence` and `critic` in round-robin
order, twice each. Each step records a prediction before observation and a reflection
after observation. The shared board contains prior proposals, predictions, compact
SEEG observations and reflections. The single agent sees the same information with
one generalist profile. In Gemini mode both policies use 16 calls. Random selection
uses the same eight-query cap and the same downstream localizer, with no LLM calls.

The frozen protocol, response-bank hash and implementation hashes are saved before
strategy execution. This is a local protocol freeze, not an externally registered
clinical or statistical preregistration. Invalid model outputs stop explicitly;
they do not become valid measurements or silently fall back to a demo policy.
Model-call journals and run-status files retain debugging evidence. Setup failures
are accounted separately from completed matched-budget policy runs.

## Shared numerical localizer

The localizer receives only the public gain geometry and sensor covariance after
the released stimulation ends (7 seconds for this patient). The no-stimulation
control uses the same initial state, model parameters, noise seed and observation
window. Source traces are never used by the localizer.

1. Normalize each region's gain column to unit Euclidean norm, producing `G`.
2. Form `R = inverse(G.T @ G + 0.1 * I) @ G.T`.
3. Estimate regional response power as the diagonal of `R @ C @ R.T`, where `C`
   is the observed sensor covariance. Apply the identical inverse to the control.
4. Divide excess estimated power by control estimated power, with a small fixed
   numerical floor; clip negative excess to zero and take `log1p`.
5. Aggregate each region's maximum response evidence across queried experiments.

The ridge value, floor rule and aggregation were fixed before scoring strategies.
This is a heuristic for diagnostic response ranking. It is not full VEP Bayesian
inversion, a validated seizure detector, or a calibrated EZ posterior. Gain
normalization and a control do not remove inverse ambiguity, electrode-coverage
bias, nonlinear model mismatch or the possibility of strong non-epileptogenic
responses. A weak result remains a weak result; the metric is not a success label.

The legacy catalog field `spatial_spread` is actually the sum of normalized
stimulation weights. It has no millimeter unit. Diagnostic prompts rename it
`profile_mass` and explicitly state its dimensionless meaning. Coordinates alone
use millimeters. Simulated power remains in model units squared.

## Hidden reference and metrics

Only after all policy decisions are complete does the judge read the stimulation
generator parameters. For the verified two-level `sub-003` stimulation parameter
map, positives are the elevated-x0 subgroup, with an independent check that it is
exactly the subgroup with accumulation thresholds below the maximum threshold.
The evaluator rejects another patient/parameter structure rather than inventing
a universal threshold. The five positive regions are a **generator-defined
susceptibility reference**, not a clinical EZ gold standard or necessarily the
complete spontaneous-seizure EZ hypothesis.

All 162 regions are evaluated. Primary: average precision (AP). Secondary: AUROC,
Recall@10, Precision@10 and trapezoid-averaged AP over query counts 0 through 8.
AP uses threshold-grouped ties; AUROC uses average ranks; top-k scores use expected
hits when a tied group crosses the boundary. A completely tied ranking has AP
equal to positive prevalence and AUROC 0.5, without an atlas-order advantage.

Twenty random-policy seeds give a reference distribution on this same frozen
patient and noise realization. They are not twenty independent patients or a
confidence interval for clinical performance. One run of each Gemini policy
cannot establish superiority. A no-sharing ablation and a better validated
inference model are further work, not completed results of this comparison.

## Real data

`ds003029` contributes a separate passive contact-level check against clinical
SOZ annotations. `ds004370` contributes recorded-stimulation response extraction.
No new stimulation response can be inferred from a passive recording or obtained
for an unperformed pair merely by replaying an existing dataset. Clinical SOZ,
resection masks, structural lesions and generator labels are distinct references.
See `docs/real-datasets.md` for resource roles, restrictions and provenance.

## Compute

The response bank runs on CPU. Four workers were used on the local i9-14900K
workstation with approximately 192 GiB physical memory. The first 17-condition
55-second bank took approximately 1,001 seconds of wall time. Gemini decisions run
on Google's API. Additional GPUs are unnecessary for this numerical benchmark.
The full cached arrays occupy about 1.74 GB; the 17 covariance files occupy about
3.88 MB. The existing 1 GiB cloud result bucket is too small for all of these raw
arrays; this run kept them on the workstation and created no additional cloud resource.
Independent patients and simulation-noise seeds can later be batched on Nebius
CPU workers. Large EEG representation learning is a different compute workload.

Source context: [VEP dataset](https://doi.org/10.25493/XWG2-S8X),
[Dollomaja et al., 2025](https://doi.org/10.1371/journal.pcbi.1012911),
[Wang et al., 2025](https://doi.org/10.1038/s43588-025-00841-6).
