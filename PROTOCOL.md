# Protocol: virtual resection and retrospective outcome prediction

Version: 0.1, 2026-11-01. Status: **design; not a frozen confirmatory protocol**.
Resolve the explicitly listed implementation choices on development data, then
commit an immutable run manifest before exposing held-out results. These gates
are methodological checks; they do not require routine user reconfirmation.

## P0. Primary hypothesis and estimand

Under the same observations, localization estimate, admissible intervention set
and resource constraints, a planner using patient-specific network dynamics
selects resections with greater seizure-burden reduction than a predefined
SOZ-guided rule on unseen virtual patients.

The unit of independent evaluation is the patient, not the seizure, random seed,
simulator trajectory, cortical parcel or agent call. Alternative EZ hypotheses
and seizures for one person stay together in the split.

For patient p and action a, define a development-frozen ictal-state detector
I(p, r, t, a). Define burden over a fixed post-transient horizon T as:

    B(p, a) = sum over regions r of w(p, r) * integral over T I(p, r, t, a) dt
    R(p, a) = 1 - B(p, a) / B(p, no_action)

Weights w are verified tissue volumes if available. Use the original intact
region set and weights for all actions; removed tissue contributes zero ictal
activity but is charged as intervention cost. Do not renormalize by remaining
tissue or discard difficult surviving regions. Pair no-action and intervention
seeds/scenarios. Define a baseline-burden eligibility threshold on development
data; zero/near-zero denominators are handled by the frozen rule and reported.

Primary patient difference:

    Delta(p) = R(p, personalized_selection) - R(p, SOZ_guided_selection)

Report mean paired difference, patient-level distribution and a 95% patient
bootstrap interval. Preserve all prespecified failures. The practical effect
threshold, bootstrap procedure and formal decision rule must be frozen before
test evaluation. Positive findings must exceed that threshold with uncertainty
reported; a failed or uncertain comparison remains a valid result.

## P1. Action and budget semantics

Start with whole-parcel virtual resection as an explicitly coarse model. Define
the operator in equations and code: remove a parcel's dynamic contribution and
incident coupling according to the chosen simulator's connectivity convention.
Verify orientation, units and the handling of removed nodes. Retain intact-model
coupling normalization across actions; otherwise normalization changes can become
an unintended second intervention. Report any justified exception in advance.

Inspect whether VEC provides or permits reconstruction of individual parcel
volumes. If not, do not invent them. A parcel-count or reference-volume pilot is
allowed but must be labeled a proxy-cost benchmark; it cannot substantiate a
claim of smaller patient tissue volume. Select the formal cost definition only
after this audit and preserve that limitation in every report.

All policies share the same candidate set, anatomical feasibility rules, maximum
budget and treatment-cost calculation. Use matched costs or explicit budget bins
when exact size matching is impossible. Do not compare low-cost SOZ choices with
high-cost network choices as if costs were equal. Report realized costs and the
benefit/cost frontier. Contiguity and atlas constraints are approximations to
feasibility, not proof of surgical accessibility or cognitive safety.

Disconnection has a separate edge/path cost. Stimulation has waveform, amplitude,
location, timing and field constraints. Neither is silently substituted for a
resection, and neither shares a tissue-volume budget without a justified mapping.

## P2. Comparator policies and information boundaries

Required comparators:

1. SOZ-guided targeting: a frozen ranking-to-action rule with deterministic
   tie-breaking and the same feasibility and budget constraints.
2. Simple numerical network heuristic, e.g. a prespecified centrality rule.
3. Generic-network planner: same planning machinery and localization information,
   using a training-only reference network. A shuffled-patient network is a
   sensitivity control, not an independent dataset.
4. Personalized-network planner using an individual model.
5. Established virtual-resection method where inputs and implementation are
   compatible; document limitations rather than relabeling an unrelated method.

Give the primary SOZ-guided and personalized planners the identical localization
estimate. Map contact-level SOZ to atlas regions explicitly; do not silently
equate contacts, parcels and source excitability. If a simulated onset estimate
is used, name its derivation and distinguish it from generator EZ parameters.

The evaluator may possess hidden generating parameters. The realistic planner
may use only the specified recordings, anatomy and training/development assets.
Known-parameter runs are allowed as labeled oracle/development controls; they
do not establish performance under incomplete patient information. Separate
planner inputs and evaluation assets in code and manifests.

For a manageable candidate set, evaluate all admissible actions to compute regret
relative to the best candidate in that set. Call this a finite-set oracle, not
the globally optimal clinical intervention. Its outcomes cannot guide the tested
planner. Match simulation/search budgets across competing planning algorithms;
report compute alongside performance and distinguish model value from more search.

## P3. Patients, model fitting and evaluation

Audit all eligible VEC cases and prior exposure before assigning patient-level
splits. Include Anya's previously exercised sub-003 in development. Choose and
record the remaining development/test allocation deterministically before
examining intervention outcomes. With 30 eligible unique patients, a reasonable
initial allocation is 6 development / 24 untouched evaluation cases; finalize
eligibility and sample-size limitations before freezing, not after seeing scores.

All model choices, seizure detectors, costs, constraints, fitting tolerances,
hyperparameters, candidate generation, seeds, stopping rules and exclusions are
fixed using development cases. Within a held-out patient, prespecified fitting to
permitted pretreatment recordings is personalization, not population training.
Withhold separate recordings/conditions for dynamics prediction checks.

First reproduce untreated dynamics and assess withheld propagation/response
features. Then evaluate intervention choices on fresh noise/initial-condition
realizations and parameter uncertainty. Identical generator/planner assumptions
are a best-case test; add mismatch in coupling, excitability, observations or a
structurally different model before claiming robustness beyond that setting.
Do not let an optimizer validate itself only on its own predicted objective.

After a held-out result has informed a design change, label subsequent reuse as
exploratory or introduce a genuinely untouched evaluation set. More trajectories
do not repair patient-level data reuse or enlarge the independent cohort.

## P4. Secondary endpoints

- Remaining onset-region activity and recruited non-onset volume/duration,
  separately: blocking spread is not necessarily seizure elimination.
- Cost required to reach a development-defined burden-reduction threshold.
- Finite-candidate regret and intervention ranking agreement.
- Fraction of uncertainty scenarios meeting the predefined control objective.
- Effect of personalization and of localization quality in separate ablations.
- Held-out untreated dynamics prediction and uncertainty calibration.
- Runtime, simulator calls, failure rate and reproducibility.

Do not interpret parcel-level silence or a favorable network statistic as proven
long-term seizure freedom. Brain-function/cognitive risks are outside the primary
model unless a separately justified data/model component measures them.

## P5. Retrospective outcome study

Audit IDEAS I/II first, and compatible iEEG/outcome cohorts as alternatives.
Verify patient identity, available modalities, preoperative timing, resection
description, outcome definition and follow-up. Do not assume that all imaging
cases have connectivity, SOZ labels, SEEG and usable outcomes. Do not join distinct
cohorts into fictional multimodal patients.

Target a single consistently encoded postoperative endpoint at a fixed horizon
(preferably 12 months if the metadata support it). Preserve distinctions between
complete seizure freedom and freedom from disabling seizures, and between Engel
and ILAE categories. Define missing-data and censoring rules before evaluation.

Predict the outcome of the operation actually performed. The observed resection
mask may describe that action, but cannot define pretreatment SOZ or leak outcome
information into fitted excitability. Using postoperative anatomy to specify the
delivered action is a retrospective analysis, not a prospective planning test.

Compare patient-isolated/nested validation against outcome prevalence, clinical
features where present, resection location/volume and a simple network baseline.
Use training-only preprocessing and calibration. Report AUROC, Brier score,
calibration and patient-level uncertainty; use site-held-out validation if feasible.
Report eligibility counts and overlap with previous cohorts. This study verifies
prediction for observed surgeries, not causal superiority of unperformed options.

## P6. Freeze and audit record

Before formal evaluation, commit a run manifest containing source revisions,
input hashes, subject split, intervention operator, budgets, ictal detector,
outcomes, candidates, policy definitions, fit/evaluation seeds, statistical rule,
runtime limits and expected outputs. Record whether the phase is engineering,
development, frozen evaluation or exploratory follow-up.

Validate meaningful invariants: null intervention; removed-node/coupling behavior;
cost accounting; unchanged normalization; mapping/permutation invariance; hidden
label exclusion; patient separation; seed pairing; reproducible metrics. Have
an independent calculation replay saved results. Numerical replay is not proof
of model validity. Preserve negative results, failed runs and versioned amendments.

## Development execution amendment, 2026-11-01 (not a confirmatory freeze)

The user moved experiments to the existing GCP host before any local simulation.
All numerical tests and trajectories run in the isolated GCP research directory,
with one worker/thread at reduced priority. The initial pilot retains the
60-minute experiment wall-time ceiling and 2-GiB dataset acquisition ceiling.

`manifests/split-v1.json` reserves 6 development and 24 held-out patients before
intervention outcomes. `manifests/pilot-v1.json`, committed in `8d24247` before
interventions, defines a one-patient engineering/development comparison with
sub-010 used only as an anatomical donor. Six common single-parcel candidates
are derived from pretreatment simulated-SEEG onset proxies and anatomy. Cost is
exactly one parcel, a proxy; the available volume file's physical units are not
yet independently verified. No clinical SOZ annotation is substituted for the
proxy, and neither is equated with generator excitability.

The pilot uses the released 6D Epileptor parameters, zero-delay difference
coupling, dt=0.05, period=1, an 8,000-model-unit horizon and a 1,000-unit transient.
A development detector thresholds the 100-sample mean of x1 at -1.0; -1.2 and
-0.8 are sensitivity analyses. Time remains model units. Missing release noise
correlation/seed metadata is handled by a declared development assumption:
ntau=1 as in inspected upstream spontaneous-seizure code, fresh paired seeds.
Current TVB 2.10.0 is pinned and numerically cross-checked, not asserted to be
the exact unreleased original environment. No bitwise original-seizure claim.

Both dynamical planners are known-parameter ORACLE controls. They share the
same generator parameters and candidate-call budget, varying only the network.
A one-donor network is a sensitivity control, not a well-estimated population
reference. Their comparison against observation-only policies has an information
advantage and cannot test P0's equal-information primary hypothesis. The
observation-only policies are the SOZ proxy and a localization-weighted outgoing
strength heuristic. There is no fitted realistic dynamical planner yet.

Planning uses seed 101. The selected actions are saved before evaluating all
candidates with seed 201, and seed 202 with coupling multiplied by 1.2 and x0
shifted by +0.05. Those two scenarios are not independent patients or a confidence
interval. Candidate regret is finite-set regret only. No held-out cohort is run.

The legacy diagnostic names `onset_generator_burden` and
`nononset_generator_burden` in raw pilot rows actually partition x0 > -2.1 versus
x0 <= -2.1. These are susceptibility partitions, not verified initiation versus
propagation. Exported summaries rename these fields. A proper initiation/spread
endpoint remains a formal-protocol gate.

Post-pilot zero-coupling and repeatability controls are exploratory development
checks motivated by the early planning outputs. They preserve the original
pilot manifest and outcomes. Independent metric replay uses cumulative sums
and saved x1 arrays without importing the primary metric implementation.

## CPU development v2 amendment, 2026-11-01 (not a confirmatory freeze)

The user authorized continuation on existing GCP CPUs. The committed
`development-v2.json`, `forecast-v2.json` and `epileptor-fit-v2.json` define a
one-worker, 60-minute development stage with no held-out patient access. New
work comprises source/sensor consistency checks, untreated network controls,
observation-only linear forecasting and a bounded mechanistic covariance fit.
The original pilot is preserved. No new intervention comparisons were run.

Sub-010's source/sensor mapping is consistent across three recordings. Sub-003's
source NPZ/TSV excerpts disagree and its sensor reconstruction remains imperfect;
it is excluded from this mechanistic fitting experiment. Fitting uses the strict
`PlannerObservation` schema. An access guard additionally rejects source and
generator file paths during the Epileptor fit. Priors are development-informed,
not investigator-blinded. Hidden parameters are read by the separate evaluator.

Temporal forecasting uses a 60/20/20 split with 50-sample embargoes. The later
Epileptor fit reuses the final development block and is exploratory; it matches
stationary covariance, not seizure timing. All networks get equal fit-candidate
budgets. Another-recording check has its own manifest addendum, specified after
some initial results and without any refitting. Different generating conditions
in those recordings are reported explicitly. They remain one development patient.

The nine invariant tests and independent replays pass, while scientific model
prediction and network-value gates do not. These fits are retained as baselines,
not promoted into the primary intervention study. The 24 reserved patients remain
untouched. Next development must address nonstationary seizure dynamics and an
informative, scientifically justified network comparison before formal freezing.

## CPU development v3 amendment, 2026-11-01 (exploratory timing diagnostic)

`manifests/development-v3.json` freezes a bounded test of timing alignment before
new feature scores are computed. It reuses sub-010 run-01 and the observation-only
v2 fitted trajectories. A causal 50-sample variance envelope, scaled from the
first 60% only, measures activity evolution without claiming clinical seizure
labels or verified physical time. The window differs from BVEP's clinical
high-pass log-power preprocessing; this is not a BVEP replication.

For each of the same four candidates per network, fit a global time offset using
the first 60%; select a candidate using the next 20%. Refit only the offset on
the first 80%, then predict the final 20% without further observation updates.
Compare with validation-selected persistence, mean, zero, damped trend and
autoregressive baselines with the same observation boundary. Target windows are
embargoed by 50 samples. Fit offsets for two fresh noise realizations per selected
network to check sensitivity, without selecting seeds or changing coupling.

All data are reused development data, including the previously inspected final
block. A phase search is an approximate initial-state diagnostic, not full
parameter inference. The six new simulations and all tests/replays remain on
the existing GCP CPU, one worker/thread at nice 10, under a 60-minute ceiling.
No downloads, held-out patients, clinical labels or intervention searches enter
this stage. Both failure and success require reporting before further planning.
