> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).

# Anya / NeuroTarget Swarm

ScienceClaw + Gemini agents study diagnostic stimulation in a Virtual Epileptic Patient (VEP) sandbox and SOZ contact ranking from real SPES recordings.

The project integrates public VEP data, Epileptor simulations, trained EEG models and native ScienceClaw components. The VEP track proposes stimulation sites from connectome features and measured SEEG summaries. The real-SPES track reconstructs numerical models and tests matched-budget agent policies. Decisions and measurements produce traceable artifacts.

**Research prototype: the diagnostic benchmark scores a generator-defined susceptibility reference, using a fixed SEEG-only response-ranking heuristic. Clinical localization accuracy and treatment benefit have not been established.**

## Schema-constrained acquisition follow-up

The sparse-training follow-up completed ten fits without improvement; those
weights are not adopted ([English v5 report](docs/spes-agent-v5-report.md)).
The feedback-matched v6 pilot completed 120 attempts, but an over-strict
validator rejected 15 real nested evidence references. Its actual fallback
outcomes stay recorded and cannot isolate collaboration ability
([English v6 report](docs/spes-agent-v6-report.md)).

The v7 interface attaches **JSON Schema to the actual Gemini request**: only
currently available query IDs and real evidence-root IDs may be returned.
Two synthetic interface checks and the complete eight-call smoke trajectory
passed. The formal five-patient pilot accepted **120/120 decisions**. Single,
multi and rationale-hidden policies all scored **0.7904**, equal to coverage;
the pilot is not expanded. Reviewers chose maximum coverage in 58/60 decisions
and replaced 13 proposals, always moving to coverage. See the
[English v7 report](docs/spes-agent-v7-report.md). Old studies remain frozen;
no multi-agent superiority or treatment benefit is claimed.

The [v8 experiment](experiments/spes_agent_v8/README.md) completed 20 inner
HAT fits, 110 small regressors, 6,720 correlated replay rows and 350 numerical
trajectories. All policies shared four initial measurements, four adaptive
measurements and unchanged final HAT. The fixed primary scored **0.7222** versus
random **0.7357**; its spending gate failed. The phase-zero secondary scored
**0.7475**, but its gain interval included zero and it does not replace the
primary. All rows passed independent feature/target/budget/outcome auditing.
See the [English v8 report](docs/spes-agent-v8-report.md).

The [v9 prototype](experiments/spes_agent_v9/README.md) stopped after 106/315
numerical trajectories when the physical action assumption failed; 209 remaining
or unfinished trajectories were canceled. The v10 contact models were fitted but
outer evaluation was blocked. All records are retained as column-ablation work.

The replacement [physical-pair environment](experiments/spes_physical/README.md)
maps real recorded anatomical pairs across different compact response columns.
All 35 patients show mixed physical pairs within old columns. Geometry verifies
114,021 cells; 253 ambiguous cells are excluded. The two-position smoke and an
actual HAT remap-invariance check passed. See the
[English action-grain correction](docs/spes-physical-action-audit.md).
The [first physically valid comparison](docs/spes-agent-v11-report.md) completed
175 trajectories on 35 patients. Score-focus selection scored **0.7620** versus
random **0.7405**, with paired gain interval **[-0.0117, +0.0520]**. This remains
uncertain; it does not establish agent value.

The [physical-state specialist test](docs/spes-agent-v12-report.md) completed ten
patient-isolated contact models. Logistic scored **0.7337** versus HAT **0.7405**
on identical eight-query inputs and is not adopted as a stronger scorer. Its
audit replayed 42,340 feature rows and 280 scorer/state outcomes. A separate
diagnostic found raw HAT scores average **0.826**, compared with mean patient
SOZ prevalence **0.132**. Training uses both class-weighted sampling and a
positive-weighted loss.

The [calibration and adaptive-boundary test](docs/spes-agent-v13-report.md)
completed 75 calibrators and 105 physical trajectories. Patient-centered
calibration modestly reduced Brier error (**0.1117** versus constant **0.1152**),
without improving eight-query ranking. Adaptive-boundary selection scored
**0.7440** versus score-focus **0.7620**; gain **-0.0179**, interval
**[-0.0431, +0.0053]**. Both fixed practical-gain gates fail. This hypothesis
does not trigger new Gemini calls. All 420 calibration outcomes and 420 adaptive
choices passed replay checks; no multi-agent improvement is established.

The [physical query-value expert test](docs/spes-agent-v14-report.md) fitted
25 regressors from 3,360 source-held-out physical counterfactual targets.
Consensus scored **0.7272**, all-feature boosting **0.7061**, and ridge **0.7234**,
versus score-focus **0.7620**. Consensus gain **-0.0348**, interval
**[-0.0775, +0.0112]**. Experts disagreed in 132/140 steps, but diversity did
not produce a better policy. The spending gate fails. All 105 trajectories and
420 adaptive choices passed independent replay; no Gemini call was added.

The [sparse-readout adaptation](docs/spes-agent-v15-report.md) completed ten
small readouts on frozen pretrained HAT representations. The context primary
scored **0.7343** versus original HAT **0.7405**; gain **-0.0061**, interval
**[-0.0465, +0.0351]**. The simpler readout's small positive secondary also has
an interval crossing zero. The gate fails and neither is adopted as an improved
localizer. Audit refitted all ten readouts and checked 57,159 correlated fitting
rows and 280 held-out scorer/state outcomes. No new Gemini call or neural fit
was used. A supported multi-agent improvement remains an open research goal.

The [independent physical-proposal pilot](docs/spes-agent-v16-report.md)
completed **120/120 accepted decisions** on five patients. Multi AUROC **0.7944**
versus single **0.7955** and score-focus **0.8119** does not pass expansion.
The paired multi/single gain interval is **[-0.0164, +0.0151]**. All fifteen
physical trajectories were re-inferred and 420 native artifacts verified.
Independent multi roles agreed on the first choice in 18/20 rounds; hiding peer
rationale did not reduce the mean outcome. The remaining 30 patients are not
run. Nine offline tests pass; code and aggregate results are committed, while
requests and patient-derived traces stay local. Twenty-one ambiguous recording
names remain explicit. See [reproducible commands](experiments/spes_agent_v16/README.md).

## Current research: complementary models and real-feedback acquisition

The [v4 research track](experiments/spes_agent_v4/README.md) preserves all earlier
negative results and keeps the stopped v3 study stopped. Its cheap numerical
screen did not justify more static-agent calls. Fifteen new patient-separated
fits test trial averaging, mean trial embeddings and hierarchical attention on
newly released multi-trial SPES arrays (35 patients / 2,117 contacts / 296 SOZ).
A [cohort identity audit](examples/spes_cohort_versions_summary.json) found only
34 patients overlap the older reconstruction: patient 39 replaces patient 47.
Same-data comparisons remain valid; cross-version scores are not directly comparable.
HAT scored **0.7495** versus the retrained same-data baseline **0.7138**;
the paired gain's 95% interval includes zero. This is an exploratory numerical
result, not demonstrated multi-agent superiority or clinical benefit.

The new acquisition task lets agents reveal eight real stimulation-column
responses and reconsider the next query after measured feedback. Single-agent,
four-role independent-first collaboration, a no-exchange ablation, random and
numerical strategies share the measurement budget. The final numerical scorer
is identical across arms. The five-patient pilot completed 120 attempts: multi **0.6331**, single
**0.6673**, no-exchange **0.5985**, compared on the same five patients. It is
not expanded. A [sparse-observation training follow-up](experiments/spes_agent_v5/README.md)
tests the mismatch between trial masking and whole missing stimulation columns
without further LLM requests. Feedback cadence also needs to be matched in any
later controller comparison. See the
[English research report](docs/spes-agent-v4-report.md) for current completion
counts and limitations. Training/inference run on the local RTX 4090; Gemini 4
Argon runs through Google's API. No new Nebius resources are created.

## Independent deliberation and trained fusion

The [v3 conservative-selection track](experiments/spes_agent_v3/README.md) unifies the Transformer default and OOF reliability evidence across all policies, prohibits handwritten probabilities, and fits a patient-isolated conservative selector from **60 newly trained inner models**. It compares two numerical policies and four Gemini policies under common decision/query ceilings, with actual model costs reported separately. See the [English v3 report](docs/spes-agent-v3-report.md) for completion status and effective decision opportunities. The earlier v1/v2 results below remain frozen.

The 60 inner fits and five selectors completed and passed independent leakage/coefficient audits. None of the 15 fold/candidate combinations cleared the fixed development gate; all 35 numerical selector outputs equal Transformer (AUROC **0.7274**). The uninformative agent run was stopped after **500 recorded attempts**, preserving partial traces and leaving **620 attempts unspent**. This is an analytically forced ranking tie, not a completed 35-patient agent comparison or evidence of general multi-agent failure. A new launch guard blocks identical-ordering and user-stopped studies before creating a model client.

The [SPES v2 experiment](experiments/spes_agent_v2/README.md) separates communication from numerical fusion. It adds independent-first and no-exchange controls, a validation-only logistic fusion model, and native ScienceClaw tools for measured responses and trained predictions. Single-agent and multi-agent policies receive the same eight-request budget. The executable control uses **Gemini 4 Argon with medium reasoning**; a separately logged Argon/Nemotron adapter is available when a dedicated model API key is configured.

The fixed numerical fusion experiment completed on all 35 patients over five repeats: prediction-only fusion AUROC **0.6944**, response-feature fusion **0.6595**, versus the reconstructed Transformer **0.7201**. Fifty fits took less than one CPU second locally, with patient-disjoint fitting and independent prediction replay. These particular fusion models did not improve the baseline. Communication results, exact completion status and limitations are in the [English follow-up report](docs/spes-agent-v2-report.md) and [aggregate evidence](examples/spes_agent_v2_summary.json). The new agent experiment uses the first repeat only; its scores must be compared with the matching first-repeat numerical baseline.

The Gemini 4 Argon comparison completed **35 patients × six arms**, **1,680 formal attempts** and **2,450 verified native artifacts**. Independent-first specialists scored **0.6870**, versus serial specialists **0.6897**; the paired difference was **-0.0027**, 95% patient-bootstrap interval **[-0.0305, +0.0234]**. Numerical-tool single and multi-agent policies scored **0.7235** and **0.7222**, respectively, compared with the same-repeat Transformer **0.7274**. Neither communication comparison demonstrates a consistent multi-agent benefit. One malformed vector and one connection failure remain counted through the frozen carry rule; no spent request was replayed. The API run took about 23 minutes; no additional Nebius GPU job was created.

## Paper reconstruction and same-task agents

The new track reconstructs Norris et al. (MLHC 2024) on public SPES recordings before comparing agents on the same SOZ electrode-classification task. All 48 recordings have been downloaded and checksum-verified. Public preprocessing yields 35 patients / 2,145 electrodes / 304 SOZ positives; the paper reports 35 / 2,066 / 298. The complete public-code rerun was authorized with this discrepancy disclosed. It must not be described as an exact reproduction of the paper cohort.

The complete schedule finished: **75 early-stopped GPU training/evaluation runs**, channel sensitivity, **175 held-out agent cases** and **2,800 formal Gemini request attempts**. Independent replay checked all predictions, patient splits, budgets, source hashes and 2,975 native ScienceClaw artifacts.

| Method | Observed mean patient AUROC |
|---|---:|
| Trained CNN Transformer | 0.7201 |
| Fixed three-model mean ensemble | 0.6837 |
| Single generalist | 0.6960 |
| Four specialists, two rounds | 0.6988 |

Multi-agent minus single-agent: **+0.0027**, 95% patient-bootstrap interval **[-0.0091, +0.0148]**. This comparison does **not demonstrate a consistent multi-agent advantage**. Numerical models were trained; Gemini was not fine-tuned. Both policies receive the same measured features and frozen predictions, eight requests each, and no hidden-label feedback. AUROC measures SOZ ranking, not treatment benefit. The paper's Transformer AUROC is 0.730; cohort and implementation differences prevent exact replication claims.

See the [English report](examples/spes_reconstruction_report.html), [machine-readable summary](examples/spes_reconstruction_summary.json), [reconstruction README](reproduction/norris2024/README.md) and [protocol](reproduction/norris2024/protocol.json) for all three model results, threshold rules, channel sensitivity, uncertainty and invalid-request accounting. Full local evidence is under `outputs/reproduction/norris2024/`. Numerical computation used the local RTX 4090 and CPU; Gemini used Google's API. No new Nebius job was created. The original VEP results below are a separate task.

## Diagnostic selection benchmark

Complete installation and the VEP download below before running these commands.

The multi-agent policy runs two rounds of four specialists. Every experiment records a prediction, one measured response and a reflection. Later agents see earlier proposals, observations and revisions. A single-agent policy receives the same tools and observations; a random policy receives the same experiment budget.

All policies share the candidate pool, waveform, simulation seed, calibration control and numerical localizer. Generator labels and accuracy scores are available only to the final judge. Cached queries still consume experiment budget. Gemini collective and single-agent runs each use 8 proposal calls plus 8 reflection calls. Random repetitions describe variability on the same patient, not independent patient validation.

```powershell
# Build 16 stimulation conditions and one no-stimulation control on four CPU workers.
.\.venv\Scripts\python.exe -m neurotarget.precompute --candidate-count 16 --seconds 55 --workers 4
# Replace BANK_JSON with the bank.json path printed by the previous command.
.\.venv\Scripts\python.exe -m neurotarget diagnostic --bank BANK_JSON --mode demo --budget 8 --random-repeats 20
.\.venv\Scripts\python.exe -m neurotarget diagnostic --bank BANK_JSON --mode gemini --budget 8 --random-repeats 20
```

Reports are `outputs/reports/diagnostic_demo.html` and `diagnostic_gemini.html`. Each run saves its frozen protocol, source hashes, native artifact lineage, per-step predictions/reflections and localization learning curves under `outputs/diagnostic/`.
`scripts/verify_diagnostic_run.py --session outputs/diagnostic/SESSION` replays the
rankings and scores from the selected sensor covariances, checks native lineage,
and audits call/query budgets and the prompt observation gate without new API calls.

The primary metric is average precision; secondary metrics are AUROC, Recall@10, Precision@10 and mean learning-curve AP. The common localizer estimates control-normalized response changes through a gain-normalized ridge inverse. These scores evaluate a development heuristic, not full Bayesian VEP inversion. See [the diagnostic method and limitations](docs/diagnostic-method.md).

Executed Gemini comparison (one virtual patient, eight queries each):

| Policy | Average precision | AUROC | Recall@10 |
|---|---:|---:|---:|
| Multi-agent | 0.0411 | 0.5338 | 0.000 |
| Single agent | 0.0406 | 0.5274 | 0.000 |
| Random, seed 42 | 0.0425 | 0.5427 | 0.000 |

The 20 random-policy repetitions have mean final AP 0.0475 on the same frozen
patient. The current experiment does not demonstrate successful localization or
a multi-agent advantage. [Portable report](examples/diagnostic_benchmark_report.html)
and [machine-readable run evidence](examples/diagnostic_benchmark_summary.json)
include the learning curves, predictions, reflections, source hashes and separate
real-data checks. The two initial schema-debugging API calls are separately accounted.

## Real-recording checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-real-data.txt
.\.venv\Scripts\python.exe -m neurotarget.real_data
.\.venv\Scripts\python.exe -m neurotarget datasets
```

Two public examples are implemented and exercised:

- **OpenNeuro ds003029:** one recorded seizure, clinical SOZ contact annotations and bad-contact exclusions. A fixed early-ictal/preictal variance-ratio baseline is scored at the contact level. This is passive localization, not an active stimulation experiment.
- **PRIOS / ds004370:** five pulses at each of two distinct recorded stimulation pairs, with artifact windows, bad channels and stimulating contacts excluded. Only dimensionless normalized responses are reported. No independent SOZ reference was verified for this example, so no localization accuracy is invented.

Only short, exact HTTP byte ranges are downloaded, with ETag locking and per-range SHA-256 records. Small annex assets are checked against their published checksum. Raw recordings and clinical annotation tables stay outside Git. See [the real-data registry](docs/real-datasets.md), which also covers every resource in the supplied `More possible datasets.txt` list.

## Actual integration

| Component | Implementation |
|---|---|
| Tool discovery | Pinned upstream `SkillRegistry` discovers project VEP and SPES skills, including bounded response-bank and evidence queries |
| Tool execution | Upstream `SkillExecutor` runs each scientific tool as a JSON subprocess |
| Provenance | Native `Artifact` / `ArtifactStore` format, hashes, parent lineage and shared index |
| Gemini | Native `LLMClient(backend="gemini")` calls the official Google endpoint |
| Experiment control | Project adapter bounds simulation/LLM budgets, validates IDs and restricts observations |

The multi-agent loop is a bounded project adapter with independent profiles and shared artifacts. It does not run upstream ArtifactReactor, Infinite publishing, heartbeat daemons or optional PQC attestation. Registry/store subclasses keep state inside the project and omit publication/key-generation side effects. See `docs/integration.md`.

## Install

Use Python **3.10** for the tested numerical stack. Only the listed ScienceClaw components are integrated here; the complete upstream application recommends newer Python and has additional dependencies.

```powershell
git clone https://github.com/ScienceClaw-Anya/anya.git
cd anya
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-scienceclaw.txt
.\.venv\Scripts\python.exe scripts/setup_upstreams.py
```

Read and accept the [EBRAINS access terms](https://kg.ebrains.eu/search-terms-of-use.html?v=2.2) before downloading:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_vep.py --patient sub-003 --accept-ebrains-terms
.\.venv\Scripts\python.exe -m neurotarget inspect
.\.venv\Scripts\python.exe -m neurotarget minimum
```

`sub-003` has 162 regions and 179 available adjacent contact pairs. The minimum compares two distinct locations for 55 seconds each. Downloads, raw data, upstream source checkouts and generated outputs are ignored by Git.

## ScienceClaw + Gemini

The `scienceclaw` command below is the earlier integration smoke workflow. Use
`diagnostic` above for scored multi-round comparisons with explicit reflection.

Copy `.env.example` to `.env`, privately fill `GEMINI_API_KEY` from your activity project's pre-provisioned Workshop key, and save. A GCP email/password is a sign-in credential, not the API key. The configured model must be available to that key; it is not silently replaced.

```dotenv
LLM_BACKEND=gemini
GEMINI_MODEL=gemini-3.6-flash
GEMINI_API_KEY=YOUR_PRIVATE_WORKSHOP_KEY
GEMINI_REASONING_EFFORT=low
GEMINI_THINKING_HEADROOM=1024
GEMINI_MAX_RETRIES=0
```

```powershell
# Native tool/artifact integration without a key.
.\.venv\Scripts\python.exe -m neurotarget scienceclaw --mode demo --budget 4 --candidate-count 16 --seconds 12 --max-steps 4
# Real Gemini decisions through ScienceClaw's LLMClient.
.\.venv\Scripts\python.exe -m neurotarget scienceclaw --mode gemini --budget 4 --candidate-count 16 --seconds 12 --max-steps 4
# Matched observation-isolation ablation.
.\.venv\Scripts\python.exe -m neurotarget scienceclaw --mode gemini --budget 4 --candidate-count 16 --seconds 12 --max-steps 4 --isolated
```

Profiles are in `config/agents.json`: structure, dynamics, evidence and critic. Candidate count must exceed the experiment budget. Both SDK and upstream retries are disabled. Unknown targets/evidence IDs, invalid JSON and API failures stop explicitly, with no silent demo fallback.

The original `scienceclaw` command exposes geometry/network scores, attributed evidence cards and SEEG summaries. The new `diagnostic` command additionally shares predictions, reflections and a numerical ranking inferred from SEEG. Neither command exposes generator x0/EZ labels, true source activity, accumulation thresholds or reference accuracy to agents. This observation gate alone does not establish a clinical localization benchmark.

## Results and checks

- Original numerical baseline: two sites, 55 seconds each, run locally and on Nebius CPU. Ten local/cloud source-variance and threshold-count summaries agreed; full cloud waveforms were not compared point by point.
- Native integration: four real Gemini decisions, four 12-second simulations among 16 candidates, ten artifacts with verified hashes and lineage. Public portable summaries are in `examples/`.
- Reports: `outputs/reports/minimum.html`, `scienceclaw_demo_shared.html` and `scienceclaw_gemini_shared.html`.
- Reports, agent explanations and project documentation are generated in English.
- Native state: `outputs/scienceclaw/<session>/session.json`, `artifacts/<agent>/store.jsonl` and `artifacts/global_index.jsonl`.
- A 12-second experiment demonstrates early responses, not a full-seizure outcome. Very short runs are smoke checks only.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The full local suite requires downloaded VEP data. CI runs deterministic offline native-integration tests against the pinned upstream, without Gemini credentials or automatic acceptance of data terms.

## Infinite connection and draft

The optional Infinite access-gateway session is stored in an ignored local auth
file. Gateway username/password authentication is distinct from an agent API
identity. The project can check the configured gateway and prepare a local draft:

```powershell
.\.venv\Scripts\python.exe -m neurotarget infinite-check
.\.venv\Scripts\python.exe -m neurotarget infinite-draft
```

The draft is written to `outputs/infinite/draft.json`. These commands do not
register an agent or publish a post. Publication requires an authorized destination
and an existing or separately authorized agent API identity. Do not commit gateway
sessions, tokens, passwords or private endpoint configuration.

## Scientific limits of the original VEP track

1. Uses the author's EpileptorStim2Populations and custom stochastic Heun integrator. Its coupling declaration `[0]` conflicts with three channels read by the compiled derivative; our subclass explicitly declares `[0,3,6]`. This is a documented model variant, not confirmed exact reproduction of the released paper run.
2. Released initial-state metadata is truncated and its random seed is unavailable. Our explicit initial state comes from the author script. Time conversion uses released `sfreq=400` scaling.
3. Accumulation threshold crossings are model diagnostics, not validated seizure labels. Signals are model units, not calibrated clinical voltage.
4. The original VEP track performs forward simulation without training a network or fitting a patient-specific inference model. The separate SPES reconstruction above trains three supervised models.
5. One virtual patient cannot establish localization accuracy, pathway causality, clinical benefit or multi-agent superiority. Additional held-out VEP patients, coverage controls and model-mismatch evaluations remain future work. The separate SPES comparison uses 35 held-out patients across repeated splits.

## Literature and cloud

The team's [epilepsy-virtual-brain review](https://github.com/ScienceClaw-Anya/epilepsy-virtual-brain) supplies attributed method context. Five cards in `research/evidence.json` record the source commit, hashes and URLs. This integration does not independently repeat its full literature audit or implement every proposed onset/bottleneck/control method.

See [cloud/README.md](cloud/README.md) for Nebius. TVB currently uses CPU; Gemini is an external API. The current diagnostic bank ran on the local workstation with four CPU workers, not on a GPU or a new Nebius job. Keys, private PDFs, user-specific cloud IDs and existing storage configuration are not included in public code or generated bundles.

References: VEP dataset [10.25493/XWG2-S8X](https://doi.org/10.25493/XWG2-S8X), CC BY 4.0; Dollomaja et al. (2025), [10.1371/journal.pcbi.1012911](https://doi.org/10.1371/journal.pcbi.1012911); [ScienceClaw](https://github.com/lamm-mit/scienceclaw); [author code](https://github.com/BalanceKey/virtual_epilepsy_patient_cohort); [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai).

See `THIRD_PARTY_NOTICES.md` for attribution and licensing boundaries.
