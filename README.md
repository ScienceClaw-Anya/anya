# Anya / NeuroTarget Swarm

ScienceClaw + Gemini agents select diagnostic stimulation experiments, share feedback and revise localization hypotheses in a Virtual Epileptic Patient (VEP) sandbox.

The project integrates public VEP data, Epileptor simulations and native ScienceClaw components. Four specialist roles propose the next stimulation site using connectome features, the team's literature evidence and measured SEEG summaries. Each proposal and measurement produces a traceable artifact.

**Research prototype: the diagnostic benchmark scores a generator-defined susceptibility reference, using a fixed SEEG-only response-ranking heuristic. Clinical localization accuracy and treatment benefit have not been established.**

## Independent deliberation and trained fusion

The [SPES v2 experiment](experiments/spes_agent_v2/README.md) separates communication from numerical fusion. It adds independent-first and no-exchange controls, a validation-only logistic fusion model, and native ScienceClaw tools for measured responses and trained predictions. Single-agent and multi-agent policies receive the same eight-request budget. The executable control uses **Gemini 4 Argon with medium reasoning**; a separately logged Argon/Nemotron adapter is available when a dedicated model API key is configured.

The fixed numerical fusion experiment completed on all 35 patients over five repeats: prediction-only fusion AUROC **0.6944**, response-feature fusion **0.6595**, versus the reconstructed Transformer **0.7201**. Fifty fits took less than one CPU second locally, with patient-disjoint fitting and independent prediction replay. These particular fusion models did not improve the baseline. Communication results, exact completion status and limitations are in the [English follow-up report](docs/spes-agent-v2-report.md) and [aggregate evidence](examples/spes_agent_v2_summary.json). The new agent experiment uses the first repeat only; its scores must be compared with the matching first-repeat numerical baseline.

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
| Tool discovery | Pinned upstream `SkillRegistry` discovers four project skills, including bounded response-bank queries |
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
