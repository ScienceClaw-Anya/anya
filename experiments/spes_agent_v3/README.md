# SPES v3: patient-isolated conservative selection

This version fixes the information and output-interface differences identified in the [v2 follow-up](../../docs/spes-agent-v2-report.md). Every policy receives the same explicit Transformer default, patient-isolated cross-prediction evidence, conservative-selector outputs and distribution summaries. Agents select executable scorers; they cannot write probabilities or weights.

Results and limitations are in the [English v3 report](../../docs/spes-agent-v3-report.md) and [portable aggregate evidence](../../examples/spes_agent_v3_summary.json). Frozen v1/v2 sources, predictions and results are preserved.

## Training without patient leakage

The study uses the first original five-fold repeat: seven held-out patients per fold, 35 distinct patients overall. Within each outer fold, the original 21 training and seven early-stopping patients form a 28-patient development pool.

That development pool is split into four inner prediction folds. Each inner job has **17 fitting patients, four early-stopping patients and seven prediction patients**. No outer-test patient enters any inner job, and an inner prediction patient cannot participate in fitting or early stopping. Three original model architectures are trained per job, giving **60 new local GPU fits**.

The unchanged released data loader computes normalization from the 17 fitting patients. Architecture hyperparameters and patience 10 follow the earlier reconstruction; the fixed ceiling is 1,000 epochs, with no additional hyperparameter search. Predictions use actual subject IDs, not the upstream validation-local indices. Checkpoint/prediction hashes and the full fitting/early-stop/prediction manifests are recorded for every job.

Existing predictions from other outer folds are **not** pooled as selector-training data. Their generating models may have seen the current outer test patients. The original first-repeat outer models are reused only for final held-out inference; their fitting and early-stopping patients are entirely within that outer development pool.

## What is trained

One fixed ridge selector is fitted per outer fold, using the 28 inner-OOF patients as 28 training examples. Its four unlabeled predictor-summary inputs are Transformer score spread, divergent/convergent model disagreement and top-ten Transformer/convergent overlap. The targets are each patient's AUROC gains for three fixed numerical alternatives:

| Candidate | Transformer | Convergent CNN | Divergent CNN |
|---|---:|---:|---:|
| Default | 100% | 0% | 0% |
| `convergent_10` | 90% | 10% | 0% |
| `divergent_10` | 90% | 0% | 10% |
| `both_20` | 80% | 10% | 10% |

The selector uses StandardScaler and Ridge(alpha=100), with no tuning. All fitting and scaling use development patients only. A candidate is eligible only if its inner-OOF mean patient gain has a Bonferroni one-sided bootstrap lower bound above **0.005 AUROC**. There are 10,000 patient bootstrap draws and three alternatives. For a new patient, the selector chooses the eligible candidate with the largest predicted gain, only when that prediction also exceeds 0.005. Otherwise it returns the exact Transformer vector.

This is a conservative development heuristic, not a guarantee of improved ranking or clinical safety. Inner fits overlap, individual gain predictions are not independently calibrated, and inner/outer models have different fitting sample sizes. Candidate OOF performance is training evidence for the selector; it must not be presented as independent validation of the fitted selector. Independent evaluation uses the seven outer held-out patients.

All four predictor summaries describe **model predictions**. They are not waveform-quality measurements, causal pathways or physiological biomarkers. No LLM fine-tuning is performed.

## Matched information and budgets

| Policy | Decision rule / communication | Gemini attempts per patient |
|---|---|---:|
| `numeric_transformer` | Fixed Transformer | 0 |
| `numeric_selector` | Fixed trained conservative selector | 0 |
| `single` | One generalist, serial memory | 8 |
| `multi_serial` | Four roles, serial memory | 8 |
| `multi_independent` | Independent first four decisions, then exchange | 8 |
| `multi_no_exchange` | Four roles, no exchange | 8 |

Every policy gets **eight decision slots and eight native ScienceClaw evidence queries**. Agent policies share an eight-request model cap and the same completion cap; numerical baselines use zero model calls. This compares policies under common budget ceilings. It does not claim exact monetary-cost equality or pad numerical baselines with unnecessary model calls. Actual attempts and captured token usage are reported separately.

All six policies receive the same public packet and tools, including OOF reliability. The exact Transformer vector is the shared default and failure fallback. The two executable actions are `transformer` and `conservative_selector`. Program execution averages the final four scorer vectors in full precision; the displayed packet uses six decimal places. Averaging preserves at least 80% Transformer weight.

The four role profiles are ranking, validation, distribution and auditor. Their output is strictly `{action, evidence_ids, reason, counterargument}`. Unknown evidence IDs, weights or probability vectors are rejected. Complete text fields are retained up to the declared 900-character limit; no silent 400-character truncation occurs. Role-specific profiles do not create separate trained experts. Evidence is static across rounds: critique does not reveal labels or acquire new physiological observations.

Public packets omit held-out patient identity, clinical labels, prevalence and accuracy. `judge.json` is read only after the eight policy decisions are saved. This is a programmatic observation gate, not an operating-system security boundary. Native tool scripts reject the judge and unapproved paths.

When both actions execute the same vector, the patient has no effective ranking choice. The report counts distinct-action patients explicitly. A tie forced by closed gates does not establish that collaboration generally lacks value.

## Execute locally

Use the existing Python 3.10 `.venv-repro` reconstruction environment with CUDA. The 60 new neural fits require the earlier numerical dependencies, including its pinned Torch/TorchEEG stack. The lightweight offline guards use `requirements-spes-agent-v2.txt` and do not install or train Torch models in CI. Do not modify the active numerical environment while it is training.

```powershell
python -m unittest experiments.spes_agent_v3.test_v3 -v
python -m experiments.spes_agent_v3.crossfit --project . --output outputs/spes_agent_v3 --workers 2
python -m experiments.spes_agent_v3.prepare --project . --output outputs/spes_agent_v3
python -m experiments.spes_agent_v3.audit --project . --output outputs/spes_agent_v3
python -m experiments.spes_agent_v2.providers --project . --output outputs/spes_agent_v3 --providers argon
python -m experiments.spes_agent_v3.engine --project . --output outputs/spes_agent_v3 --workers 16
python -m experiments.spes_agent_v3.audit --project . --output outputs/spes_agent_v3 --policies
python -m experiments.spes_agent_v3.report --project . --output outputs/spes_agent_v3
```

The protocol freezes the split schedule, source hashes, fixed selector configuration and candidate weights before training. The agent protocol separately freezes all decision inputs and critical sources before formal API requests. Coordinator lock files prevent concurrent coordinators against the same stage. Resume only after the previous coordinator exits; a stale lock after a crash requires inspecting its recorded process ID before removal.

Captured responses can be parsed again without another model call. Reserved or failed attempts remain spent and default to Transformer on resume. Transport errors halt new scheduling while in-flight tasks finish. Invalid schema or incomplete model responses consume their attempt. There are no automatic request retries. The synthetic interface preflight is separate from the **1,120 formal Gemini attempts** planned for four agent arms.

The engine pins **Gemini 4 Argon**, medium reasoning, 32,768 combined completion tokens, through the real pinned ScienceClaw LLMClient and the official Google endpoint. No Nemotron fallback, new Nebius job, Infinite publication or autonomous heartbeat is enabled.

## Audit and interpretation

The preparation audit verifies all 60 neural prediction/checkpoint receipts, patient separation including early stopping, released-label row order, and raw outer-source hashes. It independently reconstructs the ridge coefficients using normal equations, replays the bootstrap gate and executes all full-precision scorer vectors. Neural outputs are frozen captured predictions; CNN inference samples channels stochastically, so the audit does not claim bit-exact re-inference of every neural checkpoint.

The policy audit replays every final prediction and pairwise AUROC, verifies the public packet, complete board visibility, request hashes, native artifact hashes and parent lineage, and reconciles tool/model budgets. A complete six-arm study has 210 patient-policy cases, 1,680 tool-query/decision slots and 3,570 native artifacts. It supports `--allow-incomplete` for operational checks without declaring complete-cohort performance.

Primary performance is mean patient AUROC, with paired patient-bootstrap intervals for independent specialists minus the generalist and comparisons against the numerical rules. Average precision, harm frequency, distinct-action opportunities and actual resource use are secondary. Intervals are conditional on the recorded numerical fits and model generations; they do not quantify all retraining or generation variability. No unsuccessful patient or malformed response is discarded.

This is an exploratory follow-up on the already inspected public-code cohort (35 patients / 2,145 contacts / 304 SOZ contacts), which differs from the paper's 35 / 2,066 / 298. It measures retrospective SOZ ranking, not treatment benefit or active stimulation selection. New tuning based on these results needs a fresh evaluation protocol and ultimately an independent cohort.

Generated models, clinical labels, private API traces, and raw recordings remain outside Git under ignored `outputs/spes_agent_v3`. Only code, English documentation and aggregate evidence are committed.
