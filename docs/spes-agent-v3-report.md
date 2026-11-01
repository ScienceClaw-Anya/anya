# SPES v3: conservative selection with matched information

This experiment makes Transformer the explicit shared default, replaces free probability generation with executable decisions, and trains a small conservative selector from patient-isolated cross-predictions. The previous experiments remain frozen.

## Completion status

Inner numerical model fits: **27/60**. No incomplete stage is reported as a completed comparison.

## How the numerical gate was trained

Five outer folds each hold out seven patients. Four new inner folds within the 28-patient development pool use 17 fitting patients, four early-stopping patients and seven OOF prediction patients. All three base architectures are fitted in each inner job: 60 new fits. The outer test patients never enter fitting, normalization or early stopping.

Five fixed Ridge(alpha=100) selectors learn patient-level AUROC gains from four prediction-summary features. All candidate alternatives retain at least 80% Transformer weight. Eligibility requires a Bonferroni one-sided patient-bootstrap lower bound above 0.005 gain; the personalized predicted gain must also exceed 0.005. Otherwise the exact Transformer vector is returned. No test-driven parameter search was performed.

The gate is a conservative development heuristic, not a clinical guarantee. Overlapping inner models affect uncertainty, candidate OOF performance trains the selector, and its individual gain predictions are not independently calibrated. Inner base fits use 17 plus four early-stop patients; reused outer models use 21 plus seven. This training-size shift is a limitation.

## What is held constant

Every numerical and agent policy receives the same explicit baseline, OOF reliability, permitted scorer outputs and distribution evidence. Agents output only a scorer action and structured evidence references. Probability vectors and manual weights are rejected. Complete reasons are retained within the declared length bound.

Four Gemini policies use the same Gemini 4 Argon model, medium reasoning, 32,768 combined completion tokens and eight requests. The primary contrast is independent-first specialists versus one generalist. Serial specialists and no-exchange specialists are controls. All final scores are programmatic means of the last four executable vectors in full precision. Invalid or uncaptured attempts default to Transformer and are never replaced by an extra request.

These are static recorded-data decisions: later rounds receive discussion, not newly measured SPES data or hidden accuracy feedback. Roles share the same base model; they are not separately trained clinical experts.

## Evidence and limits

Observed sum of individual neural fit durations: **1560.5 seconds**. Fits overlap on the local RTX 4090; this sum is not elapsed wall time. No additional Nebius GPU job was created. Gemini inference uses the official Google API.

The public-code cohort contains 35 patients, 2,145 contacts and 304 SOZ contacts; the paper reports 35 / 2,066 / 298. The cohort and earlier results were already inspected. This follow-up is exploratory and requires independent patients for external validation. AUROC and AP measure retrospective SOZ ranking, not treatment benefit. Bootstrap intervals resample patients conditional on recorded fits and generations, and do not capture all retraining or sampling variability. Secondary comparisons are exploratory.

The audit verifies every inner split including early stopping, checkpoint/prediction hashes, label alignment, ridge normal-equation coefficients, bootstrap gate, exact scorer execution, public packets, request budgets and native artifact lineage. Neural outputs are frozen captured predictions; the audit does not claim bit-exact re-inference of stochastic CNN channel sampling.

See the [v3 README](../experiments/spes_agent_v3/README.md) for exact commands and the [aggregate summary](../examples/spes_agent_v3_summary.json). Full private evidence stays in outputs/spes_agent_v3; raw data, labels, model files and credentials are excluded from Git.
