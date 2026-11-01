# SPES v3: conservative selection with matched information

The conservative selector retained the exact Transformer ranking for all 35 held-out patients. No tested alternative cleared the fixed development gate. The resulting policy tie is forced by the permitted action space and cannot identify a general collaboration benefit or failure. Execution and audit status are listed below.

## Completion status

Inner numerical model fits: **60/60**. No incomplete stage is reported as a completed comparison.

## The uninformative API run was stopped

The user requested stopping unnecessary execution after the audit found no distinct rankings. **500 attempts** were recorded; **620 remaining attempts were not scheduled**. In-flight reserved attempts remain spent; their token consumption is unknown.

All 60 neural fits and five selector fits completed. The agent study did not complete all 35 patients per policy. Partial traces and completed cases are retained; they are not presented as a full observed agent comparison. Ranking equivalence for all 35 patients follows from the identical audited score vectors, not from a completed model-call study.

A new label-free launch guard now rejects identical-ordering score banks and user-stopped studies before any model client is created. The original frozen policy controller is preserved for replay.

## Numerical results and actual decision space

| Numerical policy | Mean patient AUROC, first repeat |
|---|---:|
| transformer | 0.7274 |
| selector | 0.7274 |

These two numerical scores cover all 35 prepared and independently replayed scorer vectors. The execution table below counts completed eight-step policy traces, a separate quantity; no full-cohort empirical agent score is inferred from those partial traces.

**0/35 patients** have different executable vectors, and **0/35** have different contact orderings for Transformer and the conservative selector.

The allowed scorers have the same contact ordering throughout this held-out cohort. Equal ranking performance is forced by the action space; it cannot establish that collaboration generally lacks value.

Every selector output equals the exact Transformer vector.

Of 15 development fold/candidate combinations, 3 had positive mean gains; the largest was +0.0018 AUROC. 0 had positive bootstrap lower bounds. The tested bounded CNN blends did not establish a robust development gain under this protocol.

The global eligibility veto can also suppress a candidate that helps a predictable subgroup. This experiment does not establish that such subgroup selection is impossible; it has not independently validated such a policy. Improving candidate complementarity or subgroup evidence requires a new development protocol, rather than interpreting a constrained tie as a general result about agents.

| Outer fold | Eligible alternatives |
|---|---|
| 0 | None; retain Transformer |
| 1 | None; retain Transformer |
| 2 | None; retain Transformer |
| 3 | None; retain Transformer |
| 4 | None; retain Transformer |

| Outer fold | Candidate | Inner-OOF mean gain | Gate lower bound | Eligible |
|---|---|---:|---:|---|
| 0 | convergent_10 | -0.0017 | -0.0272 | False |
| 0 | divergent_10 | -0.0104 | -0.0364 | False |
| 0 | both_20 | -0.0051 | -0.0368 | False |
| 1 | convergent_10 | -0.0002 | -0.0093 | False |
| 1 | divergent_10 | -0.0034 | -0.0198 | False |
| 1 | both_20 | -0.0048 | -0.0185 | False |
| 2 | convergent_10 | -0.0054 | -0.0289 | False |
| 2 | divergent_10 | +0.0018 | -0.0213 | False |
| 2 | both_20 | -0.0006 | -0.0312 | False |
| 3 | convergent_10 | -0.0004 | -0.0111 | False |
| 3 | divergent_10 | -0.0029 | -0.0138 | False |
| 3 | both_20 | +0.0002 | -0.0134 | False |
| 4 | convergent_10 | -0.0004 | -0.0086 | False |
| 4 | divergent_10 | -0.0092 | -0.0279 | False |
| 4 | both_20 | +0.0017 | -0.0106 | False |

These are development-stage gate statistics, not independent held-out scores of the trained selector. The 0.005 margin is a fixed research safeguard, not a validated clinical effect threshold.

## Policy execution was stopped

| Policy | Completed patients |
|---|---:|
| Numerical Transformer | 18/35 |
| Numerical conservative selector | 18/35 |
| Single generalist | 14/35 |
| Serial specialists | 14/35 |
| Independent-first specialists | 14/35 |
| Specialists without exchange | 13/35 |

No incomplete arm is used for full-cohort performance claims.

## What could justify a useful next experiment

A cheap development-only hindsight diagnostic found an average best-of-four gain of +0.0210 AUROC across the five development pools. This chooses after seeing development labels: it is optimistic, not a deployable score and not proof that a selector can identify the benefiting patients. The pools overlap and are not independent cohorts.

Ignoring the global eligibility veto while keeping the existing ridge predictions and 0.005 margin would propose alternatives for 7/35 unlabeled outer cases. This counts possible actions only; no claim is made that their held-out accuracy improves. No new training, neural inference or API calls were performed for these diagnostics.

The current global rule tests whether each candidate is reliably beneficial on average. A personalized selector instead needs independent evaluation of its complete conditional policy. A candidate can have a negative overall average yet help a predictable subgroup; the present global veto prevents testing that possibility. Any new policy must be assessed with patient isolation through every fitted component, rather than treating selector-training scores as validation.

The next useful work is numerical feasibility before agents: define a single conditional-policy hypothesis, validate it on development patients, and stop if it cannot beat Transformer reliably. This cohort has already been inspected repeatedly, so any further replay is exploratory; an independent cohort is needed for final validation. Agent comparison should resume only with a meaningful decision space and a reason to expect value beyond the numerical policy.

## How the numerical gate was trained

Five outer folds each hold out seven patients. Four new inner folds within the 28-patient development pool use 17 fitting patients, four early-stopping patients and seven OOF prediction patients. All three base architectures are fitted in each inner job: 60 new fits. The outer test patients never enter fitting, normalization or early stopping.

Five fixed Ridge(alpha=100) selectors learn patient-level AUROC gains from four prediction-summary features. All candidate alternatives retain at least 80% Transformer weight. Eligibility requires a Bonferroni one-sided patient-bootstrap lower bound above 0.005 gain; the personalized predicted gain must also exceed 0.005. Otherwise the exact Transformer vector is returned. No test-driven parameter search was performed.

The gate is a conservative development heuristic, not a clinical guarantee. Overlapping inner models affect uncertainty, candidate OOF performance trains the selector, and its individual gain predictions are not independently calibrated. Inner base fits use 17 plus four early-stop patients; reused outer models use 21 plus seven. This training-size shift is a limitation.

## What is held constant

Every numerical and agent policy receives the same explicit baseline, OOF reliability, permitted scorer outputs and distribution evidence. Agents output only a scorer action and structured evidence references. Probability vectors and manual weights are rejected. Complete reasons are retained within the declared length bound.

Four Gemini policies share Gemini 4 Argon, medium reasoning, a 32,768 combined completion-token cap and an eight-request trajectory cap. Completed trajectories use eight attempts; interrupted trajectories retain only their spent prefix. The primary contrast is independent-first specialists versus one generalist. Serial specialists and no-exchange specialists are controls. Completed final scores are programmatic means of the last four executable vectors in full precision. Invalid or uncaptured attempts default to Transformer and are never replaced by an extra request.

These are static recorded-data decisions: later rounds receive discussion, not newly measured SPES data or hidden accuracy feedback. Roles share the same base model; they are not separately trained clinical experts.

## Evidence and limits

Observed sum of individual neural fit durations: **3689.9 seconds**. Fits overlap on the local RTX 4090; this sum is not elapsed wall time. No additional Nebius GPU job was created. Gemini inference uses the official Google API.

The public-code cohort contains 35 patients, 2,145 contacts and 304 SOZ contacts; the paper reports 35 / 2,066 / 298. The cohort and earlier results were already inspected. This follow-up is exploratory and requires independent patients for external validation. AUROC and AP measure retrospective SOZ ranking, not treatment benefit. Bootstrap intervals resample patients conditional on recorded fits and generations, and do not capture all retraining or sampling variability. Secondary comparisons are exploratory.

The preparation audit verifies every inner split including early stopping, checkpoint/prediction hashes, label alignment, ridge normal-equation coefficients, bootstrap gate and exact scorer execution for all 35 cases. The policy audit verifies packets, budgets and native lineage for completed trajectories; interrupted prefixes are preserved, not certified as complete. Neural outputs are frozen captured predictions; the audit does not claim bit-exact re-inference of stochastic CNN channel sampling.

See the [v3 README](../experiments/spes_agent_v3/README.md) for exact commands and the [aggregate summary](../examples/spes_agent_v3_summary.json). Full private evidence stays in outputs/spes_agent_v3; raw data, labels, model files and credentials are excluded from Git.
