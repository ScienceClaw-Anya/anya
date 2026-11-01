# SPES follow-up: independent deliberation and trained fusion

## Interpretation of the completed comparison

This comparison did not demonstrate a consistent benefit from independent first-round deliberation. The paired mean AUROC difference was -0.0027, with a 95% patient-bootstrap interval of [-0.0305, +0.0234]. The cohort was already inspected and is small; this is not external clinical validation.

Intervals resample the 35 patients conditional on the recorded model generations and numerical checkpoints. There is one agent run per patient and policy in this first-repeat study; the intervals do not quantify all variability from rerunning stochastic model generation or retraining the numerical models.

All roles in this control use the same base model and observation packet. Independence changes which proposals are visible; it does not create independent training data, separately trained experts, or new physiological measurements. The result concerns this prompted-role architecture and cannot establish that every multi-agent architecture succeeds or fails.

A change in agreement or second-round accuracy alone does not identify anchoring as the mechanism. The matched visibility intervention is the primary evidence; agreement and reflection diagnostics help describe what the policies did.

Every arm used the same number of request attempts and the same maximum completion-token cap. Realized context length, token use and monetary cost can differ. The comparison establishes a call-budget match, not exact monetary-cost equality. No price estimate is invented.

The single generalist with numerical tools reached 0.7235, and tool-enabled specialists reached 0.7222; the matching Transformer baseline was 0.7274. Multi-agent minus single-agent in the tool comparison was -0.0013 [-0.0179, +0.0150] (95% paired patient-bootstrap interval). This comparison also does not demonstrate a consistent collaboration benefit.

The corresponding manual-score generalist reached 0.6985. Tool arms are descriptively closer to the numerical baseline, but they jointly change scorer access, validation evidence and the output interface. The data do not identify which component caused that difference. The practical recommendation is to preserve the held-out numerical baseline for contact ranking and require evidence before accepting LLM score revisions.

Serial specialists produced identical final four vectors for 6/35 patients; independent-first specialists did so for 0/35. Thus greater output diversity did not translate into a demonstrated ranking benefit. The controlled comparison does not establish first-round copying as the main performance limit.

Independent-first second-round AUROC changed by +0.0029, while no-exchange changed by +0.0047. A small second-round gain is therefore not, by itself, evidence that shared reflection added value. These round-change diagnostics are descriptive and exploratory.

Public model probabilities and response summaries are serialized to five decimal places; executable scorer vectors use six. Numerical baseline tables use the original full-precision predictions. Tiny rank-tie differences are therefore possible against a raw baseline; all policies within each communication comparison share the same serialization.

Token totals sum captured usage records. Any uncaptured request has unknown token consumption; these totals do not establish complete billed usage.

### Recorded failures and executable choices

| Policy | Attempts | Schema carries | Transport carries |
|---|---:|---:|---:|
| single_serial | 280 | 0 | 0 |
| multi_serial | 280 | 0 | 1 |
| multi_independent_first | 280 | 0 | 0 |
| multi_no_exchange | 280 | 1 | 0 |
| single_tools | 280 | 0 | 0 |
| multi_independent_tools | 280 | 0 | 0 |

All failed attempts remain in the primary comparison through the predeclared carry rule; no failed case or patient is dropped.

| Tool policy | Final-round scorer selections, 140 opportunities |
|---|---|
| single_tools | feature_fusion: 4; prediction_fusion: 52; transformer: 84 |
| multi_independent_tools | feature_fusion: 5; prediction_fusion: 26; transformer: 109 |

Scorer-selection counts describe executable decisions, not clinical effects. Averaging the final four selections can produce a numerical mixture even when no single scorer is used unanimously.

All six policies completed 35 patients each: 1,680 formal attempts, 1,678 accepted outputs and 2,450 verified native artifacts. Elapsed API-run time, estimated from local request receipts, was about 23.3 minutes with 16 local coordinators waiting on the remote API. The synthetic preflight is excluded from these counts.

## Why the fixed response-feature fusion may generalize poorly

| Model | Score on patients used for fitting / checkpoint selection | Held-out patient AUROC |
|---|---:|---:|
| transformer | 0.7636 | 0.7201 |
| prediction_fusion | 0.7566 | 0.6944 |
| feature_fusion | 0.8030 | 0.6595 |

The response-feature fusion has a larger fit-to-test gap than the Transformer. That is consistent with overfitting or weak transportability of these summaries across patients. It is not proof of one causal explanation, nor independent evidence that the added features contain no signal. Fusion scores on fitting patients are resubstitution, and the Transformer also used validation for early stopping.

This diagnostic was computed after fitting, without refitting, hyperparameter search, or modifications to the ongoing communication experiment. The next defensible model-development step would use patient-disjoint development data and a separate untouched evaluation cohort; the current test set should not become a tuning set.

Reproduce the diagnostic with `python analysis/spes_v2_fusion_diagnostics.py --project . --output outputs/spes_agent_v2`. Rebuild this interpreted report, after the offline audit, with `python analysis/finalize_spes_v2_report.py --project . --output outputs/spes_agent_v2`.

The fixed validation-trained fusion models did not beat the reconstructed Transformer. Communication experiments use a separate first-repeat, 35-patient cohort and equal call budgets. This is retrospective SOZ contact ranking, not evidence of therapeutic benefit.

## Numerical fusion: completed and independently replayed

| Scorer | Mean patient AUROC, all five repeats |
|---|---:|
| transformer | 0.7201 |
| mean_ensemble | 0.6837 |
| prediction_fusion | 0.6944 |
| feature_fusion | 0.6595 |

These results average the five held-out predictions for each of 35 distinct patients; 175 patient-repeat cases are not 175 independent patients. Fifty fixed L2 logistic models were fitted: two per split, five folds, five repeats. Total fitting time was 0.147 seconds on the local CPU. No cloud GPU was started.

The prediction-only fusion uses three numerical model probabilities. The response-feature fusion adds 15 measured SPES response summaries. Both use C=0.1, weighted scaling and equal total weight per validation patient. Only the original seven validation patients per outer split fit the fusion; the 21 numerical-training and seven test patients are disjoint. Validation also supported original model early stopping. Hyperparameters were fixed and not selected against test scores.

- mean_ensemble minus transformer: -0.0363; 95% patient-bootstrap interval [-0.0718, -0.0032].
- prediction_fusion minus transformer: -0.0257; 95% patient-bootstrap interval [-0.0480, -0.0030].
- feature_fusion minus transformer: -0.0606; 95% patient-bootstrap interval [-0.1008, -0.0198].

The negative response-feature result is evidence against this particular small, fixed fusion model on this cohort. Limited validation-patient diversity, redundant or unstable summaries, and model misspecification are plausible explanations; this experiment does not distinguish them. It does not establish that all response features or all trained fusion methods fail.

## Communication experiment

All arms use eight model request attempts and aggregate the final four score states. Four roles are signal, network, uncertainty and critic. Serial discussion exposes every previous proposal; independent-first hides all proposals for the first four calls and then permits discussion. No-exchange always hides other proposals. The single generalist has the same serial memory and model-call budget.

Tool arms execute eight fixed queries through native ScienceClaw SkillExecutor. They retrieve measured signal/network summaries, validation reliability and saved trained-model predictions. Both arms can select the same four executable numerical scorers from the first step. These are cached scientific outputs, not fresh physiological measurements. Tool arms also change the output interface to scorer selection; comparison against manual score vectors cannot isolate tool access alone.

The public packet includes three predictions, 15 response summaries and geometric neighbors. It excludes patient identity, clinical labels, outcome, SOZ prevalence and held-out accuracy. Later rounds receive critique, not answer-key feedback. Invalid or truncated answers consume budget and retain the last visible state, or the shared Transformer baseline when the policy has no visible state. Transport failures halt; resuming never repeats a spent attempt.

### Gemini 4 Argon control

Status: **complete_audited**. Recorded request attempts: 1680.

| Policy | Mean patient AUROC, first repeat only |
|---|---:|
| Single generalist, serial memory | 0.6985 |
| Four specialists, serial discussion | 0.6897 |
| Four specialists, independent first round | 0.6870 |
| Four specialists, no exchange | 0.6857 |
| Single generalist with numerical tools | 0.7235 |
| Independent-first specialists with numerical tools | 0.7222 |
| Same-repeat Transformer baseline | 0.7274 |

- multi_independent_first minus multi_serial: -0.0027; 95% paired patient-bootstrap interval [-0.0305, +0.0234].
- multi_independent_first minus single_serial: -0.0115; 95% paired patient-bootstrap interval [-0.0299, +0.0043].
- multi_independent_first minus multi_no_exchange: +0.0013; 95% paired patient-bootstrap interval [-0.0098, +0.0121].
- multi_independent_tools minus single_tools: -0.0013; 95% paired patient-bootstrap interval [-0.0179, +0.0150].

The independent-first minus serial contrast is primary. An interval spanning zero does not demonstrate a reliable benefit. Improvements against another arm alone do not establish that independence or communication is the cause. Additional contrasts are exploratory and unadjusted for multiplicity.

| Policy | Round 2 minus round 1 AUROC | Cases with identical final four vectors |
|---|---:|---:|
| Single generalist, serial memory | +0.0005 | 34/35 |
| Four specialists, serial discussion | +0.0011 | 6/35 |
| Four specialists, independent first round | +0.0029 | 0/35 |
| Four specialists, no exchange | +0.0047 | 0/35 |
| Single generalist with numerical tools | +0.0000 | 35/35 |
| Independent-first specialists with numerical tools | +0.0025 | 30/35 |

Identical final vectors quantify agreement, not correctness or a proven anchoring mechanism. The controlled visibility contrast is the more direct test of the anchoring hypothesis.

### Argon + Nemotron mixture

Not run. The Scientific AI credential failed the official Token Factory read-only model-permission test (HTTP 401). A dedicated Token Factory or NVIDIA model API credential is required. No Nemotron results are claimed.

## Scope and reproducibility

The public-code cohort has 35 patients, 2,145 contacts and 304 SOZ contacts, differing from the paper's 35 / 2,066 / 298. The already inspected cohort makes this an exploratory follow-up. The frozen earlier reconstruction is preserved. Numerical CNN/Transformer checkpoints are reused; no LLM fine-tuning or new EEG foundation model was trained.

Gemini is requested as gemini-4-argon with medium reasoning and a 32,768 combined completion-token cap. Every response records actual model identity, tokens, finish status and prompt hash. Argon uses the pinned native ScienceClaw LLMClient; Nemotron uses an explicit OpenAI-compatible adapter. No upstream autonomous heartbeat or Infinite publication is run.

See [the experiment README](../experiments/spes_agent_v2/README.md) for exact commands. Full local evidence is under outputs/spes_agent_v2; raw recordings, labels, credentials and generated model files are excluded from Git. The portable [summary](../examples/spes_agent_v2_summary.json) contains aggregates and protocols only.

Source evidence: frozen reconstruction outputs, fusion/protocol.json, all 25 fusion receipts and saved models, fusion_audit.json, study protocols and per-request native artifacts. Official API references: [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai/) and [Nemotron 3 Super](https://docs.nvidia.com/nemotron/latest/usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources/README.html).
