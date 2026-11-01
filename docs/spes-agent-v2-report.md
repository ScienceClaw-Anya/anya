# SPES follow-up: independent deliberation and trained fusion

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

Status: **incomplete**. Recorded request attempts: 197.

Incomplete arms are not used for full-cohort performance claims.

| Policy | Completed patient cases |
|---|---:|
| Single generalist, serial memory | 2/35 |
| Four specialists, serial discussion | 2/35 |
| Four specialists, independent first round | 3/35 |
| Four specialists, no exchange | 3/35 |
| Single generalist with numerical tools | 3/35 |
| Independent-first specialists with numerical tools | 3/35 |

### Argon + Nemotron mixture

Not run. The Scientific AI credential failed the official Token Factory read-only model-permission test (HTTP 401). A dedicated Token Factory or NVIDIA model API credential is required. No Nemotron results are claimed.

## Scope and reproducibility

The public-code cohort has 35 patients, 2,145 contacts and 304 SOZ contacts, differing from the paper's 35 / 2,066 / 298. The already inspected cohort makes this an exploratory follow-up. The frozen earlier reconstruction is preserved. Numerical CNN/Transformer checkpoints are reused; no LLM fine-tuning or new EEG foundation model was trained.

Gemini is requested as gemini-4-argon with medium reasoning and a 32,768 combined completion-token cap. Every response records actual model identity, tokens, finish status and prompt hash. Argon uses the pinned native ScienceClaw LLMClient; Nemotron uses an explicit OpenAI-compatible adapter. No upstream autonomous heartbeat or Infinite publication is run.

See [the experiment README](../experiments/spes_agent_v2/README.md) for exact commands. Full local evidence is under outputs/spes_agent_v2; raw recordings, labels, credentials and generated model files are excluded from Git. The portable [summary](../examples/spes_agent_v2_summary.json) contains aggregates and protocols only.

Source evidence: frozen reconstruction outputs, fusion/protocol.json, all 25 fusion receipts and saved models, fusion_audit.json, study protocols and per-request native artifacts. Official API references: [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai/) and [Nemotron 3 Super](https://docs.nvidia.com/nemotron/latest/usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources/README.html).
