> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).

# Learned diagnostic measurement value: fixed numerical feasibility test

The prespecified primary policy scored **0.722233 mean patient SOZ AUROC**, with gain **-0.013427** over the same-prefix random control. The spending gate did not pass. This is a numerical-tool experiment; it establishes no multi-agent or treatment benefit.

## All fixed policies

| Policy | Mean patient AUROC | Gain over random_0 | Paired 95% interval | Different final query sets |
|---|---:|---:|---:|---:|
| random_0 | 0.735660 | +0.000000 | [0.0, 0.0] | 0 / 35 |
| random_1 | 0.734175 | -0.001485 | [-0.03071816516944956, 0.02947970326793999] | 35 / 35 |
| coverage | 0.726479 | -0.009182 | [-0.04483694312862638, 0.024970577365528174] | 35 / 35 |
| score_focus | 0.737088 | +0.001428 | [-0.02840132634751517, 0.02968494607130221] | 35 / 35 |
| uncertainty | 0.709982 | -0.025679 | [-0.06896187270269781, 0.011366480782558065] | 35 / 35 |
| phase0_bag_mean | 0.747506 | +0.011846 | [-0.01618085926321645, 0.04046074339334954] | 35 / 35 |
| phase1_bag_mean | 0.722233 | -0.013427 | [-0.04464011682138883, 0.015324800380990843] | 35 / 35 |
| phase1_relative_mean | 0.708971 | -0.026690 | [-0.05735850020659652, 0.004662763293661377] | 35 / 35 |
| phase1_ridge | 0.733712 | -0.001948 | [-0.03203567168570731, 0.02806823147112463] | 35 / 35 |
| phase1_robust | 0.731252 | -0.004408 | [-0.028859672227173172, 0.022471646205596357] | 35 / 35 |

All policies start with the same four random_0 measurements, then choose four more. They use exactly eight real released stimulation columns and the unchanged original v4 HAT as the final scorer. These controls therefore differ from v4 numerical policies that selected all eight measurements independently. Numerical policies use zero language-model calls.

The fixed primary is phase1_bag_mean, not whichever row scores highest afterward. Its one-sided 95% patient-bootstrap lower bound versus random_0 is -0.039116. Before opening these outcomes, further LLM spending required the primary to exceed the strongest fixed numerical control by more than 0.005, have a positive lower bound versus random_0, and change at least ten final query sets. This is a cost-control rule, not clinical validation or a ranking veto.

## What was trained

Twenty inner HAT models provide source-held-out replay for each outer development set. Every target patient is absent from its source neural fit and early stopping. Each outer test group is absent from all source loading, target banks and value regressors. The 28 development patients per outer fold supply four query states each, with up to six candidate measurements per state.

Two phases were fixed before target generation: random-prefix replay, then replay using the phase-zero learned policy. Each phase contributes 3,360 target rows across outer folds; these are correlated patient-fold examples, not 6,720 independent patients. Each row predicts the subsequent change in within-patient SOZ AUROC. Clinical labels and added responses are supervised targets only. The 20 features use known geometry, observed waveform summaries and partial-response scores; hidden candidate waveforms, test labels and full-response predictions are excluded.

Each outer learner contains one weighted ridge model, five patient-exclusion gradient-boosting models for absolute gain and five for within-state relative gain. Fixed settings are shared across folds. Both phases total 110 small regressors. Bag disagreement is descriptive and is not a calibrated uncertainty interval. Phase-one development rollouts use the phase-zero learner fitted on that development set; this is iterative training, not independent validation of development performance.

This is a project-specific adaptation of the idea in [Learning Active Learning from Data](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf): learn acquisition value from replay, then address the distribution induced by the learned policy. The original acquires labels and updates a classifier; here a query acquires waveform groups and the final HAT remains fixed. It is not a reproduction of that paper. [EDDI](https://proceedings.mlr.press/v97/ma19c.html) was considered at abstract level and not implemented. Access status and atomic claims are recorded in research/spes-v8/.

## Independent checks

The audit verified 20 source neural fits, reconstructed all 6,720 feature rows from observed data, recomputed every target change and all 350 final AUROCs using pairwise comparisons, checked patient/model/bank/protocol provenance, and reproduced the original random_0 scores. No API request was made in v8.

The RMS cache is tested against original feature extraction. Unit tests change hidden waveforms without changing features, and complement supervised labels without changing features or replay choices. These checks do not replace external replication.

## Interpretation and limits

The primary tool did not pass the spending screen. No additional Gemini pilot is launched for this version, and no secondary result is substituted for the fixed primary. The next investigation must change a substantive source of predictive information or model assumptions, rather than spending more calls on the same tools.

A post-hoc development diagnostic found that the phase-one predictor selected sampled candidates about 0.0262 AUROC above the state average when evaluated by fitted models, but about 0.0022 below average using the bag that excluded that patient. Mean within-state rank correlation was about -0.0086. This indicates poor transfer of candidate ordering. It is not fully nested validation because source models for other patients may have used the excluded patient. The hindsight best of six candidates used hidden labels and is not an executable policy. All original methods and the fixed primary remain unchanged.

Possible limitations include weak candidate-specific information before measurement, only 28 development patients per fold, noisy immediate AUROC targets, and a shift from inner scorers trained on 17 patients to outer scorers trained on 21. These are hypotheses, not identified causal explanations. Training residuals are not held-out performance.

The current processed cohort is 35 patients / 2,117 contacts / 296 SOZ contacts. Thirty-four identities overlap the older reconstruction; patient 39 replaces patient 47. Comparisons within this experiment share the same cohort. Repeated inspection makes all new outcomes exploratory despite patient-separated fitting. Patient bootstrap intervals are conditional on recorded runs and do not adjust for all adaptive research iterations. A fresh cohort and repeated generations are needed for a stronger claim.

This is retrospective diagnostic acquisition. Stimulus IDs are anonymous array columns, geometry/preprocessing availability are assumed known, and treatment, prospective safety and physiological state carry-over are not modeled. SOZ ranking is not seizure control. All new training/inference ran locally on the RTX 4090 and CPU; no new Nebius resources were created.
