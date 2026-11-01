# Physical query-value specialists

The fixed conservative expert policy scored **0.727182** mean patient SOZ AUROC. Against the frozen v11 score-focus controller, gain was **-0.034795**, paired 95% interval **[-0.07749116446847969, 0.011200165142386639]**. Against the all-feature boosted controller, gain was **+0.021060**, interval **[-0.020221529282130758, 0.06955902986447879]**.

The fixed next-agent feasibility gate **fails**. These are trained numerical specialists, not a completed Gemini collaboration comparison. No multi-agent improvement or clinical benefit is established.

| Controller | Mean patient AUROC | Gain over score-focus | Paired 95% interval |
|---|---:|---:|---|
| expert_consensus | 0.727182 | -0.034795 | [-0.07749116446847969, 0.011200165142386639] |
| monolithic | 0.706122 | -0.055855 | [-0.09208622275872146, -0.01892897145568402] |
| ridge | 0.723416 | -0.038562 | [-0.08238163319945117, 0.006927181743160169] |

## What was trained

Three fixed feature views describe geometry/coverage, neural-state scores and already observed waveform morphology. Each specialist predicts the one-query SOZ-AUROC change using a Huber boosted regressor. The fixed consensus uses their mean minus half their spread. Spread is a disagreement heuristic, not a calibrated confidence interval. An all-feature boosted regressor and an all-feature ridge model receive the same source targets and patients.

The 25 fitted regressors use 3,360 correlated candidate/state targets, derived from twenty existing source-held-out HATs and 28 development patients per outer fold. Each candidate slate is determined before its future response and clinical label are used to compute the supervised target. The slate includes the next seeded random query, coverage and raw-rank focus leaders, plus seeded alternatives. All current features use selected responses only. Source target patients are absent from source neural fitting and early stopping; outer patients never enter model fitting.

Only identical, audited physical random0-prefix scores are reused from v12. Responses for alternative physical pairs are newly inferred with the correct per-recording-contact index. No old array-column utility bank or trained value learner is adopted. Relative scores use an unsupervised patient logit median/MAD transform, not learned v13 parameters or clinical probabilities.

## What the comparison can resolve

Specialists proposed different top choices in **132/140** consensus-policy steps. Diversity alone cannot demonstrate useful complementarity. The primary must beat score-focus and the same-information monolithic model by more than 0.01, each with a positive paired interval, before an agent trial for this tool version.

Every controller receives the same four initial physical response groups, then four adaptive groups. The unchanged outer HAT supplies all final rankings. Candidate choices, current scores, observed-only feature tables and every specialist output are recorded at each round. All three controllers and all five frozen v11 comparisons remain in the machine-readable summary; a secondary winner cannot replace the primary.

## Verification and limitations

Audit reconstructed all **3360** source candidate feature/target rows and **560** source states, checked frozen source/model/map/raw hashes and patient exclusion, then replayed **420** adaptive decisions across **105** trajectories. Every final AUROC was independently recomputed with scikit-learn. Source next-query scores match audited physical prefix scores. Alternative-response and runtime neural outputs were checked from stored predictions, not all re-inferred in the audit.

Random-prefix training may differ from learned-policy runtime states. Source HATs fit 17 patients and outer HATs fit 21, creating possible score/input shift. Model fitting uses patient-balanced rows, but these are not independent new participants. Repeated same-cohort research creates adaptive-selection bias; nominal bootstrap intervals are not adjusted for the full research search. A favorable result would still require a separate matched-budget Gemini comparison and external confirmation.

The local workstation performed this experiment with zero new neural fits and zero API calls. The anatomical-pair response-group budget does not equate current, pulse count, charge, time or safety. SOZ contact ranking measures retrospective diagnostic discrimination, not seizure-control efficacy.
