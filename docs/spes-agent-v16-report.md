# Independent physical-proposal pilot

Multi-agent mean patient AUROC is **0.794413**, versus single **0.795542** and score-focus **0.811940**. The frozen expansion gate **fails**. Five patients do not establish general superiority.

| Arm/control | Mean patient SOZ AUROC |
|---|---:|
| random_0 | 0.797918 |
| random_1 | 0.776599 |
| coverage | 0.805799 |
| score_focus | 0.811940 |
| uncertainty | 0.789102 |
| single | 0.795542 |
| multi | 0.794413 |
| multi_reason_hidden | 0.798779 |

| Comparator | Multi gain | Paired nominal 95% interval |
|---|---:|---|
| single | -0.001129 | [-0.016422466422466364, 0.015140932513813876] |
| multi_reason_hidden | -0.004365 | [-0.013095238095238049, 0.0] |
| random_0 | -0.003504 | [-0.08181018604747414, 0.07480165075304186] |
| random_1 | +0.017814 | [-0.08526885650326643, 0.08424908424908431] |
| coverage | -0.011386 | [-0.03137973137973131, 0.004761904761904745] |
| score_focus | -0.017526 | [-0.07817384175714848, 0.027914852443154326] |
| uncertainty | +0.005311 | [-0.02445054945054941, 0.032692307692307666] |

Accepted model decisions: **120/120**. Every language-agent trajectory uses eight actual response groups, eight attempts and four feedback rounds. Numerical controls use the same measurement budget with zero model calls. Actual reported input/output tokens are **3,364,204/113,699**; **0** attempts have unavailable usage, not zero usage.

Every physical trajectory was re-inferred with all three original scorers. Public evidence, history visibility, native hashes/lineage, API schemas, rank votes and final AUROC were checked. Requests never contain clinical labels or future response summaries. Twenty-one recording-name ambiguities remain explicit rather than guessed.

## What collaboration actually changed

| Arm | Independent top-choice agreement | Mean top-three Jaccard | Merged choice matches coverage leader | Matches score-focus leader |
|---|---:|---:|---:|---:|
| single | 18/20 | 0.900 | 12/20 | 12/20 |
| multi | 18/20 | 0.725 | 13/20 | 7/20 |
| multi_reason_hidden | 14/20 | 0.635 | 13/20 | 9/20 |

The two multi-agent roles selected the same first choice in **18/20 rounds**, despite independently fixed current-round prompts. The generalist pair also agreed in 18/20. Specialized prompts reduced agreement lower in the three-choice lists, without creating a better first-choice decision. Both roles use the same foundation model and all common evidence; their role names do not imply independently trained epilepsy specialists.

Multi matched the coverage leader in 13/20 rounds and score-focus in 7/20; these leader counts can overlap. This is consistent with convergence on shared geometric heuristics. It does not establish that a particular heuristic caused the performance gap. Unlike the old serial design, no reviewer can veto the other proposal.

Full and rationale-hidden multi-agent runs had different final rankings in only **1/5 patients**. Hiding peer explanations did not worsen the mean outcome. With one stochastic run per arm and five patients, this does not isolate a general causal effect of communication. Access to feedback is verified from requests and lineage; useful reflection is not proven by rationale text.

Recorded stimulation-condition counts and uniform-candidate reference counts are retained in the machine-readable behavior summary. Raw 4/8 mA counts alone cannot demonstrate preference because availability differs; pulse/current/charge budgets are not equated.

The fixed gate fails, so the 30-patient extension is not run. All120 specifically authorized attempts are consumed; no replacement or additional request is made for this failed hypothesis. The next local hypothesis is to train relative candidate-value rankings from existing source-held-out physical counterfactuals, instead of relying on prompt specialties or absolute one-query gain estimates. Such numerical feasibility would still not establish Gemini collaboration value.

A changed ranking or rationale does not prove communication improved decisions. Backend stochasticity, the small selected pilot and repeated cohort inspection limit causal interpretation. Nominal intervals are not adjusted for the complete research search. No treatment, safety or external validation claim follows. Requests and patient-derived traces remain outside Git.
