# Verified physical-pair acquisition: numerical controls

This is the first completed acquisition comparison using one real anatomical stimulation pair per query. Earlier v4-v10 column-ablation means are a different task and are not comparator results here. No agent or treatment benefit is established.

| Policy | Mean patient SOZ AUROC | Gain over random_0 | Paired 95% interval |
|---|---:|---:|---:|
| random_0 | 0.740463 | +0.000000 | [0.0, 0.0] |
| random_1 | 0.744937 | +0.004474 | [-0.03307324193677806, 0.04675996438740033] |
| coverage | 0.753486 | +0.013023 | [-0.022044527814476696, 0.04546587972550444] |
| score_focus | 0.761978 | +0.021514 | [-0.011708958754301214, 0.05202433363765379] |
| uncertainty | 0.758997 | +0.018534 | [-0.016998982421641697, 0.05360990606331313] |

The fixed primary is score_focus minus random_0. Its observed gain is about 0.0215, but the interval includes zero. This does not demonstrate a stable improvement. The highest-scoring secondary policy is not substituted for the primary.

All 35 patients have different final rankings under the fixed query rules. Mean within-patient AUROC range across five policies is 0.150587. This passes a predefined query-sensitivity check: actions matter. It is not evidence that an LLM or collaboration will select better actions.

## Same task and budgets

Five policies share the same four initial random_0 measurements and choose four further anatomical pairs. Each trajectory uses exactly eight distinct verified pairs, with the same original patient-held-out HAT as final scorer. Baseline and MTE are also recomputed from observed measurements to provide descriptive model disagreement. Every fit, normalizer and early-stop set excludes the tested patient. Numerical policies use no language-model requests.

Responses are gathered from different compact columns at different recording contacts when required by the physical-pair map. Geometry-ambiguous cells remain absent. Only the first twenty trials are used. Every candidate has one recorded parameter tuple. Provider anatomy is fsaverage with millimetre coordinates. Released preprocessing availability is assumed known and may exceed prospective knowledge.

The independent audit checked **175 trajectories / 1400 physical-pair query slots**, model/source/map lineage, patient exclusion, common prefixes, exact fixed-policy decisions and final AUROCs by pairwise comparison. **1367** query slots required different compact columns across recording contacts. This reinforces why a shared-column action was invalid.

## Next decision

Rebuild source-held-out development states with these physical queries before training contact-posterior or value models. The old v8/v9 utility models and v10 column-state classifiers are not adopted. The 20 existing inner HATs remain reusable because their original full-response contact-classification task did not require global action alignment. Fixed feature/learner recipes may be reused, but the training states must be replaced.

A subsequent single/multi-agent comparison must share tools, observations, measurements, request budget and feedback timing. Real anatomical identities can now support neuroanatomy and waveform/uncertainty roles. Independent proposals and an explicit communication ablation should test collaboration without a unilateral reviewer defaulting to one proxy. Direct numerical selection with those same tools remains a control.

All patient identities have been repeatedly inspected. These 35 patients / 2,117 contacts / 296 SOZ contacts are one exploratory cohort, not 175 independent patients. Thirty-four patients overlap the older reconstruction; patient 39 replaces patient 47. Paired bootstrap intervals condition on the recorded run and do not adjust for all research iterations. Independent confirmation is required. SOZ ranking does not measure treatment efficacy.

No physical stimulation is delivered, physiological state carry-over is not modeled, and clinical safety is not established. Computation ran locally on the RTX 4090 and CPU. No new cloud resource or model API request was used.
