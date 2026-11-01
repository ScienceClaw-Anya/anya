# Physical-state contact specialists: ranking gain not established

The fixed logistic specialist scored **0.733662 mean patient SOZ AUROC** at eight verified physical measurements, versus HAT **0.740463**. Gain **-0.006801**, paired 95% interval **[-0.063261, 0.048103]**. The predefined next-tool gate fails. These classifiers are not adopted as an improved localization scorer, and no new Gemini comparison is launched for this version.

| Measurements | Scorer | Mean patient AUROC | Gain over HAT | Paired 95% interval | Mean patient Brier |
|---:|---|---:|---:|---:|---:|
| 4 | hat | 0.705031 | +0.000000 | [0.0, 0.0] | 0.6046875810892831 |
| 4 | logistic | 0.726141 | +0.021110 | [-0.044170449340038, 0.08912876263335967] | 0.114197311520325 |
| 4 | boosting | 0.667543 | -0.037488 | [-0.0910410398088938, 0.02075121218113359] | 0.11352315674159681 |
| 4 | rank_mean_boost | 0.708887 | +0.003856 | [-0.02593506215064246, 0.036840768385298484] | None |
| 8 | hat | 0.740463 | +0.000000 | [0.0, 0.0] | 0.6003415561052277 |
| 8 | logistic | 0.733662 | -0.006801 | [-0.06326115578711076, 0.04810349009847081] | 0.1133779131844974 |
| 8 | boosting | 0.700272 | -0.040191 | [-0.08986985827273891, 0.008980171533826515] | 0.11215626698129232 |
| 8 | rank_mean_boost | 0.739465 | -0.000998 | [-0.028499831735646113, 0.02725193981177095] | None |

Four-measurement logistic ranking is numerically higher than HAT, but its interval also includes zero. It remains secondary and does not replace the fixed primary. The rank-mean score is not interpreted as a probability and receives no Brier score.

## Meaningful training and independent checks

All input states use the verified physical-pair map. Ten small classifiers were fitted using 28 development patients per outer fold, equally weighted. Twenty existing inner HATs generate source-held-out scores, with the target patient absent from neural fitting and early stopping. Outer patients never enter fitted contact models. Twenty-six observed-only features describe morphology, sample-index energy, FFT-bin energy, geometry and HAT scores; unknown 128-feature semantics are not assumed. No new neural fit or LLM request was used.

The audit reconstructed **42,340 contact/state feature rows**, checked label and physical-action identity, patient/source/model provenance, replayed both fitted classifiers on every test state, and independently recomputed all **280** AUROC/Brier results. Original HAT predictions exactly match the v11 same-query control. Recorded HAT outputs were not all re-inferred by this audit.

The 42,340 rows are correlated repetitions across states/folds, not new patients. This remains a repeatedly inspected 35-patient cohort. The outer-label exclusion prevents direct leakage but does not remove adaptive research bias. No general medical or collaboration claim follows from these results.

## A separate problem: absolute score interpretation

Mean raw HAT score across held-out patients is **0.825902**, while mean patient SOZ prevalence is **0.132328**. Its eight-query Brier score is about **0.6003**, versus about **0.1134** for logistic. This gap concerns absolute output quality; it does not make the logistic ranking better. A development-prevalence constant is still needed as a proper calibration comparator.

The local training code applies inverse-class/inverse-patient sampling and a positive-class weighted BCE loss. The author [supporting methods, S6, PDF pp. 6-7](https://www.medrxiv.org/content/medrxiv/early/2026/07/31/2026.07.27.26358819/DC1/embed/media-1.pdf?download=true) describe the same combination. This is an intentional ranking/training strategy, not evidence that raw sigmoid scores equal unweighted clinical probabilities. The preprint supplement was read directly; the associated article is a preprint version, not a peer-reviewed clinical validation.

This motivates a separately frozen calibration study before agents use absolute confidence. It should compare training-only analytic prior adjustment, patient-separated sigmoid calibration, a development-prevalence constant and an adaptive decision boundary; then test whether corrected uncertainty changes physical query selection. Weighting is a mechanistic hypothesis for part of the gap, not the demonstrated sole cause. Sparse-input shift and incomplete fitting may also matter.

The supplementary protocol uses repeated pulses and variable recorded currents. The benchmark budget is eight anatomical-pair response groups, each containing up to twenty retained trials; pulse count, charge, timing and safety are not equated across query sets. Each action uses its recorded condition. Anatomical-position effects cannot be separated causally from dose with this replay.

The unused 128-feature block remains unresolved after checking the primary Methods, supplement and available model/dataset code. It is not called a prestimulus baseline or spectral block and is excluded from the specialist features.

All computation ran locally on the workstation. Reports and code are in English, raw data/labels/weights stay outside Git, and no new cloud resource was created. The endpoint is clinical SOZ ranking, not seizure-control efficacy.
