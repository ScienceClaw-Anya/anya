# Calibration improved modestly; physical query selection did not

The fixed learned-boundary acquisition policy scored **0.744035** mean patient SOZ AUROC, versus **0.761978** for the frozen score-focus numerical control. Gain **-0.017943**, paired 95% interval **[-0.043149, +0.005348]**. The acquisition gate fails. No Gemini comparison is launched for this hypothesis, and no multi-agent advantage is claimed.

## Calibration is a separate outcome

| Eight-group score | Mean patient Brier (lower is better) | Gain over development-prevalence constant | Paired 95% interval | AUROC |
|---|---:|---:|---|---:|
| raw | 0.600342 | -0.485122 | [-0.5375628194733922, -0.4317489797470239] | 0.740463 |
| analytic_prior | 0.117972 | -0.002752 | [-0.005608917819664381, 0.00012115128566430033] | 0.740463 |
| constant | 0.115220 | +0.000000 | [0.0, 0.0] | 0.500000 |
| raw_platt | 0.117401 | -0.002181 | [-0.004661928224387866, 0.0002087060768335993] | 0.740463 |
| prior_platt | 0.115632 | -0.000412 | [-0.0012575576240347235, 0.00041774472077442723] | 0.645928 |
| centered_platt | 0.111669 | +0.003551 | [0.00035833480118441225, 0.0064692448175865235] | 0.740463 |

Patient-centered sigmoid calibration beats the constant by 0.003551, with a positive paired interval, but misses the fixed practical-gain requirement of 0.005. Its eight-group AUROC is unchanged from HAT. This is a modest score-quality finding on this cohort, not better localization or clinical probabilities. Four-group outcomes remain secondary in the machine-readable summary. Some source-fitted sigmoid slopes were nonpositive and followed the predeclared constant fallback; this explains AUROC reductions for those variants rather than a silently reversed ranking.

The analytic prior hypothesis reduces the large raw-score offset, but does not beat the development-prevalence constant. Correcting average prevalence is insufficient to establish individual calibration. The ideal odds derivation is not a proof that finite, sparsely queried neural outputs have that distribution.

## All physical acquisition variants

| Policy | Mean patient AUROC | Gain over score-focus | Paired 95% interval |
|---|---:|---:|---|
| prior_uncertainty | 0.756911 | -0.005067 | [-0.02684143180295692, 0.017553411770725686] |
| learned_boundary | 0.744035 | -0.017943 | [-0.04314946196869309, 0.005348355070041046] |
| zero_boundary | 0.749923 | -0.012054 | [-0.03677032850982561, 0.012661031954747418] |

All policies receive the same four random0 initial physical groups and four adaptive choices. The final scorer is the unchanged outer HAT applied only to revealed responses. The prior-corrected disagreement policy and matched zero-logit boundary control also fail to beat score-focus. No secondary variant replaces the primary.

## Training, evidence and checks

Seventy-five small one-dimensional calibrators were fitted on 28 development patients per outer fold, using twenty existing source-held-out HATs and audited physical states. Source patients were absent from neural fitting and early stopping. Calibration weights give each patient equal influence at natural class prevalence. Median Youden boundary offsets are learned only from those development patients.

The [official HAT preprint supplement](https://www.medrxiv.org/content/medrxiv/early/2026/07/31/2026.07.27.26358819/DC1/embed/media-1.pdf?download=true), S3/S6, supports the threshold and weighted-training motivation. The acquisition rule in logit-MAD units is a new project adaptation. The source is a preprint with an inspected official supplement, not a peer-reviewed clinical validation. Atomic evidence locations are recorded in `research/spes-v13/claims.json`.

Independent audit refitted **75** calibrators, enumerated **700** patient/state boundaries independently, recomputed **420** calibration outcomes, and reconstructed **420** adaptive choices across **105** physical trajectories. All physical identities, patient splits and frozen hashes passed. Stored neural scores were not all re-inferred in the audit.

The study used the local workstation, zero new neural fits and zero API calls. Eight response groups do not equal eight pulses or equal current, charge or time. The repeatedly inspected 35-patient cohort is exploratory; SOZ ranking does not measure seizure-control efficacy. All failures are retained. The next distinct hypothesis is supervised physical query value, with genuinely separate specialist feature views, before any new paid collaboration comparison.
