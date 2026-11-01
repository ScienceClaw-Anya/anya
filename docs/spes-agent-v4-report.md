# SPES research v4: evidence before additional agent calls

This is an adaptive exploratory study. The previously stopped v3 study remains stopped; its unused API budget has not been resumed.

## Why the method changes

Previous agents mostly recombined the same fixed observations, and v3's safety gate made every executable option identical. More debate could not improve those rankings. The new programme first tests cheap learned ranking controls, then adds complementary physiological representations, then tests whether agents can acquire useful new measurements.

## Fixed cheap screen on the old cohort

All methods use the same old first-repeat 35-patient outer partitions and v3's patient-isolated OOF base predictions. No new neural fit or LLM request was used. These are exploratory outcomes on a repeatedly inspected cohort.

| Method | Mean patient AUROC | Gain over Transformer | Paired patient-bootstrap 95% interval |
|---|---:|---:|---:|
| transformer | 0.727418 | +0.000000 | [+0.000000, +0.000000] |
| conditional_ridge_no_global_veto | 0.724011 | -0.003407 | [-0.012185, +0.003910] |
| contact_logistic | 0.691029 | -0.036389 | [-0.078730, +0.001835] |
| patient_pairwise_ranker | 0.725394 | -0.002023 | [-0.015749, +0.011796] |
| pairwise_ranker_10pct | 0.728102 | +0.000685 | [-0.000148, +0.001577] |

Removing the global veto did not help. The 10% pairwise-ranker blend had only a tiny, uncertain gain. These routes did not justify another large static-agent run.

## Complementary multi-trial models

A [2026 preprint](https://doi.org/10.64898/2026.07.27.26358819) and its [public author code](https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT) motivate testing trial-averaged, mean-trial-embedding (MTE), and hierarchical-attention (HAT) representations. The preprint models cross-trial and cross-channel dependencies; it does not study LLM collaboration. Its reported SOZ improvement did not establish postsurgical benefit. Full HTML Methods/Results/Discussion were verified in Chrome. The literature search was focused and non-exhaustive.

The newly released processed arrays contain **35 patients, 2,117 contacts and 296 SOZ contacts**. The old project cohort contains 2,145 contacts and 304 SOZ contacts. These are not interchangeable denominators.

The new local runner fixes five patient-held-out folds and published architecture presets before evaluation. All fitting, normalization and early stopping exclude each outer test patient. The best validation weights are cloned. We do not load upstream checkpoint pickles. There is no new hyperparameter search. The runner uses the paper's 25-epoch maximum and patience 5, not the CLI defaults. Streamed float64 statistics, batch padding, weight snapshot handling and deterministic seed assignment differ from upstream. This is an adaptation with one repeat, not an exact reproduction of the full paper.

| Same-cohort method | Mean patient AUROC | Gain over HAT | Paired 95% interval |
|---|---:|---:|---:|
| baseline | 0.713848 | -0.035700 | [-0.097025, +0.020361] |
| mte | 0.668299 | -0.081249 | [-0.169867, +0.005512] |
| hat | 0.749548 | +0.000000 | [+0.000000, +0.000000] |
| rank_mean | 0.749627 | +0.000079 | [-0.049927, +0.048816] |
| validation_best | 0.725338 | -0.024210 | [-0.088683, +0.032205] |
| pairwise_from_validation | 0.730538 | -0.019010 | [-0.079956, +0.038941] |
| pairwise_10pct | 0.751150 | +0.001602 | [-0.002139, +0.005403] |

HAT minus the retrained same-data baseline: +0.035700, paired 95% interval [-0.020360742642110147, 0.09702506522704701]. These numerical results alone establish no multi-agent advantage.

## New task: acquire eight real SPES measurements

The active replay reveals one real stimulation-column response at a time. The three trained models recompute contact scores using only revealed columns. Hidden waveforms, SOZ labels, outcomes and full-response predictions never enter agent prompts. The final scorer is always the same trained HAT; agents choose measurements and cannot write probabilities.

Known distances and the author-released preprocessing availability map are assumed available before acquisition. That map may include knowledge unavailable prospectively. No real stimulation is delivered; clinical safety, voltage choices and prospective acquisition cost are outside this benchmark.

Every policy receives eight measurement slots and eight native ScienceClaw public-tool queries. Each LLM policy has at most eight Gemini 4 Argon attempts, medium reasoning, no retries or replacement calls. Numerical controls use no LLM calls. The single agent acts sequentially and sees all earlier feedback. Four specialist roles (coverage, hypothesis testing, uncertainty, audit) act independently in batch one, then share real feedback and rationales in batch two. The no-exchange control shares the same type of measured feedback but hides other roles' rationales. Duplicate proposals use ordered alternatives and a declared geometric fallback. This gives the single agent a strong sequential-feedback control.

The initial five patients are fixed in advance (one from each outer fold). A numerical feasibility gate must show that changing queries meaningfully changes final rankings before any paid agent pilot. A successful pilot is not a final efficacy result; later patients remain separately reported.

Numerical pilot feasibility: launch=True, mean within-patient AUROC range=0.210020.

### Matched five-patient pilot

| Active policy | Completed pilot patients | Mean AUROC at eight measurements | Accepted model decisions |
|---|---:|---:|---:|
| coverage | 5 | 0.668790 | 0 |
| multi | 5 | 0.633065 | 37 |
| multi_no_exchange | 5 | 0.598539 | 40 |
| random_0 | 5 | 0.663276 | 0 |
| random_1 | 5 | 0.646120 | 0 |
| score_focus | 5 | 0.677448 | 0 |
| single | 5 | 0.667255 | 32 |
| uncertainty | 5 | 0.659059 | 0 |

Multi minus single on the same five patients: **-0.034191**, exploratory paired 95% interval [-0.11748917748917749, 0.030181279791997716]. The multi-agent pilot is not expanded to the remaining 30 patients. This is a small negative feasibility result, not proof that all collaboration methods fail.

The 120 model attempts produced 109 accepted decisions and 11 declared fallbacks. Rejected outputs include oversized rationales and unavailable alternatives; no spent request was replayed. A transient Windows lock interrupted one numerical artifact-index write; a verified completed temporary transaction was recovered and the unfinished numerical trajectory resumed without an API request.

### Complete numerical acquisition controls

| Policy | Completed patients | Mean AUROC at eight measurements |
|---|---:|---:|
| random_0 | 35 | 0.735660 |
| random_1 | 35 | 0.715779 |
| coverage | 35 | 0.693129 |
| score_focus | 35 | 0.709189 |
| uncertainty | 35 | 0.712949 |

The 35-patient numerical means must not be directly compared with the five-patient LLM means. The next follow-up tests sparse-observation training: masking individual trials during full-response training does not necessarily train a scorer for whole missing stimulation positions. A controlled, fixed-sequence scorer test precedes further LLM calls.

## Limits and next decision

All 35 patient identities have been examined in earlier work. Patient-isolated training prevents direct train/test leakage, but repeated research on this cohort still makes improvements exploratory. Preserve all trials; do not keep tuning to a reported test winner. A meaningful claim needs stable paired gains versus both single-agent and the strongest numerical control, repeat-generation sensitivity, and ideally a fresh external cohort. SOZ agreement is not treatment effectiveness.

All current neural training and waveform inference run on the local workstation's RTX 4090. Gemini inference uses Google's API. No new Nebius resources are created.

