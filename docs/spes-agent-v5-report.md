# Sparse-observation training: negative result

Ten new local fits tested whether matching training inputs to an eight-measurement acquisition task improves the numerical scorer. The fixed experiment did **not** improve performance and these weights are **not adopted**. No Gemini requests were made.

## Controlled comparison

The same 35 patients and five outer splits were used. Each model fitted 21 patients and used seven different patients for early stopping. Every outer test patient remained excluded from fitting, normalization and early stopping. Training sampled 1-8 whole stimulation columns per patient per batch. Early stopping used two fixed eight-column validation subsets. Architecture/optimizer presets, the 25-epoch ceiling and patience 5 were fixed.

The test replays exactly the five v4 numerical policies' existing eight-stimulation sets. Only the scorer changes. No new acquisition controller is tested by this replay. The primary sequence is random_0; the other sequences are reported as declared secondary checks.

| Fixed measurement sequence | Original HAT | Sparse baseline | Sparse HAT | Sparse rank mean |
|---|---:|---:|---:|---:|
| random_0 | 0.735660 | 0.700899 | 0.618412 | 0.701245 |
| random_1 | 0.715779 | 0.709011 | 0.608124 | 0.692166 |
| coverage | 0.693129 | 0.657091 | 0.598172 | 0.659899 |
| score_focus | 0.709189 | 0.701758 | 0.602796 | 0.690726 |
| uncertainty | 0.712949 | 0.671752 | 0.599401 | 0.675879 |

Primary sparse-HAT minus original-HAT gain: **-0.117249**, paired patient-bootstrap 95% interval **[-0.18718236270230054, -0.05048790933599992]**. Sparse HAT was worse in 23 patients and better in 11; ties account for the remainder.

## What this resolves

Changing the mask alone did not produce a better scorer. Training on fewer observations also reduces the information available to learn from. Published full-response hyperparameters may not suit sparse training from scratch; extra trial masking and dropout may further weaken that signal. These are hypotheses, not causes established by this experiment. Warm-start adaptation or a learned acquisition-value model would require separate preregistered tests.

The negative result does not prove that missing-column mismatch is irrelevant, nor that all sparse models fail. It does rule out adopting this particular fixed recipe as an improvement. The next collaboration test retains the stronger original HAT and matches single/multi feedback timing.

## Audit and scope

Verified 10 new fits and 175 held-out query sets, with independent AUROC replay and source/checkpoint provenance. The local RTX 4090 was sufficient; no cloud job or API call was needed.

This is adaptive exploration on a repeatedly inspected single-centre cohort. It is not untouched external validation and it establishes no clinical or multi-agent benefit. All negative results remain recorded.
