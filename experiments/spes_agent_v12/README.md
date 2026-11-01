# Contact specialists from verified physical measurements

This completed numerical experiment rebuilt all training states using true
anatomical-pair queries. It reuses the **feature/fitting recipe only** from the
stopped v10 prototype; no column-state bank, fitted weight or old query-utility
model is adopted. Gemini is not fine-tuned here, and no API call is made.

Twenty existing inner HATs provide source-held-out scores at four, five, six,
seven and eight physical measurements. Each target patient was absent from
source fitting and early stopping; outer patients are excluded from fitted
contact models. Twenty-six observed-only morphology, sample-index, FFT-bin and
HAT features feed fixed logistic and gradient-boosting classifiers. Neither
clinical labels nor unobserved waveforms are inputs to feature extraction.

The ten small fits use 28 development patients per outer fold, equally weighted.
The 42,340 contact/state rows are correlated repetitions, not independent
patients. Settings are fixed, with no hyperparameter search. Learned score
calibration is assessed numerically and is not a clinical-risk claim.

The fixed primary is logistic minus unchanged HAT at the same eight random_0
physical measurements. Four-measurement results, boosting and the fixed rank
mean remain secondary. No post-hoc test winner replaces the primary. This stage
tests a localization tool, not agent collaboration or active-query improvement.

```powershell
python -m experiments.spes_agent_v12.experiment --project . --output outputs/spes_agent_v12 --stage freeze
python -m experiments.spes_agent_v12.experiment --project . --output outputs/spes_agent_v12 --stage banks
python -m experiments.spes_agent_v12.experiment --project . --output outputs/spes_agent_v12 --stage fit
python -m experiments.spes_agent_v12.experiment --project . --output outputs/spes_agent_v12 --stage evaluate
```

Complete each stage before the next. All five fitted-model hashes are frozen
before outer evaluation. Every test patient is held out, and the original HAT
must exactly reproduce v11 on identical verified physical queries. Read the
[action-grain correction](../../docs/spes-physical-action-audit.md) first.
Raw data, clinical labels, generated training banks and model weights remain
outside Git. Execution uses the local workstation; no new cloud resource is
created. Same-cohort outcomes remain exploratory after repeated inspection.


The fixed ranking primary failed: eight-query logistic AUROC 0.733662 versus HAT
0.740463, gain -0.006801, 95% interval [-0.063261, 0.048103]. The classifiers are
not adopted as improved localization scorers. Independent audit reconstructed all
42,340 training feature rows and replayed 280 test scorer/state results. See the
[English result report](../../docs/spes-agent-v12-report.md). No new Gemini run
was launched. Score calibration is a separate future test, not an established
ranking or collaboration improvement.
