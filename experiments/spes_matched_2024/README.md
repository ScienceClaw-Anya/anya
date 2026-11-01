# Complete legacy-cohort comparison

**Training and evaluation complete; audit stopped by the user.** All 25 candidate
fits and 175 patient-repeat evaluations are complete. The independent audit was
stopped at the user's explicit request on 2026-11-01 after 18 fold checks.
There is no complete audit receipt. All local results and partial checks remain
preserved, but this experiment is excluded from the requested presentation.
Do not resume the audit without a new user request.

This new local experiment tests an adapted HAT architecture against the exact
complete-cohort Transformer reconstruction that produced **0.7201 AUROC**.
It uses the same 35 patients, 2,145 contacts and 304 SOZ labels, all 25 original
seed/fold splits, original training-only normalization, input arrays, augmentation,
class weighting, batch size, AdamW settings and validation stopping rule.
The original 25 Transformer fits are reused by frozen hashes, without choosing
favorable folds. New test outcomes stay sealed until all 25 candidate fits finish.

The old arrays contain response means and separate standard deviations, not
individual trials. The candidate consumes only the convergent mean, just like
the original base Transformer. SD never becomes a trial or a new feature.
The HAT architecture has one mean token per response channel. Its embedding is
adapted to the legacy 509-feature representation. This cannot test the full
cross-trial architecture or establish the newer eight-position policy's value.

The unchanged legacy normalizer has an in-place aliasing quirk: after writing
distance into feature zero, its shifted response view also writes that distance
into feature one. Both comparator and candidate receive exactly this historical
representation. We preserve it for the requested matched test, document it, and
do not silently fix only one arm. The remaining 507 features carry response
samples. A corrected preprocessing experiment would need a separate protocol
and both models retrained.

HAT depth and attention heads are fixed from the published architecture.
Original baseline learning rate, dropout and augmentation are retained to reduce
changes. There is no new hyperparameter search. Parameter counts and FLOPs differ,
so equal stopping rules are not an exact compute-match claim. The entire cohort
has been examined in prior work; this is exploratory, not untouched validation.

```powershell
python -m unittest experiments.spes_matched_2024.test_matched -v
python -m experiments.spes_matched_2024.experiment --project . --output outputs/spes_matched_2024 --stage run --workers 2
python -m experiments.spes_matched_2024.audit --project . --output outputs/spes_matched_2024
python -m experiments.spes_matched_2024.report --project . --output outputs/spes_matched_2024
```

The runner requires both pinned upstream repositories and original processed
legacy data and reconstruction artifacts. It makes **zero model API calls**.
Local RTX 4090 resources are sufficient. Generated labels, normalized tensors,
patient predictions and weights remain ignored. The final paired bootstrap
averages five repeats per patient first, then resamples 35 patients. It does not
treat 175 repeated patient cases or electrodes as independent participants.

Results must distinguish this numerical architecture comparison from the
existing full-response agent comparison and the five-patient physical-query
pilot. No numerical gain alone demonstrates multi-agent superiority.

## Sources

- [Norris et al., MLHC 2024](https://proceedings.mlr.press/v252/norris24a.html),
  with the [original implementation](https://github.com/norrisjamie23/Localising_SOZ_from_SPES)
  pinned at `90062e8c5fe5593713125eaf5f573faedb739f65`.
- [HAT implementation](https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT),
  pinned at `11f65182a2114440f1425656f91d0ebec51ee726`. This experiment adapts
  its architecture and does not claim an exact reproduction of its publication.
- [OpenNeuro ds004080 v1.2.4](https://openneuro.org/datasets/ds004080/versions/1.2.4).
