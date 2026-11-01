# Relative ranking of physical diagnostic queries

**Completed, negative result:** the specialist panel scored 0.742124, versus
score-focus 0.761978 and the all-feature pairwise control 0.757274. The fixed
gate fails. All 105 physical trajectories and 420 adaptive choices passed
independent replay. No new weights are adopted. See the
[English report](../../docs/spes-agent-v17-report.md).

This new numerical feasibility study learns which of two candidate positions
produces a better next ranking under the same source-held-out patient state.
The target difference cancels the common before-query AUROC, which the runtime
controller cannot observe. It uses the existing audited v14 physical banks,
not the privileged v16 test-patient oracle choices.

For each outer fold, 28 development patients supply672candidate rows across112
states. Six candidates yield15unordered comparisons per state, but pair weights
sum to the original672row weight. Pair differences are not new participants.
Fixed Ridge alpha100 models use standardized observed-only descriptors and
state/candidate interactions. Geometry, neural-state and waveform specialists
are averaged by percentile rank. Same-data all-feature pairwise and pointwise
models serve as controls. No hyperparameter search or test-label selection is
performed. Every source target patient was excluded from its neural fitting and
early stopping, and every outer test patient is excluded from the full pipeline.

All controllers receive the same four initial and eight total verified physical
groups, compute the same full feature table, and retain the original outer HAT
as final scorer. A cached feature recipe was previously independently checked
against the old runtime to1e-12 and is newly pinned here; no frozen study is edited.

```powershell
python -m unittest experiments.spes_agent_v17.test_ranking
python -m experiments.spes_agent_v17.experiment --project . --output outputs/spes_agent_v17 --stage freeze
python -m experiments.spes_agent_v17.experiment --project . --output outputs/spes_agent_v17 --stage fit
python -m experiments.spes_agent_v17.experiment --project . --output outputs/spes_agent_v17 --stage evaluate --workers 4
```

The fixed primary is the expert panel versus frozen v11score-focus. Both this
comparison and expert-panel versus all-feature pairwise control must exceed0.01
with positive paired95%lower bounds before further tool development. A better
numerical panel would still not prove Gemini communication value. Current
Gemini authorization is fully consumed; this experiment makes zero API calls.

The local workstation is sufficient. All25small fits are frozen before new
outer outcomes. Three views have a structural inductive bias and fewer total
coefficients than the all-feature model, so this is not an exact parameter-match
proof. Same-cohort adaptation, random-prefix/runtime shift, and source/final
neural training-size differences remain. No clinical efficacy is established.
Raw data, labels, generated banks and weights stay outside Git.
