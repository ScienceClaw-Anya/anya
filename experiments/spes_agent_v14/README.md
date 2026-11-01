# Physical counterfactual query-value experts

This new numerical study trains separate geometry, neural-state and waveform
experts to predict the gain from an actual additional anatomical-pair response
group. It uses verified physical mappings and source-held-out patients. All
candidate features are fixed before added responses and SOZ labels are used to
construct supervised training targets. Alternative response targets are newly
computed; old column-ablation banks and weights are not adopted.

Three expert views share the same source patients and targets. A conservative
mean-minus-disagreement rule is compared with all-feature boosted and ridge
controls. All runtime controllers receive four common initial groups and four
adaptive choices. The unchanged outer HAT produces every final ranking. The
fixed primary is consensus versus v11 score-focus; both this comparison and
consensus versus the monolithic learner must pass before an agent trial for
these tools. Expert disagreement is not a confidence interval.

```powershell
python -m unittest experiments.spes_agent_v14.test_tools
python -m experiments.spes_agent_v14.experiment --project . --output outputs/spes_agent_v14 --stage freeze
python -m experiments.spes_agent_v14.experiment --project . --output outputs/spes_agent_v14 --stage banks --workers 4
python -m experiments.spes_agent_v14.experiment --project . --output outputs/spes_agent_v14 --stage fit
python -m experiments.spes_agent_v14.experiment --project . --output outputs/spes_agent_v14 --stage evaluate --workers 4
```

The local GPU is sufficient. This stage uses zero API calls and no new neural
fits. Twenty own source HATs are reused; twenty-five small value regressors are
newly fitted. Random-prefix training may differ from learned-policy states;
source and final HAT training sizes differ. All repeatedly inspected-cohort
results remain exploratory. No seizure-control, pulse-budget or safety claim
follows from eight recorded response groups. Raw data, labels and model assets
stay outside Git.

## Completed result

The primary fails: expert consensus AUROC 0.727182 versus score-focus 0.761978;
gain -0.034795, interval [-0.077491, 0.011200]. Consensus also does not establish
superiority over the monolithic regressor (gain 0.021060, interval [-0.020222,
0.069559]). Expert top choices differ in 132/140 steps. All variants remain in
the [English report](../../docs/spes-agent-v14-report.md).

The independent audit reconstructs all 3,360 feature/target rows, 560 source
states, 420 adaptive choices and 105 trajectories. Its cached feature formula
is independently implemented and checked against the frozen runtime formula;
the running controller was not changed to accelerate auditing. Neural scores
are checked from stored outputs rather than all re-inferred.

```powershell
python -m experiments.spes_agent_v14.audit --project . --output outputs/spes_agent_v14 --stage full
python -m experiments.spes_agent_v14.report --project . --output outputs/spes_agent_v14
```
