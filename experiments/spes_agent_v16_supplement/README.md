# Posthoc physical-budget and candidate-pool diagnostics

This supplement addresses two presentation and bottleneck questions after the
complete negative v16 pilot. It uses local computation and zero new API calls.
The diagnostic protocol is frozen before new counterfactual scores are opened.
It does not create another deployable agent or rewrite the original results.

First, compare the common four-group HAT result with actual eight-group results.
Report the 35-patient numerical cohort and five-patient agent pilot separately.
Intervals resample paired patient differences, not electrodes or agent arms.

Second, keep each actual seven-group final prefix fixed. Evaluate the union of
the two agents' three-position proposals, the executed position and score-focus
leader. Compare the privileged best proposed candidate with five seeded random
pools of the same size. The random reference averages each pool's maximum;
it does not maximize over all five pools. Slates are fixed before new scoring,
and labels are read only by the posthoc judge, never provided to an agent.

This oracle is a conditional **last-step diagnostic**, using unavailable true
labels. It is not executable agent performance or an achievable eight-step
trajectory. Multiple random pools and arms do not increase the number of
independent patients. All outcomes remain exploratory on an inspected cohort.

```powershell
python -m unittest experiments.spes_agent_v16_supplement.test_supplement
python -m experiments.spes_agent_v16_supplement.analysis --project . --output outputs/spes_agent_v16_supplement --stage freeze
python -m experiments.spes_agent_v16_supplement.analysis --project . --output outputs/spes_agent_v16_supplement --stage budget
python -m experiments.spes_agent_v16_supplement.analysis --project . --output outputs/spes_agent_v16_supplement --stage oracle --workers 2
```

Generated raw predictions, clinical evaluation labels and patient-derived traces
remain outside Git. Only code, aggregate results and English figures/report are
published. Response-group budgets do not equate pulse count, charge or safety.

## Verification and presentation output

```powershell
python -m experiments.spes_agent_v16_supplement.audit --project . --output outputs/spes_agent_v16_supplement
python -m experiments.spes_agent_v16_supplement.report --project . --output outputs/spes_agent_v16_supplement
```

The completed audit covers190patient/policy budget rows,15conditional pool rows,
197new local scorer predictions and all random slates/bootstrap intervals.
No API call or model fit is added. The English report is
[`docs/spes-agent-v16-supplement.md`](../../docs/spes-agent-v16-supplement.md).
Two figures are supplied in both PNG and editable-text SVG under`docs/images/`.
They were visually checked for labels, axes, confidence intervals and layout.
All original agent results remain unchanged. Oracle choices are not training
targets for test patients and are never used as executable model predictions.
