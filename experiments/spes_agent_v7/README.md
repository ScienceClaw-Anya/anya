# Schema-constrained proposal/review pilot

The full pilot completed **120/120 accepted decisions**. Single, multi and
rationale-hidden policies all scored **0.790440**, equal to geometric coverage
on the same five patients. The frozen expansion rule failed, so the remaining
patients are not sent to Gemini for this policy. See the
[English report](../../docs/spes-agent-v7-report.md).

Every arm observes the same four initial stimulation columns and then four
adaptive columns. Each adaptive step has a proposal and review call before
real recorded feedback. The final scorer is the unchanged original v4 HAT.
Single and multi have the same eight-call budget and feedback cadence.

The provider attaches a dynamic **JSON Schema to the actual native API call**,
constraining `next_query` to available IDs and `evidence_ids` to real roots.
No probability field or arbitrary score vector can be returned. The model's
full reason/limitation is preserved. Zero API retries; spent failures remain
spent. A short local filesystem retry never resends a model request.

```powershell
python -m experiments.spes_agent_v7.provider --project . --output outputs/spes_agent_v7
python -m experiments.spes_agent_v7.experiment --project . --output outputs/spes_agent_v7 --stage freeze
python -m experiments.spes_agent_v7.experiment --project . --output outputs/spes_agent_v7 --stage numeric-pilot --workers 4
# Eight-call smoke is part of the pilot's 120-call budget:
python -m experiments.spes_agent_v7.experiment --project . --output outputs/spes_agent_v7 --stage smoke
python -m experiments.spes_agent_v7.audit --project .
python -m experiments.spes_agent_v7.experiment --project . --output outputs/spes_agent_v7 --stage agent-pilot --workers 6
python -m experiments.spes_agent_v7.audit --project .
python -m experiments.spes_agent_v7.behavior --project .
python -m experiments.spes_agent_v7.report --project .
```

The two synthetic interface calls are separate. Saved attempts are never
replayed. Numerical controls reproduce the original random_0 scores exactly.
The new pilot excludes the two earlier LLM pilot sets, but all 35 underlying
patients have already been inspected numerically. This remains exploratory.

Reviewers followed maximum geometric coverage in 58/60 choices and redirected
13 proposals to that heuristic. This motivates learning query value from
patient-isolated data, rather than treating an unvalidated geometric score as
localization benefit. The result is not a clinical effectiveness test.
