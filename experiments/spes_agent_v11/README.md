# First verified physical-pair selection controls

The completed 35-patient comparison uses **eight real anatomical pairs**, with a
common four-query prefix and four adaptive queries. It replaces the invalid
shared-column action in historical v4-v10 work. See the
[English physical-action correction](../../docs/spes-physical-action-audit.md)
and [English v11 results](../../docs/spes-agent-v11-report.md).

The fixed primary score_focus policy scored 0.761978 mean patient SOZ AUROC,
versus random_0 0.740463: gain 0.021514, paired 95% interval
[-0.011709, 0.052024]. The interval includes zero. All five policies remain
reported; no stable improvement, multi-agent advantage or treatment benefit is
claimed. Patient-level query sensitivity is established, not agent capability.

```powershell
python -m experiments.spes_agent_v11.numeric --project . --output outputs/spes_agent_v11 --freeze-only
python -m experiments.spes_agent_v11.numeric --project . --output outputs/spes_agent_v11 --workers 4
python -m experiments.spes_agent_v11.audit --project . --output outputs/spes_agent_v11
python -m experiments.spes_agent_v11.report --project . --output outputs/spes_agent_v11
```

Requires the verified 35 physical maps and existing v4 neural models. Frozen
models, map receipts, source hashes and actual anatomical-pair identities are
checked. Ambiguous responses are excluded, future waveforms and clinical labels
are gated, and all policies share the unchanged HAT. These numerical controls
make zero model API calls and create no cloud resources.

Rebuild development states using physical_input before training new specialist
tools. Old column-utility or column-prefix classifiers are not adopted. Any
new LLM comparison needs a separate matched-budget protocol and remains
exploratory on this repeatedly inspected cohort.
