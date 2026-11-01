# Independent physical proposals and cross-round communication

This fresh ScienceClaw/Gemini pilot uses verified actual anatomical response
groups. It does not resume any stopped column-ablation experiment or adopt
failed v12-v15 predictors. Five pilot patients were fixed in v11 before its
outcomes. All cohort identities remain previously inspected and exploratory.

Each of four adaptive rounds begins with two independently frozen proposals.
Single uses two generalist passes. Multi uses persistent anatomy/network and
response/error roles. Both have all the same whitelisted anatomy, recorded
conditions, observed-only waveform quality, numerical ranks and feedback.
Current-round peer answers are hidden. Previous-round explanations are shared
only after both current decisions finish. A third arm removes peer rationale
text while retaining measured feedback and peer choices.

The same deterministic 3/2/1 rank vote merges each pair of three-position
rankings; ties use existing score-focus gain and then physical ID. No reviewer
can unilaterally replace a proposal. The first four physical groups are common,
and the final scorer is always the unchanged outer HAT after eight groups.
Agents never write contact probabilities or see test accuracy/labels.

The pilot cap is 120 Gemini attempts: three arms, five patients, eight attempts.
The first eight-call single trajectory is an included smoke check. Failure
stops expansion without automatic replay or replacement. Maximum completion
tokens and reasoning settings match between language-agent arms; actual token
usage is reported separately. Numerical controls use the same group budget
with zero language-model calls. Eight groups do not equate pulses or charge.

```powershell
python -m unittest experiments.spes_agent_v16.test_v16
python -m experiments.spes_agent_v16.experiment --project . --output outputs/spes_agent_v16 --stage freeze
python -m experiments.spes_agent_v16.experiment --project . --output outputs/spes_agent_v16 --stage smoke
python -m experiments.spes_agent_v16.experiment --project . --output outputs/spes_agent_v16 --stage pilot --workers 6
```

Native ScienceClaw SkillRegistry, SkillExecutor, LLMClient and ArtifactStore
record common tools, independent proposals, merges and measured feedback.
Actual API JSON Schema constrains choices to available pair IDs. Request
reservations count failed or incomplete attempts; retries are disabled. No
Infinite publication, new cloud resource or external message is performed.

Report every arm and every fixed control. Expansion to the remaining 30
patients requires a completed pilot audit and the frozen multi/single,
multi/score-focus, validity and rationale-diversity gates. Pilot performance is
not confirmation. Nominal intervals are not corrected for the full adaptive
research search. No multi-agent or clinical benefit is claimed in advance.

## Local review and authorization

The first paid run was rejected by automatic approval because explicit approval
for this patient-derived payload was missing. A local-only review subsequently
verified eight true response groups, eight native observation-tool executions
and twenty artifact hashes/parent links against the frozen score-focus control.
It generated the exact first request at `outputs/spes_agent_v16/data_review.html`.
No model calls were made by that local check.

The user then explicitly authorized **five patients and at most120 attempts**
to the official Google endpoint. The eight-call smoke is included and passed.
The remaining-cohort expansion is not covered by this specific authorization;
its performance gate alone does not grant permission to send more patient data.
Generated prompts, anatomical catalogs, approval records and traces remain local
and ignored by Git. Only code and aggregate results are committed.

```powershell
python -m experiments.spes_agent_v16.review --project . --output outputs/spes_agent_v16
python -m experiments.spes_agent_v16.audit --project . --output outputs/spes_agent_v16
python -m experiments.spes_agent_v16.behavior --output outputs/spes_agent_v16
python -m experiments.spes_agent_v16.report --project . --output outputs/spes_agent_v16
```

Audit and behavior stages require a completed pilot. Every real physical
trajectory is re-inferred in the audit. The evidence skill is
[`neurotarget-spes-v16`](../../skills/neurotarget-spes-v16/SKILL.md).
