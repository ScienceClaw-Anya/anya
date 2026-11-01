> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).

# Common-prefix, feedback-matched collaboration

The earlier active pilot gave a single agent more opportunities to react to
measurements than a two-batch team. This new experiment matches that timing.
It retains the stronger original v4 HAT; the negative v5 sparse-training models
are not adopted.

Every policy first receives the **same four real measurements**. It then makes
four adaptive measurements. Each adaptive choice uses two LLM calls: a proposal
and a review, followed by execution and real recorded-response feedback.
The single agent proposes and reviews its own decisions. The team alternates
hypothesis/uncertainty proposers and audit/coverage reviewers. The rationale-
hidden team sees other roles' choices and measured feedback but not their text.
All arms have the same public numerical tools and final scorer.

Total budget per LLM arm: eight measurements, eight native public-tool reads,
eight Gemini 4 Argon attempts with medium reasoning and no retries. Complete
rationales are preserved up to 6,000 characters. Invalid or uncaptured attempts
remain spent and use a declared geometric fallback. Local atomic file writes
have a short bounded retry; this never retries an API request.

Five fixed pilot patients (one per outer fold) exclude the five patients in the
v4 LLM pilot. A numerical feasibility check precedes any paid calls. Expansion
requires a frozen rule: multi exceeds single by more than 0.005, is no more than
0.02 below the best fixed numerical pilot policy, and at least 90% of model
decisions are accepted. Five patients cannot establish efficacy. All patient
identities have still been examined in prior numerical research.

```powershell
python -m experiments.spes_agent_v6.experiment --project . --output outputs/spes_agent_v6 --stage freeze
python -m experiments.spes_agent_v6.experiment --project . --output outputs/spes_agent_v6 --stage numeric-pilot --workers 4
python -m experiments.spes_agent_v6.audit --project .
python -m experiments.spes_agent_v6.experiment --project . --output outputs/spes_agent_v6 --stage agent-pilot --workers 6
python -m experiments.spes_agent_v6.audit --project .
python -m experiments.spes_agent_v6.report --project .
python -m unittest experiments.spes_agent_v6.test_v6 -v
```

Stimulus IDs refer to released array columns. Mapping them back to named clinical
electrode pairs has not been independently reconstructed. Distances and the
released structural availability map are assumed known in advance. No actual
stimulation is delivered, and no treatment effect or clinical safety is tested.

The [MDAgents proceedings paper](https://doi.org/10.52202/079017-2522) motivates
specialist review and structured synthesis. Its medical QA/diagnosis results do
not establish a transferable effect size for this SPES acquisition task.
