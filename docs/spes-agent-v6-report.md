> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).

# Feedback-matched SPES collaboration

This exploratory experiment gives every policy the same four initial measurements, then four adaptive measurements. Each adaptive query receives a proposal and review call before execution. Actual measured feedback arrives at the same points for every LLM arm. The original v4 HAT remains the final scorer; sparse v5 models are not adopted.

The single agent proposes and self-reviews; the team alternates hypothesis/uncertainty proposers and audit/coverage reviewers. A team ablation hides peer rationales while retaining query choices and physiological feedback. Every LLM arm has eight attempts and eight total measurements. Numerical policies use no LLM calls. All share the same numerical tools. No agent-generated probabilities are executed.

Five fixed pilot patients exclude the five v4 LLM pilot patients. This is still a previously inspected cohort; it is not independent external validation. The new numerical random_0 control reproduces the old exact scores, verifying the same responses and scorer.

| Policy | Completed pilot patients | Mean SOZ AUROC | Accepted model outputs |
|---|---:|---:|---:|
| random_0 | 5 | 0.755916 | 0 |
| random_1 | 5 | 0.775485 | 0 |
| coverage | 5 | 0.780685 | 0 |
| score_focus | 5 | 0.771136 | 0 |
| uncertainty | 5 | 0.702194 | 0 |
| single | 5 | 0.761578 | 35 |
| multi | 5 | 0.740060 | 34 |
| multi_reason_hidden | 5 | 0.737837 | 34 |

Partial, unequal completed counts are progress only; do not compare unpaired partial means.

## Frozen pilot decision

Multi minus single: **-0.021519**, exploratory paired 95% patient-bootstrap interval **[-0.09270212394095649, 0.0569565217391305]**. Frozen-rule expansion: **False**. Accepted fraction: **85.8%**.

The expansion rule requires >0.005 gain over single, no more than 0.02 loss versus the best fixed numerical pilot policy, and at least 90% accepted decisions. Passing is a screening signal, not a significance or clinical-benefit claim.

| Multi minus same-patient control | Gain | Exploratory paired 95% interval |
|---|---:|---:|
| single | -0.021519 | [-0.09270212394095649, 0.0569565217391305] |
| multi_reason_hidden | +0.002222 | [0.0, 0.00666666666666671] |
| random_0 | -0.015856 | [-0.09718417047184172, 0.06820512820512822] |
| random_1 | -0.035425 | [-0.187533461784504, 0.1035507246376812] |
| coverage | -0.040626 | [-0.10365296803652968, -0.001630434782608714] |
| score_focus | -0.031076 | [-0.09466216663357811, 0.02651802303976223] |
| uncertainty | +0.037866 | [-0.01734299516908213, 0.12649572649572655] |

## Interface confound found after the pilot

The executed study contained 17 fallbacks. Of these, **15** used real nested evidence references (for example `geometry.candidates`) that the top-level-only validator rejected. This was an over-strict interface design. These scores therefore do not isolate collaboration ability.

All historical outcomes remain the actually executed fallback-policy outcomes. We do not retroactively invent accuracy for the unexecuted choices: accepting a different proposal could change later observations and later model answers. The follow-up uses actual API-enforced JSON Schema, permits only available query IDs and real evidence-root IDs, and adds a full smoke trajectory before another pilot. This interface problem does not explain all older negative experiments.


## Evidence boundaries

The [MDAgents architecture](https://doi.org/10.52202/079017-2522) motivates relevant specialists and structured review; its QA/diagnosis findings are not evidence for this epilepsy task. Full Methods sections 3.1-3.5 in the official proceedings PDF were inspected, alongside the canonical Chrome record. The literature search was focused and non-exhaustive.

Public observations contain only known geometry/availability, already measured responses, and model scores recomputed from those observed columns. Hidden labels, future responses and full-response test predictions are excluded. Stimulus IDs are array columns; their mapping to clinical electrode-pair names has not been independently reconstructed. Known released preprocessing availability may overstate prospective knowledge.

All waveform computation runs locally on the RTX 4090. Gemini inference uses the official Google API, medium reasoning, no retries. Every malformed or uncaptured attempt stays spent. Full reasons are preserved; the 6,000-character cap applies equally across arms. Short local filesystem retries do not resend API calls.

SOZ concordance is a diagnostic ranking endpoint, not therapeutic benefit. No physical stimulation is delivered. Any pilot improvement needs a larger matched comparison, repeat-generation sensitivity and a fresh cohort before a strong generalization claim.
