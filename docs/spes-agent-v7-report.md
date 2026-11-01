> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).

# Schema-valid collaboration reproduces geometric coverage

The repaired interface accepted **120/120 formal model decisions**. Single-agent, multi-agent and rationale-hidden policies all scored **0.790440 mean patient SOZ AUROC**, equal to the geometric-coverage policy on the same five patients. The fixed pilot rule does **not** permit expansion. No multi-agent superiority is established.

## Matched results

| Policy | Mean patient AUROC |
|---|---:|
| random_0 | 0.761492 |
| random_1 | 0.771364 |
| coverage | 0.790440 |
| score_focus | 0.799959 |
| uncertainty | 0.775162 |
| single | 0.790440 |
| multi | 0.790440 |
| multi_reason_hidden | 0.790440 |

| Multi minus control | Mean gain | Exploratory paired 95% interval |
|---|---:|---:|
| single | +0.000000 | [0.0, 0.0] |
| multi_reason_hidden | +0.000000 | [0.0, 0.0] |
| random_0 | +0.028949 | [-0.030774991292232646, 0.13179691653375863] |
| random_1 | +0.019076 | [-0.15583333333333332, 0.2457894736842106] |
| coverage | +0.000000 | [0.0, 0.0] |
| score_focus | -0.009518 | [-0.03620689655172411, 0.010883838383838417] |
| uncertainty | +0.015279 | [-0.013448275862068915, 0.04656432748538011] |

Five patients provide a small exploratory screen. Repeatedly inspected cohort identities and adaptive project iterations prevent an untouched-test interpretation. The zero interval versus single describes these identical recorded outcomes, not a population guarantee of equivalence.

## What the agents actually did

Reviewers selected the maximum-coverage query in **58/60** final choices. They replaced **13** proposals; every replacement moved to maximum geometric coverage. **13/15** LLM trajectories selected the same final query set and **13/15** produced the same ordered ranking as the coverage control. All 15 had its same AUROC.

This is a behavioral observation. It supports a narrower explanation than a general claim about LLM medical knowledge: the workflow repeatedly returned to a common geometric proxy instead of producing an independently useful acquisition rule. The reviewer had final authority, and some distinct proposals were discarded. Neither geometric coverage nor model disagreement had been trained to predict actual localization gain. This pilot does not causally separate role authority, prompt emphasis, model priors and the available tool set.

The next fixed experiment learns measurement value from development-patient replay, using inner models that excluded each target patient from fitting and early stopping. It first tests numerical utility policies. More LLM calls are not justified merely by changing role names.

## Interface repair and fairness

The previous v6 validator rejected real nested citations. Historical v6 fallback outcomes remain unchanged. The new interface attaches [JSON Schema to the actual Gemini API request](https://ai.google.dev/gemini-api/docs/openai#structured-output): one currently available query ID, allowed evidence-root IDs, complete rationale and limitation, and no probability fields. Two synthetic calls checked the interface. A complete eight-call smoke trajectory passed before the rest of the pilot; those eight calls count inside the 120 formal attempts.

Each arm starts with the same four real measurements. Four adaptive cycles each contain a proposal call, review call, execution and real recorded-response feedback. Single and multi share exactly the same feedback schedule, measurements, numerical tools and final original HAT. The no-peer-rationale control retains measured feedback and query choices. Gemini 4 Argon uses medium reasoning and zero retries. No sparse v5 weights are adopted.

The audit verified **960 native artifacts**, raw model/executed-decision agreement, actual API schemas, request budgets, peer-rationale isolation, independent AUROC and exact reproduction of the old random_0 control. The two synthetic calls are reported separately from the formal study.

## Scope

This is a retrospective subset-acquisition replay. Stimulus IDs are array columns, with no independently reconstructed mapping to named clinical electrode pairs. Geometry and released preprocessing availability are assumed known. Physiological state changes, clinical safety and treatment effects are not simulated. The endpoint is SOZ ranking, not seizure control.

The new processed cohort has 35 patients / 2,117 contacts / 296 SOZ contacts. A source audit found 34 patients in common with the older 35-patient reconstruction: patient 39 replaces patient 47. Across 2,036 unambiguous coordinate matches in shared patients, scored labels agreed. Different patient/contact populations prevent direct cross-version architecture claims. All v7 policy comparisons use the same current cohort/pilot.

Waveform computation ran locally on the RTX 4090; language inference used the official Google API. No new cloud resources were created.
