> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).

# Learned measurement-value tools before further agent spending

This experiment asks whether supervised development-patient replay can produce
a useful acquisition tool. It does not itself test LLM collaboration. English
results are generated in `docs/spes-agent-v8-report.md` only after the independent
audit. All earlier failed studies remain frozen, including the stopped v3 run.

The previous valid v7 pilot returned to geometric coverage in 58/60 final
reviewer choices. Here the tool learns actual immediate SOZ-ranking gain instead
of renaming roles or inviting agents to invent probabilities.

## Isolation and fixed schedule

- Twenty inner HAT fits: 17 fitting, four early-stop and seven target patients
  per job, with the seven outer patients excluded from all source loading.
- Two fixed replay phases: random prefixes, then phase-zero policy prefixes.
- Twenty observed-only features and up to six candidate targets at each of
  four states. Candidate responses and clinical labels enter targets only.
- Each outer learner fits 28 development patients, equally weighted. Ridge and
  five fixed gradient-boosting bags for both absolute and relative gain are
  numerical specialists; their disagreement is not calibrated uncertainty.
- Five learned policies and five fixed controls get a common four-measurement
  prefix plus four adaptive measurements and the same original v4 HAT.
- The fixed primary and future-LLM spending rule are frozen before outer-policy
  evaluation. All secondary policies stay visible; no test-winner substitution.

The 6,720 replay rows are correlated patient-fold examples, not independent
patients. Patient isolation does not undo repeated exploration of this cohort.
No fine-tuning of Gemini or clinical-treatment claim is made.

## Run locally

Requires completed v4 training/acquisition controls and the pinned multi-trial
upstream dataset. Use the scientific Python environment from the repository.
Run each command only after the preceding stage completes successfully.

```powershell
python -m experiments.spes_agent_v8.crossfit --project . --output outputs/spes_agent_v8 --workers 2
python -m experiments.spes_agent_v8.targets --project . --output outputs/spes_agent_v8 --freeze-only
python -m experiments.spes_agent_v8.targets --project . --output outputs/spes_agent_v8 --phase 0 --workers 2
python -m experiments.spes_agent_v8.learner --project . --output outputs/spes_agent_v8 --phase 0
python -m experiments.spes_agent_v8.targets --project . --output outputs/spes_agent_v8 --phase 1 --workers 2
python -m experiments.spes_agent_v8.learner --project . --output outputs/spes_agent_v8 --phase 1
python -m experiments.spes_agent_v8.numeric --project . --output outputs/spes_agent_v8 --freeze-only
python -m experiments.spes_agent_v8.numeric --project . --output outputs/spes_agent_v8 --workers 4
python -m experiments.spes_agent_v8.audit --project . --output outputs/spes_agent_v8
python -m experiments.spes_agent_v8.report --project . --output outputs/spes_agent_v8
python -m unittest experiments.spes_agent_v8.test_v8 -v
```

Neural weights, real arrays and generated patient-level banks stay outside Git.
Only code, research provenance and compact results are committed. Computation
uses the local RTX 4090 and CPU. The numerical stages make no model API calls
and create no cloud resources.
