> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).

# Hypothetical-response specialists

This numerical experiment stopped after 106/315 trajectories when the
physical action-grain assumption failed. The other 209 are canceled or unfinished.
All partial records remain; no partial-cohort winner is reported. The stop lock
stays in place. No new LLM request was made and these tools are not adopted for
physical selection. The methods below document the retained column-ablation
prototype, rather than instructions to resume it.

Two distinct predictors construct hypothetical responses from measured waveforms:
geometry-profile averaging, and averaging with clipped radial amplitude
correction. Candidate distances and released availability are assumed known;
candidate waveforms and clinical labels never enter either predictor. Predictions
are explicitly hypothetical, not observations. Trial-index alignment across
stimulation sites is a computational assumption, not physiological simultaneity.

The same twenty source-held-out inner HATs from v8 score the imagined additions.
Candidate-specific score changes, ranking changes and response-predictor
disagreement augment the original observed-only features. Supervised targets
reuse the exact phase-zero v8 candidate rows. Outer patients never enter tool
fitting. No new neural training or model API call is needed.

Ridge and fixed patient-exclusion gradient-boosting bags learn acquisition gain.
Five numerical controls and four learned policies retain the same initial four
measurements, eight-measurement total, full candidate pool and original final HAT.
The absolute bag mean is the frozen primary. All outcomes remain exploratory
after repeated cohort inspection; no secondary test winner replaces it.

## Local commands

Run each stage only after the previous command succeeds. Requires completed v8
source fits/phase-zero banks and v4 neural models. Use the scientific environment.

```powershell
python -m unittest experiments.spes_agent_v9.test_v9 -v
python -m experiments.spes_agent_v9.banks --project . --output outputs/spes_agent_v9 --freeze-only
python -m experiments.spes_agent_v9.banks --project . --output outputs/spes_agent_v9 --workers 2
python -m experiments.spes_agent_v9.learner --project . --output outputs/spes_agent_v9
python -m experiments.spes_agent_v9.numeric --project . --output outputs/spes_agent_v9 --freeze-only
python -m experiments.spes_agent_v9.numeric --project . --output outputs/spes_agent_v9 --workers 2
```

The fixed gate requires primary mean AUROC above the best numerical control by
more than 0.005, positive one-sided 95% patient-bootstrap lower bound versus
random_0, and at least ten changed query sets. This gate controls API spending,
not clinical validity. Hypothetical model forwards are counted separately from
actual measurements and LLM calls. Local RTX 4090/CPU execution creates no new
cloud resources. Weights, real waveforms and generated banks remain outside Git.
