# Hypothetical-response specialists

This running numerical experiment tests a substantive tool change after v8's
fixed primary failed its spending gate. No new LLM pilot is launched until a
useful numerical tool is supported. No multi-agent benefit is established yet.

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
