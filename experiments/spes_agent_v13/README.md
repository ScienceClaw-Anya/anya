# Training-only calibration and patient-adaptive physical acquisition

This fresh numerical study tests whether class-weighted HAT scores are a poor
absolute confidence signal, then whether a training-patient decision boundary
improves physical diagnostic measurement selection. It is not an agent comparison.

The protocol freezes three one-dimensional sigmoid calibration recipes, an
analytic prior hypothesis and a development-prevalence constant. Source HATs
exclude each calibration patient. No outer patient enters parameter fitting.
The acquisition primary is the learned-boundary policy against the frozen v11
score-focus control, with the same four initial and eight total physical groups.
The final HAT ranking remains unchanged. All variants and failures are retained.

The analytic correction subtracts twice the log training class ratio from raw
logits. This follows an ideal weighted-risk model of the combined sampling/loss
weights; finite training and sparse-input shift can violate its assumptions.
Patient-adaptive thresholding adapts preprint supplement S3 into a new logit-MAD
acquisition heuristic. It is not a published active-stimulation result.

```powershell
python -m unittest experiments.spes_agent_v13.test_tools
python -m experiments.spes_agent_v13.experiment --project . --output outputs/spes_agent_v13 --stage freeze
python -m experiments.spes_agent_v13.experiment --project . --output outputs/spes_agent_v13 --stage fit
python -m experiments.spes_agent_v13.experiment --project . --output outputs/spes_agent_v13 --stage calibration
python -m experiments.spes_agent_v13.experiment --project . --output outputs/spes_agent_v13 --stage acquire --workers 4
```

Use the local workstation. No cloud resource, new neural fit or Gemini request
is required. The eight-group budget does not equate pulse count, current, charge,
time or clinical safety. Same-cohort exploration cannot provide external
validation or seizure-control evidence. Raw data, labels and model assets are
not committed.

## Completed result

Both fixed practical-gain gates failed. Centered calibration Brier gain over the
constant is 0.003551 (interval [0.000358, 0.006469]), below the fixed 0.005
requirement. The primary query policy scores 0.744035 versus score-focus 0.761978;
gain -0.017943 (interval [-0.043149, 0.005348]). No paid agent trial is triggered.
See the [English report](../../docs/spes-agent-v13-report.md). Independent audit
refitted all 75 calibrators, independently checked 700 source boundaries and
replayed every acquisition choice and outcome. Neural scores were not all
re-inferred. All variants and secondary outcomes remain in the summary.
