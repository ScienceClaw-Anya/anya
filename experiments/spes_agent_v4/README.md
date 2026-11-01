# SPES research v4: complementary observations before agent debate

This is an **adaptive exploratory research programme**, not an untouched-test
confirmation or a continuation of the stopped v3 agent run. All repository
reports are in English. The previous failed experiments remain available.

## Decision sequence

1. Run a small fixed numerical screen on v3's saved, outer-isolated OOF
   predictions. Test the previously suggested removal of the global veto,
   patient-balanced contact fusion, and a patient-pairwise ranking model.
   Preserve every result; do not launch an LLM comparison for negligible gains.
2. Retrain three complementary, published multi-trial SPES architectures on the
   authors' newly released processed data: trial-averaged baseline, mean trial
   embedding (MTE), and hierarchical attention (HAT). Fix five patient-held-out
   folds and all settings before evaluating the outer predictions.
3. Only proceed to collaboration tests when agents can make meaningfully
   different executable decisions or acquire genuinely new observations.
   The best trained individual model and a numerical acquisition/fusion policy
   must be controls. Better neural models alone do not prove multi-agent value.

## Literature and cohort

The motivating [2026 preprint](https://doi.org/10.64898/2026.07.27.26358819)
and [author code](https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT)
are explicitly preprint/code evidence. The released arrays used here contain
**35 patients, 2,117 contacts and 296 SOZ contacts**, which differs from both
the project's older public-code cohort and the original paper's cohort counts.
Never compare these scores as if the contacts and split were identical.

A later identity audit found **34 shared patients**: patient 39 replaces patient
47 in this processed version. Labels agree on 2,036 unambiguous reciprocal
coordinate matches. The ignored patient-47 array remains excluded; its upstream
exclusion reason is unresolved. Frozen experiment manifests are preserved.
See `analysis/audit_spes_cohort_versions.py` and
`examples/spes_cohort_versions_summary.json` for the scored-cohort comparison.

The upstream reference checkpoints were **not loaded**: tensor-only safe
loading rejected a serialized loss object, and new training provides clearer
patient provenance. The new runner freezes source hashes and split manifests,
normalizes using fitting patients only, and clones the best validation weights.
It uses published fixed presets, a 25-epoch limit and patience 5. Batch padding,
streamed float64 normalization statistics, deterministic seed assignment and
checkpoint handling differ from the upstream runner; this is a documented
adaptation, not an exact paper reproduction.

## Local commands

Run from the repository using the scientific environment:

```powershell
.\.venv-repro\Scripts\python.exe -m experiments.spes_agent_v4.screen --project . --output outputs/spes_agent_v4/screen
.\.venv-repro\Scripts\python.exe -m experiments.spes_agent_v4.neural --project . --output outputs/spes_agent_v4/multitrial --workers 2
.\.venv-repro\Scripts\python.exe -m experiments.spes_agent_v4.evaluate --output outputs/spes_agent_v4/multitrial --freeze-only
# After all 15 fits complete:
.\.venv-repro\Scripts\python.exe -m experiments.spes_agent_v4.evaluate --output outputs/spes_agent_v4/multitrial
```

The first command requires the completed v3 OOF predictions. The second
requires the pinned author repository at `vendor/norris2026_hat`, including its
Git LFS arrays. Vendor files, model weights, data and private environment files
remain ignored. The programme uses the local RTX 4090; it creates no cloud
resources and makes no LLM calls during screening or neural training.

Use `python scripts/setup_spes_multitrial.py --download-data` to obtain the
pinned upstream code and data. Install Git LFS first. Freeze the evaluation
protocol before opening outer results; the recorded project run did so.

## Real-feedback acquisition experiment

The new task reveals real responses from one stimulation position per query.
All three models recompute scores from **observed columns only**. Agents choose
measurements; the final scorer is the same fixed HAT in every arm. Geometry and
the released structural availability map are assumed known beforehand. This is
not a physical stimulation experiment, a treatment protocol or the original
full-response paper task.

```powershell
$training = 'outputs/spes_agent_v4/multitrial'
$active = 'outputs/spes_agent_v4/active'
python -m experiments.spes_agent_v4.active --project . --training $training --output $active --stage freeze
python -m experiments.spes_agent_v4.active --project . --training $training --output $active --stage numeric-pilot --workers 4
# Runs only if the numerical feasibility gate passes; maximum 120 attempts:
python -m experiments.spes_agent_v4.active --project . --training $training --output $active --stage agent-pilot --workers 6
python -m experiments.spes_agent_v4.active --project . --training $training --output $active --stage numeric-all --workers 2
python -m experiments.spes_agent_v4.audit --project .
python -m experiments.spes_agent_v4.report --project .
```

The four roles cover geometry, diagnostic hypotheses, model disagreement and
evidence audit. They propose independently in the first round, see all four
real measurements, then exchange rationales before the second round. The
no-exchange control shares physiological feedback but hides other roles'
rationales. A strong single-agent control receives sequential feedback after
each of its eight decisions. All LLM arms get eight attempts and eight actual
measurement slots; numerical controls get eight measurements and zero LLM
calls. Invalid responses remain spent and use a declared geometric fallback.
The first five pilot patients remain separately reported from later patients.

## Interpretation rules

This cohort has been repeatedly inspected; 34 patients overlap the older
reconstruction. New runs
remain exploratory even though training and scoring are patient separated.
Track the complete trial history; report patient-level uncertainty and compare
with the strongest same-information numerical control. An agent's choice must
not depend on hidden labels, future stimulus responses, or full-response
predictions when the task concerns partial acquisition. SOZ agreement is not
evidence of better seizure control or a clinically validated stimulation target.
