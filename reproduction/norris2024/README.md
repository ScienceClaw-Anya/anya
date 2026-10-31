# Norris 2024 reproduction audit and matched-task agents

This track reconstructs **SPES-based SOZ electrode classification** from [Norris et al., MLHC 2024 / PMLR 252](https://proceedings.mlr.press/v252/norris24a.html). It is separate from the VEP diagnostic stimulation-selection demonstration.

**Current status:** all 48 public BrainVision recordings for the 36 SOZ-labelled metadata candidates have been downloaded and checksum-verified. Original preprocessing retains 35 patients, **2,145 electrodes and 304 SOZ-positive electrodes**. The paper reports **35 / 2,066 / 298**. This is a real unresolved replication discrepancy. No contacts have been removed to force agreement, and no paper accuracy has been presented as an executed result.

## Pinned sources

- Official model repository: `norrisjamie23/Localising_SOZ_from_SPES`, commit `90062e8c5fe5593713125eaf5f573faedb739f65` (MIT).
- Dataset: [OpenNeuro ds004080 v1.2.4](https://openneuro.org/datasets/ds004080/versions/1.2.4), metadata commit `c4fd7418883e33b024292468eb14da1649f51aae` (CC0). Recordings are checked against the public annex SHA-256 and byte counts.
- Dataset article: van Blooijs et al., *Developmental trajectory of transmission speed in the human brain*, [Nature Neuroscience (2023)](https://doi.org/10.1038/s41593-023-01272-0).

The unmodified upstream source and generated tensors live under ignored `vendor/norris2024/`; raw recordings and receipts live under ignored `data/raw/ds004080/`. Do not commit clinical annotation tables, raw recordings, model checkpoints or credentials.

## Reproduction contract

`protocol.json` freezes the task, sources, patient split, metrics, caveats and agent comparison before viewing held-out performance.

Primary runs cover CNN divergent, CNN convergent and CNN Transformer base, each with five patient folds and five repeats: **75 training/evaluation runs**. Three folds train, one validates and one tests. The numerical score averages AUROC per patient, then within each fold, then across the 25 folds. All three models use the same saved patient splits.

Important distinctions:

- The authors' notebook explicitly states its scores differ from the paper because its example uses ten final-training epochs and one CV repeat. Appendix D's ten-epoch statement concerns hyperparameter tuning. Our reconstruction uses early stopping with patience 10; the 1,000-epoch ceiling follows the released `train_model` default and is an assumption because the paper's final-training ceiling is not stated.
- Selected full-precision hyperparameters come from the released notebook. This does **not** rerun the paper's separate ten-trial Optuna searches.
- Appendix C's per-patient median/MAD threshold and the released code's pooled validation threshold disagree. Both are scored separately. Undefined thresholds are reported explicitly; test labels never select a threshold.
- Original RNG states, file order, patient fold manifests and checkpoints are not released. This implementation freezes lexical file order, `PYTHONHASHSEED=0` and explicit training RNG seeds.
- The unavailable development PyTorch pin is replaced by released `2.1.0+cu121`. Windows paths are normalized before original path substitutions. The original TorchEEG `RandomNoise` class is loaded directly so unrelated optional spectral C extensions are not required. Numerical model source files are unchanged.

## Run on the current Windows workstation

Use the separate `.venv-repro`, leaving the working VEP environment intact. The workstation has an RTX 4090 with 24 GB GPU memory. CUDA is used for training; preprocessing uses the local CPU. No Nebius compute job is created by these scripts.

From the project root in PowerShell:

```powershell
$py = '.\.venv-repro\Scripts\python.exe'
$track = '.\reproduction\norris2024'
$out = '.\outputs\reproduction\norris2024'
$up = '.\vendor\norris2024'

& $py "$track\prepare_sources.py" --project '.'
& $py "$track\audit_data.py" --root '.\data\ds004080' --output "$out\data_manifest.json"
& $py "$track\download_data.py" --metadata-root '.\data\ds004080' --raw-root '.\data\raw\ds004080' --manifest "$out\data_manifest.json" --workers 4
& $py "$track\preprocess.py" --manifest "$out\data_manifest.json" --raw-root '.\data\raw\ds004080' --upstream $up --output $out --workers 3 --filter-jobs 2
& $py -m unittest discover -s $track -p 'test_reproduction.py' -v

# Explicitly approved public-code cohort; differences remain in the report.
$env:PYTHONHASHSEED = '0'
& $py "$track\train_reproduction.py" --upstream $up --output $out --parallel-workers 2 --allow-count-mismatch

# Omitting --allow-count-mismatch enforces the strict published-count gate.
# The flag never makes this an exact reproduction of the paper cohort.

& $py "$track\report.py" --output $out --html '.\outputs\reports\paper_reproduction.html'
```

The raw download is about 115 GiB. Downloads resume from partial files and check the complete file hash before preprocessing. Subject processing is resumable through receipts; failures remain visible in per-subject logs. Training resumes completed runs; it never substitutes demonstration scores.

For a new **Windows / Python 3.10** environment, `requirements-windows.lock.txt` contains every installed package from the tested environment. Install it with `pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 -r reproduction/norris2024/requirements-windows.lock.txt`. The complete package list supplies required dependencies; `--no-deps` avoids pulling unrelated optional TorchEEG spectral extensions. Linux requires its own platform-specific CUDA dependency resolution. `runtime.py` loads the original required TorchEEG transform unchanged.

After all stages finish, run `verify_results.py --project .` and `final_report.py --project .`. The latter writes compact, portable English results under `examples/` after requiring complete verified evidence.

## Agents on the same task

The user authorized the complete public-code reconstruction after reviewing the cohort mismatch. `agent_comparison.py` is gated on all 75 completed primary runs. An explicit `--allow-count-mismatch` records that the agent comparison uses the public-code cohort rather than an exact paper cohort. No Gemini comparison is claimed executed until result artifacts exist.

Classifier execution runs in a project wrapper; native ScienceClaw supplies `LLMClient` requests and `ArtifactStore` provenance.

- Baselines: frozen out-of-fold CNN Transformer and a fixed three-model mean ensemble.
- Single agent and four specialists share identical measured-response features and numerical predictors. Each receives eight Gemini calls per held-out case; both average their last four score vectors.
- Specialists cover signal responses, network response direction, uncertainty and critique, in two rounds of a shared board.
- All five CV repeats retain the paper's held-out patient assignments. Repeated evaluations of the same patient are not independent subjects.
- Contact names are replaced with opaque IDs. SOZ/resection/outcome annotations, patient identity, prevalence and test accuracy are absent from agent payloads. A test confirms that changing hidden labels cannot change the payload.
- Reflection critiques supplied observations and earlier proposals. It receives no test-label feedback. This is a fixed-data classification comparison, not a new active stimulation-selection experiment.
- Requests, validated score vectors, provider token usage and native ScienceClaw artifact lineage are saved. Call limits and output caps are matched; actual token counts may differ and are reported rather than assumed equal. Malformed returned predictions count against the eight-request budget and retain the previous score state (initially the common Transformer), explicitly marked as unaccepted. No extra request or invented agent rationale is used. Transport, authentication or quota failures stop the run. The shared board retains every prior state with four-decimal scores and bounded text; full outputs remain in the trace.

Run the agent stage with `.venv-repro`, which also contains the project's pinned OpenAI client, python-dotenv and PyYAML dependencies. Keep the API key in the ignored `.env`. `pipeline.py --project .` waits for all primary training evidence, runs the channel ablation and then starts the agent comparison, refreshing the English report throughout.

```powershell
& $py "$track\agent_comparison.py" --project '.' --upstream $up --output $out --workers 2 --allow-count-mismatch
```

## Additional paper figures and remaining blockers

`channel_sensitivity.py` reuses saved Transformer checkpoints at 1, 2, 4, 9, 18, 37 and 74 inference channels. Patient-level bootstrap intervals average repeated evaluations within a patient before resampling; the original sampling/bootstrapping code is not published.

The paper's Engel outcome analysis requires independently verified outcome labels and the exact modified model specification. Neither has been verified in the public model/data release, including the public derivative's available field names. SOZ or resection annotations must not stand in for Engel outcomes. The original spatial example's patient identity is also not stated. These stages cannot be claimed fully reproduced without the missing information.

The English HTML report keeps incomplete stages visibly pending and shows published and executed values in separate columns. Matching a rounded AUROC alone would not resolve cohort or protocol differences.


### Token calibration and fresh formal run

The first API trial used the inherited combined ceiling of 3,824 tokens. Six of 45 captured responses were truncated; 47 attempts (including two unresolved in-flight attempts) were stopped and archived under `token_calibration_agents_3824`. No localization scores were used to select the new ceiling. The fresh formal comparison pins a combined ceiling of **8,192 tokens**, low reasoning effort, eight requests per policy per case, eight workers and a shared maximum rate of 60 requests/minute. Calibration requests and captured tokens are reported separately from the 2,800 formal requests. The model metadata endpoint verified a 65,536-token model output limit.


### Budget-preserving continuation

A transport/API failure halts the batch. `resume_case.py --project . --case-id CASE --dry-run` checks recorded counters and frozen observations before any continuation. Recovery reuses original native artifacts and completed responses, counts the failed attempt, and runs only the unspent requests. Uncaptured output/token consumption is explicitly unknown; no provider response is invented. `formal-agent-source-lock.json` preserves the predictive source hashes from before formal inference.
