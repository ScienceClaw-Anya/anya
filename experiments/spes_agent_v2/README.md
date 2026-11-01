# SPES agent experiments, version 2

This track tests whether independent deliberation helps on the same retrospective SOZ contact-ranking task as the Norris et al. public-code reconstruction. It also tests a small, validation-trained fusion model. It preserves `reproduction/norris2024` and its earlier results.

See the [English results report](../../docs/spes-agent-v2-report.md) and [portable aggregate evidence](../../examples/spes_agent_v2_summary.json). These are research comparisons, not clinical stimulation recommendations.

## Questions and controls

1. **Does first-round independence help?** Compare independent-first specialists with serial specialists. They have the same measured observations, four role profiles, model routing and eight request attempts. Only proposal visibility changes. A no-exchange arm helps distinguish discussion from repeated independent predictions.
2. **Can trained fusion outperform hand-written LLM scores?** Fit fixed L2 logistic models on the seven original validation patients per outer split. One uses three numerical predictions; the other adds 15 measured response features. Test patients never fit these models or their scalers.
3. **Can roles use real scientific tools?** Native ScienceClaw `SkillExecutor` runs a bounded CLI returning frozen measured-response summaries, validation reliability and trained numerical predictions. Both the single generalist and specialists receive identical tool access and eight tool queries. They select an executable scorer. This bundles tool access with a different output interface; it is not an isolated tool-access ablation against manual vectors.

The role profiles are signal, measured network, uncertainty and critic. The critic can reject unsupported score changes. Reflections use visible observations and earlier reasoning; there is no clinical-label or accuracy feedback during decisions. No LLM fine-tuning is performed in this version.

| Policy | Proposal visibility | Output | Requests / patient |
|---|---|---|---:|
| `single_serial` | Every earlier generalist proposal | Manual contact scores | 8 |
| `multi_serial` | Every earlier specialist proposal | Manual contact scores | 8 |
| `multi_independent_first` | Empty first four boards, then earlier proposals | Manual contact scores | 8 |
| `multi_no_exchange` | Always empty | Manual contact scores | 8 |
| `single_tools` | Every earlier generalist proposal | Select numerical scorer | 8 |
| `multi_independent_tools` | Empty first four boards, then earlier proposals | Select numerical scorer | 8 |

All arms average their final four score states. Independent or no-exchange failures fall back to the common Transformer baseline when no proposal is visible; hidden proposals cannot enter through error recovery.

Budget equality means the same number of requests and the same maximum completion-token allowance. Realized input/output tokens and billed cost can differ because exchanged proposals add context and manual vectors are longer than scorer selections. Usage is retained; exact monetary cost is not inferred without a verified price schedule.

## Model and compute configuration

`--routing argon_control` uses **Gemini 4 Argon**, medium reasoning, for every call. This is the first executable study. Calls use the pinned native ScienceClaw `LLMClient`, the official Google endpoint, 32,768 combined completion tokens, a 900-second timeout and zero automatic retries. Actual model identity and completion status are checked and recorded. Sampling is controlled by the service; the Gemini compatibility layer does not receive a temperature setting.

`--routing mixed` requires a separate **Nemotron 3 Super 120B A12B** model API credential. Every policy uses the same repeated schedule: Argon, Nemotron, Argon, Nemotron. Thus even the single-agent control is one logical generalist routed over two base models, with four calls to each. This matches model access; it does not compare a single-base-model agent with a two-model ensemble.

The supplied Scientific AI key was tested once against the official Token Factory model catalog, with permission, and returned HTTP 401. It is not reused at other endpoints. A workbench listing a model does not establish that a separate key can call its API. The optional mixture remains unavailable until a dedicated credential is configured privately.

The 50 logistic fusion fits took less than one CPU second in total locally. Frozen CNN/Transformer outputs came from the earlier local RTX 4090 reconstruction. Gemini inference runs on Google's service. No additional Nebius GPU job or storage resource is needed for this version. A long LLM run is dominated by remote inference, so moving the small local fusion fits to a cloud GPU would not solve that delay.

## Reproduce

Use the existing reconstruction environment and complete the [numerical reconstruction](../../reproduction/norris2024/README.md) first. Alternatively install the explicit v2 requirements into a Python 3.10 environment. Do not upgrade or modify the frozen numerical environment while reproducing earlier checkpoints.

```powershell
python -m pip install -r requirements-spes-agent-v2.txt
python scripts/setup_upstreams.py --only scienceclaw
python -m unittest experiments.spes_agent_v2.test_experiment -v
python -m experiments.spes_agent_v2.data --project . --output outputs/spes_agent_v2
python -m experiments.spes_agent_v2.verify --project . --output outputs/spes_agent_v2
python -m experiments.spes_agent_v2.providers --project . --output outputs/spes_agent_v2 --providers argon
python -m experiments.spes_agent_v2.engine --project . --output outputs/spes_agent_v2 --routing argon_control --workers 16
python -m experiments.spes_agent_v2.verify --project . --output outputs/spes_agent_v2 --routing argon_control
python -m experiments.spes_agent_v2.report --project . --output outputs/spes_agent_v2
```

The run freezes source and case hashes before any formal requests. Re-running the exact command resumes unspent steps. A successful cached response can be parsed without another request; a reserved or failed attempt stays spent. Transport errors halt new scheduling after the currently active tasks finish. Never delete usage records to obtain a replacement answer. Preflight requests are synthetic, separately stored and excluded from performance. A failed preflight requires an explicitly numbered `--attempt 2` after diagnosis; it is not retried automatically.

The first-repeat communication study has **35 distinct held-out patients**, six arms and **1,680 formal request attempts**. The primary comparison is `multi_independent_first - multi_serial`. Further contrasts are exploratory. `--max-patients` selects a deterministic execution prefix for operational checks; a partial prefix is not a full-cohort result. The report withholds full-cohort performance until all specified arms and offline audit finish.

If a dedicated Nemotron credential becomes available, privately configure one of the following in ignored `.env`:

```dotenv
# Choose one; never commit or paste the value in a public document.
NEBIUS_TOKEN_FACTORY_API_KEY=
# Or:
NVIDIA_API_KEY=
```

The default official endpoints are `https://api.tokenfactory.us-central1.nebius.com/v1/` and `https://integrate.api.nvidia.com/v1/`, respectively. `V2_NEMOTRON_KEY_ENV` can explicitly select either variable. `V2_NEMOTRON_BASE_URL` must match that credential's provider; arbitrary destinations are rejected. Then run the provider preflight for `nemotron` and the engine with `--routing mixed`. The two studies have separate protocols and results. No fallback silently changes Nemotron to another model.

## Data and training gate

The public-code cohort contains 35 patients / 2,145 contacts / 304 SOZ contacts, whereas the paper reports 35 / 2,066 / 298. These experiments are not an exact paper-cohort replication. The cohort was already inspected in the earlier experiment, so the follow-up is exploratory.

For each of five folds and five repeats, the original 21 numerical-training patients, seven validation patients and seven test patients are disjoint. Upstream validation patient values are local indices `0..6`; `validation_patient_ids` explicitly maps them back to real split IDs. Treating them as actual subject IDs would break alignment. Contact order is checked against the released tensors.

Both fusion models use weighted `StandardScaler` and logistic regression (`C=0.1`, L2, lbfgs, fixed seed). Each validation patient has equal total weight. The seven validation patients also supported the original numerical model's early stopping. There is no hyperparameter search, test-driven choice of C or refitting after test inspection.

The 15 response features are six temporal RMS ranks, two valid-pair fractions, four peak magnitude/latency ranks and three outgoing-minus-incoming RMS ranks. Pair distances must exceed 13 mm and waveforms must be finite. RMS windows are samples `[0:47]`, `[47:175]`, `[175:end]` of the saved mean responses. Features are descriptive within-patient summaries; geometric neighbors and directional responses do not establish causal pathways.

`public.json` and `tools.json` omit patient identity, clinical labels, outcome, prevalence and held-out accuracy. Only the final judge reads `judge.json`, after all eight decisions are saved. This is a programmatic observation gate, not an operating-system security boundary: the host process prepares and scores data, while models receive only the serialized public packet.

## Evidence and interpretation

All generated evidence stays under ignored `outputs/spes_agent_v2`:

- `fusion/`: fixed protocol, 50 saved fusion models, predictions and source receipts.
- `fusion_audit.json`: patient isolation, weighted scaler inputs, coefficient-based prediction replay and independent pairwise AUROC.
- `cases/`: public observation/tool banks and separate hidden judges.
- `provider_preflight/`: synthetic interface checks, outside the formal budget.
- `studies/ROUTING/`: frozen protocol, per-request prompts/usage, accepted outputs or counted carries, and native ScienceClaw artifacts.

The study audit checks source/case hashes, all eight attempts per completed case, exact board visibility and artifact parents, prediction aggregation and independently recomputed AUROC. It also supports `--allow-incomplete` for operational inspection without declaring the cohort complete. Raw recordings, clinical labels, credentials, checkpoint files and full API outputs are excluded from Git; only code, documentation and aggregate evidence are committed.

AUROC evaluates SOZ ranking, not therapeutic benefit or the accuracy of choosing an unobserved stimulation site. Patient-bootstrap intervals resample the 35 distinct patients. The five-repeat fusion results cannot be compared directly with a first-repeat agent number without matching the repeat. Agreement across agents is not evidence of accuracy, and better results after model replacement would not by themselves demonstrate a collaboration benefit.
