# Anya / NeuroTarget Swarm

ScienceClaw + Gemini agents propose and compare diagnostic stimulation experiments in a Virtual Epileptic Patient (VEP) sandbox.

本项目接入公开 VEP 数据、Epileptor 数值模拟和 ScienceClaw 原生组件。四个角色根据连接组、团队文献依据和已测 SEEG 摘要提出下一刺激位置，每次提议和测量均形成可追溯的 Artifact。

**研究原型：尚未完成盲测致痫区定位或临床验证。当前刺激模型用于诊断反应探索，不能据此推荐治疗靶点或宣称抑制发作。**

## Actual integration

| Component | Implementation |
|---|---|
| Tool discovery | Pinned upstream `SkillRegistry` discovers the three project skills |
| Tool execution | Upstream `SkillExecutor` runs each scientific tool as a JSON subprocess |
| Provenance | Native `Artifact` / `ArtifactStore` format, hashes, parent lineage and shared index |
| Gemini | Native `LLMClient(backend="gemini")` calls the official Google endpoint |
| Experiment control | Project adapter bounds simulation/LLM budgets, validates IDs and restricts observations |

The multi-agent loop is a bounded project adapter with independent profiles and shared artifacts. It does not run upstream ArtifactReactor, Infinite publishing, heartbeat daemons or optional PQC attestation. Registry/store subclasses keep state inside the project and omit publication/key-generation side effects. See `docs/integration.md`.

## Install

Use Python **3.10** for the tested numerical stack. Only the listed ScienceClaw components are integrated here; the complete upstream application recommends newer Python and has additional dependencies.

```powershell
git clone https://github.com/ScienceClaw-Anya/anya.git
cd anya
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-scienceclaw.txt
.\.venv\Scripts\python.exe scripts/setup_upstreams.py
```

Read and accept the [EBRAINS access terms](https://kg.ebrains.eu/search-terms-of-use.html?v=2.2) before downloading:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_vep.py --patient sub-003 --accept-ebrains-terms
.\.venv\Scripts\python.exe -m neurotarget inspect
.\.venv\Scripts\python.exe -m neurotarget minimum
```

`sub-003` has 162 regions and 179 available adjacent contact pairs. The minimum compares two distinct locations for 55 seconds each. Downloads, raw data, upstream source checkouts and generated outputs are ignored by Git.

## ScienceClaw + Gemini

Copy `.env.example` to `.env`, privately fill `GEMINI_API_KEY` from your activity project's pre-provisioned Workshop key, and save. A GCP email/password is a sign-in credential, not the API key. The configured model must be available to that key; it is not silently replaced.

```dotenv
LLM_BACKEND=gemini
GEMINI_MODEL=gemini-3.6-flash
GEMINI_API_KEY=YOUR_PRIVATE_WORKSHOP_KEY
GEMINI_REASONING_EFFORT=low
GEMINI_THINKING_HEADROOM=1024
GEMINI_MAX_RETRIES=0
```

```powershell
# Native tool/artifact integration without a key.
.\.venv\Scripts\python.exe -m neurotarget scienceclaw --mode demo --budget 4 --candidate-count 16 --seconds 12 --max-steps 4
# Real Gemini decisions through ScienceClaw's LLMClient.
.\.venv\Scripts\python.exe -m neurotarget scienceclaw --mode gemini --budget 4 --candidate-count 16 --seconds 12 --max-steps 4
# Matched observation-isolation ablation.
.\.venv\Scripts\python.exe -m neurotarget scienceclaw --mode gemini --budget 4 --candidate-count 16 --seconds 12 --max-steps 4 --isolated
```

Profiles are in `config/agents.json`: structure, dynamics, evidence and critic. Candidate count must exceed the experiment budget. Both SDK and upstream retries are disabled. Unknown targets/evidence IDs, invalid JSON and API failures stop explicitly, with no silent demo fallback.

Agent prompts contain geometry/network scores, attributed evidence cards and SEEG summaries. Latent x0/EZ labels, source activity and accumulation-threshold readouts are withheld. This observation gate alone does not establish a clinical localization benchmark.

## Results and checks

- Original numerical baseline: two sites, 55 seconds each, run locally and on Nebius CPU. Ten local/cloud source-variance and threshold-count summaries agreed; full cloud waveforms were not compared point by point.
- Native integration: four real Gemini decisions, four 12-second simulations among 16 candidates, ten artifacts with verified hashes and lineage. Public portable summaries are in `examples/`.
- Reports: `outputs/reports/minimum.html`, `scienceclaw_demo_shared.html` and `scienceclaw_gemini_shared.html`.
- Native state: `outputs/scienceclaw/<session>/session.json`, `artifacts/<agent>/store.jsonl` and `artifacts/global_index.jsonl`.
- A 12-second experiment demonstrates early responses, not a full-seizure outcome. Very short runs are smoke checks only.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The full local suite requires downloaded VEP data. CI runs deterministic offline native-integration tests against the pinned upstream, without Gemini credentials or automatic acceptance of data terms.

## Scientific limits

1. Uses the author's EpileptorStim2Populations and custom stochastic Heun integrator. Its coupling declaration `[0]` conflicts with three channels read by the compiled derivative; our subclass explicitly declares `[0,3,6]`. This is a documented model variant, not confirmed exact reproduction of the released paper run.
2. Released initial-state metadata is truncated and its random seed is unavailable. Our explicit initial state comes from the author script. Time conversion uses released `sfreq=400` scaling.
3. Accumulation threshold crossings are model diagnostics, not validated seizure labels. Signals are model units, not calibrated clinical voltage.
4. This is forward simulation; it has not trained a new network or fitted a patient-specific inference model.
5. One virtual patient cannot establish localization accuracy, pathway causality, clinical benefit or multi-agent superiority. Held-out patients, matched-budget baselines, coverage controls and model-mismatch evaluations remain future work.

## Literature and cloud

The team's [epilepsy-virtual-brain review](https://github.com/ScienceClaw-Anya/epilepsy-virtual-brain) supplies attributed method context. Five cards in `research/evidence.json` record the source commit, hashes and URLs. This integration does not independently repeat its full literature audit or implement every proposed onset/bottleneck/control method.

See [cloud/README.md](cloud/README.md) for Nebius. TVB currently uses CPU; Gemini is an external API. Keys, private PDFs, user-specific cloud IDs and existing storage configuration are not included in public code or generated bundles.

References: VEP dataset [10.25493/XWG2-S8X](https://doi.org/10.25493/XWG2-S8X), CC BY 4.0; Dollomaja et al. (2025), [10.1371/journal.pcbi.1012911](https://doi.org/10.1371/journal.pcbi.1012911); [ScienceClaw](https://github.com/lamm-mit/scienceclaw); [author code](https://github.com/BalanceKey/virtual_epilepsy_patient_cohort); [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai).

See `THIRD_PARTY_NOTICES.md` for attribution and licensing boundaries.
