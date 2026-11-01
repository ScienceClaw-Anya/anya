# Reproduction and execution record

Local checkout: `/Users/step5/Documents/ChatGPT/ScienceClaw 2026/anya-intervention-research`.
Remote execution mirror: `/home/devstar3022/anya-intervention-research` on the
existing SSH alias `scienceclaw-gcp`. No remote checkout or existing jobs are
modified. The mirror has no `.git`; local commits are the source of record.

Audit/split milestone: `a042619`. Pre-intervention code and development manifest:
`8d24247`. Both were pushed to the explicit destination
`origin HEAD:refs/heads/codex/personalized-intervention-research`.
No main branch fetch/change/merge, integration PR or server provisioning.

Initial local checks: `git status --short --branch`, `git remote -v`,
`git config --get core.sshCommand`, `df -h .`, `uptime`, read-only `sysctl`,
`ps`, `vm_stat`. Sandbox blocked resource inspection; a read-only escalation
succeeded. The handoff main SHA was not present in this independent clone;
no main fetch was needed. Upstream code read through the GitHub connector.

`scripts/acquire_vec.py --inventory` queried the public EBRAINS Data Proxy
listing API, with pagination. Acquisition used explicitly selected file names;
URLs, checksums, bytes and retrieval times are in `manifests/acquisition.json`.
The initial selection inadvertently included 3 reference PNGs, recorded in the
download total. Only non-image development assets were copied to GCP.

On GCP, a new project-local `.venv` was created with Python 3.12.3. Packages were
installed from PyPI using pip; exact resulting versions are in
`requirements-gcp.lock.txt`. The local environment created before the remote
instruction is recorded separately in `requirements.lock.txt`; it was not used
for experiments. Numerical dependencies occupy ignored environment directories.

Run commands on GCP from its absolute directory:

```sh
cd /home/devstar3022/anya-intervention-research
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nice -n 10 .venv/bin/python -m pytest -q tests
timeout 600 nice -n 10 .venv/bin/python scripts/development_baseline.py
nice -n 10 .venv/bin/python scripts/run_pilot.py prepare
# Retrieve and commit the prepared manifest before running interventions.
timeout 3000 nice -n 10 .venv/bin/python scripts/run_pilot.py run
nice -n 10 .venv/bin/python scripts/replay_pilot.py
```

For a fresh reproduction, use a fresh output directory/execution mirror and the
pre-intervention source commit. The original run must remain immutable. The
pilot checks code hashes and the wall clock from `outputs/pilot-budget.json`.
It refuses to begin another call with less than 90 seconds remaining. The
outer process timeout supplies a second bound. Raw trajectories stay in ignored
GCP outputs; only compact summaries are copied into this branch.

This run used one process and one numerical thread at nice priority 10. Remote
resource inspection showed existing `vep_v3` jobs; none were signaled or altered.
SSH/resource/log inspection is read-only apart from this isolated directory.

IDEAS direct downloads returned HTTP 202 with a zero-length WAF response. A
browser download request was explicitly denied. No alternate download route was
attempted after denial; record-level eligibility and outcome fitting remain undone.

## CPU development v2

User-authorized continuation used one GCP worker/thread at nice 10. Pre-run
milestones: d6ca090 (scope/projection audit), d4a713e (network controls),
5182533 (linear fit/replay), c53c9d8 (Epileptor fit), eb0e3e5 (separate exploratory
cross-recording addendum). Original fitting manifest is unchanged from c53c9d8.
All commits were pushed only to the explicit isolated branch destination.

After scoped source/manifest synchronization with rsync, commands ran serially
on GCP with the same numerical-thread environment as v1:

```sh
timeout 180 nice -n 10 .venv/bin/python scripts/audit_projection_v2.py
timeout 300 nice -n 10 .venv/bin/python scripts/acquire_development_v2.py
timeout 90 nice -n 10 .venv/bin/python -m pytest -q tests
timeout 1000 nice -n 10 .venv/bin/python scripts/network_sensitivity_v2.py
timeout 60 nice -n 10 .venv/bin/python scripts/audit_sources_v2.py
timeout 120 nice -n 10 .venv/bin/python scripts/fit_forecast_v2.py
timeout 90 nice -n 10 .venv/bin/python scripts/replay_development_v2.py
timeout 700 nice -n 10 .venv/bin/python scripts/fit_epileptor_v2.py
timeout 350 nice -n 10 .venv/bin/python scripts/evaluate_epileptor_fit_v2.py
timeout 90 nice -n 10 .venv/bin/python scripts/replay_fit_v2.py
timeout 60 nice -n 10 .venv/bin/python scripts/cross_recording_fit_v2.py
timeout 60 nice -n 10 .venv/bin/python scripts/plot_development_v2.py
```

Tests were rerun as new fitting invariants were added; final result: 9 passed.
Full raw arrays remain in `outputs/development-v2` on GCP. Only compact summaries,
the aggregate figure and acquisition hashes were retrieved locally. Figure
calculations ran on GCP and the final image was visually inspected locally.
The figure was revised to show the zero-covariance reference explicitly.

Each simulation checks the shared `outputs/development-v2/budget.json` deadline.
The scripts refuse to overwrite existing stage output directories; rerun a new
versioned stage rather than modifying the recorded results. Exact source/data
hashes are in the manifests and output summaries. No other job was signaled or
changed. No local simulation, numerical test or metric replay was run.
