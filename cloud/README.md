# Nebius Cloud AI

Use a one-off self-terminating CPU job. Gemini requests go to Google; the current TVB simulator does not use a GPU or require a persistent model endpoint.

## Build the credential-free job file

After dependency installation and `scripts/setup_upstreams.py`:

```powershell
python scripts/build_nebius_bundle.py
```

Generated local files `cloud/nebius_bootstrap.py` and `cloud/bundle_manifest.json` are ignored by Git. The injected bootstrap must remain below 64 KiB. It includes a whitelist of project Python files, three skills, public evidence cards and dependency pins. It excludes `.env`, keys, private PDFs, browser profiles and user-specific resource IDs.

At launch it verifies its payload, downloads pinned VEP files and a SHA-256-verified ScienceClaw source archive, installs requirements, downloads sub-003 and runs the two-site minimum plus four ScienceClaw experiments among 16 candidates. Default agent mode is `demo`. This revised bundle is prepared for validation; the earlier successful cloud job used the previous deterministic worker implementation.

## Console settings

- Image: `python:3.10-slim`.
- Entrypoint: `python3 /mnt/files/nebius_bootstrap.py`.
- Inject the generated file at `/mnt/files/nebius_bootstrap.py`.
- Small CPU preset; the earlier tested job used 2 vCPU / 8 GiB. Set a 1-hour timeout and no automatic restart.
- Attach an existing private bucket read/write at `/artifacts`, use `NEUROTARGET_ALLOW_EPHEMERAL=0` and a unique `NEUROTARGET_RUN_TAG`.
- Choose `NEUROTARGET_CLOUD_AGENT_MODE=demo` or `gemini`. Gemini needs a privately injected `GEMINI_API_KEY` and `GEMINI_MODEL`. Never embed a key in the injected source or public job template.

Explicit `NEUROTARGET_ALLOW_EPHEMERAL=1` uses temporary disk. Arrays disappear with that disk, while printed summaries remain in job logs. Completion requires the console terminal status and `NEUROTARGET_CLOUD_RESULT` with `success: true`.

The approved 1 GiB private result bucket was created after the original ephemeral job completed. Its actual ID and future mount configuration remain in ignored local `cloud/persistent-storage.json`. Reports and log backups were saved there; original cloud arrays were not recovered. Building this file or creating a bucket does not submit an extra compute job.

Reference: https://docs.nebius.com/serverless/jobs/manage
