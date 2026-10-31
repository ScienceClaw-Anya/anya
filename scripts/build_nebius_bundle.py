"""Create a single <64 KiB, credential-free Nebius injected bootstrap file."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import zipfile


def build(project: Path):
    project = project.resolve()
    vendor = project / "vendor/virtual_epilepsy_patient_cohort"
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    files = sorted((project / "neurotarget").glob("*.py"))
    files += [project / "scripts/fetch_vep.py", project / "requirements.txt", project / "requirements-scienceclaw.txt", project / "requirements-real-data.txt"]
    files += [project / "config/upstream-lock.json", project / "config/agents.json", project / "config/diagnostic-benchmark.json", project / "research/evidence.json", project / "research/datasets.json"]
    for name in ["neurotarget-catalog", "neurotarget-simulate", "neurotarget-evidence", "neurotarget-response"]:
        directory = project / "skills" / name
        files += [directory / "SKILL.md"] + sorted((directory / "scripts").glob("*.py"))
    scienceclaw = json.loads((project / "config/upstream-lock.json").read_text(encoding="utf-8"))["scienceclaw"]
    if not scienceclaw.get("archive_sha256"):
        raise RuntimeError("The pinned ScienceClaw source archive checksum is required for cloud execution")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.writestr(path.relative_to(project).as_posix(), path.read_bytes())
    # Solid compression shares repeated source text across files and keeps the
    # expanded diagnostic package below Nebius's injected-file size limit.
    archive_bytes = lzma.compress(archive_buffer.getvalue(), preset=9)
    encoded = base64.b64encode(archive_bytes).decode()
    vendor_paths = ["src/utils_functions/model_2populations.py", "src/utils_functions/integrator.py"]
    # Hash canonical Git blobs, not Windows checkouts with autocrlf conversion.
    # Verify local content differs only by line endings before packaging.
    checksums = {}
    for path in vendor_paths:
        blob = subprocess.check_output(["git", "-C", str(vendor), "show", commit + ":" + path])
        checkout = (vendor / path).read_bytes().replace(b"\r\n", b"\n")
        if checkout != blob.replace(b"\r\n", b"\n"):
            raise RuntimeError("Local upstream code is modified; review and pin it before cloud execution")
        checksums[path] = hashlib.sha256(blob).hexdigest()
    template = r'''# Generated NeuroTarget Swarm job. Contains no secrets or private patient records.
import base64, hashlib, io, json, lzma, os, pathlib, shutil, subprocess, sys, time, urllib.request, zipfile

PAYLOAD = "__PAYLOAD__"
PAYLOAD_SHA256 = "__PAYLOAD_SHA__"
VEP_COMMIT = "__COMMIT__"
VENDOR_HASHES = __VENDOR_HASHES__
SCIENCECLAW_SOURCE = __SCIENCECLAW_SOURCE__

def main():
    tag = os.environ.get("NEUROTARGET_RUN_TAG", "neurotarget-" + time.strftime("%Y%m%d-%H%M%S"))
    if not all(c.isalnum() or c in "-_" for c in tag):
        raise ValueError("Invalid run tag")
    # Persistent output is preferred. An explicitly requested smoke-validation
    # mode uses only the job's temporary disk and prints summaries to job logs.
    ephemeral = os.environ.get("NEUROTARGET_ALLOW_EPHEMERAL") == "1"
    artifacts = pathlib.Path("/tmp/neurotarget-artifacts" if ephemeral else os.environ.get("NEUROTARGET_ARTIFACT_DIR", "/artifacts"))
    if ephemeral:
        artifacts.mkdir(parents=True, exist_ok=True)
    elif not artifacts.is_dir():
        raise RuntimeError("Attach persistent storage at /artifacts or explicitly select ephemeral validation")
    destination = artifacts / tag
    if destination.exists():
        raise RuntimeError("Output tag already exists; choose a new run tag")
    project = pathlib.Path("/tmp") / tag
    project.mkdir(parents=True, exist_ok=False)
    payload = base64.b64decode(PAYLOAD, validate=True)
    if hashlib.sha256(payload).hexdigest() != PAYLOAD_SHA256:
        raise RuntimeError("Bundle checksum mismatch")
    payload = lzma.decompress(payload, memlimit=128 * 1024 * 1024)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            relative = pathlib.PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe bundle member")
            target = project.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    for path, checksum in VENDOR_HASHES.items():
        url = "https://raw.githubusercontent.com/BalanceKey/virtual_epilepsy_patient_cohort/" + VEP_COMMIT + "/" + path
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeError("Upstream code checksum mismatch")
        target = project / "vendor/virtual_epilepsy_patient_cohort" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    archive_url = "https://codeload.github.com/lamm-mit/scienceclaw/zip/" + SCIENCECLAW_SOURCE["commit"]
    archive_bytes = urllib.request.urlopen(archive_url, timeout=90).read()
    if hashlib.sha256(archive_bytes).hexdigest() != SCIENCECLAW_SOURCE["archive_sha256"]:
        raise RuntimeError("ScienceClaw source archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        runtime_entries = [entry for entry in archive.infolist()
            if len(pathlib.PurePosixPath(entry.filename).parts) >= 3
            and pathlib.PurePosixPath(entry.filename).parts[1] in {"core", "artifacts", "deps"}
            and entry.filename.endswith(".py")]
        if sum(entry.file_size for entry in runtime_entries) > 20 * 1024 * 1024:
            raise RuntimeError("Unexpected ScienceClaw runtime source size")
        for entry in runtime_entries:
            relative = pathlib.PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe ScienceClaw archive member")
            if entry.is_dir():
                continue
            if len(relative.parts) < 2 or relative.parts[0] != "scienceclaw-" + SCIENCECLAW_SOURCE["commit"]:
                raise RuntimeError("Unexpected ScienceClaw archive prefix")
            target = project / "vendor/scienceclaw"
            target = target.joinpath(*relative.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    mode = os.environ.get("NEUROTARGET_CLOUD_AGENT_MODE", "demo")
    if mode not in {"demo", "gemini"}:
        raise ValueError("Unsupported cloud agent mode")
    if mode == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("Gemini mode requires a privately injected runtime key")
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-scienceclaw.txt"], cwd=project, check=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", NEUROTARGET_EXECUTION_BACKEND="nebius-cloud-ai")
    commands = [
        [sys.executable, "scripts/fetch_vep.py", "--patient", "sub-003", "--accept-ebrains-terms", "--no-reference"],
        [sys.executable, "-m", "neurotarget", "minimum"],
        [sys.executable, "-m", "neurotarget", "scienceclaw", "--mode", mode, "--budget", "4", "--candidate-count", "16", "--seconds", "12", "--max-steps", "4"]
    ]
    started = time.time()
    status = {"backend": "nebius-cloud-ai", "run_tag": tag, "vep_commit": VEP_COMMIT,
              "payload_sha256": PAYLOAD_SHA256, "python": sys.version, "cpu_count": os.cpu_count(),
              "scienceclaw_commit": SCIENCECLAW_SOURCE["commit"], "agent_mode": mode,
              "persisted_artifacts": not ephemeral, "ephemeral_validation": ephemeral}
    try:
        for command in commands:
            subprocess.run(command, cwd=project, env=environment, check=True)
        status["success"] = True
        for filename in ["minimum.json", "scienceclaw_" + mode + "_shared.json"]:
            document = json.loads((project / "outputs/reports" / filename).read_text())
            summaries = [{"target_id": item["target_id"], "duration_seconds": item["duration_seconds"],
                          "wall_seconds": item["wall_seconds"],
                          "post_source_variance": item["summary"]["post_source_variance"],
                          "threshold_crossing_regions": item["summary"]["threshold_crossing_regions"]}
                         for item in document["results"]]
            print("NEUROTARGET_REPORT " + json.dumps({"report": filename, "results": summaries,
                "event_count": len(document["events"])}), flush=True)
    except Exception as error:
        status["success"] = False
        status["error_type"] = type(error).__name__
        raise
    finally:
        destination.mkdir(parents=True, exist_ok=False)
        for directory in ["outputs", "data"]:
            source = project / directory
            if source.exists():
                shutil.copytree(source, destination / directory)
        status["wall_seconds"] = time.time() - started
        (destination / "cloud_run.json").write_text(json.dumps(status, indent=2))
        print("NEUROTARGET_CLOUD_RESULT " + json.dumps(status), flush=True)
        print(("TEMPORARY_ARTIFACTS " if ephemeral else "ARTIFACTS ") + str(destination), flush=True)

if __name__ == "__main__":
    main()
'''
    script = template.replace("__PAYLOAD__", encoded).replace("__PAYLOAD_SHA__", hashlib.sha256(archive_bytes).hexdigest())
    script = script.replace("__COMMIT__", commit).replace("__VENDOR_HASHES__", repr(checksums))
    script = script.replace("__SCIENCECLAW_SOURCE__", repr(scienceclaw))
    compile(script, "nebius_bootstrap.py", "exec")
    encoded_script = script.encode("utf-8")
    if len(encoded_script) > 64 * 1024:
        raise RuntimeError("Bundle exceeds Nebius's 64 KiB injected-file limit")
    output = project / "cloud/nebius_bootstrap.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded_script)
    provenance = {"commit": commit, "vendor_sha256": checksums,
                  "payload_encoding": "base64 of LZMA-compressed ZIP",
                  "bootstrap_bytes": len(encoded_script), "payload_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                  "bundle_sha256": hashlib.sha256(encoded_script).hexdigest(),
                  "included_files": [p.relative_to(project).as_posix() for p in files]}
    (project / "cloud/bundle_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
