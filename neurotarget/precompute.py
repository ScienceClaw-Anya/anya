"""Build a reusable response bank; policies only receive their queried entries."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .data import Patient
from .simulation import RunSpec, run
from .swarm import choose_candidates


def _one(job):
    project, patient_id, target, seconds, seed, scale, directory = job
    patient = Patient.load(Path(project), patient_id)
    result = run(patient, RunSpec(target, seed=seed, duration_seconds=seconds, stimulation_scale=scale))
    with np.load(Path(result["output_dir"]) / "timeseries.npz") as data:
        stop = float(patient.stimulation_parameters["stimulation_length"]) / float(patient.simulator_parameters["sfreq"])
        selected = data["time_seconds"] >= stop
        if selected.sum() < 100:
            raise ValueError("Diagnostic inference requires at least 100 post-stimulation samples")
        sensor = data["sensor_signal"][selected]
        covariance = np.cov(sensor, rowvar=False, ddof=0)
    identifier = "control" if scale == 0 else hashlib.sha256(target.encode()).hexdigest()[:12]
    feature_file = Path(directory) / (identifier + ".npz")
    np.savez_compressed(feature_file, covariance=covariance)
    return {"target_id": target, "stimulation_scale": scale, "feature_file": str(feature_file),
            "feature_sha256": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
            "record_path": str(Path(result["output_dir"]) / "result.json"),
            "window_start_seconds": stop, "samples": int(selected.sum()),
            "cache_hit": bool(result["cache_hit"]), "original_wall_seconds": result["wall_seconds"]}


def build_bank(project: Path, patient_id="sub-003", candidate_count=16, seconds=55., seed=42, workers=4):
    project = project.resolve()
    if not 2 <= candidate_count <= 179 or not 12 <= seconds <= 55 or not 1 <= workers <= 8:
        raise ValueError("Require 2..179 candidates, 12..55 seconds and 1..8 workers")
    patient = Patient.load(project, patient_id)
    catalog = choose_candidates(patient, candidate_count)
    spec = {"patient": patient_id, "candidate_count": candidate_count, "seconds": seconds,
            "seed": seed, "data_fingerprint": patient.fingerprint, "feature_version": 1}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    directory = project / "outputs/response_banks" / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "bank.json"
    # Keep completed entries recoverable if a later worker fails.
    state = {"spec": spec, "catalog": catalog, "channels": patient.channels, "entries": [], "complete": False,
             "execution": {"host": platform.node(), "platform": platform.platform(),
                           "logical_cpus": os.cpu_count(), "workers": workers,
                           "location": "local workstation", "gpu_used": False}}
    jobs = [(str(project), patient_id, c["target_id"], seconds, seed, 1., str(directory)) for c in catalog]
    jobs.insert(0, (str(project), patient_id, catalog[0]["target_id"], seconds, seed, 0., str(directory)))
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            state["entries"].append(item)
            manifest.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"BANK {len(state['entries'])}/{len(jobs)}: {item['target_id']} scale={item['stimulation_scale']}", flush=True)
    state["complete"] = True
    state["execution"]["bank_wall_seconds"] = time.perf_counter() - start
    manifest.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--patient", default="sub-003")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=55.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(build_bank(args.project, args.patient, args.candidate_count, args.seconds, args.seed, args.workers))


if __name__ == "__main__":
    main()
