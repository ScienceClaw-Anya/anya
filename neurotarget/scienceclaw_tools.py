"""JSON-only scientific tools called by the real ScienceClaw SkillExecutor."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys


def evidence(project: Path):
    return json.loads((project / "research/evidence.json").read_text(encoding="utf-8"))


def main(tool: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--format", choices=["json"], default="json")
    parser.add_argument("--patient", default="sub-003")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--target-id")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bank-path", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    if tool == "response":
        from .response_bank import read_response
        if not args.bank_path or not args.target_id:
            raise ValueError("Response queries require bank-path and target-id")
        result = read_response(project, args.bank_path, args.target_id)
    elif tool == "evidence":
        result = evidence(project)
    else:
        import numpy as np
        from .data import Patient
        from .swarm import choose_candidates
        patient = Patient.load(project, args.patient)
        if tool == "catalog":
            if not 2 <= args.limit <= 179:
                raise ValueError("Candidate limit must be between 2 and 179")
            result = {"patient": patient.patient_id, "results": choose_candidates(patient, args.limit),
                      "observation_policy": "SEEG summaries only; no x0/EZ/accumulation thresholds in agent prompts"}
        elif tool == "simulate":
            from .simulation import RunSpec, run
            if not args.target_id:
                raise ValueError("A catalog target-id is required")
            if not 0 < args.seconds <= 55:
                raise ValueError("Duration must be >0 and <=55 seconds")
            # Progress belongs on stderr; stdout must remain parseable JSON.
            with contextlib.redirect_stdout(sys.stderr):
                record = run(patient, RunSpec(args.target_id, seed=args.seed, duration_seconds=args.seconds))
            with np.load(Path(record["output_dir"]) / "timeseries.npz") as data:
                time = data["time_seconds"]
                stop = float(patient.stimulation_parameters["stimulation_length"]) / float(patient.simulator_parameters["sfreq"])
                selected = time >= stop
                fallback = not selected.any()
                if fallback:
                    selected = np.ones(len(time), dtype=bool)
                power = np.var(data["sensor_signal"][selected], axis=0)
            result = {"target_id": args.target_id, "record_path": str(Path(record["output_dir"]) / "result.json"),
                      "cache_hit": record["cache_hit"], "observation": {
                          "target_id": args.target_id, "duration_seconds": record["duration_seconds"],
                          "sensor_names": patient.channels, "sensor_power": power.tolist(),
                          "mean_sensor_power": float(power.mean()),
                          "window": "whole-record smoke check" if fallback else "after stimulation ends",
                          "units": "model units squared; not clinically calibrated voltage"}}
        else:
            raise ValueError("Unknown scientific tool")
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
