from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Patient
from .simulation import RunSpec, run
from .report import write_report


def main():
    parser = argparse.ArgumentParser(description="NeuroTarget Swarm VEP experiments")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--patient", default="sub-003")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    minimum = commands.add_parser("minimum", help="Run one patient and two real electrode-pair locations")
    minimum.add_argument("--seconds", type=float)
    minimum.add_argument("--seed", type=int, default=42)
    minimum.add_argument("--targets", nargs=2)
    agents = commands.add_parser("agents")
    agents.add_argument("--mode", choices=["demo", "llm"], default="demo")
    agents.add_argument("--budget", type=int, default=4)
    agents.add_argument("--seconds", type=float)
    agents.add_argument("--seed", type=int, default=42)
    agents.add_argument("--isolated", action="store_true")
    agents.add_argument("--baseline", choices=["collective", "single", "random", "network"], default="collective")
    framework = commands.add_parser("scienceclaw", help="Native ScienceClaw tools/artifacts with bounded demo or Gemini agents")
    framework.add_argument("--mode", choices=["demo", "gemini"], default="demo")
    framework.add_argument("--budget", type=int, default=8)
    framework.add_argument("--candidate-count", type=int, default=32)
    framework.add_argument("--seconds", type=float, default=12)
    framework.add_argument("--seed", type=int, default=42)
    framework.add_argument("--max-steps", type=int, default=8)
    framework.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    if args.command == "scienceclaw":
        from .scienceclaw_runtime import run_collective
        result = run_collective(project, patient=args.patient, mode=args.mode, budget=args.budget,
            candidate_count=args.candidate_count, seconds=args.seconds, seed=args.seed,
            isolated=args.isolated, max_steps=args.max_steps)
        print(json.dumps(result, indent=2))
        return
    patient = Patient.load(project, args.patient)
    if args.command == "inspect":
        print(json.dumps({"patient": patient.patient_id, "regions": len(patient.labels),
                          "electrode_pairs": len(patient.channels), "default_targets": patient.default_two_targets(),
                          "published_duration_seconds": float(patient.simulator_parameters["simulation_length"]) /
                                                        float(patient.simulator_parameters["sfreq"]),
                          "data_fingerprint": patient.fingerprint}, indent=2))
    elif args.command == "minimum":
        targets = args.targets or patient.default_two_targets()
        if targets[0] == targets[1]:
            parser.error("Two different targets are required")
        results = [run(patient, RunSpec(target, seed=args.seed, duration_seconds=args.seconds)) for target in targets]
        print("REPORT", write_report(project, results, name="minimum"))
    elif args.command == "agents":
        import asyncio
        from .swarm import run_swarm
        results, events = asyncio.run(run_swarm(patient, budget=args.budget, mode=args.mode,
                    duration=args.seconds, seed=args.seed, isolated=args.isolated, baseline=args.baseline))
        name = f"agents_{args.baseline}_{'isolated' if args.isolated else 'shared'}"
        print("REPORT", write_report(project, results, events, name=name))


if __name__ == "__main__":
    main()
