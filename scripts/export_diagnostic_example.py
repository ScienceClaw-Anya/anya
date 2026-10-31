"""Export English portable results and HTML; no raw arrays or private paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from neurotarget.benchmark_report import write_benchmark_report


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items() if k not in {"host", "bank_path", "feature_file", "record_path"}}
    if isinstance(value, list): return [portable(v) for v in value]
    return value


def export(project, results):
    document = portable(json.loads(Path(results).read_text(encoding="utf-8")))
    canonical = json.dumps(document["protocol"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    document["canonical_protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    real = project / "outputs/real_data/validation.json"
    if real.exists():
        document["real_data_validation"] = json.loads(real.read_text(encoding="utf-8"))
        document["real_data_validation_sha256"] = hashlib.sha256(real.read_bytes()).hexdigest()
    failed = project / "outputs/validation/diagnostic_setup_failures.json"
    setup = json.loads(failed.read_text(encoding="utf-8")) if failed.exists() else []
    document["compute"]["formal_comparison_llm_calls"] = sum(t["llm_calls"] for t in document["trials"])
    document["compute"]["separately_recorded_setup_llm_calls"] = sum(t["llm_calls"] for t in setup)
    document["compute"]["separately_recorded_setup_response_queries"] = sum(t.get("measured_queries", 0) for t in setup)
    output = project / "examples"; output.mkdir(exist_ok=True)
    summary = {"protocol": document["protocol"], "canonical_protocol_sha256": document["canonical_protocol_sha256"],
               "reference": document["judge_only_reference"], "compute": document["compute"],
               "policy_results": [{k: t[k] for k in ["strategy", "mode", "seed", "queries_charged", "llm_calls", "gemini_model",
                    "artifact_integrity_checked", "learning_curve", "final_metrics", "mean_learning_curve_average_precision", "experiments"]}
                    for t in document["trials"]], "random_reference": document["random_reference"],
               "real_data_validation": document.get("real_data_validation"), "clinical_validation": False}
    target = output / "diagnostic_benchmark_summary.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_benchmark_report(project, document)
    shutil.copy2(report, output / "diagnostic_benchmark_report.html")
    (output / "diagnostic_protocol.json").write_text(json.dumps(document["protocol"], indent=2), encoding="utf-8")
    return target, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(*export(args.project.resolve(), args.results.resolve()), sep="\n")
