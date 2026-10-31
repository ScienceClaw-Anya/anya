"""Replay a completed run's numeric rankings and audit budgets and prompt gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from neurotarget.data import Patient
from neurotarget.diagnosis import ResponseLocalizer
from neurotarget.evaluation import localization_metrics, stimulation_generator_labels
from neurotarget.response_bank import covariance
from neurotarget.scienceclaw_runtime import Framework


def all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from all_keys(child)
    elif isinstance(value, list):
        for child in value: yield from all_keys(child)


def verify(project, session):
    document = json.loads((session / "benchmark.json").read_text(encoding="utf-8"))
    protocol = document["protocol"]
    bank_path = Path(document["compute"]["bank_path"])
    assert hashlib.sha256(bank_path.read_bytes()).hexdigest() == protocol["bank_sha256"], "Bank manifest changed"
    for name, expected in protocol["implementation_sha256"].items():
        assert hashlib.sha256((project / "neurotarget" / name).read_bytes()).hexdigest() == expected, "Executed inference source changed"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    patient = Patient.load(project, protocol["patient"])
    assert patient.fingerprint == bank["spec"]["data_fingerprint"]
    truth = stimulation_generator_labels(patient)
    control = covariance(next(e for e in bank["entries"] if e["stimulation_scale"] == 0))
    budget = protocol["experiment_budget_per_strategy"]
    counts, calls = {}, {}
    for trial in document["trials"]:
        strategy = trial["strategy"]
        framework = Framework(project, session / strategy)
        for folder in (session / strategy / "artifacts").iterdir():
            if folder.is_dir(): framework.stores[folder.name] = framework.store_class(folder.name)
        counts[strategy] = framework.verify_artifacts()
        assert counts[strategy] == 3 + 4 * budget
        localizer = ResponseLocalizer(patient.projection, patient.labels, control)
        targets = [item["proposal"]["target_id"] for item in trial["experiments"]]
        assert len(targets) == len(set(targets)) == trial["queries_charged"] == budget
        assert len(trial["ranking_traces"]) == len(trial["learning_curve"]) == budget + 1
        for step in range(budget + 1):
            if step:
                target = targets[step-1]
                entry = next(e for e in bank["entries"] if e["target_id"] == target and e["stimulation_scale"] == 1)
                localizer.observe(target, covariance(entry))
            assert np.allclose(localizer.scores, trial["ranking_traces"][step], rtol=1e-10, atol=1e-12)
            scores = localization_metrics(truth, localizer.scores)
            assert all(np.isclose(value, trial["learning_curve"][step][key]) for key, value in scores.items())
        assert trial["final_metrics"] == trial["learning_curve"][-1]
        journal = session / strategy / "llm_journal.jsonl"
        records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
        expected = 2 * budget if protocol["mode"] == "gemini" and strategy != "random" else 0
        assert len(records) == trial["llm_calls"] == expected
        for i, record in enumerate(records):
            prompt = record["prompt"]
            assert record["phase"] == ("proposal" if i % 2 == 0 else "reflection")
            assert len(prompt["shared_experiment_board"]) == i // 2
            assert hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest() == record["prompt_sha256"]
            assert not {"x0", "threshold", "post_region_power", "reference_labels", "average_precision", "auroc"}.intersection(all_keys(prompt))
        calls[strategy] = len(records)
    return {"replayed_rankings_and_scores": True, "native_artifacts": counts, "formal_model_calls": calls,
            "hidden_reference_in_prompt_fields": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.project.resolve(), args.session.resolve()), indent=2))
