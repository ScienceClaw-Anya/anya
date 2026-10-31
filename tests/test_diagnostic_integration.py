import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from neurotarget import benchmark
from neurotarget.response_bank import read_response

PROJECT = Path(__file__).resolve().parents[1]


def fixture_bank(tmp_path):
    for name in ["config/agents.json", "config/diagnostic-benchmark.json", "research/evidence.json",
                 "neurotarget/benchmark.py", "neurotarget/diagnosis.py", "neurotarget/evaluation.py"]:
        (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT/name, tmp_path/name)
    directory = tmp_path/"outputs/response_banks/test"
    directory.mkdir(parents=True)
    catalog, entries = [], []
    for i in range(13):
        cov = np.eye(12)
        if i < 12: cov[i, i] = 4+i
        path = directory/f"cov-{i}.npz"; np.savez(path, covariance=cov)
        target = f"site-{i}" if i < 12 else "site-0"
        entries.append({"target_id": target, "stimulation_scale": 1 if i < 12 else 0,
                        "feature_file": str(path), "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "window_start_seconds": 7})
        if i < 12:
            catalog.append({"target_id": target, "xyz_mm": [i, 0, 0], "peak_region": f"R{i}",
                            "network_score": i+1., "spatial_spread": i+1., "profile": np.eye(12)[i].tolist()})
    bank = {"complete": True, "catalog": catalog, "entries": entries, "channels": [f"C{i}" for i in range(12)],
            "spec": {"patient": "sub-003", "data_fingerprint": "fixture", "seconds": 55, "seed": 42},
            "execution": {"location": "test fixture", "workers": 1}}
    path = directory/"bank.json"; path.write_text(json.dumps(bank))
    patient = SimpleNamespace(patient_id="sub-003", labels=[f"R{i}" for i in range(12)], fingerprint="fixture",
                              projection=np.eye(12), model_parameters={"x0": np.array([-2.07]*3+[-2.25]*9),
                                                                      "threshold": np.array([8.]*3+[20.]*9)})
    return path, patient


class FakeFramework:
    def __init__(self, project, directory):
        self.project, self.items = project, []
    def artifact(self, *args, **kwargs):
        item = SimpleNamespace(artifact_id=str(len(self.items)))
        self.items.append(item)
        return item
    def execute(self, name, parameters):
        assert name == "neurotarget-response"
        return read_response(self.project, parameters["bank_path"], parameters["target_id"])
    def verify_artifacts(self): return len(self.items)


class FakeGemini:
    prompts = []
    def __init__(self, project, max_calls, max_tokens):
        self.calls, self.cap = 0, max_calls
        self.clients = {"fake": SimpleNamespace(gemini_model="offline-test")}
    def ask(self, role, prompt):
        assert self.calls < self.cap
        self.calls += 1; self.prompts.append(prompt)
        if prompt["phase"] == "reflection":
            return {"assessment": "inconclusive", "explanation": "This updates a response hypothesis, not verified localization.",
                    "revised_hypothesis": [], "next_test": "Choose another entry point."}
        return {"target_id": prompt["allowed_targets"][0]["target_id"], "reason": "Test a different entry point.",
                "expected_observation": "A different control-normalized response.", "hypothesis_regions": [], "evidence_ids": []}


def test_full_matched_budget_loop_uses_two_rounds_feedback_and_equal_llm_caps(tmp_path, monkeypatch):
    path, patient = fixture_bank(tmp_path)
    monkeypatch.setattr(benchmark.Patient, "load", lambda *args: patient)
    monkeypatch.setattr(benchmark, "Framework", FakeFramework)
    monkeypatch.setattr(benchmark, "Gemini", FakeGemini)
    FakeGemini.prompts = []
    result = benchmark.run_benchmark(tmp_path, path, mode="gemini", budget=8, random_repeats=2)
    report = json.loads(Path(result["results"]).read_text())
    collective, single, random = report["trials"]
    assert [t["queries_charged"] for t in report["trials"]] == [8, 8, 8]
    assert [t["llm_calls"] for t in report["trials"]] == [16, 16, 0]
    assert [e["role"] for e in collective["experiments"]] == benchmark.ROLES*2
    assert [e["round"] for e in collective["experiments"]] == [1]*4+[2]*4
    assert len(collective["learning_curve"]) == 9
    assert len({e["proposal"]["target_id"] for e in random["experiments"]}) == 8
    assert collective["artifact_integrity_checked"] == 35
    second_round = [p for p in FakeGemini.prompts if p["role"] == "structure" and p["phase"] == "proposal"][1]
    assert second_round["shared_experiment_board"][3]["role"] == "critic"
    assert "reflection" in second_round["shared_experiment_board"][3]
    text = json.dumps(FakeGemini.prompts)
    assert all('"'+key+'"' not in text for key in ["x0", "threshold", "average_precision", "reference_labels"])
    html = Path(result["report"]).read_text()
    assert '<html lang="en">' in html and "Mean AP across budget" in html
    assert report["random_reference"]["repetitions"] == 2
