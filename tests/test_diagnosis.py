import json
from types import SimpleNamespace

import numpy as np
import pytest

from neurotarget.diagnosis import ResponseLocalizer
from neurotarget.evaluation import localization_metrics, stimulation_generator_labels
from neurotarget.benchmark import BudgetedOracle, diagnostic_prompt, validate_prediction, validate_reflection
from neurotarget.response_bank import covariance, load_bank


def test_ranking_recovers_a_known_sensor_source_change_without_labels():
    model = ResponseLocalizer(np.eye(12), [f"R{i}" for i in range(12)], np.eye(12))
    observed = np.eye(12); observed[4, 4] = 10
    snapshot = model.observe("site-a", observed)
    assert snapshot["ranked_regions"][0]["region"] == "R4"
    assert model.scores[4] > 0 and np.count_nonzero(model.scores) == 1
    model.observe("site-b", np.eye(12))
    assert model.scores[4] > 0
    with pytest.raises(ValueError, match="twice"):
        model.observe("site-a", observed)


def test_no_response_does_not_manufacture_localization_evidence():
    model = ResponseLocalizer(np.eye(12), [f"R{i}" for i in range(12)], np.eye(12))
    assert model.observe("site-a", np.eye(12))["ranked_regions"] == []
    assert np.all(model.scores == 0)


def test_metrics_handle_perfect_wrong_and_tied_rankings():
    truth = [1, 0, 0, 1]
    perfect = localization_metrics(truth, [4, 2, 1, 3], k=2)
    assert perfect["average_precision"] == perfect["auroc"] == perfect["recall_at_2"] == 1
    assert localization_metrics(truth, [1, 4, 3, 2], k=2)["auroc"] == 0
    tied = localization_metrics(truth, [1, 1, 1, 1], k=2)
    assert tied["average_precision"] == tied["auroc"] == tied["recall_at_2"] == .5
    permutation = [2, 0, 3, 1]
    assert localization_metrics(np.array(truth)[permutation], [1]*4, k=2) == tied
    with pytest.raises(ValueError): localization_metrics([0, 0], [1, 2], k=1)


def test_reference_is_generator_specific_and_cross_checked():
    patient = SimpleNamespace(patient_id="sub-003", labels=["A", "B", "C"],
                              model_parameters={"x0": np.array([-2.25, -2.07, -2.25]), "threshold": np.array([20., 8., 20.])})
    assert stimulation_generator_labels(patient).tolist() == [0, 1, 0]
    patient.model_parameters["threshold"] = np.array([8., 20., 20.])
    with pytest.raises(ValueError, match="disagree"): stimulation_generator_labels(patient)


def test_feedback_prompt_shares_predictions_and_reflections_without_reference():
    event = {"experiment": 1, "role": "structure", "proposal": {"target_id": "A", "reason": "Test a hub.",
             "expected_observation": "A different response."},
             "observation": {"top_channels_by_control_ratio": [{"channel": "C1", "control_ratio": 2.}], "mean_sensor_power": 1.},
             "reflection": {"assessment": "inconclusive", "explanation": "Need a control."}}
    prompt = diagnostic_prompt("critic", "Challenge hypotheses", [], [event], {"ranked_regions": []}, {"items": []}, ["R1"])
    assert prompt["shared_experiment_board"][0]["proposal"]["expected_observation"] == "A different response."
    assert prompt["shared_experiment_board"][0]["reflection"]["explanation"] == "Need a control."
    serialized = json.dumps(prompt)
    assert all('"'+name+'"' not in serialized for name in ["x0", "threshold", "true_ez", "average_precision", "reference_labels"])


def test_queries_charge_cache_hits_and_forbid_extra_experiments(monkeypatch, tmp_path):
    monkeypatch.setattr("neurotarget.benchmark.read_response", lambda *args: {"precomputed": True})
    oracle = BudgetedOracle(tmp_path, tmp_path/"bank.json", ["A", "B", "C"], 2)
    assert oracle.query("A")["precomputed"]
    with pytest.raises(ValueError): oracle.query("A")
    oracle.query("B")
    with pytest.raises(ValueError): oracle.query("C")
    assert oracle.queried == ["A", "B"]


def test_prediction_and_reflection_reject_inventions():
    value = {"target_id": "A", "reason": "A valid test", "expected_observation": "An altered response", "hypothesis_regions": ["R1"], "evidence_ids": []}
    assert validate_prediction(value, [{"target_id": "A"}], set(), ["R1"])["hypothesis_regions"] == ["R1"]
    with pytest.raises(ValueError): validate_prediction({**value, "reason": "Measured 10 uV^2"}, [{"target_id": "A"}], set(), ["R1"])
    with pytest.raises(ValueError): validate_prediction({**value, "hypothesis_regions": ["fake"]}, [{"target_id": "A"}], set(), ["R1"])
    with pytest.raises(ValueError): validate_reflection({"assessment": "definitely proven"}, ["R1"])


def test_response_bank_rejects_escaped_incomplete_and_corrupt_data(tmp_path):
    import hashlib
    directory = tmp_path/"outputs/response_banks/test"
    directory.mkdir(parents=True)
    manifest = directory/"bank.json"; manifest.write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="incomplete"): load_bank(tmp_path, manifest)
    with pytest.raises(ValueError, match="inside"): load_bank(tmp_path, tmp_path/"outside.json")
    path = directory/"cov.npz"; np.savez(path, covariance=np.eye(2))
    entry = {"feature_file": str(path), "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert np.array_equal(covariance(entry), np.eye(2))
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"): covariance(entry)
