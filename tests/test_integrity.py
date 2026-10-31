from pathlib import Path

import numpy as np
import pytest

from neurotarget.data import Patient
from neurotarget.simulation import RunSpec, construct_model, run
from neurotarget.swarm import Board, Proposal, choose_candidates, demo_proposal, llm_proposal

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def patient():
    return Patient.load(PROJECT)


def test_patient_alignment_and_saved_stimulation_profile(patient):
    assert patient.weights.shape == (162, 162)
    assert patient.projection.shape == (179, 162)
    assert np.all(np.diag(patient.weights) == 0)
    first, second = patient.default_two_targets()
    assert first != second
    assert not np.allclose(patient.stimulation_profiles[patient.channels.index(first)],
                           patient.stimulation_profiles[patient.channels.index(second)])


def test_public_agent_catalog_has_no_latent_parameters(patient):
    import json
    catalog = choose_candidates(patient)
    encoded = json.dumps(catalog)
    assert all(key not in encoded for key in ['"x0"', '"EZ"', '"threshold"', '"model_parameters"'])
    assert len({c["target_id"] for c in catalog}) == 8
    assert all(c["target_id"] in patient.channels for c in catalog)


def test_corrected_model_matches_independent_rhs(patient):
    model, _ = construct_model(patient)
    model.configure()
    assert model.cvar.tolist() == [0, 3, 6]
    n = len(patient.labels)
    rng = np.random.default_rng(9)
    x = np.array([-1.46, -9.69, 2.97, -1.05, 0., -.19, 0.])[:, None]
    x = x + rng.normal(0, .02, size=(7, n))
    c = rng.normal(0, .01, size=(3, n))
    stimulus = np.zeros((7, n, 1))
    stimulus[0, :, 0] = rng.uniform(-.5, .5, size=n)
    actual = model.dfun(x[:, :, None], c[:, :, None], stimulus=stimulus)[:, :, 0]
    s = stimulus[0, :, 0]
    ep = patient.model_parameters
    expected = np.zeros_like(x)
    a, b, tt = float(model.a[0]), float(model.b[0]), float(model.tt[0])
    h = (x[6] >= ep["threshold"]).astype(float)
    f = np.where(x[0] < 0, -a * x[0] ** 2 + b * x[0], x[3] + .6 * (x[2] - 4) ** 2)
    expected[0] = tt * (x[1] - x[2] + ep["Iext"] + 3 * s + ep["Kvf"] * c[0] + f * x[0])
    expected[1] = tt * (float(model.c[0]) - float(model.d[0]) * x[0] ** 2 - x[1])
    energy = np.where(x[2] < 0, -.1 * x[2] ** 7, 0)
    expected[2] = tt * ep["r"] * (4 * (x[0] - ep["x0"] - h) + energy - x[2] + ep["Ks"] * c[0])
    expected[3] = tt * (-x[4] + x[3] - x[3] ** 3 + ep["Iext2"] + float(model.bb[0]) * x[5] - .3 * (x[2] - 3.5) + ep["Kf"] * c[1])
    expected[4] = tt * ((np.where(x[3] < -.25, 0, float(model.aa[0]) * (x[3] + .25)) - x[4]) / float(model.tau[0]))
    expected[5] = tt * -.01 * (x[5] - .1 * x[0])
    expected[6] = tt * ep["r2"] * (-.3 * x[6] + 20 * np.abs(s) + ep["Kf"] * c[2])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_budget_and_duplicate_claims(tmp_path):
    board = Board(tmp_path / "events.sqlite", 1)
    try:
        proposal = Proposal("structure", "B2-3", "test", [], [])
        assert board.claim(proposal)
        assert not board.claim(proposal)
        assert not board.claim(Proposal("critic", "A1-2", "test", [], []))
        assert len(board.events) == 1
    finally:
        board.close()


def test_agent_feedback_changes_available_choices(patient):
    catalog = choose_candidates(patient)
    first = demo_proposal("structure", catalog, [], set(), 42)
    second = demo_proposal("structure", catalog, [], {first.target_id}, 42)
    assert first.target_id != second.target_id


def test_invalid_target_rejected_before_simulation(patient):
    with pytest.raises(ValueError, match="existing adjacent"):
        run(patient, RunSpec("invented-contact"))


def test_llm_cannot_invent_target(monkeypatch, patient):
    monkeypatch.setenv("NEUROTARGET_LLM_API_KEY", "test-only")
    monkeypatch.setenv("NEUROTARGET_LLM_MODEL", "test-model")
    monkeypatch.setenv("NEUROTARGET_LLM_BASE_URL", "https://example.invalid/v1")
    class Response:
        ok = True
        def json(self):
            return {"choices": [{"message": {"content": '{"target_id":"invented-contact","reason":"x"}'}}]}
    monkeypatch.setattr("neurotarget.swarm.requests.post", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="outside"):
        llm_proposal("critic", choose_candidates(patient), [], set(), 42)


def test_seeded_short_simulation_reproducible(patient):
    spec = RunSpec("B2-3", seed=71, duration_seconds=.05)
    first = run(patient, spec, use_cache=False)
    with np.load(Path(first["output_dir"]) / "timeseries.npz") as data:
        before = data["source_signal"].copy()
    second = run(patient, spec, use_cache=False)
    with np.load(Path(second["output_dir"]) / "timeseries.npz") as data:
        np.testing.assert_array_equal(before, data["source_signal"])


def test_isolated_workers_cannot_read_other_outcomes(tmp_path):
    board = Board(tmp_path / "board.sqlite", 2)
    try:
        board.results = [{"agent": "structure", "target_id": "A1-2"},
                         {"agent": "critic", "target_id": "B2-3"}]
        assert len(board.visible_results("critic", True)) == 1
        assert len(board.visible_results("critic", False)) == 2
    finally:
        board.close()
