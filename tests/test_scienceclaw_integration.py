import json
from pathlib import Path

import pytest

from neurotarget.scienceclaw_runtime import Framework, Gemini, _prompt, run_collective, validate_proposal

PROJECT = Path(__file__).resolve().parents[1]


def test_real_scienceclaw_discovers_and_executes_evidence(tmp_path):
    framework = Framework(PROJECT, tmp_path / "native")
    from core.skill_registry import SkillRegistry
    from core.skill_executor import SkillExecutor
    assert isinstance(framework.registry, SkillRegistry)
    assert isinstance(framework.executor, SkillExecutor)
    result = framework.execute("neurotarget-evidence", {})
    assert result["source_commit"]
    assert len(result["items"]) >= 4
    assert all(item["source_url"].startswith(result["source_repository"]) for item in result["items"])
    assert framework.registry.cache_file.is_relative_to(tmp_path)


def test_native_artifact_hashes_and_multi_parent_lineage(tmp_path):
    framework = Framework(PROJECT, tmp_path / "native")
    a = framework.artifact("anya-structure", "neurotarget-catalog", {"results": []}, "neurotarget_catalog")
    b = framework.artifact("anya-evidence", "neurotarget-evidence", {"items": []}, "research_evidence")
    c = framework.artifact("anya-dynamics", "neurotarget-proposal", {"target_id": "B2-3"},
                           "experiment_proposal", [a.artifact_id, b.artifact_id])
    from artifacts.artifact import ArtifactStore
    assert isinstance(framework.stores["anya-dynamics"], ArtifactStore)
    assert framework.verify_artifacts() == 3
    assert framework.stores["anya-dynamics"].get(c.artifact_id).parent_artifact_ids == [a.artifact_id, b.artifact_id]
    assert framework.stores["anya-dynamics"].store_path.is_relative_to(tmp_path)


def test_artifact_tampering_is_detected(tmp_path):
    framework = Framework(PROJECT, tmp_path / "native")
    framework.artifact("anya-critic", "neurotarget-evidence", {"value": 1}, "research_evidence")
    path = framework.stores["anya-critic"].store_path
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["payload"]["value"] = 2
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity"):
        framework.verify_artifacts()


def test_agent_cannot_invent_target_or_evidence():
    choices = [{"target_id": "B2-3"}]
    with pytest.raises(ValueError, match="outside"):
        validate_proposal({"target_id": "made-up", "reason": "test"}, choices, {"paper-a"})
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_proposal({"target_id": "B2-3", "reason": "test", "evidence_ids": ["fake"]}, choices, {"paper-a"})


def test_prompt_observations_withhold_model_hidden_state():
    evidence = {"items": [{"id": "test", "statement": "use observed SEEG"}]}
    choices = [{"target_id": "B2-3", "profile": [1, 2]}]
    observations = [{"observation": {"target_id": "B2-3", "duration_seconds": 12,
        "sensor_power": [1., 2.], "sensor_names": ["A1-2", "B2-3"], "mean_sensor_power": 1.5,
        "window": "post-stimulus", "x0": [99], "post_region_power": [88], "threshold": [77]},
        "artifact_id": "measurement-1"}]
    prompt = _prompt("dynamics", {"expertise": "SEEG"}, choices, observations, evidence)
    text = json.dumps(prompt)
    assert '"profile"' not in text and '"post_region_power"' not in text
    assert '"x0"' not in text and '"threshold"' not in text
    assert prompt["measured_seeg"][0]["top_seeg_channels"][0]["channel"] == "B2-3"


def test_invalid_experiment_budget_fails_before_any_tool():
    with pytest.raises(ValueError, match="max_steps"):
        run_collective(PROJECT, budget=5, max_steps=4)
    with pytest.raises(ValueError, match="exceed"):
        run_collective(PROJECT, budget=4, candidate_count=4)


def test_gemini_uses_native_client_with_bounded_requests(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    gemini = Gemini(PROJECT, max_calls=1)
    client = gemini.client("critic")
    assert client.backend == "gemini" and client.gemini_max_retries == 0
    assert client.gemini_client.max_retries == 0
    monkeypatch.setattr(client, "call", lambda *a, **k: '{"target_id":"B2-3","reason":"control","evidence_ids":[]}')
    assert gemini.ask("critic", {"task": "test"})["target_id"] == "B2-3"
    with pytest.raises(RuntimeError, match="budget"):
        gemini.ask("critic", {"task": "test"})


def test_gemini_rejects_non_google_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.invalid/v1/")
    gemini = Gemini(PROJECT, max_calls=1)
    with pytest.raises(ValueError, match="Google"):
        gemini.client("structure")


def test_only_declared_scientific_tools_are_allowed(tmp_path):
    framework = Framework(PROJECT, tmp_path / "native")
    with pytest.raises(ValueError, match="permitted"):
        framework.execute("arbitrary-shell", {})
