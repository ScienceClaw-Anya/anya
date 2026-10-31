import json
import re

import numpy as np
import pytest

from neurotarget.report import write_report
from neurotarget.swarm import demo_proposal


@pytest.mark.parametrize("mode,runtime,has_events,expected", [
    ("demo", "neurotarget", False, "No agents were run for this report."),
    ("demo", "scienceclaw", True, "deterministic specialist policies"),
    ("gemini", "scienceclaw", True, "native Gemini client"),
    ("llm", "neurotarget", True, "configured language model"),
])
def test_reports_use_english_and_preserve_results(tmp_path, mode, runtime, has_events, expected):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    arrays = run_dir / "timeseries.npz"
    np.savez(arrays, time_seconds=np.arange(4.), source_signal=np.zeros((4, 2)),
             accumulation=np.zeros((4, 2)))
    original_arrays = arrays.read_bytes()
    result = {"output_dir": str(run_dir), "target_id": "B2-3", "wall_seconds": 1.2,
              "spec": {"seed": 42}, "model_variant": "test model",
              "summary": {"post_source_variance": 0.125, "threshold_crossing_regions": 2}}
    events = [{"event": "proposal", "agent": "structure", "target_id": "B2-3",
               "reason": "Test the observed response."}] if has_events else []
    target = write_report(tmp_path, [result], events, agent_mode=mode, runtime=runtime)
    report = target.read_text(encoding="utf-8")
    assert '<html lang="en">' in report
    assert "Simulated responses by stimulation site" in report
    assert "Data and reproducibility" in report
    assert expected in report
    assert not re.search(r"[\u3400-\u9fff]", report)
    if has_events:
        assert events[0]["reason"] in report
    saved = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["results"] == [result]
    assert saved["events"] == events
    assert arrays.read_bytes() == original_arrays


def test_all_legacy_demo_roles_explain_choices_in_english():
    catalog = [{"target_id": "B2-3", "profile": [1., 0.], "network_score": 1.,
                "spatial_spread": 1., "peak_region": "region-a"}]
    for role in ["structure", "dynamics", "evidence", "critic", "single", "network", "random"]:
        proposal = demo_proposal(role, catalog, [], set(), 42)
        assert proposal.target_id == "B2-3"
        assert proposal.reason.isascii() and len(proposal.reason.split()) >= 4
