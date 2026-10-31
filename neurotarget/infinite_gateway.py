"""Read-only access-gateway check and local publication draft preparation.

Gateway authentication is distinct from a ScienceClaw agent API identity.
This module never registers an agent, creates an API key, or publishes a post.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv


def check_connection(project: Path):
    load_dotenv(project / ".env", override=False)
    config_path = Path(os.getenv("INFINITE_GATEWAY_CONFIG") or "outputs/private/infinite_gateway.auth.json")
    if not config_path.is_absolute():
        config_path = project / config_path
    if not config_path.exists():
        raise RuntimeError("No private Infinite gateway session configured")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = (os.getenv("INFINITE_API_BASE") or config["api_base"]).rstrip("/")
    origin, authenticated = urlsplit(base), urlsplit(config["api_base"])
    if origin.scheme != "https" or origin.netloc != authenticated.netloc or origin.path != "/api":
        raise ValueError("Gateway cookies may only be sent to their authenticated HTTPS API origin")
    session = requests.Session()
    for name, value in config["cookies"].items():
        session.cookies.set(name, value, domain=origin.hostname, path="/", secure=True)
    try:
        response = session.get(base + "/communities", timeout=25, allow_redirects=False)
    except requests.RequestException:
        raise RuntimeError("Infinite connection failed; no credentials or raw response were logged") from None
    if response.status_code != 200:
        raise RuntimeError("Infinite gateway session is unavailable or expired (HTTP " + str(response.status_code) + ")")
    payload = response.json()
    communities = payload.get("communities", []) if isinstance(payload, dict) else payload
    return {"gateway_authenticated": True, "communities_api_readable": True,
            "community_count": len(communities) if isinstance(communities, list) else None,
            "agent_api_identity_verified": False, "publication_enabled": False,
            "note": "Gateway login is not agent API registration. No posts or identities were created."}


def build_draft(project: Path):
    path = project / "outputs/reports/scienceclaw_gemini_shared.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("execution") != {"runtime": "scienceclaw", "agent_mode": "gemini"}:
        raise ValueError("Draft requires a real Gemini/ScienceClaw report")
    rows = [{"target_id": item["target_id"], "agent": item["agent"],
             "duration_seconds": item["duration_seconds"],
             "mean_seeg_power": item["summary"]["post_sensor_variance"]} for item in report["results"]]
    if not rows:
        raise ValueError("No completed experiments to describe")
    draft = {
        "status": "local_draft_not_published",
        "title": "Anya: ScienceClaw-Gemini VEP diagnostic stimulation integration check",
        "hypothesis": "A bounded specialist collective can propose distinct diagnostic stimulation experiments using shared simulated SEEG observations.",
        "method": "Pinned ScienceClaw SkillRegistry, SkillExecutor, ArtifactStore and Gemini LLMClient; public VEP sub-003; recorded Epileptor coupling-index variant; bounded candidate choices and experimental budget.",
        "findings": {"completed_experiments": len(rows), "results": rows,
                     "interpretation": "Execution/provenance demonstration only. Different simulated responses do not establish treatment efficacy, localization accuracy or multi-agent superiority."},
        "data_sources": ["https://doi.org/10.25493/XWG2-S8X", "https://doi.org/10.1371/journal.pcbi.1012911",
                         "https://github.com/ScienceClaw-Anya/epilepsy-virtual-brain"],
        "limitations": ["One virtual patient", "Short early-response simulations", "Modified coupling declaration; no exact released-run reproduction claim", "No held-out clinical outcomes"],
        "publication_requirements": "An explicitly authorized destination/community and an existing or separately authorized agent API identity are required. This command sends no network write."
    }
    output = project / "outputs/infinite/draft.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": draft["status"], "path": str(output), "experiments": len(rows)}
