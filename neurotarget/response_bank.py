"""Read exactly one requested, hash-checked SEEG response from a frozen bank."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .diagnosis import public_observation


def load_bank(project, path):
    project, path = Path(project).resolve(), Path(path).resolve()
    if not path.is_relative_to(project / "outputs/response_banks"):
        raise ValueError("Response bank must be inside this project's response_banks directory")
    bank = json.loads(path.read_text(encoding="utf-8"))
    if not bank.get("complete"):
        raise ValueError("Response bank is incomplete")
    return bank


def covariance(entry):
    path = Path(entry["feature_file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["feature_sha256"]:
        raise ValueError("Response covariance hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        result = data["covariance"].copy()
    if not np.isfinite(result).all():
        raise ValueError("Response covariance contains non-finite values")
    return result


def read_response(project, path, target):
    bank = load_bank(project, path)
    allowed = {r["target_id"] for r in bank["catalog"]}
    if target not in allowed:
        raise ValueError("Requested target is outside the frozen candidate pool")
    item = next(r for r in bank["entries"] if r["target_id"] == target and r["stimulation_scale"] == 1)
    control = next(r for r in bank["entries"] if r["stimulation_scale"] == 0)
    observed = covariance(item)
    return {"target_id": target, "feature_file": item["feature_file"],
            "feature_sha256": item["feature_sha256"], "precomputed": True,
            "observation": public_observation(target, observed, covariance(control), bank["channels"],
                                               bank["spec"]["seconds"], item["window_start_seconds"])}
