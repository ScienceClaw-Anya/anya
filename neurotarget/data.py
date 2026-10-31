from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd


def array_value(value, length=None):
    result = np.fromstring(str(value).replace("[", "").replace("]", ""), sep=" ")
    if not result.size or not np.isfinite(result).all():
        raise ValueError("Invalid numerical array in VEP parameter file")
    if length is not None and result.size not in (1, length):
        raise ValueError(f"Expected 1 or {length} parameters, got {result.size}")
    return result


@dataclass
class Patient:
    project: Path
    patient_id: str
    weights: np.ndarray
    labels: list[str]
    centres: np.ndarray
    gain: np.ndarray
    channels: list[str]
    projection: np.ndarray
    stimulation_profiles: np.ndarray
    contact_pairs: list[tuple[int, int]]
    coordinates: np.ndarray
    model_parameters: dict
    simulator_parameters: dict
    stimulation_parameters: dict
    fingerprint: str
    reference_path: Path

    @classmethod
    def load(cls, project: Path, patient_id="sub-003"):
        project = project.resolve()
        base = project / "data/raw/VirtualEpilepticCohort"
        derivative = base / f"derivatives/tvb/{patient_id}"
        params = derivative / "ses-02/VEPhypothesis/parameters"
        if not params.exists():
            raise FileNotFoundError("VEP data absent. Run scripts/fetch_vep.py first.")
        def row(kind):
            return pd.read_csv(params / f"{patient_id}_{kind}_parameters_run-01.tsv", sep="\t").iloc[0].to_dict()
        with zipfile.ZipFile(derivative / f"struct/{patient_id}_connectome.zip") as archive:
            weights = np.loadtxt(io.BytesIO(archive.read("weights.txt")))
            centres_table = np.loadtxt(io.BytesIO(archive.read("centres.txt")), dtype=str)
        labels = centres_table[:, 0].tolist()
        centres = centres_table[:, 1:].astype(float)
        if weights.shape != (len(labels), len(labels)) or weights.max() <= 0:
            raise ValueError("Invalid connectome")
        np.fill_diagonal(weights, 0)
        weights /= weights.max()
        gain = np.loadtxt(derivative / f"struct/{patient_id}_gain.txt")
        electrodes = pd.read_csv(base / f"{patient_id}/{patient_id}_electrodes.tsv", sep="\t")
        if gain.shape != (len(electrodes), len(labels)):
            raise ValueError("Gain matrix must match electrodes and region ordering")
        names = electrodes["name"].tolist()
        xyz = electrodes[["x", "y", "z"]].to_numpy(float)
        channel_file = base / f"{patient_id}/ses-02/ieeg/{patient_id}_ses-02_task-simulatedstimulation_run-01_channels.tsv"
        desired = pd.read_csv(channel_file, sep="\t")["name"].tolist()
        available = {}
        for i, name in enumerate(names[:-1]):
            first = re.fullmatch(r"(.+?)(\d+)", str(name))
            second = re.fullmatch(r"(.+?)(\d+)", str(names[i + 1]))
            if first and second and first[1] == second[1] and int(second[2]) == int(first[2]) + 1:
                available[f"{first[1]}{first[2]}-{second[2]}"] = (i, i + 1)
        missing = [c for c in desired if c not in available]
        if missing:
            raise ValueError(f"Unmatched bipolar channels: {missing[:5]}")
        pairs = [available[c] for c in desired]
        projection = np.asarray([np.abs(gain[j] - gain[i]) for i, j in pairs])
        # Match author bipolarize_gain_minus(..., is_minus=False), then min-max scaling.
        profiles = np.asarray([(gain[i] + gain[j]) / 2 for i, j in pairs])
        profiles -= profiles.min(axis=1, keepdims=True)
        scale = np.ptp(profiles, axis=1, keepdims=True)
        if np.any(scale <= 0):
            raise ValueError("Degenerate stimulation profile")
        profiles /= scale
        coordinates = np.asarray([(xyz[i] + xyz[j]) / 2 for i, j in pairs])
        ep, sim, stim = row("epileptor"), row("simulator"), row("stimulation")
        for key in ["x0", "threshold", "Iext", "Iext2", "r", "r2", "Ks", "Kf", "Kvf"]:
            ep[key] = array_value(ep[key], len(labels))
        saved_profile = array_value(stim["stimulation_weights"], len(labels))
        if not np.allclose(profiles[desired.index(stim["stim_channels"])], saved_profile, atol=2e-8):
            raise ValueError("Reconstructed stimulation profile disagrees with released VEP profile")
        payload = weights.tobytes() + profiles.tobytes()
        for file in sorted(params.glob("*.tsv")):
            payload += file.read_bytes()
        fingerprint = hashlib.sha256(payload).hexdigest()
        return cls(project, patient_id, weights, labels, centres, gain, desired,
                   projection, profiles, pairs, coordinates, ep, sim, stim, fingerprint,
                   derivative / f"ses-02/VEPhypothesis/{patient_id}_simulated_source_timeseries_run-01.npz")

    def public_catalog(self, candidates=None):
        indices = range(len(self.channels)) if candidates is None else [self.channels.index(c) for c in candidates]
        degree = self.weights.sum(axis=0) + self.weights.sum(axis=1)
        records = []
        for i in indices:
            profile = self.stimulation_profiles[i]
            region = int(profile.argmax())
            records.append({"target_id": self.channels[i], "xyz_mm": self.coordinates[i].tolist(),
                            "peak_region": self.labels[region],
                            "network_score": float(profile @ degree / profile.sum()),
                            "spatial_spread": float(profile.sum()),
                            "profile": profile.tolist()})
        return records

    def default_two_targets(self):
        first = str(self.stimulation_parameters["stim_channels"])
        i = self.channels.index(first)
        distance = np.linalg.norm(self.coordinates - self.coordinates[i], axis=1)
        # Anatomical distance only; do not select the second target using hidden EZ labels.
        second = self.channels[int(distance.argmax())]
        return [first, second]
