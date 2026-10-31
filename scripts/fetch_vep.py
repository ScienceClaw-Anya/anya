"""Download only public VEP assets needed for a single-patient experiment.

The dataset's EBRAINS terms must be accepted by the operator before use.
No private clinical recordings or credentials are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests

BUCKET = "https://rgw.cscs.ch/ebrains:d-c1702a5b-bc8f-486e-a0ee-4758a707ff00"
DATASET = "VirtualEpilepticCohort"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_files(prefix: str) -> list[tuple[str, int]]:
    files = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = requests.get(BUCKET, params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall("s:Contents", NS):
            files.append((item.findtext("s:Key", namespaces=NS),
                          int(item.findtext("s:Size", namespaces=NS))))
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            return files
        token = root.findtext("s:NextContinuationToken", namespaces=NS)


def fetch(project: Path, patient: str, reference: bool = True) -> Path:
    if not patient.startswith("sub-") or not patient[4:].isdigit():
        raise ValueError("Use a BIDS patient ID such as sub-003")
    raw = project / "data" / "raw"
    prefixes = [f"{DATASET}/derivatives/tvb/{patient}/", f"{DATASET}/{patient}/"]
    candidates = [entry for prefix in prefixes for entry in list_files(prefix)]
    selected = []
    for key, size in candidates:
        structural = "/struct/" in key and key.endswith(("_connectome.zip", "_gain.txt"))
        parameters = "/ses-02/VEPhypothesis/parameters/" in key
        electrodes = key.endswith("_electrodes.tsv")
        channels = "/ses-02/" in key and key.endswith(("_channels.tsv", ".vhdr"))
        original = (reference and "/ses-02/VEPhypothesis/" in key
                    and key.endswith("_simulated_source_timeseries_run-01.npz"))
        if structural or parameters or electrodes or channels or original:
            selected.append((key, size))
    selected.extend([(f"{DATASET}/dataset_description.json", None),
                     (f"{DATASET}/dataset_coordsystem.json", None)])
    if not any("stimulation_parameters" in key for key, _ in selected):
        raise RuntimeError(f"No stimulation parameters found for {patient}")
    manifest = []
    for key, expected_size in selected:
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe remote file path")
        target = raw / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = BUCKET + "/" + quote(key, safe="/")
        if not target.exists() or (expected_size is not None and target.stat().st_size != expected_size):
            response = requests.get(url, stream=True, timeout=(15, 90))
            response.raise_for_status()
            partial = target.with_suffix(target.suffix + ".part")
            with partial.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError(f"Incomplete download: {key}")
            partial.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append({"path": key, "url": url, "bytes": target.stat().st_size, "sha256": digest})
        print(f"OK {target.stat().st_size:>10} {key}", flush=True)
    manifest_path = project / "data" / f"{patient}_download_manifest.json"
    manifest_path.write_text(json.dumps({
        "dataset_doi": "10.25493/XWG2-S8X", "license": "CC-BY-4.0",
        "paper_doi": "10.1371/journal.pcbi.1012911", "patient": patient,
        "terms_accepted_by_operator": True, "files": manifest
    }, indent=2), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--patient", default="sub-003")
    parser.add_argument("--accept-ebrains-terms", action="store_true")
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()
    if not args.accept_ebrains_terms:
        parser.error("Accept EBRAINS terms first, then pass --accept-ebrains-terms")
    print(fetch(args.project.resolve(), args.patient, not args.no_reference))
