"""Small, reproducible real-recording checks; never a counterfactual simulator."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .brainvision import RangeReader, parse_header
from .evaluation import localization_metrics

REVISIONS = {"ds003029": "62b403d544f42c095035efebafe43cb9479b32bd",
             "ds004370": "db09a7fa2dc97aa33f6458e43241d8b1e5664adf"}
S3 = "https://s3.amazonaws.com/openneuro.org"


def annex_identity(content):
    match = re.search(rb"(SHA256|MD5)E?-s(\d+)--([0-9a-f]+)", content)
    if not match or len(match[3]) != {b"SHA256": 64, b"MD5": 32}[match[1]]:
        raise ValueError("Unsupported annex integrity key")
    return match[1].decode().lower(), int(match[2]), match[3].decode()


def fetch_asset(project, dataset, path):
    if dataset not in REVISIONS or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("Unapproved dataset or unsafe asset path")
    root = Path(project) / "data/real" / dataset
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    source = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset}/{REVISIONS[dataset]}/{quote(path, safe='/')}"
    response = requests.get(source, timeout=30); response.raise_for_status()
    content = response.content
    integrity = "Pinned Git revision"
    if b"annex/objects" in content:
        algorithm, expected_size, expected_hash = annex_identity(content)
        if expected_size > 2 * 1024 * 1024:
            raise ValueError("Only small full metadata assets may be downloaded")
        source = f"{S3}/{dataset}/{quote(path, safe='/')}"
        if target.exists() and hashlib.new(algorithm, target.read_bytes()).hexdigest() == expected_hash:
            content = target.read_bytes()
        else:
            response = requests.get(source, timeout=30); response.raise_for_status(); content = response.content
        if len(content) != expected_size or hashlib.new(algorithm, content).hexdigest() != expected_hash:
            raise ValueError("Public annex asset differs from the pinned version")
        integrity = "Full " + algorithm.upper() + " matches pinned Git-annex key; local SHA-256 also recorded"
    target.write_bytes(content)
    metadata = {"path": path, "url": source, "revision": REVISIONS[dataset], "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(), "integrity": integrity}
    return target, metadata


def table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def contact_annotations(value, available):
    lookup = {n.strip().upper(): n for n in available}
    matched, unmatched = set(), []
    for token in re.split(r"[;,]", str(value)):
        token = token.strip().upper()
        if not token or token in {"N/A", "NA", "NAN", "NONE"}:
            continue
        if token in lookup:
            matched.add(lookup[token]); continue
        match = re.fullmatch(r"([A-Z_']+)(\d+)\s*[-:]\s*(?:\1)?(\d+)", token)
        if not match:
            unmatched.append(token); continue
        first, last = int(match[2]), int(match[3])
        if first > last or last-first > 256:
            raise ValueError("Invalid clinical contact range")
        for number in range(first, last+1):
            name = match[1] + str(number)
            if name in lookup: matched.add(lookup[name])
            else: unmatched.append(name)
    return matched, unmatched


def _recording(project, dataset, stem):
    header_path, header_meta = fetch_asset(project, dataset, stem + "_ieeg.vhdr")
    channels_path, channels_meta = fetch_asset(project, dataset, stem + "_channels.tsv")
    events_path, events_meta = fetch_asset(project, dataset, stem + "_events.tsv")
    header = parse_header(header_path.read_text(encoding="utf-8-sig"))
    channels, events = table(channels_path), table(events_path)
    if channels["name"].tolist() != header.channels:
        raise ValueError("Header and channel sidecar ordering disagree")
    rate = pd.to_numeric(channels["sampling_frequency"]).to_numpy()
    if not np.allclose(rate, header.sfreq, rtol=1e-5):
        raise ValueError("Sampling rates disagree")
    reader = RangeReader(f"{S3}/{dataset}/{stem.rsplit('/',1)[0]}/{header.data_file}", header,
                         Path(project) / "data/real" / dataset / "segments")
    return header, channels, events, reader, [header_meta, channels_meta, events_meta]


def validate_ictal(project):
    dataset = "ds003029"
    stem = "sub-jh101/ses-presurgery/ieeg/sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-01"
    header, channels, events, reader, provenance = _recording(project, dataset, stem)
    clinical_path, metadata = fetch_asset(project, dataset, "sourcedata/clinical_data_summary.xlsx")
    provenance.append(metadata)
    import openpyxl
    rows = list(openpyxl.load_workbook(clinical_path, read_only=True, data_only=True).active.values)
    records = [dict(zip(rows[0], row)) for row in rows[1:]]
    clinical = next(r for r in records if str(r["dataset_id"]).lower() == "jh101")
    soz, unmatched = contact_annotations(clinical["soz_contacts"], header.channels)
    bad, _ = contact_annotations(clinical["bad_contacts"], header.channels)
    if unmatched or not soz:
        raise ValueError("Clinical SOZ contact mapping is incomplete")
    good = np.array([row["status"].lower() == "good" and row["type"].upper() in {"ECOG", "SEEG"}
                     and row["name"] not in bad for row in channels.to_dict("records")])
    onset_rows = events[events["trial_type"].str.contains(r"SZ EVENT.*EEG SZ|electrographic onset|eeg onset", case=False, regex=True)]
    if len(onset_rows) != 1:
        raise ValueError("A unique seizure-onset annotation is required")
    onset = float(onset_rows.iloc[0]["onset"])
    baseline = reader.read_seconds(onset-30, onset-10)
    early = reader.read_seconds(onset, onset+5)
    baseline_power = np.var(baseline, axis=0)
    score = np.log((np.var(early, axis=0) + 1e-15) / (baseline_power + 1e-15))
    names = np.asarray(header.channels)[good]
    truth = np.asarray([name in soz for name in names], dtype=int)
    metrics = localization_metrics(truth, score[good], k=min(10, len(names)))
    metrics["positive_contacts"] = metrics.pop("positive_regions")
    metrics["evaluated_contacts"] = metrics.pop("evaluated_regions")
    ranking = np.argsort(-score[good], kind="stable")[:10]
    return {"dataset": dataset, "subject": "sub-jh101", "task": "Passive early-ictal contact ranking against clinical SOZ annotations",
            "counterfactual_stimulation_evaluated": False, "clinical_ez_gold_standard": False,
            "method": "Fixed log variance ratio: first 5 ictal seconds versus 20-second preictal baseline",
            "onset_seconds": onset, "eligible_contacts": int(good.sum()), "reference_contacts": int(truth.sum()),
            "excluded_contacts": int((~good).sum()), "unmatched_soz_contacts": unmatched, "metrics": metrics,
            "top_contacts": [{"contact": str(names[i]), "score": float(score[good][i]), "clinical_soz": bool(truth[i])} for i in ranking],
            "source": "https://openneuro.org/datasets/ds003029/versions/1.0.5",
            "limitations": ["One recorded seizure; no hyperparameter fitting or held-out-patient claim",
                            "Clinical SOZ contacts are not a complete clinical EZ or proof of surgical benefit",
                            "Uses clinician-supplied seizure timing, but no contact labels in the scoring rule",
                            "Postoperative outcomes and demographic fields are not used"],
            "provenance": provenance, "downloaded_ranges": reader.ranges}


def unique_stimulation_sites(sites, count=2):
    selected, seen = [], set()
    for site in sites:
        pair = frozenset(site.upper().split("-"))
        if len(pair) != 2:
            raise ValueError("A bipolar stimulation site must contain two different contacts")
        if pair not in seen:
            seen.add(pair); selected.append(site)
        if len(selected) == count: return selected
    raise ValueError("Not enough distinct anatomical contact pairs")


def validate_spes(project):
    dataset = "ds004370"
    stem = "sub-PRIOS01/ses-1/ieeg/sub-PRIOS01_ses-1_task-SPESclin_run-021358"
    header, channels, events, reader, provenance = _recording(project, dataset, stem)
    pulses = events[events["trial_type"] == "electrical_stimulation"].copy()
    pulses["onset"] = pd.to_numeric(pulses["onset"])
    pulses = pulses.sort_values("onset")
    artefacts = events[events["trial_type"].str.contains("artefact|artifact", case=False, regex=True)]
    good = np.asarray([r["status"].lower() == "good" and r["type"].upper() in {"ECOG", "SEEG"}
                       for r in channels.to_dict("records")])
    sites = unique_stimulation_sites(pulses["electrical_stimulation_site"].drop_duplicates().tolist())
    responses = []
    pre, post = round(.2*header.sfreq), round(.3*header.sfreq)
    baseline_end, response_start = pre-round(.02*header.sfreq), pre+round(.01*header.sfreq)
    for site in sites:
        epochs = []
        for row in pulses[pulses["electrical_stimulation_site"] == site].to_dict("records"):
            onset = float(row["onset"])
            overlap = any(float(a["onset"]) < onset+.3 and float(a["onset"])+float(a["duration"]) > onset-.2
                          for a in artefacts.to_dict("records"))
            if overlap: continue
            start = round(float(row["sample_start"]))
            epochs.append(reader.read_samples(start-pre, pre+post))
            if len(epochs) == 5: break
        if len(epochs) != 5:
            raise ValueError("Five artefact-free epochs per predefined stimulation site are required")
        epochs = np.asarray(epochs)
        epochs -= epochs[:, :baseline_end].mean(axis=1, keepdims=True)
        evoked = epochs.mean(axis=0)
        noise = epochs[:, :baseline_end].reshape(-1, len(header.channels)).std(axis=0)
        snr = np.sqrt(np.mean(evoked[response_start:]**2, axis=0)) / np.maximum(noise, 1e-12)
        active = {x.upper() for x in site.split("-")}
        eligible = good & np.asarray([n.upper() not in active for n in header.channels])
        selected = np.flatnonzero(eligible)[np.argsort(-snr[eligible], kind="stable")[:5]]
        responses.append({"stimulation_site": site, "pulses": len(epochs), "eligible_response_contacts": int(eligible.sum()),
                          "top_responses": [{"contact": header.channels[i], "evoked_rms_over_baseline_std": float(snr[i])} for i in selected]})
    return {"dataset": dataset, "subject": "sub-PRIOS01", "task": "Recorded SPES response extraction",
            "offline_replay_only": True, "unperformed_stimulation_responses_available": False,
            "localization_accuracy": None, "reason_no_accuracy": "No independent SOZ/EZ contact reference was verified for this PRIOS example",
            "epoch_seconds": [-.2, .3], "baseline_seconds": [-.2, -.02], "response_seconds": [.01, .3],
            "responses": responses, "source": "https://openneuro.org/datasets/ds004370/versions/1.0.2",
            "quality_notes": ["Exclude annotated artefacts, bad/non-iEEG channels, stimulating contacts and first 10 ms after pulse",
                              "Polarity reversal is not counted as a new anatomical site; the first observed direction is retained without pooling polarities",
                              "Header units are omitted; only dimensionless within-channel normalized responses are reported",
                              "Channel cutoff fields appear reversed; no filter was inferred or applied",
                              "This is a small extraction check, not a validated N1 detector or an active-selection accuracy benchmark"],
            "provenance": provenance, "downloaded_ranges": reader.ranges}


def run_real_checks(project):
    result = {"passive_localization": validate_ictal(project), "recorded_stimulation": validate_spes(project),
              "scope": "Separate complementary checks; these results are not pooled with virtual-patient localization scores"}
    directory = Path(project) / "outputs/real_data"; directory.mkdir(parents=True, exist_ok=True)
    target = directory / "validation.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run_real_checks(args.project.resolve()))


if __name__ == "__main__":
    main()
