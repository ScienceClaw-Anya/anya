"""Matched-budget diagnostic selection with shared predictions and reflection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np

from .data import Patient
from .diagnosis import ResponseLocalizer
from .evaluation import localization_metrics, stimulation_generator_labels
from .response_bank import covariance, load_bank, read_response
from .scienceclaw_runtime import Framework, Gemini, ROLES, validate_proposal


def safe_catalog(catalog):
    # The legacy spatial_spread field was a profile sum, never millimeters.
    return [{"target_id": c["target_id"], "xyz_mm": c["xyz_mm"], "peak_region": c["peak_region"],
             "network_score": c["network_score"], "profile_mass": c["spatial_spread"],
             "profile_mass_units": "dimensionless sum; not an anatomical distance"} for c in catalog]


def diagnostic_prompt(role, expertise, choices, history, ranking, evidence, labels, phase="proposal", current=None):
    board = [{"experiment": h["experiment"], "role": h["role"],
              "proposal": h["proposal"], "observation": {
                  **{k: v for k, v in h["observation"].items() if k != "top_channels_by_control_ratio"},
                  "top_channels_by_control_ratio": h["observation"]["top_channels_by_control_ratio"][:4]},
              "reflection": h["reflection"]} for h in history[-8:]]
    task = ("Choose ONE unused diagnostic stimulation contact pair. Return a JSON object with target_id, reason, "
            "expected_observation, hypothesis_regions and evidence_ids. Predict what an observable SEEG result would "
            "support or contradict. Read the other roles' hypotheses and reflections. Revise earlier views when warranted.")
    if phase == "reflection":
        task = ("Compare the recorded expectation with this measurement and previous team hypotheses. Return JSON with "
                "assessment (supported, contradicted or inconclusive), explanation, revised_hypothesis and next_test. "
                "State what remains uncertain. Do not invent measured effects or reference accuracy.")
    schema = ({"assessment": "supported|contradicted|inconclusive", "explanation": "English string",
               "revised_hypothesis": ["exact region_vocabulary string"], "next_test": "English string"}
              if phase == "reflection" else
              {"target_id": "exact allowed target_id", "reason": "English string", "expected_observation": "English string",
               "hypothesis_regions": ["exact region_vocabulary string"], "evidence_ids": ["exact supplied evidence ID"]})
    return {"role": role, "expertise": expertise, "phase": phase, "allowed_targets": choices,
            "region_vocabulary": labels, "shared_experiment_board": board,
            "current_response_based_ranking": ranking, "current_experiment": current,
            "literature_context": evidence["items"], "task": task, "required_json_shape": schema,
            "constraints": ["All explanations must be in English. Keep reason/explanation under 80 words and predictions/next_test under 40 words. Copy region names exactly from the vocabulary.",
                "hypothesis_regions and revised_hypothesis must be arrays of region-name strings, never prose strings or objects. Use an empty array if uncertain.",
                "Signal power is in model units squared; ratios and profile_mass are dimensionless. Never use voltage units.",
                "Power changes alone do not establish after-discharges, causal pathways or a clinical EZ.",
                "The ranking is a fixed SEEG-derived heuristic, not generator source activity or calibrated probability.",
                "No reference labels or accuracy scores are available. Do not claim therapeutic efficacy."]}


def validate_prediction(value, choices, evidence_ids, labels):
    result = validate_proposal(value, choices, evidence_ids)
    expected = value.get("expected_observation")
    hypothesis = value.get("hypothesis_regions", [])
    if not isinstance(expected, str) or not expected.strip() or len(expected) > 1000:
        raise ValueError("A bounded pre-measurement expectation is required")
    if not isinstance(hypothesis, list) or len(hypothesis) > 10 or not all(isinstance(r, str) for r in hypothesis) or not set(hypothesis).issubset(labels):
        raise ValueError("Hypothesis contains unknown regions")
    text = result["reason"] + " " + expected
    if re.search(r"(?:[µμ]v|microvolt|\bu[vV]\s*\^?\s*2\b)", text, re.I):
        raise ValueError("Agent invented calibrated voltage units")
    return {**result, "expected_observation": expected, "hypothesis_regions": hypothesis}


def validate_reflection(value, labels):
    if not isinstance(value, dict) or value.get("assessment") not in {"supported", "contradicted", "inconclusive"}:
        raise ValueError("Reflection requires an explicit evidence assessment")
    regions = value.get("revised_hypothesis", [])
    if not isinstance(regions, list) or len(regions) > 10 or not all(isinstance(r, str) for r in regions) or not set(regions).issubset(labels):
        raise ValueError("Reflection contains unknown regions")
    for key in ["explanation", "next_test"]:
        if not isinstance(value.get(key), str) or not value[key].strip() or len(value[key]) > 1500:
            raise ValueError("Reflection text is absent or too long")
    if re.search(r"(?:[µμ]v|microvolt|\bu[vV]\s*\^?\s*2\b)", value["explanation"], re.I):
        raise ValueError("Reflection invented calibrated voltage units")
    return {k: value[k] for k in ["assessment", "explanation", "revised_hypothesis", "next_test"]}


def _demo_proposal(role, choices, original, selected, scores, evidence):
    profiles = np.asarray([next(c["profile"] for c in original if c["target_id"] == x["target_id"]) for x in choices])
    network = np.asarray([c["network_score"] for c in choices]); network /= max(network.max(), 1e-12)
    focus = 1 / np.asarray([c["profile_mass"] for c in choices]); focus /= max(focus.max(), 1e-12)
    novelty = np.ones(len(choices))
    if selected:
        previous = np.asarray([c["profile"] for c in original if c["target_id"] in selected])
        novelty = np.linalg.norm(profiles[:, None] - previous[None], axis=2).min(axis=1)
        novelty /= max(novelty.max(), 1e-12)
    overlap = profiles @ scores
    overlap /= max(overlap.max(), 1e-12)
    utility = {"structure": .8 * network + .2 * novelty, "dynamics": .6 * novelty + .4 * overlap,
               "evidence": .5 * focus + .5 * novelty, "critic": .7 * novelty + .3 * (1-network),
               "single": .34 * network + .33 * focus + .33 * novelty}[role]
    return {"target_id": choices[int(np.argmax(utility))]["target_id"],
            "reason": f"Deterministic {role} diagnostic-selection policy; no language-model call.",
            "expected_observation": "Test whether this entry point produces a different control-normalized SEEG response.",
            "hypothesis_regions": [], "evidence_ids": [evidence["items"][0]["id"]]}


class BudgetedOracle:
    def __init__(self, project, bank_path, allowed, budget, framework=None):
        self.project, self.bank_path = Path(project), Path(bank_path)
        self.allowed, self.budget, self.framework = set(allowed), budget, framework
        self.queried = []

    def query(self, target):
        if target not in self.allowed or target in self.queried or len(self.queried) >= self.budget:
            raise ValueError("Response query violates the candidate, uniqueness or experiment-budget constraint")
        self.queried.append(target)  # Cache hits still cost one experiment.
        if self.framework:
            return self.framework.execute("neurotarget-response", {"bank_path": str(self.bank_path), "target_id": target})
        return read_response(self.project, self.bank_path, target)


def _strategy(project, bank_path, bank, patient, budget, strategy, mode, seed, directory, native=True):
    directory.mkdir(parents=True, exist_ok=True)
    framework = Framework(project, directory) if native else None
    evidence = json.loads((project / "research/evidence.json").read_text(encoding="utf-8"))
    evidence_ids = {e["id"] for e in evidence["items"]}
    profiles = json.loads((project / "config/agents.json").read_text(encoding="utf-8"))
    catalog = safe_catalog(bank["catalog"])
    control_entry = next(e for e in bank["entries"] if e["stimulation_scale"] == 0)
    control = covariance(control_entry)
    localizer = ResponseLocalizer(patient.projection, patient.labels, control, ridge=0.1)
    oracle = BudgetedOracle(project, bank_path, [c["target_id"] for c in catalog], budget, framework)
    gemini = Gemini(project, max_calls=2 * budget, max_tokens=1000) if mode == "gemini" and strategy != "random" else None
    def ask(prompt):
        phase = prompt["phase"]
        status = {"strategy": strategy, "phase": phase, "llm_calls": gemini.calls + 1,
                  "complete": False, "note": "Calls are charged before validation; a validation failure does not become a valid experiment."}
        (directory / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        result = gemini.ask(role, prompt)
        with (directory / "llm_journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": phase, "role": role, "call": gemini.calls,
                "prompt_sha256": hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest(),
                "prompt": prompt, "response": result}, ensure_ascii=True) + "\n")
        return result
    rng = np.random.default_rng(seed)
    history, traces = [], [localizer.scores.tolist()]
    parent_ids = []
    if framework:
        parent_ids = [framework.artifact("anya-data", "neurotarget-catalog", {"results": catalog}, "neurotarget_catalog").artifact_id,
                      framework.artifact("anya-evidence", "neurotarget-evidence", evidence, "research_evidence").artifact_id,
                      framework.artifact("anya-data", "diagnostic-control", {"sha256": control_entry["feature_sha256"],
                          "window_start_seconds": control_entry["window_start_seconds"]}, "control_observation").artifact_id]
    start = time.perf_counter()
    for step in range(budget):
        role = ROLES[step % len(ROLES)] if strategy == "collective" else strategy
        choices = [c for c in catalog if c["target_id"] not in oracle.queried]
        expertise = profiles[role]["expertise"] if role in profiles else "Use all structural, SEEG and evidence tools to select informative diagnostic experiments."
        before = localizer.snapshot()
        prompt = diagnostic_prompt(role, expertise, choices, history, before, evidence, patient.labels)
        if strategy == "random":
            proposal = {"target_id": choices[int(rng.integers(len(choices)))]["target_id"],
                        "reason": "Seeded uniform random choice among unused permitted contact pairs.",
                        "expected_observation": "No response-specific prediction is used by the random policy.",
                        "hypothesis_regions": [], "evidence_ids": []}
        else:
            proposal = (ask(prompt) if gemini else
                        _demo_proposal(role, choices, bank["catalog"], oracle.queried, localizer.scores, evidence))
            proposal = validate_prediction(proposal, choices, evidence_ids, patient.labels)
        parents = list(parent_ids)
        proposed = framework.artifact("anya-" + role, "diagnostic-proposal", proposal, "experiment_proposal", parents) if framework else None
        response = oracle.query(proposal["target_id"])
        measured = framework.artifact("anya-" + role, "neurotarget-response", response, "simulation_data", [proposed.artifact_id]) if framework else None
        observed_covariance = covariance(response)
        after = localizer.observe(proposal["target_id"], observed_covariance)
        inferred = framework.artifact("anya-localizer", "seeg-response-ranking", after, "diagnostic_inference",
                                      [measured.artifact_id, *parent_ids[-1:]]) if framework else None
        current = {"proposal": proposal, "observation": response["observation"], "ranking_before": before, "ranking_after": after}
        if gemini:
            reflection = ask(diagnostic_prompt(role, expertise, choices, history, after, evidence,
                                              patient.labels, "reflection", current))
            reflection = validate_reflection(reflection, patient.labels)
        else:
            reflection = {"assessment": "inconclusive", "explanation": "The SEEG response updates the fixed response-based ranking; reference accuracy is unavailable to this policy.",
                          "revised_hypothesis": [r["region"] for r in after["ranked_regions"][:5]],
                          "next_test": "Compare an unused entry point and reassess the response pattern."}
        reflected = framework.artifact("anya-" + role, "diagnostic-reflection", reflection, "experiment_reflection",
                                       [proposed.artifact_id, measured.artifact_id, inferred.artifact_id]) if framework else None
        item = {"experiment": step + 1, "round": step // 4 + 1 if strategy == "collective" else step + 1,
                "role": role, **current, "reflection": reflection}
        if framework:
            item["artifact_ids"] = {"proposal": proposed.artifact_id, "measurement": measured.artifact_id,
                                    "inference": inferred.artifact_id, "reflection": reflected.artifact_id}
            parent_ids = [*parent_ids[:3], inferred.artifact_id, reflected.artifact_id]
        history.append(item)
        traces.append(localizer.scores.tolist())
        checkpoint = {"strategy": strategy, "mode": mode, "seed": seed, "experiments": history,
                      "ranking_traces": traces, "queries_charged": len(oracle.queried), "complete": False}
        (directory / "trial.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        print(f"DIAGNOSTIC {mode} {strategy} {step+1}/{budget}: {proposal['target_id']} -> {reflection['assessment']}", flush=True)
    checkpoint.update({"complete": True, "wall_seconds": time.perf_counter()-start,
                       "llm_calls": gemini.calls if gemini else 0,
                       "gemini_model": next(iter(gemini.clients.values())).gemini_model if gemini else None,
                       "artifact_integrity_checked": framework.verify_artifacts() if framework else None})
    (directory / "trial.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    (directory / "run_status.json").write_text(json.dumps({"complete": True, "llm_calls": checkpoint["llm_calls"],
        "experiments_completed": len(history)}, indent=2), encoding="utf-8")
    return checkpoint


def run_benchmark(project: Path, bank_path: Path, mode="demo", budget=8, seed=42, random_repeats=20):
    project, bank_path = project.resolve(), bank_path.resolve()
    bank = load_bank(project, bank_path)
    if mode not in {"demo", "gemini"} or not 8 <= budget < len(bank["catalog"]) or budget > 16 or budget % 4 or not 1 <= random_repeats <= 100:
        raise ValueError("Use demo/gemini, 8/12/16 experiments below candidate count, and 1..100 random repetitions")
    patient = Patient.load(project, bank["spec"]["patient"])
    if patient.fingerprint != bank["spec"]["data_fingerprint"]:
        raise ValueError("Patient data differ from frozen response bank")
    session = project / "outputs/diagnostic" / uuid.uuid4().hex[:12]
    session.mkdir(parents=True)
    protocol = json.loads((project / "config/diagnostic-benchmark.json").read_text(encoding="utf-8"))
    protocol.update({"mode": mode, "experiment_budget_per_strategy": budget, "random_repeats": random_repeats,
                     "strategy_seed": seed, "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                     "simulation_seconds": bank["spec"]["seconds"], "simulation_seed": bank["spec"]["seed"],
                     "candidate_count": len(bank["catalog"]), "patient": patient.patient_id,
                     "implementation_sha256": {name: hashlib.sha256((project / "neurotarget" / name).read_bytes()).hexdigest()
                                               for name in ["benchmark.py", "diagnosis.py", "evaluation.py"]}})
    protocol["fairness"]["gemini_calls_per_nonrandom_strategy"] = 2 * budget if mode == "gemini" else 0
    protocol_path = session / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    trials = [_strategy(project, bank_path, bank, patient, budget, strategy, mode, seed, session / strategy)
              for strategy in ["collective", "single", "random"]]
    random_trials = [trials[2]]
    for repeat in range(1, random_repeats):
        random_trials.append(_strategy(project, bank_path, bank, patient, budget, "random", "demo", seed+repeat,
                                       session / f"random-{seed+repeat}", native=False))
    # This is the first point at which the policy-independent judge reads labels.
    truth = stimulation_generator_labels(patient)
    for trial in [*trials, *random_trials[1:]]:
        trial["learning_curve"] = [localization_metrics(truth, s) for s in trial["ranking_traces"]]
        trial["final_metrics"] = trial["learning_curve"][-1]
        trial["mean_learning_curve_average_precision"] = float(np.trapz(
            [r["average_precision"] for r in trial["learning_curve"]], dx=1) / budget)
    random_curves = np.asarray([[r["average_precision"] for r in t["learning_curve"]] for t in random_trials])
    document = {"protocol": protocol, "protocol_sha256": protocol_hash, "session": session.name,
                "trials": trials, "random_reference": {"repetitions": random_repeats,
                    "mean_ap_curve": random_curves.mean(axis=0).tolist(),
                    "p05_ap_curve": np.quantile(random_curves, .05, axis=0).tolist(),
                    "p95_ap_curve": np.quantile(random_curves, .95, axis=0).tolist(),
                    "trials": [{"seed": t["seed"], "targets": [e["proposal"]["target_id"] for e in t["experiments"]],
                                "metrics": t["final_metrics"]} for t in random_trials]},
                "judge_only_reference": {"definition": "Released stimulation-generator elevated-excitability subgroup; not a clinical EZ gold standard",
                    "labels": [label for label, positive in zip(patient.labels, truth) if positive],
                    "positive_count": int(truth.sum()), "region_count": len(truth)},
                "compute": {**bank["execution"], "precomputed_conditions": len(bank["entries"]),
                            "control_conditions_per_policy": 1, "simulation_queries_per_policy": budget,
                            "bank_path": str(bank_path)},
                "clinical_validation": False}
    target = session / "benchmark.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    from .benchmark_report import write_benchmark_report
    report = write_benchmark_report(project, document)
    return {"session": str(session), "report": str(report), "results": str(target),
            "metrics": {t["strategy"]: t["final_metrics"] for t in trials}}
