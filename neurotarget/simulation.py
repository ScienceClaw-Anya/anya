from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np

from .data import Patient, array_value

MODEL_VARIANT = "VEP-EpileptorStim2Populations-cvar-0-3-6-v1"


@dataclass(frozen=True)
class RunSpec:
    target_id: str
    seed: int = 42
    duration_seconds: float | None = None
    coupling_scale: float = 1.0
    noise_scale: float = 1.0


def load_author_modules(project: Path):
    vendor = project / "vendor/virtual_epilepsy_patient_cohort/src/utils_functions"
    if not (vendor / "model_2populations.py").exists():
        raise FileNotFoundError("Clone the VEP author repository into vendor/virtual_epilepsy_patient_cohort")
    loaded = []
    for name, path in [("neurotarget_vep_model", vendor / "model_2populations.py"),
                       ("neurotarget_vep_integrator", vendor / "integrator.py")]:
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def construct_model(patient: Patient):
    mod, integ = load_author_modules(patient.project)
    # Author gufunc explicitly reads c_pop[0], c_pop[1], c_pop[2]. The released
    # class declares cvar=[0], causing out-of-bounds access in compiled code.
    # Declare all three intended populations explicitly. This is a documented
    # coupling correction, NOT a claim of exact published-run reproduction.
    class CorrectedEpileptorStim(mod.EpileptorStim2Populations):
        cvar = np.array([0, 3, 6], dtype=np.int32)
        stvar = np.array([0], dtype=np.int32)

    model = CorrectedEpileptorStim(variables_of_interest=["x1", "x2", "m", "z"])
    for name in ["x0", "threshold", "Iext", "Iext2", "r", "r2", "Ks", "Kf", "Kvf"]:
        setattr(model, name, np.array(patient.model_parameters[name], copy=True))
    model.Istim = np.zeros(len(patient.labels))
    model.n_stim = np.zeros(len(patient.labels))
    return model, integ


def run(patient: Patient, spec: RunSpec, use_cache=True):
    if spec.target_id not in patient.channels:
        raise ValueError("Target must be an existing adjacent SEEG contact pair")
    if not (0 < spec.coupling_scale <= 2 and 0 <= spec.noise_scale <= 3):
        raise ValueError("Unsupported perturbation")
    sim_p, stim_p = patient.simulator_parameters, patient.stimulation_parameters
    sfreq = float(sim_p["sfreq"])
    duration = float(sim_p["simulation_length"]) / sfreq if spec.duration_seconds is None else spec.duration_seconds
    if duration <= 0 or duration > 120:
        raise ValueError("Duration must be in (0, 120] seconds")
    key_payload = {"data": patient.fingerprint, "model": MODEL_VARIANT, "spec": asdict(spec)}
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()[:20]
    output = patient.project / "outputs" / patient.patient_id / key
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if use_cache and result_path.exists() and (output / "timeseries.npz").exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["cache_hit"] = True
        return result

    from tvb.datatypes import connectivity, equations, patterns
    from tvb.simulator import coupling, monitors, noise, simulator

    model, integrators = construct_model(patient)
    con = connectivity.Connectivity(weights=patient.weights.copy(),
                                    tract_lengths=np.zeros_like(patient.weights),
                                    centres=patient.centres.copy(),
                                    region_labels=np.asarray(patient.labels))
    con.configure()
    dt = float(sim_p["dt"])
    period = float(sim_p["period"])
    duration_units = duration * sfreq
    times = np.arange(0, duration_units, dt)
    phase = np.mod(times - float(stim_p["onset"]), float(stim_p["T"]))
    pulse_width = float(stim_p["pulse_width"])
    waveform = np.zeros_like(times)
    active = (times >= float(stim_p["onset"])) & (times < float(stim_p["stimulation_length"]))
    waveform[active & (phase < pulse_width)] = float(stim_p["I"])
    waveform[active & (phase >= pulse_width) & (phase < 2 * pulse_width)] = -float(stim_p["I"])

    class SampledWaveform(equations.DiscreteEquation):
        equation = equations.Final(default="emp")

    temporal = SampledWaveform()
    temporal.parameters["emp"] = waveform
    target_index = patient.channels.index(spec.target_id)
    stimulus = patterns.StimuliRegion(temporal=temporal, connectivity=con,
                                      weight=patient.stimulation_profiles[target_index].copy())
    stimulus.configure_space()
    stimulus.configure_time(np.arange(len(waveform)))
    nsig = array_value(sim_p["noise_coeffs"]) * spec.noise_scale
    stochastic = noise.Additive(nsig=nsig, random_stream=np.random.RandomState(spec.seed))
    stochastic.ntau = 1.0
    integrator = integrators.HeunStochasticAdapted(dt=dt, noise=stochastic)
    initial = np.array([-1.46242601, -9.69344913, 2.97, -1.05214059,
                        -4.95543740e-20, -0.198742113, 0.0])
    initial = np.repeat(initial, len(patient.labels)).reshape(1, 7, len(patient.labels), 1)
    engine = simulator.Simulator(model=model, initial_conditions=initial,
        connectivity=con, stimulus=stimulus,
        coupling=coupling.Difference(a=np.array([float(sim_p["coupling_factor"]) * spec.coupling_scale])),
        integrator=integrator, conduction_speed=np.inf,
        monitors=[monitors.TemporalAverage(period=period)])
    engine.configure()
    started = time.perf_counter()
    print(f"SIMULATING {patient.patient_id} {spec.target_id}: {duration:.1f}s, seed={spec.seed}", flush=True)
    monitor_time, monitored = engine.run(simulation_length=duration_units)[0]
    wall = time.perf_counter() - started
    states = monitored[:, :, :, 0]
    if not np.isfinite(states).all():
        raise FloatingPointError("Non-finite VEP state: simulation rejected")
    source = states[:, 0, :] - states[:, 1, :]
    sensor = source @ patient.projection.T
    seconds = monitor_time / sfreq
    np.savez_compressed(output / "timeseries.npz", time_seconds=seconds,
                        source_signal=source, sensor_signal=sensor,
                        accumulation=states[:, 2, :], z=states[:, 3, :])
    summary = summarize(patient, seconds, source, sensor, states[:, 2, :])
    result = {"patient": patient.patient_id, "target_id": spec.target_id,
              "spec": asdict(spec), "model_variant": MODEL_VARIANT,
              "clinical_validation": False, "published_run_reproduced": False,
              "model_time_units_per_second": sfreq, "duration_seconds": duration,
              "data_fingerprint": patient.fingerprint, "wall_seconds": wall,
              "cache_key": key, "cache_hit": False, "output_dir": str(output),
              "summary": summary,
              "limitations": ["Coupling indices corrected from [0] to [0,3,6]",
                               "Seed and full initial state of released run not reproducible from metadata",
                               "Phenomenological diagnostic stimulation; no seizure-suppression claim"]}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"FINISHED {spec.target_id}: {wall:.2f}s wall time", flush=True)
    return result


def summarize(patient, seconds, source, sensor, accumulation):
    stop = float(patient.stimulation_parameters["stimulation_length"]) / float(patient.simulator_parameters["sfreq"])
    post = seconds >= stop
    if not post.any():
        post = np.ones(len(seconds), dtype=bool)
    centred = source[post] - source[post].mean(axis=0)
    region_power = np.mean(centred ** 2, axis=0)
    threshold = np.broadcast_to(patient.model_parameters["threshold"], (len(patient.labels),))
    crossings = accumulation > threshold[None, :]
    dt_seconds = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.0
    return {"post_source_variance": float(region_power.mean()),
            "post_sensor_variance": float(np.var(sensor[post], axis=0).mean()),
            "threshold_crossing_regions": int(crossings.any(axis=0).sum()),
            "threshold_crossing_region_seconds": float(crossings.sum() * dt_seconds),
            "post_region_power": region_power.tolist(),
            "top_response_regions": [patient.labels[i] for i in np.argsort(region_power)[-5:][::-1]],
            "metric_note": "Threshold crossing describes the model accumulation variable; it is not a validated seizure annotation."}
