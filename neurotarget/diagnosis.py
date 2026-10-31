"""Fixed SEEG-only response-ranking heuristic, independent of generator labels.

This is a regularized inverse-response baseline, not Bayesian VEP inversion or
a calibrated probability of epileptogenicity. Hyperparameters are fixed before
evaluating any strategy. The model never receives x0 or accumulation thresholds.
"""
from __future__ import annotations

import numpy as np


class ResponseLocalizer:
    def __init__(self, projection, labels, control_covariance, ridge=0.1):
        projection = np.asarray(projection, dtype=float)
        self.labels = list(labels)
        self.control = np.asarray(control_covariance, dtype=float)
        if projection.ndim != 2 or projection.shape[1] != len(labels):
            raise ValueError("Projection must have one column per region")
        if self.control.shape != (projection.shape[0], projection.shape[0]):
            raise ValueError("Control covariance does not match the sensor space")
        if ridge <= 0 or not np.isfinite(projection).all() or not np.isfinite(self.control).all():
            raise ValueError("Invalid inverse-model inputs")
        norm = np.linalg.norm(projection, axis=0)
        if np.any(norm <= 0):
            raise ValueError("Every evaluated region must have nonzero sensor sensitivity")
        gain = projection / norm
        self.inverse = np.linalg.solve(gain.T @ gain + ridge * np.eye(len(labels)), gain.T)
        self.baseline = self._power(self.control)
        self.floor = max(float(np.median(self.baseline)) * 1e-6, 1e-15)
        self.scores = np.zeros(len(labels))
        self.queries = []

    def _power(self, covariance):
        return np.einsum("ic,cd,id->i", self.inverse, covariance, self.inverse)

    def observe(self, target, covariance):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != self.control.shape or not np.isfinite(covariance).all():
            raise ValueError("Invalid observed sensor covariance")
        if target in self.queries:
            raise ValueError("A contact pair cannot be queried twice within one strategy")
        excess = np.maximum((self._power(covariance) - self.baseline) /
                            np.maximum(self.baseline, self.floor), 0.)
        # A region can be recruited by only a subset of stimulation locations.
        # Maximum aggregation is fixed; it is not selected using hidden scores.
        evidence = np.log1p(excess)
        self.scores = np.maximum(self.scores, evidence)
        self.queries.append(target)
        return self.snapshot()

    def snapshot(self, count=10):
        order = np.argsort(-self.scores, kind="stable")[:count]
        return {"method": "gain-normalized ridge inverse; maximum log relative excess power",
                "interpretation": "Response-based hypothesis ranking, not calibrated EZ probabilities",
                "experiments_seen": len(self.queries),
                "ranked_regions": [{"region": self.labels[i], "score": float(self.scores[i])}
                                   for i in order if self.scores[i] > 0]}


def public_observation(target, covariance, control, channels, seconds, window_start):
    power = np.diag(covariance)
    baseline = np.diag(control)
    ratio = power / np.maximum(baseline, max(float(np.median(baseline)) * 1e-6, 1e-15))
    top = np.argsort(-ratio, kind="stable")[:8]
    return {"target_id": target, "duration_seconds": seconds,
            "window": f"after stimulation ends: {window_start:g} to {seconds:g} seconds",
            "units": "model units squared; ratios are dimensionless; not calibrated voltage",
            "mean_sensor_power": float(power.mean()),
            "top_channels_by_control_ratio": [
                {"channel": channels[i], "power": float(power[i]), "control_ratio": float(ratio[i])}
                for i in top]}
