"""Judge-only localization metrics. Never include this output in agent prompts."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def stimulation_generator_labels(patient):
    """Sub-003 reference: the elevated-x0, lower-threshold generator subgroup.

    This is the released stimulation model's susceptibility hypothesis. It is
    not a clinical gold standard or a universal x0 cutoff. Reject other parameter
    structures rather than silently inventing a label definition for them.
    """
    x0 = np.broadcast_to(patient.model_parameters["x0"], (len(patient.labels),))
    threshold = np.broadcast_to(patient.model_parameters["threshold"], x0.shape)
    levels = np.unique(x0)
    if patient.patient_id != "sub-003" or len(levels) != 2:
        raise ValueError("This preregistered label rule is only verified for sub-003's two-level stimulation model")
    labels = x0 == levels.max()
    if not np.array_equal(labels, threshold < threshold.max()):
        raise ValueError("Elevated excitability and lower stimulation thresholds disagree")
    return labels.astype(int)


def localization_metrics(labels, scores, k=10):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must be aligned vectors")
    if not np.isin(labels, [0, 1]).all() or not 0 < labels.sum() < len(labels):
        raise ValueError("Evaluation requires both positive and negative reference regions")
    if not 1 <= k <= len(labels):
        raise ValueError("Invalid top-k cutoff")
    positive = int(labels.sum())
    ranks = rankdata(scores, method="average")
    auc = (ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * (len(labels) - positive))
    # Threshold-grouped average precision handles tied predictions without
    # allowing atlas order to create apparent accuracy at the empty baseline.
    tp = fp = 0
    ap = 0.
    for value in np.unique(scores)[::-1]:
        group = scores == value
        added = int(labels[group].sum())
        tp += added
        fp += int(group.sum()) - added
        ap += added / positive * tp / (tp + fp)
    boundary = np.sort(scores)[-k]
    above, tied = scores > boundary, scores == boundary
    expected_hits = float(labels[above].sum() + (k - above.sum()) * labels[tied].mean())
    return {"average_precision": float(ap), "auroc": float(auc),
            f"recall_at_{k}": expected_hits / positive,
            f"precision_at_{k}": expected_hits / k,
            "positive_regions": positive, "evaluated_regions": len(labels),
            "positive_prevalence": positive / len(labels)}
