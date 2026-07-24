import numpy as np


def compute_det_curve(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
):
    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64)
    spoof_scores = np.asarray(spoof_scores, dtype=np.float64)

    if bonafide_scores.size == 0:
        raise ValueError("Cannot compute EER: no bonafide scores.")
    if spoof_scores.size == 0:
        raise ValueError("Cannot compute EER: no spoof scores.")

    all_scores = np.concatenate((bonafide_scores, spoof_scores))
    labels = np.concatenate(
        (
            np.ones(bonafide_scores.size),
            np.zeros(spoof_scores.size),
        )
    )

    indices = np.argsort(all_scores, kind="mergesort")
    sorted_labels = labels[indices]

    bonafide_cumsum = np.cumsum(sorted_labels)
    spoof_accepted = spoof_scores.size - (
        np.arange(1, all_scores.size + 1) - bonafide_cumsum
    )

    false_rejection_rate = np.concatenate(
        (
            np.atleast_1d(0.0),
            bonafide_cumsum / bonafide_scores.size,
        )
    )
    false_acceptance_rate = np.concatenate(
        (
            np.atleast_1d(1.0),
            spoof_accepted / spoof_scores.size,
        )
    )

    thresholds = np.concatenate(
        (
            np.atleast_1d(all_scores[indices[0]] - 0.001),
            all_scores[indices],
        )
    )

    return false_rejection_rate, false_acceptance_rate, thresholds


def compute_eer(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
):
    false_rejection_rate, false_acceptance_rate, thresholds = compute_det_curve(
        bonafide_scores,
        spoof_scores,
    )

    min_index = np.argmin(np.abs(false_rejection_rate - false_acceptance_rate))
    eer = np.mean(
        (
            false_rejection_rate[min_index],
            false_acceptance_rate[min_index],
        )
    )

    return float(eer), float(thresholds[min_index])
