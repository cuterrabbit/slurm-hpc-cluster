"""Common metrics for traffic forecasting and autoscaling evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.evaluation.pod_policy import DEFAULT_POD_POLICY, PodPolicy, required_pods


def _validate_pair(
    function_name: str,
    actual: Iterable[float] | np.ndarray,
    predicted: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    if actual_arr.size == 0 or predicted_arr.size == 0:
        raise ValueError(f"{function_name}: inputs must not be empty")
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"{function_name}: actual and predicted must have the same shape "
            f"(got {actual_arr.shape} and {predicted_arr.shape})"
        )

    return actual_arr, predicted_arr


def smape(actual: Iterable[float] | np.ndarray, predicted: Iterable[float] | np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error as a ratio.

    A lower value is better. Positions where both actual and predicted are zero
    contribute 0 instead of NaN.
    """
    actual_arr, predicted_arr = _validate_pair("smape", actual, predicted)
    denominator = (np.abs(actual_arr) + np.abs(predicted_arr)) / 2
    diff = np.abs(actual_arr - predicted_arr)
    values = np.divide(diff, denominator, out=np.zeros_like(diff), where=denominator != 0)
    return float(np.mean(values))


def pod_accuracy(
    actual_pods: Iterable[int] | np.ndarray,
    predicted_pods: Iterable[int] | np.ndarray,
) -> float:
    """Return the share of timestamps where predicted pod count matches actual."""
    actual_arr, predicted_arr = _validate_pair("pod_accuracy", actual_pods, predicted_pods)
    return float(np.mean(actual_arr == predicted_arr))


def under_provisioning_rate(
    actual_pods: Iterable[int] | np.ndarray,
    predicted_pods: Iterable[int] | np.ndarray,
) -> float:
    """Return the share of timestamps where predicted pods are below actual pods."""
    actual_arr, predicted_arr = _validate_pair(
        "under_provisioning_rate",
        actual_pods,
        predicted_pods,
    )
    return float(np.mean(predicted_arr < actual_arr))


def over_provisioning_rate(
    actual_pods: Iterable[int] | np.ndarray,
    predicted_pods: Iterable[int] | np.ndarray,
) -> float:
    """Return the share of timestamps where predicted pods exceed actual pods."""
    actual_arr, predicted_arr = _validate_pair(
        "over_provisioning_rate",
        actual_pods,
        predicted_pods,
    )
    return float(np.mean(predicted_arr > actual_arr))


def evaluate_predictions(
    actual: Iterable[float] | np.ndarray,
    predicted: Iterable[float] | np.ndarray,
    policy: PodPolicy = DEFAULT_POD_POLICY,
) -> dict[str, float]:
    """Evaluate forecast values with traffic and pod-level metrics."""
    actual_arr, predicted_arr = _validate_pair("evaluate_predictions", actual, predicted)
    actual_pods = required_pods(actual_arr, policy)
    predicted_pods = required_pods(predicted_arr, policy)

    return {
        "smape": smape(actual_arr, predicted_arr),
        "pod_accuracy": pod_accuracy(actual_pods, predicted_pods),
        "under_provisioning_rate": under_provisioning_rate(actual_pods, predicted_pods),
        "over_provisioning_rate": over_provisioning_rate(actual_pods, predicted_pods),
    }
