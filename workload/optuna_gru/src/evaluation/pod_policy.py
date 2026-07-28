"""Pod sizing policy for predictive autoscaling experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PodPolicy:
    """Configuration for converting request rate predictions into pod counts."""

    capacity_per_pod: float = 21.1
    safety_margin: float = 0.2
    min_pods: int = 1
    max_pods: int = 8

    @property
    def effective_capacity(self) -> float:
        """Request rate capacity per pod after reserving safety headroom."""
        if self.capacity_per_pod <= 0:
            raise ValueError("capacity_per_pod must be greater than 0")
        if not 0 <= self.safety_margin < 1:
            raise ValueError("safety_margin must be in the range [0, 1)")
        if self.min_pods < 0:
            raise ValueError("min_pods must be greater than or equal to 0")
        if self.max_pods < self.min_pods:
            raise ValueError("max_pods must be greater than or equal to min_pods")
        return self.capacity_per_pod * (1 - self.safety_margin)


DEFAULT_POD_POLICY = PodPolicy()


def required_pods(
    request_rate: float | Iterable[float] | np.ndarray,
    policy: PodPolicy = DEFAULT_POD_POLICY,
) -> int | np.ndarray:
    """Convert request rate values into required pod counts.

    Negative request rates are treated as zero because model predictions can dip
    slightly below zero while traffic demand cannot.
    """
    values = np.asarray(request_rate, dtype=float)
    pods = np.ceil(np.maximum(values, 0) / policy.effective_capacity).astype(int)
    pods = np.clip(pods, policy.min_pods, policy.max_pods)

    if values.ndim == 0:
        return int(pods)
    return pods

