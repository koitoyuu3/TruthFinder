"""
Shared helpers for hybrid semantic/numeric text alignment.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np

DEFAULT_NUMERIC_MATCH_TOL = 0.02
DEFAULT_SEMANTIC_NUMERIC_ALPHA = 0.5


def extract_numeric_values(text: str) -> list[float]:
    """Extract signed integer/decimal values from free-form text."""
    values: list[float] = []
    for match in re.finditer(r"-?\d+(?:\.\d+)?", str(text)):
        try:
            values.append(float(match.group()))
        except (TypeError, ValueError):
            continue
    return values


def numeric_match_ratio(
    reference: Iterable[float],
    candidate: Iterable[float],
    tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:
    """
    DataAcc-aligned numeric matching rate.

    Each value in ``reference`` is treated as a required slot. A slot matches
    if any value in ``candidate`` is within ``tol`` relative error using the
    reference value as denominator.
    """
    reference_values = [float(value) for value in reference]
    candidate_values = [float(value) for value in candidate]
    if not reference_values:
        return 1.0

    matched = 0
    for ref_value in reference_values:
        denom = max(abs(ref_value), 1e-8)
        if any(abs(ref_value - cand_value) / denom <= tol for cand_value in candidate_values):
            matched += 1
    return matched / len(reference_values)


def symmetric_numeric_match_ratio(
    left: Iterable[float],
    right: Iterable[float],
    tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:
    """
    Symmetric pairwise numeric match rate for affinity matrices.

    Pairwise affinity should not depend on argument order, so average the
    reference-directed match rate in both directions.
    """
    return 0.5 * (
        numeric_match_ratio(left, right, tol=tol)
        + numeric_match_ratio(right, left, tol=tol)
    )


def blend_semantic_numeric_scores(
    sem_score: float,
    num_score: Optional[float],
    alpha: float = DEFAULT_SEMANTIC_NUMERIC_ALPHA,
) -> float:
    """Combine semantic and numeric evidence on a shared [0, 1] scale."""
    sem = float(np.clip(sem_score, 0.0, 1.0))
    if num_score is None:
        return sem

    alpha = float(np.clip(alpha, 0.0, 1.0))
    num = float(np.clip(num_score, 0.0, 1.0))
    return float(alpha * sem + (1.0 - alpha) * num)
