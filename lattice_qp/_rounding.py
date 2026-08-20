"""Resolved rounding policies and stateful cone projection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._problem import integral_vector
from .policies import ConeRoundingPolicy, RoundingPolicy


@dataclass(frozen=True)
class ResolvedRounding:
    selection: str
    projection: str
    threshold: float
    label: str
    incidence_indptr: np.ndarray
    incidence_indices: np.ndarray
    incidence_coefficients: np.ndarray
    base_cones: np.ndarray
    minimum_cones: np.ndarray

    @property
    def has_cones(self) -> bool:
        return len(self.base_cones) != 0


def resolve_rounding(
    rounding: str | RoundingPolicy | ConeRoundingPolicy,
    multiple_rounding_threshold: float,
    integer_count: int,
) -> ResolvedRounding:
    aliases: dict[str, tuple[str, str]] = {
        "multiple": ("multiple", "nearest"),
        "greedy": ("all", "nearest"),
        "sequential": ("sequential", "nearest"),
        "all": ("all", "nearest"),
        "all-at-once": ("all", "nearest"),
        "sequential-nearest": ("sequential", "nearest"),
        "multiple-nearest": ("multiple", "nearest"),
        "all-nearest": ("all", "nearest"),
        "all-at-once-nearest": ("all", "nearest"),
    }
    if isinstance(rounding, str):
        normalized = rounding.lower().replace("_", "-")
        if normalized in aliases:
            selection, projection = aliases[normalized]
        else:
            selection = ""
            projection = ""
            for selection_alias, canonical_selection in (
                ("all-at-once", "all"),
                ("sequential", "sequential"),
                ("multiple", "multiple"),
                ("all", "all"),
            ):
                prefix = selection_alias + "-"
                if normalized.startswith(prefix):
                    selection = canonical_selection
                    projection = normalized[len(prefix) :].replace(
                        "away-from-zero", "away_from_zero"
                    )
                    break
            if not selection:
                raise ValueError(
                    "rounding must be 'multiple', 'greedy', a supported "
                    "selection-projection alias, or a rounding policy"
                )
        threshold = multiple_rounding_threshold
        label = rounding
        cone = None
    elif isinstance(rounding, (RoundingPolicy, ConeRoundingPolicy)):
        selection = str(rounding.selection).lower().replace("_", "-")
        selection = {"all-at-once": "all"}.get(selection, selection)
        projection = str(rounding.projection).lower().replace("-", "_")
        threshold = (
            multiple_rounding_threshold
            if rounding.threshold is None
            else float(rounding.threshold)
        )
        label = f"{selection}-{projection.replace('_', '-')}"
        cone = rounding if isinstance(rounding, ConeRoundingPolicy) else None
    else:
        raise TypeError("rounding must be a string or rounding policy")

    if selection not in {"sequential", "multiple", "all"}:
        raise ValueError("rounding selection must be 'sequential', 'multiple', or 'all'")
    if projection not in {"nearest", "floor", "ceil", "away_from_zero"}:
        raise ValueError(
            "rounding projection must be 'nearest', 'floor', 'ceil', or "
            "'away_from_zero'"
        )
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("rounding threshold must be finite and positive")

    if cone is None:
        return ResolvedRounding(
            selection,
            projection,
            threshold,
            label,
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    indptr = integral_vector(cone.incidence_indptr, "incidence_indptr", int32=True)
    indices = integral_vector(cone.incidence_indices, "incidence_indices", int32=True)
    coefficients = integral_vector(cone.incidence_coefficients, "incidence_coefficients")
    base = integral_vector(cone.base_cones, "base_cones")
    minimum = integral_vector(cone.minimum_cones, "minimum_cones")
    if len(base) != len(minimum):
        raise ValueError("base_cones and minimum_cones must have matching lengths")
    if len(indptr) != len(base) + 1:
        raise ValueError("incidence_indptr must have len(base_cones) + 1 entries")
    if not len(indptr) or indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("incidence_indptr must be monotone and start at zero")
    if int(indptr[-1]) != len(indices) or len(coefficients) != len(indices):
        raise ValueError("cone incidence CSR array sizes are inconsistent")
    if np.any(indices < 0) or np.any(indices >= integer_count):
        raise ValueError("incidence_indices contains an out-of-range integer position")
    if np.any(coefficients == 0):
        raise ValueError("incidence_coefficients must be nonzero")
    for cone_index in range(len(base)):
        begin = int(indptr[cone_index])
        end = int(indptr[cone_index + 1])
        if len(np.unique(indices[begin:end])) != end - begin:
            raise ValueError("each cone row must contain unique integer positions")
    return ResolvedRounding(
        selection,
        projection,
        threshold,
        label,
        indptr,
        indices,
        coefficients,
        base,
        minimum,
    )


def project_coordinate(value: float, projection: str) -> int:
    if not math.isfinite(value):
        raise RuntimeError("rounding encountered a non-finite lattice coordinate")
    if projection == "nearest":
        projected = math.floor(value + 0.5)
    elif projection == "floor":
        projected = math.floor(value)
    elif projection == "ceil":
        projected = math.ceil(value)
    else:
        projected = math.ceil(value) if value > 0.0 else math.floor(value)
    if not np.iinfo(np.int64).min <= projected <= np.iinfo(np.int64).max:
        raise RuntimeError("projected lattice coordinate exceeds int64 range")
    return projected


class ProjectionState:
    """Backend-independent state machine for value proposals and commits."""

    def __init__(self, policy: ResolvedRounding, integer_count: int) -> None:
        self._projection = policy.projection
        self._indptr = policy.incidence_indptr
        self._indices = policy.incidence_indices
        self._coefficients = policy.incidence_coefficients
        self._minimum = [int(value) for value in policy.minimum_cones]
        self._values = [int(value) for value in policy.base_cones]
        self._remaining = [
            int(self._indptr[index + 1] - self._indptr[index])
            for index in range(len(policy.base_cones))
        ]
        self._entries_by_position: list[list[tuple[int, int]]] = [
            [] for _ in range(integer_count)
        ]
        for cone_index in range(len(policy.base_cones)):
            for entry in range(
                int(self._indptr[cone_index]), int(self._indptr[cone_index + 1])
            ):
                self._entries_by_position[int(self._indices[entry])].append(
                    (cone_index, int(self._coefficients[entry]))
                )
        self.correction_count = 0

    def propose(self, position: int, coordinate: float) -> int:
        basic = project_coordinate(coordinate, self._projection)
        projected = basic
        lower: int | None = None
        upper: int | None = None
        for cone_index, coefficient in self._entries_by_position[position]:
            if self._remaining[cone_index] != 1:
                continue
            deficit = self._minimum[cone_index] - self._values[cone_index]
            if coefficient > 0:
                bound = -((-deficit) // coefficient)
                lower = bound if lower is None else max(lower, bound)
            else:
                bound = deficit // coefficient
                upper = bound if upper is None else min(upper, bound)
        if lower is not None and upper is not None and lower > upper:
            return projected
        if lower is not None:
            projected = max(projected, lower)
        if upper is not None:
            projected = min(projected, upper)
        if not np.iinfo(np.int64).min <= projected <= np.iinfo(np.int64).max:
            # Keep the basic projection when no representable terminal
            # correction exists. The final cone audit exposes the violation.
            return basic
        return projected

    def commit(self, position: int, coordinate: float) -> int:
        raw = project_coordinate(coordinate, self._projection)
        proposed = self.propose(position, coordinate)
        if proposed != raw:
            self.correction_count += 1
        for cone_index, coefficient in self._entries_by_position[position]:
            updated = self._values[cone_index] + coefficient * proposed
            if not np.iinfo(np.int64).min <= updated <= np.iinfo(np.int64).max:
                raise RuntimeError("cone accumulation exceeds int64 range")
            self._values[cone_index] = updated
            self._remaining[cone_index] -= 1
        return proposed

    def statistics(self) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
        values = np.asarray(self._values, dtype=np.int64)
        maximum_int64 = int(np.iinfo(np.int64).max)
        violations = np.asarray(
            [
                min(minimum - value, maximum_int64) if value < minimum else 0
                for minimum, value in zip(self._minimum, self._values)
            ],
            dtype=np.int64,
        )
        count = int(np.count_nonzero(violations))
        maximum = int(np.max(violations, initial=0))
        return values, violations, count, maximum, count == 0


__all__ = ["ResolvedRounding", "ProjectionState", "resolve_rounding"]
