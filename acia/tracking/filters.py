"""Plausibility filters for per-cell (per-tracklet) results.

Tracking errors -- a spurious division, a lost cell re-found as a "newborn", two
cells swapped -- show up as cell cycles that are implausibly short or long. A
:class:`TrackletFilter` states the range a per-cycle value must lie in, with
physical units::

    from acia.tracking.filters import TrackletFilter, apply_tracklet_filters

    filters = [TrackletFilter("doubling_time", "30 min", "120 min")]
    cycles = apply_tracklet_filters(doubling_df, filters)
    valid = cycles[cycles.valid]

Filtering **marks** rather than drops: the result gains a boolean ``valid`` column
and a ``rejected_by`` column naming the first filter a cell failed, so the
rejected cells can still be shown (in grey, see :mod:`acia.analysis.cell_cycle`)
and counted. The input table is not modified.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from acia import Q_
from acia.analysis.units import UNIT_ATTR, _is_dimensionless, _is_pint_column


@dataclass(frozen=True)
class TrackletFilter:
    """Keep cells whose ``column`` lies in ``[min, max]`` (inclusive).

    Args:
        column: the per-cell column to test, e.g. ``"doubling_time"`` (the
            birth-to-division duration) or ``"mu_fit_r2"``.
        min / max: bounds; ``None`` leaves that side open. A string or pint
            quantity (``"30 min"``) is converted to the column's unit, so
            ``"0.5 h"`` and ``"30 min"`` mean the same; a plain number is taken
            to be in the column's unit already.
        name: label used in ``rejected_by``; defaults to the column name.
    """

    column: str = "doubling_time"
    min: Any = None
    max: Any = None
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or self.column

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description (for ``summary.json`` / the manifest)."""
        return {
            "column": self.column,
            "min": None if self.min is None else str(self.min),
            "max": None if self.max is None else str(self.max),
        }

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean array: ``True`` where the cell passes (NaN fails)."""
        if self.column not in df.columns:
            raise KeyError(
                f"TrackletFilter on {self.column!r}, but the table has no such "
                f"column (available: {list(df.columns)})."
            )
        series = df[self.column]
        if _is_pint_column(series):
            unit = str(series.pint.units)
            values = np.asarray(series.values.quantity.magnitude, dtype=float)
        else:
            unit = str((df.attrs.get(UNIT_ATTR) or {}).get(self.column, ""))
            values = np.asarray(series, dtype=float)

        keep = ~np.isnan(values)
        lo = _bound(self.min, unit, self.column)
        hi = _bound(self.max, unit, self.column)
        if lo is not None:
            keep &= values >= lo
        if hi is not None:
            keep &= values <= hi
        return np.asarray(keep, dtype=bool)


def _bound(value, unit: str, column: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    quantity = Q_(value) if isinstance(value, str) else value
    if _is_dimensionless(unit):
        if not quantity.dimensionless:
            raise ValueError(
                f"Bound {value!r} has a unit, but column {column!r} has none -- "
                "pass a plain number."
            )
        return float(quantity.magnitude)
    return float(quantity.to(unit).magnitude)


def duration_filter(min=None, max=None) -> TrackletFilter:
    """Filter on the birth-to-division duration, e.g. ``duration_filter("30 min", "2 h")``."""
    return TrackletFilter("doubling_time", min, max, name="duration")


def apply_tracklet_filters(
    df: pd.DataFrame, filters: Iterable[TrackletFilter]
) -> pd.DataFrame:
    """Mark each row as ``valid`` or not; never drops a row.

    Returns:
        A copy of ``df`` with a boolean ``valid`` column (all filters passed)
        and a ``rejected_by`` column holding the label of the first filter
        that failed (``None`` for valid cells). ``df.attrs`` is carried over.
        An empty filter list marks everything valid.
    """
    out = df.copy()
    out.attrs = dict(df.attrs)
    rejected_by = np.full(len(out), None, dtype=object)
    for f in filters:
        failed = ~f.mask(out) & (rejected_by == None)  # noqa: E711 (object array)
        rejected_by[failed] = f.label
    out["valid"] = rejected_by == None  # noqa: E711
    out["rejected_by"] = rejected_by
    return out


def filter_report(df: pd.DataFrame) -> pd.Series:
    """How many cells each filter rejected, plus the valid count."""
    counts = df["rejected_by"].value_counts(dropna=True)
    counts["valid"] = int(df["valid"].sum())
    counts["total"] = len(df)
    return counts
