"""Group positions into experimental conditions.

A condition (a substrate concentration, a strain, a treatment) is usually
assigned to a block of microscope positions. :func:`assign_conditions` maps it
from compact position ranges -- in the 1-based numbering of the exported files
and the microscope software (``pos001`` is position 1)::

    conditions = {
        "JW_nd73-01": {"0.1 mM": "pos001-pos010", "1 mM": "11-20, 25"},
        "JW_nd74-01": {"0.1 mM": "1-10", "10 mM": "11-30"},
    }

A mapping without the experiment level applies to every experiment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

_RANGE = re.compile(r"^(?:pos)?0*(\d+)(?:\s*-\s*(?:pos)?0*(\d+))?$", re.IGNORECASE)


def parse_position_ranges(spec) -> set[int]:
    """Positions named by a range spec: ``"pos001-pos010, 12, 15-18"``.

    Accepts a string (comma separated items, each ``N`` or ``N-M``, optionally
    with the ``pos`` prefix and leading zeros), an int, or an iterable of those.
    Positions are 1-based, as in ``pos001_roi001.tiff``.

    Raises:
        ValueError: on an item that is not a position or a range, or a range
            that runs backwards.
    """
    if isinstance(spec, (int, np.integer)):
        return {int(spec)}
    if not isinstance(spec, str):
        out: set[int] = set()
        for item in spec:
            out |= parse_position_ranges(item)
        return out
    positions: set[int] = set()
    for item in filter(None, (part.strip() for part in spec.split(","))):
        match = _RANGE.match(item)
        if not match:
            raise ValueError(
                f"{item!r} is not a position or a range like 'pos001-pos010'"
            )
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if last < first:
            raise ValueError(f"range {item!r} runs backwards")
        positions.update(range(first, last + 1))
    return positions


def _per_experiment(conditions: Mapping, experiments: Iterable[str]) -> dict[str, dict]:
    """Normalize to ``{experiment: {condition: spec}}``."""
    experiments = list(experiments)
    nested = conditions and all(isinstance(v, Mapping) for v in conditions.values())
    if nested:
        return {str(k): dict(v) for k, v in conditions.items()}
    return {exp: dict(conditions) for exp in experiments}


def condition_table(conditions: Mapping, experiments: Iterable[str]) -> pd.DataFrame:
    """``experiment, position, condition`` rows for every mapped position.

    Raises:
        ValueError: if a position of an experiment is given two conditions.
    """
    rows = []
    for experiment, mapping in _per_experiment(conditions, experiments).items():
        owner: dict[int, str] = {}
        for condition, spec in mapping.items():
            for position in sorted(parse_position_ranges(spec)):
                if position in owner:
                    raise ValueError(
                        f"{experiment}: position {position} is mapped to both "
                        f"{owner[position]!r} and {condition!r}"
                    )
                owner[position] = condition
                rows.append(
                    {
                        "experiment": experiment,
                        "position": position,
                        "condition": condition,
                    }
                )
    return pd.DataFrame(rows, columns=["experiment", "position", "condition"])


def assign_conditions(
    df: pd.DataFrame,
    conditions: Mapping,
    *,
    values: Mapping[str, Any] | None = None,
    value_name: str = "condition_value",
    experiment_col: str = "experiment",
    position_col: str = "position",
) -> pd.DataFrame:
    """Add a ``condition`` column (and optionally a numeric value per condition).

    Args:
        df: any table with ``experiment`` and ``position`` (1-based) columns --
            per run, per cell or per time point.
        conditions: ``{experiment: {condition: ranges}}`` or, for all
            experiments alike, ``{condition: ranges}``; see
            :func:`parse_position_ranges`.
        values: optional ``{condition: number}``, e.g. the substrate
            concentration, added as column ``value_name``.
        value_name: name of that column (e.g. ``"substrate"``).

    Returns:
        A copy of ``df`` with ``condition`` (``None`` where a position is not
        mapped) and, with ``values``, the value column. The rows that stayed
        unmapped are listed in ``attrs["unmapped"]`` as ``(experiment,
        position)`` pairs, so a forgotten block is visible.

    Raises:
        ValueError: if a position is mapped to two conditions.
    """
    out = df.copy()
    out.attrs = dict(df.attrs)
    if experiment_col in out.columns:
        experiments = out[experiment_col].dropna().astype(str).unique()
    else:
        experiments = np.array([""])
        out[experiment_col] = ""
    table = condition_table(conditions, experiments)
    lookup = {
        (str(r.experiment), int(r.position)): r.condition for r in table.itertuples()
    }
    keys = zip(
        out[experiment_col].astype(str),
        pd.to_numeric(out[position_col], errors="coerce"),
        strict=True,
    )
    out["condition"] = [
        lookup.get((e, int(p))) if pd.notna(p) else None for e, p in keys
    ]
    missing = out[out["condition"].isna()][
        [experiment_col, position_col]
    ].drop_duplicates()
    out.attrs["unmapped"] = [
        (str(e), None if pd.isna(p) else int(p))
        for e, p in missing.itertuples(index=False)
    ]
    if values is not None:
        out[value_name] = [
            values.get(c) if c is not None else None for c in out["condition"]
        ]
        out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out
