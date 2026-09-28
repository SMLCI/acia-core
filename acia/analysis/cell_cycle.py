"""Per-cell-cycle views: one value per cell, placed on its birth-to-division span.

The tables here have one row per complete cell cycle -- a cell observed from its
birth (``start_time``) to its division (``division_time``) -- and one or more
per-cycle values: the doubling time
(:func:`acia.analysis.doubling_time.compute_doubling_times`) or a per-cell growth
rate (:func:`acia.analysis.growth_rate.compute_cell_growth_rates`). The same
plots and summary work for any of them through ``value_col``:

* :func:`plot_cycle_dots` -- one dot per division, at the time it happened;
* :func:`plot_cycle_bars` -- one horizontal bar per cell, from birth to division,
  at the height of its value; stacked up, the bars *are* the distribution;
* :func:`plot_cycle_distribution` -- the marginal, as a violin with every cell
  shown as a jittered point;
* :func:`summarize` -- ``n``, mean, sample std, median and SEM, ready for a quick
  experiment check or ``summary.json`` (see :func:`update_summary`).

A boolean ``valid`` column (e.g. from :func:`acia.tracking.filters.apply_tracklet_filters`)
is honoured by the plots: rejected cells are drawn in grey rather than hidden, so
what a filter removed stays inspectable.

Columns may be ``pint`` columns or plain floats (after
:func:`acia.analysis.units.strip_units`, whose units are read from ``df.attrs``).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from acia.analysis.units import UNIT_ATTR, _is_pint_column

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: color of the cells that passed the filters (matches the hose plot)
VALID_COLOR = "tab:blue"
#: color of the cells a filter rejected
REJECTED_COLOR = "0.65"

SUMMARY_NAME = "summary.json"


def _column(
    df: pd.DataFrame, col: str, unit: str | None = None
) -> tuple[np.ndarray, str]:
    """``(float magnitudes, unit)`` of a pint or plain-float column.

    A pint column is converted to ``unit`` when one is given.
    """
    series = df[col]
    if _is_pint_column(series):
        quantity = series.values.quantity
        if unit:
            quantity = quantity.to(unit)
        return np.asarray(quantity.magnitude, dtype=float), str(quantity.units)
    recorded = str((df.attrs.get(UNIT_ATTR) or {}).get(col, ""))
    return np.asarray(series, dtype=float), unit or recorded


def _valid_mask(df: pd.DataFrame) -> np.ndarray:
    if "valid" in df.columns:
        return np.asarray(df["valid"], dtype=bool)
    return np.ones(len(df), dtype=bool)


def _label(col: str, unit: str) -> str:
    name = col.replace("_", " ")
    return f"{name} [{unit}]" if unit else name


def _axes(ax: Axes | None, figsize=(6, 3.5)) -> tuple[Figure, Axes]:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return cast("Figure", ax.figure), ax


def _time_unit(df: pd.DataFrame, time_col: str) -> str:
    return _column(df, time_col)[1]


def plot_cycle_dots(
    df: pd.DataFrame,
    value_col: str = "doubling_time",
    *,
    time_col: str = "division_time",
    ax: Axes | None = None,
    value_label: str | None = None,
    show_rejected: bool = True,
) -> Figure:
    """One dot per cell cycle: x = when it divided, y = its value.

    Every division stays visible as its own point, so outliers, a drift over the
    experiment or a spike at the tracking resolution are easy to spot.

    Args:
        df: one row per cell cycle, with ``time_col`` and ``value_col``.
        value_col: the per-cycle value to plot (default ``"doubling_time"``).
        time_col: x position; ``"division_time"`` (default) dates each dot at
            the daughters' birth.
        ax: axes to draw into; a new figure is created when ``None``.
        value_label: y-axis label; defaults to ``"<value col> [<unit>]"``.
        show_rejected: draw cells with ``valid == False`` in grey (default) --
            ``False`` leaves them out.

    Returns:
        The matplotlib figure (not shown).
    """
    fig, ax = _axes(ax)
    t, t_unit = _column(df, time_col)
    v, v_unit = _column(df, value_col)
    valid = _valid_mask(df)

    if show_rejected and (~valid).any():
        ax.scatter(
            t[~valid], v[~valid], s=14, color=REJECTED_COLOR, label="filtered out"
        )
    ax.scatter(t[valid], v[valid], s=14, color=VALID_COLOR, label="cells")

    ax.set_xlabel(_label(time_col, t_unit))
    ax.set_ylabel(value_label or _label(value_col, v_unit))
    if show_rejected and (~valid).any():
        ax.legend()
    fig.tight_layout()
    return fig


def plot_cycle_bars(
    df: pd.DataFrame,
    value_col: str = "doubling_time",
    *,
    start_col: str = "start_time",
    end_col: str = "division_time",
    ax: Axes | None = None,
    value_label: str | None = None,
    show_rejected: bool = True,
    spread: bool = True,
) -> Figure:
    """One horizontal bar per cell, from its birth to its division.

    The bar sits at the height of the cell's value, so the bars stacked along
    the y axis are the distribution, and their extent shows which part of the
    experiment each value comes from.

    Args:
        df: one row per cell cycle with ``start_col``, ``end_col`` and
            ``value_col``.
        value_col: the per-cycle value (default ``"doubling_time"``).
        start_col / end_col: the bar's extent (birth and division by default).
        ax: axes to draw into; a new figure is created when ``None``.
        value_label: y-axis label; defaults to ``"<value col> [<unit>]"``.
        show_rejected: draw cells with ``valid == False`` in grey (default) --
            ``False`` leaves them out.
        spread: doubling times come in whole frame intervals, so many cells
            share a value and their bars would lie exactly on top of each
            other. With ``spread`` (default) bars of equal value are fanned out
            vertically within +-40% of the value spacing, so a value held by
            many cells shows as a thick band -- the distribution stays visible.

    Returns:
        The matplotlib figure (not shown).
    """
    fig, ax = _axes(ax)
    start, t_unit = _column(df, start_col)
    end, _ = _column(df, end_col, t_unit or None)
    v, v_unit = _column(df, value_col)
    valid = _valid_mask(df)
    if spread:
        v = _spread_ties(v, start)

    if show_rejected and (~valid).any():
        ax.hlines(
            v[~valid],
            start[~valid],
            end[~valid],
            color=REJECTED_COLOR,
            lw=1.5,
            label="filtered out",
        )
    ax.hlines(
        v[valid], start[valid], end[valid], color=VALID_COLOR, lw=1.5, label="cells"
    )

    ax.set_xlabel(f"time [{t_unit}]" if t_unit else "time")
    ax.set_ylabel(value_label or _label(value_col, v_unit))
    if show_rejected and (~valid).any():
        ax.legend()
    fig.tight_layout()
    return fig


def _spread_ties(values: np.ndarray, order_by: np.ndarray) -> np.ndarray:
    """Fan out equal values within +-40% of the spacing between distinct values.

    Within a group of equal values, bars are ordered by ``order_by`` (birth
    time), so the fan keeps a stable, readable order.
    """
    finite = np.isfinite(values)
    distinct = np.unique(values[finite])
    if len(distinct) == len(values[finite]):
        return values
    gaps = np.diff(distinct)
    spacing = (
        float(gaps.min()) if len(gaps) else max(abs(float(distinct[0])) * 0.1, 1.0)
    )
    out = values.astype(float).copy()
    for value in distinct:
        (members,) = np.nonzero(finite & (values == value))
        if len(members) < 2:
            continue
        members = members[np.argsort(order_by[members], kind="stable")]
        out[members] = value + np.linspace(-0.4, 0.4, len(members)) * spacing
    return out


def plot_cycle_distribution(
    df: pd.DataFrame,
    value_col: str = "doubling_time",
    *,
    ax: Axes | None = None,
    value_label: str | None = None,
    jitter: float = 0.08,
    random_state=0,
) -> Figure:
    """The marginal distribution: a violin plus every (valid) cell as a point.

    The points keep individual cells visible where the violin's smoothing would
    hide a small sample or a bimodal population.
    """
    fig, ax = _axes(ax, figsize=(2.5, 3.5))
    v, v_unit = _column(df, value_col)
    v = v[_valid_mask(df) & ~np.isnan(v)]

    if len(v) > 1:
        ax.violinplot(v, showmeans=True, showextrema=False)
    rng = np.random.default_rng(random_state)
    ax.scatter(
        1 + rng.uniform(-jitter, jitter, size=len(v)),
        v,
        s=10,
        color=VALID_COLOR,
        alpha=0.6,
        zorder=3,
    )
    ax.set_xticks([])
    ax.set_ylabel(value_label or _label(value_col, v_unit))
    fig.tight_layout()
    return fig


def summarize(values, unit: str | None = None) -> dict[str, Any]:
    """``n``, mean, sample std (ddof=1), median and SEM of a set of values.

    ``values`` may be a pint Series/Quantity (its unit is used, or converted to
    ``unit``), a plain Series, list or array. NaNs are ignored. Statistics that
    need more data than there is (std/SEM of a single value) are ``None``, so
    the result serializes to JSON as-is.
    """
    if isinstance(values, pd.Series) and _is_pint_column(values):
        values = values.values.quantity
    if hasattr(values, "magnitude") and hasattr(values, "units"):
        if unit:
            values = values.to(unit)
        unit = str(values.units)
        values = values.magnitude

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)

    def _f(x) -> float | None:
        return None if x is None or not math.isfinite(x) else float(x)

    std = float(np.std(arr, ddof=1)) if n > 1 else None
    return {
        "n": n,
        "mean": _f(arr.mean()) if n else None,
        "std": _f(std),
        "median": _f(np.median(arr)) if n else None,
        "sem": _f(std / math.sqrt(n)) if std is not None else None,
        "unit": unit or "",
    }


def format_summary(name: str, stats: dict[str, Any], digits: int = 2) -> str:
    """One line such as ``doubling time = 42.1 ± 5.3 min (median 41.0, n = 57)``."""
    if not stats.get("n"):
        return f"{name}: no values"
    unit = f" {stats['unit']}" if stats.get("unit") else ""
    spread = f" ± {stats['std']:.{digits}f}" if stats.get("std") is not None else ""
    return (
        f"{name} = {stats['mean']:.{digits}f}{spread}{unit} "
        f"(median {stats['median']:.{digits}f}, n = {stats['n']})"
    )


def update_summary(
    output_dir: str | Path, section: str, values: dict[str, Any]
) -> Path:
    """Merge ``values`` into section ``section`` of ``<output_dir>/summary.json``.

    One ``summary.json`` per population collects the headline numbers of every
    stage (doubling time, per-cell and population growth rates, ...). Each stage
    owns one section and only ever replaces that one, so stages can write in any
    order; the file is replaced atomically, so a reader never sees half of it.

    Returns:
        The path of ``summary.json``.
    """
    path = Path(output_dir) / SUMMARY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        current = {}
    current[section] = values

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".summary-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(current, handle, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_summaries(paths, keys=None) -> pd.DataFrame:
    """Flatten several ``summary.json`` files into one table, one row per file.

    Nested sections become ``section.field`` columns (e.g.
    ``doubling_time.mean``). ``keys`` maps each path to extra columns (e.g. the
    experiment/position/roi of that population).
    """
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        row = dict((keys or {}).get(path, {}))
        row.update(pd.json_normalize(data, sep=".").iloc[0].to_dict() if data else {})
        rows.append(row)
    return pd.DataFrame(rows)


def frame_interval_of(source) -> str | None:
    """The source's (median) frame interval as text, e.g. ``"10.0 minute"``.

    Recorded next to every summary: each duration and rate scales with it, so a
    wrong interval is the first thing to rule out when numbers look off.
    ``None`` for an uncalibrated or single-frame source.
    """
    timepoints = getattr(source, "timepoints", None)
    if timepoints is None or len(timepoints) < 2:
        return None
    steps = np.diff(timepoints.magnitude)
    return f"{float(np.median(steps)):g} {timepoints.units}"
