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
from collections.abc import Iterable
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
        valid = df["valid"]
        if valid.dtype == object:  # e.g. "True"/"False" read back from text
            return np.asarray(valid.astype(str).str.lower().isin(("true", "1")), bool)
        return np.asarray(valid, dtype=bool)
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


def _as_floats(values, unit: str | None = None) -> tuple[np.ndarray, str]:
    """Float magnitudes (NaNs dropped) and unit of a pint/plain series or array."""
    if isinstance(values, pd.Series) and _is_pint_column(values):
        values = values.values.quantity
    if hasattr(values, "magnitude") and hasattr(values, "units"):
        if unit:
            values = values.to(unit)
        unit = str(values.units)
        values = values.magnitude
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)], unit or ""


def plot_cycle_violins(
    groups: dict,
    *,
    unit: str | None = None,
    ax: Axes | None = None,
    value_label: str | None = None,
    jitter: float = 0.08,
    random_state=0,
) -> Figure:
    """Several distributions side by side: a violin plus every value as a point.

    Args:
        groups: ``{label: values}`` -- pint Series/Quantity (converted to
            ``unit``), plain Series or arrays. NaNs are dropped; the label shows
            the number of values.
        unit: common unit of the pint values (e.g. ``"1/hour"``).
        ax: axes to draw into; a new figure is created when ``None``.
        value_label: y-axis label; defaults to the unit.

    Returns:
        The matplotlib figure (not shown).
    """
    fig, ax = _axes(ax, figsize=(1.6 + 1.4 * len(groups), 3.5))
    rng = np.random.default_rng(random_state)
    labels, seen_unit = [], unit or ""
    for i, (label, values) in enumerate(groups.items(), start=1):
        v, u = _as_floats(values, unit)
        seen_unit = seen_unit or u
        if len(v) > 1:
            parts = ax.violinplot(v, positions=[i], showmeans=True, showextrema=False)
            for body in list(parts["bodies"]):  # type: ignore[call-overload]
                body.set_alpha(0.35)
        ax.scatter(
            i + rng.uniform(-jitter, jitter, size=len(v)),
            v,
            s=10,
            color=VALID_COLOR,
            alpha=0.6,
            zorder=3,
        )
        labels.append(f"{label}\n(n = {len(v)})")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel(value_label or (f"[{seen_unit}]" if seen_unit else ""))
    fig.tight_layout()
    return fig


#: colour per flag reason of the fit-vs-doubling-time plots
REASON_COLORS = {
    "not flagged": "tab:blue",
    "too few points": "0.6",
    "size drop (missed division?)": "tab:orange",
    "size jump (merged cells?)": "tab:purple",
    "poor fit": "tab:brown",
    "disagrees with doubling time": "tab:red",
}


def _rate_errors(df: pd.DataFrame, rate_unit: str) -> dict[str, np.ndarray]:
    """Rates and their interval half-widths (``NaN`` where a column is missing)."""
    out = {}
    for col in (
        "mu_fit",
        "mu_fit_ci_lo",
        "mu_fit_ci_hi",
        "mu_from_dt",
        "mu_from_dt_lo",
        "mu_from_dt_hi",
    ):
        out[col] = (
            _column(df, col, rate_unit)[0]
            if col in df.columns
            else np.full(len(df), np.nan)
        )
    out["y_err"] = np.vstack(
        [out["mu_fit"] - out["mu_fit_ci_lo"], out["mu_fit_ci_hi"] - out["mu_fit"]]
    )
    out["x_err"] = np.vstack(
        [
            out["mu_from_dt"] - out["mu_from_dt_lo"],
            out["mu_from_dt_hi"] - out["mu_from_dt"],
        ]
    )
    for key in ("x_err", "y_err"):
        err = out[key]
        err[~np.isfinite(err)] = 0.0
        out[key] = np.clip(err, 0, None)
    return out


def flag_reasons(df: pd.DataFrame) -> np.ndarray:
    """``fit_outlier_reason`` per row, ``"not flagged"`` where there is none."""
    if "fit_outlier_reason" not in df.columns:
        return np.full(len(df), "not flagged", dtype=object)
    reasons = df["fit_outlier_reason"].astype(object).to_numpy()
    return np.where(pd.isna(reasons), "not flagged", reasons).astype(object)


def plot_fit_vs_doubling(
    rates: pd.DataFrame,
    *,
    rate_unit: str = "1/hour",
    ax: Axes | None = None,
    error_bars: bool = True,
) -> Figure:
    """Fitted growth rate against ``ln 2 / doubling time``, one point per cell.

    How to read it: a cell on the dashed ``y = x`` line has two readouts that
    agree. The error bars are the fit's confidence interval (vertical) and
    the range of ``ln 2 / DT`` the frame timing allows (horizontal). Points
    far from the line **with small error bars** are the cells worth
    inspecting (:func:`acia.analysis.cell_inspect.show_cell`); points far off
    with long bars are just imprecise. Colours give the flag reason (see
    :func:`acia.analysis.growth_rate.compute_cell_growth_rates`).

    Args:
        rates: :func:`~acia.analysis.growth_rate.compute_cell_growth_rates` output.
        rate_unit: unit of both axes.
        ax: axes to draw into (a new figure otherwise).
        error_bars: draw the intervals (default ``True``).
    """
    fig, ax = _axes(ax, figsize=(4.5, 4.5))
    values = _rate_errors(rates, rate_unit)
    reasons = flag_reasons(rates)
    for reason, color in REASON_COLORS.items():
        sel = reasons == reason
        if not sel.any():
            continue
        x, y = values["mu_from_dt"][sel], values["mu_fit"][sel]
        if error_bars:
            ax.errorbar(
                x,
                y,
                xerr=values["x_err"][:, sel],
                yerr=values["y_err"][:, sel],
                fmt="none",
                ecolor=color,
                alpha=0.35,
                lw=0.8,
            )
        ax.scatter(x, y, s=18, color=color, label=f"{reason} ({int(sel.sum())})")
    finite = np.r_[values["mu_fit"], values["mu_from_dt"], 0.0]
    top = float(np.nanmax(finite[np.isfinite(finite)])) * 1.1 or 1.0
    ax.plot([0, top], [0, top], ls="--", color="0.5", lw=1, label="y = x")
    unit = f"{_label('', rate_unit).strip()}"
    ax.set_xlabel(f"ln 2 / doubling time {unit}")
    ax.set_ylabel(f"growth rate μ (fit) {unit}")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def pick_percentile_cells(
    df: pd.DataFrame,
    value_col: str = "mu_fit",
    percentiles=(0, 50, 100),
    *,
    exclude_flagged: bool = True,
) -> pd.DataFrame:
    """The cells whose value lies closest to each percentile of the population.

    Only valid cells (``valid`` column, if present) with a value are
    considered; with ``exclude_flagged`` also cells whose growth fit was
    flagged (``fit_outlier``) are left out, so the examples show real cells,
    not artifacts. P0 and P100 are the smallest and largest value.

    Returns:
        One row per percentile: ``percentile``, ``tracklet`` (the df index)
        and ``value`` -- empty when no cell qualifies.
    """
    values, _ = _column(df, value_col)
    ok = _valid_mask(df) & ~np.isnan(values)
    if exclude_flagged and "fit_outlier" in df.columns:
        ok &= ~np.asarray(df["fit_outlier"], dtype=bool)
    idx, v = df.index[ok], values[ok]
    rows = []
    if len(v):
        for pct in percentiles:
            target = np.percentile(v, pct)
            i = int(np.argmin(np.abs(v - target)))
            rows.append({"percentile": pct, "tracklet": idx[i], "value": float(v[i])})
    return pd.DataFrame(rows, columns=["percentile", "tracklet", "value"])


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


def _json_safe(value: Any) -> Any:
    """``value`` with every non-finite float (NaN, ±inf) replaced by ``None``.

    ``json.dump`` writes NaN as a bare ``NaN`` -- not valid JSON, and its
    ``default=`` hook is never called for floats -- so the summary is cleaned
    before it is written. Numpy scalars are converted to Python numbers.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def update_summary(
    output_dir: str | Path, section: str, values: dict[str, Any]
) -> Path:
    """Merge ``values`` into section ``section`` of ``<output_dir>/summary.json``.

    One ``summary.json`` per population collects the headline numbers of every
    stage (doubling time, per-cell and population growth rates, ...). Each stage
    owns one section and only ever replaces that one, so stages can write in any
    order; the file is replaced atomically, so a reader never sees half of it.

    The section is attributed to the stage context open on ``output_dir`` (one
    built with ``stage=`` and not yet finished), so clearing that stage --
    :func:`acia.analysis.stage.clear_stage`, ``scale(on_existing="overwrite")`` --
    removes exactly its sections and leaves the other stages' numbers in place.

    Returns:
        The path of ``summary.json``.
    """
    path = Path(output_dir) / SUMMARY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        current = {}
    current[section] = _json_safe(values)

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".summary-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(
                _json_safe(current),
                handle,
                indent=2,
                sort_keys=True,
                default=str,
                allow_nan=False,
            )
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    from acia.analysis.stage import note_summary_section  # noqa: PLC0415 -- cycle

    note_summary_section(output_dir, section)
    return path


def remove_summary_sections(
    output_dir: str | Path, sections: Iterable[str]
) -> Path | None:
    """Drop ``sections`` from ``<output_dir>/summary.json``, atomically.

    The undo of :func:`update_summary`, used when a stage is cleared so its
    numbers do not outlive its results. The file is deleted once no section is
    left. Missing sections, a missing file and an unreadable file are no-ops.

    Returns:
        The path of ``summary.json`` when something was removed, else ``None``.
    """
    path = Path(output_dir) / SUMMARY_NAME
    try:
        current = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    drop = [s for s in sections if s in current]
    if not drop:
        return None
    for section in drop:
        current.pop(section)
    if not current:
        path.unlink(missing_ok=True)
        return path

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".summary-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as handle:
            # already-parsed JSON goes back out as it came in
            json.dump(
                _json_safe(current),
                handle,
                indent=2,
                sort_keys=True,
                default=str,
                allow_nan=False,
            )
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
