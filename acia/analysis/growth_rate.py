"""Unit-aware exponential growth-rate estimation from extractor tables.

This module fits the exponential growth model ``Q(t) = Q0 * exp(mu * t)`` to a
property-extractor DataFrame (the output of
:class:`acia.analysis.ExtractorExecutor`). The fit is performed as an ordinary
least squares (OLS) regression of ``log(y)`` on time ``t`` using
:mod:`statsmodels`, which yields the growth rate ``mu`` together with its fit
uncertainty (standard error, confidence interval, p-value) and the coefficient
of determination ``R^2`` for free.

All rate-like quantities are returned as :class:`pint.Quantity` objects built
from the shared :data:`acia.ureg` registry, so the growth rate is independent of
the imaging interval and inter-operates with the rest of the unit-aware library.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import pint
import statsmodels.api as sm  # type: ignore[import-untyped]

from acia import Q_
from acia.analysis.units import UNIT_ATTR, _is_pint_column

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from acia.base import ImageSequenceSource

#: aggregation modes accepted by :func:`estimate_growth_rate`
AggMode = Literal["sum", "mean", "count"]

#: default time unit used when none can be inferred from the DataFrame or args
DEFAULT_TIME_UNIT = "hour"


@dataclass
class GrowthRateResult:
    """Result of an exponential growth-rate fit.

    All rate quantities are :class:`pint.Quantity` in ``1 / time_unit`` and
    ``doubling_time`` is a :class:`pint.Quantity` in ``time_unit``, where the
    time unit is inferred from the source DataFrame (see
    :func:`estimate_growth_rate`).

    Attributes:
        growth_rate: The fitted growth rate ``mu`` (slope of ``log(y)`` vs.
            ``t``), in ``1 / time_unit``.
        growth_rate_std_err: Standard error of the fitted growth rate, in
            ``1 / time_unit``. Only meaningful with >= 3 time points.
        growth_rate_ci: ``(low, high)`` confidence-interval bounds for the
            growth rate, in ``1 / time_unit``. Only meaningful with >= 3 time
            points.
        doubling_time: The doubling time ``ln(2) / mu``, in ``time_unit``; ``nan``
            when the growth rate is not positive (a non-growing population never
            doubles). With fewer than 3 time points the SE/CI/p-value are ``nan``.
        initial_value: The fitted initial quantity ``Q0 = exp(intercept)`` (a
            plain float in the units of the aggregated quantity).
        r_squared: Coefficient of determination of the log-linear fit.
        p_value: Two-sided p-value for the growth-rate (slope) coefficient.
    """

    growth_rate: pint.Quantity
    growth_rate_std_err: pint.Quantity
    growth_rate_ci: tuple[pint.Quantity, pint.Quantity]
    doubling_time: pint.Quantity
    initial_value: float
    r_squared: float
    p_value: float


def _aggregate(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    agg: AggMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Group ``df`` by ``time_col`` and aggregate the quantity per ``agg``.

    Args:
        df: the source DataFrame.
        time_col: name of the time column to group by.
        value_col: name of the value column to aggregate (ignored for
            ``agg="count"``).
        agg: aggregation mode -- ``"sum"``, ``"mean"`` or ``"count"``.

    Returns:
        ``(t, y)`` where ``t`` is the sorted array of distinct times and ``y`` is
        the aggregated quantity at each time.
    """
    grouped = df.groupby(time_col, sort=True)
    if agg == "count":
        series = grouped.size()
    elif agg == "sum":
        series = grouped[value_col].sum()
    elif agg == "mean":
        series = grouped[value_col].mean()
    else:  # pragma: no cover - guarded by Literal type / validation below
        raise ValueError(f"Unknown aggregation mode: {agg!r}")

    t = series.index.to_numpy(dtype=float)
    y = series.to_numpy(dtype=float)
    return t, y


def estimate_growth_rate(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    value_col: str = "area",
    agg: AggMode = "sum",
    time_unit: str | None = None,
    ci_level: float = 0.95,
    ax: Axes | None = None,
) -> tuple[GrowthRateResult, Figure]:
    """Estimate the exponential growth rate from an extractor DataFrame.

    Groups ``df`` by ``time_col``, aggregates the chosen quantity, and fits the
    exponential model ``Q(t) = Q0 * exp(mu * t)`` as an OLS regression of
    ``log(y)`` on ``t`` using :mod:`statsmodels`. From the single fit it derives
    the growth rate ``mu`` and its uncertainty.

    Args:
        df: an :class:`acia.analysis.ExtractorExecutor` output DataFrame, with a
            time column and (for ``agg`` in ``{"sum", "mean"}``) a value column.
            The time unit is read from ``df.attrs["units"][time_col]`` when
            present.
        time_col: name of the time column to group by.
        value_col: name of the value column to aggregate. Ignored when
            ``agg="count"``.
        agg: how to aggregate the quantity per time point -- ``"sum"`` (total,
            default), ``"mean"`` or ``"count"`` (number of rows per time, i.e.
            cell-count growth).
        time_unit: explicit time unit, used only when the unit cannot be inferred
            from ``df.attrs["units"]``. Defaults to ``"hour"`` if also absent.
        ci_level: confidence level for ``growth_rate_ci`` (default ``0.95``).
        ax: optional matplotlib :class:`~matplotlib.axes.Axes` to draw into. When
            ``None`` a new figure and axes are created.

    Returns:
        A ``(result, figure)`` tuple: a :class:`GrowthRateResult` and a
        matplotlib :class:`~matplotlib.figure.Figure` showing the aggregated
        quantity vs. time with the fitted exponential curve. The figure is not
        shown (no ``plt.show()``).

    Raises:
        ValueError: if there are fewer than two distinct time points, or if any
            aggregated quantity is not strictly positive (the log requires
            ``y > 0``).

    Note:
        The standard error and confidence interval are only meaningful with at
        least three time points; with exactly two points the fit is perfect
        (zero residual degrees of freedom) and the reported SE/CI degenerate.
    """
    import matplotlib.pyplot as plt

    if not 0.0 < ci_level < 1.0:
        raise ValueError(
            f"ci_level must be in the open interval (0, 1), got {ci_level}."
        )
    if time_col not in df.columns:
        raise ValueError(f"time_col {time_col!r} is not a column of the DataFrame.")
    if agg in ("sum", "mean") and value_col not in df.columns:
        raise ValueError(f"value_col {value_col!r} is not a column of the DataFrame.")

    t, y = _aggregate(df, time_col=time_col, value_col=value_col, agg=agg)

    if t.size < 2:
        raise ValueError(
            f"Need at least 2 distinct time points to fit a growth rate, got {t.size}."
        )
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError(
            "Aggregated quantities must be finite and strictly positive to fit the "
            "log-linear growth model (got a non-finite value or a value <= 0)."
        )

    # resolve the time unit: DataFrame attrs win, then the explicit arg, then hour
    resolved_time_unit = df.attrs.get(UNIT_ATTR, {}).get(time_col)
    if resolved_time_unit is None:
        resolved_time_unit = time_unit if time_unit is not None else DEFAULT_TIME_UNIT
    resolved_time_unit = str(resolved_time_unit)
    try:
        # validate the unit the way the result is built (Q_(value, unit))
        Q_(1.0, resolved_time_unit)
    except Exception as exc:  # noqa: BLE001 - re-raised as a clear ValueError
        raise ValueError(
            f"Could not interpret time unit {resolved_time_unit!r} (from "
            f"df.attrs['units'][{time_col!r}] or the time_unit argument)."
        ) from exc

    # OLS of log(y) on t: design columns are [const, t] (has_constant='add' keeps
    # the const present so the slope is reliably params index 1)
    design = sm.add_constant(t, has_constant="add")
    with warnings.catch_warnings():
        # a 2-point fit has zero residual dof; statsmodels warns and yields nan
        # SE/CI -- surfaced as explicit nan below instead of leaking the warning
        warnings.simplefilter("ignore")
        model = sm.OLS(np.log(y), design).fit()
        intercept, slope = model.params
        r_squared = float(model.rsquared)
        if t.size >= 3:
            slope_se = float(model.bse[1])
            ci_lo, ci_hi = (float(b) for b in model.conf_int(alpha=1 - ci_level)[1])
            p_value = float(model.pvalues[1])
        else:
            slope_se = ci_lo = ci_hi = p_value = float("nan")

    rate_unit = f"1 / {resolved_time_unit}"
    result = GrowthRateResult(
        growth_rate=Q_(slope, rate_unit),
        growth_rate_std_err=Q_(slope_se, rate_unit),
        growth_rate_ci=(Q_(ci_lo, rate_unit), Q_(ci_hi, rate_unit)),
        doubling_time=Q_(
            np.log(2) / slope if slope > 0 else float("nan"), resolved_time_unit
        ),
        initial_value=float(np.exp(intercept)),
        r_squared=r_squared,
        p_value=p_value,
    )

    # plot: aggregated points + fitted exponential curve
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = cast("Figure", ax.figure)

    t_dense = np.linspace(float(t.min()), float(t.max()), 200)
    y_fit = result.initial_value * np.exp(slope * t_dense)

    ax.scatter(t, y, label="data", color="tab:blue", zorder=3)
    ax.plot(t_dense, y_fit, label="fit", color="tab:red")
    ax.set_xlabel(f"time [{resolved_time_unit}]")
    ax.set_ylabel("count" if agg == "count" else value_col)
    annotation = (
        f"$\\mu$ = {slope:.3g} $\\pm$ {slope_se:.2g} 1/{resolved_time_unit}\n"
        f"$R^2$ = {r_squared:.3f}"
    )
    ax.set_title(annotation)
    ax.legend()
    fig.tight_layout()

    return result, fig


def _cell_size(cont, measure: str) -> float:
    if measure == "area":
        return float(cont.area)
    if measure == "length":
        rect = cont.polygon.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        edges = np.hypot(np.diff(xs), np.diff(ys))
        return float(edges.max()) if len(edges) else float("nan")
    raise ValueError(f"measure must be 'area' or 'length', got {measure!r}")


def compute_cell_growth_rates(
    overlay,
    cycles: pd.DataFrame,
    source: ImageSequenceSource,
    *,
    measure: Literal["area", "length"] = "area",
    rate_unit: str = "1/hour",
    min_points: int = 4,
    r2_min: float = 0.9,
    rel_tol: float = 0.3,
) -> pd.DataFrame:
    """Per-cell exponential growth rate, fitted and derived from the doubling time.

    For every cell cycle in ``cycles`` (one row per tracklet, as returned by
    :func:`acia.analysis.doubling_time.compute_doubling_times`), the cell's size
    over its life is fitted as ``size(t) = size_0 * exp(mu * t)`` -- an OLS of
    ``log(size)`` on time. The pixel calibration cancels in the slope, so sizes
    are taken in pixels.

    Two growth rates are reported side by side:

    * ``mu_fit`` -- from the size fit, with its standard error and ``r2``;
    * ``mu_from_dt`` -- ``ln(2) / doubling_time``, i.e. what the cell's cycle
      duration implies if it doubled exactly once.

    A healthy cell has a good fit and the two agree. ``fit_outlier`` flags cells
    where ``r2 < r2_min`` or ``|mu_fit - mu_from_dt| / mu_from_dt > rel_tol``
    (reason in ``fit_outlier_reason``) -- typically a segmentation or tracking
    error (a merged neighbour, a swapped identity, a missed division).

    Args:
        overlay: the tracked overlay; each contour's ``label`` is its tracklet
            id (the index of ``cycles``).
        cycles: per-cycle table with at least ``doubling_time``; its time
            columns and any ``valid``/``rejected_by`` columns are carried over.
        source: the time-calibrated image sequence (for ``timepoints``).
        measure: ``"area"`` (default) or ``"length"`` (long side of the minimal
            rotated rectangle) as the cell size.
        rate_unit: unit of the growth rates (default ``"1/hour"``).
        min_points: minimum number of observations for a fit; cells with fewer
            get ``NaN`` fit values.
        r2_min / rel_tol: thresholds of the ``fit_outlier`` flag.

    Returns:
        ``cycles`` (copied) plus ``mu_fit``, ``mu_fit_se``, ``mu_from_dt``
        (``pint[rate_unit]``), ``r2``, ``n_points``, ``fit_outlier`` and
        ``fit_outlier_reason``, indexed by tracklet.

    Raises:
        ValueError: if ``source`` has no time calibration.
    """
    timepoints = source.timepoints
    if timepoints is None:
        raise ValueError(
            "per-cell growth rates require time calibration -- the source has "
            "neither a frame_interval nor per-frame timepoints"
        )
    time_unit = str((1 / Q_(1.0, rate_unit)).units)
    t_all = np.asarray(timepoints.to(time_unit).magnitude, dtype=float)

    by_label: dict = {}
    for cont in overlay.contours:
        by_label.setdefault(cont.label, []).append(cont)

    mu, se, r2, npts = [], [], [], []
    for tracklet in cycles.index:
        conts = sorted(by_label.get(tracklet, []), key=lambda c: c.frame)
        t = np.array([t_all[int(c.frame)] for c in conts], dtype=float)
        y = np.array([_cell_size(c, measure) for c in conts], dtype=float)
        ok = np.isfinite(y) & (y > 0)
        t, y = t[ok], y[ok]
        npts.append(int(t.size))
        if t.size < max(min_points, 2):
            mu.append(np.nan)
            se.append(np.nan)
            r2.append(np.nan)
            continue
        design = sm.add_constant(t, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(np.log(y), design).fit()
        mu.append(float(model.params[1]))
        se.append(float(model.bse[1]) if t.size >= 3 else np.nan)
        r2.append(float(model.rsquared))

    out = cycles.copy()
    out.attrs = dict(cycles.attrs)
    dt_col = cycles["doubling_time"]
    if _is_pint_column(dt_col):
        doubling = np.asarray(dt_col.values.quantity.to(time_unit).magnitude, float)
    else:
        unit = (cycles.attrs.get(UNIT_ATTR) or {}).get("doubling_time") or time_unit
        doubling = np.asarray(
            Q_(np.asarray(dt_col, float), unit).to(time_unit).magnitude
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        from_dt = np.where(doubling > 0, np.log(2) / doubling, np.nan)

    mu_arr, from_dt = np.asarray(mu, float), np.asarray(from_dt, float)
    r2_arr = np.asarray(r2, float)
    pint_dtype = f"pint[{rate_unit}]"
    out["mu_fit"] = pd.array(mu_arr, dtype=pint_dtype)
    out["mu_fit_se"] = pd.array(np.asarray(se, float), dtype=pint_dtype)
    out["mu_from_dt"] = pd.array(from_dt, dtype=pint_dtype)
    out["r2"] = r2_arr
    out["n_points"] = npts

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.abs(mu_arr - from_dt) / from_dt
    reasons = np.full(len(out), None, dtype=object)
    reasons[np.isnan(mu_arr)] = "too few points"
    reasons[(reasons == None) & (r2_arr < r2_min)] = "poor fit"  # noqa: E711
    reasons[(reasons == None) & (rel_diff > rel_tol)] = "disagrees with doubling time"  # noqa: E711
    out["fit_outlier"] = reasons != None  # noqa: E711
    out["fit_outlier_reason"] = reasons

    out.attrs.setdefault(UNIT_ATTR, {})
    out.attrs[UNIT_ATTR] = {
        **out.attrs[UNIT_ATTR],
        "mu_fit": rate_unit,
        "mu_fit_se": rate_unit,
        "mu_from_dt": rate_unit,
    }
    return out
