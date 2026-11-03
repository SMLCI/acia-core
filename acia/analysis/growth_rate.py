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


#: default threshold of the size-jump flags: a frame-to-frame size ratio below
#: this (or above its inverse) marks a likely missed division (or merged cells)
DEFAULT_SIZE_JUMP_RATIO = 0.65

#: flag reasons of :func:`compute_cell_growth_rates`, in the order they are tested
FLAG_REASONS = (
    "too few points",
    "size drop (missed division?)",
    "size jump (merged cells?)",
    "poor fit",
    "disagrees with doubling time",
)


@dataclass
class _SizeFit:
    mu: float = np.nan
    se: float = np.nan
    ci_lo: float = np.nan
    ci_hi: float = np.nan
    r2: float = np.nan
    n_points: int = 0
    min_ratio: float = np.nan
    max_ratio: float = np.nan
    t_first: float = np.nan
    t_last: float = np.nan


def _fit_log_size(
    t: np.ndarray, y: np.ndarray, *, min_points: int, ci_level: float, se_type: str
) -> _SizeFit:
    """OLS of ``log(size)`` on time for one tracklet (sizes > 0 and finite only)."""
    ok = np.isfinite(y) & (y > 0)
    t, y = t[ok], y[ok]
    out = _SizeFit(n_points=int(t.size))
    if t.size:
        out.t_first, out.t_last = float(t[0]), float(t[-1])
    if t.size >= 2:
        ratios = y[1:] / y[:-1]
        out.min_ratio, out.max_ratio = float(ratios.min()), float(ratios.max())
    if t.size < max(min_points, 2):
        return out
    design = sm.add_constant(t, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(np.log(y), design).fit()
        if se_type == "hac" and t.size >= 4:
            model = model.get_robustcov_results(
                cov_type="HAC", maxlags=max(1, int(np.floor(t.size ** (1 / 3))))
            )
    params = np.asarray(model.params, float)
    out.mu = float(params[1])
    out.r2 = float(model.rsquared)
    if t.size >= 3:
        out.se = float(np.asarray(model.bse, float)[1])
        lo, hi = np.asarray(model.conf_int(alpha=1 - ci_level), float)[1]
        out.ci_lo, out.ci_hi = float(lo), float(hi)
    return out


def fit_tracklet_sizes(
    overlay,
    tracklets,
    source: ImageSequenceSource,
    *,
    measure: Literal["area", "length"] = "area",
    rate_unit: str = "1/hour",
    min_points: int = 4,
    ci_level: float = 0.95,
    se_type: Literal["ols", "hac"] = "ols",
) -> pd.DataFrame:
    """Fit ``size(t) = size_0 * exp(mu * t)`` for each tracklet, dividing or not.

    The building block of :func:`compute_cell_growth_rates`, usable on any set
    of tracklets -- including cells that never divide, where it is the only
    growth readout there is (a non-growing cell measures ``mu`` near 0, with its
    own confidence interval; nothing is imputed).

    Args:
        overlay: the tracked overlay; each contour's ``label`` is its tracklet id.
        tracklets: the tracklet ids to fit (any iterable, e.g. ``df.index``).
        source: the time-calibrated image sequence (for ``timepoints``).
        measure: ``"area"`` or ``"length"`` as the cell size.
        rate_unit: unit of the rates.
        min_points: minimum number of observations for a fit.
        ci_level: confidence level of ``mu_fit_ci_lo``/``mu_fit_ci_hi``.
        se_type: ``"ols"`` (plain standard errors) or ``"hac"`` (Newey-West,
            robust to the autocorrelated segmentation errors of consecutive
            frames; wider, more honest intervals).

    Returns:
        One row per tracklet (index ``"tracklet"``): ``mu_fit``, ``mu_fit_se``,
        ``mu_fit_ci_lo``, ``mu_fit_ci_hi`` (``pint[rate_unit]``), ``r2``,
        ``n_points``, ``size_min_ratio``/``size_max_ratio`` (smallest/largest
        frame-to-frame size ratio), ``observed`` (first to last observation,
        in ``1/rate_unit``) and ``size_change`` (fitted relative size change
        over the observation, ``exp(mu * observed) - 1``).
    """
    timepoints = source.timepoints
    if timepoints is None:
        raise ValueError(
            "per-cell growth rates require time calibration -- the source has "
            "neither a frame_interval nor per-frame timepoints"
        )
    if se_type not in ("ols", "hac"):
        raise ValueError(f"se_type must be 'ols' or 'hac', got {se_type!r}")
    time_unit = str((1 / Q_(1.0, rate_unit)).units)
    t_all = np.asarray(timepoints.to(time_unit).magnitude, dtype=float)

    by_label: dict = {}
    for cont in overlay.contours:
        by_label.setdefault(cont.label, []).append(cont)

    labels = list(tracklets)
    fits = []
    for tracklet in labels:
        conts = sorted(by_label.get(tracklet, []), key=lambda c: c.frame)
        t = np.array([t_all[int(c.frame)] for c in conts], dtype=float)
        y = np.array([_cell_size(c, measure) for c in conts], dtype=float)
        fits.append(
            _fit_log_size(
                t, y, min_points=min_points, ci_level=ci_level, se_type=se_type
            )
        )

    def col(name):
        return np.asarray([getattr(f, name) for f in fits], dtype=float)

    pint_dtype = f"pint[{rate_unit}]"
    mu = col("mu")
    observed = col("t_last") - col("t_first")
    with np.errstate(over="ignore", invalid="ignore"):
        size_change = np.expm1(mu * observed)
    df = pd.DataFrame(
        {
            "mu_fit": pd.array(mu, dtype=pint_dtype),
            "mu_fit_se": pd.array(col("se"), dtype=pint_dtype),
            "mu_fit_ci_lo": pd.array(col("ci_lo"), dtype=pint_dtype),
            "mu_fit_ci_hi": pd.array(col("ci_hi"), dtype=pint_dtype),
            "r2": col("r2"),
            "n_points": np.asarray([f.n_points for f in fits], dtype=int),
            "size_min_ratio": col("min_ratio"),
            "size_max_ratio": col("max_ratio"),
            "observed": pd.array(observed, dtype=f"pint[{time_unit}]"),
            "size_change": size_change,
        },
        index=pd.Index(labels, name="tracklet"),
    )
    df.attrs[UNIT_ATTR] = {
        "mu_fit": rate_unit,
        "mu_fit_se": rate_unit,
        "mu_fit_ci_lo": rate_unit,
        "mu_fit_ci_hi": rate_unit,
        "observed": time_unit,
    }
    return df


def _magnitudes(df: pd.DataFrame, col: str, unit: str) -> np.ndarray:
    values = df[col]
    if _is_pint_column(values):
        return np.asarray(values.values.quantity.to(unit).magnitude, float)
    src_unit = (df.attrs.get(UNIT_ATTR) or {}).get(col) or unit
    return np.asarray(Q_(np.asarray(values, float), src_unit).to(unit).magnitude)


def compute_cell_growth_rates(
    overlay,
    cycles: pd.DataFrame,
    source: ImageSequenceSource,
    *,
    measure: Literal["area", "length"] = "area",
    rate_unit: str = "1/hour",
    min_points: int = 4,
    agreement_z: float | None = 2.0,
    size_jump_ratio: float | None = DEFAULT_SIZE_JUMP_RATIO,
    r2_min: float | None = None,
    rel_tol: float | None = None,
    ci_level: float = 0.95,
    se_type: Literal["ols", "hac"] = "ols",
) -> pd.DataFrame:
    """Per-cell exponential growth rate, fitted and derived from the doubling time.

    For every cell cycle in ``cycles`` (one row per tracklet, as returned by
    :func:`acia.analysis.doubling_time.compute_doubling_times`), the cell's size
    over its life is fitted as ``size(t) = size_0 * exp(mu * t)`` -- an OLS of
    the **natural** log of size on time (:func:`fit_tracklet_sizes`). A constant
    pixel calibration cancels in the slope, so sizes are taken in pixels.

    Two growth rates are reported side by side, both natural-log rates in
    ``rate_unit`` and therefore directly comparable:

    * ``mu_fit`` -- from the size fit, with its standard error and confidence
      interval;
    * ``mu_from_dt`` -- ``ln(2) / doubling_time``, i.e. what the cell's cycle
      duration implies if it doubled exactly once. Its uncertainty
      ``mu_from_dt_se`` comes from the frame timing: birth and division are each
      only known to within one frame interval, so the doubling time is uncertain
      by ``doubling_time_sd`` (``Δ/√6`` for equal frames), propagated as
      ``ln2 * sd / DT²``. ``mu_from_dt_lo``/``_hi`` are ``ln2`` over the widest
      possible doubling times.

    **How cells are flagged** (``fit_outlier`` / ``fit_outlier_reason``; the
    first matching reason wins, see :data:`FLAG_REASONS`):

    1. ``"too few points"`` -- fewer than ``min_points`` observations.
    2. ``"size drop (missed division?)"`` -- the size falls below
       ``size_jump_ratio`` times its previous frame (default 0.65, i.e. roughly
       halves): usually a division the tracker did not see.
    3. ``"size jump (merged cells?)"`` -- the size grows by more than
       ``1 / size_jump_ratio`` between two frames: usually two cells merged by
       the segmentation.
    4. ``"poor fit"`` -- ``r2 < r2_min`` (off by default: one cycle spans only
       ``ln 2`` in log-size, so R² mostly measures ordinary segmentation noise).
    5. ``"disagrees with doubling time"`` -- the precision-scaled difference
       ``agreement_z = |mu_fit - mu_from_dt| / sqrt(mu_fit_se² + mu_from_dt_se²)``
       exceeds ``agreement_z`` (default 2). This is the difference in units of
       the two readouts' combined measurement uncertainty: ``z <= 2`` means
       they agree within what frame timing and fit noise explain; ``z > 2``
       means something about the cell is off (tracking, segmentation, or
       growth that is not exponential). Larger values are more lenient. A cell
       with a very noisy fit has a large ``mu_fit_se`` and is rarely flagged --
       check ``mu_fit_se`` for those. With ``rel_tol`` set, the old fixed
       relative-tolerance test ``|mu_fit - mu_from_dt| / mu_from_dt > rel_tol``
       is applied as well.

    Args:
        overlay: the tracked overlay; each contour's ``label`` is its tracklet
            id (the index of ``cycles``).
        cycles: per-cycle table with at least ``doubling_time``; its time
            columns and any ``valid``/``rejected_by`` columns are carried over.
            With ``doubling_time_sd`` (from ``compute_doubling_times``) the
            timing uncertainty is per cell; without it, it is estimated from
            the median frame interval.
        source: the time-calibrated image sequence (for ``timepoints``).
        measure: ``"area"`` (default) or ``"length"`` (long side of the minimal
            rotated rectangle) as the cell size.
        rate_unit: unit of the growth rates (default ``"1/hour"``).
        min_points: minimum number of observations for a fit.
        agreement_z: threshold of the precision-scaled agreement test
            (``None`` switches it off).
        size_jump_ratio: threshold of the size drop/jump flags (``None`` off).
        r2_min: optional R² threshold (``None``, the default, switches it off).
        rel_tol: optional fixed relative tolerance (``None`` off).
        ci_level: confidence level of the fit's interval.
        se_type: ``"ols"`` or ``"hac"`` standard errors (see
            :func:`fit_tracklet_sizes`).

    Returns:
        ``cycles`` (copied) plus the :func:`fit_tracklet_sizes` columns,
        ``mu_from_dt``, ``mu_from_dt_se``, ``mu_from_dt_lo``, ``mu_from_dt_hi``
        (``pint[rate_unit]``), ``agreement_z``, ``fit_outlier`` and
        ``fit_outlier_reason``, indexed by tracklet. The thresholds used are in
        ``attrs["flag_thresholds"]`` (see :func:`flag_report`).

    Raises:
        ValueError: if ``source`` has no time calibration.
    """
    fits = fit_tracklet_sizes(
        overlay,
        cycles.index,
        source,
        measure=measure,
        rate_unit=rate_unit,
        min_points=min_points,
        ci_level=ci_level,
        se_type=se_type,
    )
    time_unit = str((1 / Q_(1.0, rate_unit)).units)
    timepoints = source.timepoints
    assert timepoints is not None  # checked by fit_tracklet_sizes

    out = cycles.copy()
    out.attrs = dict(cycles.attrs)
    doubling = _magnitudes(cycles, "doubling_time", time_unit)
    if "doubling_time_sd" in cycles.columns:
        dt_sd = _magnitudes(cycles, "doubling_time_sd", time_unit)
    else:
        steps = np.diff(np.asarray(timepoints.to(time_unit).magnitude, float))
        step = float(np.median(steps)) if steps.size else np.nan
        dt_sd = np.full(len(cycles), step / np.sqrt(6))
    dt_lo = (
        _magnitudes(cycles, "doubling_time_lo", time_unit)
        if "doubling_time_lo" in cycles.columns
        else doubling - np.sqrt(6) * dt_sd
    )
    dt_hi = (
        _magnitudes(cycles, "doubling_time_hi", time_unit)
        if "doubling_time_hi" in cycles.columns
        else doubling + np.sqrt(6) * dt_sd
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        from_dt = np.where(doubling > 0, np.log(2) / doubling, np.nan)
        from_dt_se = np.where(doubling > 0, np.log(2) * dt_sd / doubling**2, np.nan)
        from_dt_lo = np.where(dt_hi > 0, np.log(2) / dt_hi, np.nan)
        from_dt_hi = np.where(dt_lo > 0, np.log(2) / dt_lo, np.inf)

    for name in fits.columns:
        out[name] = fits[name].values
    pint_dtype = f"pint[{rate_unit}]"
    out["mu_from_dt"] = pd.array(from_dt, dtype=pint_dtype)
    out["mu_from_dt_se"] = pd.array(from_dt_se, dtype=pint_dtype)
    out["mu_from_dt_lo"] = pd.array(from_dt_lo, dtype=pint_dtype)
    out["mu_from_dt_hi"] = pd.array(from_dt_hi, dtype=pint_dtype)

    mu_arr = _magnitudes(fits, "mu_fit", rate_unit)
    mu_se = _magnitudes(fits, "mu_fit_se", rate_unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(mu_arr - from_dt) / np.sqrt(
            np.nan_to_num(mu_se) ** 2 + np.nan_to_num(from_dt_se) ** 2
        )
        rel_diff = np.abs(mu_arr - from_dt) / from_dt
    z = np.where(np.isfinite(mu_arr) & np.isfinite(from_dt), z, np.nan)
    out["agreement_z"] = z

    r2_arr = fits["r2"].to_numpy(float)
    min_ratio = fits["size_min_ratio"].to_numpy(float)
    max_ratio = fits["size_max_ratio"].to_numpy(float)
    reasons = np.full(len(out), None, dtype=object)

    def flag(mask, reason):
        reasons[(reasons == None) & mask] = reason  # noqa: E711

    flag(np.isnan(mu_arr), "too few points")
    if size_jump_ratio is not None:
        flag(min_ratio < size_jump_ratio, "size drop (missed division?)")
        flag(max_ratio > 1 / size_jump_ratio, "size jump (merged cells?)")
    if r2_min is not None:
        flag(r2_arr < r2_min, "poor fit")
    if agreement_z is not None:
        flag(z > agreement_z, "disagrees with doubling time")
    if rel_tol is not None:
        flag(rel_diff > rel_tol, "disagrees with doubling time")
    out["fit_outlier"] = reasons != None  # noqa: E711
    out["fit_outlier_reason"] = reasons

    out.attrs[UNIT_ATTR] = {
        **(out.attrs.get(UNIT_ATTR) or {}),
        **fits.attrs[UNIT_ATTR],
        "mu_from_dt": rate_unit,
        "mu_from_dt_se": rate_unit,
        "mu_from_dt_lo": rate_unit,
        "mu_from_dt_hi": rate_unit,
    }
    out.attrs["flag_thresholds"] = {
        "min_points": min_points,
        "size_jump_ratio": size_jump_ratio,
        "r2_min": r2_min,
        "agreement_z": agreement_z,
        "rel_tol": rel_tol,
        "ci_level": ci_level,
        "se_type": se_type,
        "measure": measure,
    }
    return out


def flag_report(rates: pd.DataFrame) -> pd.DataFrame:
    """Count flagged cells per reason, in the order the flags are tested.

    The table to read next to the thresholds (``rates.attrs["flag_thresholds"]``,
    copied to the result's ``attrs``): change a threshold, re-run, and the
    counts show the effect at once. Every reason is listed, also with 0 cells.

    Returns:
        A DataFrame with columns ``reason`` and ``n`` -- one row per
        :data:`FLAG_REASONS` entry, then ``"not flagged"`` and ``"total"``.
    """
    reasons = rates.get("fit_outlier_reason")
    counts = (
        reasons.dropna().value_counts() if reasons is not None else pd.Series(dtype=int)
    )
    rows = [(r, int(counts.get(r, 0))) for r in FLAG_REASONS]
    n_flagged = sum(n for _, n in rows)
    rows += [("not flagged", len(rates) - n_flagged), ("total", len(rates))]
    report = pd.DataFrame(rows, columns=["reason", "n"])
    report.attrs["flag_thresholds"] = dict(rates.attrs.get("flag_thresholds") or {})
    return report
