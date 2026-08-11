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
from acia.analysis.units import UNIT_ATTR

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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
