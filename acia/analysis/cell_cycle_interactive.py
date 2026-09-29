"""Interactive (plotly) twins of the per-cell-cycle plots.

The static matplotlib figures of :mod:`acia.analysis.cell_cycle` are what gets
saved and published. These show the same data for *exploring*: hovering a point
or bar tells which cell it is -- tracklet id, experiment/position/ROI, birth and
division, doubling time, growth rate, fit quality and why it was filtered or
flagged -- so an outlier can be followed up in the tracking video or with
:func:`acia.analysis.cell_inspect.show_cell`. Legend entries toggle the valid,
filtered-out and flagged cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from acia.analysis.cell_cycle import _column, _label, _valid_mask
from acia.analysis.units import UNIT_ATTR, _is_pint_column

if TYPE_CHECKING:
    import plotly.graph_objects as go

#: colours per status (valid / filtered out by the tracklet filters / flagged fit)
STATUS_COLORS = {"valid": "#1f77b4", "flagged": "#d62728", "filtered out": "#a6a6a6"}

#: columns shown on hover when present, in this order
_HOVER_COLUMNS = (
    "experiment",
    "position",
    "roi",
    "start_time",
    "division_time",
    "doubling_time",
    "mu_fit",
    "mu_from_dt",
    "r2",
    "n_points",
    "rejected_by",
    "fit_outlier_reason",
)


def cell_status(df: pd.DataFrame) -> np.ndarray:
    """Per row: ``"filtered out"`` (``valid`` False), ``"flagged"``
    (``fit_outlier`` True) or ``"valid"``."""
    status = np.full(len(df), "valid", dtype=object)
    if "fit_outlier" in df.columns:
        status[np.asarray(df["fit_outlier"], dtype=bool)] = "flagged"
    status[~_valid_mask(df)] = "filtered out"
    return status


def hover_table(df: pd.DataFrame) -> pd.DataFrame:
    """The hover columns as plain values: ``tracklet`` plus what is present."""
    out = pd.DataFrame({"tracklet": df.index.astype(str)}, index=df.index)
    for col in _HOVER_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        if _is_pint_column(series):
            unit = f"{series.pint.units:~P}"
            out[col] = [
                "" if pd.isna(v) else f"{float(v):.3g} {unit}"
                for v in series.values.quantity.magnitude
            ]
        elif pd.api.types.is_float_dtype(series):
            unit = (df.attrs.get(UNIT_ATTR) or {}).get(col, "")
            out[col] = [
                "" if pd.isna(v) else f"{v:.3g}{' ' + unit if unit else ''}"
                for v in series
            ]
        else:
            out[col] = series.fillna("").astype(str)
    return out


def _hovertemplate(columns) -> str:
    lines = ["<b>tracklet %{customdata[0]}</b>"]
    lines += [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(columns) if i > 0]
    return "<br>".join(lines) + "<extra></extra>"


def _figure(title: str, x_title: str, y_title: str) -> go.Figure:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=380,
        legend_title_text="cells",
    )
    return fig


def plot_cycle_dots_interactive(
    df: pd.DataFrame,
    value_col: str = "doubling_time",
    *,
    time_col: str = "division_time",
    value_label: str | None = None,
    title: str = "",
) -> go.Figure:
    """One point per cell at its division time; hover shows the cell."""
    import plotly.graph_objects as go

    t, t_unit = _column(df, time_col)
    v, v_unit = _column(df, value_col)
    hover = hover_table(df)
    status = cell_status(df)
    fig = _figure(
        title, _label(time_col, t_unit), value_label or _label(value_col, v_unit)
    )
    for name, color in STATUS_COLORS.items():
        sel = status == name
        if not sel.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=t[sel],
                y=v[sel],
                mode="markers",
                name=f"{name} ({int(sel.sum())})",
                marker={"color": color, "size": 7, "opacity": 0.8},
                customdata=hover[sel].to_numpy(),
                hovertemplate=_hovertemplate(hover.columns),
            )
        )
    return fig


def plot_cycle_bars_interactive(
    df: pd.DataFrame,
    value_col: str = "doubling_time",
    *,
    start_col: str = "start_time",
    end_col: str = "division_time",
    value_label: str | None = None,
    title: str = "",
) -> go.Figure:
    """One horizontal bar per cell from birth to division; hover shows the cell."""
    import plotly.graph_objects as go

    start, t_unit = _column(df, start_col)
    end, _ = _column(df, end_col, t_unit or None)
    v, v_unit = _column(df, value_col)
    hover = hover_table(df).to_numpy()
    status = cell_status(df)
    fig = _figure(
        title,
        f"time [{t_unit}]" if t_unit else "time",
        value_label or _label(value_col, v_unit),
    )
    for name, color in STATUS_COLORS.items():
        rows = np.nonzero(status == name)[0]
        if not len(rows):
            continue
        xs, ys, custom = [], [], []
        for i in rows:
            xs += [start[i], end[i], None]
            ys += [v[i], v[i], None]
            custom += [hover[i], hover[i], hover[i]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"{name} ({len(rows)})",
                line={"color": color, "width": 3},
                marker={"size": 4, "color": color},
                customdata=np.array(custom, dtype=object),
                hovertemplate=_hovertemplate(hover_table(df).columns),
            )
        )
    return fig


def plot_distributions_interactive(
    groups: dict[str, pd.DataFrame],
    value_col: str,
    *,
    unit: str | None = None,
    value_label: str | None = None,
    title: str = "",
) -> go.Figure:
    """Violins with every cell as a hoverable point, one per group.

    Args:
        groups: ``{label: per-cell table}`` -- e.g. all valid cells and the
            valid, non-flagged ones -- each plotted from its ``value_col``.
        unit: unit to convert pint values to (e.g. ``"1/hour"``).
    """
    import plotly.graph_objects as go

    fig = _figure(title, "", value_label or value_col.replace("_", " "))
    for label, df in groups.items():
        series = df[value_col]
        if _is_pint_column(series):
            q = series.values.quantity
            vals = np.asarray((q.to(unit) if unit else q).magnitude, dtype=float)
        else:
            vals = np.asarray(series, dtype=float)
        ok = ~np.isnan(vals)
        hover = hover_table(df)[ok]
        fig.add_trace(
            go.Violin(
                y=vals[ok],
                name=f"{label} (n={int(ok.sum())})",
                points="all",
                jitter=0.4,
                pointpos=0,
                meanline_visible=True,
                customdata=hover.to_numpy(),
                hovertemplate=_hovertemplate(hover.columns),
            )
        )
    fig.update_layout(showlegend=False)
    return fig


def plot_fit_vs_doubling_interactive(
    rates: pd.DataFrame, *, title: str = ""
) -> go.Figure:
    """Fitted growth rate against ``ln 2 / doubling time`` per cell, coloured by R²."""
    import plotly.graph_objects as go

    fit = rates["mu_fit"].values.quantity.to("1/hour").magnitude
    from_dt = rates["mu_from_dt"].values.quantity.to("1/hour").magnitude
    hover = hover_table(rates)
    status = cell_status(rates)
    fig = _figure(title, "ln 2 / doubling time [1/h]", "growth rate μ (fit) [1/h]")
    fig.add_trace(
        go.Scatter(
            x=from_dt,
            y=fit,
            mode="markers",
            name="cells",
            marker={
                "color": rates["r2"],
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": 1,
                "size": 8,
                "colorbar": {"title": "R²"},
                "line": {
                    "color": [
                        STATUS_COLORS["flagged"] if s == "flagged" else "white"
                        for s in status
                    ],
                    "width": [2 if s == "flagged" else 0.5 for s in status],
                },
            },
            customdata=hover.to_numpy(),
            hovertemplate=_hovertemplate(hover.columns),
        )
    )
    top = float(np.nanmax(np.r_[fit, from_dt, 0])) * 1.1 or 1.0
    fig.add_trace(
        go.Scatter(
            x=[0, top],
            y=[0, top],
            mode="lines",
            name="identity",
            line={"dash": "dash", "color": "grey"},
            hoverinfo="skip",
        )
    )
    return fig
