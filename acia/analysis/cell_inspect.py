"""Look at single cells behind a per-cell statistic.

A distribution plot tells that a cell is unusual; these helpers show *what* the
cell looked like, so an outlier can be judged as biology or as an artifact:

* :func:`cell_trace` -- one cell's size over its life, as a table;
* :func:`plot_cell_traces` -- several cells in one chart (e.g. the cells at the
  0th, 50th and 100th percentile of the growth rate, from
  :func:`acia.analysis.cell_cycle.pick_percentile_cells`), with each cell's
  fitted exponential;
* :func:`show_cell` -- a filmstrip of one cell's image crops with its outline,
  above its size curve.

A cell is identified by its tracklet id: the ``label`` of its contours in the
tracked overlay and the index of ``doubling_times.csv`` /
``cell_growth_rates.csv`` (and the red number in the tracking video).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from acia.analysis.growth_rate import _cell_size

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _contours_of(overlay, tracklet_id) -> list:
    conts = [c for c in overlay.contours if c.label == tracklet_id]
    if not conts:
        raise KeyError(f"no contour with label (tracklet id) {tracklet_id!r}")
    return sorted(conts, key=lambda c: c.frame)


def cell_trace(overlay, tracklet_id, source, *, measure: str = "area") -> pd.DataFrame:
    """One cell's size over its life.

    Args:
        overlay: the tracked overlay (contour ``label`` = tracklet id).
        tracklet_id: the cell.
        source: the time-calibrated image sequence (for ``timepoints``).
        measure: ``"area"`` (px²) or ``"length"`` (px, long side of the minimal
            rotated rectangle).

    Returns:
        Columns ``frame``, ``time``, ``time_since_birth`` (in the source's time
        unit, recorded in ``attrs["time_unit"]``) and ``size``.
    """
    timepoints = source.timepoints
    if timepoints is None:
        raise ValueError("cell traces need a time-calibrated source")
    conts = _contours_of(overlay, tracklet_id)
    t = np.asarray(timepoints.magnitude, dtype=float)
    frames = [int(c.frame) for c in conts]
    df = pd.DataFrame(
        {
            "frame": frames,
            "time": t[frames],
            "size": [_cell_size(c, measure) for c in conts],
        }
    )
    df["time_since_birth"] = df["time"] - df["time"].iloc[0]
    df.attrs["time_unit"] = str(timepoints.units)
    df.attrs["measure"] = measure
    return df


def _plain_log_axis(ax) -> None:
    """Log y axis labelled with plain numbers (50, 60, ...) instead of 5×10¹."""
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _fit_line(trace: pd.DataFrame) -> tuple[float, float] | None:
    ok = np.isfinite(trace["size"]) & (trace["size"] > 0)
    if ok.sum() < 2:
        return None
    slope, intercept = np.polyfit(
        trace.loc[ok, "time_since_birth"], np.log(trace.loc[ok, "size"]), 1
    )
    return float(slope), float(intercept)


def plot_cell_traces(
    overlay,
    picks,
    source,
    *,
    rates: pd.DataFrame | None = None,
    measure: str = "area",
    ax: Axes | None = None,
    log: bool = True,
) -> Figure:
    """Several cells' size over time since birth, in one chart.

    Args:
        overlay: the tracked overlay.
        picks: tracklet ids, or the table of
            :func:`~acia.analysis.cell_cycle.pick_percentile_cells` (its
            ``percentile`` then labels the curves, e.g. "P50").
        source: the time-calibrated image sequence.
        rates: optional :func:`~acia.analysis.growth_rate.compute_cell_growth_rates`
            table; adds μ and the doubling time to each label.
        measure: ``"area"`` or ``"length"``.
        ax: axes to draw into; a new figure is created when ``None``.
        log: log-scaled y axis, where exponential growth is a straight line.

    Returns:
        The matplotlib figure: dots are the measurements, the dashed line the
        cell's exponential fit.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    else:
        fig = cast("Figure", ax.figure)

    items: list[tuple[object, str | None]]
    if isinstance(picks, pd.DataFrame):
        items = [(r.tracklet, f"P{r.percentile:g}") for r in picks.itertuples()]
    else:
        items = [(tid, None) for tid in picks]

    unit = ""
    for i, (tid, tag) in enumerate(items):
        trace = cell_trace(overlay, tid, source, measure=measure)
        unit = trace.attrs["time_unit"]
        color = plt.get_cmap("tab10")(i % 10)
        parts = [tag] if tag else []
        parts.append(f"#{tid}")
        if rates is not None and tid in rates.index:
            row = rates.loc[tid]
            mu = row["mu_fit"]
            dt = row["doubling_time"]
            mu_v = float(getattr(mu, "magnitude", mu))
            dt_v = float(getattr(dt, "magnitude", dt))
            parts.append(f"μ {mu_v:.2f}/h")
            dt_unit = f"{dt.units:~P}" if hasattr(dt, "units") else unit
            parts.append(f"DT {dt_v:.0f} {dt_unit}")
        label = " · ".join(parts)
        ax.plot(
            trace["time_since_birth"],
            trace["size"],
            "o",
            color=color,
            ms=4,
            label=label,
        )
        fit = _fit_line(trace)
        if fit is not None:
            x = np.linspace(0, trace["time_since_birth"].max(), 50)
            ax.plot(x, np.exp(fit[1] + fit[0] * x), "--", color=color, lw=1)

    if log:
        _plain_log_axis(ax)
    ax.set_xlabel(f"time since birth [{unit}]")
    ax.set_ylabel("area [px²]" if measure == "area" else "length [px]")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    else:  # nothing to show (e.g. no cell with a fit): say so instead of a blank axes
        ax.text(0.5, 0.5, "no cells", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def show_cell(
    overlay,
    source,
    tracklet_id,
    *,
    cycles: pd.DataFrame | None = None,
    measure: str = "area",
    max_frames: int = 12,
    pad: int = 8,
    channel: int = 0,
) -> Figure:
    """A filmstrip of one cell over its life, above its size curve.

    Each panel is the cell's neighbourhood (the box around all its outlines,
    plus ``pad`` px) on one frame, with the cell's outline in red, so a merged
    neighbour, a swapped identity or a missed division is visible at a glance.

    Args:
        overlay: the tracked overlay (contour ``label`` = tracklet id).
        source: the image sequence the overlay belongs to.
        tracklet_id: the cell, e.g. from the hover of an interactive plot.
        cycles: optional per-cell table (doubling times / growth rates) whose
            row for this cell is summarised in the title.
        measure: size measure of the curve.
        max_frames: at most this many panels, spread evenly over the cell's life.
        pad: context around the cell, in px.
        channel: image channel to show.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    conts = _contours_of(overlay, tracklet_id)
    coords = np.concatenate(
        [np.asarray(c.coordinates, dtype=float).reshape(-1, 2) for c in conts]
    )
    first = np.asarray(source.get_frame(int(conts[0].frame)).raw)
    height, width = first.shape[:2]
    x0 = max(0, int(np.floor(coords[:, 0].min())) - pad)
    x1 = min(width, int(np.ceil(coords[:, 0].max())) + pad + 1)
    y0 = max(0, int(np.floor(coords[:, 1].min())) - pad)
    y1 = min(height, int(np.ceil(coords[:, 1].max())) + pad + 1)

    pick = np.unique(
        np.linspace(0, len(conts) - 1, min(max_frames, len(conts))).round()
    )
    shown = [conts[int(i)] for i in pick]

    fig = plt.figure(figsize=(max(6.0, 1.2 * len(shown)), 4.5))
    grid = fig.add_gridspec(2, len(shown), height_ratios=[1, 1.4])
    times = (
        np.asarray(source.timepoints.magnitude)
        if source.timepoints is not None
        else None
    )
    for i, cont in enumerate(shown):
        ax = fig.add_subplot(grid[0, i])
        raw = np.asarray(source.get_frame(int(cont.frame)).raw)
        plane = raw[..., channel] if raw.ndim == 3 else raw
        crop = plane[y0:y1, x0:x1].astype(float)
        lo, hi = np.percentile(crop, [1, 99]) if crop.size else (0, 1)
        ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1)
        xy = np.asarray(cont.coordinates, dtype=float).reshape(-1, 2)
        ax.plot(
            np.r_[xy[:, 0], xy[:1, 0]] - x0, np.r_[xy[:, 1], xy[:1, 1]] - y0, "r-", lw=1
        )
        stamp = f"f{int(cont.frame)}"
        if times is not None:
            stamp += f"\n{times[int(cont.frame)]:.0f}"
        ax.set_title(stamp, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_trace = fig.add_subplot(grid[1, :])
    if source.timepoints is not None:
        trace = cell_trace(overlay, tracklet_id, source, measure=measure)
        ax_trace.plot(trace["time_since_birth"], trace["size"], "o-", ms=3)
        fit = _fit_line(trace)
        if fit is not None:
            x = trace["time_since_birth"].to_numpy()
            ax_trace.plot(x, np.exp(fit[1] + fit[0] * x), "--", color="0.4", lw=1)
        _plain_log_axis(ax_trace)
        ax_trace.set_xlabel(f"time since birth [{trace.attrs['time_unit']}]")
        ax_trace.set_ylabel("area [px²]" if measure == "area" else "length [px]")

    title = f"tracklet #{tracklet_id}"
    if cycles is not None and tracklet_id in cycles.index:
        row = cycles.loc[tracklet_id]
        extras = []
        for col, fmt in (
            ("doubling_time", "DT {:.0f}"),
            ("mu_fit", "μ fit {:.2f}"),
            ("mu_from_dt", "ln2/DT {:.2f}"),
            ("r2", "R² {:.2f}"),
        ):
            if col in row.index and pd.notna(row[col]):
                extras.append(
                    fmt.format(float(getattr(row[col], "magnitude", row[col])))
                )
        for col in ("rejected_by", "fit_outlier_reason"):
            if col in row.index and isinstance(row[col], str) and row[col]:
                extras.append(row[col])
        if extras:
            title += " · " + " · ".join(extras)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
