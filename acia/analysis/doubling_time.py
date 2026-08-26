"""Per-cell doubling time from lineage topology, and its temporal evolution.

This module walks a ``TrackastraTracker``-style ``tracklet_graph`` (one node
per cell cycle, with ``start_frame``/``end_frame`` attributes; edges encode
divisions) to compute, for every *cleanly resolved* division, the real-time
duration between a mother cell's birth and its daughters' birth --
:func:`compute_doubling_times`. :func:`plot_doubling_time_hose` then turns the
resulting per-cell table into a "temporal hose": a mean doubling-time curve
with percentile-bootstrap confidence bands, showing how division timing
evolves over the course of an experiment.

This is unrelated to :func:`acia.analysis.growth_rate.estimate_growth_rate`,
which fits a whole-population exponential curve and derives an aggregate
doubling time from its slope -- nothing here is per-cell or lineage-aware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from acia.analysis.units import UNIT_ATTR

if TYPE_CHECKING:
    import networkx as nx
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from acia.base import ImageSequenceSource

#: default confidence levels for the temporal hose plot's bootstrap bands
DEFAULT_CI_LEVELS = (0.95,)


def compute_doubling_times(
    tracklet_graph: nx.DiGraph, source: ImageSequenceSource
) -> pd.DataFrame:
    """Compute the real-time doubling duration of every cleanly resolved division.

    A tracklet ``n`` qualifies as a "clean division" only if it has exactly
    one identified mother and exactly two daughters:
    ``tracklet_graph.in_degree(n) == 1 and tracklet_graph.out_degree(n) == 2``.
    Tracklets with ``in_degree(n) == 0`` (a root, present at the movie start --
    birth time unknown/left-censored), ``out_degree(n) == 0`` (alive at the
    movie end, exited the field of view, or lost tracking -- division time
    unknown/right-censored), or ``out_degree(n) > 2`` (a tracking/merge
    artifact, not a clean split) are excluded.

    Time is always read via ``source.timepoints[frame_idx]`` -- a per-frame
    pint ``Quantity`` array resolved once at the source level -- never
    computed by hand as ``frame_delta * frame_interval``. This is what makes
    the result correct for both a nominal fixed frame interval and genuinely
    irregular real timestamps.

    Args:
        tracklet_graph: one node per tracklet, keyed by an arbitrary hashable
            label, with ``start_frame``/``end_frame`` int attributes; edges
            ``parent -> child`` encode a division (see
            :func:`acia.tracking.formats.read_ctc_tracklet_graph`).
        source: the time-calibrated image sequence the tracklets were derived
            from (see :attr:`acia.base.ImageSequenceSource.timepoints`).

    Returns:
        A DataFrame indexed by the qualifying tracklet's label (index name
        ``"tracklet"``), with columns ``start_time``, ``end_time`` and
        ``doubling_time`` as ``pint[<unit>]`` columns (unit-safe, matching
        :func:`acia.analysis.attach_units`'s pint representation), where
        ``<unit>`` is ``source.timepoints``'s unit. The unit is also recorded
        in ``df.attrs["units"]`` for consistency with the rest of
        ``acia.analysis``. Empty (no qualifying division) if none is found.

    Raises:
        ValueError: if ``source.timepoints is None`` (no ``frame_interval``
            and no explicit ``timepoints`` were set on ``source``) -- raised
            immediately, before ``tracklet_graph`` is touched at all.
    """
    timepoints = source.timepoints
    if timepoints is None:
        raise ValueError(
            "doubling-time analysis requires time calibration -- source has "
            "neither an explicit frame_interval nor per-frame timepoints set "
            "(see ImageSequenceSource.with_frame_interval / .with_timepoints)"
        )

    unit_str = str(timepoints.units)

    labels: list[Any] = []
    start_times: list[float] = []
    end_times: list[float] = []
    doubling_times: list[float] = []

    n_timepoints = len(timepoints)

    def _check_frame(label: Any, frame: int, what: str) -> None:
        if not 0 <= frame < n_timepoints:
            raise ValueError(
                f"tracklet {label}: {what} frame index {frame} out of bounds "
                f"for {n_timepoints} timepoints"
            )

    for n in tracklet_graph.nodes:
        if tracklet_graph.in_degree(n) != 1 or tracklet_graph.out_degree(n) != 2:
            continue

        node_attrs = tracklet_graph.nodes[n]
        # Both daughters are documented to start at the same division event
        # (same start_frame); pick deterministically by minimum start_frame
        # (tie-broken by node label) rather than relying on the arbitrary
        # order networkx.successors() happens to yield.
        children = list(tracklet_graph.successors(n))
        child = min(children, key=lambda c: (tracklet_graph.nodes[c]["start_frame"], c))

        _check_frame(n, node_attrs["start_frame"], "start_frame")
        _check_frame(n, node_attrs["end_frame"], "end_frame")
        _check_frame(n, tracklet_graph.nodes[child]["start_frame"], "child start_frame")

        start_time = timepoints[node_attrs["start_frame"]]
        end_time = timepoints[node_attrs["end_frame"]]
        doubling_time = (
            timepoints[tracklet_graph.nodes[child]["start_frame"]] - start_time
        )

        if doubling_time.magnitude < 0:
            raise ValueError(
                f"tracklet {n}: negative doubling time ({doubling_time}) -- "
                "malformed tracklet_graph (end_frame < start_frame or child "
                "starts before parent)"
            )

        labels.append(n)
        start_times.append(start_time.to(unit_str).magnitude)
        end_times.append(end_time.to(unit_str).magnitude)
        doubling_times.append(doubling_time.to(unit_str).magnitude)

    pint_dtype = f"pint[{unit_str}]"
    df = pd.DataFrame(
        {
            "start_time": pd.array(start_times, dtype=pint_dtype),
            "end_time": pd.array(end_times, dtype=pint_dtype),
            "doubling_time": pd.array(doubling_times, dtype=pint_dtype),
        },
        index=pd.Index(labels, name="tracklet"),
    )
    df.attrs[UNIT_ATTR] = {
        "start_time": unit_str,
        "end_time": unit_str,
        "doubling_time": unit_str,
    }
    return df


def _compute_hose_bands(
    doubling_times_df: pd.DataFrame,
    time_grid,
    ci_levels,
    min_n: int,
    n_bootstrap: int,
    random_state,
) -> tuple[np.ndarray, np.ndarray, dict[float, tuple[np.ndarray, np.ndarray]], str]:
    """Pure computation behind :func:`plot_doubling_time_hose` (no plotting).

    Factored out from the plotting function so the percentile-bootstrap
    numerics (mean, per-``ci_levels`` bounds, the ``min_n`` NaN-gap rule) are
    directly unit-testable without depending on matplotlib artifact internals
    (e.g. parsing ``fill_between`` polygon vertices).

    Returns:
        ``(t_grid, mean_arr, bands, unit_str)`` where ``t_grid``/``mean_arr``
        are plain-float arrays (magnitudes in ``unit_str``) aligned with
        ``time_grid``, and ``bands`` maps each ``ci_levels`` entry to its
        ``(lo, hi)`` plain-float arrays, aligned the same way. Grid points
        with fewer than ``min_n`` alive-and-qualifying cells are ``NaN`` in
        ``mean_arr`` and every band.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")
    if min_n < 1:
        raise ValueError(f"min_n must be >= 1, got {min_n}.")

    # dedupe while preserving order, so both the returned `bands` dict keys
    # and whatever the caller iterates over (e.g. plot_doubling_time_hose's
    # drawing loop) don't draw/report the same level twice
    ci_levels = list(dict.fromkeys(ci_levels))

    for p in ci_levels:
        if not 0.0 < p < 1.0:
            raise ValueError(f"ci_levels entries must be in (0, 1), got {p}.")

    unit_str = str(time_grid.units)
    t_grid = time_grid.to(unit_str).magnitude

    if len(doubling_times_df) > 0:
        start = doubling_times_df["start_time"].values.quantity.to(unit_str).magnitude
        end = doubling_times_df["end_time"].values.quantity.to(unit_str).magnitude
        values = (
            doubling_times_df["doubling_time"].values.quantity.to(unit_str).magnitude
        )
    else:
        start = end = values = np.array([], dtype=float)

    n_points = len(t_grid)
    mean_arr = np.full(n_points, np.nan)
    bands: dict[float, tuple[np.ndarray, np.ndarray]] = {
        p: (np.full(n_points, np.nan), np.full(n_points, np.nan)) for p in ci_levels
    }

    rng = np.random.default_rng(random_state)

    for i, t in enumerate(t_grid):
        alive = values[(start <= t) & (t <= end)]
        if alive.size < min_n:
            continue

        mean_arr[i] = alive.mean()
        boot_means = rng.choice(
            alive, size=(n_bootstrap, alive.size), replace=True
        ).mean(axis=1)
        for p in ci_levels:
            lo, hi = np.percentile(boot_means, [(1 - p) / 2 * 100, (1 + p) / 2 * 100])
            bands[p][0][i] = lo
            bands[p][1][i] = hi

    return t_grid, mean_arr, bands, unit_str


def plot_doubling_time_hose(
    doubling_times_df: pd.DataFrame,
    source: ImageSequenceSource,
    *,
    time_grid=None,
    ci_levels=DEFAULT_CI_LEVELS,
    min_n: int = 5,
    n_bootstrap: int = 1000,
    random_state=0,
    ax: Axes | None = None,
) -> Figure:
    """Plot mean doubling time evolving over the experiment as a "temporal hose".

    At every time ``t`` in ``time_grid``, a cell contributes its
    ``doubling_time`` to the cross-section at ``t`` for as long as it was
    alive -- i.e. whenever ``start_time <= t <= end_time`` -- not just at the
    single instant it divided. Grid points with fewer than ``min_n``
    alive-and-qualifying cells are left as ``NaN`` (a gap in the plotted line
    and bands, rather than a value computed from too few observations).

    Why a **percentile bootstrap** instead of a parametric mean +/- SEM band:
    a symmetric normal-approximation interval can produce a negative lower
    bound whenever the sample is small or noisy relative to its mean -- which
    happens easily near the ``min_n`` threshold -- and a negative doubling
    time is nonsensical for a duration. A percentile-of-observed-values
    bootstrap is built entirely from resampled means of actually-observed,
    strictly positive values, so it structurally cannot produce a negative
    bound. It also yields every requested ``ci_levels`` percentile pair from
    the *same* bootstrap sample at no extra resampling cost, unlike computing
    each parametric band separately.

    Args:
        doubling_times_df: the output of :func:`compute_doubling_times`
            (``start_time``/``end_time``/``doubling_time`` pint columns).
        source: the time-calibrated image sequence the tracklets were derived
            from; used to default ``time_grid`` to ``source.timepoints``.
        time_grid: pint ``Quantity`` array of times to evaluate the hose at.
            Defaults to ``source.timepoints`` when ``None``.
        ci_levels: confidence levels for the bootstrap bands, e.g. ``(0.95,)``
            (default) or ``(0.5, 0.8, 0.95)`` for a multi-level fan chart. Each
            level is read from the same bootstrap sample at no extra cost.
        min_n: minimum number of alive-and-qualifying cells required at a grid
            point to compute a mean/CI there; below this, the point is
            recorded as ``NaN`` (default ``5``). Must be ``>= 1``. Note:
            ``min_n=1`` produces a mathematically-correct-but-zero-width band
            at any point with exactly one alive-and-qualifying cell, since
            the bootstrap then resamples that same single value every time --
            this is expected behavior reflecting genuinely low sample
            confidence at that point, not a bug.
        n_bootstrap: number of bootstrap resamples per grid point (default
            ``1000``). Must be ``>= 1`` (same zero-width-band caveat as
            ``min_n=1`` applies at ``n_bootstrap=1``).
        random_state: seed (or :class:`numpy.random.Generator`-compatible
            value) for :func:`numpy.random.default_rng`, fixed by default
            (``0``) so repeated calls are reproducible.
        ax: optional matplotlib :class:`~matplotlib.axes.Axes` to draw into.
            When ``None`` a new figure and axes are created.

    Returns:
        A matplotlib :class:`~matplotlib.figure.Figure` with the mean
        doubling-time curve and one shaded ``fill_between`` band per
        ``ci_levels`` entry (multiple levels nest as a fan chart around the
        same mean line). ``NaN`` regions render as natural gaps. The figure
        is not shown (no ``plt.show()``).

    Raises:
        ValueError: if ``time_grid`` is ``None`` and ``source.timepoints`` is
            also ``None`` (no time calibration to default to).
    """
    import matplotlib.pyplot as plt

    if time_grid is None:
        time_grid = source.timepoints
    if time_grid is None:
        raise ValueError(
            "doubling-time analysis requires time calibration -- source has "
            "neither an explicit frame_interval nor per-frame timepoints set "
            "(see ImageSequenceSource.with_frame_interval / .with_timepoints)"
        )

    t_grid, mean_arr, bands, unit_str = _compute_hose_bands(
        doubling_times_df, time_grid, ci_levels, min_n, n_bootstrap, random_state
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = cast("Figure", ax.figure)

    # widest band first (lowest alpha), narrower bands drawn on top (higher
    # alpha) so multiple ci_levels nest visibly as a fan chart. Iterate over
    # `bands` (already deduped by _compute_hose_bands) rather than the raw
    # `ci_levels` argument, so a duplicate entry (e.g. (0.95, 0.95)) doesn't
    # draw the same band/legend entry twice.
    for p in sorted(bands, reverse=True):
        lo, hi = bands[p]
        alpha = 0.15 + 0.35 * (1 - p)
        ax.fill_between(
            t_grid, lo, hi, alpha=alpha, color="tab:blue", label=f"{p:.0%} CI"
        )
    ax.plot(t_grid, mean_arr, color="tab:blue", lw=2, label="mean", zorder=5)

    ax.set_xlabel(f"time [{unit_str}]")
    ax.set_ylabel(f"doubling time [{unit_str}]")
    ax.legend()
    fig.tight_layout()

    return fig
