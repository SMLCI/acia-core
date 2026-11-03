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

from acia.analysis.units import UNIT_ATTR, _is_pint_column

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
        ``"tracklet"``), with columns ``start_time`` (birth: first frame),
        ``end_time`` (last frame the cell is seen), ``division_time`` (the
        daughters' first frame) and ``doubling_time`` (``division_time -
        start_time``, i.e. birth to birth) as ``pint[<unit>]`` columns (unit-safe, matching
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
    division_times: list[float] = []
    doubling_times: list[float] = []
    dt_lo: list[float] = []
    dt_hi: list[float] = []
    dt_sd: list[float] = []
    frame_gaps: list[int] = []

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
        # the division is dated by the daughters' first frame: a cell seen on
        # frames s..e lives e - s + 1 frames, so its cycle runs to the next
        # birth, one interval past its own last frame (end_time)
        division_time = timepoints[tracklet_graph.nodes[child]["start_frame"]]
        doubling_time = division_time - start_time

        if doubling_time.magnitude < 0:
            raise ValueError(
                f"tracklet {n}: negative doubling time ({doubling_time}) -- "
                "malformed tracklet_graph (end_frame < start_frame or child "
                "starts before parent)"
            )

        # Frame timing: the true birth lies after the frame before the first
        # observation, and the true division after the mother's last frame --
        # each end is only known to within one frame interval. The widest
        # possible cycle runs from the frame before birth to the daughters'
        # first frame, the narrowest from birth to the frame before that.
        s_frame = node_attrs["start_frame"]
        c_frame = tracklet_graph.nodes[child]["start_frame"]
        before_birth = timepoints[s_frame - 1] if s_frame > 0 else start_time
        before_division = timepoints[c_frame - 1] if c_frame > 0 else division_time
        birth_step = (start_time - before_birth).to(unit_str).magnitude
        division_step = (division_time - before_division).to(unit_str).magnitude
        dt_lo.append((before_division - start_time).to(unit_str).magnitude)
        dt_hi.append((division_time - before_birth).to(unit_str).magnitude)
        # each end is uniform over its interval: variance step**2 / 12
        dt_sd.append(float(np.sqrt((birth_step**2 + division_step**2) / 12)))
        frame_gaps.append(int(c_frame - node_attrs["end_frame"] - 1))

        labels.append(n)
        start_times.append(start_time.to(unit_str).magnitude)
        end_times.append(end_time.to(unit_str).magnitude)
        division_times.append(division_time.to(unit_str).magnitude)
        doubling_times.append(doubling_time.to(unit_str).magnitude)

    pint_dtype = f"pint[{unit_str}]"
    df = pd.DataFrame(
        {
            "start_time": pd.array(start_times, dtype=pint_dtype),
            "end_time": pd.array(end_times, dtype=pint_dtype),
            "division_time": pd.array(division_times, dtype=pint_dtype),
            "doubling_time": pd.array(doubling_times, dtype=pint_dtype),
            "doubling_time_lo": pd.array(dt_lo, dtype=pint_dtype),
            "doubling_time_hi": pd.array(dt_hi, dtype=pint_dtype),
            "doubling_time_sd": pd.array(dt_sd, dtype=pint_dtype),
            "frame_gap": np.asarray(frame_gaps, dtype=int),
        },
        index=pd.Index(labels, name="tracklet"),
    )
    df.attrs[UNIT_ATTR] = {
        col: unit_str
        for col in (
            "start_time",
            "end_time",
            "division_time",
            "doubling_time",
            "doubling_time_lo",
            "doubling_time_hi",
            "doubling_time_sd",
        )
    }
    return df


#: tracklet status values of :func:`classify_tracklets`
TRACKLET_STATUSES = ("divided", "censored", "excluded")


def classify_tracklets(
    tracklet_graph: nx.DiGraph, source: ImageSequenceSource
) -> pd.DataFrame:
    """Say for *every* tracklet whether it yields a doubling time, and why not.

    :func:`compute_doubling_times` keeps only clean divisions; this is the
    complete account of the rest, so that cells which never divide do not
    silently disappear. Each tracklet gets a ``status`` and a ``reason``:

    * ``divided`` -- one mother, two daughters: a doubling time exists
      (``reason`` is ``"clean division"``, or ``"clean division (frame gap)"``
      when the daughters appear more than one frame after the mother's last
      frame).
    * ``censored`` -- the cycle was not seen to its end or from its start, so
      it says nothing about growth by itself: ``"alive at movie end"``,
      ``"lost (no division seen)"`` (ends before the movie does without
      daughters: left the field, or the tracker lost it) or
      ``"present at start (birth unknown)"``.
    * ``excluded`` -- a topology a doubling time cannot be read from:
      ``"one daughter (other lost)"``, ``"more than two daughters"`` or
      ``"more than one mother"``.

    Args:
        tracklet_graph: as for :func:`compute_doubling_times`.
        source: the time-calibrated image sequence.

    Returns:
        One row per tracklet (index ``"tracklet"``): ``start_frame``,
        ``end_frame``, ``n_mothers``, ``n_daughters``, ``observed`` (first to
        last frame, ``pint[<time unit>]``), ``status`` and ``reason``.
    """
    timepoints = source.timepoints
    if timepoints is None:
        raise ValueError(
            "classifying tracklets requires time calibration -- source has "
            "neither an explicit frame_interval nor per-frame timepoints set"
        )
    unit_str = str(timepoints.units)
    last_frame = len(timepoints) - 1
    rows = []
    for n in tracklet_graph.nodes:
        attrs = tracklet_graph.nodes[n]
        s, e = int(attrs["start_frame"]), int(attrs["end_frame"])
        n_in = tracklet_graph.in_degree(n)
        n_out = tracklet_graph.out_degree(n)
        if n_out == 0:
            status = "censored"
            reason = (
                "alive at movie end" if e >= last_frame else "lost (no division seen)"
            )
        elif n_out == 1:
            status, reason = "excluded", "one daughter (other lost)"
        elif n_out > 2:
            status, reason = "excluded", "more than two daughters"
        elif n_in == 0:
            status, reason = "censored", "present at start (birth unknown)"
        elif n_in > 1:
            status, reason = "excluded", "more than one mother"
        else:
            child_start = min(
                tracklet_graph.nodes[c]["start_frame"]
                for c in tracklet_graph.successors(n)
            )
            status = "divided"
            reason = (
                "clean division"
                if child_start == e + 1
                else "clean division (frame gap)"
            )
        e_clamped = min(max(e, 0), last_frame)
        s_clamped = min(max(s, 0), last_frame)
        observed = (
            (timepoints[e_clamped] - timepoints[s_clamped]).to(unit_str).magnitude
        )
        rows.append((n, s, e, n_in, n_out, observed, status, reason))
    df = pd.DataFrame(
        rows,
        columns=[
            "tracklet",
            "start_frame",
            "end_frame",
            "n_mothers",
            "n_daughters",
            "observed",
            "status",
            "reason",
        ],
    ).set_index("tracklet")
    df["observed"] = pd.array(df["observed"].to_numpy(float), dtype=f"pint[{unit_str}]")
    df.attrs[UNIT_ATTR] = {"observed": unit_str}
    return df


def division_census(
    status: pd.DataFrame,
    sizes: pd.DataFrame | None = None,
    *,
    arrest_min_observation: Any = None,
    arrest_max_growth: float = 0.2,
) -> dict[str, Any]:
    """How many cells divided, were cut off, or stopped growing -- one dict.

    A report, not an analysis input: it makes zero growth visible without
    imputing anything. Cells that were not seen to divide are *not* counted as
    "no growth" by default -- most of them were simply cut off by the movie's
    end or lost by the tracker. Only a cell that was **watched long enough**
    without dividing counts towards the denominator of ``fraction_divided``,
    and only such a cell whose size did not grow counts as ``n_arrested``.

    Args:
        status: :func:`classify_tracklets` output.
        sizes: optional :func:`acia.analysis.growth_rate.fit_tracklet_sizes`
            output for (at least) the non-dividing tracklets; needed for
            ``n_arrested``.
        arrest_min_observation: how long a non-dividing cell must have been
            observed to count (``"120 min"``, a pint quantity, or a number in
            the unit of ``status["observed"]``). ``None`` disables the
            long-observation counts.
        arrest_max_growth: a long-observed, non-dividing cell is ``arrested``
            when its fitted relative size change over the observation is below
            this (default 0.2 = 20 %) or its growth-rate interval includes 0.

    Returns:
        ``n_tracklets``, ``n_divided``, ``n_censored``, ``n_excluded``,
        ``n_long_not_divided``, ``n_arrested``, ``fraction_divided``
        (``n_divided / (n_divided + n_long_not_divided)``) and the settings.
        Undefined values are ``None`` -- never NaN or infinity, so the dict is
        valid JSON.
    """
    from acia import Q_

    counts = status["status"].value_counts()
    out: dict[str, Any] = {
        "n_tracklets": int(len(status)),
        "n_divided": int(counts.get("divided", 0)),
        "n_censored": int(counts.get("censored", 0)),
        "n_excluded": int(counts.get("excluded", 0)),
        "n_long_not_divided": None,
        "n_arrested": None,
        "fraction_divided": None,
        "arrest_min_observation": None
        if arrest_min_observation is None
        else str(arrest_min_observation),
        "arrest_max_growth": arrest_max_growth,
    }
    if arrest_min_observation is None:
        return out
    unit = _unit_of(status, "observed")
    if isinstance(arrest_min_observation, (int, float)):
        threshold = float(arrest_min_observation)
    else:
        threshold = float(Q_(arrest_min_observation).to(unit).magnitude)
    observed = (
        np.asarray(status["observed"].values.quantity.to(unit).magnitude, float)
        if _is_pint_column(status["observed"])
        else np.asarray(status["observed"], float)
    )
    long_nd = (
        (status["status"].to_numpy() == "censored")
        & (status["n_daughters"].to_numpy() == 0)
        & (observed >= threshold)
    )
    out["n_long_not_divided"] = int(long_nd.sum())
    denom = out["n_divided"] + out["n_long_not_divided"]
    out["fraction_divided"] = (out["n_divided"] / denom) if denom else None
    if sizes is not None:
        ids = status.index[long_nd]
        sub = sizes.reindex(ids)
        change = sub["size_change"].to_numpy(float)
        lo_col, hi_col = sub["mu_fit_ci_lo"], sub["mu_fit_ci_hi"]
        lo = np.asarray(
            lo_col.values.quantity.magnitude if _is_pint_column(lo_col) else lo_col,
            float,
        )
        hi = np.asarray(
            hi_col.values.quantity.magnitude if _is_pint_column(hi_col) else hi_col,
            float,
        )
        arrested = (change < arrest_max_growth) | ((lo <= 0) & (hi >= 0))
        out["n_arrested"] = int(np.sum(arrested))
    return out


def _unit_of(df: pd.DataFrame, col: str) -> str:
    """Unit of a pint column, else the one recorded in ``df.attrs``, else ``""``."""
    if _is_pint_column(df[col]):
        return str(df[col].pint.units)
    return str((df.attrs.get(UNIT_ATTR) or {}).get(col, ""))


def _compute_hose_bands(
    doubling_times_df: pd.DataFrame,
    time_grid,
    ci_levels,
    min_n: int,
    n_bootstrap: int,
    random_state,
    value_col: str = "doubling_time",
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

    A cell is alive from its birth (``start_time``) until it divides
    (``division_time``, exclusive), falling back to its last observed frame
    (``end_time``, inclusive) for tables without a ``division_time`` column --
    on the movie's own frame grid both give the same cross-sections.
    ``value_col`` selects the per-cell value (in its own unit, e.g. a growth
    rate); only the time columns are converted to the grid's unit.
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

    has_division = "division_time" in doubling_times_df.columns
    if len(doubling_times_df) > 0:
        start = doubling_times_df["start_time"].values.quantity.to(unit_str).magnitude
        end_col = "division_time" if has_division else "end_time"
        end = doubling_times_df[end_col].values.quantity.to(unit_str).magnitude
        values = doubling_times_df[value_col].values.quantity
        if value_col == "doubling_time":
            values = values.to(unit_str)
        values = np.asarray(values.magnitude, dtype=float)
    else:
        start = end = values = np.array([], dtype=float)

    n_points = len(t_grid)
    mean_arr = np.full(n_points, np.nan)
    bands: dict[float, tuple[np.ndarray, np.ndarray]] = {
        p: (np.full(n_points, np.nan), np.full(n_points, np.nan)) for p in ci_levels
    }

    rng = np.random.default_rng(random_state)

    for i, t in enumerate(t_grid):
        until = (t < end) if has_division else (t <= end)
        alive = values[(start <= t) & until & ~np.isnan(values)]
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
    value_col: str = "doubling_time",
    value_label: str | None = None,
) -> Figure:
    """Plot mean doubling time evolving over the experiment as a "temporal hose".

    At every time ``t`` in ``time_grid``, a cell contributes its
    ``doubling_time`` to the cross-section at ``t`` for as long as it was
    alive -- i.e. whenever ``start_time <= t < division_time`` -- not just at the
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
        value_col: the per-cell column to average -- ``"doubling_time"``
            (default) or any other per-cycle value in the same table, e.g. a
            per-cell growth rate (:func:`acia.analysis.growth_rate.compute_cell_growth_rates`).
        value_label: y-axis label; defaults to ``"<value_col> [<unit>]"``.

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
        doubling_times_df,
        time_grid,
        ci_levels,
        min_n,
        n_bootstrap,
        random_state,
        value_col=value_col,
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
    if value_label is None:
        value_unit = (
            unit_str
            if value_col == "doubling_time"
            else _unit_of(doubling_times_df, value_col)
        )
        value_label = f"{value_col.replace('_', ' ')} [{value_unit}]"
    ax.set_ylabel(value_label)
    ax.legend()
    fig.tight_layout()

    return fig
