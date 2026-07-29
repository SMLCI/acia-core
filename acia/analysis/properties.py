"""Before/after property-distribution histograms for filtered cell populations.

This module renders a grid of density-normalized histograms for one or more
properties from a property-extractor DataFrame (the output of
:class:`acia.analysis.ExtractorExecutor`), optionally comparing an unfiltered
population against a filtered one (e.g. via
:func:`acia.segm.filter.apply_cell_filters`). Properties are laid out as
columns; when a filtered ("after") DataFrame is given, before/after are
stacked as rows within the same column and share identical bin edges and axis
limits, so outliers can be visually confirmed to "stay or vanish" under
filtering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pint
    from matplotlib.figure import Figure


def _axis_label(prop: str, units: dict[str, pint.Unit] | None) -> str:
    """Build the x-axis label for ``prop``, appending a pint unit suffix.

    Args:
        prop: the property/column name.
        units: optional column -> :class:`pint.Unit` mapping (e.g.
            :attr:`acia.analysis.ExtractorExecutor.units`). When ``prop`` is
            absent (or ``units`` is ``None``), no unit suffix is added.

    Returns:
        ``f"{prop} [{units[prop]:~P}]"`` when a unit is known, else the bare
        ``prop`` name. The ``~P`` (pretty) pint format yields Unicode (e.g.
        ``µm²``) that matplotlib renders directly; ``~L`` (LaTeX) is avoided
        because matplotlib only interprets LaTeX inside ``$...$`` and would
        otherwise show the raw ``\\mathrm{...}`` markup.
    """
    if units is not None and prop in units:
        return f"{prop} [{units[prop]:~P}]"
    return prop


def _finite(series) -> np.ndarray:
    """Return a column's finite float values as a 1-D array (drops NaN/inf)."""
    arr = np.asarray(series, dtype=float)
    return np.asarray(arr[np.isfinite(arr)])


def _hist_or_note(ax, values: np.ndarray, bins, **hist_kwargs) -> None:
    """Density-histogram ``values`` on ``ax``, or annotate "no cells" if empty.

    An empty population (no detected cells, or all cells filtered out) is a valid
    state, not an error -- the axis is drawn empty with a note instead of raising.
    """
    if values.size:
        ax.hist(values, bins=bins, density=True, **hist_kwargs)
    else:
        ax.text(
            0.5,
            0.5,
            "no cells",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.6",
        )


def plot_property_histograms(
    df_before: pd.DataFrame,
    properties: Sequence[str],
    *,
    df_after: pd.DataFrame | None = None,
    units: dict[str, pint.Unit] | None = None,
    bins: int = 50,
    log_y: bool = False,
    show_removed: bool = False,
) -> Figure:
    """Plot density-normalized before/after histograms for one or more properties.

    Properties are h-stacked as columns. When ``df_after`` is given, before
    (row 0) and after (row 1) are v-stacked as rows within each column, and
    share identical bin edges (computed from the combined range of
    ``df_before``/``df_after`` for that property) and identical x/y axis
    limits, so the two histograms for one property stay directly comparable.

    Args:
        df_before: the unfiltered (or "before") property DataFrame, e.g. the
            output of :meth:`acia.analysis.ExtractorExecutor.execute`.
        properties: column names of ``df_before`` (and ``df_after``, if given)
            to plot, one column of the grid per property.
        df_after: optional filtered (or "after") property DataFrame with the
            same columns as ``df_before``. When given, the grid gains a second
            row per property for direct before/after comparison. When
            ``None``, only ``df_before`` is plotted (one row).
        units: optional column -> :class:`pint.Unit` mapping (e.g.
            :attr:`acia.analysis.ExtractorExecutor.units`) used to format each
            column's x-axis label as ``f"{prop} [{unit:~L}]"``. Properties
            missing from ``units`` (or ``units=None``) fall back to the bare
            property name.
        bins: number of histogram bins (default ``50``).
        log_y: if ``True``, every Axes' y-scale is set to ``"log"`` (still
            density-normalized). Default is linear.
        show_removed: if ``True`` and ``df_after`` is an index-subset of
            ``df_before``, overlay the filtered-out cells (``before`` minus
            ``after``) on each "after" histogram as a red step outline, so you can
            see where in the property's range the filter cut. Density-normalized
            (shows location/shape, not count). Default ``False``.

    Returns:
        A matplotlib :class:`~matplotlib.figure.Figure` with a
        ``1 x len(properties)`` grid (``df_after=None``) or
        ``2 x len(properties)`` grid (``df_after`` given) of density
        histograms. The figure is not shown (no ``plt.show()``).

    Raises:
        TypeError: if ``properties`` is a bare ``str`` instead of a sequence
            of column names (a plausible typo, e.g. ``"area"`` instead of
            ``["area"]``, that would otherwise be silently iterated
            character-by-character).
        ValueError: if ``properties`` is empty. A property with no finite values
            to plot -- an empty population (0 rows), an empty ``df_after`` (all
            cells filtered out), or an all-NaN column -- is *not* an error: that
            axis is drawn empty with a "no cells" note.
        KeyError: if a property in ``properties`` is not a column of
            ``df_before`` (or ``df_after``, when given).
    """
    import matplotlib.pyplot as plt

    if isinstance(properties, str):
        raise TypeError(
            "properties must be a sequence of column names, not a single string"
        )

    n_props = len(properties)
    if n_props == 0:
        raise ValueError("properties must be non-empty")

    has_after = df_after is not None

    # Pass 1: validate everything and precompute finite-filtered arrays (and,
    # for the df_after case, shared bin edges) BEFORE any Figure is created,
    # so a validation failure can never leak an orphaned/partial Figure.
    before_arrays: list[np.ndarray] = []
    after_arrays: list[np.ndarray] = []
    removed_arrays: list[np.ndarray] = []
    bins_per_prop: list[np.ndarray | int] = []

    # the removed cells = before rows whose index is not among the survivors
    # (only meaningful when df_after is an index-subset of df_before)
    removed_index = None
    if (
        df_after is not None
        and show_removed
        and df_after.index.isin(df_before.index).all()
    ):
        removed_index = df_before.index.difference(df_after.index)

    for prop in properties:
        if prop not in df_before.columns:
            raise KeyError(f"'{prop}' not found in df_before")
        before_finite = _finite(df_before[prop])
        before_arrays.append(before_finite)

        if df_after is not None:
            if prop not in df_after.columns:
                raise KeyError(f"'{prop}' not found in df_after")
            after_finite = _finite(df_after[prop])
            after_arrays.append(after_finite)

            # bin edges from whatever finite values exist -- an empty "before" or
            # "after" (e.g. all cells filtered out) is drawn empty, not raised.
            populated = [a for a in (before_finite, after_finite) if a.size]
            if populated:
                combined = np.concatenate(populated)
                bins_per_prop.append(np.histogram_bin_edges(combined, bins=bins))
            else:
                bins_per_prop.append(bins)

            if removed_index is not None and len(removed_index):
                rem = np.asarray(df_before.loc[removed_index, prop], dtype=float)
                removed_arrays.append(rem[np.isfinite(rem)])
            else:
                removed_arrays.append(np.array([]))
        else:
            bins_per_prop.append(bins)

    # Pass 2: only now create the Figure and plot, using the precomputed
    # finite arrays/bin edges from Pass 1.
    fig, axes = plt.subplots(
        2 if has_after else 1,
        n_props,
        squeeze=False,
        sharex="col" if has_after else False,
        sharey="col" if has_after else False,
    )

    for j, prop in enumerate(properties):
        prop_bins = bins_per_prop[j]

        if has_after:
            ax_before = axes[0, j]
            ax_after = axes[1, j]
            _hist_or_note(ax_before, before_arrays[j], prop_bins)
            _hist_or_note(ax_after, after_arrays[j], prop_bins, label="kept")

            # overlay the removed cells in red so you can see WHERE (in this
            # property's range) the filter cut cells -- density-normalized, so it
            # shows the shape/location of the removed population, not its count.
            if show_removed and removed_arrays and removed_arrays[j].size:
                ax_after.hist(
                    removed_arrays[j],
                    bins=prop_bins,
                    density=True,
                    histtype="step",
                    color="red",
                    linewidth=1.5,
                    label="removed",
                )
                ax_after.legend(fontsize="small")

            ax_before.set_title(prop)
            ax_after.set_xlabel(_axis_label(prop, units))
            if j == 0:
                ax_before.set_ylabel("before")
                ax_after.set_ylabel("after")

            for ax in (ax_before, ax_after):
                ax.grid(True, linestyle=":", alpha=0.4)
                ax.set_axisbelow(True)
                if log_y:
                    ax.set_yscale("log")
        else:
            ax = axes[0, j]
            _hist_or_note(ax, before_arrays[j], prop_bins)
            ax.set_title(prop)
            ax.set_xlabel(_axis_label(prop, units))
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.set_axisbelow(True)
            if log_y:
                ax.set_yscale("log")

    fig.tight_layout()

    return fig
