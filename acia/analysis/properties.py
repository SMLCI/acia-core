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
        ``f"{prop} [{units[prop]:~L}]"`` when a unit is known, else the bare
        ``prop`` name.
    """
    if units is not None and prop in units:
        return f"{prop} [{units[prop]:~L}]"
    return prop


def plot_property_histograms(
    df_before: pd.DataFrame,
    properties: Sequence[str],
    *,
    df_after: pd.DataFrame | None = None,
    units: dict[str, pint.Unit] | None = None,
    bins: int = 50,
    log_y: bool = False,
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
        ValueError: if ``properties`` is empty, or if a property (after
            dropping non-finite values) has no finite values left to plot in
            ``df_before`` (or ``df_after``, when given) -- e.g. an empty
            DataFrame or an all-NaN column.
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
    bins_per_prop: list[np.ndarray | int] = []

    for prop in properties:
        if prop not in df_before.columns:
            raise KeyError(f"'{prop}' not found in df_before")
        before_arr = np.asarray(df_before[prop], dtype=float)
        before_finite = before_arr[np.isfinite(before_arr)]
        if before_finite.size == 0:
            raise ValueError(f"'{prop}' has no finite values to plot")
        before_arrays.append(before_finite)

        if df_after is not None:
            if prop not in df_after.columns:
                raise KeyError(f"'{prop}' not found in df_after")
            after_arr = np.asarray(df_after[prop], dtype=float)
            after_finite = after_arr[np.isfinite(after_arr)]
            if after_finite.size == 0:
                raise ValueError(f"'{prop}' has no finite values to plot")
            after_arrays.append(after_finite)

            combined = np.concatenate([before_finite, after_finite])
            bins_per_prop.append(np.histogram_bin_edges(combined, bins=bins))
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
            ax_before.hist(before_arrays[j], bins=prop_bins, density=True)
            ax_after.hist(after_arrays[j], bins=prop_bins, density=True)

            ax_before.set_title(prop)
            ax_after.set_xlabel(_axis_label(prop, units))
            if j == 0:
                ax_before.set_ylabel("before")
                ax_after.set_ylabel("after")

            if log_y:
                ax_before.set_yscale("log")
                ax_after.set_yscale("log")
        else:
            ax = axes[0, j]
            ax.hist(before_arrays[j], bins=prop_bins, density=True)
            ax.set_title(prop)
            ax.set_xlabel(_axis_label(prop, units))
            if log_y:
                ax.set_yscale("log")

    fig.tight_layout()

    return fig
