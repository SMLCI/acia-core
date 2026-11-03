"""Tests for the violins, percentile picks, cell inspection and plotly twins."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from acia.analysis.cell_cycle import pick_percentile_cells, plot_cycle_violins
from acia.analysis.cell_cycle_interactive import (
    cell_status,
    plot_cycle_bars_interactive,
    plot_cycle_dots_interactive,
    plot_distributions_interactive,
    plot_fit_vs_doubling_interactive,
)
from acia.analysis.cell_inspect import cell_trace, plot_cell_traces, show_cell
from acia.analysis.doubling_time import compute_doubling_times
from acia.analysis.growth_rate import compute_cell_growth_rates
from acia.base import Contour, Overlay
from acia.segm.local import THWCSequenceSource
from acia.tracking.filters import TrackletFilter, apply_tracklet_filters


@pytest.fixture
def lineage():
    """Mother 1 -> 2, 3 (each doubling every 6 frames of 10 min), plus a root."""
    g = nx.DiGraph()
    spans = {
        0: (0, 0),
        1: (0, 5),
        2: (6, 11),
        3: (6, 11),
        4: (12, 13),
        5: (12, 13),
        6: (12, 13),
        7: (12, 13),
    }
    for n, (a, b) in spans.items():
        g.add_node(n, start_frame=a, end_frame=b)
    g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)])
    contours, cid = [], 0
    for n in (1, 2, 3):
        a, b = spans[n]
        for f in range(a, b + 1):
            side = np.sqrt(100 * 2 ** ((f - a) / 6)) * (1.1 if n == 3 else 1.0)
            x0 = 5 + 25 * (n - 1)
            contours.append(
                Contour(
                    [[x0, 5], [x0 + side, 5], [x0 + side, 5 + side], [x0, 5 + side]],
                    -1,
                    f,
                    cid,
                    n,
                )
            )
            cid += 1
    src = THWCSequenceSource(
        np.random.default_rng(0).integers(0, 255, (14, 40, 90, 1), dtype=np.uint8),
        frame_interval="10 min",
    )
    overlay = Overlay(contours, frames=list(range(14)))
    cycles = apply_tracklet_filters(
        compute_doubling_times(g, src),
        [TrackletFilter("doubling_time", "30 min", "120 min")],
    )
    rates = compute_cell_growth_rates(overlay, cycles, src, min_points=3)
    return overlay, src, cycles, rates


def test_violins_and_percentile_picks(lineage):
    _, _, _, rates = lineage
    fig = plot_cycle_violins(
        {"fit": rates["mu_fit"], "ln2/DT": rates["mu_from_dt"]}, unit="1/hour"
    )
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()][0].startswith("fit")
    picks = pick_percentile_cells(rates, "mu_fit", [0, 50, 100])
    assert list(picks["percentile"]) == [0, 50, 100]
    assert set(picks["tracklet"]) <= {1, 2, 3}
    plt.close("all")


def test_traces_and_show_cell(lineage):
    overlay, src, cycles, rates = lineage
    trace = cell_trace(overlay, 2, src)
    assert trace["time_since_birth"].iloc[-1] == pytest.approx(50)
    assert trace.attrs["time_unit"] == "minute"
    picks = pick_percentile_cells(rates, "mu_fit", [0, 100])
    fig = plot_cell_traces(overlay, picks, src, rates=rates)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[0].startswith("P0 · #") and "μ" in labels[0]
    fig = show_cell(overlay, src, 2, cycles=rates, max_frames=4)
    assert fig._suptitle.get_text().startswith("tracklet #2")
    assert len(fig.axes) == 5  # 4 crops + the trace
    with pytest.raises(KeyError):
        cell_trace(overlay, 99, src)
    plt.close("all")


def test_interactive_figures_carry_tracklet_ids(lineage):
    _, _, cycles, rates = lineage
    rates = lineage[3].copy()
    rates.loc[3, "fit_outlier"] = True
    assert list(cell_status(rates)) == ["valid", "valid", "flagged"]

    dots = plot_cycle_dots_interactive(rates)
    ids = {row[0] for trace in dots.data for row in trace.customdata}
    assert ids == {"1", "2", "3"}
    assert {t.name.split(" (")[0] for t in dots.data} == {"valid", "flagged"}

    bars = plot_cycle_bars_interactive(rates, "mu_fit")
    assert any(x is None for x in bars.data[0].x)  # one segment per cell

    violins = plot_distributions_interactive({"all": rates}, "mu_fit", unit="1/hour")
    assert len(violins.data[0].y) == 3
    scatter = plot_fit_vs_doubling_interactive(rates)
    assert "tracklet" in scatter.data[0].hovertemplate
    assert isinstance(rates, pd.DataFrame)


def test_fit_vs_doubling_plots_group_by_reason(lineage):
    from acia.analysis.cell_cycle import plot_fit_vs_doubling
    from acia.analysis.cell_cycle_interactive import plot_fit_vs_doubling_interactive

    rates = lineage[3].copy()
    rates["fit_outlier_reason"] = rates["fit_outlier_reason"].astype(object)
    first = rates.index[0]
    rates.loc[first, "fit_outlier_reason"] = "size drop (missed division?)"
    fig = plot_fit_vs_doubling(rates)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any(lab.startswith("size drop") for lab in labels)
    ifig = plot_fit_vs_doubling_interactive(rates)
    names = [t.name for t in ifig.data]
    assert any(n.startswith("size drop") for n in names) and "y = x" in names


def test_hover_table_prefers_tracklet_column(lineage):
    from acia.analysis.cell_cycle_interactive import hover_table

    rates = lineage[3]

    flat = rates.reset_index()
    assert list(hover_table(flat)["tracklet"]) == [str(i) for i in rates.index]
