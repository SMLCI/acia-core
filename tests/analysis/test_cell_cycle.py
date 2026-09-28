"""Tests for :mod:`acia.analysis.cell_cycle` (per-cycle plots and summaries)."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from acia import ureg
from acia.analysis.cell_cycle import (
    format_summary,
    plot_cycle_bars,
    plot_cycle_distribution,
    plot_cycle_dots,
    read_summaries,
    summarize,
    update_summary,
)
from acia.analysis.doubling_time import compute_doubling_times
from acia.analysis.units import strip_units
from acia.segm.local import THWCSequenceSource


def _doubling_df():
    """Mother 1 (frames 0-2) -> 2, 3 (frames 3-5) -> 4..7 (frames 6-9)."""
    g = nx.DiGraph()
    g.add_node(1, start_frame=0, end_frame=2)
    for n in (2, 3):
        g.add_node(n, start_frame=3, end_frame=5)
        g.add_edge(1, n)
    for parent, kids in ((2, (4, 5)), (3, (6, 7))):
        for k in kids:
            g.add_node(k, start_frame=6, end_frame=9)
            g.add_edge(parent, k)
    root = 0
    g.add_node(root, start_frame=0, end_frame=0)
    g.add_edge(root, 1)
    g.add_node(8, start_frame=1, end_frame=9)
    g.add_edge(root, 8)
    src = THWCSequenceSource(np.zeros((10, 4, 4, 1), np.uint8), frame_interval="10 min")
    return compute_doubling_times(g, src)


def test_doubling_time_runs_birth_to_birth():
    df = _doubling_df()
    # cell 2 is seen on frames 3, 4, 5 and its daughters appear on frame 6:
    # it lived 3 frames = 30 min, not end - start = 20 min
    row = df.loc[2]
    assert row["doubling_time"].magnitude == pytest.approx(30)
    assert row["division_time"].magnitude == pytest.approx(60)
    assert (row["end_time"] - row["start_time"]).magnitude == pytest.approx(20)


def test_dots_and_bars_are_separate_figures_and_mark_rejected():
    df = _doubling_df()
    df["valid"] = [True, False, True]

    dots = plot_cycle_dots(df)
    bars = plot_cycle_bars(df)
    assert dots is not bars

    (rejected, kept) = dots.axes[0].collections
    assert len(rejected.get_offsets()) == 1 and len(kept.get_offsets()) == 2
    # x of the dots is the division time, y the doubling time
    np.testing.assert_allclose(sorted(kept.get_offsets()[:, 0]), [30, 60])

    segments = [s for c in bars.axes[0].collections for s in c.get_segments()]
    spans = sorted((s[0][0], s[1][0]) for s in segments)
    assert spans == [(0, 30), (30, 60), (30, 60)]
    assert bars.axes[0].get_ylabel() == "doubling time [minute]"
    plt.close("all")


def test_plots_accept_stripped_float_columns():
    stripped, units = strip_units(_doubling_df())
    stripped.attrs["units"] = units
    fig = plot_cycle_dots(stripped)
    assert fig.axes[0].get_ylabel() == "doubling time [minute]"
    plot_cycle_distribution(stripped)
    plt.close("all")


def test_summarize_quantity_series_and_edge_cases():
    df = _doubling_df()
    stats = summarize(df["doubling_time"])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(30)
    assert stats["std"] == pytest.approx(0)
    assert stats["unit"] == "minute"

    hours = summarize(df["doubling_time"], unit="hour")
    assert hours["mean"] == pytest.approx(0.5)

    single = summarize([4.0, np.nan])
    assert single["n"] == 1 and single["std"] is None and single["sem"] is None
    assert summarize([])["mean"] is None

    s = summarize(ureg.Quantity([1.0, 3.0], "1/hour"))
    assert s["std"] == pytest.approx(np.sqrt(2))
    assert format_summary("x", s).startswith("x = 2.00 ± 1.41 1 / hour")


def test_update_summary_merges_sections_and_reads_back(tmp_path):
    update_summary(tmp_path / "a", "doubling_time", summarize([1.0, 2.0]))
    update_summary(tmp_path / "a", "growth_rate_population", {"cc": {"mu": 0.5}})
    update_summary(tmp_path / "a", "doubling_time", summarize([3.0]))

    data = json.loads((tmp_path / "a" / "summary.json").read_text())
    assert set(data) == {"doubling_time", "growth_rate_population"}
    assert data["doubling_time"]["n"] == 1

    path = tmp_path / "a" / "summary.json"
    table = read_summaries([path], keys={path: {"experiment": "exp1", "roi": 1}})
    assert table.loc[0, "experiment"] == "exp1"
    assert table.loc[0, "growth_rate_population.cc.mu"] == 0.5
    assert isinstance(table, pd.DataFrame)
