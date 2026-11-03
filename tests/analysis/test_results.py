"""Tests for the cross-run analysis layer (acia.analysis.results / conditions)."""

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tifffile

from acia.analysis import StageContext, units_in_header, write_units_csv
from acia.analysis.conditions import assign_conditions, parse_position_ranges
from acia.analysis.results import (
    ResultSet,
    distribution,
    division_census_table,
    dynamics,
    fit_monod,
    group_palette,
    growth_rates,
    per_run,
    plot_monod,
    plot_over_time,
    step_through,
)


def test_parse_position_ranges():
    assert parse_position_ranges("pos001-pos003, 7, 10-11") == {1, 2, 3, 7, 10, 11}
    assert parse_position_ranges(["1-2", 5]) == {1, 2, 5}
    with pytest.raises(ValueError, match="backwards"):
        parse_position_ranges("5-3")
    with pytest.raises(ValueError, match="not a position"):
        parse_position_ranges("a-b")


def test_assign_conditions_nested_flat_overlap_unmapped():
    df = pd.DataFrame(
        {"experiment": ["e1", "e1", "e1", "e2"], "position": [1, 2, 9, 1]}
    )
    out = assign_conditions(
        df,
        {"e1": {"low": "1", "high": "2"}, "e2": {"low": "1-3"}},
        values={"low": 0.1, "high": 1.0},
        value_name="substrate",
    )
    assert list(out["condition"]) == ["low", "high", None, "low"]
    assert list(out["substrate"].fillna(-1)) == [0.1, 1.0, -1, 0.1]
    assert out.attrs["unmapped"] == [("e1", 9)]
    flat = assign_conditions(df, {"a": "1-2"})
    assert list(flat["condition"]) == ["a", "a", None, "a"]
    with pytest.raises(ValueError, match="both"):
        assign_conditions(df, {"a": "1-2", "b": "2-3"})


def _run(root, experiment, position, roi, mu, n_cells=6, seed=0, census=None):
    """One run's output folder with manifest, doubling times, growth rates, properties."""
    rng = np.random.default_rng(seed)
    exports = root / "exports" / experiment
    exports.mkdir(parents=True, exist_ok=True)
    image = exports / f"pos{position:03d}_roi{roi:03d}.tiff"
    tifffile.imwrite(image, np.zeros((4, 4), np.uint8))
    (exports / "roi_export.json").write_text(json.dumps({"experiment": experiment}))
    ctx = StageContext.for_image(image, root / "out", stage="DoublingTime")

    dt = np.log(2) / mu * 60 * rng.uniform(0.95, 1.05, n_cells)  # minutes
    cycles = pd.DataFrame(
        {
            "start_time": pd.array(np.arange(n_cells) * 10.0, dtype="pint[minute]"),
            "division_time": pd.array(
                np.arange(n_cells) * 10.0 + dt, dtype="pint[minute]"
            ),
            "doubling_time": pd.array(dt, dtype="pint[minute]"),
            "valid": [True] * (n_cells - 1) + [False],
        },
        index=pd.Index(range(1, n_cells + 1), name="tracklet"),
    )
    units_in_header(ctx.keyed(cycles)).to_csv(ctx.output_path("doubling_times.csv"))
    rates = cycles.assign(
        mu_fit=pd.array(np.log(2) / dt * 60, dtype="pint[1/hour]"),
        mu_from_dt=pd.array(np.log(2) / dt * 60, dtype="pint[1/hour]"),
        fit_outlier=False,
    )
    units_in_header(ctx.keyed(rates)).to_csv(ctx.output_path("cell_growth_rates.csv"))
    props = pd.DataFrame(
        {
            "time": pd.array(np.repeat([0.0, 10.0, 20.0], 2), dtype="pint[minute]"),
            "area": pd.array(
                [1.0, 1.0, 1.2, 1.3, 1.5, 1.6], dtype="pint[micrometer**2]"
            ),
            "id": range(6),
        }
    )
    write_units_csv(ctx.keyed(props), ctx.output_path("cell_properties.csv"))
    pd.DataFrame(
        [{"method": "TSCA", "growth_rate": mu, "growth_rate_unit": "1/h", **ctx.keys}]
    ).to_csv(ctx.output_path("growth_estimates.csv"), index=False)
    if census is not None:
        from acia.analysis.cell_cycle import update_summary

        update_summary(ctx.output_dir, "division_census", census)
    ctx.finish()


@pytest.fixture
def two_batches(tmp_path):
    flat = tmp_path / "exp1_batch"
    nested = tmp_path / "exp2_batch"
    for position, mu in ((1, 0.2), (2, 0.2), (3, 0.5), (4, 0.5), (5, 0.8), (6, 0.8)):
        _run(flat / f"pos{position:03d}_roi001", "e1", position, 1, mu, seed=position)
    for position, mu in ((1, 0.3), (2, 0.6), (3, 0.85)):
        _run(
            nested / "e2" / f"pos{position:03d}_roi001",
            "e2",
            position,
            1,
            mu,
            seed=10 + position,
        )
    return flat, nested


def test_result_set_opens_any_depth_and_reads_tables(two_batches):
    rs = ResultSet.open(list(two_batches))
    assert len(rs.runs) == 9
    assert set(rs.runs["experiment"]) == {"e1", "e2"}
    assert rs.tables()["doubling_times"] == 9

    dt = rs.table("doubling_times")
    assert {"tracklet", "experiment", "position", "roi"} <= set(dt.columns)
    assert str(dt["doubling_time"].pint.units) == "minute"
    assert len(dt) == 9 * 6

    only_e1 = rs.where(experiment="e1", position=range(1, 4))
    assert list(only_e1.runs["position"]) == [1, 2, 3]
    summaries = rs.summaries()
    assert isinstance(summaries, pd.DataFrame)


def test_conditions_and_analyses(two_batches):
    rs = ResultSet.open(list(two_batches))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rs = rs.with_conditions(
            {
                "e1": {"low": "1-2", "mid": "3-4", "high": "5-6"},
                "e2": {"low": "1", "mid": "2", "high": "3"},
            },
            values={"low": 0.1, "mid": 1.0, "high": 10.0},
            value_name="substrate",
        )
    assert not caught  # everything mapped

    dt = rs.table("doubling_times")
    stats, fig = distribution(dt, "doubling_time", by="condition")
    assert set(stats.index) == {"low", "mid", "high"}
    assert stats.loc["low", "n"] == 3 * 5  # the invalid cell of each run is dropped
    _, ifig = distribution(
        dt, "doubling_time", by=["experiment", "condition"], interactive=True
    )
    assert "tracklet" in str(ifig.data[0].hovertemplate)

    cells = rs.table("cell_properties")
    curves, _ = dynamics(cells, "area", by="condition", agg="sum")
    assert set(curves["condition"]) == {"low", "mid", "high"}
    assert curves["runs"].max() == 3
    counts, _ = dynamics(cells, None, by="experiment", agg="count", interactive=True)
    assert counts["mean"].iloc[0] == 2

    table, _ = per_run(dt, "doubling_time", stat="median")
    assert len(table) == 9 and "condition" in table
    plt.close("all")


def test_growth_rates_sources_and_monod(two_batches):
    rs = (
        ResultSet.open(list(two_batches))
        .with_conditions(
            {"low": "1-2", "mid": "3-4", "high": "5-6"},
            values={"low": 0.1, "mid": 1.0, "high": 10.0},
            value_name="substrate",
        )
        .where(experiment="e1")
    )
    per_cell = growth_rates(rs, "fit")
    assert per_cell["mu"].between(0.1, 1.0).all()
    population = growth_rates(rs, "population:TSCA")
    assert sorted(population["mu"].round(2)) == [0.2, 0.2, 0.5, 0.5, 0.8, 0.8]

    fit = fit_monod(per_cell, S="substrate", s_unit="mM")
    assert len(fit.points) == 3
    assert fit.to_dict()["ks_unit"] == "mM"
    plot_monod(fit)
    plot_monod(fit, interactive=True, log_x=True)
    plt.close("all")


def test_fit_monod_recovers_parameters():
    rng = np.random.default_rng(1)
    rows = []
    for s in (0.05, 0.1, 0.3, 1.0, 3.0, 10.0):
        for _ in range(40):
            rows.append(
                {
                    "condition": f"{s} mM",
                    "substrate": s,
                    "mu": 0.9 * s / (0.4 + s) + rng.normal(0, 0.02),
                }
            )
    fit = fit_monod(pd.DataFrame(rows), s_unit="mM")
    assert fit.mu_max == pytest.approx(0.9, rel=0.05)
    assert fit.ks == pytest.approx(0.4, rel=0.15)
    assert fit.ks_ci[0] < 0.4 < fit.ks_ci[1]
    assert fit.r2 > 0.98
    with pytest.raises(ValueError, match="three"):
        fit_monod(pd.DataFrame(rows[:80]))


def test_condition_of_a_position_covers_all_its_rois(tmp_path):
    """Conditions are mapped per position; every ROI of that position gets it."""
    root = tmp_path / "batch"
    _run(root / "pos001_roi001", "e1", 1, 1, 0.5, seed=1)
    _run(root / "pos001_roi002", "e1", 1, 2, 0.5, seed=2)
    _run(root / "pos002_roi001", "e1", 2, 1, 0.8, seed=3)

    rs = ResultSet.open(root).with_conditions({"low": "pos001", "high": "pos002"})
    runs = rs.runs_with_conditions().sort_values(["position", "roi"])
    assert list(runs["condition"]) == ["low", "low", "high"]

    dt = rs.table("doubling_times")
    low = dt[dt["condition"] == "low"]
    assert set(low["roi"]) == {1, 2}  # both ROIs of position 1 are in the group
    stats, _ = distribution(dt, "doubling_time", by="condition")
    assert stats.loc["low", "n"] == 2 * 5 and stats.loc["high", "n"] == 5
    by_roi, _ = distribution(dt, "doubling_time", by=["position", "roi"])
    assert len(by_roi) == 3
    plt.close("all")


def _conditioned(two_batches):
    return ResultSet.open(list(two_batches)).with_conditions(
        {
            "e1": {"low": "1-2", "mid": "3-4", "high": "5-6"},
            "e2": {"low": "1", "mid": "2", "high": "3"},
        },
        values={"low": 0.1, "mid": 1.0, "high": 10.0},
        value_name="substrate",
    )


def test_order_and_palette_select_and_colour_groups(two_batches):
    dt = _conditioned(two_batches).table("doubling_times")
    stats, fig = distribution(
        dt, "doubling_time", order=["high", "low"], palette={"high": "#ff0000"}
    )
    assert list(stats.index) == ["high", "low"]
    labels = [t.get_text().split("\n")[0] for t in fig.axes[0].get_xticklabels()]
    assert labels == ["high", "low"]
    _, ifig = distribution(
        dt,
        "doubling_time",
        order=["high"],
        palette={"high": "#ff0000"},
        interactive=True,
    )
    assert [tr.marker.color for tr in ifig.data] == ["#ff0000"]
    cells = _conditioned(two_batches).table("cell_properties")
    curves, _ = dynamics(cells, "area", agg="sum", order=["mid"])
    assert set(curves["condition"]) == {"mid"}
    assert group_palette(["a", "b"]) == {"a": "#1f77b4", "b": "#ff7f0e"}
    plt.close("all")


def test_monod_band_contains_the_curve_and_covers_truth():
    rng = np.random.default_rng(2)
    rows = [
        {
            "condition": f"{s}",
            "substrate": s,
            "mu": 0.9 * s / (0.4 + s) + rng.normal(0, 0.03),
        }
        for s in (0.05, 0.1, 0.3, 1.0, 3.0, 10.0)
        for _ in range(30)
    ]
    fit = fit_monod(pd.DataFrame(rows), s_unit="mM")
    grid = np.linspace(0.01, 10, 50)
    lo, hi = fit.band(grid)
    mu = fit.predict(grid)
    assert np.all(lo <= mu) and np.all(mu <= hi) and np.all(hi - lo > 0)
    truth = 0.9 * grid / (0.4 + grid)
    assert np.mean((lo <= truth) & (truth <= hi)) > 0.9
    fig = plot_monod(fit)
    assert any("CI" in t.get_text() for t in fig.axes[0].get_legend().get_texts())
    ifig = plot_monod(fit, interactive=True, ci=None)
    assert not any("CI" in (tr.name or "") for tr in ifig.data)
    plt.close("all")


def test_plot_over_time_layers_and_mean(two_batches):
    rs = _conditioned(two_batches)
    dt = rs.table("doubling_times")
    table, fig = plot_over_time(
        dt, "doubling_time", kind=("violin", "dots", "bars"), order=["low", "high"]
    )
    assert set(table["group"]) == {"low", "high"}
    stats, _ = distribution(dt, "doubling_time", order=["low", "high"])
    dashed = [ln for ln in fig.axes[0].lines if ln.get_linestyle() == "--"]
    assert sorted(round(ln.get_ydata()[0], 6) for ln in dashed) == sorted(
        round(float(m), 6) for m in stats["mean"]
    )
    _, ifig = plot_over_time(
        dt, "doubling_time", kind=("dots", "bars"), interactive=True
    )
    groups = {tr.legendgroup for tr in ifig.data}
    assert groups == {"low", "mid", "high"}
    assert ifig.layout.legend.groupclick == "toggleitem"
    with pytest.raises(ValueError, match="kind"):
        plot_over_time(dt, "doubling_time", kind="lines")
    plt.close("all")


def test_step_through_png_sequence_and_slider(two_batches, tmp_path):
    dt = _conditioned(two_batches).table("doubling_times")
    order = ["low", "mid", "high"]
    paths = step_through(
        plot_over_time, dt, order, "doubling_time", out_dir=tmp_path / "steps"
    )
    assert [p.name for p in paths] == ["step_01.png", "step_02.png", "step_03.png"]
    assert all(p.stat().st_size > 0 for p in paths)
    fig = step_through(
        plot_over_time, dt, order[::-1], "doubling_time", interactive=True
    )
    steps = fig.layout.sliders[0].steps
    assert [st.label for st in steps] == ["high", "mid", "low"]
    first = steps[0].args[0]["visible"]
    shown = {tr.legendgroup for tr, v in zip(fig.data, first, strict=True) if v}
    assert shown == {"high"}
    plt.close("all")


def test_division_census_table_warns_about_groups_without_divisions(tmp_path):
    root = tmp_path / "batch"
    for position, divided in ((1, 5), (2, 4), (3, 0)):
        _run(
            root / f"pos{position:03d}_roi001",
            "e1",
            position,
            1,
            0.5,
            seed=position,
            census={
                "n_tracklets": 10,
                "n_divided": divided,
                "n_censored": 10 - divided,
                "n_excluded": 0,
                "n_long_not_divided": 3 if not divided else 0,
                "n_arrested": 3 if not divided else 0,
                "fraction_divided": (1.0 if divided else 0.0),
            },
        )
    rs = ResultSet.open([root]).with_conditions({"grow": "1-2", "starved": "3"})
    with pytest.warns(UserWarning, match="starved: no cell divided"):
        table = division_census_table(rs)
    assert table.loc["grow", "n_divided"] == 9
    assert table.loc["starved", "fraction_divided"] == 0.0
    assert table.loc["starved", "n_arrested"] == 3


def test_natural_sorted_mixed_labels():
    from acia.analysis.results import natural_sorted

    assert natural_sorted(["10 mM", "high", "0.1 mM", "1 mM"]) == [
        "0.1 mM",
        "1 mM",
        "10 mM",
        "high",
    ]
