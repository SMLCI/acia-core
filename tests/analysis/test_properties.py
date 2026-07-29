"""Tests for :func:`acia.analysis.properties.plot_property_histograms`."""

import unittest

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acia import ureg
from acia.analysis.properties import plot_property_histograms


def _hist_density_sum(ax) -> float:
    """Sum(bin_height * bin_width) over every bar patch drawn on ``ax``."""
    return sum(p.get_height() * p.get_width() for p in ax.patches)


def _make_df(n: int, seed: int, **columns_and_scales) -> pd.DataFrame:
    """Build a DataFrame of ``n`` normally-distributed rows per column."""
    rng = np.random.default_rng(seed)
    data = {
        col: rng.normal(loc=scale, scale=scale / 5.0, size=n)
        for col, scale in columns_and_scales.items()
    }
    return pd.DataFrame(data)


class TestPlotPropertyHistograms(unittest.TestCase):
    """Cover every row of the spec's I/O & edge-case matrix."""

    def test_single_property_no_after_1x1_grid(self):
        df = _make_df(200, seed=1, area=100.0)
        fig = plot_property_histograms(df, ["area"])
        self.assertEqual(fig.axes[0].get_subplotspec().get_gridspec().nrows, 1)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_multi_property_no_after_1xN_grid(self):
        df = _make_df(200, seed=2, area=100.0, length=20.0, width=10.0)
        fig = plot_property_histograms(df, ["area", "length", "width"])
        gs = fig.axes[0].get_subplotspec().get_gridspec()
        self.assertEqual(gs.nrows, 1)
        self.assertEqual(gs.ncols, 3)
        self.assertEqual(len(fig.axes), 3)
        plt.close(fig)

    def test_before_after_single_property_2x1_shared_bins_and_limits(self):
        df_before = _make_df(300, seed=3, area=100.0)
        df_after = df_before.iloc[:150].copy()  # a filtered subset

        fig = plot_property_histograms(df_before, ["area"], df_after=df_after)
        gs = fig.axes[0].get_subplotspec().get_gridspec()
        self.assertEqual(gs.nrows, 2)
        self.assertEqual(gs.ncols, 1)
        self.assertEqual(len(fig.axes), 2)

        ax_before, ax_after = fig.axes[0], fig.axes[1]
        self.assertEqual(ax_before.get_xlim(), ax_after.get_xlim())
        self.assertEqual(ax_before.get_ylim(), ax_after.get_ylim())

        # bin edges identical: recover them from the patch x-positions
        before_edges = sorted(p.get_x() for p in ax_before.patches)
        after_edges = sorted(p.get_x() for p in ax_after.patches)
        np.testing.assert_allclose(before_edges, after_edges)
        plt.close(fig)

    def test_before_after_multi_property_2xN_independent_columns(self):
        df_before = _make_df(300, seed=4, area=100.0, length=20.0)
        df_after = df_before.iloc[:120].copy()

        fig = plot_property_histograms(df_before, ["area", "length"], df_after=df_after)
        gs = fig.axes[0].get_subplotspec().get_gridspec()
        self.assertEqual(gs.nrows, 2)
        self.assertEqual(gs.ncols, 2)
        self.assertEqual(len(fig.axes), 4)

        # fig.axes order for a (2, 2) grid created via subplots is row-major
        area_before, length_before, area_after, length_after = fig.axes

        # each column shares limits...
        self.assertEqual(area_before.get_xlim(), area_after.get_xlim())
        self.assertEqual(length_before.get_xlim(), length_after.get_xlim())

        # ...independently of the other column (different property scales)
        self.assertNotEqual(area_before.get_xlim(), length_before.get_xlim())
        plt.close(fig)

    def test_density_normalization(self):
        df_before = _make_df(300, seed=5, area=100.0, length=20.0)
        df_after = df_before.iloc[:100].copy()

        fig = plot_property_histograms(df_before, ["area", "length"], df_after=df_after)
        for ax in fig.axes:
            self.assertAlmostEqual(_hist_density_sum(ax), 1.0, places=6)
        plt.close(fig)

        df = _make_df(200, seed=6, area=100.0)
        fig2 = plot_property_histograms(df, ["area"])
        self.assertAlmostEqual(_hist_density_sum(fig2.axes[0]), 1.0, places=6)
        plt.close(fig2)

    def test_log_y_sets_log_scale_on_every_axes(self):
        df_before = _make_df(200, seed=7, area=100.0, length=20.0)
        df_after = df_before.iloc[:80].copy()

        fig = plot_property_histograms(
            df_before, ["area", "length"], df_after=df_after, log_y=True
        )
        for ax in fig.axes:
            self.assertEqual(ax.get_yscale(), "log")
        plt.close(fig)

        # default (log_y=False) stays linear
        fig2 = plot_property_histograms(df_before, ["area", "length"])
        for ax in fig2.axes:
            self.assertEqual(ax.get_yscale(), "linear")
        plt.close(fig2)

    def test_units_formatting_uses_pint_pretty_unicode(self):
        # ~P (pretty Unicode, e.g. "µm²") is used rather than ~L (LaTeX):
        # matplotlib only interprets LaTeX inside $...$, so a raw ~L string would
        # render as literal "\mathrm{...}" markup on the axis.
        df = _make_df(200, seed=8, area=100.0)
        area_unit = ureg.Unit("micrometer ** 2")
        fig = plot_property_histograms(df, ["area"], units={"area": area_unit})
        label = fig.axes[0].get_xlabel()
        self.assertIn(f"{area_unit:~P}", label)
        self.assertNotIn("\\mathrm", label)
        self.assertTrue(label.startswith("area ["))
        plt.close(fig)

    def test_axes_have_grid(self):
        # every subplot gets a readability grid, both single-row and
        # before/after (two-row) layouts
        df = _make_df(200, seed=11, area=100.0)
        for kwargs in ({}, {"df_after": _make_df(80, seed=12, area=100.0)}):
            fig = plot_property_histograms(df, ["area"], **kwargs)
            for ax in fig.axes:
                self.assertTrue(any(line.get_visible() for line in ax.get_xgridlines()))
                self.assertTrue(any(line.get_visible() for line in ax.get_ygridlines()))
            plt.close(fig)

    def test_units_missing_falls_back_to_bare_property_name(self):
        df = _make_df(200, seed=9, area=100.0, length=20.0)

        # units=None
        fig = plot_property_histograms(df, ["area"])
        self.assertEqual(fig.axes[0].get_xlabel(), "area")
        plt.close(fig)

        # units={} (property absent from mapping)
        fig2 = plot_property_histograms(df, ["length"], units={})
        self.assertEqual(fig2.axes[0].get_xlabel(), "length")
        plt.close(fig2)

        # units given for a different property only
        fig3 = plot_property_histograms(
            df, ["length"], units={"area": ureg.Unit("micrometer ** 2")}
        )
        self.assertEqual(fig3.axes[0].get_xlabel(), "length")
        plt.close(fig3)

    def test_unknown_property_raises_key_error(self):
        df = _make_df(50, seed=10, area=100.0)
        with self.assertRaises(KeyError):
            plot_property_histograms(df, ["not_a_column"])

    def test_dimensionless_unit_does_not_crash(self):
        df = _make_df(100, seed=11, circularity=1.0)
        fig = plot_property_histograms(
            df, ["circularity"], units={"circularity": ureg.Unit("1")}
        )
        label = fig.axes[0].get_xlabel()
        self.assertTrue(label.startswith("circularity ["))
        plt.close(fig)

    def test_empty_properties_raises_value_error_no_figure_leak(self):
        df = _make_df(50, seed=30, area=100.0)
        before_fignums = plt.get_fignums()
        with self.assertRaises(ValueError):
            plot_property_histograms(df, [])
        self.assertEqual(plt.get_fignums(), before_fignums)

    def test_string_properties_raises_type_error(self):
        df = _make_df(50, seed=31, area=100.0)
        with self.assertRaises(TypeError):
            plot_property_histograms(df, "area")

    def test_missing_column_in_df_after_raises_key_error_no_figure_leak(self):
        df_before = _make_df(100, seed=32, area=100.0, length=20.0)
        df_after = df_before[["area"]].iloc[:50].copy()  # missing "length"

        before_fignums = plt.get_fignums()
        with self.assertRaises(KeyError):
            plot_property_histograms(df_before, ["area", "length"], df_after=df_after)
        self.assertEqual(plt.get_fignums(), before_fignums)

    def test_empty_df_after_raises_value_error_no_finite_values_no_leak(self):
        df_before = _make_df(50, seed=33, area=100.0)
        df_after = df_before.iloc[:0].copy()  # 0 rows

        before_fignums = plt.get_fignums()
        with self.assertRaises(ValueError) as ctx:
            plot_property_histograms(df_before, ["area"], df_after=df_after)
        self.assertIn("no finite values", str(ctx.exception))
        self.assertEqual(plt.get_fignums(), before_fignums)

    def test_nan_values_dropped_and_still_plot_both_paths(self):
        df_before = _make_df(200, seed=34, area=100.0)
        df_before.loc[0:5, "area"] = np.nan
        df_after = df_before.iloc[:100].copy()
        df_after.loc[10, "area"] = np.inf

        # df_after given
        fig = plot_property_histograms(df_before, ["area"], df_after=df_after)
        ax_before, ax_after = fig.axes
        self.assertAlmostEqual(_hist_density_sum(ax_before), 1.0, places=6)
        self.assertAlmostEqual(_hist_density_sum(ax_after), 1.0, places=6)
        plt.close(fig)

        # no df_after
        fig2 = plot_property_histograms(df_before, ["area"])
        self.assertAlmostEqual(_hist_density_sum(fig2.axes[0]), 1.0, places=6)
        plt.close(fig2)

    def test_all_nan_column_raises_value_error_no_leak(self):
        df = _make_df(50, seed=35, area=100.0)
        df["area"] = np.nan

        before_fignums = plt.get_fignums()
        with self.assertRaises(ValueError) as ctx:
            plot_property_histograms(df, ["area"])
        self.assertIn("no finite values", str(ctx.exception))
        self.assertEqual(plt.get_fignums(), before_fignums)

    def test_no_after_mode_sets_title_per_axes(self):
        df = _make_df(100, seed=36, area=100.0, length=20.0)
        fig = plot_property_histograms(df, ["area", "length"])
        for ax, prop in zip(fig.axes, ["area", "length"], strict=True):
            self.assertEqual(ax.get_title(), prop)
        plt.close(fig)

    def test_custom_bins_count_produces_exact_bar_count(self):
        df = _make_df(200, seed=37, area=100.0)
        fig = plot_property_histograms(df, ["area"], bins=10)
        self.assertEqual(len(fig.axes[0].patches), 10)
        plt.close(fig)


class TestAcceptanceCriteria(unittest.TestCase):
    """Directly exercise the spec's three Acceptance Criteria."""

    def test_no_after_single_property_returns_one_axes(self):
        df = _make_df(100, seed=20, area=100.0)
        fig = plot_property_histograms(df, ["area"])
        self.assertEqual(len(fig.axes), 1)
        gs = fig.axes[0].get_subplotspec().get_gridspec()
        self.assertEqual((gs.nrows, gs.ncols), (1, 1))
        plt.close(fig)

    def test_two_property_before_after_2x2_grid_shared_column_limits(self):
        df_before = _make_df(300, seed=21, area=100.0, length=20.0)
        df_after = df_before.iloc[:150].copy()

        fig = plot_property_histograms(df_before, ["area", "length"], df_after=df_after)
        gs = fig.axes[0].get_subplotspec().get_gridspec()
        self.assertEqual((gs.nrows, gs.ncols), (2, 2))
        self.assertEqual(len(fig.axes), 4)

        area_before, length_before, area_after, length_after = fig.axes
        self.assertEqual(area_before.get_xlim(), area_after.get_xlim())
        self.assertEqual(area_before.get_ylim(), area_after.get_ylim())
        self.assertEqual(length_before.get_xlim(), length_after.get_xlim())
        self.assertEqual(length_before.get_ylim(), length_after.get_ylim())
        plt.close(fig)

    def test_units_dict_from_extractor_execute_style_produces_unicode_label(self):
        df = _make_df(100, seed=22, area=100.0)
        ex_units = {"area": ureg.Unit("micrometer ** 2")}
        fig = plot_property_histograms(df, ["area"], units=ex_units)
        label = fig.axes[0].get_xlabel()
        self.assertIn(f"{ex_units['area']:~P}", label)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()


class TestShowRemoved(unittest.TestCase):
    def test_show_removed_overlays_red_and_legend(self):
        df_before = _make_df(300, seed=1, area=10.0)
        df_after = df_before.iloc[:150].copy()  # index-subset of df_before
        fig = plot_property_histograms(
            df_before, ["area"], df_after=df_after, show_removed=True
        )
        ax_before, ax_after = fig.axes[0], fig.axes[1]
        # the removed step outline adds exactly one extra artist on the after axis
        self.assertEqual(len(ax_after.patches), len(ax_before.patches) + 1)
        leg = ax_after.get_legend()
        self.assertIsNotNone(leg)
        self.assertEqual({t.get_text() for t in leg.get_texts()}, {"kept", "removed"})
        plt.close(fig)

    def test_show_removed_default_off(self):
        df_before = _make_df(200, seed=2, area=10.0)
        df_after = df_before.iloc[:100].copy()
        fig = plot_property_histograms(df_before, ["area"], df_after=df_after)
        self.assertIsNone(fig.axes[1].get_legend())
        self.assertEqual(len(fig.axes[0].patches), len(fig.axes[1].patches))
        plt.close(fig)


class TestEmptyPopulation(unittest.TestCase):
    def test_empty_before_returns_labeled_grid_no_raise(self):
        df = pd.DataFrame({"area": [], "length": []})
        fig = plot_property_histograms(df, ["area", "length"])
        self.assertEqual(len(fig.axes), 2)  # 1 row x 2 props
        plt.close(fig)

    def test_empty_before_and_after_two_rows(self):
        df = pd.DataFrame({"area": []})
        fig = plot_property_histograms(df, ["area"], df_after=df.copy())
        self.assertEqual(len(fig.axes), 2)  # 2 rows x 1 prop
        plt.close(fig)

    def test_empty_still_validates_property_name(self):
        df = pd.DataFrame({"area": []})
        with self.assertRaises(KeyError):
            plot_property_histograms(df, ["nope"])
