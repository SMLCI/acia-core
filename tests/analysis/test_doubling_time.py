"""Tests for :mod:`acia.analysis.doubling_time`."""

import unittest

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; no display required

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from acia import Q_, ureg
from acia.analysis.doubling_time import (
    _compute_hose_bands,
    compute_doubling_times,
    plot_doubling_time_hose,
)
from acia.segm.local import THWCSequenceSource


def _src(t=20, frame_interval=None, timepoints=None):
    stack = np.zeros((t, 4, 4, 1), dtype=np.uint8)
    return THWCSequenceSource(
        stack, frame_interval=frame_interval, timepoints=timepoints
    )


def _mixed_lineage_graph() -> nx.DiGraph:
    """One clean division plus each exclusion case isolated to its own tracklet.

    - 1: in_degree 0, out_degree 2 -> excluded (root, no identified mother)
    - 2: in_degree 1, out_degree 0 -> excluded (still alive / lost track)
    - 3: in_degree 1, out_degree 2 -> INCLUDED (clean division), children 30, 31
    - 30, 31: in_degree 1, out_degree 0 -> excluded (still alive)
    - 40: in_degree 0, out_degree 1 -> excluded (root, and not a division anyway)
    - 41: in_degree 1, out_degree 3 -> excluded (merge/split artifact, not a clean split)
    - 42, 43, 44: in_degree 1, out_degree 0 -> excluded (still alive)
    """
    g = nx.DiGraph()
    g.add_node(1, start_frame=0, end_frame=2)
    g.add_node(2, start_frame=3, end_frame=10)
    g.add_node(3, start_frame=3, end_frame=5)
    g.add_edge(1, 2)
    g.add_edge(1, 3)

    g.add_node(30, start_frame=6, end_frame=12)
    g.add_node(31, start_frame=6, end_frame=12)
    g.add_edge(3, 30)
    g.add_edge(3, 31)

    g.add_node(40, start_frame=0, end_frame=2)
    g.add_node(41, start_frame=3, end_frame=5)
    g.add_node(42, start_frame=6, end_frame=10)
    g.add_node(43, start_frame=6, end_frame=10)
    g.add_node(44, start_frame=6, end_frame=10)
    g.add_edge(40, 41)
    g.add_edge(41, 42)
    g.add_edge(41, 43)
    g.add_edge(41, 44)

    return g


class TestComputeDoublingTimes(unittest.TestCase):
    """Every relevant I/O Matrix row for CAP-2."""

    def test_uncalibrated_source_raises_before_touching_graph(self):
        src = _src()  # no frame_interval, no timepoints -> uncalibrated
        self.assertIsNone(src.timepoints)

        class ExplodingGraph:
            """Any attribute access is a test failure -- graph must not be touched."""

            def __getattr__(self, name):
                raise AssertionError(
                    f"tracklet_graph.{name} accessed before the ValueError check"
                )

        with self.assertRaises(ValueError):
            compute_doubling_times(ExplodingGraph(), src)

    def test_clean_division_included_root_leaf_merge_excluded(self):
        g = _mixed_lineage_graph()
        src = _src(frame_interval=15 * ureg.minute)

        df = compute_doubling_times(g, src)

        self.assertEqual(set(df.index), {3})
        self.assertNotIn(1, df.index)  # root: in_degree == 0
        self.assertNotIn(2, df.index)  # leaf/still-alive: out_degree == 0
        self.assertNotIn(30, df.index)
        self.assertNotIn(31, df.index)
        self.assertNotIn(40, df.index)  # root
        self.assertNotIn(41, df.index)  # merge/split artifact: out_degree == 3
        self.assertNotIn(42, df.index)
        self.assertNotIn(43, df.index)
        self.assertNotIn(44, df.index)

    def test_doubling_time_matches_frame_interval_formula(self):
        # Acceptance Criteria #2: value == (child.start_frame - tracklet.start_frame) * frame_interval
        g = _mixed_lineage_graph()
        interval = 15 * ureg.minute
        src = _src(frame_interval=interval)

        df = compute_doubling_times(g, src)

        expected = (
            6 - 3
        ) * interval  # tracklet 3 starts at frame 3, children at frame 6
        self.assertAlmostEqual(
            df.loc[3, "doubling_time"].to("minute").magnitude,
            expected.to("minute").magnitude,
        )
        self.assertAlmostEqual(df.loc[3, "start_time"].to("minute").magnitude, 3 * 15)
        self.assertAlmostEqual(df.loc[3, "end_time"].to("minute").magnitude, 5 * 15)

    def test_fixed_interval_vs_irregular_timestamps_differ_correctly(self):
        # Same frame deltas, two sources: one via a scalar frame_interval, one
        # via explicit non-uniform timepoints. This proves source.timepoints[...]
        # indexing is used -- not a hand-rolled frame_delta * frame_interval
        # shortcut, which would give the identical (wrong) answer for both.
        g = nx.DiGraph()
        g.add_node(
            0, start_frame=0, end_frame=0
        )  # dummy parent so 1 has in_degree == 1
        g.add_node(1, start_frame=1, end_frame=3)
        g.add_node(2, start_frame=5, end_frame=7)
        g.add_node(3, start_frame=5, end_frame=7)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)

        fixed_src = _src(t=8, frame_interval=10 * ureg.minute)
        # irregular: frame 4 arrives much later than a uniform 10-minute step
        # would predict (40 -> 100), same array length/dtype otherwise
        irregular_tp = Q_(
            np.array([0.0, 10.0, 20.0, 30.0, 100.0, 110.0, 120.0, 130.0]), "minute"
        )
        irregular_src = _src(t=8, timepoints=irregular_tp)

        df_fixed = compute_doubling_times(g, fixed_src)
        df_irregular = compute_doubling_times(g, irregular_src)

        fixed_dt = df_fixed.loc[1, "doubling_time"].to("minute").magnitude
        irregular_dt = df_irregular.loc[1, "doubling_time"].to("minute").magnitude

        self.assertAlmostEqual(fixed_dt, 40.0)  # (5 - 1) frames * 10 minute interval
        self.assertAlmostEqual(
            irregular_dt, 100.0
        )  # timepoints[5] - timepoints[1] = 110 - 10
        self.assertNotAlmostEqual(fixed_dt, irregular_dt)

    def test_deterministic_child_selection_uses_min_start_frame(self):
        # Divergent daughter start_frames (6 vs 9) -- the documented
        # assumption ("both daughters start at the same division event") is
        # violated here on purpose. The fix must deterministically pick the
        # MINIMUM start_frame, regardless of networkx successor iteration
        # order. Add the later-starting child first so a `next(iter(...))`
        # style bug would pick the wrong (start_frame=9) child.
        g = nx.DiGraph()
        g.add_node(0, start_frame=0, end_frame=0)  # dummy parent
        g.add_node(1, start_frame=3, end_frame=5)
        g.add_node(2, start_frame=9, end_frame=15)  # later child, added first
        g.add_node(3, start_frame=6, end_frame=15)  # earlier child, added second
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)

        interval = 15 * ureg.minute
        src = _src(t=20, frame_interval=interval)

        df = compute_doubling_times(g, src)

        expected = (6 - 3) * interval  # must use child 3 (start_frame=6), not child 2
        self.assertAlmostEqual(
            df.loc[1, "doubling_time"].to("minute").magnitude,
            expected.to("minute").magnitude,
        )

        # deterministic across repeated calls too
        df2 = compute_doubling_times(g, src)
        self.assertAlmostEqual(
            df.loc[1, "doubling_time"].to("minute").magnitude,
            df2.loc[1, "doubling_time"].to("minute").magnitude,
        )

    def test_negative_frame_index_raises(self):
        # end_frame=-1 must raise rather than silently numpy-wrap to the
        # array's last timepoint.
        g = nx.DiGraph()
        g.add_node(0, start_frame=0, end_frame=0)  # dummy parent
        g.add_node(1, start_frame=3, end_frame=-1)
        g.add_node(2, start_frame=6, end_frame=8)
        g.add_node(3, start_frame=6, end_frame=8)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        src = _src(t=20, frame_interval=15 * ureg.minute)

        with self.assertRaisesRegex(ValueError, r"tracklet 1.*-1"):
            compute_doubling_times(g, src)

    def test_out_of_range_frame_index_raises(self):
        # end_frame >= len(timepoints) must raise rather than IndexError
        # deep inside numpy, or (for small enough values) silently wrapping.
        g = nx.DiGraph()
        g.add_node(0, start_frame=0, end_frame=0)  # dummy parent
        g.add_node(1, start_frame=3, end_frame=100)
        g.add_node(2, start_frame=6, end_frame=8)
        g.add_node(3, start_frame=6, end_frame=8)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        src = _src(t=20, frame_interval=15 * ureg.minute)

        with self.assertRaisesRegex(ValueError, r"tracklet 1.*100"):
            compute_doubling_times(g, src)

    def test_negative_doubling_time_raises(self):
        # Malformed tracklet: end_frame < start_frame, and the chosen
        # child's start_frame precedes the parent's own start_frame -- the
        # doubling_time computation (child.start_frame - own.start_frame)
        # comes out negative and must raise rather than silently store it.
        g = nx.DiGraph()
        g.add_node(0, start_frame=0, end_frame=0)  # dummy parent
        g.add_node(1, start_frame=5, end_frame=4)  # end_frame < start_frame
        g.add_node(2, start_frame=2, end_frame=8)  # child starts before parent
        g.add_node(3, start_frame=2, end_frame=8)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        src = _src(t=20, frame_interval=15 * ureg.minute)

        with self.assertRaisesRegex(ValueError, r"tracklet 1.*negative doubling time"):
            compute_doubling_times(g, src)


class TestPlotDoublingTimeHose(unittest.TestCase):
    """Every relevant I/O Matrix row for CAP-3."""

    def _synthetic_cells_df(self, n_cells, lifespan_frames, unit="minute"):
        """n_cells qualifying cells, each alive for [i, i+lifespan_frames).

        Each mother gets its own dummy parent tracklet so it qualifies as a
        clean division (in_degree == 1, out_degree == 2) rather than being
        excluded as a root.
        """
        g = nx.DiGraph()
        label = 0
        for i in range(n_cells):
            parent = label
            g.add_node(parent, start_frame=0, end_frame=0)
            label += 1
            mother = label
            g.add_node(mother, start_frame=i, end_frame=i + lifespan_frames)
            label += 1
            c1, c2 = label, label + 1
            g.add_node(
                c1, start_frame=i + lifespan_frames, end_frame=i + lifespan_frames + 5
            )
            g.add_node(
                c2, start_frame=i + lifespan_frames, end_frame=i + lifespan_frames + 5
            )
            g.add_edge(parent, mother)
            g.add_edge(mother, c1)
            g.add_edge(mother, c2)
            label += 2
        n_frames = n_cells + lifespan_frames + 10
        src = _src(t=n_frames, frame_interval=Q_(1, unit))
        df = compute_doubling_times(g, src)
        return df, src

    def test_cross_sectional_contribution(self):
        # A single qualifying cell alive on [start_time, end_time]; its value
        # must count only for grid points inside that interval.
        g = nx.DiGraph()
        g.add_node(-1, start_frame=0, end_frame=0)  # dummy parent for in_degree == 1
        g.add_node(0, start_frame=2, end_frame=5)
        g.add_node(1, start_frame=6, end_frame=8)
        g.add_node(2, start_frame=6, end_frame=8)
        g.add_edge(-1, 0)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        src = _src(t=12, frame_interval=1 * ureg.minute)
        df = compute_doubling_times(g, src)
        self.assertEqual(len(df), 1)

        # min_n=1 so a single alive cell is enough to produce a value
        fig = plot_doubling_time_hose(df, src, min_n=1, n_bootstrap=50)
        ax = fig.axes[0]
        mean_line = ax.get_lines()[0]
        y = mean_line.get_ydata()
        x = mean_line.get_xdata()
        plt.close(fig)

        start_t = df.loc[0, "start_time"].to("minute").magnitude
        end_t = df.loc[0, "end_time"].to("minute").magnitude
        for xi, yi in zip(x, y, strict=False):
            if start_t <= xi <= end_t:
                self.assertFalse(np.isnan(yi))
            else:
                self.assertTrue(np.isnan(yi))

    def test_min_n_nan_gap(self):
        # Only 2 qualifying cells alive at once; min_n=5 -> every point is NaN.
        df, src = self._synthetic_cells_df(n_cells=2, lifespan_frames=3)
        fig = plot_doubling_time_hose(df, src, min_n=5, n_bootstrap=50)
        ax = fig.axes[0]
        mean_line = ax.get_lines()[0]
        y = mean_line.get_ydata()
        plt.close(fig)

        self.assertTrue(np.all(np.isnan(y)))

    def _gap_free_cells_df(self, n_mothers=6):
        """n_mothers all alive over the same wide window -> no NaN gaps anywhere.

        Each mother has its own dummy parent (in_degree == 1) and two
        children whose start_frame varies per mother, giving a spread of
        doubling-time values while every mother stays alive across the whole
        window -- so any time_grid strictly inside [0, 50] always has all
        ``n_mothers`` alive, guaranteeing a single unbroken fill_between
        polygon (no NaN-induced path splitting) that's safe to parse back out
        for a direct numeric check.
        """
        g = nx.DiGraph()
        label = 0
        for i in range(n_mothers):
            parent = label
            g.add_node(parent, start_frame=0, end_frame=0)
            label += 1
            mother = label
            g.add_node(mother, start_frame=0, end_frame=50)
            label += 1
            child_start = 51 + i
            c1, c2 = label, label + 1
            g.add_node(c1, start_frame=child_start, end_frame=child_start + 5)
            g.add_node(c2, start_frame=child_start, end_frame=child_start + 5)
            g.add_edge(parent, mother)
            g.add_edge(mother, c1)
            g.add_edge(mother, c2)
            label += 2
        src = _src(t=60, frame_interval=Q_(1, "minute"))
        df = compute_doubling_times(g, src)
        return df, src

    def test_bootstrap_ci_sanity_and_nesting(self):
        # Numeric sanity is checked directly against the pure computation
        # (_compute_hose_bands) rather than by parsing matplotlib's
        # fill_between polygon geometry, which is an internal rendering
        # detail, not part of the function's documented contract.
        df, src = self._gap_free_cells_df(n_mothers=6)
        time_grid = Q_(np.linspace(10, 40, 7), "minute")

        t_grid, mean_arr, bands, unit_str = _compute_hose_bands(
            df, time_grid, (0.5, 0.95), min_n=3, n_bootstrap=500, random_state=0
        )
        self.assertEqual(unit_str, "minute")
        self.assertTrue(np.all(~np.isnan(mean_arr)))  # no gaps by construction

        lo50, hi50 = bands[0.5]
        lo95, hi95 = bands[0.95]
        for lo, mean, hi in zip(lo50, mean_arr, hi50, strict=True):
            self.assertLessEqual(lo, mean)
            self.assertLessEqual(mean, hi)
        # narrower (0.5) nests inside wider (0.95) at every grid point
        for lo50_i, hi50_i, lo95_i, hi95_i in zip(lo50, hi50, lo95, hi95, strict=True):
            self.assertGreaterEqual(lo50_i, lo95_i)
            self.assertLessEqual(hi50_i, hi95_i)

        # the rendered figure exposes the same mean values via its Line2D
        fig = plot_doubling_time_hose(
            df,
            src,
            time_grid=time_grid,
            min_n=3,
            ci_levels=(0.5, 0.95),
            n_bootstrap=500,
            random_state=0,
        )
        ax = fig.axes[0]
        np.testing.assert_array_almost_equal(ax.get_lines()[0].get_ydata(), mean_arr)
        self.assertEqual(len(ax.collections), 2)
        plt.close(fig)

    def test_multi_level_ci_levels_produces_three_bands(self):
        df, src = self._synthetic_cells_df(n_cells=8, lifespan_frames=3)
        fig = plot_doubling_time_hose(
            df, src, min_n=3, ci_levels=[0.5, 0.8, 0.95], n_bootstrap=200
        )
        self.assertEqual(len(fig.axes[0].collections), 3)
        plt.close(fig)

    def test_accepts_external_ax(self):
        df, src = self._synthetic_cells_df(n_cells=8, lifespan_frames=3)
        fig, ax = plt.subplots()
        out_fig = plot_doubling_time_hose(df, src, ax=ax, min_n=3, n_bootstrap=50)
        self.assertIs(out_fig, fig)
        plt.close(fig)

    def test_time_grid_defaults_to_source_timepoints(self):
        df, src = self._synthetic_cells_df(n_cells=8, lifespan_frames=3)
        fig = plot_doubling_time_hose(df, src, min_n=3, n_bootstrap=50)
        ax = fig.axes[0]
        x = ax.get_lines()[0].get_xdata()
        np.testing.assert_array_almost_equal(x, src.timepoints.magnitude)
        plt.close(fig)

    def test_n_bootstrap_zero_or_negative_raises(self):
        df = pd.DataFrame()
        time_grid = Q_(np.array([0.0, 1.0]), "minute")

        with self.assertRaisesRegex(ValueError, "n_bootstrap"):
            _compute_hose_bands(
                df, time_grid, (0.95,), min_n=5, n_bootstrap=0, random_state=0
            )
        with self.assertRaisesRegex(ValueError, "n_bootstrap"):
            _compute_hose_bands(
                df, time_grid, (0.95,), min_n=5, n_bootstrap=-1, random_state=0
            )

    def test_min_n_zero_raises(self):
        df = pd.DataFrame()
        time_grid = Q_(np.array([0.0, 1.0]), "minute")

        with self.assertRaisesRegex(ValueError, "min_n"):
            _compute_hose_bands(
                df, time_grid, (0.95,), min_n=0, n_bootstrap=50, random_state=0
            )

    def test_duplicate_ci_levels_produce_single_band(self):
        df, src = self._synthetic_cells_df(n_cells=8, lifespan_frames=3)
        fig = plot_doubling_time_hose(
            df, src, min_n=3, ci_levels=(0.95, 0.95), n_bootstrap=200
        )
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections), 1)

        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(legend_labels.count("95% CI"), 1)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
