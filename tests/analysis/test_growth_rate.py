"""Tests for :func:`acia.analysis.estimate_growth_rate`."""

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acia import ureg
from acia.analysis import GrowthRateResult, estimate_growth_rate


def _exp_df(
    Q0: float,
    mu: float,
    t: np.ndarray,
    *,
    rows_per_time: int = 1,
    time_unit: str = "hour",
    area_unit: str = "micrometer ** 2",
) -> pd.DataFrame:
    """Build a synthetic extractor-style DataFrame with ``area = Q0*exp(mu*t)``.

    ``rows_per_time`` rows are emitted per time point so that ``agg="sum"``
    yields ``rows_per_time * Q0 * exp(mu*t)`` and ``agg="count"`` yields a
    constant per-time row count.
    """
    times = np.repeat(t, rows_per_time)
    area = Q0 * np.exp(mu * times)
    df = pd.DataFrame({"time": times, "area": area})
    df.attrs["units"] = {"time": time_unit, "area": area_unit}
    return df


class TestEstimateGrowthRate(unittest.TestCase):
    """Cover every row of the spec's I/O & edge-case matrix."""

    def test_exact_recovery_sum(self):
        Q0, mu = 5.0, 0.3
        t = np.linspace(0, 10, 11)
        df = _exp_df(Q0, mu, t)

        result, fig = estimate_growth_rate(df, agg="sum")
        plt.close(fig)

        self.assertIsInstance(result, GrowthRateResult)
        self.assertAlmostEqual(result.growth_rate.to("1/hour").magnitude, mu, places=6)
        self.assertAlmostEqual(result.initial_value, Q0, places=4)
        self.assertAlmostEqual(result.r_squared, 1.0, places=8)
        # SE ~ 0 for noise-free data
        self.assertLess(result.growth_rate_std_err.to("1/hour").magnitude, 1e-6)
        # CI brackets mu
        lo = result.growth_rate_ci[0].to("1/hour").magnitude
        hi = result.growth_rate_ci[1].to("1/hour").magnitude
        self.assertLessEqual(lo, mu)
        self.assertLessEqual(mu, hi)
        # doubling time = ln(2)/mu
        self.assertAlmostEqual(
            result.doubling_time.to("hour").magnitude, np.log(2) / mu, places=5
        )

    def test_units_are_quantities(self):
        df = _exp_df(2.0, 0.2, np.linspace(0, 5, 6))
        result, fig = estimate_growth_rate(df)
        plt.close(fig)

        self.assertEqual(result.growth_rate.units, (1 / ureg.hour).units)
        self.assertEqual(result.growth_rate_std_err.units, (1 / ureg.hour).units)
        self.assertEqual(result.doubling_time.units, ureg.hour)
        self.assertEqual(result.growth_rate_ci[0].units, (1 / ureg.hour).units)

    def test_count_mode(self):
        # cell count grows exponentially: round(N0*exp(mu*t)) rows per time
        N0, mu = 2.0, 0.3
        t = np.arange(0, 11, dtype=float)
        times = np.concatenate(
            [np.full(int(round(N0 * np.exp(mu * ti))), ti) for ti in t]
        )
        df = pd.DataFrame({"time": times, "area": np.ones_like(times)})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}

        result, fig = estimate_growth_rate(df, agg="count")
        plt.close(fig)

        # count is the group size -> recovers the count growth rate approximately
        self.assertAlmostEqual(result.growth_rate.to("1/hour").magnitude, mu, places=1)
        self.assertGreater(result.r_squared, 0.99)

    def test_mean_mode(self):
        Q0, mu = 3.0, 0.25
        t = np.linspace(0, 6, 7)
        # duplicate rows per time: the mean still recovers Q0*exp(mu*t)
        df = _exp_df(Q0, mu, t, rows_per_time=3)

        result, fig = estimate_growth_rate(df, agg="mean")
        plt.close(fig)

        self.assertAlmostEqual(result.growth_rate.to("1/hour").magnitude, mu, places=6)
        self.assertAlmostEqual(result.initial_value, Q0, places=4)

    def test_noisy_data_se_and_ci(self):
        rng = np.random.default_rng(42)
        Q0, mu = 10.0, 0.4
        t = np.linspace(0, 10, 30)
        y = Q0 * np.exp(mu * t) * np.exp(rng.normal(0, 0.1, size=t.size))
        df = pd.DataFrame({"time": t, "area": y})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}

        result, fig = estimate_growth_rate(df, agg="sum")
        plt.close(fig)

        se = result.growth_rate_std_err.to("1/hour").magnitude
        lo = result.growth_rate_ci[0].to("1/hour").magnitude
        hi = result.growth_rate_ci[1].to("1/hour").magnitude
        gr = result.growth_rate.to("1/hour").magnitude

        self.assertGreater(se, 0.0)
        self.assertGreater(hi - lo, 0.0)
        self.assertLessEqual(lo, gr)
        self.assertLessEqual(gr, hi)

    def test_ci_level_widens(self):
        rng = np.random.default_rng(7)
        t = np.linspace(0, 10, 30)
        y = 5.0 * np.exp(0.3 * t) * np.exp(rng.normal(0, 0.1, size=t.size))
        df = pd.DataFrame({"time": t, "area": y})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}

        res95, fig95 = estimate_growth_rate(df, ci_level=0.95)
        res99, fig99 = estimate_growth_rate(df, ci_level=0.99)
        plt.close(fig95)
        plt.close(fig99)

        width95 = (
            (res95.growth_rate_ci[1] - res95.growth_rate_ci[0]).to("1/hour").magnitude
        )
        width99 = (
            (res99.growth_rate_ci[1] - res99.growth_rate_ci[0]).to("1/hour").magnitude
        )
        self.assertGreater(width99, width95)

    def test_minute_unit_inference(self):
        df = _exp_df(2.0, 0.05, np.linspace(0, 20, 11), time_unit="minute")
        result, fig = estimate_growth_rate(df)
        plt.close(fig)

        self.assertEqual(result.growth_rate.units, (1 / ureg.minute).units)
        self.assertEqual(result.growth_rate_std_err.units, (1 / ureg.minute).units)
        self.assertEqual(result.growth_rate_ci[0].units, (1 / ureg.minute).units)
        self.assertEqual(result.doubling_time.units, ureg.minute)
        self.assertAlmostEqual(
            result.growth_rate.to("1/minute").magnitude, 0.05, places=6
        )

    def test_time_unit_arg_fallback(self):
        # no attrs units -> falls back to the explicit time_unit argument
        t = np.linspace(0, 5, 6)
        df = pd.DataFrame({"time": t, "area": 2.0 * np.exp(0.2 * t)})
        result, fig = estimate_growth_rate(df, time_unit="minute")
        plt.close(fig)
        self.assertEqual(result.growth_rate.units, (1 / ureg.minute).units)

    def test_default_time_unit_hour(self):
        # no attrs units, no time_unit arg -> defaults to hour
        t = np.linspace(0, 5, 6)
        df = pd.DataFrame({"time": t, "area": 2.0 * np.exp(0.2 * t)})
        result, fig = estimate_growth_rate(df)
        plt.close(fig)
        self.assertEqual(result.growth_rate.units, (1 / ureg.hour).units)

    def test_non_positive_value_raises(self):
        df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "area": [1.0, -2.0, 3.0]})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}
        with self.assertRaises(ValueError):
            estimate_growth_rate(df, agg="sum")

    def test_too_few_points_raises(self):
        df = pd.DataFrame({"time": [1.0, 1.0, 1.0], "area": [1.0, 2.0, 3.0]})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}
        with self.assertRaises(ValueError):
            estimate_growth_rate(df, agg="sum")

    def test_figure_returned(self):
        from matplotlib.figure import Figure

        df = _exp_df(2.0, 0.2, np.linspace(0, 5, 6))
        result, fig = estimate_growth_rate(df)
        self.assertIsInstance(fig, Figure)
        # data + fit -> at least 2 plotted artists present
        self.assertGreaterEqual(len(fig.axes), 1)
        plt.close(fig)

    def test_accepts_external_ax(self):
        fig, ax = plt.subplots()
        df = _exp_df(2.0, 0.2, np.linspace(0, 5, 6))
        result, out_fig = estimate_growth_rate(df, ax=ax)
        self.assertIs(out_fig, fig)
        plt.close(fig)


class TestReviewRegressions(unittest.TestCase):
    def test_decay_gives_nan_doubling_time(self):
        # declining population: negative growth rate, doubling time undefined (nan)
        t = np.linspace(0, 10, 11)
        df = pd.DataFrame({"time": t, "area": 100.0 * np.exp(-0.2 * t)})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}
        result, fig = estimate_growth_rate(df)
        plt.close(fig)
        self.assertLess(result.growth_rate.to("1/hour").magnitude, 0.0)
        self.assertTrue(np.isnan(result.doubling_time.magnitude))

    def test_two_points_se_ci_are_nan_without_warning(self):
        df = pd.DataFrame({"time": [0.0, 1.0], "area": [1.0, 2.0]})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any leaked RuntimeWarning fails the test
            result, fig = estimate_growth_rate(df)
        plt.close(fig)
        self.assertTrue(np.isnan(result.growth_rate_std_err.magnitude))
        self.assertTrue(np.isnan(result.growth_rate_ci[0].magnitude))

    def test_non_finite_value_raises(self):
        df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "area": [1.0, np.nan, 3.0]})
        df.attrs["units"] = {"time": "hour", "area": "micrometer ** 2"}
        with self.assertRaises(ValueError):
            estimate_growth_rate(df)

    def test_invalid_ci_level_raises(self):
        df = _exp_df(2.0, 0.2, np.linspace(0, 5, 6))
        for bad in (0.0, 1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                estimate_growth_rate(df, ci_level=bad)

    def test_unparseable_time_unit_raises(self):
        t = np.linspace(0, 5, 6)
        df = pd.DataFrame({"time": t, "area": 2.0 * np.exp(0.2 * t)})
        df.attrs["units"] = {"time": "bogus_unit_xyz", "area": "micrometer ** 2"}
        with self.assertRaises(ValueError):
            estimate_growth_rate(df)

    def test_missing_value_col_raises(self):
        df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "size": [1.0, 2.0, 4.0]})
        with self.assertRaises(ValueError):
            estimate_growth_rate(df, value_col="area", agg="sum")


if __name__ == "__main__":
    unittest.main()


class TestCellGrowthRates(unittest.TestCase):
    """compute_cell_growth_rates on synthetic exponentially growing cells."""

    def _setup(self, noisy_label=None):
        import networkx as nx

        from acia.analysis.doubling_time import compute_doubling_times
        from acia.base import Contour, Overlay
        from acia.segm.local import THWCSequenceSource

        # 10 min frames; mother 1 (frames 0-5) divides at frame 6 into 2 and 3,
        # which live frames 6-11 and divide at 12 into 4..7 (frames 12-13).
        # A cell doubling every 60 min grows at mu = ln2 / h.
        mu = np.log(2)  # per hour
        g = nx.DiGraph()
        spans = {0: (0, 0), 1: (0, 5), 2: (6, 11), 3: (6, 11)}
        for k in (4, 5, 6, 7):
            spans[k] = (12, 13)
        for n, (a, b) in spans.items():
            g.add_node(n, start_frame=a, end_frame=b)
        g.add_edges_from(
            [(0, 1), (0, 8), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
        )
        g.add_node(8, start_frame=1, end_frame=13)

        contours = []
        next_id = 0
        for n in (1, 2, 3):
            a, b = spans[n]
            for f in range(a, b + 1):
                side = np.sqrt(100 * np.exp(mu * (f - a) / 6))
                if n == noisy_label and f % 2:
                    side *= 1.6
                sq = [[0, 0], [side, 0], [side, side], [0, side]]
                contours.append(Contour(sq, -1, f, next_id, n))
                next_id += 1
        src = THWCSequenceSource(
            np.zeros((14, 4, 4, 1), np.uint8), frame_interval="10 min"
        )
        cycles = compute_doubling_times(g, src)
        return Overlay(contours, frames=list(range(14))), cycles, src

    def test_fit_matches_doubling_time_for_clean_cells(self):
        from acia.analysis.growth_rate import compute_cell_growth_rates

        overlay, cycles, src = self._setup()
        rates = compute_cell_growth_rates(overlay, cycles, src)

        self.assertEqual(sorted(rates.index), [1, 2, 3])
        mu_fit = rates["mu_fit"].values.quantity.to("1/hour").magnitude
        np.testing.assert_allclose(mu_fit, np.log(2), rtol=1e-6)
        np.testing.assert_allclose(
            rates["mu_from_dt"].values.quantity.magnitude, np.log(2), rtol=1e-6
        )
        self.assertTrue((rates["r2"] > 0.999).all())
        self.assertFalse(rates["fit_outlier"].any())
        self.assertTrue((rates["n_points"] == 6).all())

    def test_noisy_cell_is_flagged(self):
        from acia.analysis.growth_rate import compute_cell_growth_rates

        overlay, cycles, src = self._setup(noisy_label=2)
        rates = compute_cell_growth_rates(overlay, cycles, src)

        self.assertTrue(rates.loc[2, "fit_outlier"])
        self.assertEqual(rates.loc[2, "fit_outlier_reason"], "poor fit")
        self.assertFalse(rates.loc[1, "fit_outlier"])

    def test_too_few_points(self):
        from acia.analysis.growth_rate import compute_cell_growth_rates

        overlay, cycles, src = self._setup()
        rates = compute_cell_growth_rates(overlay, cycles, src, min_points=10)
        self.assertTrue(rates["mu_fit"].isna().all())
        self.assertEqual(set(rates["fit_outlier_reason"]), {"too few points"})
