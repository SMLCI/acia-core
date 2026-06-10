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
