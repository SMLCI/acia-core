"""Tests for :func:`acia.analysis.extract_growth` (extract + growth-rate fit)."""

import numpy as np
import pytest

from acia.analysis import extract_growth
from acia.analysis.growth_rate import GrowthRateResult
from acia.base import Contour, Overlay
from acia.segm.local import THWCSequenceSource

MU = 0.4  # ground-truth growth rate (1/hour)


def _source(t=6):
    return THWCSequenceSource(
        np.zeros((t, 80, 80, 1), dtype=np.uint8),
        pixel_size="1 micrometer",
        frame_interval="1 hour",
    )


def _exponential_overlay(t=6, per_frame=3):
    """Squares whose total area per frame grows as exp(MU * t)."""
    contours = []
    cid = 0
    for frame in range(t):
        side = float(np.sqrt(np.exp(MU * frame)) * 4.0)  # area = 16 * exp(MU*t)
        for _ in range(per_frame):
            contours.append(
                Contour(
                    [[0, 0], [side, 0], [side, side], [0, side]],
                    score=-1,
                    frame=frame,
                    id=cid,
                )
            )
            cid += 1
    return Overlay(contours)


def test_extract_growth_returns_table_result_figure():
    table, result, figure = extract_growth(_exponential_overlay(), _source())

    assert {"frame", "time", "area"}.issubset(table.columns)
    assert isinstance(result, GrowthRateResult)
    assert figure is not None  # a matplotlib Figure
    # ground-truth growth rate recovered from the synthetic exponential
    assert result.growth_rate.to("1/hour").magnitude == pytest.approx(MU, abs=1e-3)
    assert result.r_squared > 0.999


def test_extract_growth_respects_time_unit_and_agg():
    # count aggregation: a constant per-frame cell count -> ~zero growth, but the
    # call must still run and honor the requested time unit on the table.
    table, result, _ = extract_growth(
        _exponential_overlay(), _source(), time_unit="minute", agg="count"
    )
    assert table.attrs["units"]["time"] == "minute"
