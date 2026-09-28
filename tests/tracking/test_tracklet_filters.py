"""Tests for :mod:`acia.tracking.filters`."""

import numpy as np
import pandas as pd
import pytest

from acia.analysis.units import strip_units
from acia.tracking.filters import (
    TrackletFilter,
    apply_tracklet_filters,
    duration_filter,
    filter_report,
)


def _cycles():
    return pd.DataFrame(
        {
            "doubling_time": pd.array([10.0, 30.0, 60.0, 120.0, 200.0], "pint[minute]"),
            "r2": [0.99, 0.5, 0.95, np.nan, 0.9],
        },
        index=pd.Index([1, 2, 3, 4, 5], name="tracklet"),
    )


def test_duration_bounds_are_inclusive_and_unit_aware():
    df = apply_tracklet_filters(_cycles(), [duration_filter("0.5 h", "120 min")])
    assert list(df["valid"]) == [False, True, True, True, False]
    assert list(df["rejected_by"]) == ["duration", None, None, None, "duration"]


def test_first_failing_filter_is_reported_and_input_untouched():
    cycles = _cycles()
    df = apply_tracklet_filters(
        cycles,
        [duration_filter(max="150 min"), TrackletFilter("r2", min=0.8)],
    )
    assert list(df["rejected_by"]) == [None, "r2", None, "r2", "duration"]
    assert "valid" not in cycles.columns

    report = filter_report(df)
    assert report["r2"] == 2 and report["duration"] == 1
    assert report["valid"] == 2 and report["total"] == 5


def test_stripped_float_columns_use_recorded_units():
    stripped, units = strip_units(_cycles())
    stripped.attrs["units"] = units
    df = apply_tracklet_filters(stripped, [duration_filter("1 h", None)])
    assert list(df["valid"]) == [False, False, True, True, True]


def test_no_filters_marks_everything_valid_and_bad_input_raises():
    assert apply_tracklet_filters(_cycles(), [])["valid"].all()
    with pytest.raises(KeyError, match="no such column"):
        apply_tracklet_filters(_cycles(), [TrackletFilter("length", 1)])
    with pytest.raises(ValueError, match="has a unit"):
        apply_tracklet_filters(_cycles(), [TrackletFilter("r2", "3 min")])
