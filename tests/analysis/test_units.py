"""Tests for the three units representations of property-extractor tables."""

import numpy as np
import pandas as pd
import pint
import pint_pandas
import pytest

from acia import Q_, ureg
from acia.analysis import (
    AreaEx,
    CircularityEx,
    ExtractorExecutor,
    FrameEx,
    LengthEx,
    PerimeterEx,
    attach_units,
    from_header,
    strip_units,
)
from acia.analysis.units import UNIT_ATTR
from acia.base import Contour, Overlay
from acia.segm.local import LocalImageSource

PS = 0.07  # pixel size in micrometer


def _overlay():
    # area 2*3 = 6 px, length 3 px, width 2 px
    return Overlay([Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=0, id=23)])


def _extractors():
    return [
        FrameEx(),
        AreaEx(input_unit=(PS * ureg.micrometer) ** 2),
        LengthEx(input_unit=PS * ureg.micrometer),
        PerimeterEx(input_unit=PS * ureg.micrometer),
        CircularityEx(),
    ]


def _run(units="none"):
    image_source = LocalImageSource.from_array(np.zeros((50, 50)))
    return ExtractorExecutor().execute(
        overlay=_overlay(),
        images=image_source,
        extractors=_extractors(),
        units=units,
    )


def test_default_none_is_float_with_attrs():
    df = _run("none")

    # values are plain floats (unchanged behaviour)
    assert isinstance(df["area"].iloc[0], float)
    assert df["area"].iloc[0] == pytest.approx(6 * PS**2)
    assert df["length"].iloc[0] == pytest.approx(3 * PS)

    # the unit map travels with the table
    assert df.attrs[UNIT_ATTR]["area"] == "micrometer ** 2"
    assert df.attrs[UNIT_ATTR]["length"] == "micrometer"


def test_invalid_units_value():
    with pytest.raises(ValueError):
        _run("furlongs")


def test_pint_mode_is_unit_safe():
    df = _run("pint")

    # dimensioned columns carry the pint dtype
    assert isinstance(df["area"].dtype, pint_pandas.PintType)
    assert str(df["area"].pint.units) == "micrometer ** 2"
    assert df["area"].pint.magnitude.iloc[0] == pytest.approx(6 * PS**2)

    # arithmetic propagates units ...
    ratio = df["area"] / df["length"]
    assert str(ratio.pint.units) == "micrometer"

    # ... and dimensional mismatches raise
    with pytest.raises(pint.DimensionalityError):
        _ = df["area"] + df["length"]

    # dimensionless columns stay plain numbers (index/merge friendly)
    assert not isinstance(df["frame"].dtype, pint_pandas.PintType)
    assert not isinstance(df["circularity"].dtype, pint_pandas.PintType)


def test_attach_strip_roundtrip():
    floats = _run("none")

    pinted = attach_units(floats)
    assert isinstance(pinted["area"].dtype, pint_pandas.PintType)

    stripped, units = strip_units(pinted)
    assert units["area"] == "micrometer ** 2"
    assert not isinstance(stripped["area"].dtype, pint_pandas.PintType)
    np.testing.assert_allclose(stripped["area"].to_numpy(), floats["area"].to_numpy())


def test_header_mode_roundtrip():
    df = _run("header")

    # unit lives in a column-index level, values are plain floats
    assert isinstance(df.columns, pd.MultiIndex)
    assert ("area", "micrometer ** 2") in list(df.columns)
    area_value = df[("area", "micrometer ** 2")].iloc[0]
    assert float(area_value) == pytest.approx(6 * PS**2)

    # recover the unit-safe pint dtype
    requantified = from_header(df)
    assert isinstance(requantified["area"].dtype, pint_pandas.PintType)


def test_registry_consistency_with_quantity():
    df = _run("pint")

    converted = df["area"].pint.to("mm ** 2")
    expected = Q_(6 * PS**2, "micrometer ** 2").to("mm ** 2").magnitude
    assert converted.pint.magnitude.iloc[0] == pytest.approx(expected)


def test_write_read_units_csv_round_trip_and_derived_units(tmp_path):
    import pandas as pd

    from acia.analysis import read_units_csv, write_units_csv

    df = pd.DataFrame(
        {"area": [3.0, 4.0], "time": [10.0, 10.0], "label": ["a", "b"]},
        index=pd.Index([1, 2], name="id"),
    )
    df.attrs["units"] = {"area": "micrometer ** 2", "time": "minute"}

    path = tmp_path / "cells.csv"
    assert write_units_csv(df, path) == str(path)

    back = read_units_csv(path)
    # units survived the file and are back as pint dtypes
    assert "micrometer ** 2" in str(back["area"].dtype)
    assert "minute" in str(back["time"].dtype)
    # non-unit column stays plain
    assert back["label"].tolist() == ["a", "b"]
    # derived column derives its unit automatically
    back["rate"] = back["area"] / back["time"]
    assert "micrometer ** 2 / minute" in str(back["rate"].dtype)
