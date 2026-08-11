"""Python-surface unit tests for ``acia.notebook.FilterExplorer``.

These cover the Python/traitlet surface only: construction (one spec per filter,
unit/data-range, precomputed per-contour values, seeded handles), the calibration
guard, ``.params`` open-bound semantics, ``configured_filters()`` /
``filtered_overlay()``, and ``save()``. The ESM JavaScript (sliders + live mask
recolouring) is exercised separately by the headless Playwright suite.
"""

import json

import numpy as np
import pytest

pytest.importorskip("anywidget")

from acia import Q_  # noqa: E402
from acia.base import Contour, Overlay  # noqa: E402
from acia.notebook import FilterExplorer  # noqa: E402
from acia.segm.filter import (  # noqa: E402
    AreaFilter,
    CircularityFilter,
    LengthFilter,
    apply_cell_filters,
)
from acia.segm.local import THWCSequenceSource  # noqa: E402


def _square(side, *, x0=0.0, y0=0.0, frame=0, id=0):
    return Contour(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        score=-1,
        frame=frame,
        id=id,
    )


def _source(h=200, w=200, t=1, pixel_size="0.5 micrometer"):
    return THWCSequenceSource(
        np.zeros((t, h, w, 1), dtype=np.uint8), pixel_size=pixel_size
    )


def _overlay():
    # at 0.5 µm/px: 4px->4µm², 8px->16µm², 12px->36µm²
    return Overlay(
        [
            _square(4, id="small"),
            _square(8, x0=20, id="mid"),
            _square(12, x0=50, id="big"),
        ]
    )


def _explorer(filters=None):
    if filters is None:
        filters = [AreaFilter(), LengthFilter(), CircularityFilter()]
    overlay, images = _overlay(), _source()
    return FilterExplorer(
        overlay, images, filters, _properties(overlay, images, filters)
    )


# --- construction ------------------------------------------------------------


def test_one_spec_per_filter_with_units_and_ranges():
    fe = _explorer()
    names = [s["name"] for s in fe.filter_specs]
    assert names == ["area", "length", "circularity"]
    by = {s["name"]: s for s in fe.filter_specs}
    assert by["area"]["unit"] == "micrometer ** 2"
    assert by["length"]["unit"] == "micrometer"
    assert by["circularity"]["unit"] == "dimensionless"
    # data ranges span the measured values (area 4..36 µm², length 2..6 µm)
    assert by["area"]["lo"] == pytest.approx(4.0)
    assert by["area"]["hi"] == pytest.approx(36.0)
    assert by["length"]["lo"] == pytest.approx(2.0)
    assert by["length"]["hi"] == pytest.approx(6.0)


def test_precomputed_contour_values_align_with_specs():
    fe = _explorer()
    # three contours, each a [area, length, circularity] vector
    vals = [r["values"] for r in fe.contours]
    assert len(vals) == 3
    areas = sorted(v[0] for v in vals)
    assert areas == pytest.approx([4.0, 16.0, 36.0])
    # squares: circularity = 4*pi*A / P^2 = pi/4
    for v in vals:
        assert v[2] == pytest.approx(np.pi / 4)


def test_image_traits_populated():
    fe = _explorer()
    assert fe.image_b64.startswith("data:image/png;base64,")
    assert fe.image_w == 200
    assert fe.image_h == 200


def test_selection_initialised_open_keep_all():
    fe = _explorer()
    # no filter had bounds -> handles span the full data range == keep everything
    for spec, sel in zip(fe.filter_specs, fe.selection, strict=False):
        assert sel["vmin"] == pytest.approx(spec["lo"])
        assert sel["vmax"] == pytest.approx(spec["hi"])
    assert all(c.id for c in fe.filtered_overlay().contours)  # nothing dropped
    assert len(fe.filtered_overlay().contours) == 3


def test_existing_bounds_seed_handles():
    fe = _explorer([AreaFilter(Q_(5, "um**2"), Q_(30, "um**2"))])
    spec = fe.filter_specs[0]
    sel = fe.selection[0]
    assert sel["vmin"] == pytest.approx(5.0)
    assert sel["vmax"] == pytest.approx(30.0)
    # and seeds stay within [lo, hi]
    assert spec["lo"] <= sel["vmin"] <= sel["vmax"] <= spec["hi"]


def test_out_of_range_seed_is_lossless():
    """A bound outside the data range widens the track instead of being clamped.

    Regression for the silent-corruption bug: ``area >= 50 µm²`` over data
    [4, 36] µm² must NOT collapse to ``area >= 36`` -- the threshold is
    preserved (the track widens to include it) and round-trips through
    ``.params`` untouched.
    """
    fe = _explorer([AreaFilter(Q_(50, "um**2"))])  # vmin=50, vmax=None
    spec = fe.filter_specs[0]
    assert spec["hi"] >= 50.0  # track widened to include the seed
    assert fe.selection[0]["vmin"] == pytest.approx(50.0)  # not clamped to 36
    p = fe.params[0]
    assert p["vmin"] == Q_(50.0, "micrometer ** 2")  # preserved exactly
    assert p["vmax"] is None  # open upper side


def test_below_range_upper_bound_is_lossless():
    """An upper bound below all data (drops everything) round-trips, not clamped up."""
    fe = _explorer([AreaFilter(None, Q_(2, "um**2"))])  # keep area <= 2 µm²
    spec = fe.filter_specs[0]
    assert spec["lo"] <= 2.0
    p = fe.params[0]
    assert p["vmin"] is None
    assert p["vmax"] == Q_(2.0, "micrometer ** 2")  # not clamped up to 4
    # and it really drops every contour (all are >= 4 µm²)
    assert fe.filtered_overlay().contours == []


def test_non_finite_value_is_sanitised_to_zero():
    """A NaN magnitude would serialize as invalid JSON; coerce it to 0.

    ``nan`` is what an extractor reports for a contour it cannot measure. The
    widget still has to ship a number to the browser, but the contour is
    genuinely unfilterable -- see the companion assertion that
    ``filtered_overlay`` drops it regardless of where the handles sit.
    """
    import numpy as np

    filters = [AreaFilter()]
    overlay, images = _overlay(), _source()
    table = _properties(overlay, images, filters)
    table.loc["mid", "area"] = np.nan

    fe = FilterExplorer(overlay, images, filters, table)
    values = [rec["values"][0] for rec in fe.contours]
    assert values[1] == 0.0  # not NaN
    assert all(np.isfinite(v) for v in values)

    # and it never survives filtering, however the handles are set
    assert "mid" not in {c.id for c in fe.filtered_overlay().contours}


# --- calibration guard -------------------------------------------------------


def test_requires_calibrated_source():
    with pytest.raises(ValueError, match="pixel_size"):
        FilterExplorer(_overlay(), None, [AreaFilter()], None)
    with pytest.raises(ValueError, match="pixel_size"):
        FilterExplorer(_overlay(), _source(pixel_size=None), [AreaFilter()], None)


# --- empty overlay -----------------------------------------------------------


def test_empty_overlay_builds_controls_with_fallback_range():
    filters = [AreaFilter(), LengthFilter()]
    empty, images = Overlay([]), _source()
    fe = FilterExplorer(empty, images, filters, _properties(empty, images, filters))
    assert len(fe.filter_specs) == 2
    assert fe.contours == []
    for spec in fe.filter_specs:
        assert spec["lo"] == pytest.approx(0.0)
        assert spec["hi"] == pytest.approx(1.0)


# --- params / configured_filters / filtered_overlay --------------------------


def _set_handle(fe, index, *, vmin=None, vmax=None):
    sel = [dict(s) for s in fe.selection]
    spec = fe.filter_specs[index]
    sel[index] = {
        "vmin": spec["lo"] if vmin is None else vmin,
        "vmax": spec["hi"] if vmax is None else vmax,
    }
    fe.selection = sel


def test_params_open_bound_semantics():
    fe = _explorer()
    # narrow only the area upper bound to 20 µm²; everything else stays open
    _set_handle(fe, 0, vmax=20.0)
    params = {p["name"]: p for p in fe.params}
    assert params["area"]["vmin"] is None  # handle at lo -> open
    assert params["area"]["vmax"] == Q_(20.0, "micrometer ** 2")
    assert params["length"]["vmin"] is None and params["length"]["vmax"] is None


def test_configured_filters_and_filtered_overlay_match_apply():
    fe = _explorer()
    _set_handle(fe, 0, vmax=20.0)  # keep area <= 20 µm² -> small + mid
    configured = fe.configured_filters()
    # bounds were written back onto the SAME filter instances
    assert configured[0].vmin is None
    assert configured[0].vmax == Q_(20.0, "micrometer ** 2")

    got = fe.filtered_overlay()
    assert {c.id for c in got.contours} == {"small", "mid"}

    # equals a direct apply_cell_filters with the same configured filters
    overlay, images = _overlay(), _source()
    expected = apply_cell_filters(
        overlay, configured, properties=_properties(overlay, images, configured)
    )
    assert {c.id for c in got.contours} == {c.id for c in expected.contours}


# --- save --------------------------------------------------------------------


def test_save_writes_filter_params_json(tmp_path):
    fe = _explorer()
    _set_handle(fe, 0, vmax=20.0)
    out = tmp_path / "filter_params.json"
    returned = fe.save(out)
    on_disk = json.loads(out.read_text())["filters"]
    assert on_disk == returned
    area = next(f for f in on_disk if f["name"] == "area")
    assert area["unit"] == "micrometer ** 2"
    assert area["vmin"] is None  # open
    assert area["vmax"] == pytest.approx(20.0)


# --- properties-backed construction ------------------------------------------


def _properties(overlay, images, filters):
    """Extractor table covering every filter in ``filters``."""
    from acia.analysis import (
        AreaEx,
        BoundaryClosenessEx,
        CircularityEx,
        ExtractorExecutor,
        LengthEx,
        PerimeterEx,
        WidthEx,
    )

    by_name = {
        "area": AreaEx,
        "perimeter": PerimeterEx,
        "length": LengthEx,
        "width": WidthEx,
        "circularity": CircularityEx,
        "boundary_closeness": BoundaryClosenessEx,
    }
    # circularity is derived from the area/perimeter columns, so those two must
    # run before it regardless of the order the filters were given in
    needed = {f.name for f in filters}
    if "circularity" in needed:
        needed |= {"area", "perimeter"}
    ordered = [by_name[n]() for n in by_name if n in needed]
    return ExtractorExecutor().execute(overlay, images, ordered)


def test_specs_match_the_per_contour_measurement():
    """Slider seeding from the table must match measuring each contour.

    ``CellFilter.value()`` remains the single-contour reference; the widget now
    reads columns instead, and the two must agree or the handles would land in
    different places than the thresholds they represent.
    """
    filters = [AreaFilter(), LengthFilter(), CircularityFilter()]
    overlay, images = _overlay(), _source()
    fe = FilterExplorer(overlay, images, filters, _properties(overlay, images, filters))

    for spec, f in zip(fe.filter_specs, filters, strict=True):
        measured = [float(f.value(c, images=images).magnitude) for c in overlay]
        assert spec["lo"] == pytest.approx(min(measured))
        if max(measured) > min(measured):
            assert spec["hi"] == pytest.approx(max(measured))
        else:
            # `_axis` gives a single-valued track a non-zero width
            assert spec["hi"] == pytest.approx(min(measured) + 1.0)
        assert spec["unit"] == f"{f.value(next(iter(overlay)), images=images).units}"


def test_properties_backed_filtered_overlay_matches():
    filters = [AreaFilter(Q_(2, "um**2"), Q_(20, "um**2"))]
    overlay, images = _overlay(), _source()
    table = _properties(overlay, images, filters)

    fe = FilterExplorer(overlay, images, filters, table)
    got = fe.filtered_overlay()

    expected = apply_cell_filters(overlay, fe.configured_filters(), properties=table)
    assert {c.id for c in got.contours} == {c.id for c in expected.contours}
    assert {c.id for c in got.contours} == {"small", "mid"}
