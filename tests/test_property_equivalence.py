"""Equivalence gate for the property-extraction / cell-filter speed work.

Property extraction and filtering are being optimised in stages. Each stage must
be a **pure speed** change, so everything a user could observe is pinned here
against ``tests/equivalence/golden.npz``, a snapshot taken from the
pre-optimisation implementation:

* the property values, held to the tolerances in ``scenes.COLUMN_TOLERANCE``
  (exact for everything except ``circularity``, which is a division);
* the unit attached to each column;
* every contour's polygon;
* the set of cells each filter configuration keeps -- the scientific contract,
  since that is what decides which cells enter the downstream analysis.

The tolerances are deliberate and are **not** to be relaxed to make a failing
test pass: a stage that cannot meet them is a stage that gets redesigned or
dropped, which is the whole point of the gate.
"""

from __future__ import annotations

import json
import unittest
import warnings
from pathlib import Path

import numpy as np
import shapely

from acia import Q_
from acia.analysis import BoundaryClosenessEx, ExtractorExecutor
from acia.base import Instance, Overlay
from acia.segm.filter import BoundaryClosenessFilter, apply_cell_filters

from .equivalence.scenes import (
    COLUMN_TOLERANCE,
    FRAME,
    GOLDEN_SCENES,
    extractors,
    filter_battery,
    filter_extractors,
    scene_absent_label,
    scene_degenerate,
    scene_empty,
    source,
)

GOLDEN_PATH = Path(__file__).parent / "equivalence" / "golden.npz"


def _load_golden():
    data = np.load(GOLDEN_PATH, allow_pickle=True)
    meta = json.loads(str(data["__meta__"][0]))
    return data, meta


class TestGoldenEquivalence(unittest.TestCase):
    """Property values, units, polygons and kept-id sets match the snapshot."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.meta = _load_golden()

    def test_property_values_match_golden(self):
        for scene_name, build in GOLDEN_SCENES.items():
            with self.subTest(scene=scene_name):
                overlay, images = build()
                df = ExtractorExecutor().execute(overlay, images, extractors())

                expected_columns = self.meta[scene_name]["columns"]
                self.assertEqual(
                    list(df.columns),
                    expected_columns,
                    f"{scene_name}: column set/order changed",
                )
                np.testing.assert_array_equal(
                    np.array([str(i) for i in df.index], dtype=object),
                    self.data[f"{scene_name}/ids"],
                    err_msg=f"{scene_name}: row identity/order changed",
                )

                for column in expected_columns:
                    got = df[column].to_numpy(dtype=np.float64)
                    want = self.data[f"{scene_name}/col/{column}"]
                    tolerance = COLUMN_TOLERANCE.get(column)
                    if tolerance is None:
                        np.testing.assert_array_equal(
                            got,
                            want,
                            err_msg=(
                                f"{scene_name}.{column}: expected EXACT equality with "
                                "the pre-optimisation values"
                            ),
                        )
                    else:
                        np.testing.assert_allclose(
                            got,
                            want,
                            rtol=tolerance,
                            atol=0.0,
                            err_msg=f"{scene_name}.{column}: outside pinned tolerance",
                        )

    def test_units_match_golden(self):
        for scene_name, build in GOLDEN_SCENES.items():
            with self.subTest(scene=scene_name):
                overlay, images = build()
                executor = ExtractorExecutor()
                df = executor.execute(overlay, images, extractors())
                got = {k: str(v) for k, v in df.attrs.get("units", {}).items()}
                self.assertEqual(got, self.meta[scene_name]["units"])

    def test_polygons_match_golden(self):
        for scene_name, build in GOLDEN_SCENES.items():
            with self.subTest(scene=scene_name):
                overlay, _ = build()
                want_ids = self.data[f"{scene_name}/wkb_ids"]
                want_wkb = self.data[f"{scene_name}/wkb"]

                got_ids = np.array([str(c.id) for c in overlay], dtype=object)
                np.testing.assert_array_equal(got_ids, want_ids)

                for contour, blob in zip(overlay, want_wkb, strict=True):
                    expected = shapely.from_wkb(bytes(blob)) if bytes(blob) else None
                    actual = contour.polygon
                    if expected is None:
                        self.assertIsNone(actual, f"{scene_name}/{contour.id}")
                        continue
                    self.assertIsNotNone(actual, f"{scene_name}/{contour.id}")
                    self.assertTrue(
                        actual.equals(expected),
                        f"{scene_name}/{contour.id}: polygon changed",
                    )

    def test_kept_ids_match_golden(self):
        """The same cells survive when filters read the extracted table.

        The golden kept-id sets were recorded from the row-wise path, which
        measured each contour itself. Reading the values off the extractor
        table instead must select exactly the same cells -- that equivalence is
        the whole basis for dropping the second measurement pass.
        """
        for scene_name, build in GOLDEN_SCENES.items():
            overlay, images = build()
            table = ExtractorExecutor().execute(overlay, images, filter_extractors())
            for battery_name, filters in filter_battery().items():
                with self.subTest(scene=scene_name, filters=battery_name):
                    result = apply_cell_filters(overlay, filters, properties=table)
                    got = sorted(str(c.id) for c in result)
                    self.assertEqual(
                        got,
                        self.meta[scene_name]["kept"][battery_name],
                        f"{scene_name}/{battery_name}: different cells survive",
                    )

    def test_deprecated_images_path_still_matches_golden(self):
        """The row-wise ``images=`` path keeps working while it is deprecated."""
        for scene_name, build in GOLDEN_SCENES.items():
            overlay, images = build()
            for battery_name, filters in filter_battery().items():
                with self.subTest(scene=scene_name, filters=battery_name):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        result = apply_cell_filters(overlay, filters, images=images)
                    got = sorted(str(c.id) for c in result)
                    self.assertEqual(got, self.meta[scene_name]["kept"][battery_name])

    def test_boundary_closeness_column_matches_the_filter(self):
        """The new column reproduces ``BoundaryClosenessFilter.value()`` exactly.

        ``BoundaryClosenessFilter`` was the one filter without a backing
        extractor, so this column is new rather than pre-existing and cannot be
        checked against the golden. Pin it against the measurement it replaces.
        """
        for scene_name, build in GOLDEN_SCENES.items():
            with self.subTest(scene=scene_name):
                overlay, images = build()
                table = ExtractorExecutor().execute(
                    overlay, images, [BoundaryClosenessEx()]
                )
                cell_filter = BoundaryClosenessFilter(Q_(0.5, "um"))
                measured = np.array(
                    [
                        float(cell_filter.value(c, images=images).magnitude)
                        for c in overlay
                    ]
                )
                from_column = table.loc[
                    [c.id for c in overlay], "boundary_closeness"
                ].to_numpy(dtype=float)
                np.testing.assert_array_equal(from_column, measured)


class TestHarnessIsMeaningful(unittest.TestCase):
    """Guards against the gate silently becoming vacuous.

    A battery where every configuration keeps every cell would pass forever
    while testing nothing, so assert the snapshot actually discriminates.
    """

    @classmethod
    def setUpClass(cls):
        cls.data, cls.meta = _load_golden()

    def test_some_configs_keep_a_strict_subset(self):
        partitioning = 0
        for scene_meta in self.meta.values():
            total = scene_meta["n_contours"]
            for kept in scene_meta["kept"].values():
                if 0 < len(kept) < total:
                    partitioning += 1
        self.assertGreater(
            partitioning,
            3,
            "filter battery no longer discriminates -- the kept-id assertions "
            "would pass trivially",
        )

    def test_battery_covers_empty_and_full_results(self):
        any_empty = any(
            len(kept) == 0
            for scene in self.meta.values()
            for kept in scene["kept"].values()
        )
        any_full = any(
            len(kept) == scene["n_contours"]
            for scene in self.meta.values()
            for kept in scene["kept"].values()
        )
        self.assertTrue(any_empty, "no configuration rejects everything")
        self.assertTrue(any_full, "no configuration keeps everything")


class TestGeometryInvariants(unittest.TestCase):
    """Properties that must hold regardless of how the geometry is computed."""

    def test_instance_polygon_is_independent_of_frame_size(self):
        """The same cell in a bigger frame must give the same polygon.

        This is the current performance defect stated as correctness: today the
        polygon is traced by scanning the whole frame, so a change to bounding-box
        cropping must not move a single vertex.
        """
        polygons = []
        for size in (64, 256, 1024):
            mask = np.zeros((size, size), np.int32)
            mask[20:31, 25:33] = 7
            polygons.append(Instance(mask=mask, frame=0, label=7, id=7).polygon)

        for other in polygons[1:]:
            self.assertTrue(
                polygons[0].equals(other),
                "Instance.polygon depends on the frame it sits in",
            )

    def test_supplied_bbox_matches_the_derived_one(self):
        """A caller-supplied bounding box must give the same geometry as none.

        ``overlay_from_masks`` hands each instance the box ``find_objects``
        found, skipping the per-instance scan. An off-by-one there would shift
        or clip an outline, so assert the two routes agree exactly.
        """
        mask = np.zeros((96, 96), np.int32)
        mask[13:27, 41:52] = 4
        mask[60:64, 5:9] = 4  # a second, disconnected part

        derived = Instance(mask=mask, frame=0, label=4, id=4)
        supplied = Instance(
            mask=mask,
            frame=0,
            label=4,
            id=4,
            bbox=(slice(13, 64), slice(5, 52)),
        )

        self.assertTrue(derived.polygon.equals(supplied.polygon))
        self.assertEqual(derived.area, supplied.area)
        self.assertEqual(derived.center, supplied.center)

    def test_overlay_from_masks_matches_per_instance_derivation(self):
        """The bulk path and the lazy path agree, cell for cell."""
        from acia.segm.formats import overlay_from_masks

        rng = np.random.default_rng(7)
        stack = np.zeros((2, 80, 80), np.int32)
        for frame in range(2):
            for label in range(1, 9):
                y = int(rng.integers(0, 70))
                x = int(rng.integers(0, 70))
                stack[frame, y : y + 6, x : x + 9] = label

        from_bulk = overlay_from_masks(stack)
        for contour in from_bulk:
            lazy = Instance(
                mask=stack[contour.frame], frame=contour.frame, label=contour.label
            )
            self.assertTrue(contour.polygon.equals(lazy.polygon))
            self.assertEqual(contour.area, lazy.area)

    def test_geometry_survives_mask_reassignment(self):
        """Changing mask/label must drop the cached box, not reuse a stale one."""
        first = np.zeros((64, 64), np.int32)
        first[10:14, 10:14] = 1
        second = np.zeros((64, 64), np.int32)
        second[40:50, 30:44] = 1

        instance = Instance(mask=first, frame=0, label=1, id=1)
        self.assertEqual(instance.area, 16.0)

        instance.mask = second
        self.assertEqual(instance.area, 140.0)
        self.assertTrue(
            instance.polygon.equals(Instance(mask=second, frame=0, label=1).polygon)
        )

    def test_supplied_bbox_is_dropped_when_mask_changes(self):
        """A passed-in box describes one (mask, label) pair only."""
        first = np.zeros((64, 64), np.int32)
        first[10:14, 10:14] = 1
        second = np.zeros((64, 64), np.int32)
        second[40:50, 30:44] = 1

        instance = Instance(
            mask=first, frame=0, label=1, id=1, bbox=(slice(10, 14), slice(10, 14))
        )
        self.assertEqual(instance.area, 16.0)

        instance.mask = second  # the old box now points at empty pixels
        self.assertEqual(instance.area, 140.0)

    def test_contour_polygon_cache_follows_coordinates(self):
        """The cached ``Contour`` polygon must track a coordinate change."""
        from acia.base import Contour

        contour = Contour(
            np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]),
            score=-1,
            frame=0,
            id=1,
        )
        self.assertEqual(contour.area, 16.0)

        contour.scale(2.0)
        self.assertEqual(contour.area, 64.0)

        contour.coordinates = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        self.assertEqual(contour.area, 4.0)

    def test_raster_polygon_area_equals_pixel_count(self):
        """``polygon.area`` equals the mask pixel count, exactly.

        Filtering reads properties off the extracted table rather than
        re-measuring each contour, which is only equivalent because rasterio
        polygonises along pixel edges. Pinned here because the design leans on
        it: were it merely approximate, the area-derived filters would start
        selecting different cells.
        """
        mask = np.zeros((64, 64), np.int32)
        mask[10:21, 12:19] = 3  # 11 x 7 = 77 px
        mask[40:44, 40:44] = 3  # + 16 px, disconnected
        instance = Instance(mask=mask, frame=0, label=3, id=3)

        self.assertEqual(instance.area, 93.0)
        self.assertEqual(instance.polygon.area, 93.0)

    def test_fragmented_area_counts_all_parts(self):
        """A split mask keeps every pixel in ``area`` though its outline is one part."""
        mask = np.zeros((64, 64), np.int32)
        mask[10:14, 10:14] = 1  # 16 px
        mask[30:32, 30:32] = 1  # 4 px
        instance = Instance(mask=mask, frame=0, label=1, id=1)

        self.assertTrue(instance.is_fragmented)
        self.assertEqual(instance.area, 20.0)
        # coordinates follow the largest part only
        self.assertEqual(len(instance.coordinates), 4)


class TestEdgeSceneBehaviour(unittest.TestCase):
    """Pins how degenerate inputs behave, so a change to it is deliberate.

    The row-wise filter path is degenerate-safe (``_rotated_rect_coords`` maps a
    collapsed minimum rotated rectangle to a 0 measurement) but the extractors
    are **not** -- they raise ``AttributeError``. Filtering off the extracted
    table inherits whatever extraction can do, so a contour that extraction
    cannot measure can no longer be filtered out either: it now fails the run
    instead of being silently dropped.

    Pinned rather than fixed here, because making the extractors degenerate-safe
    changes values (from "crash" to 0) and belongs in its own change.
    """

    def _filter_via_table(self, overlay, images, battery_name):
        table = ExtractorExecutor().execute(overlay, images, filter_extractors())
        return apply_cell_filters(
            overlay, list(filter_battery()[battery_name]), properties=table
        )

    def test_row_wise_path_tolerates_degenerate_contours(self):
        overlay, images = scene_degenerate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = apply_cell_filters(
                overlay, list(filter_battery()["length_two_sided"]), images=images
            )
        self.assertEqual(sorted(str(c.id) for c in result), ["triangle"])

    def test_row_wise_path_tolerates_absent_label(self):
        overlay, images = scene_absent_label()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = apply_cell_filters(
                overlay, list(filter_battery()["boundary_only"]), images=images
            )
        self.assertEqual(list(result), [])

    def test_table_path_cannot_filter_what_extraction_cannot_measure(self):
        """The inherited limitation, stated explicitly rather than discovered."""
        for scene in (scene_degenerate, scene_absent_label):
            with self.subTest(scene=scene.__name__):
                overlay, images = scene()
                with self.assertRaises(AttributeError):
                    self._filter_via_table(overlay, images, "length_two_sided")

    def test_extractors_fail_on_degenerate_contours(self):
        overlay, images = scene_degenerate()
        with self.assertRaises(AttributeError):
            ExtractorExecutor().execute(overlay, images, extractors())

    def test_extractors_fail_on_absent_label(self):
        overlay, images = scene_absent_label()
        with self.assertRaises(AttributeError):
            ExtractorExecutor().execute(overlay, images, extractors())

    def test_empty_overlay_extracts_and_filters(self):
        overlay, images = scene_empty()
        df = ExtractorExecutor().execute(overlay, images, extractors())
        self.assertEqual(len(df), 0)
        result = self._filter_via_table(overlay, images, "notebook_combo")
        self.assertEqual(list(result), [])

    def test_missing_column_names_the_extractor_to_add(self):
        """A filter without its column fails loudly, not by silently re-measuring."""
        overlay, images = GOLDEN_SCENES["instance_basic"]()
        # table WITHOUT boundary_closeness
        table = ExtractorExecutor().execute(overlay, images, extractors())
        with self.assertRaises(KeyError) as ctx:
            apply_cell_filters(
                overlay, list(filter_battery()["boundary_only"]), properties=table
            )
        self.assertIn("boundary_closeness", str(ctx.exception))

    def test_table_must_describe_every_contour(self):
        overlay, images = GOLDEN_SCENES["instance_basic"]()
        table = ExtractorExecutor().execute(overlay, images, filter_extractors())
        with self.assertRaises(ValueError) as ctx:
            apply_cell_filters(
                overlay,
                list(filter_battery()["area_two_sided"]),
                properties=table.iloc[:-2],
            )
        self.assertIn("missing from the properties", str(ctx.exception))

    def test_bound_dimension_mismatch_still_raises(self):
        """A µm bound against a µm² column must fail, as it did per contour."""
        import pint

        from acia.segm.filter import AreaFilter

        overlay, images = GOLDEN_SCENES["instance_basic"]()
        table = ExtractorExecutor().execute(overlay, images, filter_extractors())
        with self.assertRaises(pint.DimensionalityError):
            apply_cell_filters(
                overlay, [AreaFilter(Q_(1, "um"), None)], properties=table
            )

    def test_scene_helpers_are_deterministic(self):
        """Two builds of a scene must agree, or the golden is meaningless."""
        first, _ = GOLDEN_SCENES["instance_basic"]()
        second, _ = GOLDEN_SCENES["instance_basic"]()
        self.assertEqual(
            [c.id for c in first],
            [c.id for c in second],
        )
        for a, b in zip(first, second, strict=True):
            self.assertTrue(a.polygon.equals(b.polygon))

    def test_source_shape_matches_scene_frame(self):
        images = source()
        self.assertEqual((images.size_h, images.size_w), (FRAME, FRAME))
        self.assertIsInstance(Overlay([]), Overlay)


if __name__ == "__main__":
    unittest.main()
