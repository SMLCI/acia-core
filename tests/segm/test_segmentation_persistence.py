"""Round-trip tests for :func:`acia.segm.formats.save_segmentation` / ``load_segmentation``.

These artifacts are the handoff between an analysis step that segments and a
later step that only analyses -- so what must survive is exactly: detection
ids/labels/geometry (including sub-pixel coordinates), the *full* frame extent
of the movie, and the time calibration (restored from the source, since the file
deliberately carries no time model).

The archive is a compressed binary polygon format; the loader additionally
accepts plain and gzipped simple-segmentation JSON, the interchange format other
tools read.
"""

import gzip
import unittest
import warnings
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from acia.base import Contour, Instance, Overlay
from acia.segm.formats import (
    gen_simple_segmentation,
    load_segmentation,
    save_segmentation,
)
from acia.segm.local import THWCSequenceSource


def _square(cx, cy, r=1.0):
    return np.array(
        [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]],
        dtype=np.float32,
    )


def _source(num_frames=4, **calibration):
    return THWCSequenceSource(
        np.zeros((num_frames, 16, 16, 1), dtype=np.uint8), **calibration
    )


def _overlay(frames=(0, 1, 2, 3)):
    """One detection per requested frame, ids 100.. and labels 'cell'."""
    return Overlay(
        [
            Contour(_square(f, f), score=1.0, frame=f, id=100 + f, label="cell")
            for f in frames
        ]
    )


class TestSegmentationRoundTrip(unittest.TestCase):
    def test_preserves_ids_labels_and_geometry(self):
        overlay = _overlay()

        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "output" / "segmentation", overlay)
            loaded = load_segmentation(path)

        self.assertEqual(len(loaded), len(overlay))
        self.assertEqual([c.id for c in loaded], [c.id for c in overlay])
        self.assertEqual([c.label for c in loaded], [c.label for c in overlay])
        self.assertEqual([c.frame for c in loaded], [c.frame for c in overlay])
        for before, after in zip(overlay, loaded, strict=True):
            np.testing.assert_allclose(
                np.asarray(after.coordinates, dtype=float),
                np.asarray(before.coordinates, dtype=float),
            )

    def test_preserves_subpixel_coordinates(self):
        # segmenter output is not on the pixel grid; a mask round-trip would
        # quantize these away, the polygon archive must not
        poly = np.array(
            [[10.25, 4.75], [11.5, 4.125], [11.0, 6.875], [9.875, 6.5]],
            dtype=np.float32,
        )
        overlay = Overlay([Contour(poly, score=1.0, frame=0, id=7, label="cell")])

        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", overlay)
            loaded = load_segmentation(path)

        np.testing.assert_array_equal(
            np.asarray(next(iter(loaded)).coordinates, dtype=np.float32), poly
        )

    def test_normalizes_suffix_and_creates_parents(self):
        with TemporaryDirectory() as tmp:
            written = save_segmentation(
                Path(tmp) / "nested" / "segmentation", _overlay()
            )

            self.assertEqual(written.name, "segmentation.npz")
            self.assertTrue(written.is_file())
            # the file really is a numpy archive, not just named like one
            self.assertTrue(zipfile.is_zipfile(written))

    def test_load_accepts_the_unsuffixed_path(self):
        with TemporaryDirectory() as tmp:
            stem = Path(tmp) / "segmentation"
            save_segmentation(stem, _overlay())

            # what save() accepted, load() accepts -- callers keep one name
            self.assertEqual(len(load_segmentation(stem)), 4)

    def test_load_accepts_plain_uncompressed_json(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "segmentation.json"
            path.write_text(gen_simple_segmentation(_overlay()), encoding="utf-8")

            # the format is sniffed from the magic bytes, not the suffix
            self.assertEqual(len(load_segmentation(path)), 4)

    def test_load_accepts_gzipped_json(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "segmentation.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(gen_simple_segmentation(_overlay()))

            loaded = load_segmentation(path)

        self.assertEqual(len(loaded), 4)
        self.assertEqual([c.id for c in loaded], [100, 101, 102, 103])

    def test_mixed_id_types_raise(self):
        overlay = Overlay(
            [
                Contour(_square(0, 0), score=1.0, frame=0, id=1, label="cell"),
                Contour(_square(1, 1), score=1.0, frame=1, id="two", label="cell"),
            ]
        )

        with TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError) as ctx:
                save_segmentation(Path(tmp) / "segmentation", overlay)

            self.assertIn("mixed types", str(ctx.exception))

    def test_string_ids_and_labels_round_trip(self):
        overlay = Overlay(
            [Contour(_square(0, 0), score=1.0, frame=0, id="a1", label="rod")]
        )

        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", overlay)
            loaded = load_segmentation(path)

        cont = next(iter(loaded))
        self.assertEqual(cont.id, "a1")
        self.assertEqual(cont.label, "rod")

    def test_missing_artifact_raises(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_segmentation(Path(tmp) / "segmentation.json.gz")

            self.assertIn("save_segmentation", str(ctx.exception))

    def test_empty_overlay_round_trips(self):
        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", Overlay([]))
            loaded = load_segmentation(path, _source(num_frames=4))

        self.assertEqual(len(loaded), 0)
        self.assertEqual(loaded.numFrames(), 4)


class TestSegmentationFrameExtent(unittest.TestCase):
    """A movie's length is not recoverable from its detections alone."""

    def test_trailing_empty_frames_are_restored_from_source(self):
        # 4-frame movie, but nothing survived filtering after frame 1
        overlay = _overlay(frames=(0, 1))

        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", overlay)

            without_source = load_segmentation(path)
            with_source = load_segmentation(path, _source(num_frames=4))

        self.assertEqual(without_source.numFrames(), 2)  # last populated frame
        self.assertEqual(with_source.numFrames(), 4)  # the movie's actual length
        self.assertEqual(len(with_source), 2)  # no detections invented

    def test_stored_extent_is_used_when_no_source_is_given(self):
        # an overlay that knows it spans 6 frames keeps that without a source
        overlay = Overlay(list(_overlay(frames=(0, 1))), frames=list(range(6)))

        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", overlay)
            loaded = load_segmentation(path)

        self.assertEqual(loaded.numFrames(), 6)

    def test_source_overrides_the_stored_extent(self):
        overlay = Overlay(list(_overlay(frames=(0, 1))), frames=list(range(6)))

        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", overlay)
            loaded = load_segmentation(path, _source(num_frames=4))

        self.assertEqual(loaded.numFrames(), 4)

    def test_leading_empty_frames_keep_their_frame_index(self):
        overlay = _overlay(frames=(2, 3))

        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", overlay)
            loaded = load_segmentation(path, _source(num_frames=4))

        self.assertEqual([c.frame for c in loaded], [2, 3])
        self.assertEqual(loaded.numFrames(), 4)


class TestSegmentationCalibration(unittest.TestCase):
    def test_timepoints_are_attached_from_the_source(self):
        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", _overlay())
            loaded = load_segmentation(
                path, _source(num_frames=4, frame_interval="5 min")
            )

        np.testing.assert_allclose(loaded.timepoints.magnitude, [0, 5, 10, 15])
        self.assertEqual(f"{loaded.timepoints.units:~P}", "min")
        # every detection is stamped, which is what the tracking graph reads later
        np.testing.assert_allclose(
            [c.time.magnitude for c in loaded], [0.0, 5.0, 10.0, 15.0]
        )

    def test_uncalibrated_source_invents_no_time(self):
        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", _overlay())
            loaded = load_segmentation(path, _source(num_frames=4))

        self.assertIsNone(loaded.timepoints)

    def test_without_source_the_overlay_is_returned_as_stored(self):
        with TemporaryDirectory() as tmp:
            path = save_segmentation(Path(tmp) / "segmentation", _overlay())
            loaded = load_segmentation(path)

        self.assertIsNone(loaded.timepoints)
        self.assertEqual(len(loaded), 4)


class TestFragmentedDetections(unittest.TestCase):
    """Instances whose masks have disconnected components.

    Such a mask has no single outline, so only its largest part can be stored
    as a polygon. That used to raise `AttributeError: 'MultiPolygon' object has
    no attribute 'exterior'` partway through a save.
    """

    @staticmethod
    def _fragmented_overlay():
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[3:8, 3:8] = 1
        mask[12:15, 12:15] = 1  # detached speck sharing the label
        solid = np.zeros((20, 20), dtype=np.int32)
        solid[3:8, 3:8] = 1
        return Overlay(
            [
                Instance(mask, frame=0, label=1, id=1),
                Instance(solid, frame=0, label=1, id=2),
            ],
            frames=[0],
        )

    def test_saving_a_fragmented_detection_succeeds(self):
        with TemporaryDirectory() as d:
            with self.assertWarns(UserWarning) as ctx:
                path = save_segmentation(
                    Path(d) / "segmentation", self._fragmented_overlay()
                )
            self.assertIn("connected component", str(ctx.warning))
            self.assertEqual(len(load_segmentation(path)), 2)

    def test_no_warning_when_every_detection_is_one_piece(self):
        solid = np.zeros((20, 20), dtype=np.int32)
        solid[3:8, 3:8] = 1
        overlay = Overlay([Instance(solid, frame=0, label=1, id=1)], frames=[0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with TemporaryDirectory() as d:
                save_segmentation(Path(d) / "segmentation", overlay)

    def test_contours_never_trigger_the_warning(self):
        """A Contour is already a single outline -- it has no `polygon` to be
        multi-part, and the check must not assume otherwise."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with TemporaryDirectory() as d:
                save_segmentation(Path(d) / "segmentation", _overlay())


if __name__ == "__main__":
    unittest.main()
