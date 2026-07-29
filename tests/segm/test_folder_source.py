"""Unit tests for :mod:`acia.segm.folder_source` (a folder of per-timepoint TIFFs).

Everything runs against synthetic folders written into a temp dir -- no fixture
files. Laziness is asserted by swapping the module's ``tifffile`` for a decode-
counting stand-in: listing/`size_t`/positions must decode nothing, and reading a
frame must decode exactly the one file that backs it.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

from acia import ureg
from acia.segm import folder_source
from acia.segm.folder_source import (
    FolderSequenceSource,
    natural_key,
    resolve_layout,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _plane(h=6, w=8, seed=0, dtype=np.uint16):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 1000).astype(dtype)


def _write_folder(folder, names, *, h=6, w=8, dtype=np.uint16, **imwrite_kwargs):
    """Write one single-plane TIFF per name; returns the planes in listed order."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    planes = []
    for i, name in enumerate(names):
        plane = _plane(h=h, w=w, seed=i, dtype=dtype)
        tifffile.imwrite(folder / name, plane, **imwrite_kwargs)
        planes.append(plane)
    return planes


class _CountingTiff:
    """Stand-in for the ``tifffile`` module that records every decode."""

    def __init__(self):
        self.reads: list[str] = []

    def imread(self, handle):
        path = getattr(handle, "path", None) or getattr(handle, "name", repr(handle))
        self.reads.append(Path(str(path)).name)
        return tifffile.imread(handle)


@contextlib.contextmanager
def _count_reads():
    counter = _CountingTiff()
    with mock.patch.object(folder_source, "tifffile", counter):
        yield counter


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #
class TestNaturalOrdering(unittest.TestCase):
    def test_numeric_runs_sort_numerically(self):
        names = ["img_10.tif", "img_2.tif", "img_1.tif"]
        self.assertEqual(
            sorted(names, key=natural_key), ["img_1.tif", "img_2.tif", "img_10.tif"]
        )

    def test_key_uses_basename_only(self):
        self.assertEqual(natural_key("/a/b/t_3.tif"), natural_key("t_3.tif"))

    def test_frames_follow_numeric_order(self):
        with tempfile.TemporaryDirectory() as d:
            names = [f"img_{i}.tif" for i in range(1, 11)]  # img_1 .. img_10
            planes = _write_folder(d, names)
            src = FolderSequenceSource(d)

            self.assertEqual(src.size_t, 10)
            self.assertEqual(Path(src.files[9]).name, "img_10.tif")
            # frame 9 is img_10, NOT img_2 (the lexicographic answer)
            np.testing.assert_array_equal(src.get_frame(9).raw[..., 0], planes[9])
            np.testing.assert_array_equal(src.get_frame(1).raw[..., 0], planes[1])


# --------------------------------------------------------------------------- #
# frames, shape and laziness
# --------------------------------------------------------------------------- #
class TestFolderSequenceSource(unittest.TestCase):
    def test_size_t_decodes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, [f"t{i:03d}.tif" for i in range(12)])
            with _count_reads() as counter:
                src = FolderSequenceSource(d)
                self.assertEqual(src.size_t, 12)
                self.assertEqual(len(src.files), 12)
                self.assertEqual(counter.reads, [])

    def test_get_frame_decodes_exactly_one_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, [f"t{i:03d}.tif" for i in range(12)])
            with _count_reads() as counter:
                frame = FolderSequenceSource(d).get_frame(7)
                self.assertEqual(counter.reads, ["t007.tif"])
            self.assertEqual(frame.raw.shape, (6, 8, 1))

    def test_repeated_access_reuses_cached_frame(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.tif"])
            src = FolderSequenceSource(d)
            with _count_reads() as counter:
                src.get_frame(1)
                src.get_frame(1)
                self.assertEqual(counter.reads, ["b.tif"])
                src.close()
                src.get_frame(1)
                self.assertEqual(counter.reads, ["b.tif", "b.tif"])

    def test_grayscale_gains_channel_axis(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.tif"], h=12, w=9)
            src = FolderSequenceSource(d)
            self.assertEqual((src.size_h, src.size_w, src.size_c), (12, 9, 1))
            self.assertEqual(src.num_channels, 1)
            self.assertEqual(src.get_frame(0).raw.shape, (12, 9, 1))
            self.assertEqual(str(src.dtype), "uint16")

    def test_channel_axis_zero_moves_channels_last(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                arr = np.stack([_plane(12, 9, seed=i), _plane(12, 9, seed=i + 10)])
                tifffile.imwrite(Path(d) / f"t{i}.tif", arr)  # (C, H, W)
            src = FolderSequenceSource(d, channel_axis=0)
            self.assertEqual((src.size_h, src.size_w, src.size_c), (12, 9, 2))
            self.assertEqual(src.get_frame(2).raw.shape, (12, 9, 2))

    def test_ambiguous_leading_axis_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            arr = np.stack([_plane(12, 9), _plane(12, 9, seed=1)])  # (2, 12, 9)
            tifffile.imwrite(Path(d) / "t0.tif", arr)
            with self.assertRaises(ValueError) as ctx:
                FolderSequenceSource(d).get_frame(0)  # default channel_axis=-1
            self.assertIn("channel_axis=0", str(ctx.exception))

    def test_multipage_stack_per_file_is_rejected_at_any_page_count(self):
        # a z-stack/multi-page file read as (H, W, C) would give an image of
        # `pages` rows with `width` channels -- garbage that segments happily
        for pages in (2, 5, 10):
            with tempfile.TemporaryDirectory() as d:
                tifffile.imwrite(
                    Path(d) / "t0.tif", np.zeros((pages, 64, 64), np.uint16)
                )
                with self.assertRaises(ValueError, msg=f"{pages} pages"):
                    FolderSequenceSource(d).get_frame(0)

    def test_explicit_channel_axis_two_asserts_hwc(self):
        with tempfile.TemporaryDirectory() as d:
            tifffile.imwrite(Path(d) / "t0.tif", np.zeros((4, 8, 64), np.uint16))
            src = FolderSequenceSource(d, channel_axis=2)  # "yes, really (H, W, C)"
            self.assertEqual((src.size_h, src.size_w, src.size_c), (4, 8, 64))

    def test_one_dimensional_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tifffile.imwrite(Path(d) / "t0.tif", np.zeros((16,), np.uint16))
            with self.assertRaises(ValueError) as ctx:
                FolderSequenceSource(d).get_frame(0)
            self.assertIn("one plane per file", str(ctx.exception))

    def test_hwc_file_is_kept_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                tifffile.imwrite(
                    Path(d) / f"t{i}.tif", np.zeros((12, 9, 3), dtype=np.uint8)
                )
            src = FolderSequenceSource(d)
            self.assertEqual((src.size_h, src.size_w, src.size_c), (12, 9, 3))

    def test_iteration_and_materialize(self):
        with tempfile.TemporaryDirectory() as d:
            planes = _write_folder(d, [f"t{i}.tif" for i in range(4)])
            src = FolderSequenceSource(d)
            self.assertEqual(len(src), 4)
            self.assertEqual([f.raw.shape for f in src], [(6, 8, 1)] * 4)
            frozen = src.materialize()
            np.testing.assert_array_equal(frozen.get_frame(3).raw[..., 0], planes[3])

    def test_invalid_channel_axis_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            FolderSequenceSource("/tmp", channel_axis=1)

    # --- error rows --------------------------------------------------------- #

    def test_ragged_folder_names_the_offending_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, [f"t{i}.tif" for i in range(5)], h=12, w=9)
            tifffile.imwrite(Path(d) / "t5.tif", _plane(6, 6))  # different geometry
            src = FolderSequenceSource(d)
            src.get_frame(0)
            with self.assertRaises(ValueError) as ctx:
                src.get_frame(5)
            message = str(ctx.exception)
            # both files are named: whichever was decoded first defines "correct",
            # so the mismatch alone does not say which of the two is the odd one
            self.assertIn("t5.tif", message)
            self.assertIn("t0.tif", message)

    def test_dtype_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["t0.tif"], dtype=np.uint16)
            tifffile.imwrite(Path(d) / "t1.tif", _plane(dtype=np.uint8))
            src = FolderSequenceSource(d)
            src.get_frame(0)
            with self.assertRaises(ValueError):
                src.get_frame(1)

    def test_stack_inside_a_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tifffile.imwrite(Path(d) / "t0.tif", np.zeros((2, 3, 8, 10), np.uint16))
            with self.assertRaises(ValueError) as ctx:
                FolderSequenceSource(d).get_frame(0)
            self.assertIn("one plane per file", str(ctx.exception))

    def test_empty_folder_names_folder_and_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "notes.txt").write_text("nothing to see")
            with self.assertRaises(ValueError) as ctx:
                _ = FolderSequenceSource(d).size_t
            message = str(ctx.exception)
            self.assertIn(d, message)
            self.assertIn(".tif", message)

    def test_frame_out_of_range(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.tif"])
            with self.assertRaises(IndexError):
                FolderSequenceSource(d).get_frame(5)

    # --- file selection ----------------------------------------------------- #

    def test_mixed_suffix_and_case_with_junk_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.TIFF"])
            (Path(d) / "notes.txt").write_text("x")
            (Path(d) / ".DS_Store").write_text("x")
            tifffile.imwrite(Path(d) / "._a.tif", _plane())  # AppleDouble sibling
            self.assertEqual(FolderSequenceSource(d).size_t, 2)

    def test_broad_pattern_still_requires_a_tiff_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["img_a.tif", "img_b.tif"])
            (Path(d) / "img_metadata.txt").write_text("x")
            (Path(d) / "DisplaySettings.json").write_text("{}")
            # '*' must not pull non-TIFFs in and fail deep inside a decode later
            self.assertEqual(FolderSequenceSource(d, pattern="*").size_t, 2)
            self.assertEqual(FolderSequenceSource(d, pattern="img_*").size_t, 2)

    def test_leading_zero_ties_are_broken_deterministically(self):
        with tempfile.TemporaryDirectory() as d:
            # img_1 and img_01 share a numeric key; order must not depend on
            # whatever sequence the filesystem happened to list them in
            _write_folder(d, ["img_01.tif", "img_1.tif", "img_2.tif"])
            names = [Path(f).name for f in FolderSequenceSource(d).files]
            self.assertEqual(names, ["img_01.tif", "img_1.tif", "img_2.tif"])

    def test_custom_pattern_selects_a_subset(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(
                d,
                [
                    "img_t001_c000.tif",
                    "img_t001_c001.tif",
                    "img_t002_c000.tif",
                    "img_t002_c001.tif",
                ],
            )
            # channel-split folders are out of scope: the default pattern sees all
            # four files as four frames, which is wrong but visible (2x the frames)
            self.assertEqual(FolderSequenceSource(d).size_t, 4)
            # the documented workaround: one channel per source
            src = FolderSequenceSource(d, pattern="*_c000.tif")
            self.assertEqual(src.size_t, 2)
            self.assertEqual(
                [Path(f).name for f in src.files],
                ["img_t001_c000.tif", "img_t002_c000.tif"],
            )

    def test_subfolders_are_not_flattened_into_frames(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.tif"])
            _write_folder(Path(d) / "deeper", ["c.tif", "d.tif"])
            self.assertEqual(FolderSequenceSource(d).size_t, 2)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
class TestCalibration(unittest.TestCase):
    def _imagej_folder(self, d, pixel_size_um=0.0733):
        _write_folder(
            d,
            [f"t{i}.tif" for i in range(3)],
            imagej=True,
            resolution=(1 / pixel_size_um, 1 / pixel_size_um),
            metadata={"unit": "um"},
        )

    def test_pixel_size_read_from_first_file(self):
        with tempfile.TemporaryDirectory() as d:
            self._imagej_folder(d)
            src = FolderSequenceSource(d)
            self.assertAlmostEqual(
                src.pixel_size.to("micrometer").magnitude, 0.0733, places=4
            )
            self.assertEqual(src.calibration_source, "imagej")

    def test_user_override_wins(self):
        with tempfile.TemporaryDirectory() as d:
            self._imagej_folder(d)
            src = FolderSequenceSource(d, pixel_size="0.5 um", frame_interval="5 min")
            self.assertAlmostEqual(
                src.pixel_size.to("micrometer").magnitude, 0.5, places=6
            )
            gap = src.timepoints[1] - src.timepoints[0]
            self.assertAlmostEqual(gap.to("minute").magnitude, 5.0, places=6)

    def test_per_file_plane_timepoints_are_not_adopted(self):
        # A per-timepoint OME-TIFF carries its OWN <Plane DeltaT>, describing that
        # one file. Adopting it would give a length-1 timepoints array for an
        # N-frame folder (and silently outrank the caller's frame_interval).
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                tifffile.imwrite(
                    Path(d) / f"t{i}.tif",
                    _plane(seed=i),
                    ome=True,
                    metadata={
                        "PhysicalSizeX": 0.5,
                        "PhysicalSizeXUnit": "µm",
                        "Plane": {"DeltaT": [float(i * 300)], "DeltaTUnit": ["s"]},
                    },
                )
            src = FolderSequenceSource(d, frame_interval="5 min")
            self.assertEqual(len(src.timepoints), 5)  # not the file's length-1 array
            gap = src.timepoints[1] - src.timepoints[0]
            self.assertAlmostEqual(gap.to("minute").magnitude, 5.0, places=6)
            # pixel size still comes from the same file's metadata
            self.assertAlmostEqual(
                src.pixel_size.to("micrometer").magnitude, 0.5, places=6
            )
            # ... and the whole SequenceFile surface stays usable
            from acia.segm.open import open_sequence

            self.assertEqual(
                open_sequence(d, frame_interval="5 min").metadata.num_timepoints, 5
            )

    def test_with_calibration_setters_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            self._imagej_folder(d)  # file says 0.0733 um
            src = FolderSequenceSource(d).with_pixel_size("0.25 um")
            self.assertAlmostEqual(
                src.pixel_size.to("micrometer").magnitude, 0.25, places=6
            )
            src = FolderSequenceSource(d).with_frame_interval("2 min")
            gap = src.timepoints[1] - src.timepoints[0]
            self.assertAlmostEqual(gap.to("minute").magnitude, 2.0, places=6)

    def test_uncalibrated_folder_reports_none(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.tif"])
            src = FolderSequenceSource(d)
            self.assertIsNone(src.pixel_size)
            self.assertIsNone(src.timepoints)

    def test_full_user_calibration_reads_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.tif"])
            src = FolderSequenceSource(
                d,
                pixel_size="0.5 um",
                frame_interval="5 min",
                timepoints=ureg.Quantity(np.array([0.0, 300.0]), "second"),
            )
            with mock.patch.object(
                folder_source, "read_tiff_calibration"
            ) as read_calibration:
                self.assertIsNotNone(src.pixel_size)
                read_calibration.assert_not_called()


# --------------------------------------------------------------------------- #
# layout resolution (flat vs. nested)
# --------------------------------------------------------------------------- #
class TestResolveLayout(unittest.TestCase):
    def test_flat_folder_is_one_position(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["a.tif", "b.tif"])
            folders, nested = resolve_layout(d)
            self.assertEqual(folders, [d])
            self.assertFalse(nested)

    def test_subfolders_become_positions_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as d:
            for i in [1, 2, 10]:  # pos10 must sort after pos2
                _write_folder(Path(d) / f"pos{i}", ["a.tif", "b.tif"])
            folders, nested = resolve_layout(d)
            self.assertTrue(nested)
            self.assertEqual([Path(f).name for f in folders], ["pos1", "pos2", "pos10"])

    def test_flat_wins_when_both_are_present_but_warns(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(d, ["overview.tif"])  # one stray file next to positions
            for i in range(1, 4):
                _write_folder(Path(d) / f"pos{i}", ["a.tif", "b.tif"])
            with self.assertWarns(UserWarning) as ctx:
                folders, nested = resolve_layout(d)
            self.assertEqual((folders, nested), ([d], False))
            # silently reading a 1-frame movie instead of 3 positions must not pass
            # unremarked -- the warning names both counts
            self.assertIn("3 subfolder", str(ctx.warning))

    def test_symlinked_position_folder_is_found(self):
        import os

        with tempfile.TemporaryDirectory() as d:
            real = Path(d) / "real"
            _write_folder(real, ["a.tif", "b.tif"])
            parent = Path(d) / "parent"
            parent.mkdir()
            os.symlink(real, parent / "pos1")  # fsspec lists this as type "other"
            folders, nested = resolve_layout(parent)
            self.assertTrue(nested)
            self.assertEqual([Path(f).name for f in folders], ["pos1"])
            self.assertEqual(FolderSequenceSource(folders[0]).size_t, 2)

    def test_subfolder_without_tiffs_is_not_a_position(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(Path(d) / "pos1", ["a.tif"])
            (Path(d) / "notes").mkdir()
            (Path(d) / "notes" / "readme.txt").write_text("x")
            folders, _ = resolve_layout(d)
            self.assertEqual([Path(f).name for f in folders], ["pos1"])

    def test_nothing_anywhere_raises(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "empty").mkdir()
            with self.assertRaises(ValueError) as ctx:
                resolve_layout(d)
            self.assertIn(d, str(ctx.exception))

    def test_two_levels_deep_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _write_folder(Path(d) / "a" / "b", ["t0.tif"])
            with self.assertRaises(ValueError):
                resolve_layout(d)

    def test_layout_decodes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                _write_folder(Path(d) / f"pos{i}", ["a.tif", "b.tif"])
            with _count_reads() as counter:
                resolve_layout(d)
                self.assertEqual(counter.reads, [])


if __name__ == "__main__":
    unittest.main()
