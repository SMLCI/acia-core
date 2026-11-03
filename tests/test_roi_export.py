"""Unit tests for :func:`acia.roi_export.export_rois`.

The source is a real multi-position handle (a folder of per-timepoint TIFFs, one
subfolder per position), so ``position()``, ``metadata`` and the lazy slicing all
behave as they do on an ND2/CZI. Pixels encode their ``(position, frame)`` so a
crop can be traced back to the frame it came from.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import tifffile

from acia.base import RotatedCropSpec
from acia.roi_export import (
    _register_position,
    _roi_bytes,
    _roi_groups,
    export_rois,
)
from acia.segm.open import open_sequence
from acia.segm.tiff_export import save_tiff_stack
from acia.selection import RoiSelection, SelectionManifest, make_source_block

SIZE = 64


def _textured(seed: int = 0, size: int = SIZE) -> np.ndarray:
    """Structure a registration method can actually lock onto."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (size // 4, size // 4))
    frame = np.kron(base, np.ones((4, 4)))
    frame = frame + rng.normal(0, 0.05, frame.shape)
    frame -= frame.min()
    return (frame / frame.max() * 60000).astype(np.uint16)


def _write_positions(root: Path, num_positions: int, num_t: int, *, shift=0.0):
    """One subfolder per position, one TIFF per timepoint."""
    import cv2

    ref = _textured(0)
    for p in range(num_positions):
        folder = root / f"pos{p:03d}"
        folder.mkdir(parents=True)
        for t in range(num_t):
            if shift:
                matrix = np.float32([[1, 0, shift * t], [0, 1, -shift * t]])
                frame = cv2.warpAffine(ref, matrix, (SIZE, SIZE))
            else:
                frame = np.full((SIZE, SIZE), _encode(p, t), np.uint16)
            tifffile.imwrite(folder / f"t{t:03d}.tif", frame)
    return root


def _encode(p: int, t: int) -> int:
    return 1000 * (p + 1) + t


def _manifest(seqfile, specs):
    """specs: (position, center, size, label, anchor_frame)."""
    return SelectionManifest(
        source=make_source_block(seqfile),
        roi_mode="multi",
        selections=[
            RoiSelection(
                position=p,
                roi=RotatedCropSpec(center=c, size=s, angle=0.0),
                label=label,
                id=f"sel{i}",
                anchor_frame=anchor,
            )
            for i, (p, c, s, label, anchor) in enumerate(specs)
        ],
    )


class _Case(unittest.TestCase):
    num_positions = 2
    num_t = 4
    shift = 0.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_positions(
            self.root / "acq", self.num_positions, self.num_t, shift=self.shift
        )
        self.out = str(self.root / "out")
        self.seqfile = open_sequence(str(self.root / "acq"))

    def tearDown(self):
        self._tmp.cleanup()


class TestExportWithoutRegistration(_Case):
    def _export(self, specs, **kwargs):
        manifest = _manifest(self.seqfile, specs)
        kwargs.setdefault("progress", False)
        return manifest, export_rois(manifest, self.out, source=self.seqfile, **kwargs)

    def test_one_tiff_per_selection(self):
        _, written = self._export(
            [
                (0, (20.0, 20.0), (10, 8), "a", 0),
                (0, (40.0, 40.0), (6, 6), "b", 0),
                (1, (30.0, 30.0), (12, 12), "c", 0),
            ]
        )
        self.assertEqual(
            [os.path.basename(p) for p in written],
            ["pos001_roi001.tiff", "pos001_roi002.tiff", "pos002_roi001.tiff"],
        )
        self.assertTrue(all(os.path.exists(p) for p in written))

    def test_roi_nb_is_per_position_and_naming_template_applies(self):
        _, written = self._export(
            [
                (0, (20.0, 20.0), (8, 8), "left", 0),
                (1, (20.0, 20.0), (8, 8), "right", 0),
            ],
            naming_template="{label}_p{position}_r{roi_nb}.tiff",
        )
        self.assertEqual(
            [os.path.basename(p) for p in written],
            ["left_p1_r1.tiff", "right_p2_r1.tiff"],
        )

    def test_crop_holds_that_position_s_pixels(self):
        _, written = self._export([(1, (30.0, 30.0), (10, 10), "c", 0)])
        stack = tifffile.imread(written[0])
        self.assertEqual(stack.shape[0], self.num_t)
        self.assertEqual(
            [int(np.unique(frame)[0]) for frame in stack.reshape(self.num_t, -1)],
            [_encode(1, t) for t in range(self.num_t)],
        )

    def test_position_without_rois_is_skipped(self):
        _, written = self._export([(1, (30.0, 30.0), (8, 8), "c", 0)])
        self.assertEqual(len(written), 1)
        self.assertIn("pos002", os.path.basename(written[0]))

    def test_max_frames_truncates_the_stack(self):
        _, written = self._export([(0, (20.0, 20.0), (8, 8), "a", 0)], max_frames=2)
        self.assertEqual(tifffile.imread(written[0]).shape[0], 2)

    def test_reopens_the_manifest_source_when_no_source_is_given(self):
        manifest = _manifest(self.seqfile, [(0, (20.0, 20.0), (8, 8), "a", 0)])
        written = export_rois(manifest, self.out, progress=False)
        self.assertEqual(len(written), 1)

    def test_anchor_frames_are_ignored_without_registration(self):
        # the crop is taken from the raw source, whose coordinates the ROI was
        # drawn in, so a late anchor needs no mapping and no error
        _, written = self._export([(0, (20.0, 20.0), (8, 8), "a", self.num_t - 1)])
        self.assertEqual(len(written), 1)


class TestAnchorValidation(_Case):
    def test_anchor_beyond_max_frames_fails_before_registering(self):
        manifest = _manifest(self.seqfile, [(0, (20.0, 20.0), (8, 8), "late", 3)])
        with self.assertRaises(ValueError) as ctx:
            export_rois(
                manifest,
                self.out,
                registration_method_name="GradientECC",
                source=self.seqfile,
                max_frames=2,
                progress=False,
            )
        message = str(ctx.exception)
        self.assertIn("drawn on frame 3", message)
        self.assertIn("max_frames", message)
        # nothing was written, and registration never started
        self.assertFalse(
            os.path.exists(os.path.join(self.out, "registration_transforms.json"))
        )


class TestExportWithRegistration(_Case):
    num_positions = 1
    num_t = 4
    shift = 2.0  # px per frame

    def test_registers_then_crops_and_records_transforms(self):
        import pytest

        pytest.importorskip("anywidget")
        manifest = _manifest(self.seqfile, [(0, (32.0, 32.0), (12, 12), "a", 0)])
        written = export_rois(
            manifest,
            self.out,
            registration_method_name="GradientECC",
            source=self.seqfile,
            progress=False,
        )
        self.assertEqual(len(written), 1)
        self.assertEqual(tifffile.imread(written[0]).shape[0], self.num_t)

        from acia.registration_persistence import RegistrationManifest

        transforms_json = os.path.join(self.out, "registration_transforms.json")
        record = RegistrationManifest.load(transforms_json).records[0]
        self.assertEqual(record.position, 0)
        self.assertEqual(len(record.transforms), self.num_t)
        # the fixture drifts by (+2, -2) px per frame and the record stores the
        # measured drift, so frame 3 sits near (+6, -6) (the empty border the
        # warp leaves behind costs a few tenths of a pixel)
        last = record.transforms[3]
        self.assertAlmostEqual(last.dx, 6.0, delta=1.0)
        self.assertAlmostEqual(last.dy, -6.0, delta=1.0)
        # the reference frame's own transform is *estimated*, not a stored
        # identity -- which is why a frame-0 ROI is passed through untouched
        self.assertAlmostEqual(record.transforms[0].dx, 0.0, places=6)

    def test_per_frame_warnings_are_folded_into_the_position_report(self):
        import warnings

        import pytest

        pytest.importorskip("anywidget")
        from acia.registration import LowConfidenceWarning

        manifest = _manifest(self.seqfile, [(0, (32.0, 32.0), (12, 12), "a", 0)])
        lines = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with unittest.mock.patch("acia.roi_export.tqdm.write", lines.append):
                export_rois(
                    manifest,
                    self.out,
                    registration_method_name="GradientECC",
                    # an unreachable gate: every frame is "low confidence"
                    method_kwargs={"min_confidence": 0.999999},
                    source=self.seqfile,
                )
        self.assertFalse(
            [w for w in caught if issubclass(w.category, LowConfidenceWarning)]
        )
        report = [line for line in lines if line.startswith("pos 1:")]
        self.assertEqual(len(report), 1)
        self.assertIn("below 0.999999", report[0])
        self.assertFalse([line for line in lines if line.startswith("wrote ")])

    def test_a_late_anchor_is_carried_onto_the_corrected_view(self):
        import pytest

        pytest.importorskip("anywidget")
        drawn_on_last = _manifest(
            self.seqfile, [(0, (32.0, 32.0), (12, 12), "late", self.num_t - 1)]
        )
        on_reference = _manifest(self.seqfile, [(0, (32.0, 32.0), (12, 12), "ref", 0)])
        late = export_rois(
            drawn_on_last,
            self.out,
            naming_template="late.tiff",
            registration_method_name="GradientECC",
            source=self.seqfile,
            progress=False,
        )
        ref = export_rois(
            on_reference,
            self.out,
            naming_template="ref.tiff",
            registration_method_name="GradientECC",
            source=self.seqfile,
            progress=False,
        )
        # same box, different anchor frames: the late one is shifted by the drift
        # accumulated up to its anchor, so the two crops must differ
        self.assertFalse(
            np.array_equal(tifffile.imread(late[0]), tifffile.imread(ref[0]))
        )


if __name__ == "__main__":
    unittest.main()


class TestExportInfo(_Case):
    """roi_export.json describes the export for batch analysis."""

    def test_written_with_experiment_and_1_based_positions(self):
        import json

        from acia.roi_export import experiment_name

        manifest = _manifest(
            self.seqfile,
            [(0, (20.0, 20.0), (8, 8), "a", 0), (1, (30.0, 30.0), (8, 8), "b", 0)],
        )
        export_rois(manifest, self.out, source=self.seqfile, progress=False)
        info = json.loads((Path(self.out) / "roi_export.json").read_text())

        self.assertEqual(info["schema"], "acia.roi_export/v2")
        self.assertEqual(
            [(r["file"], r["position"], r["position_index"]) for r in info["rois"]],
            [("pos001_roi001.tiff", 1, 0), ("pos002_roi001.tiff", 2, 1)],
        )
        self.assertEqual(experiment_name(self.out), info["experiment"])

        # a later export into the same folder keeps the earlier ROIs
        export_rois(
            _manifest(self.seqfile, [(1, (30.0, 30.0), (8, 8), "b", 0)]),
            self.out,
            source=self.seqfile,
            progress=False,
            experiment="custom",
        )
        info = json.loads((Path(self.out) / "roi_export.json").read_text())
        self.assertEqual(len(info["rois"]), 2)
        self.assertEqual(experiment_name(self.out), "custom")

    def test_default_export_dir_and_fallbacks(self):
        from acia.roi_export import default_export_dir, experiment_name

        self.assertEqual(
            default_export_dir("/data/exp/JW_nd73-01.czi"),
            Path("/data/exp/roi_exports"),
        )
        empty = Path(self.out) / "some_folder"
        empty.mkdir(parents=True)
        self.assertEqual(experiment_name(empty), "some_folder")

    def test_refuses_to_mix_two_acquisitions_in_one_folder(self):
        import json

        manifest = _manifest(self.seqfile, [(0, (20.0, 20.0), (8, 8), "a", 0)])
        export_rois(manifest, self.out, source=self.seqfile, progress=False)
        info_path = Path(self.out) / "roi_export.json"
        info = json.loads(info_path.read_text())
        info["source"]["path"] = "/elsewhere/other_acquisition.nd2"
        info_path.write_text(json.dumps(info))

        with self.assertRaisesRegex(ValueError, "other_acquisition.nd2"):
            export_rois(manifest, self.out, source=self.seqfile, progress=False)


class TestExportReview(_Case):
    """Rendering exported ROIs to preview videos for review."""

    def test_preview_is_rendered_cached_and_listed(self):
        from acia.notebook import roi_export_viewer
        from acia.roi_export import export_entries, preview_path, render_roi_preview

        manifest = _manifest(
            self.seqfile,
            [(0, (20.0, 20.0), (10, 8), "a", 0), (1, (30.0, 30.0), (12, 12), "b", 0)],
        )
        export_rois(manifest, self.out, source=self.seqfile, progress=False)

        entries = export_entries(self.out)
        self.assertEqual(
            [e["file"] for e in entries], ["pos001_roi001.tiff", "pos002_roi001.tiff"]
        )

        target = preview_path(self.out, entries[0]["file"], fps=5, max_size=64)
        info = render_roi_preview(
            os.path.join(self.out, entries[0]["file"]), target, fps=5, max_size=64
        )
        self.assertTrue(target.is_file() and target.stat().st_size > 0)
        self.assertEqual(info["frames"], self.num_t)

        panel = roi_export_viewer(self.out, fps=5, max_size=64)
        picker, caption, video = panel.children
        self.assertEqual(picker.options[1][0], "pos 002 · roi 001 · b")
        self.assertIn("cached preview", caption.value)  # rendered above already
        self.assertGreater(len(video.value), 0)
        picker.value = "pos002_roi001.tiff"
        self.assertIn("rendered in", caption.value)
        # a fixed box: switching to an ROI of another aspect never resizes it
        self.assertEqual((video.layout.width, video.layout.height), ("64px", "64px"))
        self.assertEqual(video.layout.object_fit, "contain")
        boxed = roi_export_viewer(self.out, fps=5, max_size=64, box_size=200)
        self.assertEqual(boxed.children[2].layout.height, "200px")


# --------------------------------------------------------------------------- #
# one read and one correction per frame, however many ROIs
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _count_reads():
    """``{(position folder, frame): reads}`` for every decode during the block."""
    from acia.segm.folder_source import FolderSequenceSource

    counts: dict[tuple[str, int], int] = {}
    original = FolderSequenceSource.get_frame

    def counting(self, index):
        key = (os.path.basename(self.folder), int(index))
        counts[key] = counts.get(key, 0) + 1
        return original(self, index)

    with unittest.mock.patch.object(FolderSequenceSource, "get_frame", counting):
        yield counts


def _reads_per_frame(counts, folder="pos000", skip_first=True):
    """Reads of each frame of one position; frame 0 is special, see the tests."""
    return [
        n
        for (f, t), n in sorted(counts.items())
        if f == folder and (t or not skip_first)
    ]


class _ThreeRois:
    THREE_ROIS = [
        (0, (16.0, 16.0), (8, 8), "a", 0),
        (0, (32.0, 32.0), (8, 8), "b", 0),
        (0, (48.0, 48.0), (8, 8), "c", 0),
    ]


class TestFramesOuterPass(_ThreeRois, _Case):
    """A position is walked once, not once per ROI."""

    num_positions = 1
    num_t = 6

    def test_each_frame_is_read_once_without_registration(self):
        manifest = _manifest(self.seqfile, self.THREE_ROIS)
        with _count_reads() as counts:
            export_rois(manifest, self.out, source=self.seqfile, progress=False)
        # Three ROIs, one read per frame -- not one pass per ROI.
        self.assertEqual(_reads_per_frame(counts), [1] * (self.num_t - 1))

    def test_all_three_rois_hold_their_own_pixels(self):
        manifest = _manifest(self.seqfile, self.THREE_ROIS)
        written = export_rois(manifest, self.out, source=self.seqfile, progress=False)
        self.assertEqual(len(written), 3)
        for path in written:
            stack = tifffile.imread(path)
            self.assertEqual(stack.shape, (self.num_t, 8, 8))
            for t in range(self.num_t):
                self.assertTrue((stack[t] == _encode(0, t)).all())


class TestFramesOuterPassRegistered(_ThreeRois, _Case):
    """Same, with drift correction in play (which needs frames to lock onto)."""

    num_positions = 1
    num_t = 6
    shift = 2.0  # px per frame

    def test_each_frame_is_read_twice_with_registration(self):
        manifest = _manifest(self.seqfile, self.THREE_ROIS)
        with _count_reads() as counts:
            export_rois(
                manifest,
                self.out,
                registration_method_name="GradientECC",
                source=self.seqfile,
                progress=False,
            )
        # Once to estimate the transform, once to crop -- still independent of
        # how many ROIs the position holds.
        self.assertEqual(_reads_per_frame(counts), [2] * (self.num_t - 1))

    def test_no_frame_is_ever_corrected_as_a_whole(self):
        """The drift correction is composed into each crop, not applied first."""
        import acia.registration

        manifest = _manifest(self.seqfile, self.THREE_ROIS)
        calls = []
        original = acia.registration.apply_correction

        def counting(frame, transform):
            calls.append(transform)
            return original(frame, transform)

        with unittest.mock.patch.object(
            acia.registration, "apply_correction", counting
        ):
            written = export_rois(
                manifest,
                self.out,
                registration_method_name="GradientECC",
                source=self.seqfile,
                progress=False,
            )
        self.assertEqual(len(written), 3)
        # A per-ROI pass used to do T*3 of these and the frames-outer pass T;
        # composing the correction into the crop warp does away with them.
        self.assertEqual(calls, [])

    def test_the_crop_is_one_warp_per_roi_per_frame(self):
        import acia.base

        manifest = _manifest(self.seqfile, self.THREE_ROIS)
        calls = []
        original = acia.base.warp_rotated_crop

        def counting(raw, spec):
            calls.append(spec)
            return original(raw, spec)

        with unittest.mock.patch.object(acia.base, "warp_rotated_crop", counting):
            export_rois(
                manifest,
                self.out,
                registration_method_name="GradientECC",
                source=self.seqfile,
                progress=False,
            )
        self.assertEqual(len(calls), 3 * self.num_t)


class TestRoiGroups(unittest.TestCase):
    """The ROI buffers are split into budget-sized passes, in manifest order."""

    @staticmethod
    def _specs(count, size=10):
        return [
            RotatedCropSpec(center=(20.0, 20.0), size=(size, size), angle=0.0)
            for _ in range(count)
        ]

    def test_no_budget_is_a_single_pass(self):
        groups = _roi_groups(self._specs(4), 10, 1, 2, None)
        self.assertEqual(groups, [[0, 1, 2, 3]])

    def test_consecutive_groups_fit_the_budget_and_keep_the_order(self):
        specs = self._specs(5)  # 10 * 10 * 10 * 1 * 2 = 2000 bytes each
        groups = _roi_groups(specs, 10, 1, 2, 4500)
        self.assertEqual(groups, [[0, 1], [2, 3], [4]])
        for group in groups:
            total = sum(_roi_bytes(specs[i], 10, 1, 2) for i in group)
            self.assertLessEqual(total, 4500)

    def test_an_roi_that_does_not_fit_stands_alone(self):
        specs = self._specs(3)
        groups = _roi_groups(specs, 10, 1, 2, 100)
        self.assertEqual(groups, [[0], [1], [2]])


class TestRoiBufferBudget(_Case):
    num_positions = 1
    num_t = 5

    SPECS = [
        (0, (16.0, 16.0), (8, 8), "a", 0),
        (0, (32.0, 32.0), (8, 8), "b", 0),
    ]

    def _export(self, out, **kwargs):
        manifest = _manifest(self.seqfile, self.SPECS)
        kwargs.setdefault("progress", False)
        return export_rois(manifest, out, source=self.seqfile, **kwargs)

    def test_a_small_budget_splits_the_pass_but_not_the_output(self):
        unlimited = self._export(str(self.root / "a"))
        with _count_reads() as counts:
            limited = self._export(
                str(self.root / "b"), max_roi_buffer_bytes=5 * 8 * 8 * 2
            )
        for one, other in zip(unlimited, limited, strict=True):
            np.testing.assert_array_equal(tifffile.imread(one), tifffile.imread(other))
        # One ROI per pass now, so each frame is decoded once per pass.
        self.assertEqual(_reads_per_frame(counts), [2] * (self.num_t - 1))

    def test_an_roi_too_big_to_buffer_still_exports(self):
        lazy = self._export(str(self.root / "c"), max_roi_buffer_bytes=1)
        buffered = self._export(str(self.root / "d"))
        for one, other in zip(lazy, buffered, strict=True):
            np.testing.assert_array_equal(tifffile.imread(one), tifffile.imread(other))


class TestBudgetWithRegistration(_Case):
    """The streamed and buffered paths must correct pixels the same way."""

    num_positions = 1
    num_t = 5
    shift = 2.0  # px per frame

    def _export(self, out, **kwargs):
        manifest = _manifest(
            self.seqfile,
            [(0, (40.0, 40.0), (12, 12), "a", 0), (0, (56.0, 56.0), (12, 12), "b", 0)],
        )
        kwargs.setdefault("progress", False)
        return export_rois(
            manifest,
            out,
            registration_method_name="GradientECC",
            source=self.seqfile,
            **kwargs,
        )

    def test_the_lazy_fallback_writes_the_same_pixels_as_the_buffers(self):
        # An ROI that cannot be buffered streams through `crop_rotated`, which
        # composes the correction into the crop just as the buffered pass does;
        # if one of the two ever went back to correcting whole frames, the two
        # outputs would quietly stop matching.
        lazy = self._export(str(self.root / "lazy"), max_roi_buffer_bytes=1)
        buffered = self._export(str(self.root / "buffered"))
        self.assertEqual(len(lazy), 2)
        for one, other in zip(lazy, buffered, strict=True):
            np.testing.assert_array_equal(tifffile.imread(one), tifffile.imread(other))


class TestChunkFrames(_Case):
    num_positions = 1
    num_t = 7

    def test_read_ahead_does_not_change_the_output(self):
        specs = [(0, (16.0, 16.0), (8, 8), "a", 0), (0, (40.0, 40.0), (8, 8), "b", 0)]
        out = {}
        for depth in (1, 4):
            manifest = _manifest(self.seqfile, specs)
            out[depth] = export_rois(
                manifest,
                str(self.root / f"out{depth}"),
                source=self.seqfile,
                progress=False,
                chunk_frames=depth,
            )
        for one, other in zip(out[1], out[4], strict=True):
            np.testing.assert_array_equal(tifffile.imread(one), tifffile.imread(other))

    def test_chunk_frames_must_be_at_least_one(self):
        manifest = _manifest(self.seqfile, [(0, (16.0, 16.0), (8, 8), "a", 0)])
        with self.assertRaisesRegex(ValueError, "chunk_frames"):
            export_rois(
                manifest, self.out, source=self.seqfile, progress=False, chunk_frames=0
            )


class TestMissingTransformReport(_Case):
    """The per-position "no transform" line survives the single-pass crop."""

    num_positions = 1
    num_t = 5
    shift = 0.6

    def test_a_frame_without_a_transform_is_still_reported(self):
        manifest = _manifest(
            self.seqfile,
            [(0, (24.0, 24.0), (8, 8), "a", 0), (0, (40.0, 40.0), (8, 8), "b", 0)],
        )
        real = _register_position

        def drop_one(*args, **kwargs):
            record = real(*args, **kwargs)
            transforms = dict(record.transforms)
            transforms.pop(2, None)
            return dataclasses.replace(record, transforms=transforms)

        lines = []
        with (
            unittest.mock.patch("acia.roi_export._register_position", drop_one),
            unittest.mock.patch(
                "tqdm.auto.tqdm.write", lambda text: lines.append(text)
            ),
        ):
            export_rois(
                manifest,
                self.out,
                registration_method_name="GradientECC",
                source=self.seqfile,
                progress=True,
            )
        reported = [text for text in lines if "had no transform of their own" in text]
        self.assertEqual(len(reported), 1, lines)
        self.assertIn("were corrected with the nearest frame", reported[0])


# --------------------------------------------------------------------------- #
# re-running: export only what is new, and pick up an interrupted run
# --------------------------------------------------------------------------- #
class TestIncrementalExport(_Case):
    num_positions = 2
    num_t = 4

    FIRST = [(0, (16.0, 16.0), (8, 8), "a", 0)]
    BOTH = [(0, (16.0, 16.0), (8, 8), "a", 0), (1, (32.0, 32.0), (8, 8), "b", 0)]

    def _rois(self):
        return json.loads((Path(self.out) / "roi_export.json").read_text())["rois"]

    def test_only_a_newly_drawn_roi_is_exported(self):
        export_rois(
            _manifest(self.seqfile, self.FIRST),
            self.out,
            source=self.seqfile,
            progress=False,
        )
        first = Path(self.out) / "pos001_roi001.tiff"
        stamp = first.stat().st_mtime_ns

        with _count_reads() as counts:
            written = export_rois(
                _manifest(self.seqfile, self.BOTH),
                self.out,
                source=self.seqfile,
                progress=False,
            )

        self.assertEqual([os.path.basename(p) for p in written], ["pos002_roi001.tiff"])
        self.assertEqual(first.stat().st_mtime_ns, stamp, "pos 1 was rewritten")
        # The finished position was never opened again.
        self.assertEqual(_reads_per_frame(counts, "pos000", skip_first=False), [])
        self.assertEqual(
            {r["file"] for r in self._rois()},
            {"pos001_roi001.tiff", "pos002_roi001.tiff"},
        )

    def test_an_roi_inserted_before_another_keeps_the_old_numbering(self):
        later = [(0, (16.0, 16.0), (8, 8), "a", 0)]
        export_rois(
            _manifest(self.seqfile, later),
            self.out,
            source=self.seqfile,
            progress=False,
        )
        # The new ROI sorts *first* in the manifest but must not take roi001.
        manifest = _manifest(
            self.seqfile,
            [(0, (40.0, 40.0), (8, 8), "new", 0), (0, (16.0, 16.0), (8, 8), "a", 0)],
        )
        manifest.selections[1].id = "sel0"  # the already-exported selection
        manifest.selections[0].id = "selnew"
        written = export_rois(manifest, self.out, source=self.seqfile, progress=False)
        self.assertEqual([os.path.basename(p) for p in written], ["pos001_roi002.tiff"])
        by_id = {r["id"]: r["roi_nb"] for r in self._rois()}
        self.assertEqual(by_id, {"sel0": 1, "selnew": 2})

    def test_a_moved_roi_is_re_exported(self):
        export_rois(
            _manifest(self.seqfile, self.FIRST),
            self.out,
            source=self.seqfile,
            progress=False,
        )
        moved = _manifest(self.seqfile, [(0, (40.0, 40.0), (8, 8), "a", 0)])
        written = export_rois(moved, self.out, source=self.seqfile, progress=False)
        self.assertEqual([os.path.basename(p) for p in written], ["pos001_roi001.tiff"])

    def test_changed_settings_re_export(self):
        manifest = _manifest(self.seqfile, self.FIRST)
        export_rois(manifest, self.out, source=self.seqfile, progress=False)
        written = export_rois(
            manifest, self.out, source=self.seqfile, progress=False, max_frames=2
        )
        self.assertEqual([os.path.basename(p) for p in written], ["pos001_roi001.tiff"])
        self.assertEqual(tifffile.imread(written[0]).shape[0], 2)

    def test_a_truncated_file_is_detected_and_redone(self):
        manifest = _manifest(self.seqfile, self.FIRST)
        export_rois(manifest, self.out, source=self.seqfile, progress=False)
        target = Path(self.out) / "pos001_roi001.tiff"
        tifffile.imwrite(target, tifffile.imread(target)[:2])  # a short stack

        written = export_rois(manifest, self.out, source=self.seqfile, progress=False)
        self.assertEqual([os.path.basename(p) for p in written], ["pos001_roi001.tiff"])
        self.assertEqual(tifffile.imread(target).shape[0], self.num_t)

    def test_verify_existing_false_trusts_the_file(self):
        manifest = _manifest(self.seqfile, self.FIRST)
        export_rois(manifest, self.out, source=self.seqfile, progress=False)
        target = Path(self.out) / "pos001_roi001.tiff"
        tifffile.imwrite(target, tifffile.imread(target)[:2])

        written = export_rois(
            manifest,
            self.out,
            source=self.seqfile,
            progress=False,
            verify_existing=False,
        )
        self.assertEqual(written, [])
        self.assertEqual(tifffile.imread(target).shape[0], 2)

    def test_overwrite_redoes_everything(self):
        manifest = _manifest(self.seqfile, self.BOTH)
        export_rois(manifest, self.out, source=self.seqfile, progress=False)
        written = export_rois(
            manifest,
            self.out,
            source=self.seqfile,
            progress=False,
            on_existing="overwrite",
        )
        self.assertEqual(len(written), 2)

    def test_error_refuses_to_touch_an_existing_export(self):
        manifest = _manifest(self.seqfile, self.FIRST)
        export_rois(manifest, self.out, source=self.seqfile, progress=False)
        with self.assertRaisesRegex(ValueError, "on_existing='error'"):
            export_rois(
                manifest,
                self.out,
                source=self.seqfile,
                progress=False,
                on_existing="error",
            )

    def test_an_unknown_on_existing_policy_is_rejected(self):
        manifest = _manifest(self.seqfile, self.FIRST)
        with self.assertRaisesRegex(ValueError, "Unknown on_existing"):
            export_rois(
                manifest,
                self.out,
                source=self.seqfile,
                progress=False,
                on_existing="resume",
            )


class TestInterruptedExport(_Case):
    """A run killed mid-write leaves no half-TIFF, and the next one finishes it."""

    num_positions = 2
    num_t = 4
    BOTH = [(0, (16.0, 16.0), (8, 8), "a", 0), (1, (32.0, 32.0), (8, 8), "b", 0)]

    def test_a_crash_records_what_finished_and_leaves_no_broken_tiff(self):
        manifest = _manifest(self.seqfile, self.BOTH)
        real = save_tiff_stack
        seen = []

        def fail_on_the_second(source, path, **kwargs):
            seen.append(path)
            if len(seen) == 2:
                raise KeyboardInterrupt("killed mid-write")
            return real(source, path, **kwargs)

        with (
            unittest.mock.patch(
                "acia.segm.tiff_export.save_tiff_stack", fail_on_the_second
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            export_rois(manifest, self.out, source=self.seqfile, progress=False)

        # The first ROI is complete; the second never appeared under its name.
        self.assertEqual(
            sorted(p.name for p in Path(self.out).glob("*.tif*")),
            ["pos001_roi001.tiff"],
        )
        # ... and what did finish is on record, although the run never reached
        # the end where the description used to be written.
        rois = json.loads((Path(self.out) / "roi_export.json").read_text())["rois"]
        self.assertEqual([r["file"] for r in rois], ["pos001_roi001.tiff"])

        # The next run picks up exactly what is missing.
        written = export_rois(manifest, self.out, source=self.seqfile, progress=False)
        self.assertEqual([os.path.basename(p) for p in written], ["pos002_roi001.tiff"])

    def test_a_half_written_tiff_is_never_visible_under_its_real_name(self):
        manifest = _manifest(self.seqfile, self.BOTH)
        real = save_tiff_stack

        def die_mid_write(source, path, **kwargs):
            real(source, path, **kwargs)  # the temporary file now exists
            raise KeyboardInterrupt("killed just before the rename")

        with (
            unittest.mock.patch("acia.segm.tiff_export.save_tiff_stack", die_mid_write),
            self.assertRaises(KeyboardInterrupt),
        ):
            export_rois(manifest, self.out, source=self.seqfile, progress=False)

        self.assertEqual(
            sorted(p.name for p in Path(self.out).glob("*.tif*")),
            ["pos001_roi001.part.tiff"],
        )
        self.assertFalse((Path(self.out) / "pos001_roi001.tiff").exists())

        # The leftover is cleared and the ROI exported properly.
        written = export_rois(manifest, self.out, source=self.seqfile, progress=False)
        self.assertEqual(len(written), 2)
        self.assertEqual(list(Path(self.out).glob("*.part.*")), [])
