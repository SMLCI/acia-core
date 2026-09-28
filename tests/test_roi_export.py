"""Unit tests for :func:`acia.roi_export.export_rois`.

The source is a real multi-position handle (a folder of per-timepoint TIFFs, one
subfolder per position), so ``position()``, ``metadata`` and the lazy slicing all
behave as they do on an ND2/CZI. Pixels encode their ``(position, frame)`` so a
crop can be traced back to the frame it came from.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from acia.base import RotatedCropSpec
from acia.roi_export import export_rois
from acia.segm.open import open_sequence
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

        self.assertEqual(info["schema"], "acia.roi_export/v1")
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
