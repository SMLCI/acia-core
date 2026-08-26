"""Python-surface tests for :class:`acia.notebook.RegistrationDashboard`.

anywidget must be installed (the ``widget`` extra). The ESM JavaScript is NOT
exercised here -- it is validated only by a real Jupyter/Colab/marimo run (no
ESM/Playwright suite for this widget in v1, per the spec). These tests cover
the traits, the ``on_msg`` handlers (``mask_frame``/``verify``/``batch_apply``/
``save``), the mask-rect point-fit observer, the ``manifest``/``save`` build,
batch-apply resumability + per-frame/per-position failure isolation, and
import-safety without anywidget.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import cv2
import ipywidgets
import numpy as np
import pytest
import tifffile

pytest.importorskip("anywidget")

import traitlets  # noqa: E402

from acia.base import RotatedCropSpec  # noqa: E402
from acia.notebook import RegistrationDashboard  # noqa: E402
from acia.registration import (  # noqa: E402
    FrameTransform,
    LowConfidenceWarning,
)
from acia.registration_persistence import (  # noqa: E402
    RegistrationManifest,
    RegistrationRecord,
)
from acia.segm.open import open_sequence  # noqa: E402


class _Recorder:
    """Captures ``widget.send`` calls (content, buffers)."""

    def __init__(self):
        self.sent = []

    def __call__(self, content, buffers=None):
        self.sent.append((content, buffers))


def _textured(seed: int, size: int = 48) -> np.ndarray:
    """A blurred-noise frame with real texture but no straight edges."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (size, size), dtype=np.uint8).astype(np.float32)
    return cv2.GaussianBlur(frame, (3, 3), 0).astype(np.uint8)


def _warp(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)


def _tif_translating(d, t=4, size=48, shift=2.0, positions=1):
    """A small TIFF stack that translates a fixed amount per frame.

    Real (non-wraparound) structure + a small, recoverable per-frame shift --
    unlike pure random noise, this reliably converges for
    ``PhaseCorrelationHighpass`` so the "before/after available" verify path
    is actually exercised, not just its no-correction fallback.
    """
    ref = _textured(0, size=size)
    stack = np.stack([_warp(ref, shift * i, -shift * i) for i in range(t)])
    path = Path(d) / "stack.tif"
    tifffile.imwrite(path, stack)
    return path


def _tif_random(d, t=3, h=20, w=20):
    path = Path(d) / "stack.tif"
    tifffile.imwrite(path, (np.random.rand(t, h, w) * 1000).astype(np.uint16))
    return path


_AXIS_ALIGNED = [[10, 10], [30, 10], [30, 20], [10, 20]]  # 4 corners -> angle ~0


class TestConstruct(unittest.TestCase):
    def test_traits_populated(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            self.assertEqual(dash.metadata["num_positions"], 1)
            self.assertEqual(dash.metadata["num_timepoints"], 3)
            self.assertEqual(len(dash.positions), 1)
            self.assertEqual(dash.method_name, "GradientECC")
            self.assertEqual(dash.n_sample_frames, 8)
            self.assertIsNone(dash.mask_rect)
            self.assertFalse(dash.batch_running)

    def test_construct_from_path_and_method_name(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(
                str(_tif_random(d)), method_name="PhaseCorrelationHighpass"
            )
            self.assertEqual(dash.method_name, "PhaseCorrelationHighpass")

    def test_invalid_method_name_rejected(self):
        with (
            tempfile.TemporaryDirectory() as d,
            self.assertRaises(traitlets.TraitError),
        ):
            RegistrationDashboard(open_sequence(_tif_random(d)), method_name="Bogus")

    def test_method_name_trait_validated_after_construction(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            with self.assertRaises(traitlets.TraitError):
                dash.method_name = "NotAMethod"

    def test_n_sample_frames_must_be_positive(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            with self.assertRaises(traitlets.TraitError):
                dash.n_sample_frames = 0

    def test_repr_html(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            html = dash._repr_html_()
            self.assertIn("RegistrationDashboard", html)
            self.assertIn("GradientECC", html)

    def test_does_not_shadow_widget_dispatcher(self):
        """Regression guard, mirrors the SequenceDashboard test: a custom
        on_msg callback must not be named ``_handle_msg`` -- that name
        collides with ipywidgets' own comm-message dispatcher.
        """
        self.assertIs(RegistrationDashboard._handle_msg, ipywidgets.Widget._handle_msg)


class TestMaskPointsObserver(unittest.TestCase):
    def test_points_seed_mask_rect(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            dash.mask_points = _AXIS_ALIGNED
            self.assertAlmostEqual(dash.mask_angle, 0.0, delta=1e-3)
            self.assertEqual({dash.mask_width, dash.mask_height}, {20, 10})
            spec = dash.mask_rect
            self.assertIsNotNone(spec)
            self.assertEqual(spec.size, (dash.mask_width, dash.mask_height))

    def test_too_few_points_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            before = (dash.mask_width, dash.mask_height)
            dash.mask_points = [[0, 0], [1, 1]]
            self.assertEqual((dash.mask_width, dash.mask_height), before)
            self.assertIsNone(dash.mask_rect)


class TestMaskFrameMessage(unittest.TestCase):
    def test_mask_frame_sets_image_traits(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            rec = _Recorder()
            dash.send = rec
            dash._on_custom_msg(dash, {"type": "mask_frame", "position": 0}, None)
            self.assertTrue(dash.mask_image_b64.startswith("data:image/png;base64,"))
            self.assertGreater(dash.mask_image_w, 0)
            self.assertGreater(dash.mask_image_h, 0)
            content, buffers = rec.sent[0]
            self.assertEqual(content, {"type": "mask_frame", "position": 0})
            self.assertIsNone(buffers)

    def test_mask_frame_error_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            rec = _Recorder()
            dash.send = rec
            dash._on_custom_msg(dash, {"type": "mask_frame", "position": 5}, None)
            content, buffers = rec.sent[0]
            self.assertEqual(content["type"], "error")
            self.assertEqual(content["kind"], "mask_frame")
            self.assertIsNone(buffers)


class TestVerifyMessage(unittest.TestCase):
    def test_verify_reports_progress_and_full_frame_buffers(self):
        """verify sends a per-frame "progress" message (phase="verify") via
        run_comparison's on_progress callback, then a final verify_result
        carrying one uncorrected PNG buffer per sampled frame plus a
        corrected one wherever has_correction[i] is true (the comparison
        player's data source)."""
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            dash.n_sample_frames = 3
            rec = _Recorder()
            dash.send = rec
            dash._on_custom_msg(
                dash,
                {"type": "verify", "position": 0, "method": "PhaseCorrelationHighpass"},
                None,
            )

            progress = [c for c, _ in rec.sent if c.get("type") == "progress"]
            content, buffers = rec.sent[-1]
            self.assertEqual(content["type"], "verify_result")

            num_frames = len(content["frame_indices"])
            self.assertEqual(len(progress), num_frames)
            for i, msg in enumerate(progress):
                self.assertEqual(msg["phase"], "verify")
                self.assertEqual(msg["frame"], i)
                self.assertEqual(msg["num_frames"], num_frames)

            self.assertEqual(content["position"], 0)
            self.assertEqual(content["reference_frame"], 0)
            self.assertEqual(len(content["frame_indices"]), len(content["transforms"]))
            self.assertEqual(len(content["has_correction"]), num_frames)
            # a translating stack must converge for at least one sampled frame
            self.assertTrue(any(content["has_correction"]))

            expected_buffers = 1 + sum(
                2 if available else 1 for available in content["has_correction"]
            )
            self.assertEqual(len(buffers), expected_buffers)
            for png in buffers:
                self.assertTrue(png.startswith(b"\x89PNG"))
            # transforms are plain JSON-safe dicts (dx/dy/theta), not objects
            for t in content["transforms"]:
                self.assertTrue(t is None or {"dx", "dy", "theta"} <= t.keys())

    def test_verify_defaults_to_widget_method_and_position(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            rec = _Recorder()
            dash.send = rec
            dash._on_custom_msg(dash, {"type": "verify"}, None)
            content, _buffers = rec.sent[-1]
            self.assertEqual(content["type"], "verify_result")
            self.assertEqual(content["method"], "GradientECC")
            self.assertEqual(content["position"], 0)

    def test_verify_masked_template_without_mask_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(
                open_sequence(_tif_random(d)), method_name="MaskedTemplateCorrelation"
            )
            rec = _Recorder()
            dash.send = rec
            dash._on_custom_msg(dash, {"type": "verify"}, None)
            content, buffers = rec.sent[0]
            self.assertEqual(content["type"], "error")
            self.assertEqual(content["kind"], "verify")
            self.assertIn("mask rect", content["message"])
            self.assertIsNone(buffers)


class TestBatchApply(unittest.TestCase):
    def test_batch_apply_writes_manifest_with_progress(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=3)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            rec = _Recorder()
            dash.send = rec
            out = Path(d) / "out"
            summary = dash.batch_apply(directory=out)

            self.assertEqual(summary["completed"], [0])
            self.assertEqual(summary["skipped"], [])
            self.assertEqual(summary["failed_positions"], [])
            self.assertTrue(Path(summary["path"]).exists())

            progress = [c for c, _ in rec.sent if c.get("type") == "progress"]
            self.assertEqual(len(progress), 3)  # one per frame (0, 1, 2)
            first = progress[0]
            self.assertEqual(
                {
                    k: first[k]
                    for k in (
                        "type",
                        "position",
                        "num_positions",
                        "frame",
                        "num_frames",
                    )
                },
                {
                    "type": "progress",
                    "position": 0,
                    "num_positions": 1,
                    "frame": 0,
                    "num_frames": 3,
                },
            )
            # best-effort elapsed/ETA fields (ETA may be None early on, before
            # a rate can be estimated)
            self.assertIsInstance(first["elapsed_seconds"], float)
            self.assertGreaterEqual(first["elapsed_seconds"], 0)
            self.assertIn("eta_seconds", first)
            last = progress[-1]
            self.assertIsInstance(last["elapsed_seconds"], float)

            manifest = RegistrationManifest.load(summary["path"])
            self.assertEqual(len(manifest.records), 1)
            self.assertEqual(manifest.records[0].position, 0)
            self.assertEqual(set(manifest.records[0].transforms.keys()), {0, 1, 2})

    def test_batch_apply_message_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=2)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            rec = _Recorder()
            dash.send = rec
            out = Path(d) / "out"
            dash._on_custom_msg(
                dash, {"type": "batch_apply", "directory": str(out)}, None
            )
            kinds = [c["type"] for c, _ in rec.sent]
            self.assertIn("progress", kinds)
            self.assertEqual(kinds[-1], "batch_done")
            self.assertTrue((out / "registration_transforms.json").exists())

    def test_batch_apply_resumes_and_skips_completed_positions(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=2)
            out = Path(d) / "out"

            dash1 = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            dash1.send = _Recorder()
            first = dash1.batch_apply(directory=out)
            self.assertEqual(first["completed"], [0])

            # a fresh widget instance pointed at the same output path must skip
            # the already-completed position instead of recomputing it.
            dash2 = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            rec2 = _Recorder()
            dash2.send = rec2
            second = dash2.batch_apply(directory=out)
            self.assertEqual(second["completed"], [])
            self.assertEqual(second["skipped"], [0])
            # no progress messages -- the position was skipped, never touched
            self.assertFalse(any(c.get("type") == "progress" for c, _ in rec2.sent))

    def test_batch_apply_resumes_partial_position_from_first_uncomputed_frame(self):
        """A partial checkpoint (some but not all frames done) must resume
        from the first uncomputed frame -- not be wrongly fully skipped
        (today's bug once partial checkpoints exist) and not be fully redone.
        """
        from acia.registration import FrameTransform
        from acia.registration_persistence import RegistrationRecord, save_registration

        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=5)
            out = Path(d) / "out"

            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )

            # Seed a manifest as if a prior run got through frames 0-1 of a
            # 5-frame position before being interrupted. A sentinel transform
            # value that a real estimate() would never produce, so recompute
            # vs. preserve is unambiguous.
            sentinel = FrameTransform(dx=999.0, dy=999.0, theta=0.0)
            partial = RegistrationRecord(
                position=0,
                method="PhaseCorrelationHighpass",
                transforms={0: sentinel, 1: sentinel},
                reference_frame=0,
                failed_frames={},
            )
            manifest = RegistrationManifest(
                source=dash.manifest.source,
                records=[partial],
                method="PhaseCorrelationHighpass",
            )
            save_registration(manifest, out)

            rec = _Recorder()
            dash.send = rec
            summary = dash.batch_apply(directory=out)

            self.assertEqual(summary["completed"], [0])
            self.assertEqual(summary["skipped"], [])

            progress = [c for c, _ in rec.sent if c.get("type") == "progress"]
            # only the previously-uncomputed frames (2, 3, 4) are processed
            self.assertEqual([p["frame"] for p in progress], [2, 3, 4])

            final = RegistrationManifest.load(summary["path"]).records[0]
            self.assertEqual(
                set(final.transforms) | set(final.failed_frames), {0, 1, 2, 3, 4}
            )
            # frames 0/1 were not recomputed -- the sentinel survives untouched
            self.assertEqual(final.transforms[0].dx, 999.0)
            self.assertEqual(final.transforms[1].dx, 999.0)

    def test_batch_apply_checkpoints_within_a_position(self):
        """The manifest is persisted every CHECKPOINT_INTERVAL frames within a
        position, not only after it fully completes."""
        from unittest import mock

        import acia.registration_persistence as reg_persistence
        from acia.notebook import CHECKPOINT_INTERVAL

        with tempfile.TemporaryDirectory() as d:
            n_frames = CHECKPOINT_INTERVAL + 5
            path = _tif_translating(d, t=n_frames)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            dash.send = _Recorder()
            out = Path(d) / "out"

            real_save = reg_persistence.save_registration
            calls = []

            def _wrapped(manifest, directory):
                if manifest.records:
                    rec0 = manifest.records[0]
                    calls.append(len(rec0.transforms) + len(rec0.failed_frames))
                else:
                    calls.append(0)
                return real_save(manifest, directory)

            with mock.patch.object(
                reg_persistence, "save_registration", side_effect=_wrapped
            ):
                summary = dash.batch_apply(directory=out)

            self.assertEqual(summary["completed"], [0])
            # a from-scratch single-position run without mid-position
            # checkpointing would only ever call save_registration twice (once
            # right after the position finishes inside the loop, once more
            # after the loop) -- a 3rd call proves a mid-position checkpoint
            # fired at CHECKPOINT_INTERVAL frames.
            self.assertGreater(len(calls), 2)
            self.assertIn(CHECKPOINT_INTERVAL, calls)

    def test_batch_apply_isolates_per_frame_failures(self):
        """One frame raising during estimate() must not abort the position."""
        with tempfile.TemporaryDirectory() as d:
            # random (unstructured) frames: PhaseCorrelationHighpass tends to
            # fail per-frame on pure noise -- exercise the failed_frames path
            # without aborting the run.
            path = _tif_random(d, t=3, h=16, w=16)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            dash.send = _Recorder()
            out = Path(d) / "out"
            summary = dash.batch_apply(directory=out)
            self.assertEqual(summary["completed"], [0])
            manifest = RegistrationManifest.load(summary["path"])
            record = manifest.records[0]
            # every frame accounted for, either as a transform or a failure
            self.assertEqual(
                set(record.transforms) | set(record.failed_frames), {0, 1, 2}
            )

    def test_batch_apply_isolates_whole_position_failure(self):
        """A position whose reference-frame read raises must not abort the run."""
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=2)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            dash.send = _Recorder()

            def _boom(self, pos, method_name, mask_rect, num_positions, **kwargs):
                raise RuntimeError("simulated read failure")

            dash._register_position = _boom.__get__(dash, RegistrationDashboard)
            out = Path(d) / "out"
            summary = dash.batch_apply(directory=out)
            self.assertEqual(summary["failed_positions"], [0])
            self.assertEqual(summary["completed"], [0])
            manifest = RegistrationManifest.load(summary["path"])
            self.assertEqual(manifest.records[0].transforms, {})
            self.assertIn("simulated read failure", manifest.records[0].notes)

    def test_batch_apply_failure_after_checkpoint_preserves_progress(self):
        """A whole-position failure that happens *after* a checkpoint fired
        for the resumed attempt must not wipe out the checkpointed progress
        -- the persisted manifest must still contain those frames, not an
        empty record clobbering the checkpoint.
        """
        from unittest import mock

        from acia.registration import FrameTransform
        from acia.registration_persistence import RegistrationRecord, save_registration

        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=6)
            out = Path(d) / "out"

            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )

            # Seed a manifest as if a prior run got through frames 0-1 of a
            # 6-frame position before being interrupted.
            sentinel = FrameTransform(dx=999.0, dy=999.0, theta=0.0)
            partial = RegistrationRecord(
                position=0,
                method="PhaseCorrelationHighpass",
                transforms={0: sentinel, 1: sentinel},
                reference_frame=0,
                failed_frames={},
            )
            manifest = RegistrationManifest(
                source=dash.manifest.source,
                records=[partial],
                method="PhaseCorrelationHighpass",
            )
            save_registration(manifest, out)

            # A `send` that blows up right after the frame-3 progress message
            # -- with CHECKPOINT_INTERVAL patched to 2, a checkpoint for
            # frames {0, 1, 2, 3} fires (inside the resumed attempt, which
            # starts at frame 2) just before that same message is sent, so
            # the failure lands strictly after the checkpoint.
            class _RaiseAfterFrame:
                def __init__(self, boom_on_frame):
                    self.sent = []
                    self.boom_on_frame = boom_on_frame

                def __call__(self, content, buffers=None):
                    self.sent.append((content, buffers))
                    if (
                        content.get("type") == "progress"
                        and content.get("frame") == self.boom_on_frame
                    ):
                        raise RuntimeError("simulated send failure after checkpoint")

            dash.send = _RaiseAfterFrame(boom_on_frame=3)

            with mock.patch("acia.notebook.CHECKPOINT_INTERVAL", 2):
                summary = dash.batch_apply(directory=out)  # must not raise

            self.assertEqual(summary["failed_positions"], [0])
            self.assertEqual(summary["completed"], [0])

            final = RegistrationManifest.load(summary["path"]).records[0]
            # frames 0-3 (checkpointed before the simulated failure) survive
            # -- not wiped to an empty record.
            self.assertEqual(
                set(final.transforms) | set(final.failed_frames), {0, 1, 2, 3}
            )
            self.assertEqual(final.transforms[0].dx, 999.0)
            self.assertEqual(final.transforms[1].dx, 999.0)
            self.assertIn("simulated send failure after checkpoint", final.notes)

    def test_batch_apply_isolates_size_t_lookup_failure(self):
        """A resume/skip ``size_t`` lookup raising for one position must not
        abort the whole batch-apply run -- it falls through to the existing
        per-position failure path instead."""
        from acia.registration import FrameTransform
        from acia.registration_persistence import RegistrationRecord, save_registration

        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=4)
            out = Path(d) / "out"

            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )

            # A partial (not complete) prior record so batch_apply's
            # resume/skip logic actually performs a size_t lookup.
            sentinel = FrameTransform(dx=999.0, dy=999.0, theta=0.0)
            partial = RegistrationRecord(
                position=0,
                method="PhaseCorrelationHighpass",
                transforms={0: sentinel},
                reference_frame=0,
                failed_frames={},
            )
            manifest = RegistrationManifest(
                source=dash.manifest.source,
                records=[partial],
                method="PhaseCorrelationHighpass",
            )
            save_registration(manifest, out)

            class _BoomSource:
                @property
                def size_t(self):
                    raise RuntimeError("simulated size_t failure")

            real_position = dash._file.position
            dash._file.position = lambda i: (
                _BoomSource() if i == 0 else real_position(i)
            )

            dash.send = _Recorder()
            summary = dash.batch_apply(directory=out)  # must not raise

            self.assertEqual(summary["failed_positions"], [0])
            self.assertEqual(summary["completed"], [0])
            final = RegistrationManifest.load(summary["path"]).records[0]
            # the previously-checkpointed frame survives the size_t failure
            self.assertEqual(final.transforms[0].dx, 999.0)
            self.assertIn("simulated size_t failure", final.notes)

    def test_batch_apply_ignores_existing_record_from_different_method(self):
        """A prior record computed under a different method must not be
        treated as resume/skip data for the newly-selected method -- the
        position is processed from scratch instead of merging across
        methods."""
        from acia.registration import FrameTransform
        from acia.registration_persistence import RegistrationRecord, save_registration

        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=3)
            out = Path(d) / "out"

            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )

            # A *complete* record, but recorded under a different method.
            sentinel = FrameTransform(dx=999.0, dy=999.0, theta=0.0)
            old_method_record = RegistrationRecord(
                position=0,
                method="GradientECC",
                transforms={0: sentinel, 1: sentinel, 2: sentinel},
                reference_frame=0,
                failed_frames={},
            )
            manifest = RegistrationManifest(
                source=dash.manifest.source,
                records=[old_method_record],
                method="GradientECC",
            )
            save_registration(manifest, out)

            rec = _Recorder()
            dash.send = rec
            summary = dash.batch_apply(directory=out)

            # not skipped -- reprocessed from scratch under the new method
            self.assertEqual(summary["skipped"], [])
            self.assertEqual(summary["completed"], [0])
            progress = [c for c, _ in rec.sent if c.get("type") == "progress"]
            self.assertEqual([p["frame"] for p in progress], [0, 1, 2])

            final = RegistrationManifest.load(summary["path"]).records[0]
            self.assertEqual(final.method, "PhaseCorrelationHighpass")
            # sentinel values from the old method are gone -- real estimates
            self.assertNotEqual(final.transforms[0].dx, 999.0)

    def test_batch_apply_blocked_for_masked_template_without_mask(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(
                open_sequence(_tif_random(d)), method_name="MaskedTemplateCorrelation"
            )
            with self.assertRaises(ValueError):
                dash.batch_apply(directory=Path(d) / "out")

    def test_batch_apply_rejects_concurrent_runs(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            dash.batch_running = True
            with self.assertRaises(RuntimeError):
                dash.batch_apply(directory=Path(d) / "out")

    def test_batch_apply_positions_subset(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=2)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            dash.send = _Recorder()
            out = Path(d) / "out"
            summary = dash.batch_apply(directory=out, positions=[0])
            self.assertEqual(summary["completed"], [0])

    def test_batch_apply_sources_override_limits_frames(self):
        # A `sources` override (e.g. a lazily-sliced `position[:n]`) restricts
        # registration to just those frames instead of the whole position.
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=10)
            seqfile = open_sequence(path)
            dash = RegistrationDashboard(
                seqfile, method_name="PhaseCorrelationHighpass"
            )
            dash.send = _Recorder()
            out = Path(d) / "out"
            limited = seqfile.position(0)[:4]
            summary = dash.batch_apply(
                directory=out, positions=[0], sources={0: limited}
            )
            self.assertEqual(summary["completed"], [0])
            manifest = RegistrationManifest.load(summary["path"])
            record = manifest.records[0]
            self.assertEqual(
                set(record.transforms) | set(record.failed_frames), {0, 1, 2, 3}
            )

    def test_batch_apply_sources_override_resume_uses_sliced_length(self):
        # The "already complete" resume check must compare against the
        # override source's size_t, not the full position's -- otherwise a
        # 4-frame-limited position recorded as complete would look "partial"
        # against the full 10-frame position and be reprocessed forever.
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=10)
            seqfile = open_sequence(path)
            dash = RegistrationDashboard(
                seqfile, method_name="PhaseCorrelationHighpass"
            )
            dash.send = _Recorder()
            out = Path(d) / "out"
            limited = seqfile.position(0)[:4]
            dash.batch_apply(directory=out, positions=[0], sources={0: limited})

            dash2 = RegistrationDashboard(
                seqfile, method_name="PhaseCorrelationHighpass"
            )
            dash2.send = _Recorder()
            summary2 = dash2.batch_apply(
                directory=out, positions=[0], sources={0: limited}
            )
            self.assertEqual(summary2["skipped"], [0])
            self.assertEqual(summary2["completed"], [])


class TestManifestAndSave(unittest.TestCase):
    def test_manifest_empty_before_batch_apply(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            manifest = dash.manifest
            self.assertIsInstance(manifest, RegistrationManifest)
            self.assertEqual(manifest.records, [])
            self.assertEqual(manifest.method, "GradientECC")
            self.assertIn("pixel_size_um", manifest.source)

    def test_save_message_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=2)
            dash = RegistrationDashboard(
                open_sequence(path), method_name="PhaseCorrelationHighpass"
            )
            dash.send = _Recorder()
            out = Path(d) / "out"
            dash.batch_apply(directory=out)

            rec = _Recorder()
            dash.send = rec
            with tempfile.TemporaryDirectory() as d2:
                import os as _os

                cwd = _os.getcwd()
                try:
                    _os.chdir(d2)
                    dash._on_custom_msg(dash, {"type": "save"}, None)
                finally:
                    _os.chdir(cwd)
                content, buffers = rec.sent[0]
                self.assertEqual(content["type"], "saved")
                self.assertTrue(Path(content["path"]).exists())
                self.assertIsNone(buffers)


class TestImportSafety(unittest.TestCase):
    """``acia.notebook`` must stay importable without anywidget (optional dep)."""

    def test_import_safe_without_anywidget(self):
        real_import = builtins.__import__
        blocked = {"anywidget", "traitlets"}

        def fake_import(name, *args, **kwargs):
            if name in blocked or name.split(".")[0] in blocked:
                raise ImportError(f"simulated missing dependency: {name}")
            return real_import(name, *args, **kwargs)

        affected = [
            "acia.notebook",
            "acia.base",
            "acia.segm.local",
            "acia.segm.nd2_source",
        ]
        saved = {name: sys.modules.get(name) for name in affected}
        for name in affected:
            sys.modules.pop(name, None)

        try:
            builtins.__import__ = fake_import
            notebook = importlib.import_module("acia.notebook")
            self.assertFalse(notebook._HAS_ANYWIDGET)
            with self.assertRaisesRegex(ImportError, r"acia\[widget\]"):
                notebook.RegistrationDashboard(object())
        finally:
            builtins.__import__ = real_import
            for name, mod in saved.items():
                if mod is not None:
                    sys.modules[name] = mod
                else:
                    sys.modules.pop(name, None)
            importlib.reload(importlib.import_module("acia.notebook"))


def _tif_growing_colony(d, n_frames=24, size=256):
    """A TIFF whose ROI interiors fill with cells while the whole field drifts.

    A scaled-down twin of ``tests/test_registration.py``'s
    ``_growing_colony_series`` fixture, written to disk so it can be driven
    through the real ``batch_apply`` path the notebook uses.
    """
    rois = ((60, 50, 45, 130), (150, 50, 45, 130))
    rng = np.random.default_rng(11)
    base = rng.integers(0, 25, (size, size)).astype(np.float32)
    for x in (20, 102, 125, 235):
        cv2.line(base, (x, 0), (x, size), 220, 2)
    for y in (15, 240):
        cv2.line(base, (0, y), (size, y), 220, 2)
    for x, y, w, h in rois:
        cv2.rectangle(base, (x, y), (x + w, y + h), 255, 3)
    base = cv2.GaussianBlur(base, (0, 0), 1.0)

    frames = []
    for t in range(n_frames):
        frame = base.copy()
        cells = np.random.default_rng(100)
        for x, y, w, h in rois:
            for _ in range(int(t * 2.0)):
                cx = int(cells.integers(x + 5, x + w - 5))
                cy = int(cells.integers(y + 5, y + h - 5))
                cv2.circle(frame, (cx, cy), int(cells.integers(2, 5)), 200, -1)
        frame = cv2.GaussianBlur(frame, (0, 0), 1.0)
        frames.append(
            cv2.warpAffine(
                frame,
                np.float32([[1, 0, 0.10 * t], [0, 1, -0.07 * t]]),
                (size, size),
                borderMode=cv2.BORDER_REFLECT,
            ).astype(np.uint16)
        )

    path = Path(d) / "colony.tif"
    tifffile.imwrite(path, np.stack(frames))
    specs = [
        RotatedCropSpec(
            center=(x + w / 2.0, y + h / 2.0), size=(int(w), int(h)), angle=0.0
        )
        for x, y, w, h in rois
    ]
    return path, specs


class TestMethodKwargs(unittest.TestCase):
    """Constructor settings for the registration method, without monkeypatching."""

    def test_method_kwargs_reach_the_constructed_method(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(
                open_sequence(_tif_random(d)),
                method_kwargs={"early_stop_delta_px": 0.5, "min_confidence": 0.8},
            )
            method = dash._build_method("GradientECC")
        self.assertEqual(method.early_stop_delta_px, 0.5)
        self.assertEqual(method.min_confidence, 0.8)

    def test_per_call_overrides_win_over_instance_kwargs(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(
                open_sequence(_tif_random(d)), method_kwargs={"min_confidence": 0.8}
            )
            method = dash._build_method("GradientECC", min_confidence=0.5)
        self.assertEqual(method.min_confidence, 0.5)

    def test_no_kwargs_builds_the_method_with_its_own_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            method = dash._build_method("GradientECC")
        self.assertEqual(method.min_confidence, 0.9)
        self.assertIsNone(method.early_stop_delta_px)

    def test_mask_rect_still_required_by_the_method_that_needs_one(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            with self.assertRaises(ValueError):
                dash._build_method("MaskedTemplateCorrelation")

    def test_mask_rect_is_passed_to_any_method_accepting_one(self):
        rect = RotatedCropSpec(center=(10.0, 10.0), size=(8, 8), angle=0.0)
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
            method = dash._build_method("MaskedTemplateCorrelation", rect)
        self.assertEqual(method.mask_rect, rect)


class TestReferenceMode(unittest.TestCase):
    """The reference policy batch-apply registers under."""

    def test_defaults_to_reanchor(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(open_sequence(_tif_random(d)))
        self.assertEqual(dash._reference_mode, "reanchor")

    def test_rejects_an_unknown_mode(self):
        with tempfile.TemporaryDirectory() as d:
            source = open_sequence(_tif_random(d))
            with self.assertRaises(ValueError):
                RegistrationDashboard(source, reference_mode="sideways")

    def test_mode_is_recorded_on_every_record_and_in_method_params(self):
        with tempfile.TemporaryDirectory() as d:
            dash = RegistrationDashboard(
                open_sequence(_tif_translating(d)), reference_mode="fixed"
            )
            dash.send = _Recorder()
            dash.batch_apply(directory=d)
            manifest = RegistrationManifest.load(
                Path(d) / "registration_transforms.json"
            )
        self.assertEqual(manifest.method_params["reference_mode"], "fixed")
        for record in manifest.records:
            self.assertEqual(record.reference_mode, "fixed")


class TestGrowingColonyEndToEnd(unittest.TestCase):
    """The reported failure and its fix, at the layer the notebook calls."""

    def _run(
        self,
        directory,
        path,
        *,
        reference_mode,
        method_kwargs=None,
        low_confidence="reject",
    ):
        # These cases are about what a *rejected* frame does to the run, so
        # they opt into the gate; the default "keep" policy has its own case
        # below.
        dash = RegistrationDashboard(
            open_sequence(path),
            reference_mode=reference_mode,
            low_confidence=low_confidence,
        )
        dash.send = _Recorder()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowConfidenceWarning)
            dash.batch_apply(directory=directory, method_kwargs=method_kwargs)
        return dash.manifest.records[0]

    def test_fixed_reference_without_exclusion_fails_on_late_frames(self):
        with tempfile.TemporaryDirectory() as d:
            path, _specs = _tif_growing_colony(d)
            record = self._run(d, path, reference_mode="fixed")
        self.assertTrue(
            record.failed_frames,
            "expected a fixed reference on changing content to reject late frames",
        )

    def test_default_policy_stores_a_transform_for_every_frame(self):
        """The same fixed-reference run that leaves holes above leaves none
        under the default policy -- which is the whole point of it."""
        with tempfile.TemporaryDirectory() as d:
            path, _specs = _tif_growing_colony(d)
            n_frames = open_sequence(path).position(0).size_t
            record = self._run(d, path, reference_mode="fixed", low_confidence="keep")
        self.assertEqual(record.failed_frames, {})
        self.assertEqual(set(record.transforms), set(range(n_frames)))
        # The weak frames are still identifiable -- by their score, not by
        # their absence.
        self.assertLess(min(t.confidence for t in record.transforms.values()), 0.9)

    def test_the_policy_is_recorded_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            path, _specs = _tif_growing_colony(d)
            dash = RegistrationDashboard(open_sequence(path))
            self.assertEqual(dash.manifest.method_params["low_confidence"], "keep")
            dash = RegistrationDashboard(open_sequence(path), low_confidence="reject")
            self.assertEqual(dash.manifest.method_params["low_confidence"], "reject")

    def test_unknown_policy_is_rejected_at_construction(self):
        with tempfile.TemporaryDirectory() as d:
            path, _specs = _tif_growing_colony(d)
            with self.assertRaisesRegex(ValueError, "low_confidence"):
                RegistrationDashboard(open_sequence(path), low_confidence="warn")

    def test_excluding_the_roi_interiors_registers_every_frame(self):
        with tempfile.TemporaryDirectory() as d:
            path, specs = _tif_growing_colony(d)
            record = self._run(
                d,
                path,
                reference_mode="fixed",
                method_kwargs={"exclude_rects": specs, "exclude_shrink_px": 8.0},
            )
        self.assertEqual(record.failed_frames, {})

    def test_reanchoring_alone_also_clears_the_failures(self):
        with tempfile.TemporaryDirectory() as d:
            path, _specs = _tif_growing_colony(d)
            record = self._run(d, path, reference_mode="reanchor")
        self.assertEqual(record.failed_frames, {})
        # ...and records which frames needed a mid-sequence anchor.
        self.assertTrue(record.reference_frames)

    def test_confidences_are_persisted_for_auditing(self):
        with tempfile.TemporaryDirectory() as d:
            path, specs = _tif_growing_colony(d)
            self._run(
                d, path, reference_mode="fixed", method_kwargs={"exclude_rects": specs}
            )
            reloaded = RegistrationManifest.load(
                Path(d) / "registration_transforms.json"
            ).records[0]
        self.assertTrue(
            all(t.confidence is not None for t in reloaded.transforms.values())
        )


class TestReferenceModeResume(unittest.TestCase):
    """Progress is only reused where the policy that produced it still applies."""

    N_FRAMES = 4

    def _seed_manifest(self, directory, path, *, reference_mode, failed_frames=None):
        """Write a *complete* prior record under ``reference_mode``.

        Built by hand rather than by running batch-apply, so the record's
        contents (and specifically whether it holds any failures) are exactly
        what each case is about, independent of whether the small fixture
        happens to satisfy the method's own confidence gate.
        """
        failed_frames = failed_frames or {}
        n_transforms = self.N_FRAMES - len(failed_frames)
        record = RegistrationRecord(
            position=0,
            method="GradientECC",
            transforms={
                t: FrameTransform(dx=float(t), dy=0.0, theta=0.0)
                for t in range(n_transforms)
            },
            failed_frames=failed_frames,
            reference_mode=reference_mode,
        )
        manifest = RegistrationManifest(
            source={"path": str(path)}, records=[record], method="GradientECC"
        )
        manifest.save(Path(directory) / "registration_transforms.json")

    def _rerun(self, directory, path, mode):
        dash = RegistrationDashboard(open_sequence(path), reference_mode=mode)
        dash.send = _Recorder()
        return dash.batch_apply(directory=directory)

    def test_same_mode_record_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=self.N_FRAMES)
            self._seed_manifest(d, path, reference_mode="reanchor")
            summary = self._rerun(d, path, "reanchor")
        self.assertEqual(summary["skipped"], [0])

    def test_a_clean_fixed_record_is_reused_under_reanchor(self):
        """Changing the default policy must not silently discard a long run
        whose frames re-anchoring would have produced identically anyway --
        re-anchoring is a pure fallback, so it never fires on a frame that
        already succeeded against the reference."""
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=self.N_FRAMES)
            self._seed_manifest(d, path, reference_mode="fixed")
            summary = self._rerun(d, path, "reanchor")
        self.assertEqual(summary["skipped"], [0])

    def test_a_fixed_record_with_failures_is_not_reused_under_reanchor(self):
        """Those failures are exactly what re-anchoring exists to retry."""
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=self.N_FRAMES)
            self._seed_manifest(
                d, path, reference_mode="fixed", failed_frames={3: "boom"}
            )
            summary = self._rerun(d, path, "reanchor")
        self.assertEqual(summary["skipped"], [])
        self.assertEqual(summary["completed"], [0])

    def test_a_chained_record_is_never_reused_under_reanchor(self):
        """Chained transforms are measured against the previous frame, so they
        are not a valid prefix of a reference-anchored run at all."""
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=self.N_FRAMES)
            self._seed_manifest(d, path, reference_mode="chained")
            summary = self._rerun(d, path, "reanchor")
        self.assertEqual(summary["skipped"], [])
        self.assertEqual(summary["completed"], [0])

    def test_a_reanchor_record_is_not_reused_under_fixed(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif_translating(d, t=self.N_FRAMES)
            self._seed_manifest(d, path, reference_mode="reanchor")
            summary = self._rerun(d, path, "fixed")
        self.assertEqual(summary["skipped"], [])


if __name__ == "__main__":
    unittest.main()
