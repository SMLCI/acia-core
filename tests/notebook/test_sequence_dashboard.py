"""Python-surface tests for :class:`acia.notebook.SequenceDashboard`.

anywidget must be installed (the ``widget`` extra). The ESM JavaScript is NOT
exercised here — it is validated by a real notebook run / the Playwright suite in
the devcontainer. These tests cover the traits, the ``on_msg`` handlers (lazy
thumbnail/frame bytes + point-fit), the manifest build, and ``save``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import ipywidgets
import numpy as np
import pytest
import tifffile

pytest.importorskip("anywidget")

from acia.notebook import SequenceDashboard  # noqa: E402
from acia.segm.open import open_sequence  # noqa: E402
from acia.selection import SelectionManifest  # noqa: E402


class _Recorder:
    """Captures ``widget.send`` calls (content, buffers)."""

    def __init__(self):
        self.sent = []

    def __call__(self, content, buffers=None):
        self.sent.append((content, buffers))


def _tif(d, t=3, h=8, w=10):
    path = Path(d) / "stack.tif"
    tifffile.imwrite(path, (np.random.rand(t, h, w) * 1000).astype(np.uint16))
    return path


_AXIS_ALIGNED = [[0, 0], [10, 0], [10, 8], [0, 8]]  # 4 corners -> angle ~0


class TestConstruct(unittest.TestCase):
    def test_traits_populated(self):
        with tempfile.TemporaryDirectory() as d:
            path = _tif(d)
            dash = SequenceDashboard(open_sequence(path))
            self.assertEqual(dash.metadata["num_positions"], 1)
            self.assertEqual(dash.metadata["num_timepoints"], 3)
            self.assertEqual(len(dash.positions), 1)
            self.assertEqual(dash.selections, [])
            self.assertEqual(dash.roi_mode, "single")
            self.assertTrue(dash.auto_save)

    def test_metadata_carries_source_identity(self):
        """The header's read-only Source field renders ``metadata['path']``."""
        with tempfile.TemporaryDirectory() as d:
            path = _tif(d)
            dash = SequenceDashboard(open_sequence(path))
            self.assertEqual(dash.metadata["path"], str(path))
            self.assertTrue(dash.metadata["format"])

    def test_metadata_path_falls_back_for_in_memory_source(self):
        """Synthetic/in-memory sources are only SequenceFile-*compatible*."""
        with tempfile.TemporaryDirectory() as d:
            seqfile = open_sequence(_tif(d))

            class _NoPath:
                metadata = seqfile.metadata
                positions = seqfile.positions

            dash = SequenceDashboard(_NoPath())
            self.assertEqual(dash.metadata["path"], "")
            self.assertEqual(dash.metadata["format"], "")

    def test_construct_from_path_and_mode(self):
        with tempfile.TemporaryDirectory() as d:
            dash = SequenceDashboard(str(_tif(d)), roi_mode="multi")
            self.assertEqual(dash.roi_mode, "multi")

    def test_repr_html(self):
        with tempfile.TemporaryDirectory() as d:
            dash = SequenceDashboard(open_sequence(_tif(d)))
            self.assertIn("SequenceDashboard", dash._repr_html_())

    def test_does_not_shadow_widget_dispatcher(self):
        """Regression test: a custom on_msg callback must not be named
        ``_handle_msg`` -- that name collides with the base class's own
        comm-message dispatcher (registered via ``self.comm.on_msg(self._handle_msg)``
        in ``ipywidgets.Widget``), silently breaking every custom message
        (frame/thumb/fit/save) the widget sends or receives.
        """
        self.assertIs(SequenceDashboard._handle_msg, ipywidgets.Widget._handle_msg)


class TestMessages(unittest.TestCase):
    def _dash(self, d):
        dash = SequenceDashboard(open_sequence(_tif(d)))
        rec = _Recorder()
        dash.send = rec  # shadow the widget's send
        return dash, rec

    def test_thumb_sends_png(self):
        with tempfile.TemporaryDirectory() as d:
            dash, rec = self._dash(d)
            dash._on_custom_msg(dash, {"type": "thumb", "pos": 0}, None)
            self.assertEqual(len(rec.sent), 1)
            content, buffers = rec.sent[0]
            self.assertEqual(content, {"type": "thumb", "pos": 0})
            self.assertTrue(buffers[0].startswith(b"\x89PNG"))

    def test_thumb_error_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            dash, rec = self._dash(d)
            dash._on_custom_msg(
                dash, {"type": "thumb", "pos": 5}, None
            )  # position 5 doesn't exist (1-position tiff)
            self.assertEqual(len(rec.sent), 1)
            content, buffers = rec.sent[0]
            self.assertEqual(content["type"], "error")
            self.assertEqual(content["kind"], "thumb")
            self.assertEqual(content["pos"], 5)
            self.assertIsNone(buffers)

    def test_frame_sends_png(self):
        with tempfile.TemporaryDirectory() as d:
            dash, rec = self._dash(d)
            dash._on_custom_msg(dash, {"type": "frame", "pos": 0, "t": 1}, None)
            content, buffers = rec.sent[0]
            self.assertEqual(content["type"], "frame")
            self.assertEqual(content["t"], 1)
            self.assertTrue(buffers[0].startswith(b"\x89PNG"))

    def test_fit_returns_spec(self):
        with tempfile.TemporaryDirectory() as d:
            dash, rec = self._dash(d)
            dash._on_custom_msg(dash, {"type": "fit", "points": _AXIS_ALIGNED}, None)
            content, _ = rec.sent[0]
            self.assertEqual(content["type"], "fit")
            self.assertIn("roi", content)
            self.assertIn("center", content["roi"])

    def test_fit_too_few_points_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            dash, rec = self._dash(d)
            # the ESM always sends exactly 4 points, so this should never happen
            # in practice -- but if it did, it must be reported, not swallowed
            # (the old silent-ignore behavior left the UI stuck with no
            # feedback, which is exactly the bug this replaces).
            dash._on_custom_msg(dash, {"type": "fit", "points": [[0, 0], [1, 1]]}, None)
            self.assertEqual(len(rec.sent), 1)
            content, buffers = rec.sent[0]
            self.assertEqual(content["type"], "error")
            self.assertEqual(content["kind"], "fit")
            self.assertIn("message", content)
            self.assertIsNone(buffers)

    def test_frame_error_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            dash, rec = self._dash(d)
            # out-of-range timepoint: the read itself should fail, and that
            # failure must be reported to the frontend, not raised into the
            # comm dispatcher (where the ESM would never learn about it and
            # the pane would stay in its "loading" state forever).
            dash._on_custom_msg(dash, {"type": "frame", "pos": 0, "t": 999}, None)
            self.assertEqual(len(rec.sent), 1)
            content, buffers = rec.sent[0]
            self.assertEqual(content["type"], "error")
            self.assertEqual(content["kind"], "frame")
            self.assertEqual(content["pos"], 0)
            self.assertIn("message", content)
            self.assertIsNone(buffers)


class TestManifestAndSave(unittest.TestCase):
    _SEL = {
        "id": "s0",
        "position": 0,
        "label": "colony A",
        "ci": 0,
        "roi": {"center": [5.0, 4.0], "size": [6, 4], "angle": 0.0},
    }

    def test_manifest_from_selections(self):
        with tempfile.TemporaryDirectory() as d:
            dash = SequenceDashboard(open_sequence(_tif(d)))
            dash.selections = [self._SEL]
            m = dash.manifest
            self.assertIsInstance(m, SelectionManifest)
            self.assertEqual(len(m.selections), 1)
            self.assertEqual(m.selections[0].spec.size, (6, 4))
            self.assertEqual(m.source["format"], "tiff")
            self.assertIn("pixel_size_um", m.source)

    def test_save_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            dash = SequenceDashboard(open_sequence(_tif(d)))
            dash.selections = [self._SEL]
            out = Path(d) / "curation"
            path = dash.save(out)
            self.assertTrue(Path(path).exists())
            reloaded = SelectionManifest.load(path)
            self.assertEqual(reloaded.selections[0].label, "colony A")

    def test_save_dir_is_the_default_output(self):
        """Auto-save calls ``save()`` with no argument -- it must land in save_dir."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "curation"
            dash = SequenceDashboard(open_sequence(_tif(d)), save_dir=out)
            dash.selections = [self._SEL]
            self.assertEqual(Path(dash.save()), out / "selection.json")

    def test_explicit_directory_beats_save_dir(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "curation"
            other = Path(d) / "elsewhere"
            dash = SequenceDashboard(open_sequence(_tif(d)), save_dir=out)
            dash.selections = [self._SEL]
            self.assertEqual(Path(dash.save(other)), other / "selection.json")
            self.assertFalse((out / "selection.json").exists())


class TestResume(unittest.TestCase):
    _SELS = [
        {
            "id": 1,
            "position": 0,
            "label": "colony A",
            "ci": 0,
            "roi": {"center": [5.0, 4.0], "size": [6, 4], "angle": 0.0},
        },
        {
            "id": 2,
            "position": 0,
            "label": "colony B",
            "ci": 1,
            "roi": {"center": [3.0, 2.0], "size": [4, 2], "angle": 15.0},
        },
    ]

    def _saved(self, d):
        tif_path = _tif(d)
        dash = SequenceDashboard(open_sequence(tif_path), roi_mode="multi")
        dash.selections = self._SELS
        out = Path(d) / "curation"
        dash.save(out)
        return tif_path, out

    def test_resume_restores_selections_and_mode(self):
        with tempfile.TemporaryDirectory() as d:
            _tif_path, out = self._saved(d)
            resumed = SequenceDashboard.resume(
                out
            )  # directory containing selection.json
            self.assertEqual(resumed.roi_mode, "multi")
            self.assertEqual(len(resumed.selections), 2)
            labels = [s["label"] for s in resumed.selections]
            self.assertEqual(labels, ["colony A", "colony B"])
            # ids re-numbered sequentially, per-position ci assigned in order
            self.assertEqual([s["id"] for s in resumed.selections], [1, 2])
            self.assertEqual([s["ci"] for s in resumed.selections], [0, 1])
            self.assertEqual(resumed.selections[1]["roi"]["angle"], 15.0)

    def test_resume_from_explicit_file_path(self):
        with tempfile.TemporaryDirectory() as d:
            _tif_path, out = self._saved(d)
            resumed = SequenceDashboard.resume(out / "selection.json")
            self.assertEqual(len(resumed.selections), 2)

    def test_resume_keeps_saving_where_it_resumed_from(self):
        with tempfile.TemporaryDirectory() as d:
            _tif_path, out = self._saved(d)
            self.assertEqual(
                Path(SequenceDashboard.resume(out).save()), out / "selection.json"
            )
            self.assertEqual(
                Path(SequenceDashboard.resume(out / "selection.json").save()),
                out / "selection.json",
            )

    def test_resume_roi_mode_override(self):
        with tempfile.TemporaryDirectory() as d:
            _tif_path, out = self._saved(d)
            resumed = SequenceDashboard.resume(out, roi_mode="single")
            self.assertEqual(resumed.roi_mode, "single")

    def test_resume_with_explicit_source(self):
        with tempfile.TemporaryDirectory() as d:
            tif_path, out = self._saved(d)
            resumed = SequenceDashboard.resume(out, source=open_sequence(tif_path))
            self.assertEqual(len(resumed.selections), 2)

    def test_resume_out_of_range_position_raises(self):
        with tempfile.TemporaryDirectory() as d:
            tif_path = _tif(d)
            dash = SequenceDashboard(open_sequence(tif_path))
            dash.selections = [{**self._SELS[0], "position": 5}]
            out = Path(d) / "curation"
            dash.save(out)
            with self.assertRaises(ValueError):
                SequenceDashboard.resume(out)


if __name__ == "__main__":
    unittest.main()
