"""Tests for recording a stage as it happens, and for showing what is recorded.

``StageContext`` used to write one manifest entry at the very end, from
``record()``. A stage that died before that recorded nothing -- and under
``scale(exist_skip=True)`` a half-finished stage is skipped on every later run,
so the run that died is exactly the one whose record was wanted.

Naming the stage up front (``for_image(..., stage=...)``) opens the entry
immediately and every ``log_*`` call persists straight away. These tests pin the
two properties that makes worth having: what is logged is on disk *before* the
stage finishes, and an unfinished stage is distinguishable from a finished one.
"""

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from acia.analysis import (  # noqa: E402
    MANIFEST_NAME,
    StageContext,
    read_manifest,
    stage_table,
)


class _StageFolder(unittest.TestCase):
    """A temp folder holding one population's source and output."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.image_id = self.root / "pos001_roi002.tiff"
        self.image_id.write_bytes(b"not really a tiff")
        self.output = self.root / "output"

    def tearDown(self):
        self._tmp.cleanup()

    def context(self, stage="01_Segment", **kwargs):
        return StageContext.for_image(self.image_id, self.output, stage=stage, **kwargs)

    def entry(self, stage="01_Segment"):
        return read_manifest(self.output)["stages"][stage]


class TestRecordingHappensImmediately(_StageFolder):
    """What is logged must be on disk before the stage ends."""

    def test_building_a_context_writes_nothing(self):
        """A folder that only *built* a context has not run a stage."""
        self.context()
        self.assertEqual(read_manifest(self.output), {})

    def test_params_and_metrics_are_readable_before_finish(self):
        ctx = self.context()
        ctx.log_params(backend="omnipose", diameter_px=12.0)
        ctx.log_metrics(n_cells=1843)

        entry = self.entry()
        self.assertEqual(entry["params"], {"backend": "omnipose", "diameter_px": 12.0})
        self.assertEqual(entry["metrics"], {"n_cells": 1843})

    def test_an_unfinished_stage_is_marked_running(self):
        """The distinction a crashed stage depends on: running is not ok."""
        ctx = self.context()
        ctx.log_params(backend="omnipose")

        entry = self.entry()
        self.assertEqual(entry["status"], "running")
        self.assertNotIn("finished_at", entry)

    def test_finish_marks_it_ok_and_stamps_the_end(self):
        ctx = self.context()
        ctx.log_metrics(n_cells=3)
        ctx.finish()

        entry = self.entry()
        self.assertEqual(entry["status"], "ok")
        self.assertIn("finished_at", entry)
        self.assertIn("duration_s", entry)

    def test_values_that_are_not_json_are_stored_as_text(self):
        """A pint quantity or a Path must not be able to break the manifest."""
        from acia import Q_

        ctx = self.context()
        ctx.log_params(pixel_size=Q_(0.065, "micrometer"), where=Path("/tmp/x"))

        entry = self.entry()
        self.assertIn("micrometer", entry["params"]["pixel_size"])
        self.assertEqual(entry["params"]["where"], "/tmp/x")

    def test_logging_without_a_stage_name_is_refused(self):
        """The error has to say how to fix it, not just that it failed."""
        ctx = StageContext.for_image(self.image_id, self.output)
        with self.assertRaises(ValueError) as caught:
            ctx.log_params(backend="omnipose")
        self.assertIn("stage=", str(caught.exception))


class TestDeclaredPaths(_StageFolder):
    """input_path / output_path state intent next to what was observed."""

    def test_output_path_creates_the_folder_but_not_the_file(self):
        ctx = self.context()
        target = ctx.output_path("figures/size.png")

        self.assertTrue(target.parent.is_dir())
        self.assertFalse(target.exists())

    def test_declared_output_that_is_written_is_not_missing(self):
        """The collapse fix: figures/size.png is observed as the artifact figures/."""
        ctx = self.context()
        ctx.output_path("figures/size.png").write_bytes(b"png")
        ctx.finish()

        self.assertNotIn("missing", self.entry()["io"])

    def test_declared_output_never_written_is_reported(self):
        ctx = self.context()
        ctx.output_path("typo.csv")
        ctx.finish()

        self.assertEqual(self.entry()["io"]["missing"], ["typo.csv"])

    def test_input_path_records_the_declaration_and_the_producer(self):
        first = self.context("01_Segment")
        first.output_path("segmentation.npz").write_bytes(b"seg")
        first.finish()

        second = self.context("02_Track")
        second.input_path("segmentation.npz")
        second.finish()

        entry = self.entry("02_Track")
        self.assertEqual(entry["declared_inputs"], ["segmentation.npz"])
        produced = {
            item["path"]: item.get("produced_by") for item in entry["io"]["inputs"]
        }
        self.assertEqual(produced["segmentation.npz"], "01_Segment")

    def test_input_path_keeps_the_actionable_error(self):
        ctx = self.context("02_Track")
        with self.assertRaises(FileNotFoundError) as caught:
            ctx.input_path("segmentation.npz", "01_Segment.ipynb")
        self.assertIn("01_Segment.ipynb", str(caught.exception))

    def test_writing_outside_the_output_folder_warns(self):
        """A chain hands results over through one folder; leaving it hides them."""
        ctx = self.context()
        with self.assertWarns(UserWarning):
            ctx.output_path("../elsewhere.csv")


class TestFigures(_StageFolder):
    """A figure should carry its caption, not land in the folder anonymously."""

    def _figure(self):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([1, 2, 3])
        return fig

    def test_log_figure_writes_and_records_it(self):
        ctx = self.context()
        target = ctx.log_figure(self._figure(), "size_dist", caption="cell areas")

        self.assertTrue(target.exists())
        figures = self.entry()["figures"]
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0]["path"], "figures/size_dist.png")
        self.assertEqual(figures[0]["caption"], "cell areas")

    def test_relogging_the_same_name_replaces_it(self):
        ctx = self.context()
        ctx.log_figure(self._figure(), "size_dist", caption="first")
        ctx.log_figure(self._figure(), "size_dist", caption="second")

        figures = self.entry()["figures"]
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0]["caption"], "second")

    def test_thumbnails_stay_out_of_the_manifest(self):
        """They are for display; base64 in the manifest would bloat every read."""
        ctx = self.context()
        ctx.log_figure(self._figure(), "size_dist")

        self.assertNotIn("base64", (self.output / MANIFEST_NAME).read_text())

    def test_figures_do_not_become_recorded_reads(self):
        """Re-reading a figure to display it would forge a dependency edge.

        The thumbnail is captured from the live figure precisely so that a later
        stage rendering its own view never *reads* an earlier stage's figures --
        which would make stage_graph() grow an edge that no analysis implies.
        """
        first = self.context("01_Segment")
        first.log_figure(self._figure(), "size_dist")
        first.finish()

        second = self.context("02_Track")
        second._repr_html_()
        second.finish()

        read = {item["path"] for item in self.entry("02_Track")["io"]["inputs"]}
        self.assertNotIn("figures/", read)


class TestCalibrationIsRecordedAndChecked(_StageFolder):
    """The source stays authoritative; the record exists to catch disagreement."""

    class _Source:
        def __init__(self, pixel_size, frame_interval, origin="ome-xml"):
            from acia import Q_

            self.pixel_size = Q_(pixel_size, "micrometer")
            self._frame_interval = Q_(frame_interval, "second")
            self.calibration_source = origin

    def test_resolved_values_are_recorded(self):
        ctx = self.context()
        ctx.log_calibration(self._Source(0.065, 300.0))

        calibration = self.entry()["calibration"]
        self.assertAlmostEqual(calibration["pixel_size_um"], 0.065)
        self.assertAlmostEqual(calibration["frame_interval_s"], 300.0)
        self.assertEqual(calibration["origin"], "ome-xml")

    def test_agreement_between_stages_is_silent(self):
        first = self.context("01_Segment")
        first.log_calibration(self._Source(0.065, 300.0))
        first.finish()

        second = self.context("02_Track")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            second.log_calibration(self._Source(0.065, 300.0))

    def test_disagreement_between_stages_warns(self):
        """Two stages on one movie with different calibration are incomparable."""
        first = self.context("01_Segment")
        first.log_calibration(self._Source(0.065, 300.0))
        first.finish()

        second = self.context("02_Track")
        with self.assertWarns(UserWarning) as caught:
            second.log_calibration(self._Source(0.065, 600.0))
        self.assertIn("frame interval", str(caught.warning))

    def test_read_back_returns_what_was_recorded(self):
        ctx = self.context()
        ctx.log_calibration(self._Source(0.065, 300.0))
        ctx.finish()

        self.assertAlmostEqual(ctx.calibration("01_Segment")["pixel_size_um"], 0.065)


class TestBackCompat(_StageFolder):
    """record() keeps writing exactly what it always wrote."""

    def test_record_writes_extras_flat(self):
        """ctx.stage('X')['pixel_size'] and stage_table columns both read these."""
        ctx = StageContext.for_image(self.image_id, self.output)
        ctx.record("01_Segment", artifacts=[], n_detections=17, pixel_size="0.065 um")

        entry = self.entry()
        self.assertEqual(entry["n_detections"], 17)
        self.assertEqual(entry["pixel_size"], "0.065 um")
        self.assertEqual(entry["status"], "ok")

    def test_record_can_take_the_stage_name_from_the_context(self):
        ctx = self.context("02_Track")
        ctx.record(n_tracklets=412)

        self.assertEqual(self.entry("02_Track")["n_tracklets"], 412)

    def test_record_without_any_stage_name_is_refused(self):
        ctx = StageContext.for_image(self.image_id, self.output)
        with self.assertRaises(ValueError):
            ctx.record()

    def test_a_context_without_a_stage_writes_nothing_until_record(self):
        """The opt-in rule: a legacy notebook behaves exactly as it did."""
        ctx = StageContext.for_image(self.image_id, self.output)
        ctx.path("x.npz").write_bytes(b"x")
        self.assertEqual(read_manifest(self.output), {})
        ctx.record("01_Segment")
        self.assertIn("01_Segment", read_manifest(self.output)["stages"])


class TestStageTable(_StageFolder):
    """Logged values have to tabulate like recorded ones."""

    def test_params_and_metrics_flatten_into_columns(self):
        ctx = self.context()
        ctx.log_params(backend="omnipose")
        ctx.log_metrics(n_cells=1843)
        ctx.log_calibration(TestCalibrationIsRecordedAndChecked._Source(0.065, 300.0))
        ctx.finish()

        table = stage_table(self.root, pattern="output")
        row = table.iloc[0]
        self.assertEqual(row["backend"], "omnipose")
        self.assertEqual(row["n_cells"], 1843)
        self.assertAlmostEqual(row["pixel_size_um"], 0.065)
        self.assertEqual(row["status"], "ok")

    def test_structured_blocks_do_not_become_object_columns(self):
        ctx = self.context()
        ctx.log_params(backend="omnipose")
        ctx.finish()

        table = stage_table(self.root, pattern="output")
        for absent in ("params", "metrics", "figures", "calibration", "io", "schema"):
            self.assertNotIn(absent, table.columns)

    def test_a_failed_stage_reports_why_not_only_that(self):
        """Across a fan-out this separates one ROI's OOM from a broken chain."""
        ctx = self.context()
        ctx.log_params(backend="omnipose")
        ctx._note_cell(type("R", (), {"error_in_exec": RuntimeError("CUDA OOM")})())

        row = stage_table(self.root, pattern="output").iloc[0]

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error_type"], "RuntimeError")
        self.assertIn("CUDA OOM", row["error_message"])

    def test_a_stage_that_succeeded_has_no_error(self):
        ctx = self.context()
        ctx.log_metrics(n_cells=3)
        ctx.finish()

        row = stage_table(self.root, pattern="output").iloc[0]

        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["error_type"])

    def test_a_setting_cannot_shadow_a_derived_column(self):
        """A param called duration_s must not overwrite the real duration."""
        ctx = self.context()
        ctx.log_params(duration_s="not a duration")
        ctx.finish()

        table = stage_table(self.root, pattern="output")
        self.assertIsInstance(table.iloc[0]["duration_s"], float)


class TestHtmlView(_StageFolder):
    """The view must be informative, safe, and never the thing that breaks."""

    def test_it_shows_the_population_stage_and_files(self):
        ctx = self.context()
        ctx.output_path("segmentation.npz").write_bytes(b"seg")
        ctx.log_params(backend="omnipose")

        rendered = ctx._repr_html_()
        self.assertIn("pos001_roi002", rendered)
        self.assertIn("01_Segment", rendered)
        self.assertIn("segmentation.npz", rendered)
        self.assertIn("omnipose", rendered)

    def test_it_renders_for_an_empty_folder(self):
        ctx = self.context()
        self.assertIsInstance(ctx._repr_html_(), str)

    def test_a_corrupt_manifest_degrades_to_text_instead_of_raising(self):
        """A raising repr prints a traceback that reads as a failed analysis."""
        ctx = self.context()
        (self.output / MANIFEST_NAME).write_text("{not json")

        rendered = ctx._repr_html_()
        self.assertIn("pos001_roi002", rendered)
        self.assertIn("<pre>", rendered)

    def test_recorded_values_are_escaped(self):
        ctx = self.context()
        ctx.log_params(note="<script>alert(1)</script>")

        rendered = ctx._repr_html_()
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)


class TestSurvivesAKilledProcess(_StageFolder):
    """The headline claim: a run that dies keeps what it logged."""

    SCRIPT = """
import os, signal, sys
from acia.analysis import StageContext
ctx = StageContext.for_image(sys.argv[1], sys.argv[2], stage="01_Segment")
ctx.log_params(backend="omnipose")
ctx.log_metrics(n_cells=1843)
{ending}
"""

    def _run(self, ending):
        script = self.SCRIPT.format(ending=ending)
        subprocess.run(
            [sys.executable, "-c", script, str(self.image_id), str(self.output)],
            check=False,
            capture_output=True,
        )

    def test_params_survive_a_sigkill(self):
        """SIGKILL runs no exit hook, so only the as-you-go writes can save this."""
        self._run("os.kill(os.getpid(), signal.SIGKILL)")

        manifest = json.loads((self.output / MANIFEST_NAME).read_text())
        entry = manifest["stages"]["01_Segment"]
        self.assertEqual(entry["params"], {"backend": "omnipose"})
        self.assertEqual(entry["metrics"], {"n_cells": 1843})
        self.assertEqual(entry["status"], "running")
        self.assertNotIn("finished_at", entry)

    def test_the_manifest_stays_valid_json_through_a_sigkill(self):
        """Atomic writes: a truncated manifest would poison the whole folder."""
        self._run("os.kill(os.getpid(), signal.SIGKILL)")

        json.loads((self.output / MANIFEST_NAME).read_text())
        StageContext.for_image(output_folder=self.output)  # must not raise

    def test_the_exit_hook_collects_io_when_the_process_ends_normally(self):
        """No IPython in that interpreter, so this isolates the atexit path."""
        self._run(
            "open(os.path.join(sys.argv[2], 'segmentation.npz'), 'wb').write(b'x')"
        )

        entry = self.entry()
        written = {item["path"] for item in (entry.get("io") or {}).get("outputs", [])}
        self.assertIn("segmentation.npz", written)

    def test_no_temp_file_is_left_behind_or_recorded(self):
        self._run(
            "open(os.path.join(sys.argv[2], 'segmentation.npz'), 'wb').write(b'x')"
        )

        leftovers = [p.name for p in self.output.iterdir() if ".tmp" in p.name]
        self.assertEqual(leftovers, [])
        recorded = json.dumps(read_manifest(self.output))
        self.assertNotIn(".tmp", recorded)


class TestCellBoundaryFlush(_StageFolder):
    """Under IPython the cell boundary is what closes a stage, not exit."""

    def _shell(self):
        from IPython.core.interactiveshell import InteractiveShell

        return InteractiveShell.instance()

    def test_a_raising_cell_marks_the_stage_failed(self):
        shell = self._shell()
        ctx = self.context()
        ctx._note_cell(type("R", (), {"error_in_exec": ValueError("boom")})())

        entry = self.entry()
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["error"]["type"], "ValueError")
        self.assertIn("boom", entry["error"]["message"])
        del shell

    def test_a_later_good_cell_clears_the_failure(self):
        ctx = self.context()
        ctx._note_cell(type("R", (), {"error_in_exec": ValueError("boom")})())
        ctx._note_cell(type("R", (), {"error_in_exec": None})())

        entry = self.entry()
        self.assertEqual(entry["status"], "running")
        self.assertNotIn("error", entry)


class TestProducerHintIsChecked(_StageFolder):
    """A stated producer that is wrong reads as a checked dependency and is not one.

    ``produced_by`` names a **stage**, matched exactly. Accepting the notebook file
    name as well would be worse than strict: a stage called ``Segment`` living in
    ``01_Segment.ipynb`` would then match under ``scale()`` -- which records the
    notebook -- and not match interactively, so a wrong hint would pass in one
    place and warn in the other.
    """

    def _chain(self):
        first = self.context("Segment")
        first.output_path("segmentation.npz").write_bytes(b"seg")
        first.finish()

    def _consume(self, produced_by, stage="Track"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.context(stage).input_path("segmentation.npz", produced_by)
        # "produced" matches both messages -- "produced by stage" and "produced_by
        # names a notebook". Matching the spaced form only would let the notebook
        # warning slip past every assertion here.
        return [str(w.message) for w in caught if "produced" in str(w.message)]

    def test_a_correct_stage_name_is_silent(self):
        self._chain()
        self.assertEqual(self._consume("Segment"), [])

    def test_the_notebook_file_name_is_refused_with_the_fix(self):
        """Naming the notebook is the easy mistake, so say what to write instead."""
        self._chain()

        warned = self._consume("01_Segment.ipynb")

        self.assertEqual(len(warned), 1)
        self.assertIn("names a notebook", warned[0])
        self.assertIn("stage name", warned[0])
        self.assertIn("Segment", warned[0])  # what is actually recorded here

    def test_naming_a_stage_that_never_ran_here_warns(self):
        self._chain()

        warned = self._consume("Nonexistent")

        self.assertEqual(len(warned), 1)
        self.assertIn("no such stage has run", warned[0])
        self.assertIn("Segment", warned[0])  # what did run, so the typo is visible

    def test_naming_a_stage_that_wrote_something_else_warns(self):
        self._chain()
        other = self.context("Track")
        other.output_path("tracking.npz").write_bytes(b"t")
        other.finish()

        warned = self._consume("Track", stage="GrowthRate")

        self.assertEqual(len(warned), 1)
        self.assertIn("written by stage 'Segment'", warned[0])

    def test_an_empty_manifest_has_nothing_to_check_against(self):
        """A hand-placed file in a fresh folder is the caller's business."""
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "segmentation.npz").write_bytes(b"hand-placed")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StageContext.for_image(self.image_id, self.output).require(
                "segmentation.npz", "Segment"
            )

        self.assertEqual([w for w in caught if "produced" in str(w.message)], [])

    def test_an_unattributed_artifact_is_not_contradicted(self):
        """Capture can fail; a missing attribution is not evidence of a wrong hint."""
        self._chain()
        # strip the recorded I/O, as a run whose capture failed would leave it
        manifest = read_manifest(self.output)
        manifest["stages"]["Segment"].pop("io", None)
        (self.output / MANIFEST_NAME).write_text(json.dumps(manifest))

        self.assertEqual(self._consume("Segment"), [])

    def test_no_hint_means_no_check(self):
        self._chain()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.context("Track").input_path("segmentation.npz")
        self.assertEqual([w for w in caught if "produced" in str(w.message)], [])

    def test_a_missing_artifact_still_raises_rather_than_warns(self):
        self._chain()
        with self.assertRaises(FileNotFoundError):
            self.context("Track").input_path("absent.npz", "Segment")


class TestRerunningARecordedStage(_StageFolder):
    """Re-running is allowed, but it must not leave two runs mixed into one entry."""

    def _first_run(self):
        first = self.context("Segment")
        first.log_params(backend="omnipose")
        first.log_metrics(n_cells=100)
        first.output_path("segmentation.npz").write_bytes(b"seg")
        first.finish()

        downstream = self.context("Track")
        downstream.input_path("segmentation.npz")
        downstream.finish()

    def test_it_warns_and_names_what_goes_stale(self):
        """The part not visible from this notebook: Track now describes old results."""
        self._first_run()

        with self.assertWarns(UserWarning) as caught:
            self.context("Segment")

        message = str(caught.warning)
        self.assertIn("already ran", message)
        self.assertIn("Track", message)

    def test_the_new_run_does_not_inherit_the_old_one(self):
        """Otherwise a re-run that dies half-way reads as one complete run."""
        self._first_run()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = self.context("Segment")
        again.log_params(backend="cellpose_sam")

        entry = self.entry("Segment")
        self.assertEqual(entry["status"], "running")
        self.assertEqual(entry["params"], {"backend": "cellpose_sam"})
        self.assertNotIn("metrics", entry)  # the old run's 100 cells are gone
        self.assertNotIn("io", entry)  # ... and so are its outputs
        self.assertNotIn("finished_at", entry)

    def test_an_unfinished_previous_run_warns_differently(self):
        """Nothing consumed it, so there is no staleness to claim."""
        crashed = self.context("Segment")
        crashed.log_params(backend="omnipose")  # never finished

        with self.assertWarns(UserWarning) as caught:
            self.context("Segment")

        message = str(caught.warning)
        self.assertIn("did not finish", message)
        self.assertNotIn("out of date", message)

    def test_a_stage_that_never_ran_here_is_silent(self):
        self._first_run()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.context("GrowthRate")

        self.assertEqual([w for w in caught if "already ran" in str(w.message)], [])

    def test_the_legacy_record_path_is_unaffected(self):
        """A notebook that never passes stage= behaves exactly as it did."""
        ctx = StageContext.for_image(self.image_id, self.output)
        ctx.record("Segment", n_cells=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            again = StageContext.for_image(self.image_id, self.output)
            again.record("Segment", n_cells=2)

        self.assertEqual([w for w in caught if "already ran" in str(w.message)], [])
        self.assertEqual(self.entry("Segment")["n_cells"], 2)


class TestClearDoesNotTakeAnotherStagesWork(_StageFolder):
    """figures/ is shared, so clearing one stage must not rmtree it."""

    def test_a_shared_directory_artifact_is_left_alone(self):
        first = self.context("01_Segment")
        first.output_path("figures/seg.png").write_bytes(b"png")
        first.finish()

        second = self.context("02_Track")
        second.output_path("figures/track.png").write_bytes(b"png")
        second.finish()

        second.clear("02_Track")

        self.assertTrue((self.output / "figures" / "seg.png").exists())


if __name__ == "__main__":
    unittest.main()
