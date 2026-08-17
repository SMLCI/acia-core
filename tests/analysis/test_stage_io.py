"""Tests for the recorded stage I/O (acia.analysis._stage_io + StageContext).

The point of the feature is that a stage's record reflects what the *filesystem
observed*, not what its author remembered to declare -- so these tests do real I/O
with the libraries the notebooks actually use, rather than asserting against mocks.

The safety tests carry the most weight. acia is used by people who came for the
biology, so provenance must never be the reason an analysis fails: if capture breaks,
`record()` has to write exactly the manifest it wrote before this feature existed.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import tifffile

from acia.analysis import (
    StageContext,
    _stage_io,
    check_stale,
    stage_graph,
    stage_table,
)


def _source(tmp_path, name="pos001_roi002.tiff"):
    path = tmp_path / name
    tifffile.imwrite(path, np.zeros((4, 4), np.uint8))
    return path


def _io(manifest_path, stage):
    return json.loads(Path(manifest_path).read_text())["stages"][stage]["io"]


# --------------------------------------------------------------------------- #
# what gets captured
# --------------------------------------------------------------------------- #


def test_records_reads_and_writes_of_the_usual_libraries(tmp_path):
    source = _source(tmp_path)
    ctx = StageContext.for_image(source, tmp_path / "out")

    tifffile.imread(str(source))
    np.savez_compressed(ctx.path("segmentation.npz"), a=np.arange(5))
    pd.DataFrame({"a": [1]}).to_csv(ctx.path("cell_properties.csv"), index=False)

    io = _io(ctx.record("Segment"), "Segment")
    assert [entry["path"] for entry in io["outputs"]] == [
        "cell_properties.csv",
        "segmentation.npz",
    ]
    assert [entry["path"] for entry in io["inputs"]] == ["../pos001_roi002.tiff"]
    assert all("size" in entry and "mtime" in entry for entry in io["outputs"])


def test_records_writes_python_cannot_see(tmp_path):
    """OpenCV writes through C++, raising no audit event -- the diff must catch it.

    This is not a corner case: the .mp4 artifacts every stage produces are written by
    cv2.VideoWriter or by ffmpeg, so a hook-only design would miss exactly the files
    the notebooks bother to declare.
    """
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    cv2.imwrite(str(ctx.path("raw.png")), np.zeros((4, 4, 3), np.uint8))

    io = _io(ctx.record("Segment"), "Segment")
    assert "raw.png" in [entry["path"] for entry in io["outputs"]]


def test_capture_does_not_depend_on_ctx_path(tmp_path):
    """Tracking keys off where a file is, not on how its path was built."""
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    with open(f"{ctx.output_dir}/hand_built.txt", "w") as handle:
        handle.write("x")

    io = _io(ctx.record("Segment"), "Segment")
    assert "hand_built.txt" in [entry["path"] for entry in io["outputs"]]


def test_a_directory_artifact_is_one_entry(tmp_path):
    """CTC tracking output is a folder per frame -- it must not flood the record."""
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    tracking = ctx.path("tracking")
    tracking.mkdir()
    for frame in range(50):
        (tracking / f"mask{frame:03}.tif").write_text("x")

    io = _io(ctx.record("Track"), "Track")
    assert [entry["path"] for entry in io["outputs"]] == ["tracking/"]


def test_unrelated_files_are_not_recorded(tmp_path):
    """Reads outside the tracked roots would drown the record in site-packages."""
    outside = tmp_path.parent / "elsewhere.csv"
    pd.DataFrame({"a": [1]}).to_csv(outside, index=False)
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    pd.read_csv(outside)

    io = _io(ctx.record("Segment"), "Segment")
    assert not any("elsewhere" in entry["path"] for entry in io["inputs"])
    outside.unlink()


def test_notebooks_are_code_not_data(tmp_path):
    """A notebook must never land in the recorded I/O.

    Under `scale()` papermill executes a copy in the output folder and rewrites it
    *after* the stage records, so recording it as an input would leave every stage
    permanently stale. Notebooks are identified by the `code` block instead.
    """
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    (tmp_path / "out" / "01_Segment.ipynb").write_text('{"cells": []}')
    (tmp_path / "out" / "01_Segment.ipynb").read_text()

    io = _io(ctx.record("Segment"), "Segment")

    recorded = [entry["path"] for entry in io["inputs"] + io["outputs"]]
    assert not any(path.endswith(".ipynb") for path in recorded)


def test_the_manifest_is_not_its_own_artifact(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    ctx.record("Segment")

    ctx2 = StageContext.for_image(output_folder=tmp_path / "out")
    io = _io(ctx2.record("Track"), "Track")

    recorded = [e["path"] for e in io["inputs"] + io["outputs"]]
    assert not any("stage_manifest" in path for path in recorded)


def test_declared_artifacts_that_were_never_written_are_flagged(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    ctx.path("real.csv").write_text("x")

    io = _io(ctx.record("Segment", artifacts=["real.csv", "typo.csv"]), "Segment")
    assert io["missing"] == ["typo.csv"]


def test_two_records_in_one_notebook_get_separate_windows(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    ctx.path("first.txt").write_text("x")
    ctx.record("Segment")
    ctx.path("second.txt").write_text("x")
    manifest_path = ctx.record("Track")

    assert [e["path"] for e in _io(manifest_path, "Segment")["outputs"]] == [
        "first.txt"
    ]
    assert [e["path"] for e in _io(manifest_path, "Track")["outputs"]] == ["second.txt"]


# --------------------------------------------------------------------------- #
# derived dependencies, staleness, undo
# --------------------------------------------------------------------------- #


def test_the_dependency_edge_is_inferred_not_declared(tmp_path):
    source = _source(tmp_path)
    first = StageContext.for_image(source, tmp_path / "out")
    np.savez(first.path("segmentation.npz"), a=np.arange(3))
    first.record("Segment")

    second = StageContext.for_image(source, tmp_path / "out")
    np.load(second.require("segmentation.npz"))
    manifest_path = second.record("Track")

    inputs = {
        e["path"]: e.get("produced_by") for e in _io(manifest_path, "Track")["inputs"]
    }
    assert inputs["segmentation.npz"] == "Segment"
    assert stage_graph(tmp_path / "out") == [("Segment", "segmentation.npz", "Track")]


def test_a_changed_input_makes_a_stage_stale(tmp_path):
    source = _source(tmp_path)
    first = StageContext.for_image(source, tmp_path / "out")
    first.path("segmentation.npz").write_text("x")
    first.record("Segment")

    second = StageContext.for_image(source, tmp_path / "out")
    second.require("segmentation.npz")
    second.record("Track")

    assert check_stale(tmp_path / "out") == []

    time.sleep(0.01)
    second.path("segmentation.npz").write_text("re-segmented, longer")

    stale = check_stale(tmp_path / "out")
    assert [entry["stage"] for entry in stale] == ["Track"]
    assert stale[0]["path"] == "segmentation.npz"


def test_building_a_context_warns_about_stale_stages(tmp_path):
    source = _source(tmp_path)
    first = StageContext.for_image(source, tmp_path / "out")
    first.path("segmentation.npz").write_text("x")
    first.record("Segment")
    second = StageContext.for_image(source, tmp_path / "out")
    second.require("segmentation.npz")
    second.record("Track")

    time.sleep(0.01)
    second.path("segmentation.npz").write_text("changed and longer")

    with pytest.warns(UserWarning, match="Track may be stale"):
        StageContext.for_image(source, tmp_path / "out")


def test_clear_removes_exactly_what_a_stage_recorded(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    ctx.path("keep.txt").write_text("x")
    ctx.record("Segment")

    ctx2 = StageContext.for_image(output_folder=tmp_path / "out")
    ctx2.path("drop.txt").write_text("x")
    (ctx2.path("tracking")).mkdir()
    (ctx2.path("tracking") / "res.txt").write_text("x")
    ctx2.record("Track")

    removed = ctx2.clear("Track")

    assert {p.name for p in removed} == {"drop.txt", "tracking"}
    assert not ctx2.path("drop.txt").exists()
    assert not ctx2.path("tracking").exists()
    assert ctx2.path("keep.txt").exists()  # another stage's output is untouched
    assert list(json.loads(ctx2.manifest_path.read_text())["stages"]) == ["Segment"]


def test_clear_is_a_noop_for_a_stage_that_never_ran(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    assert ctx.clear("Track") == []


def test_stage_returns_an_upstream_setting(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    ctx.record("Segment", pixel_size="0.065 micrometer")

    ctx2 = StageContext.for_image(output_folder=tmp_path / "out")
    assert ctx2.stage("Segment")["pixel_size"] == "0.065 micrometer"
    assert ctx2.stage("NeverRan") is None


# --------------------------------------------------------------------------- #
# when and how it ran
# --------------------------------------------------------------------------- #


def test_records_when_it_ran_and_how_long(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    entry = json.loads(ctx.record("Segment").read_text())["stages"]["Segment"]

    assert entry["started_at"].endswith("+00:00")
    assert entry["finished_at"].endswith("+00:00")
    assert entry["duration_s"] >= 0
    assert entry["env"]["python"] and entry["env"]["host"]


def test_records_which_notebook_ran(tmp_path, monkeypatch):
    notebook = tmp_path / "01_Segment.ipynb"
    notebook.write_text('{"cells": []}')
    monkeypatch.setenv(_stage_io.STAGE_NOTEBOOK_ENV, str(notebook))

    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    code = json.loads(ctx.record("Segment").read_text())["stages"]["Segment"]["code"]

    assert code["notebook"] == "01_Segment.ipynb"
    assert len(code["sha256"]) == 64
    assert code["mtime"].endswith("+00:00")

    # editing the analysis changes the digest -- this is what makes "these two
    # populations disagree" answerable with "they ran different code"
    notebook.write_text('{"cells": [1]}')
    ctx2 = StageContext.for_image(_source(tmp_path), tmp_path / "out2")
    other = json.loads(ctx2.record("Segment").read_text())["stages"]["Segment"]["code"]
    assert other["sha256"] != code["sha256"]


def test_the_code_block_is_omitted_when_the_notebook_is_unknown(tmp_path, monkeypatch):
    """Absent, never guessed -- a plain `python script.py` has no notebook."""
    monkeypatch.delenv(_stage_io.STAGE_NOTEBOOK_ENV, raising=False)
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)

    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    entry = json.loads(ctx.record("Segment").read_text())["stages"]["Segment"]

    assert "code" not in entry


# --------------------------------------------------------------------------- #
# recovering the source from the folder
# --------------------------------------------------------------------------- #


def test_a_later_stage_recovers_the_source_from_the_folder(tmp_path):
    source = _source(tmp_path)
    StageContext.for_image(source, tmp_path / "out").record("Segment")

    ctx = StageContext.for_image(output_folder=tmp_path / "out")

    assert Path(ctx.image_id).name == source.name
    assert ctx.population_id == "pos001_roi002"
    assert ctx.keys == {"population_id": "pos001_roi002", "position": 1, "roi": 2}


def test_passing_a_different_source_warns(tmp_path):
    StageContext.for_image(_source(tmp_path), tmp_path / "out").record("Segment")
    other = _source(tmp_path, "pos002_roi001.tiff")

    with pytest.warns(UserWarning, match="results from two"):
        StageContext.for_image(other, tmp_path / "out")


def test_recovery_without_a_manifest_names_the_stage_to_run_first(tmp_path):
    with pytest.raises(ValueError, match="first stage"):
        StageContext.for_image(output_folder=tmp_path / "out")


def test_a_moved_run_resolves_through_the_relative_source(tmp_path):
    """An absolute path dies when a run moves between machines; the relative one lives."""
    original = tmp_path / "run"
    (original).mkdir()
    source = _source(original)
    StageContext.for_image(source, original / "out").record("Segment")

    moved = tmp_path / "moved"
    moved.mkdir()
    (original / "out").rename(moved / "out")
    source.rename(moved / source.name)

    ctx = StageContext.for_image(output_folder=moved / "out")
    assert Path(ctx.image_id).resolve() == (moved / source.name).resolve()


# --------------------------------------------------------------------------- #
# require()'s messages
# --------------------------------------------------------------------------- #


def test_require_names_the_recorded_producer(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    ctx.path("segmentation.npz").write_text("x")
    ctx.record("Segment")
    ctx.path("segmentation.npz").unlink()

    ctx2 = StageContext.for_image(output_folder=tmp_path / "out")
    with pytest.raises(FileNotFoundError) as excinfo:
        ctx2.require("segmentation.npz")

    assert "'Segment'" in str(excinfo.value)


def test_require_reports_an_empty_folder_as_a_wrong_directory(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    with pytest.raises(FileNotFoundError, match="wrong working directory"):
        ctx.require("segmentation.npz")


def test_require_records_the_read_it_hands_back(tmp_path):
    """The explicit path, for readers the hook cannot see (OpenCV, remote sources)."""
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    ctx.path("upstream.npz").write_text("x")
    ctx.record("Segment")

    ctx2 = StageContext.for_image(output_folder=tmp_path / "out")
    ctx2.require("upstream.npz")
    io = _io(ctx2.record("Track"), "Track")

    assert "upstream.npz" in [entry["path"] for entry in io["inputs"]]


# --------------------------------------------------------------------------- #
# opting out, and the optional track() region
# --------------------------------------------------------------------------- #


def _existing(output_dir, name):
    """A file that pre-dates the stage, so reading it is an input and not an output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text("x\n")
    return name


def test_a_file_this_stage_wrote_is_an_output_not_an_input(tmp_path):
    """Reading back what you just produced does not make it a dependency."""
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")
    ctx.path("own.csv").write_text("x\n")
    ctx.path("own.csv").read_text()

    io = _io(ctx.record("Segment"), "Segment")
    assert "own.csv" in [entry["path"] for entry in io["outputs"]]
    assert "own.csv" not in [entry["path"] for entry in io["inputs"]]


def test_track_io_false_records_nothing_it_was_not_told(tmp_path):
    output = tmp_path / "out"
    upstream = _existing(output, "upstream.csv")
    ctx = StageContext.for_image(_source(tmp_path), output, track_io=False)

    ctx.path(upstream).read_text()

    io = _io(ctx.record("Segment"), "Segment")
    # only the source, which is always recorded -- the read went unobserved
    assert [entry["path"] for entry in io["inputs"]] == ["../pos001_roi002.tiff"]


def test_a_track_region_captures_reads_when_tracking_is_off(tmp_path):
    output = tmp_path / "out"
    upstream = _existing(output, "upstream.csv")
    ctx = StageContext.for_image(_source(tmp_path), output, track_io=False)

    with ctx.track():
        pd.read_csv(ctx.path(upstream))

    io = _io(ctx.record("Segment"), "Segment")
    assert "upstream.csv" in [entry["path"] for entry in io["inputs"]]


def test_track_regions_accumulate_across_blocks(tmp_path):
    """A `with` cannot span notebook cells, so regions must be re-enterable."""
    output = tmp_path / "out"
    _existing(output, "a.csv")
    _existing(output, "b.csv")
    ctx = StageContext.for_image(_source(tmp_path), output, track_io=False)

    with ctx.track():
        ctx.path("a.csv").read_text()
    with ctx.track():
        ctx.path("b.csv").read_text()

    read = [entry["path"] for entry in _io(ctx.record("Segment"), "Segment")["inputs"]]
    assert "a.csv" in read and "b.csv" in read


def test_a_region_does_not_switch_off_the_capture_around_it(tmp_path):
    output = tmp_path / "out"
    _existing(output, "after.csv")
    ctx = StageContext.for_image(_source(tmp_path), output)

    with ctx.track():
        pass
    ctx.path("after.csv").read_text()  # still captured after the region closed

    io = _io(ctx.record("Segment"), "Segment")
    assert "after.csv" in [entry["path"] for entry in io["inputs"]]


# --------------------------------------------------------------------------- #
# safety: provenance must never break an analysis
# --------------------------------------------------------------------------- #


def test_a_broken_recorder_cannot_break_open(tmp_path):
    """An exception in an audit hook propagates into the open() that triggered it."""
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    class _Exploding:
        def note(self, *args):
            raise RuntimeError("boom")

    _stage_io.activate(_Exploding())
    try:
        ctx.path("still_works.txt").write_text("x")
        assert ctx.path("still_works.txt").read_text() == "x"
    finally:
        _stage_io.activate(None)


def test_record_still_writes_the_manifest_when_capture_fails(tmp_path):
    ctx = StageContext.for_image(_source(tmp_path), tmp_path / "out")

    class _Broken:
        def apply_diff(self, *args):
            raise RuntimeError("boom")

        def reset(self, *args):
            pass

    object.__setattr__(ctx, "_recorder", _Broken())
    entry = json.loads(ctx.record("Segment", n=3).read_text())["stages"]["Segment"]

    assert entry["n"] == 3  # the pre-feature manifest, intact
    assert "io" not in entry


def test_arming_twice_installs_one_hook(tmp_path, monkeypatch):
    installed = []
    monkeypatch.setattr(
        _stage_io.sys, "addaudithook", lambda hook: installed.append(hook)
    )
    monkeypatch.setattr(_stage_io, "_ARMED", False)

    _stage_io.arm()
    _stage_io.arm()

    assert len(installed) == 1


def test_roots_are_compared_resolved_on_both_sides(tmp_path):
    """On macOS /tmp is a symlink to /private/tmp; comparing the two forms matches
    nothing, which silently empties the record."""
    recorder = _stage_io._Recorder([tmp_path.resolve()])
    unresolved = Path(os.path.join(str(tmp_path), "x.txt"))

    recorder.note(str(unresolved), "r")

    assert recorder.reads == {unresolved.resolve()}


def test_a_manifest_without_io_still_loads(tmp_path):
    """Runs recorded before this feature must stay readable."""
    output = tmp_path / "out"
    output.mkdir()
    (output / "stage_manifest.json").write_text(
        json.dumps(
            {
                "population": {"population_id": "pos001_roi002"},
                "stages": {"01_Segment": {"artifacts": ["x.npz"], "n": 1}},
            }
        )
    )

    assert check_stale(output) == []
    assert stage_graph(output) == []
    table = stage_table(tmp_path, pattern="out")
    assert table.loc[0, "stage"] == "01_Segment"
    assert table.loc[0, "n_inputs"] is None


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #


def test_stage_table_is_one_row_per_run_with_settings_as_columns(tmp_path):
    for name in ("pos001_roi001.tiff", "pos002_roi001.tiff"):
        source = _source(tmp_path, name)
        folder = tmp_path / Path(name).stem / "output"
        ctx = StageContext.for_image(source, folder)
        ctx.path("segmentation.npz").write_text("x")
        ctx.record("Segment", pixel_size="0.065 micrometer")
        StageContext.for_image(source, folder).record("Track", n_tracklets=7)

    table = stage_table(tmp_path)

    assert len(table) == 4
    assert set(table["stage"]) == {"Segment", "Track"}
    assert set(table["population_id"]) == {"pos001_roi001", "pos002_roi001"}
    assert (
        table[table.stage == "Segment"]["pixel_size"].tolist()
        == ["0.065 micrometer"] * 2
    )
    assert table[table.stage == "Track"]["n_tracklets"].tolist() == [7, 7]
    assert not table["stale"].any()


def test_stage_table_is_empty_for_a_folder_with_no_runs(tmp_path):
    assert stage_table(tmp_path).empty


# --------------------------------------------------------------------------- #
# optional OpenLineage export
# --------------------------------------------------------------------------- #


def test_openlineage_export_round_trips_the_recorded_graph(tmp_path):
    pytest.importorskip("openlineage.client", reason="needs the acia[lineage] extra")
    from acia.analysis.lineage import to_openlineage

    source = _source(tmp_path)
    folder = tmp_path / "pos001_roi002" / "output"
    first = StageContext.for_image(source, folder)
    np.savez(first.path("segmentation.npz"), a=np.arange(3))
    first.record("Segment")
    second = StageContext.for_image(output_folder=folder)
    np.load(second.require("segmentation.npz"))
    second.record("Track")

    out = to_openlineage(tmp_path, tmp_path / "lineage.jsonl")

    events = [json.loads(line) for line in out.read_text().strip().splitlines()]
    jobs = {event["job"]["name"]: event for event in events}
    assert set(jobs) == {"Segment/pos001_roi002", "Track/pos001_roi002"}
    assert all(event["eventType"] == "COMPLETE" for event in events)
    # the edge survives the export: what Segment output, Track takes as input
    produced = {ds["name"] for ds in jobs["Segment/pos001_roi002"]["outputs"]}
    consumed = {ds["name"] for ds in jobs["Track/pos001_roi002"]["inputs"]}
    assert produced & consumed


def test_openlineage_export_is_not_needed_to_import_acia():
    """The extra must never become a hidden hard dependency."""
    import importlib

    assert importlib.import_module("acia.analysis") is not None
