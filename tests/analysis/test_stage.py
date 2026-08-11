"""Tests for the shared notebook-stage run context (acia.analysis.stage)."""

import json

import pandas as pd
import pytest

from acia import __version__
from acia.analysis import (
    StageContext,
    population_id_of,
    read_manifest,
    stages_run,
)


def _ctx(tmp_path, name="pos001_roi002.tiff", **kwargs):
    image = tmp_path / name
    image.write_text("")  # a file source; contents irrelevant here
    return StageContext.for_image(image, tmp_path / "output", **kwargs)


def test_population_id_of_file_drops_extension(tmp_path):
    image = tmp_path / "pos001_roi002.tiff"
    image.write_text("")
    assert population_id_of(image) == "pos001_roi002"


def test_population_id_of_folder_keeps_dotted_name(tmp_path):
    folder = tmp_path / "pos001_roi002.long.name"
    folder.mkdir()
    # a blind .stem would truncate at the dot and split one population in two
    assert population_id_of(folder) == "pos001_roi002.long.name"


def test_for_image_derives_keys_and_creates_output(tmp_path):
    ctx = _ctx(tmp_path)

    assert ctx.population_id == "pos001_roi002"
    assert ctx.keys == {"population_id": "pos001_roi002", "position": 1, "roi": 2}
    assert ctx.output_dir.is_dir()
    assert str(ctx).startswith("population pos001_roi002")


def test_for_image_unmatched_pattern_keeps_keys_as_none(tmp_path):
    ctx = _ctx(tmp_path, name="some_other_movie.tiff")

    assert ctx.population_id == "some_other_movie"
    assert ctx.keys == {
        "population_id": "some_other_movie",
        "position": None,
        "roi": None,
    }


def test_for_image_without_key_pattern(tmp_path):
    ctx = _ctx(tmp_path, key_pattern=None)
    assert ctx.keys == {"population_id": "pos001_roi002"}


def test_for_image_custom_key_pattern(tmp_path):
    ctx = _ctx(tmp_path, name="well_B7.tiff", key_pattern=r"well_(?P<well>\w+)")
    assert ctx.keys == {"population_id": "well_B7", "well": "B7"}


def test_key_pattern_without_named_groups_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="named groups"):
        _ctx(tmp_path, key_pattern=r"pos(\d+)")


def test_for_image_create_false_leaves_folder_absent(tmp_path):
    ctx = _ctx(tmp_path, create=False)
    assert not ctx.output_dir.exists()


def test_require_returns_existing_and_reports_producer(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.path("segmentation.npz").write_text("x")

    assert ctx.require("segmentation.npz", "01_Segment.ipynb") == ctx.path(
        "segmentation.npz"
    )
    assert ctx.has("segmentation.npz")

    with pytest.raises(FileNotFoundError) as excinfo:
        ctx.require("tracking", "02_Track.ipynb")
    message = str(excinfo.value)
    assert "02_Track.ipynb" in message  # what to run
    assert "working directory" in message  # and where it would have run


def test_record_appends_stages_and_keeps_population(tmp_path):
    ctx = _ctx(tmp_path)

    ctx.record("01_Segment", ["segmentation.npz"], n_detections=17)
    manifest_path = ctx.record("02_Track", ["tracking/"], mode="greedy")

    manifest = json.loads(manifest_path.read_text())
    assert list(manifest["stages"]) == [
        "01_Segment",
        "02_Track",
    ]  # appended, not replaced
    assert manifest["stages"]["01_Segment"]["artifacts"] == ["segmentation.npz"]
    assert manifest["stages"]["01_Segment"]["n_detections"] == 17
    assert manifest["stages"]["02_Track"]["mode"] == "greedy"
    assert manifest["population"] == {
        "population_id": "pos001_roi002",
        "position": 1,
        "roi": 2,
        "image_id": str((tmp_path / "pos001_roi002.tiff").resolve()),
    }


def test_record_stamps_provenance(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.record("01_Segment", [])

    entry = ctx.manifest()["stages"]["01_Segment"]
    assert entry["acia_version"] == __version__
    assert entry["finished_at"].endswith("+00:00")  # UTC, so runs stay comparable


def test_record_rerun_replaces_only_its_own_entry(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.record("01_Segment", ["segmentation.npz"], n_detections=17)
    ctx.record("02_Track", ["tracking/"])
    ctx.record("01_Segment", ["segmentation.npz"], n_detections=42)

    stages = ctx.manifest()["stages"]
    assert stages["01_Segment"]["n_detections"] == 42
    assert "02_Track" in stages


def test_read_manifest_and_stages_run(tmp_path):
    ctx = _ctx(tmp_path)
    assert read_manifest(ctx.output_dir) == {}  # nothing ran yet -> not an error
    assert stages_run(ctx.output_dir) == []

    ctx.record("01_Segment", ["segmentation.npz"])
    assert stages_run(ctx.output_dir) == ["01_Segment"]
    assert read_manifest(ctx.output_dir)["population"]["roi"] == 2


def test_keyed_adds_key_columns_and_keeps_units(tmp_path):
    ctx = _ctx(tmp_path)
    df = pd.DataFrame({"area": [1.0, 2.0]})
    df.attrs["units"] = {"area": "micrometer ** 2"}

    keyed = ctx.keyed(df)

    assert list(keyed["population_id"]) == ["pos001_roi002"] * 2
    assert list(keyed["position"]) == [1, 1]
    assert list(keyed["roi"]) == [2, 2]
    # assign() drops attrs -- the unit map must survive, or the CSV loses its units
    assert keyed.attrs["units"] == {"area": "micrometer ** 2"}
    assert "population_id" not in df.columns  # input untouched


def test_keyed_without_units_still_has_the_attr(tmp_path):
    keyed = _ctx(tmp_path).keyed(pd.DataFrame({"area": [1.0]}))
    assert keyed.attrs["units"] == {}
