"""``scale(on_existing=...)`` and what clearing a stage removes."""

import json
from unittest.mock import patch

import pytest

from acia.analysis import scale
from acia.analysis._stage_io import STAGE_NOTEBOOK_ENV
from acia.analysis.cell_cycle import (
    SUMMARY_NAME,
    remove_summary_sections,
    update_summary,
)
from acia.analysis.stage import StageContext, clear_stage, read_manifest


def _summary(output):
    return json.loads((output / SUMMARY_NAME).read_text())


def _run_stage(monkeypatch, image, output, stage, notebook, files, sections):
    """Run a fake stage the way a stage notebook does under scale()."""
    monkeypatch.setenv(STAGE_NOTEBOOK_ENV, str(notebook))
    ctx = StageContext.for_image(image, output, stage=stage)
    for name in files:
        ctx.output_path(name).write_text(name)
    for section in sections:
        update_summary(output, section, {"value": 1})
    ctx.finish()
    monkeypatch.delenv(STAGE_NOTEBOOK_ENV)


@pytest.fixture
def population(tmp_path, monkeypatch):
    """One execution folder in which two stages ran."""
    folder = tmp_path / "out" / "pos001_roi001"
    output = folder / "output"
    image = tmp_path / "pos001_roi001.tiff"
    image.write_bytes(b"x")
    scripts = []
    for name in ("03_GrowthRate.ipynb", "03_DoublingTime.ipynb"):
        script = tmp_path / name
        script.write_text("{}")
        scripts.append(script)
    _run_stage(
        monkeypatch,
        image,
        output,
        "GrowthRate",
        scripts[0],
        ["growth_estimates.csv", "figures/growth.png"],
        ["growth_rate_population"],
    )
    _run_stage(
        monkeypatch,
        image,
        output,
        "DoublingTime",
        scripts[1],
        ["doubling_times.csv", "figures/dt.png"],
        ["doubling_time", "growth_rate_per_cell"],
    )
    for script in scripts:  # the executed copies scale() leaves behind
        (folder / script.name).write_text("{}")
    (folder / "03_DoublingTime.html").write_text("<html/>")
    return {"folder": folder, "output": output, "image": image, "scripts": scripts}


# -- ownership and clear_stage ------------------------------------------------


def test_update_summary_records_the_owning_stage(population):
    stages = read_manifest(population["output"])["stages"]
    assert stages["GrowthRate"]["summary_sections"] == ["growth_rate_population"]
    assert stages["DoublingTime"]["summary_sections"] == [
        "doubling_time",
        "growth_rate_per_cell",
    ]


def test_clear_stage_removes_only_its_own_work(population):
    output = population["output"]
    removed = clear_stage(output, "DoublingTime")

    assert not (output / "doubling_times.csv").exists()
    assert not (output / "figures" / "dt.png").exists()
    # summary.json is shared: only this stage's sections go
    assert _summary(output) == {"growth_rate_population": {"value": 1}}
    assert output / SUMMARY_NAME in removed
    # the other stage's results and entry are untouched
    assert (output / "growth_estimates.csv").exists()
    assert (output / "figures" / "growth.png").exists()
    assert set(read_manifest(output)["stages"]) == {"GrowthRate"}


def test_clearing_the_last_owner_deletes_summary_json(population):
    output = population["output"]
    clear_stage(output, "DoublingTime")
    clear_stage(output, "GrowthRate")
    assert not (output / SUMMARY_NAME).exists()


def test_clear_of_a_stage_that_never_ran_is_a_noop(population):
    assert clear_stage(population["output"], "Nope") == []
    assert len(_summary(population["output"])) == 3


def test_a_manifest_without_ownership_keeps_summary_json(tmp_path):
    """Entries written before ownership existed must not cost the whole file."""
    image = tmp_path / "pos001_roi001.tiff"
    image.write_bytes(b"x")
    output = tmp_path / "output"
    ctx = StageContext.for_image(image, output, stage="DoublingTime")
    update_summary(output, "doubling_time", {"value": 1})
    ctx.finish()
    manifest = read_manifest(output)
    manifest["stages"]["DoublingTime"].pop("summary_sections")
    (output / "stage_manifest.json").write_text(json.dumps(manifest))

    clear_stage(output, "DoublingTime")
    assert _summary(output) == {"doubling_time": {"value": 1}}


def test_update_summary_without_an_open_stage_records_nothing(tmp_path):
    update_summary(tmp_path, "section", {"value": 1})
    assert not (tmp_path / "stage_manifest.json").exists()
    assert _summary(tmp_path) == {"section": {"value": 1}}


def test_remove_summary_sections_noops(tmp_path):
    assert remove_summary_sections(tmp_path, ["a"]) is None  # no file
    update_summary(tmp_path, "a", {"value": 1})
    assert remove_summary_sections(tmp_path, ["b"]) is None  # no such section
    assert remove_summary_sections(tmp_path, ["a"]) == tmp_path / SUMMARY_NAME
    assert not (tmp_path / SUMMARY_NAME).exists()


# -- scale(on_existing=...) ---------------------------------------------------


def _scale(population, scripts, **kwargs):
    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(
            population["folder"].parent,
            scripts,
            image_ids=[str(population["image"])],
            **kwargs,
        )
    return exec_nb


def test_skip_keeps_a_stage_that_already_ran(population):
    exec_nb = _scale(population, population["scripts"], on_existing="skip")
    exec_nb.assert_not_called()
    assert (population["output"] / "doubling_times.csv").exists()


def test_overwrite_clears_and_reruns_only_the_listed_stages(population):
    output = population["output"]
    doubling = population["scripts"][1]

    exec_nb = _scale(population, [doubling], on_existing="overwrite")

    assert exec_nb.call_count == 1
    assert not (output / "doubling_times.csv").exists()
    assert not (population["folder"] / "03_DoublingTime.html").exists()
    assert _summary(output) == {"growth_rate_population": {"value": 1}}
    # the stage that was not listed is kept, results and notebook
    assert (output / "growth_estimates.csv").exists()
    assert (population["folder"] / "03_GrowthRate.ipynb").exists()
    assert "GrowthRate" in read_manifest(output)["stages"]


def test_error_refuses_a_folder_where_a_stage_ran(population, caplog):
    with caplog.at_level("WARNING"):
        exec_nb = _scale(population, population["scripts"], on_existing="error")
    exec_nb.assert_not_called()
    assert any("Scaling failed" in rec.message for rec in caplog.records)
    assert (population["output"] / "doubling_times.csv").exists()


def test_error_is_the_default_and_runs_a_fresh_folder(tmp_path):
    script = tmp_path / "stage.ipynb"
    script.write_text("{}")
    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(tmp_path / "out", script, image_ids=["a.tif"])
    assert exec_nb.call_count == 1


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ({"exist_skip": True}, "skip"),
        ({"exist_ok": True, "exist_skip": True}, "skip"),
        ({"exist_ok": True}, "overwrite"),
        ({"exist_ok": False}, "error"),
    ],
)
def test_the_legacy_flags_still_work_but_warn(tmp_path, legacy, expected):
    script = tmp_path / "stage.ipynb"
    script.write_text("{}")
    with (
        patch("acia.analysis._scale_execute_one", return_value=[]) as run,
        pytest.warns(DeprecationWarning, match="on_existing"),
    ):
        scale(tmp_path / "out", script, image_ids=["a.tif"], **legacy)
    assert run.call_args.args[2] == expected


def test_on_existing_is_validated(tmp_path):
    script = tmp_path / "stage.ipynb"
    script.write_text("{}")
    with pytest.raises(ValueError, match="on_existing"):
        scale(tmp_path / "out", script, image_ids=["a.tif"], on_existing="keep")
    with pytest.raises(TypeError, match="not both"):
        scale(
            tmp_path / "out",
            script,
            image_ids=["a.tif"],
            on_existing="skip",
            exist_skip=True,
        )
