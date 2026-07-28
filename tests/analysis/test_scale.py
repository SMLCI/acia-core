"""Tests for the source-aware notebook scaling (acia.analysis.scale)."""

from unittest.mock import patch

import pytest

from acia.analysis import default_execution_naming, scale


def _make_script(tmp_path):
    script = tmp_path / "analysis.ipynb"
    script.write_text("{}")  # contents irrelevant; papermill is mocked
    return script


def test_default_execution_naming_int():
    assert default_execution_naming(42) == "execution_42"


def test_default_execution_naming_path():
    assert default_execution_naming("/data/exp/pos1.tif") == "pos1"


def test_default_execution_naming_url():
    assert (
        default_execution_naming("smb://fileserver.lab/data/pos3.ome.tif") == "pos3.ome"
    )


def test_default_execution_naming_unsupported():
    with pytest.raises(ValueError):
        default_execution_naming({"image_id": 1})


def test_scale_int_ids_backward_compatible(tmp_path):
    script = _make_script(tmp_path)
    out = tmp_path / "out"

    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(out, script, image_ids=[1, 2])

    # folders named execution_<id>
    assert (out / "execution_1").is_dir()
    assert (out / "execution_2").is_dir()
    # image_id injected per execution
    injected = [
        call.kwargs["parameters"]["image_id"] for call in exec_nb.call_args_list
    ]
    assert injected == [1, 2]


def test_scale_paths_use_stem_and_inject(tmp_path):
    script = _make_script(tmp_path)
    out = tmp_path / "out"
    paths = ["smb://srv/data/posA.tif", "/local/data/posB.tif"]

    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(out, script, image_ids=paths)

    assert (out / "posA").is_dir()
    assert (out / "posB").is_dir()
    injected = [
        call.kwargs["parameters"]["image_id"] for call in exec_nb.call_args_list
    ]
    assert injected == paths


def test_scale_custom_parameter_name(tmp_path):
    script = _make_script(tmp_path)
    out = tmp_path / "out"

    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(
            out,
            script,
            image_ids=["smb://srv/data/x.tif"],
            parameter_name="image_path",
        )

    params = exec_nb.call_args_list[0].kwargs["parameters"]
    assert params["image_path"] == "smb://srv/data/x.tif"
    assert "image_id" not in params


def test_scale_dict_entries_merged(tmp_path):
    script = _make_script(tmp_path)
    out = tmp_path / "out"

    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(
            out,
            script,
            image_ids=[{"host": "srv", "share": "data", "path": "x.tif"}],
            execution_naming=lambda item: item["path"].replace("/", "_"),
        )

    params = exec_nb.call_args_list[0].kwargs["parameters"]
    assert params["host"] == "srv"
    assert params["share"] == "data"
    assert params["path"] == "x.tif"
    assert (out / "x.tif").is_dir()


def test_scale_warns_on_name_collision(tmp_path, caplog):
    script = _make_script(tmp_path)
    out = tmp_path / "out"
    # two different folders, same file stem -> collision on stem naming
    paths = ["smb://srv/a/pos1.tif", "smb://srv/b/pos1.tif"]

    with (
        patch("acia.analysis.pm.execute_notebook"),
        caplog.at_level("WARNING"),
    ):
        scale(out, script, image_ids=paths, exist_ok=True)

    assert any("maps to multiple sources" in rec.message for rec in caplog.records)


def _make_marker_script(tmp_path, name="marker.ipynb", fail_on_bad=False):
    """A minimal *real* notebook (the parallel path spawns processes, so papermill
    can't be mocked across them). It writes the injected image_id into a marker
    file inside its storage_folder, optionally failing when image_id contains 'bad'.
    """
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "python3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    params = nbformat.v4.new_code_cell("storage_folder = ''\nimage_id = None")
    params.metadata["tags"] = ["parameters"]
    guard = "assert 'bad' not in str(image_id), 'boom'\n" if fail_on_bad else ""
    body = nbformat.v4.new_code_cell(
        "from pathlib import Path\n"
        f"{guard}"
        "Path(storage_folder, 'done.txt').write_text(str(image_id))"
    )
    nb.cells = [params, body]
    path = tmp_path / name
    nbformat.write(nb, str(path))
    return path


def test_scale_parallel_runs_all(tmp_path):
    script = _make_marker_script(tmp_path)
    out = tmp_path / "out"
    paths = [f"/data/pos{i}.tif" for i in range(4)]

    result = scale(out, script, image_ids=paths, max_workers=2, kernel_name="python3")

    assert len(result) == 4
    # each ran in its own process (own cwd), writing its own marker -- no collision
    for i in range(4):
        assert (out / f"pos{i}" / "done.txt").read_text() == f"/data/pos{i}.tif"


def test_scale_parallel_isolates_failures(tmp_path):
    script = _make_marker_script(tmp_path, fail_on_bad=True)
    out = tmp_path / "out"
    paths = ["/data/good1.tif", "/data/bad.tif", "/data/good2.tif"]

    result = scale(out, script, image_ids=paths, max_workers=2, kernel_name="python3")

    # the failing ROI does not abort the batch; the good ones still produced output
    assert len(result) == 2
    assert (out / "good1" / "done.txt").exists()
    assert (out / "good2" / "done.txt").exists()
    assert not (out / "bad" / "done.txt").exists()


def test_scale_rejects_bad_max_workers(tmp_path):
    script = _make_script(tmp_path)
    with pytest.raises(ValueError):
        scale(tmp_path / "out", script, image_ids=[1], max_workers=0)
