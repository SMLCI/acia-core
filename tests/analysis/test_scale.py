"""Tests for the source-aware notebook scaling (acia.analysis.scale)."""

from unittest.mock import patch

import papermill as pm
import pytest

from acia.analysis import _source_label, default_execution_naming, scale


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


def test_source_label_prefers_input_file_name():
    # path/URL sources: the file name (not the execution folder name)
    assert _source_label("/data/exp/pos1_roi2.tiff", "pos1_roi2") == "pos1_roi2.tiff"
    assert (
        _source_label("smb://srv/data/pos3.ome.tif?a=1", "pos3.ome") == "pos3.ome.tif"
    )
    # ids and dicts have no file name -> the execution name identifies them
    assert _source_label(42, "execution_42") == "execution_42"
    assert _source_label({"path": "x"}, "my_run") == "my_run"


class _StubBar:
    """Minimal tqdm stand-in that records the descriptions it is given."""

    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable
        self.descriptions: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._iterable)

    def set_description(self, desc, **kwargs):
        self.descriptions.append(desc)

    def update(self, n=1):
        pass


def test_scale_progress_reports_stage_and_source(tmp_path):
    """The bar says which stage notebook runs on which input, not just a count."""
    stages = []
    for name in ("01_Segment.ipynb", "02_Track.ipynb"):
        script = tmp_path / name
        script.write_text("{}")
        stages.append(script)
    bars = []

    def _bar(*args, **kwargs):
        bars.append(_StubBar(*args, **kwargs))
        return bars[-1]

    with (
        patch("acia.analysis.pm.execute_notebook"),
        patch("acia.analysis.tqdm", _bar),
    ):
        scale(tmp_path / "out", stages, image_ids=["/data/exp/pos1_roi2.tiff"])

    descriptions = bars[0].descriptions
    assert descriptions[0] == "pos1_roi2.tiff"  # source, before the first stage
    assert "01_Segment.ipynb | pos1_roi2.tiff" in descriptions
    assert "02_Track.ipynb | pos1_roi2.tiff" in descriptions


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


def test_scale_points_the_kernel_at_the_source_notebook(tmp_path):
    """The template, not the executed copy.

    papermill writes outputs back into the copy as it runs, so every execution
    folder's copy has different bytes. Hashing those would make the recorded code
    digest differ per image and destroy the one question it answers -- did this whole
    batch run the same code?
    """
    import os

    from acia.analysis._stage_io import STAGE_NOTEBOOK_ENV

    script = _make_script(tmp_path)
    seen = []

    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        exec_nb.side_effect = lambda *a, **k: seen.append(
            os.environ[STAGE_NOTEBOOK_ENV]
        )
        scale(tmp_path / "out", script, image_ids=[1, 2])

    assert seen == [str(script), str(script)]  # the template, for both images


def test_scale_rejects_bad_stage_progress(tmp_path):
    script = _make_script(tmp_path)
    with pytest.raises(ValueError):
        scale(tmp_path / "out", script, image_ids=[1], stage_progress="louder")


def test_scale_labels_the_papermill_cell_bar(tmp_path):
    """The per-notebook bar says which stage runs on which source, not "Executing"."""
    stages = []
    for name in ("01_Segment.ipynb", "02_Track.ipynb"):
        stage = tmp_path / name
        stage.write_text("{}")
        stages.append(stage)

    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(tmp_path / "out", stages, image_ids=["/data/exp/pos1_roi2.tiff"])

    bars = [call.kwargs["progress_bar"] for call in exec_nb.call_args_list]
    assert bars == [
        {"desc": "  ↳ 01_Segment.ipynb | pos1_roi2.tiff", "leave": True},
        {"desc": "  ↳ 02_Track.ipynb | pos1_roi2.tiff", "leave": True},
    ]


def test_scale_stage_progress_collapse_does_not_leave_bars(tmp_path):
    script = _make_script(tmp_path)
    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(tmp_path / "out", script, image_ids=[1], stage_progress="collapse")

    assert exec_nb.call_args_list[0].kwargs["progress_bar"]["leave"] is False


def test_scale_stage_progress_off_disables_the_cell_bar(tmp_path):
    script = _make_script(tmp_path)
    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(tmp_path / "out", script, image_ids=[1], stage_progress="off")

    assert exec_nb.call_args_list[0].kwargs["progress_bar"] is False


def test_scale_storage_parameter_name_none_omits_it(tmp_path):
    script = _make_script(tmp_path)
    out = tmp_path / "out"
    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(out, script, image_ids=[1], storage_parameter_name=None)
    params = exec_nb.call_args_list[0].kwargs["parameters"]
    assert "storage_folder" not in params


def test_scale_storage_parameter_name_custom(tmp_path):
    script = _make_script(tmp_path)
    out = tmp_path / "out"
    with patch("acia.analysis.pm.execute_notebook") as exec_nb:
        scale(out, script, image_ids=[1], storage_parameter_name="output_folder")
    params = exec_nb.call_args_list[0].kwargs["parameters"]
    assert "storage_folder" not in params
    assert params["output_folder"].endswith("execution_1")


# --- failure isolation and kernel-start retry ---------------------------------
#
# scale() used to catch only PapermillExecutionError ("a cell raised"). A kernel
# that never starts raises a bare RuntimeError, which escaped and aborted the
# whole batch -- the sources after the failing one were never attempted.


def _kernel_died():
    return RuntimeError("Kernel died before replying to kernel_info")


def test_scale_isolates_kernel_start_failure(tmp_path):
    """A kernel that never starts fails its own source, not the batch."""
    script = _make_script(tmp_path)
    out = tmp_path / "out"

    def flaky(*args, **kwargs):
        if kwargs["parameters"]["image_id"] == 2:
            raise _kernel_died()

    with patch("acia.analysis.pm.execute_notebook", side_effect=flaky) as exec_nb:
        scale(out, script, image_ids=[1, 2, 3])

    # every source was attempted (the retry gives the failing one two calls)
    attempted = [c.kwargs["parameters"]["image_id"] for c in exec_nb.call_args_list]
    assert sorted(set(attempted)) == [1, 2, 3]
    assert (out / "execution_1").is_dir()
    assert (out / "execution_3").is_dir()


def test_scale_isolates_arbitrary_failure(tmp_path):
    """Any exception is isolated, not just papermill's own."""
    script = _make_script(tmp_path)
    out = tmp_path / "out"

    def flaky(*args, **kwargs):
        if kwargs["parameters"]["image_id"] == 1:
            raise OSError("no space left on device")

    with patch("acia.analysis.pm.execute_notebook", side_effect=flaky) as exec_nb:
        scale(out, script, image_ids=[1, 2])

    attempted = [c.kwargs["parameters"]["image_id"] for c in exec_nb.call_args_list]
    assert attempted == [1, 2]


def test_scale_retries_a_kernel_that_failed_to_start(tmp_path):
    """One transient kernel-start failure is retried, and the source succeeds."""
    script = _make_script(tmp_path)
    out = tmp_path / "out"
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _kernel_died()

    with patch("acia.analysis.pm.execute_notebook", side_effect=flaky):
        result = scale(out, script, image_ids=[1])

    assert calls["n"] == 2  # failed once, retried once, succeeded
    assert len(result) == 1  # counted as a success, not a failure


def test_scale_does_not_retry_a_cell_error(tmp_path):
    """A notebook whose cell raised is a real failure -- re-running it is waste."""
    script = _make_script(tmp_path)
    out = tmp_path / "out"
    calls = {"n": 0}

    def always_fails(*args, **kwargs):
        calls["n"] += 1
        raise pm.PapermillExecutionError(
            exec_count=1,
            source="1/0",
            ename="ZeroDivisionError",
            evalue="division by zero",
            traceback=[],
        )

    with patch("acia.analysis.pm.execute_notebook", side_effect=always_fails):
        result = scale(out, script, image_ids=[1])

    assert calls["n"] == 1
    assert result == []


def test_scale_gives_up_after_the_kernel_retry(tmp_path):
    """A kernel that never starts at all is isolated once the retry is spent."""
    script = _make_script(tmp_path)
    out = tmp_path / "out"
    calls = {"n": 0}

    def always_dies(*args, **kwargs):
        calls["n"] += 1
        raise _kernel_died()

    with patch("acia.analysis.pm.execute_notebook", side_effect=always_dies):
        result = scale(out, script, image_ids=[1, 2])

    assert calls["n"] == 4  # two sources x (initial attempt + one retry)
    assert result == []
