"""Tests for scale()'s progress reporting (acia.analysis._scale_progress).

The interesting part -- workers reporting instead of drawing -- is exercised
in-process here: the queue is a plain ``queue.Queue`` and the papermill engine
runs in this process, so a broken engine override shows up as a test failure
rather than as a garbled display inside a spawned worker.
"""

import queue

import papermill as pm
import pytest

from acia.analysis._scale_progress import (
    _QueuePBar,
    _WorkerBars,
    register_engine,
)
from tests.analysis.test_scale import _make_marker_script


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


class _StubBar:
    """tqdm stand-in recording what the parent-side renderer does to it."""

    def __init__(self, iterable=None, total=None, desc=None, **kwargs):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.n = 0
        self.closed = False
        self.kwargs = kwargs
        self.descriptions = [] if desc is None else [desc]

    def reset(self, total=None):
        # tqdm keeps the old total unless a new one is passed
        self.n = 0
        if total is not None:
            self.total = total

    def update(self, n=1):
        self.n += n

    def set_description(self, desc, **kwargs):
        self.desc = desc
        self.descriptions.append(desc)

    def refresh(self):
        pass

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# child side
# --------------------------------------------------------------------------- #


def test_queue_pbar_reports_total_progress_and_end():
    q = queue.Queue()
    bar = _QueuePBar(q, key=7, total=3)

    bar.update()
    bar.update()
    bar.refresh()  # forced, so the coalescing cannot hide it
    bar.close()

    events = _drain(q)
    assert events[0] == ("total", 7, 3)
    assert ("progress", 7, 2) in events
    assert events[-1] == ("stage_end", 7)
    assert all(event[1] == 7 for event in events)


def test_queue_pbar_reports_absolute_counts():
    """papermill assigns `pbar.n` directly when it force-completes a bar."""
    q = queue.Queue()
    bar = _QueuePBar(q, key="w", total=10)

    bar.update(4)
    bar.n = 10  # what NotebookExecutionManager.complete_pbar() does
    bar.refresh()

    progress = [event for event in _drain(q) if event[0] == "progress"]
    assert progress[-1] == ("progress", "w", 10)


def test_queue_pbar_coalesces_fast_updates_but_flushes_on_close():
    q = queue.Queue()
    bar = _QueuePBar(q, key=1, total=100)

    for _ in range(50):
        bar.update()
    bar.close()

    progress = [event for event in _drain(q) if event[0] == "progress"]
    assert len(progress) < 50  # coalesced -- a fast notebook cannot flood the queue
    assert progress[-1] == ("progress", 1, 50)  # but the final count is exact


def test_queue_pbar_survives_a_broken_queue():
    """A dead queue must never take the analysis down with it."""

    class _Broken:
        def put(self, event):
            raise BrokenPipeError("gone")

    bar = _QueuePBar(_Broken(), key=1, total=2)
    bar.update()
    bar.close()  # no exception


def test_reporting_engine_executes_and_reports(tmp_path):
    """The engine override must still run the notebook, and report cell progress."""
    script = _make_marker_script(tmp_path)
    out = tmp_path / "executed.ipynb"
    q = queue.Queue()

    pm.execute_notebook(
        str(script),
        str(out),
        parameters={"storage_folder": str(tmp_path), "image_id": "pos1"},
        cwd=str(tmp_path),
        kernel_name="python3",
        engine_name=register_engine(),
        progress_queue=q,
        progress_key=42,
    )

    # the notebook really ran ...
    assert (tmp_path / "done.txt").read_text() == "pos1"
    # ... and reported instead of drawing
    events = _drain(q)
    kinds = [event[0] for event in events]
    assert kinds[0] == "total"
    assert "progress" in kinds
    assert kinds[-1] == "stage_end"
    assert all(event[1] == 42 for event in events)


# --------------------------------------------------------------------------- #
# parent side
# --------------------------------------------------------------------------- #


def test_worker_bars_label_and_fill_per_worker():
    bars = _WorkerBars(total_sources=3, bar_factory=_StubBar)

    bars.on_event(("stage_start", 101, "pos1_roi2.tiff", "01_Segment.ipynb"))
    bars.on_event(("total", 101, 21))
    bars.on_event(("progress", 101, 5))
    bars.on_event(("stage_start", 202, "pos3_roi1.tiff", "01_Segment.ipynb"))

    first, second = bars._bars[101], bars._bars[202]
    assert first.desc == "  [w1] pos1_roi2.tiff | 01_Segment.ipynb"
    assert second.desc == "  [w2] pos3_roi1.tiff | 01_Segment.ipynb"
    assert (first.total, first.n) == (21, 5)


def test_worker_bars_reuse_a_slot_for_the_next_stage():
    """One bar per worker, not one per stage -- the display must not grow."""
    bars = _WorkerBars(total_sources=2, bar_factory=_StubBar)

    bars.on_event(("stage_start", 101, "pos1.tiff", "01_Segment.ipynb"))
    bars.on_event(("total", 101, 21))
    bars.on_event(("progress", 101, 21))
    bars.on_event(("stage_end", 101))
    bars.on_event(("stage_start", 101, "pos1.tiff", "02_Track.ipynb"))

    assert len(bars._bars) == 1
    bar = bars._bars[101]
    assert bar.desc == "  [w1] pos1.tiff | 02_Track.ipynb"
    # reset, and indeterminate again until the new cell count arrives
    assert bar.n == 0
    assert bar.total is None


def test_worker_bars_drain_applies_queued_events():
    bars = _WorkerBars(total_sources=1, bar_factory=_StubBar)
    q = queue.Queue()
    for event in [
        ("stage_start", 1, "pos1.tiff", "01_Segment.ipynb"),
        ("total", 1, 4),
        ("progress", 1, 3),
        ("something-else", 1),  # unknown kinds are ignored, not fatal
    ]:
        q.put(event)

    bars.drain(q)

    assert bars._bars[1].n == 3
    assert q.empty()


def test_worker_bars_advance_and_close():
    bars = _WorkerBars(total_sources=2, bar_factory=_StubBar)
    bars.on_event(("stage_start", 1, "pos1.tiff", "01_Segment.ipynb"))

    bars.advance_total("done pos1.tiff")
    bars.close()

    assert bars.total_bar.n == 1
    assert bars.total_bar.desc == "done pos1.tiff"
    assert bars.total_bar.closed
    assert bars._bars[1].closed


@pytest.mark.parametrize(
    ("mode", "leave"),
    [("keep", True), ("collapse", False)],
)
def test_worker_bars_mode_controls_whether_bars_are_left(mode, leave):
    bars = _WorkerBars(total_sources=1, mode=mode, bar_factory=_StubBar)
    bars.on_event(("stage_start", 1, "pos1.tiff", "01_Segment.ipynb"))

    assert bars._bars[1].kwargs["leave"] is leave
