"""Progress reporting for :func:`acia.analysis.scale`.

``scale()`` runs notebooks with papermill, which draws its own per-cell ``tqdm``
bar. That works fine when the notebook runs in this process, but with
``max_workers > 1`` every notebook runs in a *spawned child*: ``tqdm.auto`` there
has no ipykernel, falls back to the text bar and ``\\r``-redraws into the stderr
that all children share. Jupyter's output area does not emulate a terminal
cursor, so those bars overwrite each other and the display turns into garbage.

The fix implemented here is that **children never draw**. They report what they
are doing over a queue and the parent -- the process that owns the display --
renders one *worker slot* bar per pool worker:

.. code-block:: text

    Sources:  33%|███       | 1/3 [04:41<09:23, 281.63s/source]
      [w1] pos001_roi001.tiff | 02_Track.ipynb:   57%|█████ | 12/21 [00:32<00:35, 3.91s/cell]
      [w2] pos002_roi001.tiff | 01_Segment.ipynb: 19%|██    |  4/21 [00:00<00:03, 4.88cell/s]

Three pieces make that work:

* :class:`_QueuePBar` -- a duck-typed stand-in for papermill's ``tqdm``, living in
  the child, that turns bar updates into queue events instead of output.
* :class:`_ReportingEngine` -- a papermill engine that installs a
  :class:`_QueuePBar` on the notebook execution manager.
* :class:`_WorkerBars` -- the parent-side renderer consuming those events.

Everything in this module is private to ``acia.analysis``.
"""

from __future__ import annotations

import logging
import time
from queue import Empty
from typing import Any

from papermill.engines import (
    NBClientEngine,
    NotebookExecutionManager,
    papermill_engines,
)
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

#: Name under which :class:`_ReportingEngine` is registered with papermill.
ENGINE_NAME = "acia-scale-progress"

#: Minimum seconds between two ``progress`` events of one child. Cell updates are
#: coalesced in between so a notebook with many fast cells cannot flood the queue.
_MIN_EVENT_INTERVAL = 0.05


# --------------------------------------------------------------------------- #
# child side
# --------------------------------------------------------------------------- #


class _QueuePBar:
    """Reports papermill's cell progress to the parent instead of drawing a bar.

    Duck-types the part of ``tqdm`` that ``NotebookExecutionManager`` uses:
    ``update()``, ``set_description()``, ``refresh()``, ``close()`` and a plain
    ``n`` attribute (papermill assigns to it directly when it force-completes the
    bar). Progress is reported as an *absolute* cell count, so a coalesced or lost
    event cannot desynchronise the parent's bar.

    ``key`` identifies the reporting process (``os.getpid()`` in practice); the
    parent maps it to a worker slot.
    """

    def __init__(self, queue: Any, key: Any, total: int | None) -> None:
        self._queue = queue
        self._key = key
        self.n = 0
        self.total = total
        self._last_sent = -1
        self._last_time = 0.0
        self._put(("total", key, total))

    def _put(self, event: tuple) -> None:
        try:
            self._queue.put(event)
        except Exception:  # a broken queue must never fail the analysis itself
            logger.debug("could not report progress event %r", event, exc_info=True)

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (
            self.n == self._last_sent or now - self._last_time < _MIN_EVENT_INTERVAL
        ):
            return
        self._last_sent = self.n
        self._last_time = now
        self._put(("progress", self._key, self.n))

    def update(self, n: int = 1) -> None:
        self.n += n
        self._emit()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        # papermill only sets a description for cells carrying the
        # `papermill_description=` escape string; the stage label the user cares
        # about is sent once by the caller, so this is deliberately ignored.
        pass

    def refresh(self) -> None:
        self._emit(force=True)

    def close(self) -> None:
        self._emit(force=True)
        self._put(("stage_end", self._key))


class _ReportingEngine(NBClientEngine):
    """papermill engine that reports cell progress over a queue.

    Mirrors ``papermill.engines.Engine.execute_notebook`` but builds the
    ``NotebookExecutionManager`` with ``progress_bar=False`` and installs a
    :class:`_QueuePBar` instead, so nothing is written to the child's stderr.
    ``progress_queue``/``progress_key`` are consumed here and deliberately *not*
    forwarded -- ``execute_managed_notebook`` passes its leftovers to nbclient,
    which would reject them.
    """

    @classmethod
    def execute_notebook(  # type: ignore[override]
        cls,
        nb,
        kernel_name,
        output_path=None,
        progress_bar=True,  # accepted for signature compatibility; never drawn
        log_output=False,
        autosave_cell_every=30,
        progress_queue=None,
        progress_key=None,
        **kwargs,
    ):
        nb_man = NotebookExecutionManager(
            nb,
            output_path=output_path,
            progress_bar=False,
            log_output=log_output,
            autosave_cell_every=autosave_cell_every,
        )
        if progress_queue is not None:
            nb_man.pbar = _QueuePBar(progress_queue, progress_key, len(nb.cells))

        nb_man.notebook_start()
        try:
            cls.execute_managed_notebook(
                nb_man, kernel_name, log_output=log_output, **kwargs
            )
        finally:
            nb_man.cleanup_pbar()
            nb_man.notebook_complete()

        return nb_man.nb


def register_engine() -> str:
    """Register :class:`_ReportingEngine` with papermill and return its name.

    Called from the worker rather than at import time so that importing
    ``acia.analysis`` has no side effect on papermill's engine registry.
    Registration is a dict assignment, so calling this repeatedly is harmless.
    """
    papermill_engines.register(ENGINE_NAME, _ReportingEngine)
    return ENGINE_NAME


# --------------------------------------------------------------------------- #
# parent side
# --------------------------------------------------------------------------- #


class _WorkerBars:
    """Renders the total bar plus one bar per pool worker, in the parent process.

    Slot bars are created lazily, the first time a worker reports in, and reused
    for every stage that worker runs afterwards -- so the display has at most
    ``max_workers`` of them regardless of how many sources are processed.

    ``mode`` is :func:`acia.analysis.scale`'s ``stage_progress``: ``"keep"``
    leaves the slot bars on screen at their final state, ``"collapse"`` removes
    them when the run ends.
    """

    #: minimum seconds between two full refreshes driven by :meth:`tick`
    _TICK_INTERVAL = 1.0

    def __init__(
        self,
        total_sources: int,
        mode: str = "keep",
        bar_factory: Any = tqdm,
    ) -> None:
        self._mode = mode
        self._bar_factory = bar_factory
        self._slots: dict[Any, int] = {}
        self._bars: dict[Any, Any] = {}
        self._last_tick = 0.0
        self.total_bar = bar_factory(
            total=total_sources, unit="source", desc="Sources", position=0
        )

    def _bar_for(self, key: Any) -> Any:
        """The slot bar of a worker, created on its first event."""
        if key not in self._bars:
            slot = self._slots.setdefault(key, len(self._slots))
            self._bars[key] = self._bar_factory(
                total=None,
                unit="cell",
                desc=f"  [w{slot + 1}]",
                position=slot + 1,
                leave=self._mode == "keep",
            )
        return self._bars[key]

    def on_event(self, event: tuple) -> None:
        """Apply one event from a worker to the display."""
        kind, key = event[0], event[1]
        bar = self._bar_for(key)

        if kind == "stage_start":
            _, _, label, stage = event
            bar.reset()
            # the cell count is only known once papermill has read the notebook;
            # until the ("total", ...) event arrives the bar just counts up
            bar.total = None
            bar.set_description(f"  [w{self._slots[key] + 1}] {label} | {stage}")
        elif kind == "total":
            bar.total = event[2]
            bar.refresh()
        elif kind == "progress":
            delta = event[2] - bar.n
            if delta > 0:
                bar.update(delta)
            elif delta < 0:  # a new stage started; bar.reset() may have been missed
                bar.n = event[2]
                bar.refresh()
        elif kind == "stage_end":
            # keep the finished stage visible until this worker starts the next one
            bar.refresh()
        else:
            logger.debug("ignoring unknown progress event %r", event)

    def drain(self, queue: Any) -> None:
        """Apply every event currently waiting on the queue."""
        while True:
            try:
                event = queue.get_nowait()
            except Empty:
                return
            except (EOFError, OSError, BrokenPipeError):
                logger.debug("progress queue is gone", exc_info=True)
                return
            self.on_event(event)

    def advance_total(self, description: str) -> None:
        """One more source is done (or failed)."""
        self.total_bar.set_description(description)
        self.total_bar.update(1)

    def tick(self) -> None:
        """Redraw so elapsed times keep moving while no event arrives."""
        now = time.monotonic()
        if now - self._last_tick < self._TICK_INTERVAL:
            return
        self._last_tick = now
        self.total_bar.refresh()
        for bar in self._bars.values():
            bar.refresh()

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()
        self.total_bar.close()
