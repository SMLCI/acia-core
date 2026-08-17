"""File-access capture behind :class:`acia.analysis.StageContext`.

A stage's manifest entry used to record what its author *remembered to declare* --
a hand-written ``artifacts`` list, no inputs at all. That drifts from reality the
moment someone adds an output or changes an input. This module records what the
filesystem **observed** instead, so the claim and the evidence cannot silently
disagree, and it does so without any notebook writing a single extra line.

Two mechanisms, each covering the other's blind spot:

* a **PEP 578 audit hook** (:func:`sys.addaudithook`) reports every ``open()`` with
  its path *and* mode -- this is how reads are seen, which no directory listing
  could show;
* a **snapshot/diff of the output folder** -- this is how writes by code Python
  cannot observe are seen (``cv2.VideoWriter`` and ffmpeg write the ``.mp4``
  artifacts without ever calling Python's ``open``).

Neither depends on paths coming from :meth:`StageContext.path`; capture keys off
*where* a file is and *when* it was touched, so ``f"{out}/x.npz"`` and
``ctx.path("x.npz")`` are identical here.

**Nothing in this module may ever raise into user code.** Provenance is a bonus;
an analysis that would have worked before must still work. Every entry point is
wrapped, and a failure degrades to "no record", never to an error.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: environment variable through which :func:`acia.analysis.scale` tells a kernel
#: which stage notebook it is executing (the kernel cannot know otherwise)
STAGE_NOTEBOOK_ENV = "ACIA_STAGE_NOTEBOOK"

#: modes that mean "this file is being produced" -- everything else is a read
_WRITE_MODES = set("wax+")

#: never recorded as data. A notebook is *code*: it is identified by
#: :func:`code_fingerprint` instead, and under :func:`acia.analysis.scale` the
#: executed copy is rewritten by papermill after a stage records, which would make
#: every stage permanently look stale.
_CODE_SUFFIXES = {".ipynb"}

#: the one mutable slot the process-global audit hook consults; ``None`` means
#: nothing is being recorded and the hook returns immediately
_CURRENT: _Recorder | None = None

#: whether the audit hook is installed. PEP 578 hooks cannot be removed, so a
#: notebook kernel re-running its setup cell must not stack them.
_ARMED = False


def _utc(timestamp: float | None = None) -> str:
    """UTC ISO-8601, seconds precision -- the format the manifest already uses."""
    when = (
        datetime.now(timezone.utc)
        if timestamp is None
        else datetime.fromtimestamp(timestamp, timezone.utc)
    )
    return when.isoformat(timespec="seconds")


def fingerprint(path: str | Path) -> dict[str, Any]:
    """A cheap ``(size, mtime)`` fingerprint of a file or folder.

    Deliberately not a content hash: a single image stack runs to gigabytes, so
    hashing every artifact would cost more than the analysis. This is a
    move/change *detector*, the same trade-off (and the same rationale) as the
    curation manifest's :func:`acia.selection._fingerprint`.

    A folder aggregates its contents -- ``tracking/`` is a directory artifact, and
    its identity is the sum of what it holds.
    """
    try:
        target = Path(path)
        if target.is_dir():
            files = [f for f in target.rglob("*") if f.is_file()]
            stats = [f.stat() for f in files]
            return {
                "size": sum(s.st_size for s in stats),
                "mtime": max((s.st_mtime for s in stats), default=0.0),
            }
        stat = target.stat()
        return {"size": int(stat.st_size), "mtime": float(stat.st_mtime)}
    except OSError:
        return {}


def code_fingerprint() -> dict[str, Any]:
    """Identity of the stage notebook being executed, or ``{}`` if unknown.

    Records *which version of the analysis ran*, which is what turns "these two
    populations disagree" into "they ran different code" instead of a guess.
    Hashing is affordable here precisely where it is not for data: a notebook is
    kilobytes.

    The notebook is located from the environment variable
    :data:`STAGE_NOTEBOOK_ENV` (exported by :func:`acia.analysis.scale`) or, when
    running interactively, ipykernel's ``JPY_SESSION_NAME``. If neither is
    available -- a plain ``python script.py``, say -- the block is **omitted**
    rather than guessed.
    """
    try:
        raw = os.environ.get(STAGE_NOTEBOOK_ENV) or os.environ.get("JPY_SESSION_NAME")
        if not raw:
            return {}
        notebook = Path(raw)
        info: dict[str, Any] = {"notebook": notebook.name}
        if notebook.is_file():
            info["mtime"] = _utc(notebook.stat().st_mtime)
            info["sha256"] = hashlib.sha256(notebook.read_bytes()).hexdigest()
        return info
    except Exception:
        logger.debug("could not fingerprint the stage notebook", exc_info=True)
        return {}


def env_info() -> dict[str, Any]:
    """The cheap facts that explain a result which will not reproduce elsewhere."""
    try:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": socket.gethostname(),
        }
    except Exception:  # pragma: no cover - platform calls are not expected to fail
        logger.debug("could not collect environment info", exc_info=True)
        return {}


def _is_write(mode: str | None) -> bool:
    return bool(set(mode or "") & _WRITE_MODES)


class _Recorder:
    """Accumulates one stage's observed reads and writes.

    ``roots`` is an allowlist of resolved directories. Without it the record would
    drown: a notebook opens thousands of files under ``site-packages``, matplotlib's
    font cache and the temp dir before it touches any data.

    Both sides of every root comparison are resolved, because on macOS ``/tmp`` is a
    symlink to ``/private/tmp`` and comparing one form against the other silently
    matches nothing.
    """

    def __init__(self, roots: list[Path]) -> None:
        self.roots = roots
        self.reads: set[Path] = set()
        self.writes: set[Path] = set()
        self.snapshot: dict[Path, tuple[int, float]] = {}
        self.depth = 0  # re-entrant track() regions

    def _in_scope(self, path: Path) -> bool:
        return any(root == path or root in path.parents for root in self.roots)

    def note(self, raw_path: Any, mode: Any) -> None:
        """Record one observed ``open()``. Called from the audit hook -- hot path."""
        path = Path(os.fsdecode(raw_path)).resolve()
        if path.suffix in _CODE_SUFFIXES or not self._in_scope(path):
            return
        (self.writes if _is_write(mode) else self.reads).add(path)

    def declare_read(self, path: str | Path) -> None:
        """Record a read that the hook cannot see (a C-level or remote reader)."""
        try:
            self.reads.add(Path(path).resolve())
        except Exception:
            logger.debug("could not declare read of %r", path, exc_info=True)

    # -- snapshot / diff ---------------------------------------------------

    def take_snapshot(self, output_dir: Path) -> None:
        """State of the output folder before this stage did anything."""
        self.snapshot = _dir_state(output_dir)

    def reset(self, output_dir: Path) -> None:
        """Start a fresh window -- a notebook may record more than one stage."""
        self.reads.clear()
        self.writes.clear()
        self.take_snapshot(output_dir)

    def apply_diff(self, output_dir: Path) -> None:
        """Everything new or changed in the output folder counts as written.

        This is what covers writers Python cannot observe, and it is the mechanism
        Sumatra has used for the same purpose for over a decade. It runs once per
        stage, at ``record()``, independently of any :meth:`StageContext.track`
        regions -- a file this stage produced is its output whether or not the
        author bracketed the code that produced it.
        """
        for path, state in _dir_state(output_dir).items():
            if path.suffix in _CODE_SUFFIXES:
                continue
            if self.snapshot.get(path) != state:
                self.writes.add(path)


def _dir_state(directory: Path) -> dict[Path, tuple[int, float]]:
    try:
        return {
            f.resolve(): (f.stat().st_size, f.stat().st_mtime)
            for f in Path(directory).rglob("*")
            if f.is_file()
        }
    except OSError:
        logger.debug("could not snapshot %s", directory, exc_info=True)
        return {}


# --------------------------------------------------------------------------- #
# the process-global hook
# --------------------------------------------------------------------------- #


def _audit_hook(event: str, args: tuple) -> None:
    """Route ``open`` events to whichever recorder is currently active.

    Must be as cheap as possible and must **never** raise: an exception here
    propagates out of the ``open()`` that triggered it, which would turn a
    provenance bug into a broken analysis.
    """
    if event != "open" or _CURRENT is None:
        return
    try:  # noqa: SIM105 - contextlib.suppress allocates; this runs on every open()
        _CURRENT.note(args[0], args[1])
    except Exception:
        pass


def arm() -> None:
    """Install the audit hook, at most once per process."""
    global _ARMED
    if _ARMED:
        return
    try:
        sys.addaudithook(_audit_hook)
        _ARMED = True
    except Exception:  # pragma: no cover - only on exotic interpreters
        logger.debug("could not install the file-access audit hook", exc_info=True)


def activate(recorder: _Recorder | None) -> None:
    """Point the hook at ``recorder`` (or at nothing)."""
    global _CURRENT
    _CURRENT = recorder


def current() -> _Recorder | None:
    return _CURRENT


class _TrackRegion:
    """Context manager turning read capture on for a region -- :meth:`StageContext.track`.

    Under the default (``track_io=True``) capture is already running for the whole
    stage, so a region is simply a no-op that costs nothing: it exists for the
    notebook built with ``track_io=False``, whose exploratory cells should not
    count and which opts back in around the parts that should.

    Re-entrant on purpose: a ``with`` block cannot span notebook cells, so a stage
    opens one per cell and everything accumulates into the next ``record()``.
    Restores rather than clears the previous state on exit, so a region nested in
    an already-capturing stage does not switch capture off behind its back.

    Only the *hook* is scoped. The output-folder diff runs once, at ``record()``.
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._previous: _Recorder | None = None

    def __enter__(self) -> Any:
        try:
            recorder = self._ctx._recorder
            if recorder is not None:
                if recorder.depth == 0:
                    self._previous = current()
                recorder.depth += 1
                arm()
                activate(recorder)
        except Exception:
            logger.debug("could not start a tracking region", exc_info=True)
        return self._ctx

    def __exit__(self, *exc: Any) -> None:
        try:
            recorder = self._ctx._recorder
            if recorder is not None:
                recorder.depth = max(0, recorder.depth - 1)
                if recorder.depth == 0:
                    activate(self._previous)
        except Exception:
            logger.debug("could not close a tracking region", exc_info=True)
