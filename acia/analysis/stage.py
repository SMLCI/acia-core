"""Shared run context for a chain of analysis notebooks ("stages").

A staged workflow splits one analysis into several notebooks (segment -> track ->
measure) that hand their results over as **files in a shared output folder**, one
folder per imaged population. Every stage therefore has to answer the same three
questions before it can do anything: where do results go, which population is this,
and did the stage I depend on actually run here?

:class:`StageContext` answers them once:

.. code-block:: python

    from acia.analysis import StageContext

    ctx = StageContext.for_image(image_id, "./output", stage="Track")

    seg = ctx.input_path("segmentation.npz")            # fail early, actionably
    ctx.log_params(mode="greedy")                       # on disk immediately
    write_units_csv(ctx.keyed(props), ctx.output_path("cell_properties.csv"))
    ctx.log_metrics(n_tracklets=len(graph))
    ctx.finish()

    ctx                                                 # -> the folder, rendered

Naming the stage up front is what lets the record be written *as the stage runs*.
Everything logged is on disk the moment it happens, so a notebook that dies
half-way still says what it was doing and under which settings -- which is
exactly the run whose record is wanted, and the one that used to record nothing.
An unfinished stage is marked ``"running"`` rather than passing for a complete
one. A context built without a ``stage=`` name behaves as it always did: nothing
is written until :meth:`~StageContext.record`.

The recorded ``stage_manifest.json`` makes a finished run self-describing -- which
stages ran, what each produced and under which settings -- and is what
:func:`acia.analysis.scale` batches read back to summarise a fan-out. Stages
*append* to it, so a chain that grows a stage later keeps the earlier entries.

Beyond what a stage states, the context records what it **did**: the files it was
observed to read and write, when it ran, and a digest of the notebook that ran (see
:mod:`acia.analysis._stage_io`). Three things follow, none of which a notebook has to
ask for:

* the dependency graph is *derived* -- tracking read the file segmentation wrote, so
  there is an edge, and it cannot fall out of date the way a written-down one does
  (:func:`stage_graph`);
* a stage whose input has changed since it ran says so (:func:`check_stale`), which is
  the one thing a folder of results could not tell you before -- re-segmenting leaves
  stale tracking output looking perfectly current;
* a whole batch becomes searchable as one table (:func:`stage_table`).

Provenance is a bonus and never a risk: if capture fails, ``record()`` writes exactly
the manifest it would have written without it.
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import html
import json
import logging
import os
import re
import shutil
import time
import warnings
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from acia import __version__
from acia.analysis import _stage_io
from acia.analysis.units import UNIT_ATTR

logger = logging.getLogger(__name__)

#: file name of the per-population manifest inside the output folder
MANIFEST_NAME = "stage_manifest.json"

#: schema of the recorded-I/O block inside a stage entry, following the same
#: convention as ``acia.selection/v1`` and ``acia.registration/v1``
IO_SCHEMA = "acia.stage_io/v1"

#: default identity pattern: ``pos001_roi002`` -> ``position=1, roi=2``. It is only
#: a naming convention -- pass your own ``key_pattern`` (named groups) or ``None``
#: for sources that don't follow it; unmatched names simply become ``None``.
DEFAULT_KEY_PATTERN = r"pos(?P<position>\d+)_roi(?P<roi>\d+)"

#: schema of one stage entry. Versioned on the entry rather than the manifest,
#: like :data:`IO_SCHEMA`, because one folder legitimately holds entries written
#: by different acia versions. An entry with no ``schema`` key is a v1 entry.
STAGE_SCHEMA = "acia.stage/v2"

#: schema of the recorded calibration block inside a stage entry
CALIBRATION_SCHEMA = "acia.calibration/v1"

#: most figure thumbnails inlined into the HTML view; beyond this they render as
#: text. A notebook is saved with its outputs, so every inlined byte is paid for
#: on every open.
_MAX_THUMBNAILS = 6

#: longest side of an inlined thumbnail, in pixels
_THUMBNAIL_PX = 320.0

#: shortest gap between two folder diffs, in seconds (see :meth:`StageContext._io_due`)
_IO_MIN_INTERVAL = 2.0

#: how long a run may sit idle before the exit hook stops trusting its folder
#: snapshot. An interactive kernel lives for hours; diffing against a snapshot
#: taken that long ago would attribute every file written since -- by any
#: notebook -- to this stage.
_ATEXIT_STALE_S = 900.0

#: stage runs that have not been finished, by ``id(run)``. Module-level, so a
#: notebook that rebuilds its context twenty times does not stack twenty exit
#: hooks; entries are popped by :meth:`StageContext.finish`.
_UNFINISHED: dict[int, tuple[weakref.ref, Any]] = {}


def _flush_unfinished() -> None:
    """Last-chance persist for stages still open at interpreter exit.

    A safety net, not the main path. The cheap parts of an entry are already on
    disk -- they are written as they happen, which is the point of logging as you
    go -- and the only thing deferred is the folder diff, because it rglobs the
    whole output folder and a CTC ``tracking/`` folder holds one mask per frame.

    A run that has been idle for :data:`_ATEXIT_STALE_S` gets its entry written
    *without* that diff: under a long-lived Jupyter kernel this hook can fire
    hours after the stage stopped, and by then the snapshot it would diff against
    describes a folder that several other notebooks have written to.
    """
    for ref, run in list(_UNFINISHED.values()):
        ctx = ref()
        if ctx is None:
            continue
        with contextlib.suppress(Exception):
            idle = time.monotonic() - (run.last_persist or 0.0)
            ctx._persist(collect_io=idle <= _ATEXIT_STALE_S)


atexit.register(_flush_unfinished)


def _get_ipython() -> Any:
    """The active IPython shell, or ``None`` outside one."""
    try:
        from IPython import get_ipython  # noqa: PLC0415 -- optional, and only here

        return get_ipython()
    except Exception:  # pragma: no cover - IPython absent or misbehaving
        return None


def _register_autoflush(ctx: StageContext) -> None:
    """Persist this context's stage at every cell boundary, and at exit.

    ``post_run_cell`` is the primary trigger rather than ``atexit`` because it is
    the only one that coincides with the *stage*: IPython fires it after every
    cell including one that raised, and swallows a callback's exceptions. An exit
    hook, by contrast, fires when the kernel dies -- which interactively may be
    hours later, or never.
    """
    run = ctx._run
    if run is None:
        return
    _UNFINISHED[id(run)] = (weakref.ref(ctx), run)

    shell = _get_ipython()
    if shell is None:
        return  # a plain interpreter: the exit hook is all there is
    ref = weakref.ref(ctx)

    def _on_cell(result: Any = None) -> None:
        target = ref()
        if target is None or target._run is None or target._run.finalized:
            return
        with contextlib.suppress(Exception):
            target._note_cell(result)

    with contextlib.suppress(Exception):
        shell.events.register("post_run_cell", _on_cell)
        object.__setattr__(ctx, "_cell_hook", _on_cell)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Replace a manifest atomically.

    ``Path.write_text`` truncates before it writes. With one write per stage that
    window was academic; logging as you go makes it a hundred times wider, and a
    truncated ``stage_manifest.json`` is not a lost record but a poisoned folder --
    every later :meth:`StageContext.for_image` reads it while resolving the source,
    so the next run would die on a ``JSONDecodeError`` instead of just missing an
    entry. The temp file is named per-process so two writers cannot collide.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class _StageRun:
    """Mutable accumulators for the stage in progress.

    Kept off the frozen :class:`StageContext` (a value type) and off
    ``_recorder`` (which is ``None`` whenever capture fails -- while the whole
    point of :meth:`StageContext.log_params` is that it survives that).
    """

    stage: str | None = None
    started_at: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    declared_inputs: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] | None = None
    status: str = "running"
    error: dict[str, str] | None = None
    #: ``(name, data-uri)`` thumbnails, in memory only -- never in the manifest
    thumbnails: list[tuple[str, str]] = field(default_factory=list)
    finalized: bool = False
    #: whether *this* run has collected its own observed I/O yet
    io_collected: bool = False
    last_io_at: float = 0.0
    last_io_cost: float = 0.0
    last_persist: float = 0.0


def _jsonable(value: Any) -> Any:
    """A JSON-serialisable view of a recorded value.

    Settings arrive as pint quantities, paths and numpy scalars at least as often
    as plain numbers. A manifest that records ``"0.065 micrometer"`` as a string is
    worth far more than one that could not be written at all.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalar -> python scalar
    if callable(item):
        with contextlib.suppress(Exception):
            return _jsonable(item())
    return str(value)


def population_id_of(path: str | Path) -> str:
    """Identity of one imaged population, derived from its source path.

    A **folder** source (one image per timepoint) keeps its full name; a **file**
    source drops the extension. Stripping a ``.tiff``-like suffix only makes sense
    for a file -- folder names may legitimately contain dots, which a blind
    ``.stem`` would truncate.
    """
    path = Path(path)
    return path.name if path.is_dir() else path.stem


def _parse_keys(population_id: str, key_pattern: str | None) -> dict[str, Any]:
    """Named groups of ``key_pattern`` matched against ``population_id``.

    All-digit captures become ``int``; a pattern that doesn't match still yields
    every key, with value ``None`` (the population stays usable, it just has no
    position/roi to group by).
    """
    if not key_pattern:
        return {}
    regex = re.compile(key_pattern)
    if not regex.groupindex:
        raise ValueError(
            f"key_pattern {key_pattern!r} has no named groups -- it cannot name the "
            r"keys it extracts. Use e.g. r'pos(?P<position>\d+)_roi(?P<roi>\d+)'."
        )
    match = regex.match(population_id)
    keys: dict[str, Any] = {}
    for name in regex.groupindex:
        raw = match.group(name) if match else None
        keys[name] = int(raw) if raw is not None and raw.isdigit() else raw
    return keys


def _recorded_source(output_dir: Path) -> str | None:
    """The source an existing manifest says this folder belongs to, if any.

    Prefers the path *relative* to the output folder when it resolves: the absolute
    one breaks the moment an execution folder moves between machines (a run on a GPU
    node, analysed on a laptop), while the relative one survives the move.
    """
    population = read_manifest(output_dir).get("population") or {}
    relative = population.get("image_id_relative")
    if relative:
        candidate = (Path(output_dir) / relative).resolve()
        if candidate.exists():
            return str(candidate)
    absolute = population.get("image_id")
    return str(absolute) if absolute else None


def _resolve_source(image_id: str | Path | None, output_dir: Path) -> str | Path:
    """The source for this context: the one given, or the one already recorded.

    Passing a source that disagrees with the folder's record warns rather than
    fails -- it is very likely a mistake (two populations' results landing in one
    folder), but the caller may genuinely be re-pointing a folder, and this library
    warns rather than blocks, as the curation manifest does for its fingerprints.
    """
    recorded = _recorded_source(output_dir)
    if image_id is None:
        if recorded is None:
            ran = stages_run(output_dir)
            raise ValueError(
                f"No source given and {Path(output_dir).resolve()} records none yet "
                f"(stages recorded here: {ran or 'none'}). The first stage of a chain "
                "must be told its image_id; later ones can recover it from the folder."
            )
        return recorded

    if recorded is not None and Path(recorded).name != Path(image_id).name:
        warnings.warn(
            f"{Path(output_dir).resolve()} records {Path(recorded).name!r} as its "
            f"source, but {Path(image_id).name!r} was passed -- results from two "
            "populations would be mixed in one folder.",
            stacklevel=3,
        )
    return image_id


def _warn_if_stale(output_dir: Path) -> None:
    for entry in check_stale(output_dir):
        warnings.warn(
            f"{entry['stage']} may be stale -- its input {entry['path']!r} changed "
            f"after it ran (recorded {entry['recorded']}, file modified "
            f"{entry['actual']}).",
            stacklevel=4,
        )


def _warn_on_producer_mismatch(
    output_dir: Path, name: str | Path, produced_by: str
) -> None:
    """Warn when a stated producer disagrees with what the folder recorded.

    ``produced_by`` used to be consulted only when the artifact was *missing*, to
    say what to run. But a hint that is wrong while the file happens to exist is
    the more dangerous case: it reads as a checked dependency and is not one. The
    manifest already knows who wrote what, so the claim can simply be verified --
    the same declared-versus-observed check that ``io.missing`` applies to outputs.

    It names a **stage**, matched exactly against the manifest's keys. Accepting
    the notebook file name too would be worse than strict: a stage named
    ``"Segment"`` in ``01_Segment.ipynb`` would then match under
    :func:`~acia.analysis.scale` (which records the notebook) and not match
    interactively, so a wrong hint would pass in one place and warn in the other.

    Reports, never decides: the recorded producer stays *derived* from what stages
    actually wrote, and this never changes which file is returned.
    """
    manifest = read_manifest(output_dir)
    stages = manifest.get("stages") or {}
    if not stages:
        # nothing has ever run here, so there is nothing to check the claim
        # against -- the artifact was put here by other means, which is the
        # caller's business
        return

    if produced_by not in stages:
        if Path(produced_by).suffix == ".ipynb":
            # a notebook file name where a stage name belongs: point at the fix
            # rather than reporting it as an unknown stage
            warnings.warn(
                f"produced_by names a notebook ({produced_by!r}), but it takes the "
                f"stage name -- the name passed to StageContext.for_image(stage=...). "
                f"Stages recorded here: {', '.join(stages)}.",
                stacklevel=3,
            )
            return
        warnings.warn(
            f"{str(name)!r} is said to be produced by stage {produced_by!r}, but no "
            f"such stage has run in this folder -- recorded here: "
            f"{', '.join(stages)}. Either the name is wrong, or this is not the "
            "folder you meant.",
            stacklevel=3,
        )
        return

    recorded = _producers(manifest).get(str(name))
    if recorded is None:
        # the artifact has no recorded producer (capture off, or written before
        # the stage that claims it). The named stage did run, so there is nothing
        # solid enough to contradict -- provenance is a bonus, not a gate
        return
    if recorded != produced_by:
        warnings.warn(
            f"{str(name)!r} is said to be produced by stage {produced_by!r}, but "
            f"this folder records it as written by stage {recorded!r}. The stage "
            "this input actually depends on is the one that wrote it.",
            stacklevel=3,
        )


def _warn_on_rerun(output_dir: Path, stage: str) -> None:
    """Warn that a stage already recorded here is being run again.

    Replacing the entry is the right thing -- a stage is re-run because its
    settings changed, and the folder should describe the run that produced the
    files now in it. What is worth saying out loud is the part that is *not*
    visible from this notebook: the stages downstream of this one still describe
    results computed from the previous version of these outputs, so they are
    stale until re-run. :func:`check_stale` will say so on their next run, but
    that is too late to be useful when the decision to re-run is being made here.
    """
    entry = (read_manifest(output_dir).get("stages") or {}).get(stage)
    if not entry:
        return

    status = entry.get("status", "ok")
    if status != "ok":
        # a previous attempt that never finished: nothing downstream can have
        # consumed it, so there is no staleness to report -- but silently
        # resuming over a failed run hides that it failed
        warnings.warn(
            f"stage {stage!r} was started in this folder before and did not finish "
            f"(status {status!r}); its entry will be replaced by this run.",
            stacklevel=3,
        )
        return

    when = entry.get("finished_at")
    ran = f" (finished {when})" if when else ""
    downstream = sorted(
        {d for upstream, _, d in stage_graph(output_dir) if upstream == stage}
    )
    consequence = (
        f" Stages that read its results are now out of date until they are re-run "
        f"too: {', '.join(downstream)}."
        if downstream
        else ""
    )
    warnings.warn(
        f"stage {stage!r} already ran in this folder{ran}; this run replaces its "
        f"entry.{consequence}",
        stacklevel=3,
    )


@dataclass(frozen=True)
class StageContext:
    """Where one stage writes, which population it is on, and what it may read.

    Build it with :meth:`for_image`; the fields are derived, not meant to be
    assembled by hand.

    Attributes:
        image_id: the source this run analyses, exactly as it was passed in.
        output_dir: the folder every stage of this population reads and writes.
        population_id: identity of the population (see :func:`population_id_of`).
        keys: the columns that key this population's rows in exported tables --
            ``population_id`` plus whatever ``key_pattern`` extracted (by default
            ``position`` and ``roi``). Splat it into a table: ``df.assign(**ctx.keys)``.
        stage: name of the stage this context is running, when it was given one.
            Naming it up front is what lets the record be written *as it happens*
            rather than in one call at the end -- see :meth:`log_params`.
    """

    image_id: str
    output_dir: Path
    population_id: str
    keys: Mapping[str, Any]
    stage_name: str | None = None
    #: observed reads/writes of the stage currently in progress; ``None`` when
    #: capture is off. Private -- see :meth:`track` and :meth:`record`.
    _recorder: Any = field(default=None, compare=False, repr=False)
    #: when the stage in progress started, UTC ISO-8601
    _started_at: str | None = field(default=None, compare=False, repr=False)
    #: mutable accumulators for the stage in progress (``_StageRun``), or ``None``
    #: when the context was built without a stage name. Deliberately not on
    #: ``_recorder``, which is ``None`` whenever capture fails -- logged parameters
    #: have to survive exactly that.
    _run: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def for_image(
        cls,
        image_id: str | Path | None = None,
        output_folder: str | Path = "./output",
        stage: str | None = None,
        *,
        key_pattern: str | None = DEFAULT_KEY_PATTERN,
        create: bool = True,
        track_io: bool = True,
        track_roots: Iterable[str | Path] = (),
    ) -> StageContext:
        """Resolve the run context for one source.

        Args:
            image_id: path to the movie this run analyses -- a single stack file or
                a folder of per-timepoint images. **Optional**: a folder that already
                holds a manifest knows its own source, so a downstream stage can be
                re-run there without being told it again. Passing it anyway is
                *verified* against what the folder records, and warns on a mismatch --
                running the wrong source in an existing folder mixes two populations
                into one set of results.
            output_folder: where this population's artifacts live. A **relative**
                value resolves against the working directory, which is what makes a
                stage chain work unchanged both interactively (the notebook folder)
                and under :func:`acia.analysis.scale` (each population's own
                execution folder).
            stage: name of the stage about to run, e.g. ``"02_Track"``. Naming it
                here opens its manifest entry immediately, so everything logged
                afterwards (:meth:`log_params`, :meth:`log_metrics`,
                :meth:`log_figure`) is on disk the moment it happens rather than
                waiting for a call at the end that a failing notebook never
                reaches. Omitted, the context behaves exactly as before and the
                stage is named by the deprecated :meth:`record`.
            key_pattern: regex with named groups, matched against the population id
                to derive grouping keys. ``None`` disables key extraction.
            create: create ``output_folder`` if it does not exist.
            track_io: record the files this stage reads and writes (see
                :meth:`record`). On by default and free -- notebooks need no change.
                Set it to ``False`` to capture only inside explicit :meth:`track`
                regions, e.g. when exploratory cells should not count.
            track_roots: extra directories whose reads should be recorded. Reads are
                otherwise limited to the working directory, the output folder and the
                source, without which a notebook's thousands of ``site-packages``
                opens would drown the record.
        """
        output_dir = Path(output_folder)
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)

        image_id = _resolve_source(image_id, output_dir)
        population_id = population_id_of(image_id)

        ctx = cls(
            image_id=str(image_id),
            output_dir=output_dir,
            population_id=population_id,
            keys={
                "population_id": population_id,
                **_parse_keys(population_id, key_pattern),
            },
            stage_name=stage,
            _started_at=_stage_io._utc(),
        )
        ctx._begin(track_io=track_io, track_roots=track_roots)
        _warn_if_stale(output_dir)
        if stage is not None:
            _warn_on_rerun(output_dir, stage)
            # Nothing is written here on purpose: a folder that has only *built* a
            # context has not run a stage, and read_manifest() must still be empty.
            # The first write is the first log_*/cell boundary/finish.
            object.__setattr__(
                ctx, "_run", _StageRun(stage=stage, started_at=ctx._started_at)
            )
            injected = _stage_io.injected_params()
            if injected:
                # what scale() actually parameterised this run with, recorded
                # without the notebook having to repeat its parameter cell
                ctx._run.params.update(_jsonable(injected))
            _register_autoflush(ctx)
        return ctx

    # -- capture -----------------------------------------------------------

    def _begin(self, *, track_io: bool, track_roots: Iterable[str | Path]) -> None:
        """Start a fresh capture window. Never raises -- provenance is a bonus."""
        try:
            roots = [Path.cwd().resolve(), self.output_dir.resolve()]
            source = Path(self.image_id).resolve()
            roots.append(source if source.is_dir() else source.parent)
            roots += [Path(r).resolve() for r in track_roots]

            recorder = _stage_io._Recorder(roots)
            recorder.take_snapshot(self.output_dir)
            object.__setattr__(self, "_recorder", recorder)
            if track_io:
                _stage_io.arm()
                _stage_io.activate(recorder)
        except Exception:
            logger.debug("could not start I/O capture", exc_info=True)
            object.__setattr__(self, "_recorder", None)

    def track(self) -> Any:
        """Capture reads inside this block -- **advanced, rarely needed**.

        With the default ``track_io=True`` the whole stage is already captured and
        this is a no-op. It earns its keep with ``track_io=False``, where nothing is
        recorded except inside these regions -- the way to keep exploratory cells out
        of a stage's record:

        .. code-block:: python

            ctx = StageContext.for_image(image_id, "./output", track_io=False)
            ...                                      # scratch work, not recorded
            with ctx.track():
                overlay = segment(source)            # this is the real analysis

        Re-entrant, because a ``with`` block cannot span notebook cells: open one per
        cell and everything accumulates into the next :meth:`record`.
        """
        return _stage_io._TrackRegion(self)

    # -- artifacts ---------------------------------------------------------

    def path(self, name: str | Path) -> Path:
        """Path of an artifact in this population's output folder."""
        return self.output_dir / name

    def has(self, name: str | Path) -> bool:
        """Whether an artifact is already present (e.g. an optional upstream stage)."""
        return self.path(name).exists()

    def require(self, name: str | Path, produced_by: str | None = None) -> Path:
        """Path of an artifact that must exist, or an actionable error.

        Args:
            name: artifact inside this population's output folder.
            produced_by: **optional** hint naming the notebook that makes it. The
                producer recorded in the manifest is always *derived* from what the
                earlier stages actually wrote, never from this string -- but the
                error path cannot use that, because a missing artifact usually means
                the producing stage never ran here and the manifest cannot know
                either. So the message degrades by what is genuinely known: the
                recorded producer, else which stages *did* run here (which
                distinguishes "wrong folder" from "not run yet"), else this hint.
        """
        return self.input_path(name, produced_by)

    def open(self, name: str | Path, mode: str = "r", **kwargs: Any) -> Any:
        """Open an artifact in this population's output folder.

        A convenience only -- it joins the path and notes the access. It is **not**
        how tracking works and is never required: almost nothing in this stack takes
        a file handle (``np.savez``, ``pd.read_csv``, ``tifffile.imwrite`` and
        ``cv2.VideoWriter`` all take *paths*), so capture watches the filesystem
        rather than handles.
        """
        target = self.path(name)
        recorder = self._recorder
        if recorder is not None and "r" in mode and "+" not in mode:
            recorder.declare_read(target)
        return open(target, mode, **kwargs)

    def input_path(self, name: str | Path, produced_by: str | None = None) -> Path:
        """Path of an artifact this stage reads, which must already exist.

        The reading half of the pair with :meth:`output_path`. Beyond what
        :meth:`require` has always done -- failing early with a message that says
        which stage makes the file -- it records the name as a declared input, so
        the stage says what it *meant* to read next to what it was observed to
        read. Direction itself is still observed, not taken from this call.

        Args:
            name: artifact inside this population's output folder.
            produced_by: optional name of the **stage** that makes it -- the name
                passed to :meth:`for_image` as ``stage=``, not the notebook file.
                It is *checked*: naming a stage that never ran here, or one that
                did not write this file, warns. It never decides anything -- the
                producer recorded in the manifest stays derived from what stages
                actually wrote -- and it still shapes the error when the artifact
                is missing (see :meth:`require`).
        """
        target = self.path(name)
        if not target.exists():
            raise FileNotFoundError(_missing_artifact_message(self, name, produced_by))
        if produced_by is not None:
            _warn_on_producer_mismatch(self.output_dir, name, produced_by)
        recorder = self._recorder
        if recorder is not None:
            # explicit, so it survives readers the audit hook cannot see -- anything
            # going through OpenCV's C++ layer, or a remote fsspec/OMERO source
            recorder.declare_read(target)
        run = self._run
        if run is not None:
            run.declared_inputs.append(str(name))
            self._persist(collect_io=False)
        return target

    def output_path(self, name: str | Path, *, parents: bool = True) -> Path:
        """Path of an artifact this stage writes, with its folder ready.

        The writing half of the pair with :meth:`input_path`. Creates the parent
        directory and records the name as a declared output, which is then checked
        against what was actually written -- a name declared here but never written
        shows up in the manifest's ``io.missing``, which is usually a typo.

        Nothing is created for the file itself, and the path is an ordinary
        :class:`~pathlib.Path`: write to it with whatever writer you already use.
        """
        target = self.path(name)
        try:
            resolved = target.resolve()
            root = self.output_dir.resolve()
            if not resolved.is_relative_to(root):
                warnings.warn(
                    f"{name!r} resolves outside this population's output folder "
                    f"({resolved} not under {root}); a stage chain hands results "
                    "over through that one folder, so writing elsewhere hides the "
                    "artifact from the next stage.",
                    stacklevel=2,
                )
        except (OSError, ValueError):  # pragma: no cover - exotic paths
            pass
        if parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        run = self._run
        if run is not None:
            run.artifacts.append(str(name))
            self._persist(collect_io=False)
        return target

    # -- recording as you go -----------------------------------------------

    def _require_run(self, what: str) -> _StageRun:
        run: _StageRun | None = self._run
        if run is None or run.stage is None:
            raise ValueError(
                f"{what} needs to know which stage it belongs to -- build the "
                "context with StageContext.for_image(..., stage='Segment')."
            )
        return run

    def log_params(self, **params: Any) -> None:
        """Record the settings this stage is running with, right away.

        Parameters are the part of a run that cannot be recovered by re-running it,
        so they are written to the manifest the moment they are known rather than
        at the end -- a notebook that dies half-way still says what it was doing.
        Values that are not JSON-serialisable (pint quantities, paths, numpy
        scalars) are stored as their string form.
        """
        run = self._require_run("log_params()")
        run.params.update(_jsonable(params))
        self._persist(collect_io=False)

    def log_metrics(self, **metrics: Any) -> None:
        """Record numbers this stage obtained -- counts, rates, scores.

        Same immediacy as :meth:`log_params`; the split is only about meaning, and
        both flatten into :func:`stage_table` columns.
        """
        run = self._require_run("log_metrics()")
        run.metrics.update(_jsonable(metrics))
        self._persist(collect_io=False)

    def log_artifact(
        self,
        path: str | Path,
        *,
        kind: str | None = None,
        caption: str | None = None,
    ) -> Path:
        """Note that a file belongs to this stage, optionally describing it.

        The observed writes already catch the file; this adds the intent and the
        description a bare filename cannot carry.
        """
        run = self._require_run("log_artifact()")
        name = str(path)
        run.artifacts.append(name)
        if kind or caption:
            run.extra.setdefault("artifact_meta", {})[name] = {
                key: value
                for key, value in (("kind", kind), ("caption", caption))
                if value is not None
            }
        self._persist(collect_io=False)
        return self.path(path)

    def log_figure(
        self,
        figure: Any,
        name: str,
        *,
        caption: str | None = None,
        dpi: int = 150,
        subfolder: str = "figures",
    ) -> Path:
        """Save a figure into the output folder and record it with its caption.

        A bare ``savefig`` lands in the folder anonymously: the diff notices a new
        file and nothing says what it shows or which step produced it. This writes
        it, records it in order with its caption, and keeps a small thumbnail for
        the context's own display.

        Args:
            figure: a matplotlib ``Figure`` (anything with ``savefig``), or the path
                of an image already written.
            name: file name; ``.png`` is appended when it has no suffix.
            caption: what the figure shows.
            dpi: resolution for the saved file (not the thumbnail).
            subfolder: where figures go inside the output folder.

        Returns:
            The path written.
        """
        run = self._require_run("log_figure()")
        if not Path(name).suffix:
            name = f"{name}.png"
        target = self.output_path(f"{subfolder}/{name}" if subfolder else name)

        saver = getattr(figure, "savefig", None)
        if callable(saver):
            saver(target, dpi=dpi, bbox_inches="tight")
        else:
            source = Path(figure)
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)

        entry = {
            "name": Path(name).stem,
            "path": _as_recorded_path(target.resolve(), self.output_dir),
            "logged_at": _stage_io._utc(),
        }
        if caption:
            entry["caption"] = caption
        run.figures = [f for f in run.figures if f["path"] != entry["path"]]
        run.figures.append(entry)
        self._capture_thumbnail(run, figure, entry)
        self._persist(collect_io=False)
        return target

    def _capture_thumbnail(self, run: _StageRun, figure: Any, entry: dict) -> None:
        """Keep a small in-memory PNG of a figure for :meth:`_repr_html_`.

        Captured from the live figure rather than re-read from the file on
        display. Reading it back would be an *observed read* of a file this stage
        wrote -- and, for a later stage, of a file an earlier one wrote, which
        would grow a spurious edge in :func:`stage_graph`.
        """
        if len(run.thumbnails) >= _MAX_THUMBNAILS:
            return
        saver = getattr(figure, "savefig", None)
        if not callable(saver):
            return
        with contextlib.suppress(Exception):
            import io as _io  # noqa: PLC0415 -- only needed here

            buffer = _io.BytesIO()
            size = max(figure.get_size_inches())
            saver(
                buffer,
                format="png",
                dpi=max(1.0, _THUMBNAIL_PX / float(size)),
                bbox_inches="tight",
            )
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            run.thumbnails.append((entry["path"], f"data:image/png;base64,{encoded}"))

    # -- manifest ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        """Path of this population's ``stage_manifest.json``."""
        return self.path(MANIFEST_NAME)

    def manifest(self) -> dict[str, Any]:
        """The manifest as recorded so far (``{}`` before the first stage)."""
        return read_manifest(self.output_dir)

    def _run_or_open(self, stage: str | None = None) -> _StageRun:
        """The stage in progress, opening one if the context was built without."""
        run: _StageRun | None = self._run
        if run is None:
            run = _StageRun(stage=stage or self.stage_name, started_at=self._started_at)
            object.__setattr__(self, "_run", run)
        if stage is not None:
            run.stage = stage
        return run

    def _io_due(self) -> bool:
        """Whether the folder diff has earned another run.

        The diff rglobs the whole output folder, so its cost scales with the
        folder rather than with what changed -- a CTC ``tracking/`` folder holds
        one mask per frame. Rather than a fixed interval, the gap is ten times the
        last diff's own cost, which caps it at roughly a tenth of runtime whether
        the folder is small and local or large and on a share.
        """
        run: _StageRun | None = self._run
        if run is None:
            return False
        gap = max(_IO_MIN_INTERVAL, 10.0 * run.last_io_cost)
        return bool((time.monotonic() - run.last_io_at) >= gap)

    def _persist(self, *, collect_io: bool, status: str | None = None) -> Path | None:
        """Write the stage in progress into the manifest. The single write path.

        Cheap by design: everything but the folder diff is dict work plus one
        small atomic write, so it can run at every cell boundary. ``collect_io``
        adds the diff, which the caller schedules (see :meth:`_io_due`).
        """
        run = self._run
        if run is None or run.stage is None:
            return None

        manifest = self.manifest()
        population = dict(self.keys, image_id=str(Path(self.image_id).resolve()))
        relative = _relative_source(self.image_id, self.output_dir)
        if relative is not None:
            population["image_id_relative"] = relative
        manifest.setdefault("population", {}).update(population)

        previous = dict((manifest.get("stages") or {}).get(run.stage) or {})
        if status is not None:
            run.status = status

        entry: dict[str, Any] = {
            "schema": STAGE_SCHEMA,
            "status": run.status,
            "artifacts": list(dict.fromkeys(run.artifacts)),
            "acia_version": __version__,
            # legacy record(**extra) keys stay flat at the top of the entry: that
            # is what ctx.stage("Segment")["pixel_size"] and stage_table's
            # settings-as-columns both read
            **_jsonable(run.extra),
        }
        if run.params:
            entry["params"] = run.params
        if run.metrics:
            entry["metrics"] = run.metrics
        if run.figures:
            entry["figures"] = run.figures
        if run.calibration:
            entry["calibration"] = run.calibration
        if run.error:
            entry["error"] = run.error
        if run.declared_inputs:
            entry["declared_inputs"] = sorted(set(run.declared_inputs))

        started = run.started_at or self._started_at
        if started:
            entry["started_at"] = started
        if run.finalized:
            entry["finished_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            if started:
                entry["duration_s"] = _elapsed_since(started)

        code = _stage_io.code_fingerprint()
        if code:
            entry["code"] = code
        env = _stage_io.env_info()
        if env:
            entry["env"] = env

        # An earlier persist *of this run* may already have collected io, and a cheap
        # one must not drop it. A previous *run* of the same stage is different: its
        # outputs describe work this run has not done, and carrying them would show a
        # half-old, half-new entry as if it were one complete run.
        io = dict(previous.get("io") or {}) if run.io_collected else {}
        if collect_io:
            began = time.monotonic()
            fresh = self._collect_io(manifest, entry["artifacts"])
            run.last_io_cost = time.monotonic() - began
            run.last_io_at = time.monotonic()
            if fresh is not None:
                io.update(fresh)
                run.io_collected = True
        if io:
            entry["io"] = io

        manifest.setdefault("stages", {})[run.stage] = entry
        _write_manifest(self.manifest_path, manifest)
        run.last_persist = time.monotonic()
        return self.manifest_path

    def _note_cell(self, result: Any = None) -> None:
        """Persist at a notebook cell boundary, marking a raising cell failed.

        IPython fires ``post_run_cell`` from a ``finally``, so this runs for a cell
        that raised too -- which is the case that used to record nothing at all. A
        later successful cell clears the mark, so an interactive fix-and-re-run
        does not leave the stage looking broken.
        """
        error = getattr(result, "error_in_exec", None) or getattr(
            result, "error_before_exec", None
        )
        run = self._run
        if run is None:
            return
        if error is None:
            run.error = None
            status = "running"
        else:
            run.error = {"type": type(error).__name__, "message": str(error)[:500]}
            status = "failed"
        self._persist(collect_io=self._io_due(), status=status)

    def flush(self) -> Path | None:
        """Persist everything recorded so far, including the observed I/O.

        Rarely needed -- logging already persists, and a cell boundary collects the
        rest. Useful before a long step whose record you want on disk first.
        """
        return self._persist(collect_io=True)

    def finish(self, status: str = "ok", **extra: Any) -> Path | None:
        """Close this stage: stamp it finished and collect its I/O one last time.

        Args:
            status: how the stage ended, ``"ok"`` unless you know better.
            **extra: last settings/counts to record, same rules as :meth:`record`.
        """
        run = self._run_or_open()
        if run.stage is None:
            raise ValueError(
                "this context has no stage name -- pass stage= to "
                "StageContext.for_image(), or call ctx.record('StageName')."
            )
        if extra:
            run.extra.update(_jsonable(extra))
        run.finalized = True
        path = self._persist(collect_io=True, status=status)

        _UNFINISHED.pop(id(run), None)
        # the next stage in this notebook (if any) starts from here
        object.__setattr__(self, "_started_at", _stage_io._utc())
        recorder = self._recorder
        if recorder is not None:
            recorder.reset(self.output_dir)
        return path

    def record(
        self, stage: str | None = None, artifacts: Iterable[str] = (), **extra: Any
    ) -> Path:
        """Append this stage's entry to the manifest and return its path.

        The explicit finalize. Equivalent to logging ``**extra`` and calling
        :meth:`finish`; still the whole story for a context built without a
        ``stage=`` name, where nothing is written until this call.

        Besides what the caller states, the entry carries what was *observed*: the
        files this stage read and wrote (each with a ``(size, mtime)`` fingerprint),
        when it ran and how long it took, and which version of the notebook produced
        it. That is what lets a later run notice that an input has changed underneath
        a result, and what makes "these two populations disagree" answerable with
        "they ran different code".

        Args:
            stage: name of the stage, e.g. ``"Segment"``. Optional when the context
                was built with ``stage=``. Re-running a stage replaces its own entry
                and leaves the others alone.
            artifacts: what this stage produced, as names inside the output folder.
                **Optional** -- the observed writes are recorded regardless, and
                :meth:`output_path` adds to this list on its own; pass it only to
                state an intent that can then be checked against reality.
            **extra: whatever makes the run reproducible -- settings used, counts
                obtained. Stored verbatim at the top of the entry, so keep it
                JSON-serialisable.
        """
        run = self._run_or_open(stage)
        if run.stage is None:
            raise ValueError(
                "record() needs a stage name -- pass one here, or build the context "
                "with StageContext.for_image(..., stage='Segment')."
            )
        run.artifacts.extend(str(a) for a in artifacts)
        path = self.finish(**extra)
        assert path is not None  # finish() only returns None without a stage name
        return path

    def _collect_io(
        self, manifest: dict[str, Any], artifacts: list[str]
    ) -> dict[str, Any] | None:
        """The observed reads and writes of the stage now finishing.

        Returns ``None`` when nothing was captured, in which case ``record()`` writes
        exactly the manifest it wrote before this feature existed. Provenance is a
        bonus; it must never be the reason an analysis fails.
        """
        recorder = self._recorder
        if recorder is None:
            return None
        try:
            recorder.apply_diff(self.output_dir)
            collapse = {_collapse(p, self.output_dir) for p in recorder.writes}

            # the manifest is this machinery's own bookkeeping, not a stage artifact,
            # and every context reads it while resolving the folder's source. Matched
            # by name prefix, not identity, so the atomic-write temp file
            # (stage_manifest.json.<pid>.tmp) is excluded too
            def _own(candidate: Path) -> bool:
                return candidate.name.startswith(MANIFEST_NAME)

            writes = {p for p in collapse if not _own(p)}
            reads = {
                _collapse(p, self.output_dir) for p in recorder.reads if not _own(p)
            } - writes

            source = Path(self.image_id).resolve()
            if source.exists():
                reads.add(source)

            produced = _producers(manifest)
            outputs = [_entry_for(p, self.output_dir) for p in sorted(writes)]
            inputs = []
            for path in sorted(reads):
                item = _entry_for(path, self.output_dir)
                owner = produced.get(item["path"])
                if owner:
                    item["produced_by"] = owner
                inputs.append(item)

            # compare like with like: an observed write is collapsed to its
            # top-level artifact, so `figures/size.png` is recorded as `figures/`.
            # Comparing the raw declared string against that would report every
            # file inside a subfolder as missing
            declared = {
                _as_recorded_path(
                    _collapse((self.output_dir / a).resolve(), self.output_dir),
                    self.output_dir,
                ).rstrip("/")
                for a in artifacts
            }
            recorded = {o["path"].rstrip("/") for o in outputs}
            io: dict[str, Any] = {
                "schema": IO_SCHEMA,
                "inputs": inputs,
                "outputs": outputs,
            }
            missing = sorted(declared - recorded)
            if missing:
                # declared but never written: usually a typo in `artifacts`
                io["missing"] = missing
            return io
        except Exception:
            logger.debug("could not collect stage I/O", exc_info=True)
            return None

    def stage(self, name: str) -> dict[str, Any] | None:
        """The recorded entry of a stage in this folder, or ``None``.

        Lets a downstream notebook read a setting off an upstream one -- the pixel
        size segmentation actually used, say -- instead of re-deriving it and risking
        a different answer.
        """
        entry = self.manifest().get("stages", {}).get(name)
        return dict(entry) if entry is not None else None

    def clear(self, stage: str) -> list[Path]:
        """Delete a stage's recorded outputs and drop its manifest entry.

        The honest undo for a stage that failed half-way: ``scale(exist_skip=True)``
        keys on the copied notebook existing, so a half-finished stage is skipped on
        every later run until its traces are gone. Removing exactly what the stage
        recorded is more precise than deleting a folder -- it also catches whatever
        it wrote elsewhere -- and it is a no-op for a stage that never ran.
        """
        manifest = self.manifest()
        entry = manifest.get("stages", {}).pop(stage, None)
        if entry is None:
            return []
        # a directory artifact may be shared -- every stage's figures land in
        # figures/ -- so only a directory no other stage also claims may be removed
        # wholesale; otherwise clearing one stage would delete another's results
        others = {
            item["path"]
            for other, other_entry in (manifest.get("stages") or {}).items()
            if other != stage
            for item in (other_entry.get("io") or {}).get("outputs", [])
        }
        removed = []
        for item in (entry.get("io") or {}).get("outputs", []):
            target = (self.output_dir / item["path"]).resolve()
            if target.is_dir():
                if item["path"] in others:
                    continue
                shutil.rmtree(target, ignore_errors=True)
                removed.append(target)
            elif target.exists():
                target.unlink()
                removed.append(target)
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
        return removed

    # -- calibration -------------------------------------------------------

    def log_calibration(
        self, source: Any, *, origin: str | None = None, check: bool = True
    ) -> dict[str, Any]:
        """Record the pixel size and frame interval this stage actually resolved.

        The movie stays the single authority on calibration -- this records what
        was resolved, it does not become a second place to read it from. That
        distinction matters: a stage that took its interval from an override
        parameter and one that read it from the file's OME-XML currently leave
        indistinguishable records, so "what interval did this run use?" has no
        answer, and two stages of one population silently disagreeing has nothing
        to notice it.

        With ``check`` (the default), a value that disagrees with what an earlier
        stage of this population recorded warns -- the same reports-never-decides
        stance as :func:`check_stale`, because the caller may know exactly what
        they changed.

        Args:
            source: the opened image sequence, after calibration is resolved.
            origin: where the calibration came from, when the source cannot say
                (it is read from ``source.calibration_source`` otherwise).
            check: warn when this disagrees with an earlier stage's record.

        Returns:
            The recorded calibration block.
        """
        run = self._require_run("log_calibration()")
        resolved: dict[str, Any] = {"schema": CALIBRATION_SCHEMA}

        pixel_size = getattr(source, "pixel_size", None)
        if pixel_size is not None:
            resolved["pixel_size"] = str(pixel_size)
            with contextlib.suppress(Exception):
                resolved["pixel_size_um"] = float(pixel_size.to("micrometer").magnitude)

        interval = _resolve_frame_interval(source)
        if interval is not None:
            resolved["frame_interval"] = str(interval)
            with contextlib.suppress(Exception):
                resolved["frame_interval_s"] = float(interval.to("second").magnitude)

        stated = origin or getattr(source, "calibration_source", None)
        if stated:
            resolved["origin"] = str(stated)

        if check:
            self._warn_on_calibration_mismatch(resolved)

        run.calibration = resolved
        self._persist(collect_io=False)
        return resolved

    def calibration(self, stage: str | None = None) -> dict[str, Any] | None:
        """The calibration recorded by a stage of this population.

        With no argument, the most recent one recorded here. Use it to *check* a
        value, not to replace opening the source -- see :meth:`log_calibration`.
        """
        stages = (self.manifest().get("stages") or {}).items()
        if stage is not None:
            entry = dict(stages).get(stage) or {}
            recorded = entry.get("calibration")
            return dict(recorded) if recorded else None
        found = [e["calibration"] for _, e in stages if e.get("calibration")]
        return dict(found[-1]) if found else None

    def _warn_on_calibration_mismatch(self, resolved: Mapping[str, Any]) -> None:
        """Warn when this stage resolved a different calibration than an earlier one."""
        for name, entry in (self.manifest().get("stages") or {}).items():
            if name == (self._run.stage if self._run else None):
                continue
            recorded = entry.get("calibration") or {}
            for key, label in (
                ("pixel_size_um", "pixel size"),
                ("frame_interval_s", "frame interval"),
            ):
                before, now = recorded.get(key), resolved.get(key)
                if before is None or now is None:
                    continue
                # metadata reads round in the last digit or two; only a real
                # disagreement is worth a warning
                if abs(before - now) > 1e-6 * max(abs(before), abs(now), 1e-12):
                    warnings.warn(
                        f"{label} {now} disagrees with {before}, recorded by stage "
                        f"{name!r} of this population -- the two stages analysed the "
                        "same movie with different calibration, so their results are "
                        "not comparable.",
                        stacklevel=3,
                    )

    # -- tables ------------------------------------------------------------

    def keyed(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` with this population's key columns added, ``attrs`` preserved.

        Exported tables carry their keys so many populations concatenate cleanly.
        ``DataFrame.assign`` drops ``attrs``, which would silently lose the unit
        map (``df.attrs["units"]``) that the CSV writers rely on -- this restores it.
        """
        keyed = df.assign(**self.keys)
        keyed.attrs = dict(df.attrs)
        keyed.attrs.setdefault(UNIT_ATTR, {})
        return keyed

    def __str__(self) -> str:
        return f"population {self.population_id}  ->  {self.output_dir.resolve()}"

    # -- display -----------------------------------------------------------

    def summary(self, *, figures: bool = True) -> Any:
        """This population's state as a rich display object.

        The same panel :meth:`_repr_html_` renders, as a value you can display
        from anywhere in a notebook rather than only as a cell's last expression.
        """
        from IPython.display import HTML  # noqa: PLC0415 -- display-time only

        return HTML(self._html(figures=figures))

    def _repr_html_(self) -> str:
        """Render the panel, or degrade to text -- but never raise.

        A repr is evaluated on a cell's last expression, so one that raises prints
        a traceback that reads as though the *analysis* failed. The same rule the
        capture layer follows applies here with more force: this is a view of the
        work, and it must never be mistaken for the work going wrong.
        """
        try:
            return self._html()
        except Exception:
            logger.debug("could not render the stage summary", exc_info=True)
            return f"<pre>{html.escape(str(self))}</pre>"

    def _html(self, *, figures: bool = True) -> str:
        manifest = self.manifest()
        stages = manifest.get("stages") or {}
        run = self._run
        current = run.stage if run is not None else None

        parts = [
            '<div style="font:13px/1.5 -apple-system,BlinkMacSystemFont,'
            "'Segoe UI',sans-serif;max-width:920px\">",
            self._html_header(current, run),
            self._html_current(run, figures=figures),
            self._html_stages(stages, current),
            self._html_files(manifest, stages),
            "</div>",
        ]
        return "".join(parts)

    @staticmethod
    def _html_status(status: str) -> str:
        color = {"ok": "#1a7f37", "failed": "#cf222e"}.get(status, "#9a6700")
        return (
            f'<span style="background:{color};color:#fff;border-radius:3px;'
            f'padding:1px 6px;font-size:11px">{html.escape(status)}</span>'
        )

    def _html_header(self, current: str | None, run: Any) -> str:
        title = html.escape(self.population_id)
        if current:
            title += f" &middot; {html.escape(current)} {self._html_status(run.status)}"
        return (
            f'<div style="font-size:15px;font-weight:600;margin-bottom:2px">{title}</div>'
            f'<div style="color:#57606a;font-size:12px;margin-bottom:10px">'
            f"{html.escape(str(self.output_dir.resolve()))}<br>"
            f"source: {html.escape(str(self.image_id))}</div>"
        )

    @staticmethod
    def _html_kv(title: str, mapping: Mapping[str, Any]) -> str:
        if not mapping:
            return ""
        rows = "".join(
            f'<tr><td style="padding:1px 12px 1px 0;color:#57606a;white-space:nowrap">'
            f'{html.escape(str(key))}</td><td style="padding:1px 0">'
            f"{html.escape(str(value))}</td></tr>"
            for key, value in mapping.items()
        )
        return (
            f'<div style="margin:6px 0"><b style="font-size:12px">{html.escape(title)}'
            f'</b><table style="border-collapse:collapse;font-size:12px">{rows}</table></div>'
        )

    def _html_current(self, run: Any, *, figures: bool) -> str:
        if run is None:
            return ""
        blocks = [
            self._html_kv("Parameters", run.params),
            self._html_kv("Metrics", run.metrics),
        ]

        if run.calibration:
            shown = {
                k: v
                for k, v in run.calibration.items()
                if k in ("pixel_size", "frame_interval", "origin")
            }
            blocks.append(self._html_kv("Calibration", shown))

        if run.error:
            blocks.append(
                f'<div style="margin:6px 0;padding:6px 8px;background:#ffebe9;'
                f'border-left:3px solid #cf222e;font-size:12px">'
                f"<b>{html.escape(run.error.get('type', 'Error'))}</b>: "
                f"{html.escape(run.error.get('message', ''))}</div>"
            )

        if figures and run.figures:
            blocks.append(self._html_figures(run))

        body = "".join(b for b in blocks if b)
        return body

    def _html_figures(self, run: Any) -> str:
        thumbs = dict(run.thumbnails)
        cards = []
        for figure in run.figures:
            caption = html.escape(figure.get("caption") or figure.get("name", ""))
            uri = thumbs.get(figure["path"])
            image = (
                f'<img src="{uri}" style="max-width:200px;height:auto;display:block">'
                if uri
                else f'<div style="color:#57606a;font-size:11px">'
                f"{html.escape(figure['path'])}</div>"
            )
            cards.append(
                f'<div style="margin:0 10px 10px 0">{image}'
                f'<div style="font-size:11px;color:#57606a;max-width:200px">{caption}</div></div>'
            )
        return (
            '<div style="margin:6px 0"><b style="font-size:12px">Figures</b>'
            f'<div style="display:flex;flex-wrap:wrap;margin-top:4px">{"".join(cards)}</div></div>'
        )

    def _html_stages(self, stages: Mapping[str, Any], current: str | None) -> str:
        if not stages:
            return (
                '<div style="color:#57606a;font-size:12px;margin:8px 0">'
                "No stage has run in this folder yet.</div>"
            )
        stale = {entry["stage"] for entry in check_stale(self.output_dir)}
        rows = []
        for name, entry in stages.items():
            marks = []
            if name in stale:
                marks.append('<span style="color:#9a6700">stale inputs</span>')
            duration = entry.get("duration_s")
            rows.append(
                "<tr>"
                f'<td style="padding:2px 12px 2px 0;{"font-weight:600" if name == current else ""}">'
                f"{html.escape(name)}</td>"
                f'<td style="padding:2px 12px 2px 0">'
                f"{self._html_status(entry.get('status', 'ok'))}</td>"
                f'<td style="padding:2px 12px 2px 0;color:#57606a">'
                f"{'' if duration is None else f'{duration:.1f} s'}</td>"
                f'<td style="padding:2px 12px 2px 0;color:#57606a">'
                f"{len((entry.get('io') or {}).get('outputs') or [])} out</td>"
                f'<td style="padding:2px 0">{" ".join(marks)}</td>'
                "</tr>"
            )
        return (
            '<div style="margin:10px 0"><b style="font-size:12px">Stages</b>'
            '<table style="border-collapse:collapse;font-size:12px;margin-top:2px">'
            f"{''.join(rows)}</table></div>"
        )

    def _html_files(
        self, manifest: Mapping[str, Any], stages: Mapping[str, Any]
    ) -> str:
        produced = _producers(manifest)
        try:
            # iterdir, never rglob: artifacts are the direct children of the output
            # folder (see _collapse), and a view must never be the thing that stats
            # a thousand CTC masks
            entries = sorted(self.output_dir.iterdir(), key=lambda q: q.name)
        except OSError:
            return ""
        rows = []
        for item in entries:
            if item.name.startswith(MANIFEST_NAME):
                continue
            recorded = f"{item.name}/" if item.is_dir() else item.name
            owner = produced.get(recorded) or produced.get(item.name)
            info = _stage_io.fingerprint(item)
            rows.append(
                "<tr>"
                f'<td style="padding:2px 12px 2px 0">{html.escape(recorded)}</td>'
                f'<td style="padding:2px 12px 2px 0;color:#57606a;text-align:right">'
                f"{_human_size(info.get('size'))}</td>"
                f'<td style="padding:2px 0;color:#57606a">'
                f"{html.escape(owner) if owner else '&mdash;'}</td>"
                "</tr>"
            )
        if not rows:
            return (
                '<div style="color:#57606a;font-size:12px;margin:8px 0">'
                "The output folder is empty.</div>"
            )
        header = (
            '<tr><th style="text-align:left;padding:2px 12px 2px 0;font-weight:600">file</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;font-weight:600">size</th>'
            '<th style="text-align:left;padding:2px 0;font-weight:600">produced by</th></tr>'
        )
        return (
            '<div style="margin:10px 0"><b style="font-size:12px">Files</b>'
            '<table style="border-collapse:collapse;font-size:12px;margin-top:2px">'
            f"{header}{''.join(rows)}</table></div>"
        )


def read_manifest(output_dir: str | Path) -> dict[str, Any]:
    """The ``stage_manifest.json`` of one population's output folder, or ``{}``."""
    manifest_path = Path(output_dir) / MANIFEST_NAME
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def stages_run(output_dir: str | Path) -> list[str]:
    """Names of the stages recorded in one population's output folder, in order."""
    return list(read_manifest(output_dir).get("stages", {}))


# --------------------------------------------------------------------------- #
# recorded I/O: helpers
# --------------------------------------------------------------------------- #


def _collapse(path: Path, output_dir: Path) -> Path:
    """An observed path reduced to the *artifact* it belongs to.

    Artifacts are the direct children of the output folder, and some of them are
    directories: CTC tracking output is a folder of one mask per frame. Recording
    every file inside it would bury the record under a thousand entries that all say
    the same thing, so anything below a subfolder is attributed to that subfolder --
    ``output/tracking/mask042.tif`` is the artifact ``tracking/``.
    """
    try:
        relative = path.relative_to(output_dir.resolve())
    except ValueError:
        return path
    return output_dir.resolve() / relative.parts[0] if relative.parts else path


def _as_recorded_path(path: Path, output_dir: Path) -> str:
    """A path as the manifest stores it: relative to the output folder if possible.

    Keeps the record readable and keeps a run relocatable; falls back to a
    ``..``-relative or absolute form for anything outside (a source on a share).
    """
    try:
        name = str(path.relative_to(output_dir.resolve()))
        return f"{name}/" if path.is_dir() else name
    except ValueError:
        try:
            return str(Path("..") / path.relative_to(output_dir.resolve().parent))
        except ValueError:
            return str(path)


def _entry_for(path: Path, output_dir: Path) -> dict[str, Any]:
    return {"path": _as_recorded_path(path, output_dir), **_stage_io.fingerprint(path)}


def _producers(manifest: Mapping[str, Any]) -> dict[str, str]:
    """``{recorded path: stage that wrote it}`` from the stages recorded so far.

    This is where the dependency graph comes from: nobody declares that tracking
    depends on segmentation, it is inferred from tracking having read the file
    segmentation wrote.
    """
    produced: dict[str, str] = {}
    for name, entry in (manifest.get("stages") or {}).items():
        for item in (entry.get("io") or {}).get("outputs", []):
            produced[item["path"]] = name
    return produced


def _relative_source(image_id: str, output_dir: Path) -> str | None:
    """The source path relative to the output folder, when they share an ancestor.

    The absolute path stops resolving the moment an execution folder moves between
    machines; this survives that, and recovery prefers whichever actually exists.
    """
    try:
        import os.path

        source = Path(image_id).resolve()
        base = output_dir.resolve()
        relative = os.path.relpath(source, base)
        return relative if not relative.startswith(os.pardir * 3) else None
    except (OSError, ValueError):
        return None


#: entry keys that are structure rather than a recorded setting. Everything else
#: in an entry becomes a :func:`stage_table` column, so a new structured block has
#: to be listed here or it lands in the table as an object column.
_ENTRY_RESERVED = frozenset(
    {
        "io",
        "code",
        "env",
        "artifacts",
        "started_at",
        "finished_at",
        "duration_s",
        "acia_version",
        "schema",
        "status",
        "params",
        "metrics",
        "figures",
        "calibration",
        "error",
        "declared_inputs",
    }
)


def _human_size(size: Any) -> str:
    """A byte count as something readable in a table cell."""
    if not isinstance(size, int | float):
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return ""


def _resolve_frame_interval(source: Any) -> Any:
    """The source's frame interval, however it is calibrated.

    A source may carry a scalar interval, or per-frame timepoints it was built
    from; both answer "how far apart are the frames?", and a recorded calibration
    that only understood one of them would be blank for half the sources here.
    """
    interval = getattr(source, "_frame_interval", None)
    if interval is not None:
        return interval
    with contextlib.suppress(Exception):
        timepoints = source.timepoints
        if timepoints is not None and len(timepoints) > 1:
            steps = timepoints[1:] - timepoints[:-1]
            return steps[0] if len(steps) else None
    return None


def _elapsed_since(started_at: str) -> float | None:
    try:
        start = datetime.fromisoformat(started_at)
        return round((datetime.now(timezone.utc) - start).total_seconds(), 1)
    except ValueError:
        return None


def _missing_artifact_message(
    ctx: StageContext, name: str | Path, produced_by: str | None
) -> str:
    """The most useful thing we can say about an artifact that isn't there."""
    target = ctx.path(name)
    manifest = ctx.manifest()
    owner = _producers(manifest).get(str(name))
    if owner:
        code = (manifest["stages"][owner].get("code") or {}).get("notebook")
        via = f" ({code})" if code else ""
        return (
            f"{target} not found -- it is produced by stage {owner!r}{via}; re-run it "
            f"in this folder (working directory: {Path.cwd()})."
        )
    ran = stages_run(ctx.output_dir)
    if produced_by:
        # the caller named a producer: say so, and add the folder's own evidence,
        # which is what separates "not run yet" from "wrong working directory"
        nothing_ran = " -- no stage has ever run here" if not ran else ""
        return (
            f"{target} not found{nothing_ran} -- run {produced_by} in this folder "
            f"first (working directory: {Path.cwd()})."
        )
    if not ran:
        return (
            f"{target} not found, and no stage has ever run in this folder -- you are "
            f"probably in the wrong working directory: {Path.cwd()}."
        )
    return (
        f"{target} not found. Stages recorded here: {', '.join(ran)} -- none of them "
        f"produced it (working directory: {Path.cwd()})."
    )


def check_stale(output_dir: str | Path) -> list[dict[str, Any]]:
    """Recorded stages whose inputs have changed since they ran.

    This is the gap that makes a staged workflow quietly wrong: re-segmenting with a
    new filter leaves the tracking output looking current forever, because
    ``exist_skip`` keys on a notebook file existing, not on freshness. Comparing each
    recorded input against the file on disk surfaces it.

    Reports, never decides -- like the curation manifest's fingerprint check, a
    mismatch is a warning, because the caller may know exactly what they changed.
    """
    stale = []
    output_dir = Path(output_dir)
    for name, entry in (read_manifest(output_dir).get("stages") or {}).items():
        for item in (entry.get("io") or {}).get("inputs", []):
            current = _stage_io.fingerprint(output_dir / item["path"])
            if not current or "mtime" not in item:
                continue
            if current["size"] != item.get("size") or current["mtime"] > item["mtime"]:
                stale.append(
                    {
                        "stage": name,
                        "path": item["path"],
                        "recorded": _stage_io._utc(item["mtime"]),
                        "actual": _stage_io._utc(current["mtime"]),
                    }
                )
    return stale


def stage_graph(output_dir: str | Path) -> list[tuple[str, str, str]]:
    """``(upstream, artifact, downstream)`` edges derived from what stages read.

    Nothing declares this graph -- it is what the recorded inputs and outputs say
    actually happened.
    """
    manifest = read_manifest(output_dir)
    edges = []
    for name, entry in (manifest.get("stages") or {}).items():
        for item in (entry.get("io") or {}).get("inputs", []):
            upstream = item.get("produced_by")
            if upstream and upstream != name:
                edges.append((upstream, item["path"], name))
    return edges


def stage_table(
    root: str | Path, pattern: str = "*/output", stale: bool = True
) -> pd.DataFrame:
    """Every recorded stage run under ``root``, as one table -- one row per run.

    Turns a folder of results into something searchable without a query language:
    the population keys, the timings, the code that ran and every setting a stage
    recorded become columns, so questions are pandas.

    .. code-block:: python

        runs = stage_table("automated_executions_stages")
        runs[runs.stale]                                  # what needs redoing
        runs[runs.stage == "Segment"].pixel_size.value_counts()
        runs.groupby(["stage", "code_sha256"]).size()     # did the batch run one code?

    Args:
        root: folder holding one subfolder per population.
        pattern: glob from ``root`` to each population's output folder. The default
            matches the layout :func:`acia.analysis.scale` produces.
        stale: check each run's inputs against the files on disk and add a ``stale``
            column. Set it to ``False`` for a large tree where the extra ``stat``
            calls are not worth it.

    Returns:
        pd.DataFrame: one row per (population, stage). Besides the population's key
            columns it carries ``status`` and, for a stage that failed,
            ``error_type``/``error_message``; the timing, notebook and
            ``code_sha256`` of the run; ``n_inputs``/``n_outputs``/``n_figures``;
            the resolved ``pixel_size_um``/``frame_interval_s``; and every
            parameter and metric the stages logged, flattened into columns of its
            own name so a whole fan-out can be compared in one table.
    """
    rows: list[dict[str, Any]] = []
    for output_dir in sorted(Path(root).glob(pattern)):
        manifest = read_manifest(output_dir)
        if not manifest:
            continue
        population = manifest.get("population") or {}
        stale_stages = {s["stage"] for s in check_stale(output_dir)} if stale else set()
        for name, entry in (manifest.get("stages") or {}).items():
            row: dict[str, Any] = {**population, "stage": name}
            row["output_dir"] = str(output_dir)
            io = entry.get("io") or {}
            code = entry.get("code") or {}
            calibration = entry.get("calibration") or {}
            error = entry.get("error") or {}

            # Settings the stage recorded become columns, so they can be compared
            # across a batch. Logged parameters and metrics flatten into the *same*
            # namespace as the flat keys record(**extra) writes, so a folder whose
            # stages use either API still tabulates the same way.
            row.update(entry.get("params") or {})
            row.update(entry.get("metrics") or {})
            row.update(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in _ENTRY_RESERVED
                }
            )
            # derived columns last: a setting named `duration_s` must not shadow
            # the real one
            row.update(
                {
                    "status": entry.get("status", "ok"),
                    "started_at": entry.get("started_at"),
                    "finished_at": entry.get("finished_at"),
                    "duration_s": entry.get("duration_s"),
                    "acia_version": entry.get("acia_version"),
                    "notebook": code.get("notebook"),
                    "code_sha256": code.get("sha256"),
                    "n_inputs": len(io.get("inputs", [])) if io else None,
                    "n_outputs": len(io.get("outputs", [])) if io else None,
                    "n_figures": len(entry.get("figures") or []),
                    # what went wrong, not merely that something did: across a
                    # fan-out this is the column that separates "one ROI ran out
                    # of GPU memory" from "the chain is broken for everything"
                    "error_type": error.get("type"),
                    "error_message": error.get("message"),
                    "pixel_size_um": calibration.get("pixel_size_um"),
                    "frame_interval_s": calibration.get("frame_interval_s"),
                    "stale": name in stale_stages,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
