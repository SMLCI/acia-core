"""Shared run context for a chain of analysis notebooks ("stages").

A staged workflow splits one analysis into several notebooks (segment -> track ->
measure) that hand their results over as **files in a shared output folder**, one
folder per imaged population. Every stage therefore has to answer the same three
questions before it can do anything: where do results go, which population is this,
and did the stage I depend on actually run here?

:class:`StageContext` answers them once:

.. code-block:: python

    from acia.analysis import StageContext

    ctx = StageContext.for_image(image_id, "./output")
    print(ctx)                                          # population pos001_roi002 -> ...

    seg = ctx.require("segmentation.npz")               # fail early, actionably
    write_units_csv(ctx.keyed(props), ctx.path("cell_properties.csv"))
    ctx.record("Track", n_tracklets=len(graph))

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

import json
import logging
import re
import shutil
import warnings
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
    """

    image_id: str
    output_dir: Path
    population_id: str
    keys: Mapping[str, Any]
    #: observed reads/writes of the stage currently in progress; ``None`` when
    #: capture is off. Private -- see :meth:`track` and :meth:`record`.
    _recorder: Any = field(default=None, compare=False, repr=False)
    #: when the stage in progress started, UTC ISO-8601
    _started_at: str | None = field(default=None, compare=False, repr=False)

    @classmethod
    def for_image(
        cls,
        image_id: str | Path | None = None,
        output_folder: str | Path = "./output",
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
            _started_at=_stage_io._utc(),
        )
        ctx._begin(track_io=track_io, track_roots=track_roots)
        _warn_if_stale(output_dir)
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
        target = self.path(name)
        if not target.exists():
            raise FileNotFoundError(_missing_artifact_message(self, name, produced_by))
        recorder = self._recorder
        if recorder is not None:
            # explicit, so it survives readers the audit hook cannot see -- anything
            # going through OpenCV's C++ layer, or a remote fsspec/OMERO source
            recorder.declare_read(target)
        return target

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

    # -- manifest ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        """Path of this population's ``stage_manifest.json``."""
        return self.path(MANIFEST_NAME)

    def manifest(self) -> dict[str, Any]:
        """The manifest as recorded so far (``{}`` before the first stage)."""
        return read_manifest(self.output_dir)

    def record(self, stage: str, artifacts: Iterable[str] = (), **extra: Any) -> Path:
        """Append this stage's entry to the manifest and return its path.

        Besides what the caller states, the entry carries what was *observed*: the
        files this stage read and wrote (each with a ``(size, mtime)`` fingerprint),
        when it ran and how long it took, and which version of the notebook produced
        it. That is what lets a later run notice that an input has changed underneath
        a result, and what makes "these two populations disagree" answerable with
        "they ran different code".

        Args:
            stage: name of the stage, e.g. ``"Segment"``. Re-running a stage replaces
                its own entry and leaves the others alone.
            artifacts: what this stage produced, as names inside the output folder.
                **Optional** -- the observed writes are recorded regardless; pass it
                only to state an intent that can then be checked against reality.
            **extra: whatever makes the run reproducible -- settings used, counts
                obtained. Stored verbatim, so keep it JSON-serialisable.
        """
        manifest = self.manifest()
        population = dict(self.keys, image_id=str(Path(self.image_id).resolve()))
        relative = _relative_source(self.image_id, self.output_dir)
        if relative is not None:
            population["image_id_relative"] = relative
        manifest.setdefault("population", {}).update(population)

        entry: dict[str, Any] = dict(
            artifacts=list(artifacts),
            acia_version=__version__,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        )
        if self._started_at:
            entry["started_at"] = self._started_at
            entry["duration_s"] = _elapsed_since(self._started_at)
        code = _stage_io.code_fingerprint()
        if code:
            entry["code"] = code
        env = _stage_io.env_info()
        if env:
            entry["env"] = env
        io = self._collect_io(manifest, entry["artifacts"])
        if io is not None:
            entry["io"] = io

        manifest.setdefault("stages", {})[stage] = entry
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

        # the next stage in this notebook (if any) starts from here
        object.__setattr__(self, "_started_at", _stage_io._utc())
        recorder = self._recorder
        if recorder is not None:
            recorder.reset(self.output_dir)
        return self.manifest_path

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
            manifest_path = self.manifest_path.resolve()
            collapse = {_collapse(p, self.output_dir) for p in recorder.writes}
            # the manifest is this machinery's own bookkeeping, not a stage artifact,
            # and every context reads it while resolving the folder's source
            writes = {p for p in collapse if p != manifest_path}
            reads = {
                _collapse(p, self.output_dir)
                for p in recorder.reads
                if p != manifest_path
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

            declared = {str(Path(a)).rstrip("/") for a in artifacts}
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
        removed = []
        for item in (entry.get("io") or {}).get("outputs", []):
            target = (self.output_dir / item["path"]).resolve()
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(target)
            elif target.exists():
                target.unlink()
                removed.append(target)
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
        return removed

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
            row.update(
                {
                    "started_at": entry.get("started_at"),
                    "finished_at": entry.get("finished_at"),
                    "duration_s": entry.get("duration_s"),
                    "acia_version": entry.get("acia_version"),
                    "notebook": code.get("notebook"),
                    "code_sha256": code.get("sha256"),
                    "n_inputs": len(io.get("inputs", [])) if io else None,
                    "n_outputs": len(io.get("outputs", [])) if io else None,
                    "stale": name in stale_stages,
                }
            )
            # settings the stage recorded become columns, so they can be compared
            row.update(
                {
                    key: value
                    for key, value in entry.items()
                    if key
                    not in {
                        "io",
                        "code",
                        "env",
                        "artifacts",
                        "started_at",
                        "finished_at",
                        "duration_s",
                        "acia_version",
                    }
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
