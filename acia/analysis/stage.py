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

    seg = ctx.require("segmentation.npz", "01_Segment.ipynb")   # fail early, actionably
    write_units_csv(ctx.keyed(props), ctx.path("cell_properties.csv"))
    ctx.record("02_Track", artifacts=["tracking/"], n_tracklets=len(graph))

The recorded ``stage_manifest.json`` makes a finished run self-describing -- which
stages ran, what each produced and under which settings -- and is what
:func:`acia.analysis.scale` batches read back to summarise a fan-out. Stages
*append* to it, so a chain that grows a stage later keeps the earlier entries.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from acia import __version__
from acia.analysis.units import UNIT_ATTR

#: file name of the per-population manifest inside the output folder
MANIFEST_NAME = "stage_manifest.json"

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

    @classmethod
    def for_image(
        cls,
        image_id: str | Path,
        output_folder: str | Path = "./output",
        *,
        key_pattern: str | None = DEFAULT_KEY_PATTERN,
        create: bool = True,
    ) -> StageContext:
        """Resolve the run context for one source.

        Args:
            image_id: path to the movie this run analyses -- a single stack file or
                a folder of per-timepoint images.
            output_folder: where this population's artifacts live. A **relative**
                value resolves against the working directory, which is what makes a
                stage chain work unchanged both interactively (the notebook folder)
                and under :func:`acia.analysis.scale` (each population's own
                execution folder).
            key_pattern: regex with named groups, matched against the population id
                to derive grouping keys. ``None`` disables key extraction.
            create: create ``output_folder`` if it does not exist.
        """
        population_id = population_id_of(image_id)
        output_dir = Path(output_folder)
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            image_id=str(image_id),
            output_dir=output_dir,
            population_id=population_id,
            keys={
                "population_id": population_id,
                **_parse_keys(population_id, key_pattern),
            },
        )

    # -- artifacts ---------------------------------------------------------

    def path(self, name: str | Path) -> Path:
        """Path of an artifact in this population's output folder."""
        return self.output_dir / name

    def has(self, name: str | Path) -> bool:
        """Whether an artifact is already present (e.g. an optional upstream stage)."""
        return self.path(name).exists()

    def require(self, name: str | Path, produced_by: str) -> Path:
        """Path of an artifact that must exist, or an actionable error.

        Fails with the stage that would have produced it *and* the working
        directory, because the usual cause is running a stage in the wrong folder.
        """
        target = self.path(name)
        if not target.exists():
            raise FileNotFoundError(
                f"{target} not found -- run {produced_by} in this folder first "
                f"(working directory: {Path.cwd()})."
            )
        return target

    # -- manifest ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        """Path of this population's ``stage_manifest.json``."""
        return self.path(MANIFEST_NAME)

    def manifest(self) -> dict[str, Any]:
        """The manifest as recorded so far (``{}`` before the first stage)."""
        return read_manifest(self.output_dir)

    def record(self, stage: str, artifacts: Iterable[str], **extra: Any) -> Path:
        """Append this stage's entry to the manifest and return its path.

        Args:
            stage: name of the stage, e.g. ``"02_Track"``. Re-running a stage
                replaces its own entry and leaves the others alone.
            artifacts: what this stage produced, as names inside the output folder.
            **extra: whatever makes the run reproducible -- settings used, counts
                obtained. Stored verbatim, so keep it JSON-serialisable.
        """
        manifest = self.manifest()
        manifest.setdefault("population", {}).update(
            **self.keys, image_id=str(Path(self.image_id).resolve())
        )
        manifest.setdefault("stages", {})[stage] = dict(
            artifacts=list(artifacts),
            acia_version=__version__,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        )
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
        return self.manifest_path

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
