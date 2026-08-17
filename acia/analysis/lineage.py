"""Export recorded stage runs as OpenLineage events -- **optional**.

``stage_manifest.json`` stays the source of truth: it is per-population *state*, and
that is what :meth:`StageContext.require` and :func:`stages_run` read to answer "has
this stage run here?". OpenLineage is an append-only event *log*, which cannot answer
that without replay -- so this is an export, not the storage format.

It exists for the day these runs have to leave the project: OpenLineage is an
LF AI & Data standard with an established object model (Job, Run, Dataset), so a
lineage graph exported this way can be loaded into Marquez or handed to a data
platform without anyone writing a parser for acia's manifest.

Requires the optional dependency::

    pip install "acia[lineage]"

Events are written to a **file**, not shipped to a server::

    from acia.analysis.lineage import to_openlineage

    to_openlineage("automated_executions_stages", "lineage.jsonl")
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from acia import __version__
from acia.analysis import _stage_io
from acia.analysis.stage import read_manifest

#: namespace for the datasets acia records -- plain files on a filesystem
NAMESPACE = "file"

#: namespace for the jobs, i.e. the stages of an acia notebook chain
JOB_NAMESPACE = "acia"

#: what produced these events, as OpenLineage requires
PRODUCER = f"https://pypi.org/project/acia/{__version__}"


def _require_client() -> Any:
    try:
        from openlineage.client import OpenLineageClient
        from openlineage.client.transport.file import FileConfig, FileTransport
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "OpenLineage export needs the optional dependency: "
            'pip install "acia[lineage]"'
        ) from error
    return OpenLineageClient, FileConfig, FileTransport


def _dataset(item: dict[str, Any], output_dir: Path, dataset_cls: Any) -> Any:
    """One recorded file as an OpenLineage dataset, named by its absolute path.

    The recorded paths are relative to the output folder (so a run stays
    relocatable); lineage across populations only lines up if they are resolved
    back to one namespace here.
    """
    return dataset_cls(
        namespace=NAMESPACE, name=str((output_dir / item["path"]).resolve())
    )


def to_openlineage(
    root: str | Path,
    out_path: str | Path,
    pattern: str = "*/output",
) -> Path:
    """Write every recorded stage run under ``root`` as OpenLineage events.

    Each stage run becomes a ``COMPLETE`` event whose job is the stage name and whose
    input/output datasets are the files the stage was observed to read and write.
    Stages recorded before I/O capture existed carry no datasets and are exported as
    bare job runs rather than skipped, so the log stays a faithful list of what ran.

    Args:
        root: folder holding one subfolder per population.
        out_path: file the events are appended to, one JSON object per line.
        pattern: glob from ``root`` to each population's output folder.

    Returns:
        The path written to.
    """
    from openlineage.client.event_v2 import (
        InputDataset,
        Job,
        OutputDataset,
        Run,
        RunEvent,
        RunState,
    )

    client_cls, file_config, file_transport = _require_client()
    out_path = Path(out_path)
    client = client_cls(
        transport=file_transport(file_config(log_file_path=str(out_path), append=True))
    )

    for output_dir in sorted(Path(root).glob(pattern)):
        manifest = read_manifest(output_dir)
        population = (manifest.get("population") or {}).get("population_id", "unknown")
        for name, entry in (manifest.get("stages") or {}).items():
            io = entry.get("io") or {}
            # eventTime is required by the spec; a manifest predating the timestamps
            # still has finished_at, but be defensive rather than emit an invalid event
            finished_at = entry.get("finished_at") or _stage_io._utc()
            client.emit(
                RunEvent(
                    eventType=RunState.COMPLETE,
                    eventTime=finished_at,
                    run=Run(runId=str(uuid.uuid4())),
                    job=Job(namespace=JOB_NAMESPACE, name=f"{name}/{population}"),
                    inputs=[
                        _dataset(item, output_dir, InputDataset)
                        for item in io.get("inputs", [])
                    ],
                    outputs=[
                        _dataset(item, output_dir, OutputDataset)
                        for item in io.get("outputs", [])
                    ],
                    producer=PRODUCER,
                )
            )
    return out_path
