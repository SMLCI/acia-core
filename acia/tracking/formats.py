"""Module to convert tracking formats"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import tifffile

from acia.base import Contour, ImageSequenceSource, Overlay
from acia.segm.formats import read_ctc_segmentation_native


def parse_simple_tracking(file_content: str) -> tuple[Overlay, nx.DiGraph]:
    """Parse simple tracking format from file content string

    Args:
        file_content (str): simple tracking format file content

    Returns:
        Tuple[Overlay, nx.DiGraph]: segmentation overlay and tracking graph
    """

    data = json.loads(file_content)

    segmentation_data = data["segmentation"]
    tracking_data = data["tracking"]

    # deal with the segmentation first
    all_detections = []

    # create contours
    for det in segmentation_data:
        det_id = det["id"]
        # try to convert to integer
        with contextlib.suppress(ValueError):
            det_id = int(det_id)

        all_detections.append(
            Contour(det["contour"], -1, det["frame"], det_id, det["label"])
        )

    segmentation_overlay = Overlay(all_detections)

    # deal with the tracking
    tracking = nx.DiGraph()

    tracking.add_nodes_from(map(lambda cont: cont.id, segmentation_overlay))
    node_set = set(tracking.nodes)
    for cont in segmentation_overlay:
        tracking.nodes[cont.id]["frame"] = cont.frame

    # create graph from id links
    for link in tracking_data:
        if link["sourceId"] in node_set and link["targetId"] in node_set:
            tracking.add_edge(link["sourceId"], link["targetId"])

    return segmentation_overlay, tracking


def gen_simple_tracking(overlay: Overlay, tracking_graph: nx.Graph) -> str:
    """Create a simple tracking format from overlay and tracking graph

    Args:
        overlay (Overlay): segmentation overlay
        tracking_graph (nx.Graph): tracking graph

    Returns:
        str: simple tracking format string
    """

    segmentation_data = []
    for cont in overlay:
        coordinates = cont.coordinates

        if isinstance(coordinates, np.ndarray):
            coordinates = coordinates.tolist()

        segmentation_data.append(
            dict(label=cont.label, contour=coordinates, id=cont.id, frame=cont.frame)
        )

    tracking_data = []
    for edge in tracking_graph.edges:
        tracking_data.append(dict(sourceId=edge[0], targetId=edge[1]))

    simpleTracking = dict(segmentation=segmentation_data, tracking=tracking_data)

    return json.dumps(simpleTracking)


def read_ctc_tracklet_graph(file: Path):
    colnames = ["label", "t_start", "t_end", "parent"]

    ctc_df = pd.read_csv(
        file,
        names=colnames,
        header=None,
        dtype={"label": "int", "t_start": "int", "t_end": "int", "parent": "int"},
        sep=" ",
    )

    tracklet_graph = nx.DiGraph()

    for _, row in ctc_df.iterrows():
        label, t_start, t_end, parent_label = row.to_list()
        tracklet_graph.add_node(label, start_frame=t_start, end_frame=t_end)

        if parent_label != 0:
            tracklet_graph.add_edge(parent_label, label)

    return tracklet_graph


def read_ctc_tracking(input_path: Path) -> tuple[Overlay, nx.DiGraph, nx.DiGraph]:
    """Read ctc tracking information

    Args:
        input_path (Path): Path to the ctc tracking folder

    Returns:
        tuple[Overlay, nx.DiGraph, nx.DiGraph]: segmentation overlay, tracklet graph (every cell cycle is a node), tracking graph (every cell detection is a node)
    """
    input_path = Path(input_path)
    track_file = input_path / "man_track.txt"

    ov = read_ctc_segmentation_native(input_path)
    tracklet_graph = read_ctc_tracklet_graph(track_file)

    tracking_graph = ctc_track_graph(ov, tracklet_graph)

    return ov, tracklet_graph, tracking_graph


def write_ctc_tracking(
    output_path: Path,
    images: ImageSequenceSource,
    overlay: Overlay,
    tracklet_graph: nx.DiGraph,
):
    """Write ctc tracking to output folder

    Args:
        output_path (Path): output folder for writing
        images (ImageSequenceSource): image time-lapse (only used to compute mask sizes)
        overlay (Overlay): segmentation overlay
        tracklet_graph (nx.DiGraph): tracklet graph (every cell cycle is a node)
    """

    output_path = Path(output_path)

    # Write tracklet information
    data = []
    for n in tracklet_graph.nodes:
        predecessors = list(tracklet_graph.predecessors(n))
        if len(predecessors) == 0:
            parent_label = 0
        else:
            parent_label = predecessors[0]

            if len(predecessors) > 1:
                logging.warning(
                    "Tracklet has more than one parent. This indicates a malformed tracklet graph!"
                )

        data.append(
            {
                "label": n,
                "t_start": tracklet_graph.nodes[n]["start_frame"],
                "t_end": tracklet_graph.nodes[n]["end_frame"],
                "parent": parent_label,
            }
        )

    df_ctc = pd.DataFrame(data)
    df_ctc.to_csv(output_path / "man_track.txt", sep=" ", header=False, index=False)

    # get the image size
    height, width = next(iter(images)).raw.shape[:2]

    # Write segmentation information
    for i, frame_overlay in enumerate(overlay.timeIterator()):
        local_mask = np.zeros((height, width), dtype=np.uint16)
        for cont in frame_overlay:
            cont_mask = cont.toMask(height, width).astype(np.uint16) * cont.label
            local_mask = np.maximum(local_mask, cont_mask)

        # zlib-compress the label masks: they are mostly background and highly
        # repetitive, so this is ~50x smaller than the uncompressed write (1 GiB
        # -> 18 MiB for a 500-frame 1024x1024 movie) at no readability cost --
        # tifffile decompresses transparently, and it stays a valid TIFF.
        tifffile.imwrite(
            output_path / f"man_track{i:04d}.tif", local_mask, compression="zlib"
        )


def save_tracking(
    path: str | Path,
    images: ImageSequenceSource,
    overlay: Overlay,
    tracklet_graph: nx.DiGraph,
) -> Path:
    """Store a tracking result as a CTC folder, guaranteeing full frame coverage.

    The counterpart of :func:`load_tracking`, and the recommended way to persist
    a tracker's ``(overlay, tracklet_graph)`` output. It wraps
    :func:`write_ctc_tracking` and adds the one guarantee that function cannot
    give on its own: the written mask stack is aligned with ``images``.

    ``write_ctc_tracking`` names its masks by enumerating
    :meth:`~acia.base.Overlay.timeIterator`, which starts at the overlay's
    *first populated* frame when the overlay carries no explicit frame list. An
    overlay whose frame 0 happens to hold no detections would therefore write a
    stack shifted against the movie -- every reloaded detection landing on the
    wrong frame, with no error anywhere. This function re-wraps such an overlay
    over ``range(images.size_t)`` first (the caller's overlay is not mutated).

    Args:
        path: output directory (created if missing). It must be owned by this
            artifact: :func:`load_tracking` reads *every* ``*.tif`` in it.
        images: the image sequence the tracking was computed on -- used for the
            mask size and the frame extent.
        overlay: tracked overlay; ``label`` carries the tracklet id.
        tracklet_graph: one node per tracklet (``start_frame``/``end_frame``).

    Returns:
        The directory written.

    Raises:
        ValueError: if the overlay holds a frame beyond ``images.size_t``, i.e.
            overlay and images do not belong to the same sequence.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    size_t = images.size_t
    frames = [int(f) for f in overlay.frames()]

    if frames and frames[-1] >= size_t:
        raise ValueError(
            f"Overlay holds frame {frames[-1]} but the image sequence has only "
            f"{size_t} frames -- overlay and images do not match."
        )

    # re-wrap (never mutate) when the overlay does not already span the movie, so
    # timeIterator emits exactly one mask per image frame, starting at frame 0
    if not frames or frames[0] != 0 or frames[-1] + 1 != size_t:
        overlay = Overlay(overlay.contours, frames=list(range(size_t)))

    write_ctc_tracking(path, images, overlay, tracklet_graph)

    return path


def load_tracking(
    path: str | Path, source: ImageSequenceSource
) -> tuple[Overlay, nx.DiGraph, nx.DiGraph]:
    """Load a tracking stored by :func:`save_tracking`, with time re-attached.

    Returns the same ``(overlay, tracklet_graph, tracking_graph)`` triple, in the
    same order, that a tracking processor (e.g.
    :class:`~acia.tracking.processor.trackastra.TrackastraTracker`) returns -- so
    a step that loads is a drop-in for a step that tracked.

    This is **not** equivalent to :func:`read_ctc_tracking`. That function builds
    the tracking graph while the reloaded overlay is still uncalibrated, and
    :func:`ctc_track_graph` reads each detection's ``time`` to stamp its nodes --
    so re-attaching the time model afterwards leaves the graph timeless, and a
    lineage plotted over ``time_feature="time"`` silently has nothing to plot.
    Here the calibration is attached *first* and the tracking graph is built from
    the calibrated overlay.

    Detection ids are **not** stable across this round-trip: the CTC mask format
    stores label images, so ids are renumbered on load (this is already true of
    the trackers themselves, which round-trip through the same format). After
    tracking, ``label`` -- the tracklet id -- is the stable key; do not join these
    ids against a property table exported before tracking.

    Args:
        path: the CTC directory written by :func:`save_tracking`.
        source: the image sequence the tracking was computed on. Required: it is
            the only carrier of the time calibration.

    Returns:
        ``(overlay, tracklet_graph, tracking_graph)``. When ``source`` is
        uncalibrated, the graphs simply carry no time attributes.

    Raises:
        FileNotFoundError: if the directory or its ``man_track.txt`` is missing.
    """
    # local import: acia.tracking imports this module, so this cannot be top-level
    from acia.tracking import annotate_tracklet_times

    path = Path(path)
    track_file = path / "man_track.txt"
    if not path.is_dir():
        raise FileNotFoundError(
            f"No tracking artifact at {path} -- expected a directory written by "
            "acia.tracking.formats.save_tracking()."
        )
    if not track_file.exists():
        raise FileNotFoundError(
            f"{path} holds no man_track.txt -- it is not a CTC tracking directory."
        )

    overlay = read_ctc_segmentation_native(path)
    tracklet_graph = read_ctc_tracklet_graph(track_file)

    if overlay.numFrames() != source.size_t:
        logging.warning(
            "Tracking artifact has %d frames but the image sequence has %d -- "
            "they may not belong to the same acquisition.",
            overlay.numFrames(),
            source.size_t,
        )

    # attach the time model BEFORE building the tracking graph (see docstring)
    timepoints = source.timepoints
    if timepoints is not None:
        overlay = overlay.with_timepoints(timepoints)
        annotate_tracklet_times(tracklet_graph, timepoints)

    tracking_graph = ctc_track_graph(overlay, tracklet_graph)

    return overlay, tracklet_graph, tracking_graph


def ctc_track_graph(ov: Overlay, tracklet_graph: nx.DiGraph):
    """Computes the ctc track graph (every cell detection is a node) based on cell detections (overlay) and the tracklet graph (every tracklet is one node).

    Hint: overlay labels and tracklet_graph node ids need to align.

    Args:
        ov (Overlay): _description_
        tracklet_graph (nx.DiGraph): _description_

    Returns:
        _type_: _description_
    """

    track_graph = nx.DiGraph()

    # add all the nodes -- carrying real time (not just frame index) when the
    # overlay is time-calibrated, so the lineage can plot against real time.
    time_unit: str | None = None
    all_timed = True
    for cont in ov:
        t = getattr(cont, "time", None)
        if t is None:
            all_timed = False
            track_graph.add_node(cont.id, frame=cont.frame)
        else:
            track_graph.add_node(cont.id, frame=cont.frame, time=float(t.magnitude))
            time_unit = time_unit or f"{t.units:~P}"
    if all_timed and time_unit is not None:
        track_graph.graph["time_unit"] = time_unit

    tracklets: dict[Any, list[Any]] = {}
    for cont in ov:
        tracklets[cont.label] = tracklets.get(cont.label, []) + [cont]

    for tracklet_label in tracklets:
        tracklets[tracklet_label] = sorted(
            tracklets[tracklet_label], key=lambda c: c.frame
        )

    for tracklet_label, tracklet_nodes in tracklets.items():
        # add tracklet edges
        for contA, contB in zip(tracklet_nodes, tracklet_nodes[1:], strict=False):
            track_graph.add_edge(contA.id, contB.id)

        for pred_label in tracklet_graph.predecessors(tracklet_label):
            track_graph.add_edge(tracklets[pred_label][-1].id, tracklet_nodes[0].id)

        for succ_label in tracklet_graph.successors(tracklet_label):
            track_graph.add_edge(tracklet_nodes[-1].id, tracklets[succ_label][0].id)

    return track_graph


def tracking_to_graph(data: list[dict]) -> nx.DiGraph:
    """Populates a ctc tracking into a full tracking lineage where every detection has its own node with a unique id based on (ctc_id, frame)

    Args:
        data (list[dict]): Output of :func:`read_ctc_tracking`

    Returns:
        nx.DiGraph: A lineage graph where every detection has its unique node (id, frame) and the edges represent the linking
    """

    graph = nx.DiGraph()

    # go through every ctc line
    for item in data:
        # iterate every frame for that the cell track exists
        for frame in range(int(item["start_frame"]), int(item["end_frame"]) + 1):
            # add node with unique id (ctc_id, frame)
            graph.add_node((item["id"], frame), frame=frame)
            # add non-division links
            if graph.has_node((item["id"], frame - 1)):
                graph.add_edge((item["id"], frame - 1), (item["id"], frame))

    # add division links
    for item in data:
        if int(item["parent_id"]) != 0:
            # get the time it divides
            # split_frame = int(item["start_frame"]) - 1

            # extract source and target
            source_candidates = list(
                # pylint: disable=cell-var-from-loop
                filter(lambda n: n[0] == item["parent_id"], graph.nodes)
            )
            latest_source = np.argmax(list(map(lambda n: n[1], source_candidates)))
            source = source_candidates[
                latest_source
            ]  # (item["parent_id"], split_frame)
            target = (item["id"], int(item["start_frame"]))

            if not graph.has_node(source):
                print("Error")

            assert graph.has_node(source), f"Source: {source}"
            assert graph.has_node(target), f"Target: {target}"

            # add the edge
            graph.add_edge(source, target)

    return graph
