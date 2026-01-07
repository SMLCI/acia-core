"""Tracking dataset exporters"""

from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import tifffile
from shapely.geometry import MultiPolygon
from tqdm.auto import tqdm

from acia.base import BaseImage, Contour, ImageSequenceSource, Overlay
from acia.tracking import TrackingSource
from acia.utils import multi_mask_to_polygons


class CellTrackingChallengeDatasetGT:
    """
    Utility class to create segmentation/tracking output in form of the Cell Tracking Challenge (CTC)
    """

    def __init__(self):
        self.sources = []

    def add(self, content: tuple[ImageSequenceSource, TrackingSource]):
        self.sources.append(content)

    def write(self, base_folder: str | Path = "data", offset=0):
        """Exports a ctc ground truth dataset

        Args:
            base_folder (str | Path, optional): _description_. Defaults to "data".
        """

        base_folder = Path(base_folder)

        base_folder.mkdir(exist_ok=True, parents=True)

        for i, (image_source, tracking_source) in enumerate(self.sources):
            mode = "GT"

            image_dir = base_folder / f"{i + offset:02}"
            ann_dir = base_folder / f"{i + offset:02}_{mode}"

            if image_dir.exists():
                shutil.rmtree(image_dir)
            if ann_dir.exists():
                shutil.rmtree(ann_dir)

            image_dir.mkdir()
            ann_dir.mkdir()

            height, width = -1, -1

            # save the images
            for t, image in enumerate(image_source):
                if isinstance(image, BaseImage):
                    image = image.raw
                height, width = image.shape[:2]
                tifffile.imwrite(str(image_dir / f"t{t:04}.tif"), image[..., 0])

            tracking_helper = CTCTrackingHelper(
                tracking_source.overlay, tracking_source.tracking_graph, height, width
            )
            ctc_masks, ctc_tracking_format = tracking_helper.to_ctc_format()

            # store segmentation masks
            seg_dir = ann_dir / "SEG"
            seg_dir.mkdir(exist_ok=True)

            for t, mask in enumerate(ctc_masks):
                tifffile.imwrite(str(seg_dir / f"man_seg{t:04}.tif"), mask)

            # store tracking masks
            track_dir = ann_dir / "TRA"
            track_dir.mkdir(exist_ok=True)

            for t, mask in enumerate(ctc_masks):
                tifffile.imwrite(str(track_dir / f"man_track{t:04}.tif"), mask)

            with open(
                str(track_dir / "man_track.txt"), "w", encoding="utf-8"
            ) as output_file:
                output_file.write("\n".join(ctc_tracking_format))


class CellTrackingDatasetResult:
    """
    Utility class to create segmentation/tracking output in form of the Cell Tracking Challenge (CTC)
    """

    def __init__(self):
        self.sources = []

    def add(self, content: tuple[TrackingSource, tuple[int, int]]):
        self.sources.append(content)

    def write(self, base_folder: str | Path = "data", offset=0):
        base_folder = Path(base_folder)

        base_folder.mkdir(exist_ok=True, parents=True)

        for i, (tracking_source, (height, width)) in enumerate(self.sources):
            mode = "RES"

            ann_dir = base_folder / f"{i + offset:02}_{mode}"

            if ann_dir.exists():
                shutil.rmtree(ann_dir)

            ann_dir.mkdir()

            tracking_helper = CTCTrackingHelper(
                tracking_source.overlay, tracking_source.tracking_graph, height, width
            )
            ctc_masks, ctc_tracking_format = tracking_helper.to_ctc_format()

            for t, mask in enumerate(ctc_masks):
                tifffile.imwrite(str(ann_dir / f"mask{t:04}.tif"), mask)

            with open(
                str(ann_dir / "res_track.txt"), "w", encoding="utf-8"
            ) as output_file:
                output_file.write("\n".join(ctc_tracking_format))


class CTCTrackingHelper:
    """Helper class for the CTC format generation"""

    def __init__(
        self, overlay: Overlay, tracking_graph: nx.DiGraph, height: int, width: int
    ):
        """Create a new tracking helper object

        Args:
            tracking_graph (nx.DiGraph): tracking graph consisting of contour nodes
        """
        self.contour_lookup = {cont.id: cont for cont in overlay}
        self.tracking_graph = tracking_graph
        # compute the life-cycles of individual cells
        self.life_cycles = CTCTrackingHelper.compute_life_cycles(self.tracking_graph)
        # create lookup (cont id --> life cycle index)
        self.life_cycle_lookup = CTCTrackingHelper.create_life_cycle_lookup(
            self.life_cycles
        )
        self.overlay = overlay
        self.height = height
        self.width = width

    def to_ctc_format(self) -> tuple[list[np.ndarray], list[str]]:
        """Generate CTC format

        Returns:
            Tuple[List[np.ndarray], List[str]]: Returns a List of segm/tracking masks and a list of lines for the ctc txt tracking format.
        """

        # generate txt format
        ctc_tracking_format = CTCTrackingHelper.txt_format(
            self.tracking_graph,
            self.life_cycles,
            self.contour_lookup,
            self.life_cycle_lookup,
        )

        ctc_masks = []
        # generate masks
        for overlay in self.overlay.timeIterator():
            mask = CTCTrackingHelper.convert_overlay_to_ctc_mask(
                overlay, self.life_cycle_lookup, self.height, self.width
            )
            ctc_masks.append(mask)

        return ctc_masks, ctc_tracking_format

    @staticmethod
    def compute_life_cycles(tracking_graph: nx.DiGraph) -> list[list[str]]:
        """Track life cycles of contour observations (from birth to division).

        Args:
            tracking_graph (nx.DiGraph): tracking graph

        Returns:
            List[List[str]]: List of life cycles (consisting of a list of contour ids)
        """

        start_nodes = deque(
            filter(lambda n: tracking_graph.in_degree(n) == 0, tracking_graph.nodes)
        )

        life_cycles = []

        while len(start_nodes) > 0:
            node = start_nodes.pop()
            life_cycle = [node]
            while tracking_graph.out_degree(node) == 1:
                node = next(tracking_graph.successors(node))
                life_cycle.append(node)

            life_cycles.append(life_cycle)

            if tracking_graph.out_degree(node) > 1:
                start_nodes += list(tracking_graph.successors(node))

        return life_cycles

    @staticmethod
    def txt_format(
        tracking_graph: nx.DiGraph,
        life_cycles: list[list[any]],
        contour_lookup,
        life_cycle_lookup,
    ) -> list[str]:
        """Generate CTC tracking txt format

        Args:
            tracking_graph (nx.DiGraph): tracking graph
            life_cycles (List[List[str]]): life_cycles

        Raises:
            Exception: when a life_cycle has multiple parents

        Returns:
            List[str]: list of ctc tracking lines
        """
        lines = []
        for i, life_cycle in enumerate(life_cycles):
            start_frame = contour_lookup[life_cycle[0]].frame
            end_frame = contour_lookup[life_cycle[-1]].frame

            # get the parent and its life_cycle number
            parents = list(tracking_graph.predecessors(life_cycle[0]))
            if len(parents) == 0:
                parent = 0  # TODO: default value when no parent???
            elif len(parents) == 1:
                parent = (
                    life_cycle_lookup[parents[0]] + 1
                )  # life-cycle enumeration starts with 1 #list(filter(lambda el: el[[-1] == parents[0], enumerate(life_cycles)))[0]
            else:
                raise Exception(
                    "Multiple parents not supported by Cell Tracking Challenge format!"
                )

            lines.append(
                f"{i + 1} {start_frame} {end_frame} {parent}"
            )  # life-cycle enumeration starts with one

        return lines

    @staticmethod
    def create_life_cycle_lookup(life_cycles: list[list[Contour]]):
        """Computes a mapping from contour ids to life cycle ids

        Args:
            life_cycles (List[List[Contour]]): _description_

        Returns:
            _type_: _description_
        """
        contour_life_cycle_lookup = {}
        for i, life_cycle in enumerate(life_cycles):
            for cont_id in life_cycle:
                contour_life_cycle_lookup[cont_id] = (
                    i  # life cycle enumeration starts with 1
                )

        return contour_life_cycle_lookup

    @staticmethod
    def convert_overlay_to_ctc_mask(
        overlay: Overlay,
        contour_life_cycle_lookup: dict[str, int],
        height: int,
        width: int,
    ) -> list[np.ndarray]:
        """Creates a ctc masks with correct numbering for a frame overlay

        Args:
            overlay (Overlay): overlay containing the contours
            contour_life_cycle_lookup (Dict[str, int]): lookup for the life cycle

        Returns:
            List[np.ndarray]: list of ctc masks
        """
        assert height > 0 and width > 0

        image_mask = np.zeros((height, width), dtype=np.uint16)
        for cont in overlay:
            life_cycle_id = (
                contour_life_cycle_lookup[cont.id] + 1
            )  # lifecycle ids must start with 1
            cont_mask = cont.toMask(height, width)
            image_mask = np.maximum(
                image_mask, (cont_mask * life_cycle_id).astype(np.uint16)
            )

        return image_mask

    @staticmethod
    def __load_masks(mask_path: Path) -> Overlay:
        """Load CTC masks into an overlay

        Args:
            mask_path (Path): Path to the folder containing the masks

        Returns:
            Overlay: Overlay containing all cell detections
        """

        mask_path = Path(mask_path)

        # load masks
        mask_files = sorted(mask_path.glob("man_seg*.tif"))
        masks = [tifffile.imread(file) for file in mask_files]

        all_polygons = map(
            multi_mask_to_polygons, tqdm(masks, desc="Convert masks to polygon...")
        )

        contours = []

        for frame, frame_polygons in enumerate(
            tqdm(all_polygons, desc="Convert to overlay...")
        ):
            for id, poly in frame_polygons:
                if isinstance(poly, MultiPolygon):
                    # if it comes to parsing problems take the polygon with the largest area
                    polygons = list(poly.geoms)
                    areas = [p.area for p in polygons]
                    poly = polygons[np.argmax(areas)]

                cc = np.stack(poly.exterior.coords.xy, axis=-1)
                contours.append(Contour(cc, -1, frame, f"{frame}_{id}"))

        return Overlay(contours, frames=list(range(0, len(masks))))

    @staticmethod
    def __load_tracking(tracking_path: Path) -> nx.DiGraph:
        """Load tracking from CTC txt file

        Args:
            tracking_path (Path): path to the folder containing the man_track.txt file

        Returns:
            nx.DiGraph: tracking graph with nodes and edges of every cell detection
        """

        track_annotation = pd.read_csv(
            tracking_path / "man_track.txt",
            delimiter=" ",
            names=["track_id", "start_frame", "end_frame", "parent_id"],
            header=None,
        )

        tracking_graph = nx.DiGraph()
        lc_lookup = {}

        for _, row in track_annotation.iterrows():
            start_frame = row["start_frame"]
            end_frame = row["end_frame"]
            id = row["track_id"]
            parent_id = row["parent_id"]

            node_items = [
                f"{frame}_{id}" for frame in range(start_frame, end_frame + 1)
            ]

            # add trajectory nodes
            for frame in range(start_frame, end_frame + 1):
                lc_lookup[f"{frame}_{id}"] = id
                tracking_graph.add_node(f"{frame}_{id}", frame=frame)

            # add trajectory edges
            for a, b in zip(node_items, node_items[1:], strict=False):
                tracking_graph.add_edge(a, b)

            # add edge to parent (0 means no parent)
            if parent_id != 0:
                tracking_graph.add_edge(
                    f"{start_frame - 1}_{parent_id}", f"{start_frame}_{id}"
                )

        return tracking_graph

    @staticmethod
    def load_ctc_format(
        segmentation_path: Path, tracking_path: Path
    ) -> tuple[Overlay, nx.DiGraph]:
        """Load ctc format from paths

        Args:
            segmentation_path (Path): path to the segmentation masks
            tracking_path (Path): path to the tracking gt/st

        Returns:
            tuple[Overlay, nx.DiGraph]: the overlay and tracking graph
        """

        # load segmentation overlay
        overlay = CTCTrackingHelper.__load_masks(mask_path=segmentation_path)

        # load tracking overlay
        tracking_graph = CTCTrackingHelper.__load_tracking(tracking_path=tracking_path)

        # Remove items only available in segmentation or tracking

        seg_ids = list(cont.id for cont in overlay)
        track_ids = set(tracking_graph.nodes())

        id_sym_difference = set(seg_ids).symmetric_difference(track_ids)

        # remove nodes in graph
        nodes_to_remove = set(tracking_graph.nodes()).intersection(id_sym_difference)
        for n in nodes_to_remove:
            tracking_graph.remove_node(n)

        # remove nodes in overlay
        conts_to_remove = set(seg_ids).intersection(id_sym_difference)
        for cont_id in conts_to_remove:
            seg_ids = np.array([cont.id for cont in overlay])
            index = np.argmin(seg_ids == cont_id)
            del overlay.contours[index]

        assert (
            len(
                {cont.id for cont in overlay}.symmetric_difference(
                    tracking_graph.nodes()
                )
            )
            == 0
        )

        return overlay, tracking_graph
