"""LAP tracking processor: https://doi.org/10.1093/bioinformatics/btac799"""

import logging
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import tifffile
from laptrack import OverLapTrack

from acia.attribute import attribute_tracking
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from ..formats import read_ctc_tracking
from . import TrackingProcessor
from .utils import overlay_to_masks


class LAPTracker(TrackingProcessor):
    """Processor for LAP tracking"""

    def __init__(
        self,
        track_cost_cutoff=0.9,
        track_dist_metric_coefs=(1.0, -1.0, 0.0, 0.0, 0.0),
        gap_closing_dist_metric_coefs=(1.0, -1.0, 0.0, 0.0, 0.0),
        gap_closing_max_frame_count=1,
        splitting_cost_cutoff=0.9,
        splitting_dist_metric_coefs=(1.0, 0.0, 0.0, 0.0, -1.0),
    ):
        """Configure LAP tracker

        For the parameter configuration please refer to https://github.com/yfukai/laptrack/blob/main/docs/examples/overlap_tracking.ipynb
        """
        # define the overlap based tracking
        self.olt = OverLapTrack(
            track_cost_cutoff=track_cost_cutoff,
            track_dist_metric_coefs=track_dist_metric_coefs,
            gap_closing_dist_metric_coefs=gap_closing_dist_metric_coefs,
            gap_closing_max_frame_count=gap_closing_max_frame_count,
            splitting_cost_cutoff=splitting_cost_cutoff,
            splitting_dist_metric_coefs=splitting_dist_metric_coefs,
        )

    @staticmethod
    def __export(output_path, track_df, split_df, labels):
        output_path.mkdir(parents=True, exist_ok=True)

        df = track_df.reset_index()

        # relabel the cells in the segmentation masks
        new_labels = []

        for frame in np.unique(df["frame"]):
            frame_df = df[df["frame"] == frame]

            new_label = np.zeros_like(labels[frame])
            for _, row in frame_df.iterrows():
                label = row["label"]
                track_id = row["track_id"] + 1
                mask = labels[frame] == label
                new_label[mask] = track_id

            new_labels.append(new_label)

        # output the relabeled segmentation masks
        for i, new_label in enumerate(new_labels):
            tifffile.imwrite(
                output_path / f"t{i:04d}.tif", new_label, compression="zlib"
            )

        # Make the cell tracking format

        res = []

        for track_id in np.unique(df["track_id"]):
            frames = df[df["track_id"] == track_id]["frame"]
            start_frame = np.min(frames)
            last_frame = np.max(frames)

            if len(split_df) > 0:
                lookup = split_df[split_df["child_track_id"] == track_id]
            else:
                lookup = []

            if len(lookup) > 0:
                parent = lookup["parent_track_id"].item() + 1
            else:
                parent = 0

            res.append((track_id + 1, start_frame, last_frame, parent))

        with open(output_path / "man_track.txt", "w", encoding="utf-8") as csvfile:
            for row in res:
                csvfile.write(" ".join(map(str, row)) + "\n")

    def __call__(self, images: ImageSequenceSource, segmentation: Overlay):
        image = next(iter(images)).raw
        height, width = image.shape[:2]

        masks = overlay_to_masks(segmentation, height=height, width=width)

        if segmentation.numFrames() != len(masks):
            logging.warning("Number of segmented frames and masks is unequal!")

        track_df, split_df, _ = self.olt.predict_overlap_dataframe(masks)

        with tempfile.TemporaryDirectory() as td:
            self.__export(td, track_df, split_df, masks)

            ov, tracklet_graph, tracking_graph = read_ctc_tracking(td)

            attribute_tracking(ov, tracking_graph, tracking_graph, self)

            return ov, tracklet_graph, tracking_graph


class LAPTracker2(TrackingProcessor):
    """Processor for LAP tracking"""

    def __init__(
        self,
        laptrack_params,
    ):
        """Configure LAP tracker

        For the parameter configuration please refer to https://github.com/yfukai/laptrack/blob/main/docs/examples/overlap_tracking.ipynb
        """
        # define the overlap based tracking
        self.olt = OverLapTrack(**laptrack_params)

    @staticmethod
    def __export(output_path, track_df, split_df, labels):
        output_path.mkdir(parents=True, exist_ok=True)

        df = track_df.reset_index()

        # relabel the cells in the segmentation masks
        new_labels = []

        for frame in np.unique(df["frame"]):
            frame_df = df[df["frame"] == frame]

            new_label = np.zeros_like(labels[frame])
            for _, row in frame_df.iterrows():
                label = row["label"]
                track_id = row["track_id"] + 1
                mask = labels[frame] == label
                new_label[mask] = track_id

            new_labels.append(new_label)

        # output the relabeled segmentation masks
        for i, new_label in enumerate(new_labels):
            tifffile.imwrite(
                output_path / f"t{i:04d}.tif", new_label, compression="zlib"
            )

        # Make the cell tracking format

        res = []

        for track_id in np.unique(df["track_id"]):
            frames = df[df["track_id"] == track_id]["frame"]
            start_frame = np.min(frames)
            last_frame = np.max(frames)

            if len(split_df) > 0:
                lookup = split_df[split_df["child_track_id"] == track_id]
            else:
                lookup = []

            if len(lookup) > 0:
                parent = lookup["parent_track_id"].item() + 1
            else:
                parent = 0

            res.append((track_id + 1, start_frame, last_frame, parent))

        with open(output_path / "man_track.txt", "w", encoding="utf-8") as csvfile:
            for row in res:
                csvfile.write(" ".join(map(str, row)) + "\n")

    def __call__(self, images: ImageSequenceSource, segmentation: Overlay):
        image = next(iter(images)).raw
        height, width = image.shape[:2]

        masks = overlay_to_masks(segmentation, height=height, width=width)

        if segmentation.numFrames() != len(masks):
            logging.warning("Number of segmented frames and masks is unequal!")

        track_df, split_df, _ = self.olt.predict_overlap_dataframe(masks)

        with tempfile.TemporaryDirectory() as td:
            self.__export(Path(td), track_df, split_df, masks)

            ov, tracklet_graph, tracking_graph = read_ctc_tracking(td)

            return ov, tracklet_graph, tracking_graph


class LaptrackTracker(TrackingProcessor):
    """Processor for LAP tracking according to https://github.com/yfukai/laptrack/blob/main/docs/examples/overlap_tracking.ipynb"""

    def __init__(
        self,
        laptrack_params,
    ):
        """Configure LAP tracker

        For the parameter configuration please refer to https://github.com/yfukai/laptrack/blob/main/docs/examples/overlap_tracking.ipynb
        """
        # define the overlap based tracking
        self.olt = OverLapTrack(**laptrack_params)

    def __call__(self, images: ImageSequenceSource, segmentation: Overlay):
        image = next(iter(images)).raw
        height, width = image.shape[:2]

        mask_stack = overlay_to_masks(segmentation, height=height, width=width)

        track_df, split_df, _ = self.olt.predict_overlap_dataframe(mask_stack)

        tracking_ov = overlay_from_masks(mask_stack)

        label_lookup = {}
        for track_id, track_id_df in track_df.groupby("track_id"):
            label_lookup[track_id] = track_id_df.iloc[0].name[1]

        split_df["parent_track_label"] = split_df["parent_track_id"].apply(
            lambda id: label_lookup[id]
        )
        split_df["child_track_label"] = split_df["child_track_id"].apply(
            lambda id: label_lookup[id]
        )

        # create tracklet graph
        tracklet_graph = nx.DiGraph()

        for _, row in split_df.iterrows():
            tracklet_nodes = np.unique(track_df.reset_index()["label"])
            tracklet_graph.add_nodes_from(tracklet_nodes)
            tracklet_graph.add_edge(row["parent_track_label"], row["child_track_label"])

        # create tracking graph
        tracking_graph = nx.DiGraph()

        frame_label_lookup = {(cont.frame, cont.label): cont for cont in tracking_ov}

        track_sequences = {}

        for _track_id, track_id_df in track_df.groupby("track_id"):
            track_seq = [(frame, label) for (frame, label), _ in track_id_df.iterrows()]
            label = track_id_df.iloc[0].name[1]
            track_sequences[label] = track_seq

        for label, track_seq in track_sequences.items():
            # add sequence
            for a, b in zip(track_seq[0:-1], track_seq[1:], strict=False):
                tracking_graph.add_edge(
                    frame_label_lookup[a].id, frame_label_lookup[b].id
                )

            # add divisions
            for parent_label in tracklet_graph.predecessors(label):
                # lookup the contours
                a = frame_label_lookup[track_sequences[parent_label][-1]].id
                b = frame_label_lookup[track_seq[0]].id

                print(f"{parent_label} -> {label}")
                print(f"{a} -> {b}")

                tracking_graph.add_edge(a, b)

        return tracking_ov, tracklet_graph, tracking_graph
