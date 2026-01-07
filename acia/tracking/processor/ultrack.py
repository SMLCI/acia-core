""" Ultrack based tracking """

import networkx as nx
from ultrack import to_tracks_layer, track, tracks_to_zarr
from ultrack.utils import labels_to_contours

from acia.attribute import attribute_tracking
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import TrackingProcessor
from .utils import overlay_to_masks


class UltrackTracker(TrackingProcessor):
    """Processor for Ultrack: https://github.com/royerlab/ultrack/tree/main"""

    def __init__(self, config):
        """_summary_

        Args:
            config (_type_): Ultrack configuration
        """

        self.config = config

    def __call__(self, images: ImageSequenceSource, segmentation: Overlay):

        image = next(iter(images)).raw
        height, width = image.shape[:2]

        mask_stack = overlay_to_masks(segmentation, height=height, width=width)

        detection, edges = labels_to_contours(mask_stack, sigma=4.0)

        # perform the tracking
        track(
            foreground=detection,
            edges=edges,
            config=self.config,
            overwrite=True,
        )

        # convert back
        tracks_df, graph = to_tracks_layer(self.config)
        labels = tracks_to_zarr(self.config, tracks_df)

        # parse the tracking overlay from instance stack (TxHxW)
        tracking_ov = overlay_from_masks(labels)

        # create the tracklet graph
        tracklet_graph = nx.DiGraph()
        for child, parent in graph.items():
            tracklet_graph.add_edge(parent, child)

        # compute tracking graph

        label_contours = {}
        for cont in tracking_ov:
            label_contours[cont.label] = label_contours.get(cont.label, []) + [cont]

        tracking_graph = nx.DiGraph()

        for label, contours in label_contours.items():
            sorted(contours, key=lambda c: c.frame)
            # print([c.frame for c in contours])

            tracklet_graph.nodes[label]["start_frame"] = contours[0].frame
            tracklet_graph.nodes[label]["end_frame"] = contours[-1].frame

            tracking_graph.add_nodes_from([c.id for c in contours])

            # add sequential tracks
            for c1, c2 in zip(contours[:-1], contours[1:], strict=False):
                tracking_graph.add_edge(c1.id, c2.id)

            # add divisions
            parent_labels = list(tracklet_graph.predecessors(label))

            # add connection to every parent
            for pl in parent_labels:
                last_cont = label_contours[pl][-1]
                tracking_graph.add_edge(last_cont.id, contours[0].id)

        attribute_tracking(tracking_ov, tracklet_graph, tracking_graph, self)

        return tracking_ov, tracklet_graph, tracking_graph
