"""Unit test cases for tracking subsampling"""

import unittest

import networkx as nx
import numpy as np

from acia.base import Contour, Overlay
from acia.tracking import TrackingSource, TrackingSourceInMemory, subsample_tracking
from acia.tracking.utils import delete_nodes, subsample_lineage


class TestSubsampling(unittest.TestCase):
    """Test the linearization of z and t stacks"""

    @staticmethod
    def linear_track(
        length, frame_start, id_start=None, successors: list[TrackingSource] = None
    ):

        if successors is None:
            successors = []

        if id_start is None:
            id_start = 0
            for succ in successors:
                id_start = np.max([id_start, *[cont.id for cont in succ.overlay]]) + 1

        ids = [id_start + i for i in range(length)]
        frames = [frame_start + i for i in range(length)]

        overlay = Overlay(
            [Contour(None, -1, frame, id) for id, frame in zip(ids, frames, strict=False)]
        )
        tracking_graph = nx.DiGraph()
        tracking_graph.add_nodes_from([cont.id for cont in overlay])
        tracking_graph.add_edges_from(
            [(a.id, b.id) for a, b in zip(overlay.contours, overlay.contours[1:], strict=False)]
        )

        linear_ts = TrackingSourceInMemory(overlay, tracking_graph)

        for succ in successors:
            # merge the two trackings
            linear_ts.merge(succ)

            # add edge between last and first overlay contour
            linear_ts.tracking_graph.add_edge(
                overlay.contours[-1].id, succ.overlay.contours[0].id
            )

        return linear_ts

    def test_linear(self):
        """Test subsampling of a non-dividing contour"""

        # create tracking source
        tsim = TestSubsampling.linear_track(10, 0)

        # subsample tracking
        tr_source = subsample_tracking(tsim, 2)

        self.assertEqual(tr_source.tracking_graph.number_of_nodes(), 5)
        self.assertEqual(tr_source.tracking_graph.number_of_edges(), 4)

        self.assertSetEqual(
            set(map(lambda c: c.id, tr_source.overlay)), {0, 2, 4, 6, 8}
        )

    def test_division(self):
        """Test subsampling of a dividing contour"""

        # create tracking source
        tsim = TestSubsampling.linear_track(
            5,
            0,
            successors=[
                TestSubsampling.linear_track(3, 5, id_start=0),
                TestSubsampling.linear_track(5, 5, id_start=3),
            ],
        )

        # subsample tracking
        tr_source = subsample_tracking(tsim, 3)

        self.assertEqual(tr_source.tracking_graph.number_of_nodes(), 5)
        self.assertEqual(tr_source.tracking_graph.number_of_edges(), 4)
        self.assertSetEqual(
            set(map(lambda c: c.id, tr_source.overlay)), {8, 11, 1, 4, 7}
        )
        self.assertEqual(tr_source.tracking_graph.out_degree(11), 2)


def linear_track(start_frame: int, num_frames: int) -> nx.DiGraph:
    """Creates a tracklet for a single cell

    Args:
        start_frame (int): start frame
        num_frames (int): number of frames

    Returns:
        nx.DiGraph: Simple lineage with a single cell track (frame is node attribute)
    """

    lineage = nx.DiGraph()
    nodes = list(range(start_frame, start_frame + num_frames))

    lineage.add_nodes_from(nodes)
    for n in lineage.nodes:
        lineage.nodes[n]["frame"] = n

    for a, b in zip(nodes, nodes[1:], strict=False):
        lineage.add_edge(a, b)

    return lineage


class TestNewSubsampling(unittest.TestCase):
    """Test the linearization of z and t stacks"""

    def test_node_deletion(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3)])

        new_graph = delete_nodes(graph, [2])

        self.assertEqual(set(new_graph.nodes), {1, 3})
        self.assertEqual(set(new_graph.edges), {(1, 3)})

    def test_node_subsampling(self):
        lineage = linear_track(0, 11)

        sub_lineage = subsample_lineage(lineage, 5)

        self.assertEqual(set(sub_lineage.nodes), {0, 5, 10})
        self.assertEqual(set(sub_lineage.edges), {(0, 5), (5, 10)})

    def test_node_sub_with_div(self):
        lineage = nx.DiGraph()

        ########
        #           | 3 - 4 - 5
        # 0 - 1 - 2 - 6 - 7 - 8
        # 0 | 1 | 2 | 3 | 4 | 5 (frame index)

        # add nodes with frame information
        lineage.add_nodes_from(
            [
                (0, dict(frame=0)),
                (1, dict(frame=1)),
                (2, dict(frame=2)),
                (3, dict(frame=3)),
                (4, dict(frame=4)),
                (5, dict(frame=5)),
                (6, dict(frame=3)),
                (7, dict(frame=4)),
                (8, dict(frame=5)),
            ],
        )

        # add edges
        lineage.add_edges_from(
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (2, 7), (2, 8)]
        )

        # subsample lineage
        new_graph = subsample_lineage(lineage, 2)

        # check nodes and connectivity
        self.assertEqual({0, 2, 4, 7}, set(new_graph.nodes))
        self.assertEqual({(0, 2), (2, 4), (2, 7)}, set(new_graph.edges))


if __name__ == "__main__":
    unittest.main()
