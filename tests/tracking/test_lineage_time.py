"""Real-time propagation into the lineage graphs.

Covers the "calibration leak" fix: a time-calibrated overlay must carry real
time (``cont.time``) into :func:`acia.tracking.ctc_track_graph`'s nodes, and
:func:`acia.tracking.annotate_tracklet_times` must stamp ``start_time``/
``end_time`` on the tracklet graph -- so downstream lineage plots read real
time off the nodes without the caller re-supplying any timepoints.
"""

import unittest

import networkx as nx
import numpy as np

from acia.base import Contour, Overlay
from acia.tracking import annotate_tracklet_times, ctc_track_graph


def _square(cx, cy, r=1.0):
    return np.array(
        [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]],
        dtype=np.float32,
    )


def build_overlay():
    """One cell (label 1) tracked across frames 0..2 -> ids 0,1,2."""
    conts = [
        Contour(_square(0, 0), score=1.0, frame=f, id=f, label=1) for f in range(3)
    ]
    return Overlay(conts, frames=[0, 1, 2])


def build_tracklet_graph():
    g = nx.DiGraph()
    g.add_node(1, start_frame=0, end_frame=2)
    return g


class TestCtcTrackGraphTime(unittest.TestCase):
    def test_calibrated_overlay_stamps_node_time(self):
        # 5-minute frame interval -> timepoints 0, 5, 10 min
        ov = build_overlay().with_frame_interval("5 min")
        tracklet_graph = build_tracklet_graph()

        g = ctc_track_graph(ov, tracklet_graph)

        self.assertEqual(g.nodes[0]["frame"], 0)
        self.assertEqual(g.nodes[0]["time"], 0.0)
        self.assertEqual(g.nodes[1]["time"], 5.0)
        self.assertEqual(g.nodes[2]["time"], 10.0)
        # graph-level unit lets the plot auto-label the axis
        self.assertEqual(g.graph["time_unit"], "min")

    def test_uncalibrated_overlay_has_no_time(self):
        ov = build_overlay()  # no time model
        g = ctc_track_graph(ov, build_tracklet_graph())

        self.assertEqual(g.nodes[0]["frame"], 0)
        self.assertNotIn("time", g.nodes[0])
        self.assertNotIn("time_unit", g.graph)


class TestAnnotateTrackletTimes(unittest.TestCase):
    def test_stamps_start_end_time(self):
        ov = build_overlay().with_frame_interval("5 min")
        tracklet_graph = build_tracklet_graph()

        annotate_tracklet_times(tracklet_graph, ov.timepoints)

        self.assertEqual(tracklet_graph.nodes[1]["start_time"], 0.0)
        self.assertEqual(tracklet_graph.nodes[1]["end_time"], 10.0)
        self.assertEqual(tracklet_graph.graph["time_unit"], "min")

    def test_none_timepoints_is_noop(self):
        tracklet_graph = build_tracklet_graph()
        annotate_tracklet_times(tracklet_graph, None)

        self.assertNotIn("start_time", tracklet_graph.nodes[1])
        self.assertNotIn("time_unit", tracklet_graph.graph)


if __name__ == "__main__":
    unittest.main()
