"""Tests for :func:`acia.viz.tracklet_graph_to_segments`/:func:`acia.viz.plot_tracklet_lineage`."""

import unittest

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; no display required

import networkx as nx
import plotly.graph_objects as go

from acia.viz import (
    plot_tracklet_lineage,
    plotly_cell_lineage,
    tracklet_graph_to_segments,
)


def build_tracklet_graph() -> nx.DiGraph:
    """3 tracklets: one root (5) with two children (6, 7) -- algorithm-notes.md's worked example."""
    g = nx.DiGraph()
    g.add_node(5, start_frame=0, end_frame=3)
    g.add_node(6, start_frame=4, end_frame=7)
    g.add_node(7, start_frame=4, end_frame=6)
    g.add_edge(5, 6)
    g.add_edge(5, 7)
    return g


class TestTrackletGraphToSegments(unittest.TestCase):
    """Segment-structure correctness against the hand-built synthetic graph."""

    def test_segment_structure(self):
        g = build_tracklet_graph()
        segments = tracklet_graph_to_segments(g)

        expected_nodes = {
            (5, "start"): 0,
            (5, "end"): 3,
            (6, "start"): 4,
            (6, "end"): 7,
            (7, "start"): 4,
            (7, "end"): 6,
        }
        self.assertEqual(set(segments.nodes), set(expected_nodes))
        for node, frame in expected_nodes.items():
            self.assertEqual(segments.nodes[node]["frame"], frame)

        expected_edges = {
            ((5, "start"), (5, "end")),
            ((6, "start"), (6, "end")),
            ((7, "start"), (7, "end")),
            ((5, "end"), (6, "start")),
            ((5, "end"), (7, "start")),
        }
        self.assertEqual(set(segments.edges), expected_edges)

    def test_no_extra_nodes_or_edges(self):
        g = build_tracklet_graph()
        segments = tracklet_graph_to_segments(g)

        self.assertEqual(segments.number_of_nodes(), 2 * g.number_of_nodes())
        # one intra-tracklet edge per tracklet + one inter-tracklet edge per division edge
        self.assertEqual(
            segments.number_of_edges(), g.number_of_nodes() + g.number_of_edges()
        )

    def test_renders_via_plotly_cell_lineage_directly(self):
        g = build_tracklet_graph()
        segments = tracklet_graph_to_segments(g)

        fig = plotly_cell_lineage(segments, time_feature="frame")
        self.assertIsInstance(fig, go.Figure)

    def test_start_end_time_attaches_real_time(self):
        # tracklet nodes carrying real time (as stamped by
        # annotate_tracklet_times) forward it to the point nodes' "time" attr
        g = build_tracklet_graph()
        for _n, a in g.nodes(data=True):
            a["start_time"] = a["start_frame"] * 5.0
            a["end_time"] = a["end_frame"] * 5.0
        g.graph["time_unit"] = "min"

        segments = tracklet_graph_to_segments(g)

        # frame attribute is still present, plus a real-time "time" attribute
        self.assertEqual(segments.nodes[(5, "start")]["frame"], 0)
        self.assertEqual(segments.nodes[(5, "start")]["time"], 0.0)
        self.assertEqual(segments.nodes[(5, "end")]["time"], 15.0)
        self.assertEqual(segments.nodes[(6, "end")]["time"], 35.0)
        # graph-level unit is forwarded so the axis can auto-label
        self.assertEqual(segments.graph["time_unit"], "min")

    def test_no_time_attribute_without_start_end_time(self):
        g = build_tracklet_graph()
        segments = tracklet_graph_to_segments(g)
        self.assertNotIn("time", segments.nodes[(5, "start")])
        self.assertNotIn("time_unit", segments.graph)


class TestPlotTrackletLineage(unittest.TestCase):
    """Wrapper's one-call render and kwargs forwarding."""

    def test_renders_in_one_call(self):
        g = build_tracklet_graph()
        fig = plot_tracklet_lineage(g)
        self.assertIsInstance(fig, go.Figure)

    def test_equivalent_to_manual_composition(self):
        g = build_tracklet_graph()
        wrapped_fig = plot_tracklet_lineage(g)
        manual_fig = plotly_cell_lineage(
            tracklet_graph_to_segments(g), time_feature="frame"
        )
        self.assertEqual(len(wrapped_fig.data), len(manual_fig.data))

    def test_mark_births_kwarg_forwarded(self):
        g = build_tracklet_graph()

        fig_without = plot_tracklet_lineage(g, mark_births=False)
        fig_with = plot_tracklet_lineage(g, mark_births=True)

        names_without = {trace.name for trace in fig_without.data}
        names_with = {trace.name for trace in fig_with.data}

        self.assertNotIn("Cell birth", names_without)
        self.assertIn("Cell birth", names_with)

    def test_time_feature_not_overridable(self):
        g = build_tracklet_graph()
        with self.assertRaises(TypeError):
            plot_tracklet_lineage(g, time_feature="frame")

    def test_real_time_drives_axis_and_autolabels(self):
        g = build_tracklet_graph()
        for _n, a in g.nodes(data=True):
            a["start_time"] = a["start_frame"] * 5.0
            a["end_time"] = a["end_frame"] * 5.0
        g.graph["time_unit"] = "min"

        fig = plot_tracklet_lineage(g)
        self.assertIsInstance(fig, go.Figure)
        # axis auto-labels from the graph's unit -- no caller input needed
        self.assertEqual(fig.layout.xaxis.title.text, "Time [min]")
        # x positions span real minutes (max end_frame 7 -> 35 min), not the
        # raw frame indices (max 7)
        max_x = max(x for trace in fig.data if trace.x is not None for x in trace.x)
        self.assertEqual(max_x, 35.0)

    def test_falls_back_to_frame_without_time(self):
        g = build_tracklet_graph()
        fig = plot_tracklet_lineage(g)
        # no real time -> generic "Time" title and frame-indexed x positions
        self.assertEqual(fig.layout.xaxis.title.text, "Time")
        max_x = max(x for trace in fig.data if trace.x is not None for x in trace.x)
        self.assertEqual(max_x, 7.0)


if __name__ == "__main__":
    unittest.main()
