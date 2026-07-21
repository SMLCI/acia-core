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


if __name__ == "__main__":
    unittest.main()
