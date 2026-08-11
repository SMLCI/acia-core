"""Round-trip tests for :func:`acia.tracking.formats.save_tracking` / ``load_tracking``.

Two defects motivate these wrappers, and each has a dedicated regression here:

* ``load_tracking`` must attach the time model **before** building the tracking
  graph -- :func:`acia.tracking.formats.ctc_track_graph` stamps node ``time``
  from ``cont.time``, so :func:`read_ctc_tracking` (which builds the graph
  internally, while the reloaded overlay is still uncalibrated) yields a
  timeless graph that a lineage plot cannot lay out on a real-time axis.
* ``save_tracking`` must write one mask per image frame starting at frame 0 --
  ``write_ctc_tracking`` enumerates ``timeIterator()``, which starts at the
  first *populated* frame, so an overlay with an empty frame 0 writes a stack
  shifted against the movie.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np

from acia.base import Contour, Overlay
from acia.segm.local import THWCSequenceSource
from acia.tracking.formats import load_tracking, read_ctc_tracking, save_tracking


def _square(cx, cy, r=3.0):
    return np.array(
        [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]],
        dtype=np.float32,
    )


def _source(num_frames=4, **calibration):
    return THWCSequenceSource(
        np.zeros((num_frames, 32, 32, 1), dtype=np.uint8), **calibration
    )


def _tracked(frames=(0, 1, 2, 3), label=1):
    """One cell (tracklet ``label``) present in each requested frame."""
    overlay = Overlay(
        [
            Contour(_square(10, 10), score=1.0, frame=f, id=f, label=label)
            for f in frames
        ]
    )
    graph = nx.DiGraph()
    graph.add_node(label, start_frame=min(frames), end_frame=max(frames))
    return overlay, graph


def _dividing():
    """Mother tracklet 1 (frames 0-1) dividing into daughters 2 and 3 (frames 2-3)."""
    contours = [
        Contour(_square(10, 10), score=1.0, frame=f, id=f, label=1) for f in (0, 1)
    ]
    contours += [
        Contour(_square(6, 10), score=1.0, frame=f, id=10 + f, label=2) for f in (2, 3)
    ]
    contours += [
        Contour(_square(20, 20), score=1.0, frame=f, id=20 + f, label=3) for f in (2, 3)
    ]

    graph = nx.DiGraph()
    graph.add_node(1, start_frame=0, end_frame=1)
    graph.add_node(2, start_frame=2, end_frame=3)
    graph.add_node(3, start_frame=2, end_frame=3)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)

    return Overlay(contours, frames=[0, 1, 2, 3]), graph


class TestTrackingRoundTrip(unittest.TestCase):
    def test_preserves_tracklet_topology(self):
        overlay, tracklet_graph = _dividing()
        source = _source()

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )
            loaded_ov, loaded_tracklets, loaded_tracking = load_tracking(path, source)

        self.assertEqual(set(loaded_tracklets.nodes), {1, 2, 3})
        self.assertEqual(set(loaded_tracklets.edges), {(1, 2), (1, 3)})
        for node in (1, 2, 3):
            self.assertEqual(
                loaded_tracklets.nodes[node]["start_frame"],
                tracklet_graph.nodes[node]["start_frame"],
            )
            self.assertEqual(
                loaded_tracklets.nodes[node]["end_frame"],
                tracklet_graph.nodes[node]["end_frame"],
            )

        self.assertEqual(len(loaded_ov), len(overlay))
        # per-detection graph: 6 detections, 3 intra-tracklet + 2 division edges
        self.assertEqual(len(loaded_tracking.nodes), 6)
        self.assertEqual(len(loaded_tracking.edges), 5)

    def test_returns_the_tracker_triple_order(self):
        overlay, tracklet_graph = _tracked()
        source = _source()

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )
            result = load_tracking(path, source)

        self.assertEqual(len(result), 3)
        loaded_ov, loaded_tracklets, loaded_tracking = result
        self.assertIsInstance(loaded_ov, Overlay)
        self.assertIsInstance(loaded_tracklets, nx.DiGraph)
        self.assertIsInstance(loaded_tracking, nx.DiGraph)
        # the tracklet graph is the small one, the tracking graph the per-detection one
        self.assertLess(len(loaded_tracklets.nodes), len(loaded_tracking.nodes))

    def test_missing_directory_raises(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_tracking(Path(tmp) / "tracking", _source())

            self.assertIn("save_tracking", str(ctx.exception))

    def test_directory_without_man_track_raises(self):
        with TemporaryDirectory() as tmp:
            empty = Path(tmp) / "tracking"
            empty.mkdir()

            with self.assertRaises(FileNotFoundError) as ctx:
                load_tracking(empty, _source())

            self.assertIn("man_track.txt", str(ctx.exception))


class TestTrackingTimeReattached(unittest.TestCase):
    """Regression: the graph must be built *after* the calibration is attached."""

    def test_tracking_graph_nodes_carry_real_time(self):
        overlay, tracklet_graph = _tracked()
        source = _source(num_frames=4, frame_interval="5 min")

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )
            loaded_ov, loaded_tracklets, loaded_tracking = load_tracking(path, source)

        times = sorted(data["time"] for _, data in loaded_tracking.nodes(data=True))
        self.assertEqual(times, [0.0, 5.0, 10.0, 15.0])
        self.assertEqual(loaded_tracking.graph["time_unit"], "min")

        # the tracklet graph is stamped too, which is what doubling time reads
        self.assertEqual(loaded_tracklets.nodes[1]["start_time"], 0.0)
        self.assertEqual(loaded_tracklets.nodes[1]["end_time"], 15.0)

        np.testing.assert_allclose(loaded_ov.timepoints.magnitude, [0, 5, 10, 15])

    def test_read_ctc_tracking_does_not_provide_this(self):
        """Pins *why* load_tracking exists rather than delegating to read_ctc_tracking."""
        overlay, tracklet_graph = _tracked()
        source = _source(num_frames=4, frame_interval="5 min")

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )
            _, raw_tracklets, raw_tracking = read_ctc_tracking(path)

        # no source is involved, so nothing carries real time
        for _, data in raw_tracking.nodes(data=True):
            self.assertNotIn("time", data)
        self.assertNotIn("time_unit", raw_tracking.graph)
        self.assertNotIn("start_time", raw_tracklets.nodes[1])

    def test_uncalibrated_source_invents_no_time(self):
        overlay, tracklet_graph = _tracked()
        source = _source(num_frames=4)

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )
            loaded_ov, loaded_tracklets, loaded_tracking = load_tracking(path, source)

        self.assertIsNone(loaded_ov.timepoints)
        self.assertNotIn("time_unit", loaded_tracking.graph)
        self.assertNotIn("start_time", loaded_tracklets.nodes[1])


class TestTrackingFrameAlignment(unittest.TestCase):
    """Regression: an overlay whose frame 0 is empty must not shift the mask stack."""

    def test_leading_empty_frames_do_not_shift_the_stack(self):
        # cell only appears from frame 2 on; overlay carries no explicit frame list,
        # so timeIterator() would start at frame 2
        overlay = Overlay(
            [
                Contour(_square(10, 10), score=1.0, frame=f, id=f, label=1)
                for f in (2, 3)
            ]
        )
        tracklet_graph = nx.DiGraph()
        tracklet_graph.add_node(1, start_frame=2, end_frame=3)
        source = _source(num_frames=4, frame_interval="5 min")

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )

            self.assertEqual(len(sorted(path.glob("*.tif"))), 4)

            loaded_ov, _, loaded_tracking = load_tracking(path, source)

        # detections stay on frames 2 and 3 -- not shifted onto 0 and 1
        self.assertEqual(sorted(c.frame for c in loaded_ov), [2, 3])
        times = sorted(data["time"] for _, data in loaded_tracking.nodes(data=True))
        self.assertEqual(times, [10.0, 15.0])

    def test_caller_overlay_is_not_mutated(self):
        overlay = Overlay([Contour(_square(10, 10), score=1.0, frame=2, id=0, label=1)])
        tracklet_graph = nx.DiGraph()
        tracklet_graph.add_node(1, start_frame=2, end_frame=2)
        source = _source(num_frames=4)

        with TemporaryDirectory() as tmp:
            save_tracking(Path(tmp) / "tracking", source, overlay, tracklet_graph)

        self.assertEqual(list(overlay.frames()), [2])

    def test_trailing_empty_frames_are_written(self):
        overlay, tracklet_graph = _tracked(frames=(0, 1))
        source = _source(num_frames=4)

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )

            self.assertEqual(len(sorted(path.glob("*.tif"))), 4)

            loaded_ov, _, _ = load_tracking(path, source)

        self.assertEqual(loaded_ov.numFrames(), 4)
        self.assertEqual(sorted(c.frame for c in loaded_ov), [0, 1])

    def test_overlay_beyond_the_sequence_raises(self):
        overlay, tracklet_graph = _tracked(frames=(0, 1, 2, 3))

        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                save_tracking(
                    Path(tmp) / "tracking",
                    _source(num_frames=2),
                    overlay,
                    tracklet_graph,
                )

            self.assertIn("do not match", str(ctx.exception))


class TestDoublingTimeAfterReload(unittest.TestCase):
    """The end the whole handoff serves: analysis must be identical after a reload."""

    def test_identical_to_the_in_memory_result(self):
        from acia.analysis.doubling_time import compute_doubling_times

        overlay, tracklet_graph = _dividing()
        source = _source(num_frames=4, frame_interval="5 min")

        from acia.tracking import annotate_tracklet_times

        annotate_tracklet_times(tracklet_graph, source.timepoints)
        before = compute_doubling_times(tracklet_graph, source)

        with TemporaryDirectory() as tmp:
            path = save_tracking(
                Path(tmp) / "tracking", source, overlay, tracklet_graph
            )
            _, loaded_tracklets, _ = load_tracking(path, source)

        after = compute_doubling_times(loaded_tracklets, source)

        self.assertEqual(len(before), len(after))
        self.assertEqual(list(before.columns), list(after.columns))
        for column in before.columns:
            np.testing.assert_allclose(
                np.asarray(before[column].values, dtype=float),
                np.asarray(after[column].values, dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
