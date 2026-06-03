"""Unit test cases for tracking subsampling"""

import unittest
from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
import tifffile

from acia.segm.local import LocalImage
from acia.tracking.formats import read_ctc_tracking, write_ctc_tracking

# reads/writes the external ctc_format dataset under test_resources/
pytestmark = pytest.mark.integration


class TestSubsampling(unittest.TestCase):
    """Test the linearization of z and t stacks"""

    def __test_tracking(self, ov, tracklet_graph, tracking_graph):
        self.assertEqual(len(ov), 67)

        self.assertEqual(len(tracklet_graph.nodes), 10)
        self.assertEqual(len(tracklet_graph.edges), 8)

        self.assertEqual(len(tracking_graph.nodes), 67)
        self.assertEqual(len(tracking_graph.edges), 65)

    def test_read_write(self):
        """Test subsampling of a non-dividing contour"""

        input_path = Path("test_resources") / "ctc_format"

        ov, tracklet_graph, tracking_graph = read_ctc_tracking(input_path=input_path)

        total_area = np.sum([cont.area for cont in ov])

        self.__test_tracking(ov, tracklet_graph, tracking_graph)

        output_path = Path("tracking_output")
        if output_path.exists():
            rmtree(output_path)
        output_path.mkdir()

        images = [LocalImage(tifffile.imread(input_path / "man_track0000.tif"))]

        write_ctc_tracking(output_path, images, ov, tracklet_graph)

        ov2, tracklet_graph2, tracking_graph2 = read_ctc_tracking(
            input_path=output_path
        )

        total_area2 = np.sum([cont.area for cont in ov2])

        self.assertEqual(total_area, total_area2)

        self.__test_tracking(ov2, tracklet_graph2, tracking_graph2)


if __name__ == "__main__":
    unittest.main()
