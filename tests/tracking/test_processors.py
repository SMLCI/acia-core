"""Unit test cases for tracking processors"""

import unittest
from pathlib import Path

import numpy as np
import pytest

# needs the trackastra heavy dependency + the external ctc_format dataset
pytest.importorskip("trackastra")

from acia.segm.local import InMemorySequenceSource  # noqa: E402
from acia.tracking.formats import read_ctc_tracking  # noqa: E402

# from acia.tracking.processor.laptrack import LAPTracker
from acia.tracking.processor.trackastra import TrackastraTracker  # noqa: E402

pytestmark = pytest.mark.integration


class TestTrackingProcessors(unittest.TestCase):
    """Test the linearization of z and t stacks"""

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

        # Just generate an empty image sequence here! It is not about doing a correct tracking!
        self.images = InMemorySequenceSource(np.zeros((30, 677, 426), dtype=np.uint8))

    def test_trackastra(self):
        """Test subsampling of a non-dividing contour"""

        input_path = Path("test_resources") / "ctc_format"

        ov, _, _ = read_ctc_tracking(input_path=input_path)

        tracker = TrackastraTracker()

        # Just make sure we can execute the method
        tracker(self.images, ov)


#    def test_lap(self):
#        """Test subsampling of a non-dividing contour"""
#
#        input_path = Path("test_resources") / "ctc_format"
#
#        ov, _, _ = read_ctc_tracking(input_path=input_path)
#
#        tracker = LAPTracker()
#
#        # Just make sure we can execute the method
#        tracker(self.images, ov)
