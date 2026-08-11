"""Testcases for single-cell property extractors"""

import unittest
from itertools import product

import numpy as np
import pytest

from acia import ureg
from acia.analysis import (
    AreaEx,
    CircularityEx,
    DynamicTimeEx,
    ExtractorExecutor,
    FluorescenceEx,
    FrameEx,
    LengthEx,
    LengthWidthEx,
    PerimeterEx,
    PositionEx,
    PropertyExtractor,
    TimeEx,
    WidthEx,
)
from acia.base import Contour, Overlay
from acia.segm.local import InMemorySequenceSource, LocalImageSource, THWCSequenceSource


class TestPropertyExtractors(unittest.TestCase):
    """Test cases for single-cell property extractors"""

    def test_unit_conversion(self):
        # test basic conversion patterns

        self.assertAlmostEqual(
            PropertyExtractor("test", "meter", "millimeter").convert(1), 1000
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "micrometer", "millimeter").convert(1), 1e-3
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "liter", "milliliter").convert(1), 1000
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "micrometer", "micrometer").convert(1), 1
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "meter", "micrometer").convert(1), 1e6
        )

    def test_extractors(self):
        # in x,y coordinates
        contours = [Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=0, id=23)]
        overlay = Overlay(contours)

        image = np.zeros((200, 200))
        image[0, 0] = 2
        image[0, 1] = 5
        image[1, 0] = 6
        image[1, 1] = 10
        image[2, 0:3] = 4
        image[0:2, 2] = 4
        image_source = LocalImageSource.from_array(image)

        # pixel size
        ps = 0.07

        # test basic extractors
        df = ExtractorExecutor().execute(
            overlay=overlay,
            images=image_source,
            extractors=[
                FrameEx(),
                AreaEx(input_unit=(ps * ureg.micrometer) ** 2),
                LengthEx(input_unit=ps * ureg.micrometer),
                WidthEx(input_unit=ps * ureg.micrometer),
                LengthWidthEx("a_", input_unit=ps * ureg.micrometer),
                TimeEx(input_unit="15 * minute"),  # one frame every 15 minutes
                PositionEx(input_unit=ps * ureg.micrometer),
                FluorescenceEx(channels=[0], channel_names=["gfp"], parallel=1),
                FluorescenceEx(
                    channels=[0],
                    channel_names=["gfp_mean"],
                    summarize_operator=np.mean,
                    parallel=1,
                ),
                PerimeterEx(input_unit=(ps * ureg.micrometer)),
                CircularityEx(),
            ],
        )

        self.assertEqual(df["area"].iloc[0], (2 * 3) * ps**2)
        self.assertEqual(df["length"].iloc[0], 3 * ps)
        self.assertEqual(df["a_length"].iloc[0], 3 * ps)
        self.assertEqual(df["width"].iloc[0], 2 * ps)
        self.assertEqual(df["a_width"].iloc[0], 2 * ps)
        self.assertEqual(df.index[0], 23)
        self.assertEqual(df["frame"].iloc[0], 0)
        self.assertEqual(df["time"].iloc[0], 0 * 15 / 60)
        np.testing.assert_almost_equal(df["position_x"].iloc[0], 2 / 2 * ps)
        np.testing.assert_almost_equal(df["position_y"].iloc[0], 3 / 2 * ps)
        self.assertEqual(df["gfp"].iloc[0], np.median(image[:3, :2]))
        self.assertEqual(df["gfp_mean"].iloc[0], np.mean(image[:3, :2]))
        self.assertEqual(df["perimeter"].iloc[0], 10 * ps)
        np.testing.assert_almost_equal(df["circularity"].iloc[0], 0.7539822368615503)

    def test_dynamic_time_extractor(self):
        # in x,y coordinates
        contours = [
            Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=0, id=23),
            Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=1, id=24),
            Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=2, id=25),
        ]
        overlay = Overlay(contours)

        timepoints = [1710326746.8015938, 1710326987.554663, 1710327228.3607492]
        rel_timepoints = np.array(timepoints) - timepoints[0]

        df = ExtractorExecutor().execute(
            overlay=overlay,
            images=THWCSequenceSource(np.zeros((3, 100, 100, 1), dtype=np.uint8)),
            extractors=[FrameEx(), DynamicTimeEx(timepoints, relative=True)],
        )

        self.assertEqual(df["time"].iloc[0], 0)
        self.assertEqual(df["time"].iloc[1], rel_timepoints[1] / 3600)
        self.assertEqual(df["time"].iloc[2], rel_timepoints[2] / 3600)

    def test_dynamic_time_extractor_failures(self):
        contours = [
            Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=0, id=23),
            Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=1, id=24),
            Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=2, id=25),
        ]
        overlay = Overlay(contours)

        with self.assertRaises(ValueError) as _:
            DynamicTimeEx([])

        with self.assertRaises(ValueError) as _:
            _ = ExtractorExecutor().execute(
                overlay=overlay,
                images=THWCSequenceSource(np.zeros((3, 100, 100, 1), dtype=np.uint8)),
                extractors=[
                    FrameEx(),
                    DynamicTimeEx(timepoints=[1, 2], relative=True),
                ],
            )

    def test_fluorescence_extractor_float(self):
        """Testing that the fluorescence exporter can work with float values"""

        contours = [Contour([[0, 0], [2, 0], [2, 3], [0, 3]], -1, frame=0, id=23)]
        overlay = Overlay(contours)

        image = np.zeros((200, 200), dtype=np.float32)
        image[0, 0] = 2.5
        image[0, 1] = 5.5
        image[1, 0] = 6
        image[1, 1] = 10.1
        image[2, 0:3] = 4
        image[0:2, 2] = -4.3
        image_source = LocalImageSource.from_array(image)

        # test basic extractors
        df = ExtractorExecutor().execute(
            overlay=overlay,
            images=image_source,
            extractors=[
                FluorescenceEx(channels=[0], channel_names=["gfp"], parallel=1),
                FluorescenceEx(
                    channels=[0],
                    channel_names=["gfp_mean"],
                    summarize_operator=np.mean,
                    parallel=1,
                ),
            ],
        )

        self.assertEqual(df["gfp"].iloc[0], np.median(image[:3, :2]))
        self.assertEqual(df["gfp_mean"].iloc[0], np.mean(image[:3, :2]))

    def test_parallel_fluorescence_extraction(self):
        squared_num = 30
        # ids must be unique across the whole overlay, not per frame
        contours = [
            Contour([[0, 0], [2, 0], [2, 2], [0, 2]], -1, frame=frame, id=id)
            for id, (_, frame) in enumerate(
                product(list(range(squared_num)), list(range(squared_num)))
            )
        ]
        overlay = Overlay(contours)

        image = np.zeros((200, 200))
        image[0, 0] = 2
        image[0, 1] = 5
        image[1, 0] = 6
        image[1, 1] = 10
        image_sources = InMemorySequenceSource(np.stack([image] * squared_num))

        self.assertTrue(image_sources is not None)

        # test basic extractors
        df = ExtractorExecutor().execute(
            overlay=overlay,
            images=image_sources,
            extractors=[
                FluorescenceEx(channels=[0], channel_names=["fl1"], parallel=3),
                FluorescenceEx(
                    channels=[0],
                    channel_names=["fl1_mean"],
                    summarize_operator=np.mean,
                    parallel=3,
                ),
            ],
        )

        np.testing.assert_array_equal(df["fl1"], [5.5] * len(df))
        np.testing.assert_array_equal(
            df["fl1_mean"], [np.mean([2, 5, 6, 10])] * len(df)
        )


if __name__ == "__main__":
    unittest.main()


def test_extractor_executor_empty_overlay_returns_empty_typed_frame():
    import numpy as np

    from acia import ureg
    from acia.analysis import AreaEx, ExtractorExecutor, PerimeterEx, PositionEx
    from acia.base import Overlay
    from acia.segm.local import THWCSequenceSource

    src = THWCSequenceSource(
        np.zeros((2, 10, 10, 1), dtype=np.uint8), pixel_size=0.1 * ureg.micrometer
    )
    df = ExtractorExecutor().execute(
        Overlay([], frames=[0, 1]), src, [AreaEx(), PerimeterEx(), PositionEx()]
    )
    assert len(df) == 0
    assert {"area", "perimeter", "position_x", "position_y"} <= set(df.columns)


def _square_overlay_source():
    """Overlay with 1/2/3 well-separated 10x10 cells and a matching 1 um/px source."""
    import numpy as np

    from acia.base import Contour, Overlay
    from acia.segm.local import THWCSequenceSource

    def square(x, y, frame, cont_id, size=10.0):
        coords = [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]
        return Contour(np.array(coords, dtype=float), 1.0, frame, cont_id)

    contours = [square(10, 10, 0, 0)]
    contours += [square(10, 10, 1, 1), square(200, 200, 1, 2)]
    contours += [square(10, 10, 2, 3), square(200, 200, 2, 4), square(400, 400, 2, 5)]

    source = THWCSequenceSource(
        np.zeros((3, 512, 512, 1), dtype=np.uint8),
        pixel_size="1 um",
        frame_interval="5 min",
    )
    return Overlay(contours), source


def test_extractor_executor_rejects_duplicate_ids():
    """Joining extractor results on a duplicated id would silently multiply rows"""
    from acia.analysis import AreaEx, ExtractorExecutor, FrameEx

    overlay, source = _square_overlay_source()
    for cont in overlay:
        cont.id = cont.frame  # what merge_cells_to_colonies used to do

    with pytest.raises(ValueError, match="duplicate contour id"):
        ExtractorExecutor().execute(overlay, source, [AreaEx(), FrameEx()])


def test_colony_areas_are_not_inflated_by_multiple_blobs_per_frame():
    """Regression: total colony area per timepoint must be the plain sum of its blobs"""
    from acia.analysis import AreaEx, ExtractorExecutor, FrameEx, TimeEx
    from acia.segm.utils import merge_cells_to_colonies

    overlay, source = _square_overlay_source()
    colonies = merge_cells_to_colonies(overlay, expand=2)

    df = ExtractorExecutor().execute(colonies, source, [AreaEx(), FrameEx(), TimeEx()])

    # one row per colony blob -- no cartesian blow-up from the id join
    assert len(df) == len(colonies)

    per_frame = df.groupby("frame")["area"].sum()
    # 1, 2 and 3 separated cells of 10x10 um each
    assert per_frame.loc[0.0] == pytest.approx(100, rel=1e-3)
    assert per_frame.loc[1.0] == pytest.approx(200, rel=1e-3)
    assert per_frame.loc[2.0] == pytest.approx(300, rel=1e-3)
