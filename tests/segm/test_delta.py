"""Unit test cases for DeltaSegmenter."""

import sys
import types
import unittest

import numpy as np
import pytest

from acia.base import Overlay
from acia.segm.local import THWCSequenceSource


class _FakeDataArray:
    """Fake xarray.DataArray: just remembers the array and its dims."""

    def __init__(self, data, dims):
        self.data = data
        self.dims = dims
        self.shape = data.shape


class _FakeROI:
    def __init__(self, seg_stack):
        self.seg_stack = seg_stack


class _FakePosition:
    """Fake delta.pipeline.Position: fabricates a seg_stack sized to the input."""

    built_with: list = []
    preprocessed_with: list = []

    def __init__(self, position_nb, config):
        self.position_nb = position_nb
        self.config = config
        self.rois = []
        type(self).built_with.append((position_nb, config))

    def preprocess(self, all_frames):
        type(self).preprocessed_with.append(all_frames)
        self._all_frames = all_frames

    def segment(self):
        n_frames, _n_channels, height, width = self._all_frames.shape
        seg_stack = []
        for _ in range(n_frames):
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[: height // 2, : width // 2] = 1
            seg_stack.append(mask)
        self.rois = [_FakeROI(seg_stack=seg_stack)]


class _FakeConfig:
    default_calls: list = []

    @classmethod
    def default(cls, regime):
        cls.default_calls.append(regime)
        return f"config[{regime}]"


def _install_fake_delta_and_xarray():
    _FakePosition.built_with = []
    _FakePosition.preprocessed_with = []
    _FakeConfig.default_calls = []

    delta_pkg = types.ModuleType("delta")
    delta_config = types.ModuleType("delta.config")
    delta_config.Config = _FakeConfig
    delta_pipeline = types.ModuleType("delta.pipeline")
    delta_pipeline.Position = _FakePosition
    delta_imgops = types.ModuleType("delta.imgops")
    delta_imgops.label_seg = lambda binary_mask: binary_mask.astype(np.uint16)
    delta_pkg.config = delta_config
    delta_pkg.pipeline = delta_pipeline
    delta_pkg.imgops = delta_imgops
    sys.modules["delta"] = delta_pkg
    sys.modules["delta.config"] = delta_config
    sys.modules["delta.pipeline"] = delta_pipeline
    sys.modules["delta.imgops"] = delta_imgops

    xarray_pkg = types.ModuleType("xarray")
    xarray_pkg.DataArray = _FakeDataArray
    sys.modules["xarray"] = xarray_pkg


class TestDeltaSegmenter(unittest.TestCase):
    """Exercise DeltaSegmenter against fake delta/xarray modules (no real deps)."""

    def setUp(self):
        self._saved = {
            name: sys.modules.get(name)
            for name in (
                "delta",
                "delta.config",
                "delta.pipeline",
                "delta.imgops",
                "xarray",
            )
        }
        _install_fake_delta_and_xarray()

        from acia.segm.processor.delta import DeltaSegmenter

        self.DeltaSegmenter = DeltaSegmenter

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("acia.segm.processor.delta", None)

    def test_loads_config_for_regime(self):
        processor = self.DeltaSegmenter(regime="mothermachine")
        image = np.zeros((2, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertEqual(_FakeConfig.default_calls, ["mothermachine"])

    def test_default_regime_is_2d(self):
        processor = self.DeltaSegmenter()
        image = np.zeros((1, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertEqual(_FakeConfig.default_calls, ["2D"])

    def test_returns_overlay_for_all_frames(self):
        processor = self.DeltaSegmenter()
        image = np.zeros((3, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 3)
        self.assertGreater(len(overlay), 0)

    def test_builds_frame_channel_y_x_array(self):
        processor = self.DeltaSegmenter(autorelease=False)
        image = np.zeros((4, 16, 16, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        (all_frames,) = _FakePosition.preprocessed_with
        self.assertEqual(all_frames.dims, ("frame", "channel", "y", "x"))
        self.assertEqual(all_frames.shape, (4, 1, 16, 16))


@pytest.mark.integration
class TestDeltaSegmenterIntegration(unittest.TestCase):
    """Real delta import; skipped unless the (optional) dependency is installed."""

    def setUp(self):
        pytest.importorskip("delta")

    def test_real_segmentation(self):
        from acia.segm.processor.delta import DeltaSegmenter

        processor = DeltaSegmenter()
        image = (np.random.rand(2, 64, 64, 1) * 255).astype(np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 2)


if __name__ == "__main__":
    unittest.main()
