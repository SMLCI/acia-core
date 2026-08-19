"""Unit test cases for StarDistSegmenter."""

import sys
import types
import unittest

import numpy as np
import pytest

from acia.base import Overlay
from acia.segm.local import THWCSequenceSource


class _FakeStarDistModel:
    """Fake StarDist2D model: records predict_instances calls."""

    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model
        self.calls = []

    def predict_instances(self, img):
        self.calls.append(img)
        # a single labeled instance covering the top-left quadrant
        labels = np.zeros(img.shape[:2], dtype=np.int32)
        labels[: img.shape[0] // 2, : img.shape[1] // 2] = 1
        return labels, {"prob": [0.9]}


class _FakeStarDist2D:
    """Fake stardist.models.StarDist2D: records from_pretrained calls."""

    built_with: list = []
    last_model: "_FakeStarDistModel | None" = None

    @classmethod
    def from_pretrained(cls, name):
        cls.built_with.append(name)
        cls.last_model = _FakeStarDistModel(name)
        return cls.last_model


def _install_fake_stardist():
    stardist_pkg = types.ModuleType("stardist")
    stardist_models = types.ModuleType("stardist.models")
    _FakeStarDist2D.built_with = []
    _FakeStarDist2D.last_model = None
    stardist_models.StarDist2D = _FakeStarDist2D
    stardist_pkg.models = stardist_models
    sys.modules["stardist"] = stardist_pkg
    sys.modules["stardist.models"] = stardist_models

    csbdeep_pkg = types.ModuleType("csbdeep")
    csbdeep_utils = types.ModuleType("csbdeep.utils")
    csbdeep_utils.normalize = lambda img: img
    csbdeep_pkg.utils = csbdeep_utils
    sys.modules["csbdeep"] = csbdeep_pkg
    sys.modules["csbdeep.utils"] = csbdeep_utils


class TestStarDistSegmenter(unittest.TestCase):
    """Exercise StarDistSegmenter against fake stardist/csbdeep modules (no real deps)."""

    def setUp(self):
        self._saved = {
            name: sys.modules.get(name)
            for name in ("stardist", "stardist.models", "csbdeep", "csbdeep.utils")
        }
        _install_fake_stardist()

        # import after faking sys.modules so the module-level `from ... import`
        # statements resolve against the fakes
        from acia.segm.processor.stardist import StarDistSegmenter

        self.StarDistSegmenter = StarDistSegmenter

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("acia.segm.processor.stardist", None)

    def test_constructs_model_with_pretrained_name(self):
        processor = self.StarDistSegmenter(pretrained_model="2D_versatile_fluo")
        image = np.zeros((2, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertEqual(_FakeStarDist2D.built_with, ["2D_versatile_fluo"])

    def test_returns_overlay(self):
        processor = self.StarDistSegmenter()
        image = np.zeros((3, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 3)
        self.assertGreater(len(overlay), 0)

    def test_squeezes_single_channel_input(self):
        processor = self.StarDistSegmenter(autorelease=False)
        image = np.zeros((1, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        called_shape = processor._model.calls[0].shape
        self.assertEqual(called_shape, (20, 20))


@pytest.mark.integration
class TestStarDistSegmenterIntegration(unittest.TestCase):
    """Real stardist import; skipped unless the (optional) dependency is installed."""

    def setUp(self):
        pytest.importorskip("stardist")

    def test_real_segmentation(self):
        from acia.segm.processor.stardist import StarDistSegmenter

        processor = StarDistSegmenter()
        image = (np.random.rand(2, 64, 64, 1) * 255).astype(np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 2)


if __name__ == "__main__":
    unittest.main()
