"""Unit test cases for MicroSAMSegmenter."""

import sys
import types
import unittest

import numpy as np
import pytest

from acia.base import Overlay
from acia.segm.local import THWCSequenceSource


class _FakePredictor:
    pass


class _FakeSegmenter:
    pass


def _fake_get_predictor_and_segmenter(calls):
    def _impl(model_type, device=None):
        calls.append((model_type, device))
        return (_FakePredictor(), _FakeSegmenter())

    return _impl


def _fake_automatic_instance_segmentation(calls):
    def _impl(predictor, segmenter, input_path, ndim):
        calls.append(input_path.shape)
        # a single labeled instance covering the top-left quadrant
        labels = np.zeros(input_path.shape[:2], dtype=np.int32)
        labels[: input_path.shape[0] // 2, : input_path.shape[1] // 2] = 1
        return labels

    return _impl


def _install_fake_micro_sam(load_calls, segment_calls):
    micro_sam_pkg = types.ModuleType("micro_sam")
    automatic_segmentation = types.ModuleType("micro_sam.automatic_segmentation")
    automatic_segmentation.get_predictor_and_segmenter = (
        _fake_get_predictor_and_segmenter(load_calls)
    )
    automatic_segmentation.automatic_instance_segmentation = (
        _fake_automatic_instance_segmentation(segment_calls)
    )
    micro_sam_pkg.automatic_segmentation = automatic_segmentation
    sys.modules["micro_sam"] = micro_sam_pkg
    sys.modules["micro_sam.automatic_segmentation"] = automatic_segmentation


class TestMicroSAMSegmenter(unittest.TestCase):
    """Exercise MicroSAMSegmenter against a fake micro_sam module (no real dep)."""

    def setUp(self):
        self._saved = {
            name: sys.modules.get(name)
            for name in ("micro_sam", "micro_sam.automatic_segmentation")
        }
        self.load_calls: list = []
        self.segment_calls: list = []
        _install_fake_micro_sam(self.load_calls, self.segment_calls)

        from acia.segm.processor.microsam import MicroSAMSegmenter

        self.MicroSAMSegmenter = MicroSAMSegmenter

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("acia.segm.processor.microsam", None)

    def test_constructs_predictor_with_model_type(self):
        processor = self.MicroSAMSegmenter(model_type="vit_b_lm")
        image = np.zeros((2, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertEqual(self.load_calls, [("vit_b_lm", None)])

    def test_returns_overlay(self):
        processor = self.MicroSAMSegmenter()
        image = np.zeros((3, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 3)
        self.assertGreater(len(overlay), 0)

    def test_squeezes_single_channel_input(self):
        processor = self.MicroSAMSegmenter()
        image = np.zeros((1, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertEqual(self.segment_calls, [(20, 20)])

    def test_model_loaded_once_per_call_with_autorelease(self):
        processor = self.MicroSAMSegmenter()
        image = np.zeros((1, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)
        processor(source)

        self.assertEqual(len(self.load_calls), 2)  # rebuilt each call (autorelease)


@pytest.mark.integration
class TestMicroSAMSegmenterIntegration(unittest.TestCase):
    """Real micro_sam import; skipped unless the (optional) dependency is installed."""

    def setUp(self):
        pytest.importorskip("micro_sam")

    def test_real_segmentation(self):
        from acia.segm.processor.microsam import MicroSAMSegmenter

        processor = MicroSAMSegmenter(device="cpu")
        image = (np.random.rand(2, 64, 64, 1) * 255).astype(np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 2)


if __name__ == "__main__":
    unittest.main()
