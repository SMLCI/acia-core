"""Unit test cases for FlowposeRTSegmenter."""

import sys
import types
import unittest

import numpy as np
import pytest

from acia.base import Overlay
from acia.segm.local import THWCSequenceSource
from acia.segm.processor.flowpose_rt import FlowposeRTSegmenter


class _FakeSegmenter:
    """Fake flowpose_rt.Segmenter: records the stack it was called with."""

    def __init__(self, model, device, precision, compile):
        self.model = model
        self.device = device
        self.precision = precision
        self.compile = compile
        self.last_stack = None
        self.call_shapes = []

    def segment(self, stack):
        self.last_stack = stack
        self.call_shapes.append(stack.shape)
        # a single labeled instance covering the top-left quadrant of every frame
        masks = np.zeros(stack.shape, dtype=np.int32)
        masks[:, : stack.shape[1] // 2, : stack.shape[2] // 2] = 1
        return masks


class _FakeFlowposeRTModule(types.ModuleType):
    def __init__(self):
        super().__init__("flowpose_rt")
        self.built_with = []

    def Segmenter(self, model, device, precision, compile):  # noqa: A002
        self.built_with.append((model, device, precision, compile))
        return _FakeSegmenter(model, device, precision, compile)


class TestFlowposeRTSegmenter(unittest.TestCase):
    """Exercise FlowposeRTSegmenter against a fake flowpose_rt module (no real dep)."""

    def setUp(self):
        self._saved_module = sys.modules.get("flowpose_rt")
        self._fake_module = _FakeFlowposeRTModule()
        sys.modules["flowpose_rt"] = self._fake_module

    def tearDown(self):
        if self._saved_module is None:
            sys.modules.pop("flowpose_rt", None)
        else:
            sys.modules["flowpose_rt"] = self._saved_module

    def test_constructs_segmenter_with_config(self):
        processor = FlowposeRTSegmenter(
            model="bact_phase_omni", device="cpu", precision="fp32", compile=False
        )
        image = np.zeros((2, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertEqual(
            self._fake_module.built_with,
            [("bact_phase_omni", "cpu", "fp32", False)],
        )

    def test_single_channel_stack_returns_overlay(self):
        processor = FlowposeRTSegmenter(device="cpu")
        image = np.zeros((3, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 3)
        self.assertGreater(len(overlay), 0)

    def test_vectorized_single_forward_over_stack(self):
        """The whole (T, H, W) stack is segmented in one call, not per-frame."""
        processor = FlowposeRTSegmenter(device="cpu", autorelease=False)
        image = np.zeros((4, 16, 16, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertEqual(processor._model.last_stack.shape, (4, 16, 16))

    def test_batch_size_splits_stack_across_calls(self):
        """batch_size < num_frames -> multiple smaller model.segment() calls."""
        processor = FlowposeRTSegmenter(device="cpu", autorelease=False, batch_size=2)
        image = np.zeros((5, 16, 16, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertEqual(
            processor._model.call_shapes,
            [(2, 16, 16), (2, 16, 16), (1, 16, 16)],
        )
        self.assertEqual(overlay.numFrames(), 5)

    def test_multi_channel_input_raises(self):
        processor = FlowposeRTSegmenter(device="cpu")
        image = np.zeros((1, 20, 20, 3), dtype=np.uint8)
        source = THWCSequenceSource(image)

        with self.assertRaises(ValueError):
            processor(source)


class _FakeCompiler:
    def __init__(self, calls):
        self._calls = calls

    def reset(self):
        self._calls.append("compiler.reset")


class _FakeCuda:
    def __init__(self, available, calls):
        self._available = available
        self._calls = calls

    def is_available(self):
        return self._available

    def empty_cache(self):
        self._calls.append("empty_cache")


def _install_fake_torch(available):
    """Install a fake ``torch`` module (cuda + compiler.reset) in sys.modules."""
    calls: list[str] = []
    fake = types.ModuleType("torch")
    fake.cuda = _FakeCuda(available, calls)  # type: ignore[attr-defined]
    fake.compiler = _FakeCompiler(calls)  # type: ignore[attr-defined]
    sys.modules["torch"] = fake
    return calls


class TestFlowposeRTReleaseModel(unittest.TestCase):
    """FlowposeRTSegmenter._release_model additionally clears the compile cache."""

    def setUp(self):
        self._saved_module = sys.modules.get("flowpose_rt")
        self._fake_module = _FakeFlowposeRTModule()
        sys.modules["flowpose_rt"] = self._fake_module

        self._saved_torch = sys.modules.get("torch")

    def tearDown(self):
        if self._saved_module is None:
            sys.modules.pop("flowpose_rt", None)
        else:
            sys.modules["flowpose_rt"] = self._saved_module

        if self._saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved_torch

    def test_compiler_reset_called_when_cuda_available(self):
        calls = _install_fake_torch(available=True)
        processor = FlowposeRTSegmenter(device="cpu")
        image = np.zeros((1, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)  # autorelease=True -> triggers _release_model

        self.assertIn("compiler.reset", calls)

    def test_compiler_reset_not_called_when_cuda_unavailable(self):
        calls = _install_fake_torch(available=False)
        processor = FlowposeRTSegmenter(device="cpu")
        image = np.zeros((1, 20, 20, 1), dtype=np.uint8)
        source = THWCSequenceSource(image)

        processor(source)

        self.assertNotIn("compiler.reset", calls)


@pytest.mark.integration
class TestFlowposeRTSegmenterIntegration(unittest.TestCase):
    """Real flowpose_rt import; skipped unless the (optional) dependency is installed."""

    def setUp(self):
        pytest.importorskip("flowpose_rt")

    def test_real_segmentation(self):
        processor = FlowposeRTSegmenter(model="bact_phase_omni", device="cpu")
        image = (np.random.rand(2, 64, 64, 1) * 255).astype(np.uint8)
        source = THWCSequenceSource(image)

        overlay = processor(source)

        self.assertIsInstance(overlay, Overlay)
        self.assertEqual(overlay.numFrames(), 2)


if __name__ == "__main__":
    unittest.main()
