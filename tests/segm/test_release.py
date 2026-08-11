"""Unit tests for the SegmentationProcessor lazy-model / autorelease machinery.

These tests use a lightweight stub subclass and a fake ``torch`` module; they
never import the real (torch/cellpose) backends, so they run without GPU or
heavy deps.
"""

import sys
import types
import unittest

from acia.base import Overlay
from acia.segm.processor import SegmentationProcessor


class StubSeg(SegmentationProcessor):
    """Stub segmenter: counts model builds, returns an empty Overlay."""

    def __init__(self, autorelease=True):
        super().__init__(autorelease=autorelease)
        self.load_count = 0

    def _load_model(self):
        self.load_count += 1
        return object()  # fake model

    def _segment(self, images):
        assert self.model is not None  # uses the lazily-built model
        return Overlay([])


class RaisingSeg(StubSeg):
    """Stub whose ``_segment`` raises after the model is built."""

    def _segment(self, images):
        assert self.model is not None
        raise RuntimeError("boom")


class _FakeCuda:
    def __init__(self, available, calls):
        self._available = available
        self._calls = calls

    def is_available(self):
        return self._available

    def empty_cache(self):
        self._calls.append("empty_cache")


def _install_fake_torch(available):
    """Install a fake ``torch`` module in ``sys.modules`` and return its calls list."""
    calls: list[str] = []
    fake = types.ModuleType("torch")
    fake.cuda = _FakeCuda(available, calls)  # type: ignore[attr-defined]
    sys.modules["torch"] = fake
    return calls


class TestAutorelease(unittest.TestCase):
    """Default autorelease behavior (one-shot)."""

    def test_default_autorelease_releases_and_rebuilds(self):
        s = StubSeg()
        ov = s("imgs")

        self.assertIsInstance(ov, Overlay)
        self.assertIsNone(s._model)  # released
        self.assertEqual(s.load_count, 1)

        # second call rebuilds the model
        s("imgs")
        self.assertEqual(s.load_count, 2)
        self.assertIsNone(s._model)

    def test_autorelease_false_keeps_model_resident(self):
        s = StubSeg(autorelease=False)
        s("a")
        s("b")

        self.assertEqual(s.load_count, 1)
        self.assertIsNotNone(s._model)


class TestLoadContextManager(unittest.TestCase):
    """The load() context manager suppresses per-call autorelease."""

    def test_load_suppresses_autorelease(self):
        seg = StubSeg()
        with seg.load() as s:
            s("a")
            s("b")
            self.assertEqual(s.load_count, 1)
            self.assertIsNotNone(s._model)

        # released on block exit
        self.assertIsNone(seg._model)

    def test_load_releases_even_if_body_raises(self):
        seg = StubSeg()
        with self.assertRaises(RuntimeError), seg.load() as s:
            s("a")
            raise RuntimeError("body boom")

        self.assertIsNone(seg._model)


class TestSegmentRaises(unittest.TestCase):
    """A raising _segment still releases when autorelease is on."""

    def test_segment_raise_still_releases(self):
        s = RaisingSeg(autorelease=True)
        with self.assertRaises(RuntimeError):
            s("imgs")

        self.assertIsNone(s._model)


class TestEmptyCache(unittest.TestCase):
    """empty_cache is only called when a CUDA device is available."""

    def setUp(self):
        self._saved_torch = sys.modules.get("torch")

    def tearDown(self):
        if self._saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved_torch

    def test_empty_cache_called_when_available(self):
        calls = _install_fake_torch(available=True)
        s = StubSeg()
        s.release()
        self.assertIn("empty_cache", calls)

    def test_empty_cache_not_called_when_unavailable(self):
        calls = _install_fake_torch(available=False)
        s = StubSeg()
        s.release()
        self.assertNotIn("empty_cache", calls)

    def test_release_works_when_torch_import_fails(self):
        # Ensure torch cannot be imported.
        sys.modules.pop("torch", None)
        sys.modules["torch"] = None  # type: ignore[assignment]
        s = StubSeg()
        # Should not raise even though importing torch fails.
        s.release()
        self.assertIsNone(s._model)


class TestIdempotencyAndDel(unittest.TestCase):
    """release() is idempotent and __del__ never raises."""

    def test_release_idempotent(self):
        s = StubSeg()
        s.release()
        s.release()  # no error
        self.assertIsNone(s._model)

    def test_del_does_not_raise(self):
        s = StubSeg()
        s.__del__()  # no raise


class _KwargSeg(SegmentationProcessor):
    """Stub whose ``_segment`` records the per-call params it receives."""

    def __init__(self, autorelease=True):
        super().__init__(autorelease=autorelease)
        self.received = None

    def _load_model(self):
        return object()

    def _segment(self, images, **kwargs):
        self.received = kwargs
        return Overlay([])


class _FailLoadSeg(SegmentationProcessor):
    """Stub whose model build always fails."""

    def _load_model(self):
        raise RuntimeError("build failed")

    def _segment(self, images):  # pragma: no cover - never reached
        return Overlay([])


class TestReviewRegressions(unittest.TestCase):
    def test_call_forwards_per_call_kwargs(self):
        # BC-1: per-call params (e.g. omnipose_parameters) must reach _segment
        s = _KwargSeg()
        s("imgs", omnipose_parameters={"batch_size": 30})
        self.assertEqual(s.received, {"omnipose_parameters": {"batch_size": 30}})

    def test_failed_load_resets_depth_no_autorelease_leak(self):
        # if the model build raises inside load(), _load_depth must return to 0
        s = _FailLoadSeg()
        with self.assertRaises(RuntimeError), s.load():
            pass  # pragma: no cover - build raises before body
        self.assertEqual(s._load_depth, 0)

    def test_nested_load_releases_only_at_outermost_exit(self):
        seg = StubSeg()
        with seg.load():
            with seg.load():
                seg("a")
                self.assertIsNotNone(seg._model)  # inner block keeps it
            # inner exit must NOT free the model the outer block still uses
            self.assertIsNotNone(seg._model)
            self.assertEqual(seg.load_count, 1)  # built once across both blocks
        self.assertIsNone(seg._model)  # freed at outermost exit


if __name__ == "__main__":
    unittest.main()
