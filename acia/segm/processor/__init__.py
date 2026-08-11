"""Segmentation Processors"""

import contextlib
from collections.abc import Iterator
from typing import Any

from acia.base import ImageSequenceSource, Overlay


class SegmentationProcessor:
    """Base class for segmentation processors.

    Provides lazy model construction and GPU-memory lifecycle management. The
    model is built on first access via the ``model`` property (backed by the
    :meth:`_load_model` hook and cached in ``self._model``). When
    ``autorelease`` is enabled (the default), :meth:`__call__` releases the
    model after each segmentation, freeing GPU memory for other processes.

    Subclasses implement :meth:`_load_model` (build and return the backend
    model) and :meth:`_segment` (run segmentation using ``self.model``), rather
    than overriding :meth:`__call__` directly.
    """

    def __init__(self, autorelease: bool = True) -> None:
        """Initialize the segmentation processor.

        Args:
            autorelease: When ``True`` (default), the model is released after
                each :meth:`__call__`, freeing GPU memory. Set to ``False`` to
                keep the model resident across calls (expert batch case).
        """
        self.autorelease = autorelease
        self._load_depth = 0

    def _load_model(self) -> Any:
        """Build and return the backend model.

        Overridden by subclasses. The returned model is cached by the
        :attr:`model` property in ``self._model``.

        Returns:
            The constructed backend model.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("Please implement this base function")

    @property
    def model(self) -> Any:
        """Lazily build and return the backend model.

        The model is constructed on first access via :meth:`_load_model` and
        cached in ``self._model`` for subsequent accesses.

        Returns:
            The (cached) backend model.
        """
        if getattr(self, "_model", None) is None:
            self._model = self._load_model()
        return self._model

    def _segment(
        self, images: ImageSequenceSource, *args: Any, **kwargs: Any
    ) -> Overlay:
        """Run segmentation on the given images using ``self.model``.

        Overridden by subclasses. This is the hook invoked by the
        :meth:`__call__` template; backends may accept extra per-call
        parameters (e.g. ``cellpose_params``/``omnipose_parameters``), which
        :meth:`__call__` forwards.

        Args:
            images: The image sequence to segment.
            *args: Extra positional per-call parameters forwarded from ``__call__``.
            **kwargs: Extra keyword per-call parameters forwarded from ``__call__``.

        Returns:
            The resulting overlay.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("Please implement this base function")

    def __call__(
        self, images: ImageSequenceSource, *args: Any, **kwargs: Any
    ) -> Overlay:
        """Segment the images and (optionally) release the model afterward.

        Template method: runs :meth:`_segment` (forwarding any extra per-call
        parameters) and, when ``autorelease`` is on and not inside a
        :meth:`load` block, calls :meth:`release` in a ``finally`` so the GPU is
        freed even if segmentation raises.

        Args:
            images: The image sequence to segment.
            *args: Extra positional parameters forwarded to :meth:`_segment`.
            **kwargs: Extra keyword parameters forwarded to :meth:`_segment`
                (e.g. ``omnipose_parameters=...``).

        Returns:
            The resulting overlay.
        """
        try:
            return self._segment(images, *args, **kwargs)
        finally:
            if getattr(self, "autorelease", True) and not getattr(
                self, "_load_depth", 0
            ):
                self.release()

    def _release_model(self) -> None:
        """Drop the cached model reference.

        Default implementation sets ``self._model`` to ``None``. Subclasses may
        override to release additional resources.
        """
        self._model = None

    def release(self) -> None:
        """Release the model and attempt to free GPU memory.

        Drops the model via :meth:`_release_model`, runs a garbage collection,
        and, if torch is importable and a CUDA device is available, calls
        ``torch.cuda.empty_cache()``. Idempotent and never raises.
        """
        self._release_model()

        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @contextlib.contextmanager
    def load(self) -> Iterator["SegmentationProcessor"]:
        """Keep the model resident for the duration of a ``with`` block.

        Builds the model up front, suppresses per-call autorelease while the
        block is active, and releases the model on exit (even if the body
        raises). Re-entrant: nested ``load()`` blocks on the same instance only
        release once the outermost block exits, via a depth counter.

        Yields:
            This processor instance, with its model loaded.
        """
        self._load_depth = getattr(self, "_load_depth", 0) + 1
        try:
            _ = self.model  # build now (inside try so a build failure resets depth)
            yield self
        finally:
            self._load_depth -= 1
            if self._load_depth == 0:
                self.release()

    def __del__(self) -> None:
        """Release resources on garbage collection. Never raises."""
        with contextlib.suppress(Exception):
            self.release()


# Import subclasses after base class is defined to avoid circular imports
from acia.segm.processor.canny import (  # noqa: E402
    CannySegmentationProcessor as CannySegmentationProcessor,
)
