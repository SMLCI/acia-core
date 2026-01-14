"""Segmentation Processors"""

from acia.base import ImageSequenceSource, Overlay


class SegmentationProcessor:
    """Base class for segmentation processors"""

    def __call__(self, images: ImageSequenceSource) -> Overlay:
        raise NotImplementedError("Please implement this base function")


# Import subclasses after base class is defined to avoid circular imports
from acia.segm.processor.canny import (  # noqa: E402
    CannySegmentationProcessor as CannySegmentationProcessor,
)
