"""Segmentation Processors"""

from acia.base import ImageSequenceSource, Overlay


class SegmentationProcessor:
    """Base class for segmentation processors"""

    def __call__(self, images: ImageSequenceSource) -> Overlay:
        raise NotImplementedError("Please implement this base function")
