"""Segmenter using Cellpose (below v4)"""

from cellpose import core, models

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks


class CellposeSegmenter:
    """Cellpose segmenter"""

    def __init__(self, model_type: str):
        self.use_GPU = core.use_gpu()
        self.model = models.CellposeModel(gpu=self.use_GPU, model_type=model_type)

    @staticmethod
    def __predict(images, model, cellpose_params=None):
        chans = [0, 0]  # this means segment based on first channel, no second channel

        if cellpose_params is None:
            cellpose_params = {}

        masks, _, _ = model.eval(images, channels=chans, **cellpose_params)

        return masks

    def __call__(self, images: ImageSequenceSource, cellpose_params=None) -> Overlay:
        imgs = []
        for image in images:
            raw_image = image.raw

            imgs.append(raw_image)

        masks = self.__predict(imgs, self.model, cellpose_params=cellpose_params)

        ov = overlay_from_masks(masks)

        attribute_segmentation(ov, self)

        return ov
