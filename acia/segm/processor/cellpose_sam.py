"""Segmenter using CellposeSAM: https://doi.org/10.1101/2025.04.28.651001"""

from cellpose import core, models

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks


class CellposeSAMSegmenter:
    """CellposeSAMSegmenter using Cellpose SAM: https://doi.org/10.1101/2025.04.28.651001"""

    def __init__(self, use_GPU=None):
        if use_GPU is None:
            self.use_GPU = core.use_gpu()
        print(f"Use GPU? {self.use_GPU}")

        # create CellPose model
        self.model = models.CellposeModel(gpu=self.use_GPU)

    @staticmethod
    def __predict(images, model, cellpose_params=None):
        chans = [0, 0]  # this means segment based on first channel, no second channel

        if cellpose_params is None:
            cellpose_params = {}

        # perform inference
        masks, _, _ = model.eval(images, channels=chans, **cellpose_params)

        return masks

    def __call__(self, images: ImageSequenceSource, cellpose_params=None) -> Overlay:
        # list of images
        imgs = [im.raw for im in images]

        # perform the prediction
        masks = self.__predict(imgs, self.model, cellpose_params=cellpose_params)

        # parse the overlay
        ov = overlay_from_masks(masks)

        attribute_segmentation(ov, self)

        return ov
