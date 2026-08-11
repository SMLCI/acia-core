"""Segmenter using CellposeSAM: https://doi.org/10.1101/2025.04.28.651001"""

import os

from cellpose import core, models
from tqdm.auto import tqdm

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import SegmentationProcessor


class CellposeSAMSegmenter(SegmentationProcessor):
    """CellposeSAMSegmenter using Cellpose SAM: https://doi.org/10.1101/2025.04.28.651001"""

    def __init__(self, use_GPU=None, pretrained_model=None, autorelease: bool = True):
        """Initialize the Cellpose-SAM segmenter.

        Args:
            use_GPU: Force GPU on/off. ``None`` (default) auto-detects via
                :func:`cellpose.core.use_gpu`.
            pretrained_model: Which Cellpose model to load. ``None`` (default)
                loads Cellpose's built-in default model. Pass a model name or a
                path to weights to override it (forwarded to
                :class:`cellpose.models.CellposeModel`). The actually loaded
                model is printed on first use.
            autorelease: Release the model after each call to free GPU memory.
        """
        super().__init__(autorelease=autorelease)
        if use_GPU is None:
            use_GPU = core.use_gpu()
        self.use_GPU = use_GPU
        self.pretrained_model = pretrained_model
        print(f"Use GPU? {self.use_GPU}")

    def _load_model(self):
        # create CellPose model
        kwargs = {"gpu": self.use_GPU}
        if self.pretrained_model is not None:
            kwargs["pretrained_model"] = self.pretrained_model
        model = models.CellposeModel(**kwargs)
        self._describe_model(model)
        return model

    @staticmethod
    def _describe_model(model) -> None:
        """Print which underlying Cellpose model / device is in use.

        Cellpose ships several different pretrained models; this makes it
        explicit which one was loaded (rather than silently using a default).
        """
        pretrained = getattr(model, "pretrained_model", None)
        if isinstance(pretrained, (list, tuple)):
            pretrained = pretrained[0] if pretrained else None

        parts = []
        if pretrained:
            parts.append(f"weights={os.path.basename(str(pretrained))}")
        device = getattr(model, "device", None)
        if device is not None:
            parts.append(f"device={device}")
        diam_mean = getattr(model, "diam_mean", None)
        if diam_mean is not None:
            parts.append(f"diam_mean={diam_mean}")

        if parts:
            print("Cellpose model loaded: " + ", ".join(str(p) for p in parts))
        else:
            print("Cellpose model loaded")

    @staticmethod
    def __predict(images, model, cellpose_params=None):
        if cellpose_params is None:
            cellpose_params = {}

        # Segment image-by-image so we can render a real progress bar.
        # Cellpose-SAM's eval() already loops per image internally when handed a
        # list (models.py: `for i in trange(nimg)`), so this does NOT cost any
        # cross-image GPU batching -- it is the same work, just with visible
        # progress. GPU batching happens at the tile level via the `batch_size`
        # cellpose param (number of 256x256 patches run at once), which still
        # applies within each per-image call. Cellpose's own progress goes to a
        # logger at 30s intervals, which is why it looked frozen in notebooks.
        # (channels= is deprecated/ignored in Cellpose v4, so it is not passed.)
        masks = []
        for image in tqdm(images, desc="CellposeSAM segmenting"):
            mask, _, _ = model.eval(image, **cellpose_params)
            masks.append(mask)

        return masks

    def _segment(self, images: ImageSequenceSource, cellpose_params=None) -> Overlay:
        # list of images
        imgs = [im.raw for im in images]

        # perform the prediction
        masks = self.__predict(imgs, self.model, cellpose_params=cellpose_params)

        # parse the overlay
        ov = overlay_from_masks(masks)

        attribute_segmentation(ov, self)

        return ov
