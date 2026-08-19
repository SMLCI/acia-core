"""Segmenter using DeLTA: https://gitlab.com/delta-microscopy/delta

Targets DeLTA 3.x (package ``delta-microscopy``, import name ``delta``), which
rewrote DeLTA on Keras 3 as a multi-backend model (TensorFlow/torch/JAX are now
interchangeable extras upstream). Installing the ``torch-cpu``/``torch-gpu``
extra keeps this backend on the same torch stack as Cellpose/Omnipose instead
of pulling in TensorFlow. Note: v3.0.0 is a very recent release relative to the
DeLTA 2.0 paper -- its ``Config``/``Position``/``ROI`` object graph is
reverse-engineered here from the published library docs and source, not a
one-line ``delta.segment(image)`` call (no such shortcut exists upstream).
"""

import numpy as np
from tqdm.auto import tqdm

from acia.attribute import attribute_segmentation
from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import SegmentationProcessor


class DeltaSegmenter(SegmentationProcessor):
    """DeltaSegmenter using DeLTA (bacteria-specific U-Net segmentation):
    https://gitlab.com/delta-microscopy/delta

    Runs only DeLTA's segmentation model -- the tracking model is never
    loaded/invoked. Per-instance labels come from DeLTA's own
    ``delta.imgops.label_seg`` connected-components helper applied to the
    segmentation U-Net's binary output, not from a second model.
    """

    def __init__(self, regime: str = "2D", autorelease: bool = True):
        """Initialize the DeLTA segmenter.

        Args:
            regime: Which DeLTA preset to load -- ``"2D"`` (agar-pad/open-growth
                imaging, default) or ``"mothermachine"`` (microfluidic trap
                imaging). Selects both the config and the pretrained
                segmentation weights DeLTA downloads on first use.
            autorelease: Release the model after each call to free GPU memory.
        """
        super().__init__(autorelease=autorelease)
        self.regime = regime

    def _load_model(self):
        from delta.config import Config

        config = Config.default(self.regime)
        print(f"DeLTA config loaded: regime={self.regime}")
        return config

    def _segment(self, images: ImageSequenceSource) -> Overlay:
        import xarray as xr
        from delta.imgops import label_seg
        from delta.pipeline import Position

        frames = []
        for image in images:
            raw_image = image.raw
            if len(raw_image.shape) == 3 and raw_image.shape[2] == 1:
                raw_image = raw_image[..., 0]
            frames.append(raw_image.astype(np.float32))

        # (frame, channel, y, x); channel 0 = phase-contrast/brightfield
        stack = np.stack(frames)[:, np.newaxis, :, :]
        all_frames = xr.DataArray(stack, dims=("frame", "channel", "y", "x"))

        position = Position(position_nb=0, config=self.model)
        position.preprocess(all_frames)
        position.segment()

        roi = position.rois[0]
        masks = [
            label_seg(binary_mask)
            for binary_mask in tqdm(roi.seg_stack, desc="DeLTA labeling")
        ]

        ov = overlay_from_masks(np.stack(masks))

        attribute_segmentation(ov, self)

        return ov
