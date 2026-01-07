"""Utility functions for the tracking processors"""

import numpy as np

from acia.base import Overlay


def overlay_to_masks(segmentation: Overlay, height: int, width: int):
    masks = []
    for frame_ov in segmentation.timeIterator():
        # deal with empty masks
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(frame_ov) > 0:
            mask = frame_ov.toMasks(height=height, width=width, binary_mask=False)[0]

        masks.append(mask)

    # Convert to numpy inputs
    masks = np.stack(masks)

    return masks
