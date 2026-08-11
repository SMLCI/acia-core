"""Module for Contour Proposal Networks"""

import celldetection as cd
import numpy as np
import torch
from tqdm.auto import tqdm

from acia.attribute import attribute_segmentation
from acia.base import Contour, Overlay


class CPNSegmenter:
    """Contour Proposal Networks segmenter: https://github.com/FZJ-INM1-BDA/celldetection"""

    def __init__(self, nms_thresh=0.4):
        # Load pretrained model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = cd.fetch_model(
            "ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c", check_hash=True
        ).to(self.device)
        self.model.nms_thresh = nms_thresh
        self.model.eval()

    def __call__(self, image_sequence):
        contours = []
        max_frame = 0
        for frame_id, img in enumerate(
            tqdm(image_sequence, desc="Perform segmentation...")
        ):
            # Load input
            img = img.raw
            print(img.dtype, img.shape, (img.min(), img.max()))

            if len(img.shape) == 3:
                # we have HxWxC
                # strip of last channel to make it grayscale
                img = img[..., 0]

            # convert to rgb
            img = np.stack((img,) * 3, axis=-1)

            # Run model
            with torch.no_grad():
                x = cd.to_tensor(
                    img, transpose=True, device=self.device, dtype=torch.float32
                )
                x = x / x.max()  # ensure 0..1 range
                x = x[
                    None
                ]  # add batch dimension: Tensor[3, h, w] -> Tensor[1, 3, h, w]
                y = self.model(x)

            frame_ov = y["contours"][0]
            torch_frame_ov = frame_ov.cpu().numpy()
            for cont in torch_frame_ov:
                contours.append(Contour(cont, -1, frame_id, 0))

            max_frame = frame_id

        overlay = Overlay(contours, frames=list(range(max_frame + 1)))

        attribute_segmentation(overlay, self)

        return overlay
