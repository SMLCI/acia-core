"""Functionality for creating outputs (export) the segmentation information"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tqdm.auto as tqdm

from acia.base import BaseImage, ImageRoISource, ImageSequenceSource, Overlay
from acia.utils import ScaleBar
from acia.viz import VideoExporter2


class DatasetExporter:
    """Base class for dataset exporters"""

    def __init__(self):
        self.sources = []

    def add(self, item: ImageRoISource | list[ImageRoISource]):
        if isinstance(item, Iterable):
            # iterable
            self.sources += item
        else:
            # not iterable
            self.sources.append(item)


class MMSegmentationDataset(DatasetExporter):
    """MMSegmentation dataset exporter"""

    def __init__(
        self,
        labels=None,
        label_coverter=lambda x: x,
    ):
        super().__init__()

        if labels is None:
            labels = ["Stem", "ThickRoot", "MediumRoot", "ThinRoot"]

        self.labels = labels
        self.label_converter = label_coverter

    def write(self, base_folder: str | Path = "data", mode="train"):
        img_path = Path(base_folder).absolute() / "img_dir" / mode
        ann_path = Path(base_folder).absolute() / "ann_dir" / mode

        img_path.mkdir(parents=True, exist_ok=True)
        ann_path.mkdir(parents=True, exist_ok=True)

        for input_index, image_roi_source in enumerate(tqdm.tqdm(self.sources)):
            for image_index, (image, rois) in enumerate(image_roi_source):
                if len(rois) == 0:
                    print("Skip")
                    continue

                # save image file
                image_file_path = img_path / f"{input_index:03d}_{image_index:03d}.png"
                mask_file_path = ann_path / f"{input_index:03d}_{image_index:03d}.png"
                cv2.imwrite(
                    str(image_file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )

                height, width = image.shape[:2]

                image_mask = np.zeros((height, width), dtype=np.uint8)

                # save roi masks
                for i, label in enumerate(self.labels):
                    label_value = i + 1
                    rois_for_label = filter(
                        partial(
                            lambda r, label: self.label_converter(r.label) == label,
                            label=label,
                        ),
                        rois,
                    )

                    label_mask = np.zeros((height, width), dtype=np.uint8)

                    for roi in rois_for_label:
                        roi_mask = roi.toMask(
                            height,
                            width,
                            fillValue=label_value,
                            outlineValue=label_value,
                        )

                        label_mask |= roi_mask

                    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

                    dil_label_mask = cv2.dilate(label_mask, kernel)
                    ero_label_mask = cv2.erode(dil_label_mask, kernel)

                    label_mask = ero_label_mask

                    image_mask[label_mask > 0] = (
                        label_mask.astype(np.uint8) * label_value
                    )[label_mask > 0]

                cv2.imwrite(str(mask_file_path), image_mask)


def no_crop(frame: int, _: Overlay):
    return (slice(0, frame.shape[0]), slice(0, frame.shape[1]))  # type: ignore[attr-defined]


def __video_export_from_str(
    filename: str, codec: Literal["vp09", "mjpg", "h264", "h265"], framerate: int
) -> VideoExporter2:
    """Create video exporter from string

    Args:
        filename (str): video filename
        codec (Literal[&quot;vp09&quot;, &quot;mjpg&quot;, &quot;h264&quot;, &quot;h265&quot;]): codec string
        framerate (int): framerate (in fps)

    Raises:
        ValueError: If codec is not found

    Returns:
        VideoExporter2: generate appropriate video exporter
    """
    if codec == "vp09":
        ve = VideoExporter2.default_vp9(filename=filename, framerate=framerate)  # type: ignore[arg-type]
    elif codec == "h264":
        ve = VideoExporter2.default_h264(filename=filename, framerate=framerate)  # type: ignore[arg-type]
    elif codec == "h265":
        ve = VideoExporter2.default_h265(filename=filename, framerate=framerate)  # type: ignore[arg-type]
    elif codec == "mjpg":
        ve = VideoExporter2.default_mjpg(filename=filename, framerate=framerate)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown/Unsupported codec: {codec}")

    return ve  # type: ignore[no-any-return]


def renderVideo(
    imageSource: ImageSequenceSource,
    roiSource=None,
    filename="output.mp4",
    framerate=3,
    codec: Literal["vp09", "mjpg", "h264", "h265"] = "vp09",
    scaleBar: ScaleBar | None = None,
    draw_frame_number=False,
    cropper=no_crop,
    filter_contours=lambda i, cont: True,
    cell_color=(255, 255, 0),
):
    """Render a video of the time-lapse.

    Args:
        imageSource (ImageSequenceSource): Your time-lapse source object.
        roiSource ([type]): Your source of RoIs for the image (e.g. cells). If None, no RoIs are visualized. Defaults to None.
        filename (str, optional): The output path of the video. Defaults to 'output.mp4'.
        framerate (int, optional): The framerate of the video. E.g. 3 means three time-lapse images per second. Defaults to 3.
        codec (str, optional): The video format codec. Defaults to "vp09".
        scaleBar (ScaleBar, optional): The scale bar object. Defaults to None.
        draw_frame_number (bool, optional): Whether to draw the frame number. Defaults to False.
        cropper ([type], optional): The frame cropper object. Defaults to no_crop.
    """

    if roiSource is None:
        # when we have no rois -> create iterator that always returns None
        def always_none():
            while True:
                yield None

        roiSource = iter(always_none())

    # make codec lower case
    codec = codec.lower()  # type: ignore[assignment]

    # create the video exporter
    ve = __video_export_from_str(filename, codec, framerate)

    with ve:
        for frame, (image, overlay) in enumerate(
            tqdm.tqdm(zip(imageSource, roiSource, strict=False))
        ):
            # extract the numpy image
            if isinstance(image, BaseImage):
                image = image.raw
            elif isinstance(image, np.ndarray):
                pass
            else:
                raise Exception("Unsupported image type!")

            # copy image as we draw onto it
            image = np.copy(image)

            crop_parameters = cropper(image, overlay)
            image = image[crop_parameters[0], crop_parameters[1]]

            height, width = image.shape[:2]

            # TODO: Draw float based contours
            # Draw overlay
            if overlay:
                image = cv2.drawContours(
                    image,
                    [
                        np.array(cont.coordinates).astype(np.int32)
                        for i, cont in enumerate(
                            overlay.croppedContours(crop_parameters)
                        )
                        if filter_contours(i, cont)
                    ],
                    -1,
                    cell_color,
                )  # RGB format

            if draw_frame_number:
                cv2.putText(
                    image,
                    f"Frame: {frame}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                )

            if scaleBar:
                image = scaleBar.draw(
                    image, width - scaleBar.pixelWidth - 10, height - 10  # type: ignore[attr-defined]
                )

            # output images
            ve.write(image)


def fast_mask_rendering(masks, im, colors, alpha=0.5):
    """
    Plot masks on image.

    Args:
        masks (tensor): Predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
        im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
        alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
        retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
    """

    colors = np.array(colors, dtype=float) / 255.0  # shape(n,3)
    colors = colors[:, None, None]  # shape(n,1,1,3)

    masks = np.expand_dims(masks, 3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = masks_color.max(axis=0)  # shape(n,h,w,3)

    im = im.astype(float) / 255
    im = im * inv_alpha_masks[-1] + mcs
    im_mask = (im * 255).astype(np.uint8)

    return im_mask


def fast_mask_rendering_torch(masks, im, colors, alpha=0.5):
    """
    Plot masks on image.

    Args:
        masks (tensor): Predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
        im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
        alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
        retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    colors = torch.tensor(colors, dtype=torch.float32) / 255.0  # shape(n,3)
    colors = colors[:, None, None]  # shape(n,1,1,3)

    masks = torch.tensor(masks)
    masks = masks.unsqueeze(3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

    im = torch.tensor(im, dtype=torch.float) / 255
    im = im * inv_alpha_masks[-1] + mcs
    im_mask = (im * 255).byte().numpy()

    return im_mask
