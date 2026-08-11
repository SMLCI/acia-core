"""Module for general visualization functionality"""

from __future__ import annotations

import contextlib
import logging
import numbers
from collections.abc import Callable, Iterable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as iio

# --- Matplotlib / Plotly imports are optional until plotting is called ---
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import networkx as nx
import numpy as np
import pint
import plotly.graph_objects as go
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from acia import ureg
from acia.base import BaseImage, ImageSequenceSource, Instance, Overlay
from acia.segm.local import InMemorySequenceSource, LocalImage, THWCSequenceSource

from .compose import ComposedSequenceSource as ComposedSequenceSource
from .compose import compose_sequences as compose_sequences
from .compose import label_sequence as label_sequence
from .utils import strfdelta

# loda the deja vu sans default font
default_font = font_manager.findfont("DejaVu Sans")


def draw_scale_bar(
    image_iterator,
    xy_position: tuple[int, int],
    size_of_pixel,
    bar_width,
    bar_height,
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color=None,
    background_margin_pixel=3,
):
    """Draws a scale bar on all images of an image sequence or iterable image array

    Args:
        image_iterator: image sequence or iterator over images
        xy_position (tuple[int, int]): lower left xy position of the scale bar
        size_of_pixel (_type_): metric size of a pixel (e.g. 0.007 * ureg.micrometer)
        bar_width (_type_): width of the scale bar (e.g. 5 * ureg.micrometer)
        short_title (str, optional): Short title of the unit to be displayed. Defaults to "μm".
        color (tuple, optional): Color of scale bar and text. Defaults to (255, 255, 255).
        font_size (int, optional): text font size. Defaults to 25.
        font_path (str, optional): text font. Defaults to "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf".
        background_color: color for a potential background rectangle (e.g. (0, 0, 0)). Defaults to None (no background drawn).
        background_margin_pixel: pixels of margin for the background rectangle

    Yields:
        np.ndarray | LocalImage: Image in numpy format or LocalImage (depending on the input format)
    """

    # create pint quantities (values and units)
    bar_width = ureg.Quantity(bar_width)
    bar_height = ureg.Quantity(bar_height)
    size_of_pixel = ureg.Quantity(size_of_pixel)

    # load font
    font = ImageFont.truetype(font_path, font_size)

    # compute width and height of the scale bar in pixels (we need to round here)
    bar_pixel_width = int(
        np.round((bar_width / size_of_pixel).to_base_units().magnitude)
    )
    bar_pixel_height = int(
        np.round((bar_height / size_of_pixel).to_base_units().magnitude)
    )

    # extract position
    xstart, ystart = xy_position

    for image in image_iterator:
        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        # compute text size
        text = f"{bar_width:~P}"
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # get size of text
        left, top, right, bottom = draw.textbbox((xstart, ystart), text, font=font)

        text_width = right - left
        text_height = bottom - top

        if background_color:
            cv2.rectangle(
                image,
                (xstart - background_margin_pixel, ystart + background_margin_pixel),
                (
                    xstart + bar_pixel_width + background_margin_pixel,
                    ystart
                    - text_height
                    - bar_pixel_height
                    - 5
                    - background_margin_pixel,
                ),
                background_color,
                -1,
            )

        # draw scale bar
        cv2.rectangle(
            image,
            (xstart, ystart),
            (xstart + bar_pixel_width, ystart - bar_pixel_height),
            (255, 255, 255),
            -1,
        )

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # draw text centered and with distance to the scale bar
        draw.text(
            (
                xstart + bar_pixel_width / 2 - text_width / 2,
                ystart - text_height - bar_pixel_height - 10,
            ),
            text,
            fill=color,
            font=font,
        )

        # convert PIL image back to numpy
        image = np.array(img_pil)

        # do the image wrapping
        if is_wrapped:
            yield LocalImage(image)
        else:
            yield image


def draw_time(
    image_iterator,
    xy_position,
    time_step,
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color=None,
    background_margin_pixel=3,
):
    """Draw time onto images

    Args:
        image_iterator (_type_): image sequence or iterator over images
        xy_position (tuple[int, int]): lower left xy position of the time text
        time_step (_type_): time step between images (e.g. 15 * ureg.minute or "15 minute")
        color (_type): Color of the time text. Defaults to (255, 255, 255) which is white.
        font_size (int, optional): text font size. Defaults to 25.
        font_path (str, optional): text font. Defaults to "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf".
        background_color: color for a potential background rectangle (e.g. (0, 0, 0)). Defaults to None (no background drawn).
        background_margin_pixel: pixels of margin for the background rectangle

    Yields:
        _type_: _description_
    """

    time_step = ureg.Quantity(time_step)

    # load font
    font = ImageFont.truetype(font_path, font_size)

    for frame, image in enumerate(image_iterator):
        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        # convert to pillow image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # extract time in hours and minutes
        time = (frame * time_step).to(ureg.hour)
        hours = int(np.floor(time.magnitude))
        minutes = int(np.round((time - hours * ureg.hour).to("minute").magnitude))

        time_text = f"Time: {hours:2d}:{minutes:02d} h"

        if background_color:
            # get size of text
            left, top, right, bottom = draw.textbbox(xy_position, time_text, font=font)

            text_width = right - left
            text_height = bottom - top

            x, y = xy_position

            cv2.rectangle(
                image,
                (x - background_margin_pixel, y - background_margin_pixel),
                (
                    x + text_width + background_margin_pixel,
                    y + text_height + background_margin_pixel + 5,
                ),
                background_color,
                -1,
            )

            # convert to pillow image
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

        # draw on image
        draw.text(xy_position, time_text, fill=color, font=font)

        # convert PIL image back to numpy
        image = np.array(pil_image)

        # do the image wrapping
        if is_wrapped:
            yield LocalImage(image)
        else:
            yield image


class VideoExporter:
    """
    Wrapper for opencv video writer. Simplifies usage
    """

    def __init__(self, filename, framerate, codec="MJPG"):
        self.filename = filename
        self.framerate = framerate
        self.out = None
        self.frame_height = None
        self.frame_width = None
        self.codec = codec

    def __del__(self):
        if self.out:
            self.close()

    def write(self, image):
        height, width = image.shape[:2]
        if self.out is None:
            self.frame_height, self.frame_width = image.shape[:2]
            self.out = cv2.VideoWriter(
                self.filename,
                cv2.VideoWriter_fourcc(*self.codec),
                self.framerate,
                (self.frame_width, self.frame_height),
            )
        if self.frame_height != height or self.frame_width != width:
            logging.warning(
                "You add images of different resolution to the VideoExporter. This may cause problems (e.g. black video output)!"
            )
        self.out.write(image)

    def close(self):
        if self.out:
            self.out.release()
            self.out = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.out is None:
            logging.warning(
                "Closing video writer without any images written and no video output generated! Did you forget to write the images="
            )
        self.close()


class VideoExporter2:
    """
    Wrapper for opencv video writer. Simplifies usage
    """

    def __init__(
        self, filename: Path, framerate: int, codec="mjpeg", ffmpeg_params=None
    ):
        self.filename = Path(filename)
        self.framerate = framerate
        self.codec = codec

        if ffmpeg_params is None:
            ffmpeg_params = []

        self.ffmpeg_params = ffmpeg_params

        self.images: list = []

    @staticmethod
    def default_vp9(
        filename: Path,
        framerate: int,
    ):
        ffmpeg_params = ["-crf", "30", "-b:v", "0", "-speed", "1"]
        return VideoExporter2(
            filename, framerate, codec="libvpx-vp9", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def fast_vp9(
        filename: Path,
        framerate: int,
    ):
        ffmpeg_params = ["-crf", "35", "-b:v", "0", "-speed", "3"]
        return VideoExporter2(
            filename, framerate, codec="libvpx-vp9", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def default_h264(
        filename: Path,
        framerate: int,
    ):
        ffmpeg_params = ["-crf", "30", "-preset", "fast"]
        return VideoExporter2(
            filename, framerate, codec="libx264", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def default_h265(filename: Path, framerate: int):
        ffmpeg_params = ["-crf", "26", "-preset", "fast"]
        return VideoExporter2(
            filename, framerate, codec="libx265", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def default_mjpg(filename: Path, framerate: int):
        ffmpeg_params: list[str] = []
        return VideoExporter2(
            filename, framerate, codec="mjpeg", ffmpeg_params=ffmpeg_params
        )

    # av1 not yet supported
    #    @staticmethod
    #    def default_av1(filename: Path, framerate: int, ffmpeg_params=["-crf", "26", "-preset", "2", "-strict", "2"]):
    #        return VideoExporter2(filename, framerate, codec="libaom-av1", ffmpeg_params=ffmpeg_params)

    def write(self, image):
        self.images.append(image)

    def close(self):
        if len(self.images) == 0:
            logging.warning(
                "Closing video writer without any images written and no video output generated! Did you forget to write the images?"
            )
        else:
            # do the video rendering
            clip = mpy.ImageSequenceClip(
                list(self.images),
                fps=self.framerate,
            )
            clip.write_videofile(
                str(self.filename.absolute()),
                codec=self.codec,
                ffmpeg_params=self.ffmpeg_params,
                # verbose=False,
                # logger=None,
            )
            self.images = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def render_segmentation(
    imageSource: ImageSequenceSource,
    overlay: Overlay,
    cell_color=(255, 255, 0),
) -> ImageSequenceSource:
    """Render a video of the time-lapse including the segmentaiton information.

    Args:
        imageSource (ImageSequenceSource): Your time-lapse source object.
        Overlay ([type]): Your source of RoIs for the image (e.g. cells).
        cell_color: rgb color of the cell outlines
    """

    if overlay is None:
        # when we have no rois -> create iterator that always returns None
        def always_none():
            while True:
                yield None

        overlay = iter(always_none())

    images = []

    for image, frame_overlay in tqdm(
        zip(imageSource, overlay.timeIterator(), strict=False),
        desc="Render cell segmentation...",
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

        if len(image.shape) == 2:
            # convert to grayscale if needed
            image = np.stack((image,) * 3, axis=-1)

        if len(image.shape) != 3 or image.shape[2] != 3:
            logging.warning(
                "Your images are in the wrong shape! The shape of an image is %s but we need (height, width, 3)! This is likely to cause an error!",
                image.shape,
            )

        # Draw overlay
        if frame_overlay:
            image = frame_overlay.draw(image, cell_color)  # RGB format

        images.append(image)

    # return as sequence source again
    return InMemorySequenceSource(np.stack(images))


def render_cell_centers(
    image_source: ImageSequenceSource | np.ndarray,
    overlay: Overlay,
    center_color=(255, 255, 0),
    center_size=3,
) -> ImageSequenceSource:
    """Render a image sequence of the time-lapse with the cell centers.

    Args:
        imageSource (ImageSequenceSource): Your time-lapse source object.
        overlay (Overlay, optional): Your source of RoIs for the image (e.g. cells).
        center_color (tuple, optional): RGB color of the cell center circle. Defaults to (255, 255, 0).
        center_size (int, optional): Radius of the cell center circle (in pixels). Defaults to 3.

    Raises:
        ValueError: If we recognize unsupported image type or format

    Returns:
        ImageSequenceSource: The rendered image sequence
    """

    if overlay is None:
        # when we have no rois -> create iterator that always returns None
        def always_none():
            while True:
                yield None

        overlay = iter(always_none())

    images = []

    for image, frame_overlay in tqdm(
        zip(image_source, overlay.timeIterator(), strict=False),
        desc="Render cell centers...",
    ):
        # extract the numpy image
        if isinstance(image, BaseImage):
            image = image.raw
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("Unsupported image type!")

        # copy image as we draw onto it
        image = np.copy(image)

        # Draw overlay
        if frame_overlay:
            # compute all centers
            centers = [cont.center for cont in frame_overlay]

            for center in centers:
                int_center = tuple(map(int, center))

                cv2.circle(image, int_center, center_size, center_color, -1)

        images.append(image)

    image_stack = np.stack(images)

    # return as a sequence source
    return InMemorySequenceSource(image_stack)


def render_tracking(
    image_source: ImageSequenceSource,
    overlay: Overlay,
    tracking_graph: nx.DiGraph,
) -> ImageSequenceSource:
    """Render the tracking to an image source

    Args:
        image_source (ImageSequenceSource): Image source
        overlay (Overlay): overla of cell detections (for center points)
        tracking_graph (nx.DiGraph): the tracking graph where every cell detection is a node in the graph.

    Returns:
        ImageSequenceSource: Rendered image source
    """

    images = []

    # Centers are constant, but deriving one is expensive (a shapely centroid
    # for Contour, a full-frame mask pass for Instance) and the draw loop below
    # needs each of them once per incident edge. Resolve them all up front.
    center_lookup = {
        cont.id: np.asarray(cont.center, dtype=np.float64).astype(np.int32)
        for cont in overlay
    }

    # marker colors
    line_color = (255, 0, 0)  # rgb: red
    division_color = (0, 0, 255)  # bgr: blue
    marker_color = (203, 192, 255)

    for image, frame_overlay in zip(
        tqdm(image_source, desc="Render cell tracking paths..."),
        overlay.timeIterator(),
        strict=False,
    ):
        raw = image.raw

        if raw.ndim != 2 and (raw.ndim != 3 or raw.shape[2] not in (1, 3)):
            logging.warning(
                "Your images are in the wrong shape! The shape of an image is %s but we need (height, width, 3)! This is likely to cause an error!",
                raw.shape,
            )

        np_image = _to_uint8_rgb(raw)

        # Draw order is kept cell-by-cell (rather than batched per color) so
        # that overlapping tracks stack exactly as they did before.
        for cont in frame_overlay:
            if cont.id not in tracking_graph.nodes:
                continue

            edges = tracking_graph.out_edges(cont.id)

            if len(edges) == 0:
                center = center_lookup[cont.id]
                cv2.rectangle(
                    np_image,
                    tuple(map(int, center - 2)),
                    tuple(map(int, center + 2)),
                    marker_color,
                )
                continue

            # more than one successor -> the cell divides in this frame
            color = division_color if len(edges) > 1 else line_color
            born = tracking_graph.in_degree(cont.id) == 0

            for edge in edges:
                source = center_lookup[edge[0]]
                target = center_lookup[edge[1]]

                cv2.line(
                    np_image,
                    tuple(map(int, source)),
                    tuple(map(int, target)),
                    color,
                    thickness=3,
                )

                if born:
                    cv2.circle(
                        np_image,
                        tuple(map(int, source)),
                        3,
                        marker_color,
                        thickness=1,
                    )

        images.append(np_image)

    return InMemorySequenceSource(images)


def render_video(
    image_source: ImageSequenceSource,
    filename: str,
    framerate: int = 10,
    codec: str = "libx264",
    pixelformat: str = "yuv420p",
    macro_block_size: int = 2,
    ffmpeg_params: list[str] | None = None,
) -> None:
    """Render video

    Args:
        image_source (ImageSequenceSource): sequence of images
        filename (str): video filename
        framerate (int): framerate of the video
        codec (str): the codec for video encoding
        pixelformat (str): output pixel format. "yuv420p" is required by most
            browsers/players (e.g. Firefox rejects other chroma subsampling).
        macro_block_size (int): frame dimensions are padded up to a multiple of
            this value. imageio's own default of 16 stretches typical
            (already-even) frame sizes noticeably; 2 is the minimum yuv420p
            allows and avoids that visible distortion.
        ffmpeg_params (list[str] | None): extra output ffmpeg arguments, applied
            in addition to "-movflags +faststart" (which moves the moov atom to
            the front of the file so it plays back in browsers instead of
            failing to load).
    """

    with iio.get_writer(
        filename,
        fps=framerate,
        codec=codec,
        pixelformat=pixelformat,
        macro_block_size=macro_block_size,
        output_params=["-movflags", "+faststart"],
        ffmpeg_params=ffmpeg_params,
    ) as writer:
        for im in tqdm(image_source, desc="Encoding video..."):
            image = im.raw

            if len(image.shape) == 2:
                # convert to grayscale if needed
                image = np.stack((image,) * 3, axis=-1)

            if len(image.shape) != 3 or image.shape[2] != 3:
                logging.warning(
                    "Your images are in the wrong shape! The shape of an image is %s but we need (height, width, 3)! This is likely to cause an error!",
                    image.shape,
                )

            writer.append_data(image)  # type: ignore[attr-defined]


def render_scalebar(
    image_source: ImageSequenceSource,
    xy_position: tuple[int | float, int | float],
    size_of_pixel: pint.Quantity,
    bar_width: pint.Quantity,
    bar_height: pint.Quantity,
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color: tuple[int, int, int] | None = None,
    background_margin_pixel=3,
    show_text=True,
) -> ImageSequenceSource:
    """Draws a scale bar on all images of an image sequence or iterable image array

    Args:
        image_source (Overlay): image sequence or iterator over images
        xy_position (tuple[int, int]): lower left xy position of the scale bar
        size_of_pixel (pint.Quantity): metric size of a pixel (e.g. 0.007 * ureg.micrometer)
        bar_width (pint.Quantity): width of the scalebar (e.g. 5 * ureg.micrometer). Also the text over the bar.
        bar_height (pint.Quantity): height of the scalebar.
        color (tuple, optional): Color of the scalebar and text. Defaults to (255, 255, 255).
        font_size (int, optional): font size of the text. Defaults to 25.
        font_path (_type_, optional): path to the font. Defaults to default_font.
        background_color (tuple[int, int, int], optional): Color of the background. None draws no background. Defaults to None.
        background_margin_pixel (int, optional): Margin of the background box. Defaults to 3.
        show_text (bool, optional): If true shows the bar width as text above the bar. Defaults to True.

    Returns:
        ImageSequenceSource: Rendered image sequence
    """

    # create pint quantities (values and units)
    bar_width = ureg.Quantity(bar_width)
    bar_height = ureg.Quantity(bar_height)
    size_of_pixel = ureg.Quantity(size_of_pixel)

    # load font
    font = ImageFont.truetype(font_path, font_size)

    # compute width and height of the scale bar in pixels (we need to round here)
    bar_pixel_width = int(
        np.round((bar_width / size_of_pixel).to_base_units().magnitude)
    )
    bar_pixel_height = int(
        np.round((bar_height / size_of_pixel).to_base_units().magnitude)
    )

    image_height, image_width = image_source.get_frame(0).raw.shape[:2]

    # extract position
    xstart, ystart = xy_position

    # Allow relative positioning
    if isinstance(xstart, float):
        if xstart > 1.0:
            raise ValueError(
                f"If using float (x,y) position coordinates they have to be below 1. Your x position is {xstart}"
            )
        xstart = int(np.round(image_width * xstart))

    if isinstance(ystart, float):
        if ystart > 1.0:
            raise ValueError(
                f"If using float (x,y) position coordinates they have to be below 1. Your x position is {xstart}"
            )
        ystart = int(np.round(image_height * ystart))

    images = []

    for image in tqdm(image_source, desc="Render scale bar..."):
        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        image = np.copy(image)

        # compute text size
        text = f"{bar_width:~P}"
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # get size of text
        left, top, right, bottom = draw.textbbox((xstart, ystart), text, font=font)

        text_width = right - left
        text_height = bottom - top

        if background_color:
            cv2.rectangle(
                image,
                (xstart - background_margin_pixel, ystart + background_margin_pixel),
                (
                    xstart + bar_pixel_width + background_margin_pixel,
                    ystart
                    - text_height
                    - bar_pixel_height
                    - 5
                    - background_margin_pixel,
                ),
                background_color,
                -1,
            )

        # draw scale bar
        cv2.rectangle(
            image,
            (xstart, ystart),
            (xstart + bar_pixel_width, ystart - bar_pixel_height),
            (255, 255, 255),
            -1,
        )

        if show_text:
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)

            # draw text centered and with distance to the scale bar
            draw.text(
                (
                    xstart + bar_pixel_width / 2 - text_width / 2,
                    ystart - text_height - bar_pixel_height - 10,
                ),
                text,
                fill=color,
                font=font,
            )

            # convert PIL image back to numpy
            image = np.array(img_pil)

        images.append(image)

    # combine all images
    image_stack = np.stack(images)

    # return as a sequence source
    return InMemorySequenceSource(image_stack)


def render_time(
    image_source: ImageSequenceSource,
    xy_position: tuple[int | float, int | float],
    timepoints: list[pint.Quantity | timedelta],
    time_format="{H:02}h {M:02}m",
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color: tuple[int, int, int] | None = None,
    background_margin_pixel=3,
) -> ImageSequenceSource:
    """Draw time onto images

    Args:
        image_source (ImageSequenceSource): image sequence of the time-lapse
        xy_position (tuple[int]): lower left xy position of the formatted time text
        timepoints (list[pint.Quantity  |  timedelta]): timepoints of the individual frames
        time_format (str, optional): Timeformat for rendering the time to the images. Defaults to "{H:02}h {M:02}m".
        color (tuple, optional): Color of the time text. Defaults to (255, 255, 255).
        font_size (int, optional): Fontsize of the time text. Defaults to 25.
        font_path (_type_, optional): Path to the rendering font. Defaults to default_font.
        background_color (tuple[int, int, int], optional): Color of the background box. None does not draw any background box. Defaults to None.
        background_margin_pixel (int, optional): Margin of the background box. Defaults to 3.

    Returns:
        ImageSequenceSource: Rendered image sequence
    """

    # load font
    font = ImageFont.truetype(font_path, font_size)

    images = []

    image_height, image_width = image_source.get_frame(0).raw.shape[:2]

    # extract position
    xstart, ystart = xy_position

    # Allow relative positioning
    if isinstance(xstart, float):
        if xstart > 1.0:
            raise ValueError(
                f"If using float (x,y) position coordinates they have to be below 1. Your x position is {xstart}"
            )
        xstart = int(np.round(image_width * xstart))

    if isinstance(ystart, float):
        if ystart > 1.0:
            raise ValueError(
                f"If using float (x,y) position coordinates they have to be below 1. Your x position is {xstart}"
            )
        ystart = int(np.round(image_height * ystart))

    for image, timepoint in zip(
        tqdm(image_source, desc="Render time..."), timepoints, strict=False
    ):
        if isinstance(timepoint, pint.Quantity):
            timepoint = timedelta(seconds=float(timepoint.to(ureg.seconds).magnitude))

        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        image = np.copy(image)

        # convert to pillow image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        time_text = strfdelta(timepoint, fmt=time_format)

        if background_color:
            # get size of text
            left, top, right, bottom = draw.textbbox(xy_position, time_text, font=font)

            text_width = right - left
            text_height = bottom - top

            x, y = (xstart, ystart)

            cv2.rectangle(
                image,
                (x - background_margin_pixel, y - background_margin_pixel),
                (
                    x + text_width + background_margin_pixel,
                    y + text_height + background_margin_pixel + 5,
                ),
                background_color,
                -1,
            )

            # convert to pillow image
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

        # draw on image
        draw.text(xy_position, time_text, fill=color, font=font)

        # convert PIL image back to numpy
        image = np.array(pil_image)

        images.append(image)

    # combine all images
    image_stack = np.stack(images)

    # return as a sequence source
    return InMemorySequenceSource(image_stack)


def colorize_instance_mask(
    instance_mask, background_color=(0, 0, 0), seed=42, color_lut=None
) -> np.ndarray:
    """
    Convert instance mask to an RGB image with random colors per instance (no loop).

    Parameters:
        instance_mask (np.ndarray): 2D array of shape (H, W) with integer instance IDs.
        background_color (tuple): RGB color for background (default black).
        seed (int): Random seed for consistent coloring.
        color_lut (np.ndarray): Ix3 lookup map for instance colors (I)

    Returns:
        np.ndarray: Colored mask of shape (H, W, 3), dtype=uint8.
    """
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background (assumed to be 0)

    if len(unique_ids) == 0:
        return np.zeros((*instance_mask.shape, 3), dtype=np.uint8)

    # Map instance IDs to color lookup table (LUT)
    rng = np.random.default_rng(seed)
    if color_lut is None:
        color_lut = np.zeros((np.max(unique_ids) + 1, 3), dtype=np.uint8)
        # color_lut[unique_ids] = rng.integers(0, 256, size=(len(unique_ids), 3), dtype=np.uint8)
        color_lut = rng.integers(
            0, 256, size=(np.max(unique_ids) + 1, 3), dtype=np.uint8
        )
        color_lut[0] = background_color

    # Map colors to mask using LUT
    colored_mask = color_lut[instance_mask]

    return np.asarray(colored_mask)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary frame into a HxWx3 uint8 RGB array.

    Overlay colors live in the 0-255 range, so blending them onto a uint16 or
    float frame requires bringing the frame into the same range first.

    The result is always a fresh, contiguous buffer -- renderers draw onto it
    with cv2, which mutates in place, so it must never alias the source data.

    Args:
        image (np.ndarray): frame in HxW, HxWx1 or HxWx3 layout (uint8/uint16/float).

    Returns:
        np.ndarray: HxWx3 uint8 copy of the frame.
    """
    im = image

    # Convert image to uint8 if necessary
    if im.dtype == np.uint16:
        im = (im / 256).astype(np.uint8)
    elif im.dtype in (np.float32, np.float64):
        im = np.clip(im * 255, 0, 255).astype(np.uint8)
    elif im.dtype != np.uint8:
        im = im.astype(np.uint8)

    # Convert grayscale images to RGB by duplicating channels
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    elif im.ndim == 3 and im.shape[2] == 1:
        im = np.stack([im[:, :, 0]] * 3, axis=-1)

    # copy=True is load-bearing: np.ascontiguousarray alone would return the
    # caller's array untouched for an already-contiguous uint8 RGB frame, and
    # the subsequent in-place cv2 drawing would then corrupt the source data.
    return np.array(im, dtype=np.uint8, order="C", copy=True)


def _contour_labels(contours: Sequence[Any], enumerate_fallback: bool) -> list[int]:
    """Resolve the integer label to rasterize for each contour.

    Args:
        contours (Sequence): contours/instances of a single frame.
        enumerate_fallback (bool): if True, contours whose ``label`` is None or
            not convertible to int fall back to their 1-based position. If
            False, such contours are skipped (label 0).

    Returns:
        list[int]: one label per contour.
    """
    labels = []
    for i, cont in enumerate(contours):
        label = i + 1 if enumerate_fallback else 0
        if cont.label is not None:
            # could not convert label to integer -> keep the fallback label
            with contextlib.suppress(ValueError, TypeError):
                label = int(cont.label)
        labels.append(label)
    return labels


def _frame_label_mask(
    contours: Sequence[Any],
    height: int,
    width: int,
    enumerate_fallback: bool = False,
) -> np.ndarray:
    """Rasterize one frame's contours into a single instance label mask.

    This is the hot path of every mask-based renderer. The naive formulation
    (one full-image ``mask == label`` plus ``np.maximum`` per cell) costs
    O(n_cells * height * width); every branch below is O(height * width).

    On overlapping pixels the higher label wins, matching the ``np.maximum``
    semantics of the original implementation.

    Args:
        contours (Sequence): contours/instances of a single frame.
        height (int): frame height.
        width (int): frame width.
        enumerate_fallback (bool): see :func:`_contour_labels`.

    Returns:
        np.ndarray: HxW uint32 label mask (0 = background).
    """
    contours = list(contours)
    if not contours:
        return np.zeros((height, width), dtype=np.uint32)

    labels = _contour_labels(contours, enumerate_fallback)

    first = contours[0]
    if isinstance(first, Instance) and all(
        isinstance(c, Instance) and c.mask is first.mask for c in contours
    ):
        # Fast path: acia.segm.formats.overlay_from_masks hands every instance
        # of a frame a reference to the same full-frame label mask, so the mask
        # we want already exists -- one LUT remap keeps the requested labels and
        # drops everything else, instead of one pass per cell.
        src = first.mask
        src_labels = np.asarray([c.label for c in contours])
        lut_size = int(max(int(src.max()), int(src_labels.max()), max(labels))) + 1
        lut = np.zeros(lut_size, dtype=np.uint32)
        lut[src_labels] = np.asarray(labels, dtype=np.uint32)
        return np.asarray(lut[src])

    # Slow path: write each contour into the shared buffer. Ascending label
    # order reproduces "higher label wins" without a per-cell np.maximum.
    # int32 rather than uint32 because cv2.fillPoly has no uint32 overload.
    local_mask = np.zeros((height, width), dtype=np.int32)

    for i in np.argsort(np.asarray(labels, dtype=np.int64), kind="stable"):
        cont = contours[i]

        if isinstance(cont, Instance):
            np.putmask(local_mask, cont.binary_mask, np.int32(labels[i]))
            continue

        # Instance must be handled above: its `coordinates` property derives a
        # shapely polygon from the mask and raises when the mask is empty.
        coordinates = getattr(cont, "coordinates", None)

        if coordinates is None:
            # anything exposing only the toMask() protocol
            np.putmask(
                local_mask, cont.toMask(height=height, width=width), np.int32(labels[i])
            )
        else:
            # cv2.fillPoly only touches the polygon bounding box, whereas
            # Contour.toMask rasterizes over the whole frame per contour.
            points = np.asarray(coordinates, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(local_mask, [points], int(labels[i]))

    return local_mask.astype(np.uint32)


def _blend_overlay(
    image: np.ndarray, colored_mask: np.ndarray, label_mask: np.ndarray, alpha: float
) -> np.ndarray:
    """Alpha-blend a colorized instance mask onto a frame, foreground only.

    Args:
        image (np.ndarray): HxWx3 uint8 frame (see :func:`_to_uint8_rgb`).
        colored_mask (np.ndarray): HxWx3 uint8 colorized instance mask.
        label_mask (np.ndarray): HxW label mask deciding what counts as foreground.
        alpha (float): weight of the original image (1.0 = original only).

    Returns:
        np.ndarray: HxWx3 uint8 blended frame.
    """
    # uint8 saturating arithmetic -- avoids six full-frame float32 temporaries
    blended = cv2.addWeighted(image, alpha, colored_mask, 1 - alpha, 0)

    # keep the original image where no overlay is available. Testing the label
    # mask rather than the colors means a cell that happens to be colored
    # (0, 0, 0) still counts as foreground.
    result = image.copy()
    np.copyto(result, blended, where=(label_mask != 0)[..., None])
    return result


def get_mask(self, height, width, binary_mask=True) -> np.ndarray:
    """
    Turn the individual overlays into masks. For every time point we create a mask of all contours.

    returns: List of masks (np.array[bool])

    height: height of the image
    width: width of the image
    """
    label_mask = _frame_label_mask(list(self), height, width, enumerate_fallback=True)

    if binary_mask:
        return np.asarray(label_mask != 0)

    return label_mask


def render_overlay_frame(
    image: np.ndarray, overlay: Overlay, frame_idx: int, alpha: float = 0.8
) -> np.ndarray:
    """Render segmentation overlay on a single frame image with alpha blending.

    This helper function applies a segmentation overlay to a single frame image,
    blending the colorized instance mask with the original image using alpha transparency.
    Used by Jupyter widget callbacks for interactive overlay rendering.

    Args:
        image (np.ndarray): Single frame image in HxW or HxWxC format (uint8 or uint16).
        overlay (Overlay): Overlay object containing contours for the current frame
            (typically from overlay.time_iterator()).
        frame_idx (int): Frame index for reference (not directly used, for compatibility).
        alpha (float, optional): Alpha blending weight for the original image.
            Default 0.8 means 80% original image, 20% overlay. Valid range: 0.0-1.0.

    Returns:
        np.ndarray: Blended image with overlay applied, dtype uint8, shape HxWx3 (RGB).
            Where the overlay has no data (background), returns the original image unchanged.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Normalize to a HxWx3 uint8 copy (never touches the caller's array)
    im = _to_uint8_rgb(image)

    # Convert overlay to instance label mask (each instance has a unique ID)
    label_mask = _frame_label_mask(
        list(overlay), height, width, enumerate_fallback=True
    )

    # Colorize the instance mask (assigns random colors per instance)
    colored_mask = colorize_instance_mask(label_mask)

    # Alpha blend the colored mask onto the image, foreground pixels only.
    # alpha controls the mix: higher alpha = more original image, lower = more overlay
    return _blend_overlay(im, colored_mask, label_mask, alpha)


def render_segmentation_mask(
    source: ImageSequenceSource,
    overlay: Overlay,
    alpha=0.8,
    *,
    colors=None,
    palette: dict | None = None,
    cmap: str = "viridis",
    default_color: tuple[int, int, int] = (120, 120, 120),
) -> THWCSequenceSource:
    """Render cell segmentation masks, colored randomly (default) or from a table.

    Args:
        source (ImageSequenceSource): the time-lapse sequence source.
        overlay (Overlay): the corresponding overlay. WARNING: all instances need
            to be based on masks!
        alpha (float, optional): weight of the *original* image in the blend --
            ``1.0`` keeps the frame unchanged, ``0.0`` shows solid mask color.
            Defaults to 0.8.
        colors: optional per-cell coloring, matched to contours by **id** -- a
            ``pandas.Series`` indexed by contour id (i.e. one column of a property
            table), a ``dict`` ``{id: value}``, or a single-column ``DataFrame``.
            Each value becomes a color: an ``(r, g, b)`` triple is used directly, a
            number is mapped through ``cmap`` (continuous), anything else is
            categorical (one distinct color per category, from ``palette`` or a
            qualitative colormap). ``None`` (default) keeps the original random
            per-instance colors.
        palette: optional ``{category: (r, g, b)}`` mapping for categorical
            ``colors`` (its keys double as the color legend).
        cmap: matplotlib colormap name used for numeric ``colors``.
        default_color: color for cells absent from ``colors`` (or with ``NaN``).

    Returns:
        THWCSequenceSource: TxHxWx3 sequence
    """
    id_to_rgb = None
    if colors is not None:
        id_to_rgb, _ = _resolve_cell_colors(
            _as_id_value_dict(colors), palette, cmap, default_color
        )

    return_images = []

    for im, ov in zip(
        tqdm(source, desc="Render segmentation masks..."),
        overlay.time_iterator(),
        strict=False,
    ):
        raw = im.raw
        height, width = raw.shape[:2]
        image = _to_uint8_rgb(raw)

        conts = list(ov)
        label_mask = _frame_label_mask(conts, height, width, enumerate_fallback=True)

        if id_to_rgb is None:
            # random per-instance colors (original behaviour)
            colored_mask = colorize_instance_mask(label_mask)
        else:
            # drive the color LUT from the table: each contour's rasterized label
            # -> the color looked up for that contour's id
            labels = _contour_labels(conts, enumerate_fallback=True)
            lut = np.zeros((int(label_mask.max()) + 1, 3), dtype=np.uint8)
            for cont, lbl in zip(conts, labels, strict=False):
                lut[lbl] = id_to_rgb.get(cont.id, default_color)
            colored_mask = colorize_instance_mask(label_mask, color_lut=lut)

        return_images.append(_blend_overlay(image, colored_mask, label_mask, alpha))

    # return the new time-lapse
    return THWCSequenceSource(np.stack(return_images, axis=0))


def _as_id_value_dict(values) -> dict:
    """Normalize a Series / single-column DataFrame / dict into ``{id: value}``."""
    if hasattr(values, "columns"):  # DataFrame
        if values.shape[1] != 1:
            raise ValueError(
                "values DataFrame must have exactly one column (pass e.g. df['col'])"
            )
        values = values.iloc[:, 0]
    if hasattr(values, "to_dict"):  # pandas Series
        return dict(values.to_dict())
    return dict(values)


def _is_rgb(v) -> bool:
    return (
        isinstance(v, (tuple, list, np.ndarray))
        and len(v) == 3
        and all(isinstance(x, numbers.Number) and not isinstance(x, bool) for x in v)
    )


def _resolve_cell_colors(id_value, palette, cmap, default_color):
    """Map ``{id: value}`` -> ``({id: (r,g,b)}, {category: (r,g,b)} legend)``.

    ``(r,g,b)`` values pass through; numeric values map through ``cmap``
    (continuous, no discrete legend); everything else is treated as categorical.
    """
    present = {
        i: v
        for i, v in id_value.items()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    }
    vals = list(present.values())
    id_to_rgb: dict = {}
    legend: dict = {}

    if vals and all(_is_rgb(v) for v in vals):
        for i, v in present.items():
            id_to_rgb[i] = tuple(int(x) for x in v)
        return id_to_rgb, legend

    if vals and all(
        isinstance(v, numbers.Number) and not isinstance(v, bool) for v in vals
    ):
        arr = np.array([float(v) for v in vals], dtype=float)
        lo = float(arr.min())
        span = float(arr.max()) - lo or 1.0
        cm = mpl.colormaps[cmap] if cmap in mpl.colormaps else mpl.colormaps["viridis"]
        for i, v in present.items():
            r, g, b, _ = cm((float(v) - lo) / span)
            id_to_rgb[i] = (int(r * 255), int(g * 255), int(b * 255))
        return id_to_rgb, legend

    # categorical: one distinct color per category
    categories = sorted({str(v) for v in vals})
    resolved = {str(k): tuple(int(x) for x in c) for k, c in (palette or {}).items()}
    qualitative = mpl.colormaps["tab10"]
    for k, cat in enumerate(categories):
        if cat not in resolved:
            r, g, b, _ = qualitative(k % qualitative.N)
            resolved[cat] = (int(r * 255), int(g * 255), int(b * 255))
    for i, v in present.items():
        id_to_rgb[i] = resolved.get(str(v), default_color)
    legend = {cat: resolved[cat] for cat in categories}
    return id_to_rgb, legend


def render_tracking_mask(
    source: ImageSequenceSource,
    overlay: Overlay,
    alpha=0.8,
    show_label_numbers=False,
    seed=42,
) -> THWCSequenceSource:
    """Render tracking and use the label colors for the masks

    Args:
        source (ImageSequenceSource): the time-lapse sequence source
        overlay (Overlay): the corresponding overlay. WARNING: all instances need to be based on masks!
        alpha (float, optional): The opacity of the masked image. Defaults to 0.8.

    Returns:
        THWCSequenceSource: TxHxWx3 sequence
    """
    return_images = []

    # generate color LUT (persistent for labels). Only the label range matters
    # here, so take the max directly instead of sorting every label.
    rng = np.random.default_rng(seed)
    max_label = max((int(cont.label) for cont in overlay), default=0)
    color_lut = rng.integers(0, 256, size=(max_label + 1, 3), dtype=np.uint8)
    color_lut[0] = (0, 0, 0)

    for im, ov in zip(
        tqdm(source, desc="Render tracking mask..."),
        overlay.time_iterator(),
        strict=False,
    ):
        raw = im.raw
        h, w = raw.shape[:2]
        image = _to_uint8_rgb(raw)

        contours = list(ov)

        # single-pass rasterization of the whole frame (see _frame_label_mask)
        label_mask = _frame_label_mask(contours, h, w)

        # render label numbers if necessary
        if show_label_numbers:
            for cont in contours:
                cv2.putText(
                    image,
                    f"{cont.label}",
                    tuple(map(int, np.array(cont.center).astype(int))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # render the masks based on the labels
        colored_mask = colorize_instance_mask(label_mask, color_lut=color_lut)

        return_images.append(_blend_overlay(image, colored_mask, label_mask, alpha))

    # return the new time-lapse
    return THWCSequenceSource(np.stack(return_images, axis=0))


###########################################
# Add new lineage rendering functionality #
###########################################


# ========== LAYOUT ==========
def compute_lineage_y(G: nx.DiGraph, time_feature: str = "t") -> dict[Any, float]:
    """
    Assign a y-position to each node using a tidy tree layout.

    Parameters
    ----------
    G : nx.DiGraph
        The lineage graph.
    time_feature : str
        The node attribute that encodes time.

    Returns
    -------
    assigned_y : dict
        Mapping from node to y-coordinate (float).
    """
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    assigned_y: dict[Any, float] = {}
    next_y = [0]

    def assign_y_iterative():
        # Assign unique y-coordinates to all tips, then propagate up for inner nodes.
        stack: list = []
        visited = set()
        # Start with roots (nodes with no parents), sorted by time
        for root in sorted(roots, key=lambda n: G.nodes[n][time_feature]):
            stack.append((root, 0))
            while stack:
                node, depth = stack.pop()
                if node in visited:
                    continue
                children = list(G.successors(node))
                if not children:
                    # Assign new y to each leaf node
                    assigned_y[node] = next_y[0]
                    next_y[0] += 1
                else:
                    # Process children before parent (postorder)
                    stack.append((node, depth))
                    for child in reversed(children):
                        if child not in visited:
                            stack.append((child, depth + 1))
                    visited.add(node)
                    continue
                visited.add(node)
        # For internal nodes, set y as average of children
        for root in roots:
            postorder = list(nx.dfs_postorder_nodes(G, source=root))
            for node in postorder:
                children = list(G.successors(node))
                if children:
                    assigned_y[node] = sum(assigned_y[c] for c in children) / len(
                        children
                    )

    assign_y_iterative()
    return assigned_y


def extract_lineage_plotdata(
    G: nx.DiGraph,
    assigned_y: dict[Any, float],
    time_feature: str = "t",
    label_name: str | None = None,
    orientation: str = "horizontal",
) -> dict[str, Any]:
    """
    Collect all node and edge positions and hover info for plotting.
    """
    xs, ys, node_ids, node_labels, hover_texts = [], [], [], [], []
    for n in G.nodes:
        t = G.nodes[n][time_feature]
        y = assigned_y[n]
        xs.append(t if orientation == "horizontal" else y)
        ys.append(y if orientation == "horizontal" else t)
        node_ids.append(n)
        # label
        label = (
            str(G.nodes[n][label_name])
            if (label_name and label_name in G.nodes[n])
            else str(n)
        )
        node_labels.append(label)
        # hover HTML
        features = G.nodes[n]
        if features:
            maxk = max((len(str(k)) for k in features), default=1)

            def fmt(k, v, maxk):
                return f"{str(k).ljust(maxk)} : {v}<br>"

            feat_lines = "".join(fmt(k, v, maxk) for k, v in features.items())
            hover_html = f"<b>Node:</b> {n}<br><span style='font-family:monospace'>{feat_lines}</span>"
        else:
            hover_html = f"<b>Node:</b> {n}"
        hover_texts.append(hover_html)

    # edges
    edge_xs, edge_ys = [], []
    for n in G.nodes:
        t0 = G.nodes[n][time_feature]
        y0 = assigned_y[n]
        for c in G.successors(n):
            t1 = G.nodes[c][time_feature]
            y1 = assigned_y[c]
            if orientation == "horizontal":
                edge_xs.append([t0, t1])
                edge_ys.append([y0, y1])
            else:
                edge_xs.append([y0, y1])
                edge_ys.append([t0, t1])

    # births & ends
    births_x, births_y, ends_x, ends_y = [], [], [], []
    for n in G.nodes:
        t = G.nodes[n][time_feature]
        y = assigned_y[n]
        if G.in_degree(n) == 0:
            births_x.append(t if orientation == "horizontal" else y)
            births_y.append(y if orientation == "horizontal" else t)
        if G.out_degree(n) == 0:
            ends_x.append(t if orientation == "horizontal" else y)
            ends_y.append(y if orientation == "horizontal" else t)

    return dict(
        xs=xs,
        ys=ys,
        node_ids=node_ids,
        node_labels=node_labels,
        hover_texts=hover_texts,
        edge_xs=edge_xs,
        edge_ys=edge_ys,
        births_x=births_x,
        births_y=births_y,
        ends_x=ends_x,
        ends_y=ends_y,
    )


# ========== COLOR HELPERS ==========
def _value_getter(
    G: nx.DiGraph,
    node_ids: Iterable[Any],
    node_color_by: str | dict[Any, Any] | Callable[[Any], Any] | None,
) -> list[Any] | None:
    """Return list of values per node for coloring."""
    if node_color_by is None:
        return None
    vals: list[Any] = []
    if isinstance(node_color_by, str):
        for n in node_ids:
            vals.append(G.nodes[n].get(node_color_by, None))
    elif callable(node_color_by):
        for n in node_ids:
            vals.append(node_color_by(n))
    elif isinstance(node_color_by, dict):
        for n in node_ids:
            vals.append(node_color_by.get(n, None))
    else:
        # Series-like (has .get)
        try:
            for n in node_ids:
                vals.append(node_color_by.get(n, None))  # type: ignore[attr-defined]
        except Exception as e:
            raise TypeError(
                "node_color_by must be a str, dict/Series-like, or callable(node)->value"
            ) from e
    return vals


def _is_numeric_series(vals: Iterable[Any]) -> bool:
    any_val = next((v for v in vals if v is not None), None)
    return isinstance(any_val, numbers.Real) and not isinstance(any_val, bool)


# ========== MATPLOTLIB PLOT ==========
def plot_cell_lineage(
    G: nx.DiGraph,
    time_feature: str = "t",
    orientation: str = "horizontal",
    show_label: bool = True,
    label_name: str | None = None,
    node_marker: str = "o",
    node_ms: int = 6,
    line_color: str = "blue",
    line_lw: int | float = 2,
    mark_births: bool = False,
    birth_color: str = "red",
    birth_marker: str | None = None,
    birth_ms: int = 12,
    mark_ends: bool = False,
    end_color: str = "orange",
    end_marker: str = "s",
    end_ms: int = 10,
    ax: plt.Axes | None = None,
    interactive_tooltip: bool = False,
    # --- coloring controls ---
    node_color_by: str | dict[Any, Any] | Callable[[Any], Any] | None = None,
    node_edge_color: str = "none",
    node_cmap: str = "viridis",
    node_na_color: str = "#bbbbbb",
    show_colorbar: bool = True,
    show_legend: bool = True,
    colorbar_title: str | None = None,  # NEW: colorbar label (numeric coloring only)
):
    """
    Draw a cell lineage tree as a static matplotlib plot.
    """
    assigned_y = compute_lineage_y(G, time_feature)
    data = extract_lineage_plotdata(
        G, assigned_y, time_feature, label_name, orientation
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = None

    # edges
    for x, y in zip(data["edge_xs"], data["edge_ys"], strict=False):
        ax.plot(x, y, "-", color=line_color, lw=line_lw)

    # coloring
    colors = None
    if node_color_by is not None:
        vals = _value_getter(G, data["node_ids"], node_color_by) or []
        if _is_numeric_series(vals):
            vmin = min(v for v in vals if v is not None)
            vmax = max(v for v in vals if v is not None)
            if vmin == vmax:  # avoid zero-range
                vmin, vmax = vmin - 0.5, vmax + 0.5
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = mpl.colormaps.get_cmap(node_cmap)
            colors = [
                cmap(norm(v)) if v is not None else mcolors.to_rgba(node_na_color)
                for v in vals
            ]

            if show_colorbar:
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cb = plt.colorbar(sm, ax=ax, pad=0.01)
                title = (
                    colorbar_title
                    if colorbar_title
                    else (
                        str(node_color_by)
                        if isinstance(node_color_by, str)
                        else "value"
                    )
                )
                cb.set_label(title)
        else:
            # categorical → discrete palette
            uniq = list(dict.fromkeys(v for v in vals if v is not None))
            if len(uniq) == 0:
                colors = [node_na_color] * len(vals)
            else:
                base = mpl.colormaps.get_cmap(node_cmap).resampled(max(len(uniq), 3))
                lut = {u: base(i / max(len(uniq) - 1, 1)) for i, u in enumerate(uniq)}
                colors = [lut.get(v, mcolors.to_rgba(node_na_color)) for v in vals]
            if show_legend and len(uniq) > 0:
                # legend proxies
                handles, labels = [], []
                for u in uniq:
                    proxy = plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=lut[u],
                        markeredgecolor="none",
                        markersize=max(6, node_ms),
                    )
                    handles.append(proxy)
                    labels.append(str(u))
                legend_title = (
                    str(node_color_by) if isinstance(node_color_by, str) else "category"
                )
                ax.legend(handles, labels, title=legend_title, loc="best", frameon=True)

    # nodes
    main_nodes = ax.scatter(
        data["xs"],
        data["ys"],
        marker=node_marker,
        color=(colors if colors is not None else line_color),
        edgecolors=node_edge_color,
        s=node_ms**2,
        zorder=3,
    )

    if show_label:
        for x, y, label in zip(
            data["xs"], data["ys"], data["node_labels"], strict=False
        ):
            ax.text(x, y + 0.12, label, fontsize=7, ha="center", va="bottom")

    # births / ends
    if mark_births and data["births_x"]:
        marker = (
            birth_marker
            if birth_marker
            else (">" if orientation == "horizontal" else "v")
        )
        ax.scatter(
            data["births_x"],
            data["births_y"],
            marker=marker,
            color=birth_color,
            s=birth_ms**2,
            zorder=5,
            alpha=0.9,
            edgecolor="k",
        )
    if mark_ends and data["ends_x"]:
        ax.scatter(
            data["ends_x"],
            data["ends_y"],
            marker=end_marker,
            color=end_color,
            s=end_ms**2,
            zorder=5,
            alpha=0.9,
            edgecolor="k",
        )

    # axes
    if orientation == "horizontal":
        ax.set_xlabel("Time")
        ax.set_ylabel("Lineage")
    else:
        ax.set_ylabel("Time")
        ax.set_xlabel("Lineage")
        ax.invert_yaxis()
    ax.autoscale()
    ax.set_aspect("auto")
    plt.tight_layout()

    # optional tooltips
    if interactive_tooltip:
        try:
            # pylint: disable=import-outside-toplevel
            import mplcursors  # type: ignore

            cursor = mplcursors.cursor(main_nodes, hover=True)
            cursor.connect(
                "add",
                lambda sel: sel.annotation.set_text(
                    data["hover_texts"][sel.index]
                    .replace("<br>", "\n")
                    .replace("<b>", "")
                    .replace("</b>", "")
                    .replace("<span style='font-family:monospace'>", "")
                    .replace("</span>", "")
                ),
            )
        except ImportError:
            print("mplcursors not installed; install for interactive node tooltips.")
    return fig


# ========== PLOTLY PLOT ==========
def plotly_cell_lineage(
    G: nx.DiGraph,
    time_feature: str = "t",
    orientation: str = "horizontal",
    show_label: bool = True,
    label_name: str | None = None,
    node_marker: str = "circle",
    node_ms: int = 10,
    line_color: str = "blue",
    line_width: int | float = 2,
    mark_births: bool = False,
    birth_color: str = "red",
    birth_marker: str | None = None,
    birth_ms: int = 16,
    mark_ends: bool = False,
    end_color: str = "orange",
    end_marker: str = "square",
    end_ms: int = 14,
    figure_title: str = "Cell Lineage",
    fig_height: int = 500,
    fig_width: int = 1000,
    # --- coloring controls ---
    node_color_by: str | dict[Any, Any] | Callable[[Any], Any] | None = None,
    node_colorscale: str = "Viridis",  # for numeric
    node_na_color: str = "lightgray",
    show_colorbar: bool = True,
    show_legend: bool = True,
    colorbar_title: str | None = None,  # NEW: numeric colorbar title
    time_axis_label: str | None = None,  # override the time-axis title
):
    """
    Plot a cell lineage tree as an interactive Plotly chart.
    Node hover shows all features as a readable (monospace) "pseudo-table".

    ``time_axis_label`` overrides the title on whichever axis encodes
    ``time_feature`` (x when horizontal, y when vertical). When it is ``None``
    (the default), the title is auto-derived: ``f"Time [{unit}]"`` when the
    graph carries a ``graph["time_unit"]`` (as stamped by the tracker when the
    source was time-calibrated), else plain ``"Time"``.
    """
    assigned_y = compute_lineage_y(G, time_feature)
    data = extract_lineage_plotdata(
        G, assigned_y, time_feature, label_name, orientation
    )
    fig = go.Figure()

    # edges
    for x, y in zip(data["edge_xs"], data["edge_ys"], strict=False):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=line_color, width=line_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # helper to add node subsets
    def add_nodes_subset(
        idx: list[int],
        name: str | None = None,
        color: str | list[float] | None = None,
        colorscale: str | None = None,
        colorbar: dict | None = None,
    ):
        xs = [data["xs"][i] for i in idx]
        ys = [data["ys"][i] for i in idx]
        labels = [data["node_labels"][i] for i in idx] if show_label else None
        hovers = [data["hover_texts"][i] for i in idx]
        marker_kw = dict(symbol=node_marker, size=node_ms, line=dict(width=0))
        if color is not None:
            marker_kw["color"] = color
        if colorscale is not None:
            marker_kw["colorscale"] = colorscale
        if colorbar is not None:
            marker_kw["colorbar"] = colorbar
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text" if show_label else "markers",
                text=labels,
                textposition="top center",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hovers,
                marker=marker_kw,
                name=name if name else "Cells",
                showlegend=(name is not None and show_legend),
            )
        )

    # coloring
    vals = (
        _value_getter(G, data["node_ids"], node_color_by)
        if node_color_by is not None
        else None
    )
    if vals is None:
        add_nodes_subset(list(range(len(data["xs"]))), name="Cells", color=line_color)
    else:
        if _is_numeric_series(vals):
            color = list(vals)
            idx_valid = [i for i, v in enumerate(color) if v is not None]
            idx_na = [i for i, v in enumerate(color) if v is None]
            add_nodes_subset(
                idx_valid,
                name=str(node_color_by) if isinstance(node_color_by, str) else "value",
                color=[color[i] for i in idx_valid],
                colorscale=node_colorscale,
                colorbar=dict(
                    title=(
                        colorbar_title
                        if colorbar_title
                        else (
                            str(node_color_by)
                            if isinstance(node_color_by, str)
                            else "value"
                        )
                    )
                )
                if show_colorbar
                else None,
            )
            if idx_na:
                add_nodes_subset(idx_na, name="NA", color=node_na_color)
        else:
            # categorical → one trace per category for legend
            vals_clean = [v if v is not None else "NA" for v in vals]
            cats = list(dict.fromkeys(vals_clean))
            for cat in cats:
                idx = [i for i, v in enumerate(vals_clean) if v == cat]
                add_nodes_subset(idx, name=str(cat))  # Plotly cycles colors

    # births / ends
    if mark_births and data["births_x"]:
        marker = (
            birth_marker
            if birth_marker
            else (">" if orientation == "horizontal" else "v")
        )
        marker_map = {
            ">": "triangle-right",
            "<": "triangle-left",
            "^": "triangle-up",
            "v": "triangle-down",
            "o": "circle",
            "s": "square",
            "d": "diamond",
            "*": "star",
            "x": "x",
            "+": "cross",
        }
        fig.add_trace(
            go.Scatter(
                x=data["births_x"],
                y=data["births_y"],
                mode="markers",
                marker=dict(
                    symbol=marker_map.get(marker, "triangle-right"),
                    color=birth_color,
                    size=birth_ms,
                    line=dict(width=1, color=birth_color),
                ),
                name="Cell birth",
                hoverinfo="skip",
                showlegend=True,
            )
        )
    if mark_ends and data["ends_x"]:
        marker_map = {
            "s": "square",
            "o": "circle",
            "d": "diamond",
            "*": "star",
            "x": "x",
            "+": "cross",
            ">": "triangle-right",
            "<": "triangle-left",
            "^": "triangle-up",
            "v": "triangle-down",
        }
        fig.add_trace(
            go.Scatter(
                x=data["ends_x"],
                y=data["ends_y"],
                mode="markers",
                marker=dict(
                    symbol=marker_map.get(end_marker, "square"),
                    color=end_color,
                    size=end_ms,
                    line=dict(width=1, color=end_color),
                ),
                name="Cell end",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # axes & layout. Auto-label the time axis with the graph's own unit (set by
    # the tracker when it stamped real time on the nodes, e.g. "min"), unless the
    # caller gave an explicit override.
    if time_axis_label is not None:
        time_title = time_axis_label
    elif G.graph.get("time_unit"):
        time_title = f"Time [{G.graph['time_unit']}]"
    else:
        time_title = "Time"
    if orientation == "horizontal":
        fig.update_xaxes(title=time_title)
        fig.update_yaxes(title="Lineage")
    else:
        fig.update_xaxes(title="Lineage")
        fig.update_yaxes(title=time_title, autorange="reversed")
    fig.update_layout(
        title=figure_title, height=fig_height, width=fig_width, plot_bgcolor="white"
    )
    return fig


# ========== TRACKLET (CELL-CYCLE) LINEAGE ==========
def tracklet_graph_to_segments(tracklet_graph: nx.DiGraph) -> nx.DiGraph:
    """Reshape a tracklet graph into a "one line per cell cycle" segment graph.

    :func:`plotly_cell_lineage`/:func:`plot_cell_lineage` (and their shared
    layout helpers :func:`compute_lineage_y`/:func:`extract_lineage_plotdata`)
    only require a graph where every node carries a unique id and a scalar
    ``time_feature`` attribute -- they know nothing about tracklets. This
    function bridges that gap by turning each tracklet node ``n`` (with
    ``start_frame``/``end_frame`` int attributes, as produced by e.g.
    :func:`acia.tracking.formats.read_ctc_tracklet_graph`) into **two** point
    nodes, so a tracklet becomes a single line segment from its start to its
    end and a division becomes a branch from a parent segment's end-point to
    each child segment's start-point.

    Args:
        tracklet_graph: one node per tracklet, keyed by an arbitrary hashable
            label (e.g. the CTC integer label), with ``start_frame``/
            ``end_frame`` int attributes; edges ``parent -> child`` encode a
            division.

    When the tracklet graph carries real time on its nodes -- ``start_time``/
    ``end_time`` float attributes plus a ``graph["time_unit"]``, as stamped by
    :func:`acia.tracking.annotate_tracklet_times` at tracking time -- those are
    forwarded to the point nodes' ``"time"`` attribute (and the unit copied to
    ``segments.graph["time_unit"]``), so the lineage lays out on a real-time
    axis with no extra input. Otherwise only ``"frame"`` is carried.

    Returns:
        A new :class:`networkx.DiGraph` where each input node ``n`` becomes
        ``(n, "start")`` and ``(n, "end")``, each carrying a ``"frame"`` int
        attribute (``start_frame``/``end_frame`` respectively) and, when the
        input carried real time, a ``"time"`` float attribute. One
        intra-tracklet edge ``(n, "start") -> (n, "end")`` per tracklet, and
        one inter-tracklet edge ``(n, "end") -> (child, "start")`` for every
        division edge ``n -> child`` in the input. Suitable for
        ``plotly_cell_lineage(segments, time_feature="frame")`` (or
        ``time_feature="time"`` when the input carried real time).
    """
    segments = nx.DiGraph()

    # Real time is present iff every tracklet node carries start_time/end_time.
    has_time = tracklet_graph.number_of_nodes() > 0 and all(
        "start_time" in a and "end_time" in a
        for _, a in tracklet_graph.nodes(data=True)
    )

    for n, attrs in tracklet_graph.nodes(data=True):
        start_attrs = {"frame": attrs["start_frame"]}
        end_attrs = {"frame": attrs["end_frame"]}
        if has_time:
            start_attrs["time"] = attrs["start_time"]
            end_attrs["time"] = attrs["end_time"]
        segments.add_node((n, "start"), **start_attrs)
        segments.add_node((n, "end"), **end_attrs)
        segments.add_edge((n, "start"), (n, "end"))

    for parent, child in tracklet_graph.edges:
        segments.add_edge((parent, "end"), (child, "start"))

    if has_time and "time_unit" in tracklet_graph.graph:
        segments.graph["time_unit"] = tracklet_graph.graph["time_unit"]

    return segments


def plot_tracklet_lineage(tracklet_graph: nx.DiGraph, **kwargs):
    """Render a tracklet graph as a lineage tree with one line per cell cycle.

    One-line convenience wrapper hiding :func:`tracklet_graph_to_segments`
    behind a single call to :func:`plotly_cell_lineage`. ``time_feature`` is
    selected automatically -- ``"time"`` when the tracklet graph carries real
    time (``start_time``/``end_time``, as stamped by
    :func:`acia.tracking.annotate_tracklet_times`), else ``"frame"`` -- and is
    not caller-overridable; a caller who also passes ``time_feature=`` gets
    Python's own "got multiple values for keyword argument" ``TypeError``,
    which is the correct, sufficient failure mode here. When real time is
    present, the axis is auto-labelled from the graph's ``time_unit``.

    Args:
        tracklet_graph: see :func:`tracklet_graph_to_segments`.
        **kwargs: forwarded to :func:`plotly_cell_lineage` unchanged (e.g.
            ``orientation``, ``show_label``, ``node_color_by``,
            ``mark_births``, ``time_axis_label``).

    Returns:
        The :class:`plotly.graph_objects.Figure` produced by
        :func:`plotly_cell_lineage`.
    """
    segments = tracklet_graph_to_segments(tracklet_graph)
    time_feature = (
        "time" if any("time" in a for _, a in segments.nodes(data=True)) else "frame"
    )
    return plotly_cell_lineage(
        segments,
        time_feature=time_feature,
        **kwargs,
    )
