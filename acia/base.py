"""All basic functionality for acia"""

from __future__ import annotations

import contextlib
import copy
import logging
import multiprocessing
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from functools import partial
from typing import Any

import cv2
import numpy as np
import tqdm
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from tqdm.contrib.concurrent import process_map

from acia.notebook import JupyterVisualizationMixin

from .utils import mask_to_polygons, polygon_to_mask


def unpack(data, function):
    return function(*data)


class Instance:
    """Cell instance based on an image mask and a label"""

    def __init__(
        self,
        mask: np.ndarray,
        frame: int,
        label: int,
        id=None,
        score: float | None = None,
    ):
        """Create an object instance

        Args:
            mask (np.ndarray): mask of the object where the object pixels are marked with [label] value
            frame (int): frame in the time-lapse
            label (int): label of the object (as marked in the mask)
            id (_type_, optional): Unique identifier for the object. Defaults to None.
            score (float, optional): E.g. confidence of the detection method. Defaults to None.
        """
        self.mask = mask
        self.frame = frame
        self.label = label
        self.id = id  # id is unique in an overlay
        self.score = score
        self.time = None  # pint timestamp, set when the overlay carries a time model

        self._polygon = None

    @property
    def binary_mask(self):
        return self.mask == self.label

    @property
    def center(self):
        # compute (x,y) center on pixel level

        bin_mask = self.binary_mask

        x = np.median(np.nonzero(np.max(bin_mask, axis=0)))
        y = np.median(np.nonzero(np.max(bin_mask, axis=1)))

        return (x, y)

    @property
    def area(self) -> float:
        """Compute the area inside the contour

        Returns:
            [float]: area
        """
        return float(np.sum(self.binary_mask))

    def toMask(self, height, width):
        """
        Render contour mask onto new image

        height: height of the image
        width: width of the image
        """
        bin_mask = self.binary_mask
        m_height, m_width = bin_mask.shape
        if m_height != height:
            logging.warning("Mask height %d != requested height %d!", m_height, height)
        if m_width != width:
            logging.warning("Mask width %s != requested width %s!", m_width, width)

        return bin_mask

    @property
    def polygon(self) -> Polygon | MultiPolygon | None:
        if self._polygon is None:
            self._polygon = mask_to_polygons(self.binary_mask)
        return self._polygon

    @property
    def coordinates(self) -> np.ndarray:
        """Extract contour coordinates

        Raises:
            ValueError: if the polygon is not valid or None

        Returns:
            np.ndarray: Nx2 contour coordinates of the polygon
        """
        poly = self.polygon
        if poly is None:
            raise ValueError("Polygon is None (empty mask).")

        if not poly.is_valid:
            raise ValueError("Invalid Shapely polygon.")

        # polygon.exterior.coords returns a coordinate sequence with first==last (closed ring)
        coords = np.array(
            poly.exterior.coords[:-1]
        )  # remove duplicate last point if needed
        return coords

    def draw(self, image, draw=None, outlineColor=(255, 255, 0), fillColor=None):
        """Draws instance onto an image

        Args:
            image (np.array | PIL.Image): the image to draw onto
            draw (PIL.ImageDraw, optional): Drawing Tool. Defaults to None.
            outlineColor (tuple, optional): Color of the Instance contour. None means no contour is drawn. Defaults to (255, 255, 0).
            fillColor (tuple, optional): Color of the contour fill. Defaults to None (no filling).

        Returns:
            np.array | PIL.Image: The image containing the drawn contour.
        """
        # TODO: make this more efficient
        if draw is None:
            draw = ImageDraw.Draw(image)

        def get_largest(poly):
            if isinstance(poly, MultiPolygon):
                return poly.geoms[np.argmax([p.area for p in poly.geoms])]
            else:
                return poly

        # get the contour coordinates
        coords = np.stack(get_largest(self.polygon).exterior.coords, axis=0).astype(int)
        # draw the polygon
        draw.polygon(tuple(coords.flatten()), outline=outlineColor, fill=fillColor)


class Contour:
    """Class for object contour detection (e.g. Cell object)"""

    def __init__(
        self, coordinates: np.ndarray, score: float, frame: int, id, label=None
    ):
        """Create Contour

        Args:
            coordinates (np.ndarray): coordinates in (x,y) list
            score (float): segmentation score
            frame (int): frame index
            id (any): unique id
            label: class-defining label of the contour
        """
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.score = score
        self.frame = frame
        self.id = id
        self.label = label
        self.time = None  # pint timestamp, set when the overlay carries a time model

    def _toMask(self, height: int, width: int) -> np.ndarray:
        """
        Render contour mask onto existing image

        img: pillow image
        fillValue: mask values inside the contour
        outlineValues: mask values on the outline (border)
        """
        # perform rasterization into mask
        result: np.ndarray = polygon_to_mask(self.polygon, height, width)
        return result

    def toMask(self, height, width):
        """
        Render contour mask onto new image

        height: height of the image
        width: width of the image
        """
        return self._toMask(height=height, width=width)

    def draw(self, image, draw=None, outlineColor=(255, 255, 0), fillColor=None):
        is_numpy = isinstance(image, np.ndarray)

        # Deal with numpy or PIL.Image
        if is_numpy:
            # convert into rgb PIL image
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)

        if draw is None:
            draw = ImageDraw.Draw(image)

        draw.polygon(
            list(map(tuple, self.coordinates)), outline=outlineColor, fill=fillColor
        )

        if is_numpy:
            # return the numpy version
            return np.asarray(image)
        else:
            # return the PIL image
            return image

    def scale(self, scale: float):
        """Apply scale factor to contour coordinates

        Args:
            scale (float): the multplication factor
        """
        self.coordinates *= scale

    @property
    def center(self):
        return np.array(Polygon(self.coordinates).centroid.coords[0], dtype=np.float32)

    @property
    def area(self) -> float:
        """Compute the area inside the contour

        Returns:
            [float]: area
        """
        return float(self.polygon.area)

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.coordinates)

    def __repr__(self) -> str:
        return str(self.id)


class Overlay:
    """Overlay contains Contours at different frames and provides functionalities iterate and modify them"""

    def __init__(
        self,
        contours: Sequence[Contour | Instance],
        frames=None,
        timepoints=None,
        frame_interval=None,
    ):
        self.contours: list[Contour | Instance] = list(contours)
        if frames is not None:
            frames = sorted(list(frames))
        self.__frames = frames

        self.cont_lookup: dict[Any, Contour | Instance] = {
            cont.id: cont for cont in self.contours
        }

        self._timepoints = None
        self._frame_interval = None
        if timepoints is not None or frame_interval is not None:
            self._set_time(timepoints=timepoints, frame_interval=frame_interval)

    def add_contour(self, contour: Contour | Instance):
        self.contours.append(contour)
        self.cont_lookup[contour.id] = contour

    def add_contours(self, contours: Sequence[Contour | Instance]):
        for cont in contours:
            self.add_contour(cont)

    def __getitem__(self, key):
        """``overlay[id]`` -> contour by id; ``overlay[slice|list]`` -> temporal
        sub-overlay (frames remapped to ``0..n-1``, time model carried)."""
        if isinstance(key, slice | list | range | np.ndarray):
            return self._slice_frames(key)
        return self.cont_lookup[key]

    def _slice_frames(self, key) -> Overlay:
        all_frames = sorted(self.frames())
        if isinstance(key, slice):
            selected = list(np.array(all_frames)[key])
        else:
            selected = [all_frames[int(i)] for i in key]

        selected_set = set(selected)
        frame_map = {old: new for new, old in enumerate(selected)}

        new_contours = []
        for cont in self.contours:
            if cont.frame in selected_set:
                new_cont = copy.deepcopy(cont)
                new_cont.frame = frame_map[cont.frame]
                new_contours.append(new_cont)

        sub = Overlay(new_contours, frames=list(range(len(selected))))
        tp = self.timepoints
        if tp is not None and len(selected) > 0:
            sub = sub.with_timepoints(tp[selected])
        return sub

    def __iter__(self):
        return iter(self.contours)

    def __add__(self, other):
        jointContours = self.contours + other.contours
        return Overlay(jointContours)

    def __len__(self):
        return len(self.contours)

    def numFrames(self):
        return len(self.frames())

    def frames(self):
        if self.__frames:
            return self.__frames
        else:
            return np.unique([c.frame for c in self.contours])

    # --- time model (pint), so detections carry timestamps ---

    def _frame_extent(self) -> int:
        fr = self.frames()
        return int(np.max(fr)) + 1 if len(fr) else 0

    @property
    def timepoints(self):
        """Per-frame pint ``Quantity`` of timepoints, or ``None`` if uncalibrated."""
        from acia.timing import resolve_timepoints

        if self._timepoints is not None:
            return self._timepoints
        if self._frame_interval is not None:
            return resolve_timepoints(
                self._frame_extent(), frame_interval=self._frame_interval
            )
        return None

    @property
    def timestamps(self):
        """Pint ``Quantity`` of per-contour timestamps (in ``contours`` order)."""
        tp = self.timepoints
        if tp is None:
            return None
        return tp[[c.frame for c in self.contours]]

    def _set_time(self, *, timepoints=None, frame_interval=None) -> Overlay:
        from acia.timing import to_quantity

        self._timepoints = timepoints
        self._frame_interval = to_quantity(frame_interval)
        tp = self.timepoints
        if tp is not None:
            for cont in self.contours:
                if 0 <= cont.frame < len(tp):
                    cont.time = tp[cont.frame]
        return self

    def with_timepoints(self, timepoints) -> Overlay:
        """Attach explicit per-frame timepoints (pint) and stamp each detection."""
        return self._set_time(timepoints=timepoints)

    def with_frame_interval(self, interval) -> Overlay:
        """Attach a scalar frame interval (pint) and stamp each detection."""
        return self._set_time(frame_interval=interval)

    def scale(self, scale: float):
        """Scale the contour with the specified scale factor

           Applies the scale factor to all coordinates individually

        Args:
            scale (float): [description]
        """
        for cont in self.contours:
            if isinstance(cont, Contour):
                cont.scale(scale)

    def croppedContours(self, cropping_parameters: tuple[slice, slice]):
        y, x = cropping_parameters
        miny, maxy, minx, maxx = y.start, y.stop, x.start, x.stop

        crop_rectangle = Polygon(
            [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        )

        def __crop_function_filter(contour: Contour | Instance) -> bool:
            if not isinstance(contour, Contour):
                return False  # Instance doesn't have coordinates attribute
            try:
                return bool(crop_rectangle.contains(Polygon(contour.coordinates)))
            # TODO: more precise exception catching here!
            # pylint: disable=W0703
            except Exception:
                # if we have problems to convert to shapely polygon, we cannot include it
                logging.warning(
                    "Have to drop Polygon: It cannot be converted into a shapely Polygon."
                )
                return False

        for cont in filter(__crop_function_filter, self.contours):
            if isinstance(cont, Contour):
                new_cont = copy.deepcopy(cont)
                new_cont.coordinates -= np.array([minx, miny])
                yield new_cont

    def time_iterator(
        self, start_frame=None, end_frame=None, frame_range=None
    ) -> Iterable[Overlay]:
        return self.timeIterator(
            startFrame=start_frame, endFrame=end_frame, frame_range=frame_range
        )

    def timeIterator(
        self, startFrame=None, endFrame=None, frame_range=None
    ) -> Iterable[Overlay]:
        """
        Creates an iterator that returns an Overlay for every frame between starFrame and endFrame

        startFrame: first frame number
        endFrame: last frame number
        """
        if len(self.frames()) == 0:
            yield Overlay([])

        if startFrame is None:
            startFrame = np.min(self.frames())

        if endFrame is None:
            endFrame = np.max(self.frames())

        assert startFrame >= 0
        assert endFrame >= 0
        assert endFrame <= np.max(self.frames())

        it_frames: Iterable[int] = range(startFrame, endFrame + 1)

        if self.__frames:
            it_frames = sorted(self.__frames)

        # frame for every contour
        frame_information = np.array(
            list(map(lambda cont: cont.frame, self.contours)), dtype=np.int64
        )
        # numpy array of contours (dtype=np.object)
        contour_array = np.array(self.contours)

        # iterate frames
        for frame in it_frames:
            if frame_range and frame not in frame_range:
                continue

            # mask for contour array for this frame
            cont_mask = frame_information == frame
            # filter sub overlay with all contours in the current frame
            yield Overlay(list(contour_array[cont_mask]))

    def toMasks(self, height, width, binary_mask=True) -> list[np.ndarray]:
        """
        Turn the individual overlays into masks. For every time point we create a mask of all contours.

        returns: List of masks (np.ndarray[bool])

        height: height of the image
        width: width of the image
        """
        masks = []
        for timeOverlay in self.timeIterator():
            if binary_mask:
                local_mask = np.zeros((height, width), dtype=bool)
            else:
                # non-binary
                local_mask = np.zeros((height, width), dtype=np.uint16)

            # combine all contours in one mask
            for i, cont in enumerate(timeOverlay):
                mask = cont.toMask(height=height, width=width)
                if not binary_mask:
                    label = i + 1
                    if cont.label is not None:
                        with contextlib.suppress(ValueError):
                            label = int(cont.label)

                    mask = mask.astype(np.uint16) * (
                        label
                    )  # convert into a non-binary mask

                # combine into a single mask
                local_mask = np.maximum(mask, local_mask)

            # append frame mask to list of masks
            masks.append(local_mask)

        return masks

    def draw(
        self,
        image: np.ndarray | Image.Image,
        outlineColor: str
        | tuple[int, ...]
        | Callable[[Contour | Instance], tuple[int, ...]]
        | None = None,
        fillColor: str
        | tuple[int, ...]
        | Callable[[Contour | Instance], tuple[int, ...]]
        | None = None,
    ) -> np.ndarray | Image.Image:
        """Draw an overly onto an image frame. Hint: overlay should only contain contours for a single frame

        Args:
            image (np.ndarray | Image): Image to draw onto
            outlineColor (str | Callable[[Contour], tuple[int]], optional): Color of the object outlines. If this is a function, the function computes the color for every contour/instance individually. Defaults to None (no contour is drawn).
            fillColor (str | Callable[[Contour], tuple[int]], optional): Fill color of the object. If this is a function, the function computes the color for every contour/instance individually. Defaults to None (no fill). Defaults to None.

        Returns:
            np.ndarray | Image: the updated image object
        """

        if self.numFrames() > 1:
            logging.warning(
                "Drawing overlay onto a frame while the overlay contains instances from multiple frames!"
            )

        is_numpy = isinstance(image, np.ndarray)
        pil_image: Image.Image

        # Deal with numpy or PIL.Image
        if is_numpy:
            assert isinstance(image, np.ndarray)
            # convert into rgb PIL image
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(image)
        else:
            assert isinstance(image, Image.Image)
            pil_image = image

        imdraw = ImageDraw.Draw(pil_image)
        for timeOverlay in self.timeIterator():
            for cont in timeOverlay:
                oc_local: str | tuple[int, ...] | None = None
                fc_local: str | tuple[int, ...] | None = None

                # compute the contour color for the object
                if outlineColor is not None:
                    if callable(outlineColor):
                        oc_local = outlineColor(cont)
                    else:
                        oc_local = outlineColor
                # compute the fill color for the object
                if fillColor is not None:
                    if callable(fillColor):
                        fc_local = fillColor(cont)
                    else:
                        fc_local = fillColor

                cont.draw(
                    pil_image, outlineColor=oc_local, fillColor=fc_local, draw=imdraw
                )

        if is_numpy:
            # return the numpy version
            return np.asarray(pil_image)
        else:
            # return the PIL image
            return pil_image


class BaseImage:
    """Base class for an image from an image source"""

    @property
    def raw(self):
        raise NotImplementedError("Please implement this function!")

    @property
    def num_channels(self):
        raise NotImplementedError()

    def get_channel(self, channel: int):
        raise NotImplementedError()


class ArrayImage(BaseImage):
    """A `BaseImage` backed directly by a numpy array (e.g. a cropped frame)."""

    def __init__(self, content: np.ndarray, frame: int | None = None):
        self.content = content
        self.frame = frame

    @property
    def raw(self):
        return self.content

    @property
    def num_channels(self):
        if len(self.raw.shape) == 2:
            return 1
        return self.raw.shape[-1]

    def get_channel(self, channel: int):
        assert channel < self.num_channels
        if self.num_channels == 1 and len(self.raw.shape) == 2:
            return self.raw
        return self.raw[..., channel]

    def __getitem__(self, item):
        return self.raw[item]


class Processor:
    """Base class for a processor"""


class ImageSequenceSource(Iterable[BaseImage], Sized):
    """Base class for an image sequence source (e.g. Tiff, OMERO, png, ...).

    Supports numpy-style indexing over the (T, H, W, C) axes:

    * ``src[5]`` -> the frame at index 5 (a :class:`BaseImage`)
    * ``src[::2]`` -> a view sequence of every second frame
    * ``src[3:23, 10:90, 10:90, 0]`` -> a cropped, single-channel subsequence
    """

    @property
    def num_channels(self) -> int:
        raise NotImplementedError()

    @property
    def size_t(self) -> int:
        raise NotImplementedError()

    @property
    def size_h(self) -> int:
        raise NotImplementedError()

    @property
    def size_w(self) -> int:
        raise NotImplementedError()

    @property
    def size_c(self) -> int:
        raise NotImplementedError()

    def get_frame(self, frame: int) -> BaseImage:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[BaseImage]:
        for i in range(self.size_t):
            yield self.get_frame(i)

    def __len__(self) -> int:
        return self.size_t

    def __getitem__(self, key):
        """numpy-style indexing over (T, H, W, C). See class docstring."""
        t_key, spatial = self._split_index(key)

        if isinstance(t_key, int | np.integer):
            idx = self._resolve_t_index(int(t_key))
            frame = self.get_frame(idx)
            if spatial:
                return ArrayImage(frame.raw[spatial], frame=idx)
            return frame

        if isinstance(t_key, slice):
            t_indices = list(range(*t_key.indices(self.size_t)))
        else:
            # fancy indexing: a list/array of frame indices
            t_indices = [self._resolve_t_index(int(i)) for i in t_key]

        return SlicedSequenceSource(self, t_indices, spatial)

    def _resolve_t_index(self, idx: int) -> int:
        n = self.size_t
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f"frame index {idx} out of range for size_t={n}")
        return idx

    # --- physical calibration (time + space), all optional and in pint units ---

    @property
    def timepoints(self):
        """Per-frame pint ``Quantity`` of timepoints, or ``None`` if uncalibrated."""
        from acia.timing import resolve_timepoints

        return resolve_timepoints(
            self.size_t,
            timepoints=getattr(self, "_timepoints_raw", None),
            frame_interval=getattr(self, "_frame_interval", None),
        )

    @property
    def pixel_size(self):
        """Pint length per pixel (scalar or ``[y, x]``), or ``None``."""
        return getattr(self, "_pixel_size", None)

    def with_frame_interval(self, interval):
        """Tag this source with a scalar frame interval (pint); returns self."""
        from acia.timing import to_quantity

        self._frame_interval = to_quantity(interval)
        self._timepoints_raw = None
        return self

    def with_timepoints(self, timepoints):
        """Tag this source with explicit per-frame timepoints (pint); returns self."""
        self._timepoints_raw = timepoints
        self._frame_interval = None
        return self

    def with_pixel_size(self, pixel_size):
        """Tag this source with a pixel size (pint length per pixel); returns self."""
        from acia.timing import to_quantity

        self._pixel_size = to_quantity(pixel_size)
        return self

    def _init_calibration(self, frame_interval=None, timepoints=None, pixel_size=None):
        """Store load-time calibration (called from concrete source constructors)."""
        from acia.timing import to_quantity

        self._frame_interval = to_quantity(frame_interval)
        self._timepoints_raw = timepoints
        self._pixel_size = to_quantity(pixel_size)

    def _split_index(self, key) -> tuple[Any, tuple]:
        """Split an index into the temporal key and the trailing spatial key.

        Expands a single ``Ellipsis`` against the frame dimensionality so that
        e.g. ``src[..., 0]`` selects channel 0 across all frames.
        """
        if not isinstance(key, tuple):
            return key, ()

        key_list = list(key)
        if Ellipsis in key_list:
            frame_ndim = self.get_frame(0).raw.ndim
            total = 1 + frame_ndim  # T + frame axes
            n_explicit = len(key_list) - 1  # all entries except the Ellipsis
            fill = max(total - n_explicit, 0)
            i = key_list.index(Ellipsis)
            key_list = key_list[:i] + [slice(None)] * fill + key_list[i + 1 :]

        return key_list[0], tuple(key_list[1:])


class SlicedSequenceSource(ImageSequenceSource, JupyterVisualizationMixin):
    """A lazy view over a parent sequence selecting frames and cropping each.

    Holds the parent source, a list of original frame indices and a trailing
    spatial/channel key applied to every frame's array. Re-slicing nests another
    view, so composition is automatic.
    """

    def __init__(
        self,
        parent: ImageSequenceSource,
        t_indices: Sequence[int],
        spatial: tuple = (),
    ):
        self.parent = parent
        self.t_indices = list(t_indices)
        self.spatial = spatial
        self._shape: tuple | None = None

    def get_frame(self, frame: int) -> BaseImage:
        idx = self.t_indices[frame]
        frame_obj = self.parent.get_frame(idx)
        if self.spatial:
            return ArrayImage(frame_obj.raw[self.spatial], frame=frame)
        return frame_obj

    @property
    def size_t(self) -> int:
        return len(self.t_indices)

    def _frame_shape(self) -> tuple:
        if self._shape is None:
            self._shape = tuple(self.get_frame(0).raw.shape)
        return self._shape

    @property
    def size_h(self) -> int:
        return int(self._frame_shape()[0])

    @property
    def size_w(self) -> int:
        return int(self._frame_shape()[1])

    @property
    def size_c(self) -> int:
        shape = self._frame_shape()
        return int(shape[2]) if len(shape) > 2 else 1

    @property
    def num_channels(self) -> int:
        return int(self.get_frame(0).num_channels)

    def _axis_step(self, axis: int) -> int:
        """Step of the spatial indexer for a frame axis (0=H, 1=W); 1 if none."""
        if axis < len(self.spatial) and isinstance(self.spatial[axis], slice):
            return self.spatial[axis].step or 1
        return 1

    @property
    def timepoints(self):
        # an explicit calibration set on the view itself wins
        if (
            getattr(self, "_timepoints_raw", None) is not None
            or getattr(self, "_frame_interval", None) is not None
        ):
            return super().timepoints
        parent_tp = self.parent.timepoints
        if parent_tp is None:
            return None
        return parent_tp[self.t_indices]

    @property
    def pixel_size(self):
        own = getattr(self, "_pixel_size", None)
        if own is not None:
            return own
        base = self.parent.pixel_size
        if base is None:
            return None
        h_step, w_step = self._axis_step(0), self._axis_step(1)
        if h_step == 1 and w_step == 1:
            return base  # crop keeps the pixel size
        if h_step == w_step:
            return base * h_step
        # anisotropic after differing spatial steps -> [y, x]
        return base * np.array([h_step, w_step])


class RoISource(Iterable[Overlay], Sized):
    """Base class for a RoI source (e.g. tiff metadata, OMERO, json, ...)"""

    def __iter__(self) -> Iterator[Overlay]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


class ImageRoISource:
    """
    Contains both, the image and the RoI Source. Provides a joint iterator
    """

    def __init__(self, imageSource: ImageSequenceSource, roiSource: RoISource):
        self.imageSource = imageSource
        self.roiSource = roiSource

    def __iter__(self) -> Iterator[tuple[BaseImage, Overlay]]:
        return zip(iter(self.imageSource), iter(self.roiSource), strict=False)  # type: ignore[return-value]

    def __len__(self):
        return min(len(self.imageSource), len(self.roiSource))

    def apply_parallel(self, function, num_workers=None):
        if num_workers is None:
            num_workers = int(np.floor(multiprocessing.cpu_count() * 2 / 3))

        return process_map(function, self, max_workers=num_workers, chunksize=4)

    def apply_parallel_star(self, function, num_workers=None):
        if num_workers is None:
            num_workers = int(np.floor(multiprocessing.cpu_count() * 2 / 3))

        return process_map(
            partial(unpack, function=function),
            self,
            max_workers=num_workers,
            chunksize=4,
        )

    def apply(self, function):
        def limit():
            for _, el in enumerate(self):
                yield el

        return list(tqdm.tqdm(map(function, limit())))

    def apply_star(self, function):
        return list(tqdm.tqdm(map(partial(unpack, function=function), self)))
