"""Local segmentation functionality dealing with files from HDD."""

import os
import os.path as osp
from urllib.parse import urlsplit

import cv2
import fsspec
import numpy as np
import roifile
import tifffile

from acia.base import (
    BaseImage,
    Contour,
    ImageSequenceSource,
    Instance,
    Overlay,
    RoISource,
)
from acia.config import resolve_storage_options
from acia.notebook import JupyterVisualizationMixin


def list_sequence_sources(
    folder: str,
    pattern: str = "*.tif",
    storage_options: dict | None = None,
    recursive: bool = False,
    **kwargs,
) -> list["LocalSequenceSource"]:
    """Discover image stacks in a folder and return one source per file.

    Works for a local directory or any fsspec URL (e.g. an ``smb://`` share),
    using the same credential resolution as the sources themselves: credentials
    for the listing are looked up by host in the acia config (see
    :mod:`acia.config`) and can be overridden via ``storage_options``.

    Args:
        folder (str): directory to scan. A plain local path or an fsspec URL,
            e.g. ``"/data/experiments"`` or ``"smb://fileserver.lab/data/exp"``.
        pattern (str): glob pattern for the file names. Defaults to ``"*.tif"``.
        storage_options (dict, optional): extra fsspec options (e.g. credentials)
            merged on top of the config entry for the folder's host.
        recursive (bool): if True, search sub-folders too (``**`` glob).
        **kwargs: forwarded to each :class:`LocalSequenceSource` (e.g.
            ``normalize_image``, ``luts``, ``channel_index``).

    Returns:
        list[LocalSequenceSource]: one source per matching file, sorted by path.
    """
    opts = resolve_storage_options(folder, storage_options)
    fs, root = fsspec.core.url_to_fs(folder, **opts)

    sep = "/**/" if recursive else "/"
    glob_path = root.rstrip("/") + sep + pattern
    matches = sorted(fs.glob(glob_path))

    is_local = "file" in fs.protocol or "local" in fs.protocol

    sources = []
    for match in matches:
        # keep plain paths for the local fs, full URLs (with host) for remotes so
        # per-file credential resolution still works at read time
        location = match if is_local else fs.unstrip_protocol(match)
        sources.append(
            LocalSequenceSource(location, storage_options=storage_options, **kwargs)
        )

    return sources


def prepare_image(image, normalize_image=True):
    """Normalize and convert image to RGB.

    Args:
        image ([type]): [description]
        normalize_image (bool, optional): Whether to normalize the image into uint8 domain (0-255). Defaults to True.
    Returns:
        [np.array]: RGB image (Width, height, 3 color channels)
    """
    # normalize image space
    if normalize_image:
        min_val = np.min(image)
        max_val = np.max(image)
        image = np.floor((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    if len(image.shape) == 2:
        # make it artificially rgb
        image = np.repeat(image[:, :, None], 3, axis=-1)

    return image


class LocalImage(BaseImage):
    """Class for a single image"""

    def __init__(self, content, frame=None):
        self.content = content
        self.frame = frame

    @property
    def raw(self):
        return self.content

    @property
    def num_channels(self):
        if len(self.raw.shape) == 2:
            # only width and height -> 1 channel
            return 1
        else:
            # multiple channels -> channels are specified at the end
            return self.raw.shape[-1]

    def get_channel(self, channel: int):
        assert channel < self.num_channels

        if self.num_channels == 1 and len(self.raw.shape) == 2:
            return self.raw
        else:
            return self.raw[..., channel]

    def __getitem__(self, item):
        return self.raw[item]


class LocalImageSource(ImageSequenceSource, JupyterVisualizationMixin):
    """Source for a single image only"""

    def __init__(self, image: LocalImage):
        self.image = image

    def __get_image(self):
        return self.image

    def __iter__(self):
        yield self.__get_image()

    def get_frame(self, frame: int):
        assert frame == 0, f"We only have a single frame, but frame={frame}"

        return self.__get_image()

    @property
    def num_channels(self) -> int:
        return int(self.__get_image().num_channels)

    @property
    def num_frames(self) -> int:
        return 1

    @property
    def size_t(self) -> int:
        return 1

    def __len__(self):
        return 1

    @staticmethod
    def from_file(file_path: str, normalize_image=True):
        image = LocalImage(prepare_image(cv2.imread(file_path), normalize_image))

        return LocalImageSource(image)

    @staticmethod
    def from_array(array):
        image = LocalImage(array)

        return LocalImageSource(image)


class InMemorySequenceSource(ImageSequenceSource, JupyterVisualizationMixin):
    """Image sequence for an in memory image stack"""

    def __init__(
        self, image_stack, frame_interval=None, timepoints=None, pixel_size=None
    ):
        self.image_stack = image_stack
        self._init_calibration(frame_interval, timepoints, pixel_size)

    def get_frame(self, frame: int) -> BaseImage:
        assert frame < len(self.image_stack)

        return LocalImage(self.image_stack[frame])

    def __len__(self):
        return len(self.image_stack)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_frame(i)

    @property
    def size_t(self) -> int:
        return len(self)

    @property
    def num_channels(self) -> int:
        return int(self.get_frame(0).num_channels)


class THWCSequenceSource(ImageSequenceSource, JupyterVisualizationMixin):
    """Image sequence for an in memory image stack [TxHxWxC]"""

    def __init__(
        self,
        image_stack: np.ndarray,
        frame_interval=None,
        timepoints=None,
        pixel_size=None,
    ):
        self.image_stack = image_stack
        self._init_calibration(frame_interval, timepoints, pixel_size)

        if len(self.image_stack.shape) != 4:
            raise ValueError(
                f"Please make sure to have TxHxWxC image stack. Currently it is: {self.image_stack.shape}"
            )

    def get_frame(self, frame: int) -> BaseImage:
        assert frame < len(self.image_stack)

        return LocalImage(self.image_stack[frame])

    def __len__(self):
        return len(self.image_stack)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_frame(i)

    @property
    def num_channels(self) -> int:
        return int(self.get_frame(0).num_channels)

    @property
    def size_c(self) -> int:
        """

        Returns:
            int: size of the C dimension
        """
        return int(self.image_stack.shape[3])

    @property
    def size_t(self) -> int:
        """

        Returns:
            int: size of the T dimension
        """
        return int(self.image_stack.shape[0])

    @property
    def size_h(self) -> int:
        """

        Returns:
            int: size of the C dimension
        """
        return int(self.image_stack.shape[1])

    @property
    def size_w(self) -> int:
        """

        Returns:
            int: size of the T dimension
        """
        return int(self.image_stack.shape[2])

    def to_channel(self, c: int) -> "THWCSequenceSource":
        """Converts multi-channel source into single-channel source

        Args:
            c (int): the channel to use

        Returns:
            THWCSequenceSource: sequence with the single channel
        """

        # select channel but make it TxHxWxC immediately
        return THWCSequenceSource(self.image_stack[..., c][..., None])

    def to_rgb(self) -> "THWCSequenceSource":
        """Convert image source into rgb space

        Raises:
            ValueError: if has wrong format

        Returns:
            InMemorySequenceSource:
        """

        if self.image_stack.shape[3] != 1:
            raise ValueError(
                f"Only works for single-channel sequences for now. You have C={self.num_channels}!"
            )

        def normalize(im: np.ndarray) -> np.ndarray:
            """Normalize image"""
            min_val = np.quantile(im, 0.01)
            max_val = np.quantile(im, 0.99)

            result: np.ndarray = (
                np.clip((im.astype(float) - min_val) / (max_val - min_val), 0.0, 1.0)
                * 255.0
            ).astype(np.uint8)
            return result

        # select the first channel
        image_stack = self.image_stack[..., 0]

        # apply normalization into unit8 space
        if self.image_stack.dtype != np.uint8:
            image_stack = normalize(image_stack)

        # repeat the channels to make a grayscale rendering
        return THWCSequenceSource(np.stack((image_stack,) * 3, axis=-1))


class LocalSequenceSource(ImageSequenceSource, JupyterVisualizationMixin):
    """Image sequence source for files in the local file system (e.g. a tif)."""

    def __init__(
        self,
        tif_file: str,
        normalize_image=True,
        luts=None,
        channel_index: int = 0,
        storage_options: dict | None = None,
        frame_interval=None,
        timepoints=None,
        pixel_size=None,
    ):
        """Create a new local image source

        Args:
            tif_file (str): path to the image file. May be a plain local path or
                any fsspec-supported URL (e.g. ``smb://``, ``s3://``, ``http://``).
            normalize_image (bool, optional): Normalizes the image pixels t0 [0, 255]. Defaults to True.
            luts: (List, optional): List of lut functions applied to the channels
            channel_index (int, optional): index in image of the channel. For example, for H,W,C dims where C is channel we should have a 2.
            storage_options (dict, optional): extra fsspec storage options (e.g.
                credentials) for remote URLs. Merged on top of any matching entry
                in the acia credentials config (see :mod:`acia.config`).
            frame_interval: scalar time between frames (pint Quantity or str like
                ``"15 minute"``) defining the imaging interval at load.
            timepoints: explicit per-frame timepoints (pint Quantity array).
            pixel_size: physical pixel size (pint length per pixel) for spatial
                calibration; extractors pull this for their units.
        """
        self.filename = tif_file
        self.normalize_image = normalize_image
        self.luts = luts
        self.channel_index = channel_index
        self.storage_options = storage_options
        self._init_calibration(frame_interval, timepoints, pixel_size)

    def _read_images(self):
        """Read the image stack via fsspec (works for local and remote URLs)."""
        opts = resolve_storage_options(self.filename, self.storage_options)
        with fsspec.open(self.filename, mode="rb", **opts) as f:
            return tifffile.imread(f)

    def __iter__(self):
        images = self._read_images()

        for image in images:
            if self.luts is not None:
                if len(image.shape) == 2:
                    # just a single channel
                    num_image_channels = 1
                else:
                    num_image_channels = image.shape[self.channel_index]

                assert len(self.luts) == num_image_channels, (
                    f"We need a LUTs function for every channel! We have {num_image_channels} channels but only {len(self.luts)} LUTs!"
                )
                # apply luts to image
                if len(image.shape) == 2:
                    # we only have one channel
                    image = self.luts[0](image)
                elif len(image.shape) == 3:
                    # we have N channels (at the front)
                    for channel in range(image.shape[self.channel_index]):
                        image[channel] = self.luts[channel](
                            image.take(channel, axis=self.channel_index)
                        )

            image = prepare_image(image, self.normalize_image)

            yield LocalImage(image)

    def get_frame(self, frame: int) -> BaseImage:
        # TODO: this is super slow access for indiviudal images
        images = self._read_images()
        assert frame < len(images)

        return LocalImage(prepare_image(images[frame]))

    @property
    def size_t(self):
        return len(self._read_images())

    @property
    def num_channels(self) -> int:
        return int(self.get_frame(0).num_channels)

    def slice(self, start, end):
        """Return a view over frames [start:end).

        Kept for backward compatibility; equivalent to ``self[start:end]``.
        """
        return self[start:end]


class SambaSequenceSource(LocalSequenceSource):
    """Image sequence source for TIFFs on an SMB/SAMBA share (via fsspec).

    This is a thin convenience wrapper around :class:`LocalSequenceSource`: it
    builds the ``smb://`` URL and forwards credentials as fsspec storage options.
    All reading, iteration and visualization logic is inherited.

    Credentials are optional. If ``username``/``password`` (etc.) are omitted,
    they are looked up in the acia credentials config by host (see
    :mod:`acia.config`), so regular usage needs no secrets in code. Any value
    passed explicitly here overrides the config.
    """

    def __init__(
        self,
        host: str,
        share: str,
        path: str,
        username: str | None = None,
        password: str | None = None,
        domain: str | None = None,
        port: int | None = None,
        **kwargs,
    ):
        """Create a new SAMBA/SMB image source.

        Args:
            host (str): SMB server host name or IP.
            share (str): name of the share.
            path (str): path to the image file within the share.
            username (str, optional): SMB username. Defaults to config lookup.
            password (str, optional): SMB password. Defaults to config lookup.
            domain (str, optional): SMB/Windows domain. Defaults to config lookup.
            port (int, optional): SMB port. Defaults to the backend default.
            **kwargs: forwarded to :class:`LocalSequenceSource` (e.g.
                ``normalize_image``, ``luts``, ``channel_index``).
        """
        url = f"smb://{host}/{share}/{path.lstrip('/')}"

        explicit = {
            "host": host,
            "username": username,
            "password": password,
            "domain": domain,
            "port": port,
        }
        # drop unset values so the config / backend defaults take over
        explicit = {k: v for k, v in explicit.items() if v is not None}

        super().__init__(url, storage_options=explicit, **kwargs)

    @classmethod
    def from_url(
        cls,
        url: str,
        username: str | None = None,
        password: str | None = None,
        domain: str | None = None,
        port: int | None = None,
        **kwargs,
    ) -> "SambaSequenceSource":
        """Create a SambaSequenceSource from a full ``smb://`` URL.

        The URL is split into host / share / path, e.g.
        ``smb://fileserver.lab/data/exp/img.tif`` -> host ``fileserver.lab``,
        share ``data``, path ``exp/img.tif``. Credentials may be embedded in the
        URL (``smb://user:pass@host/share/...``) or passed as keyword arguments;
        anything omitted is resolved from the acia credentials config.

        Args:
            url (str): an ``smb://`` URL pointing at the image file.
            username/password/domain/port: optional credential overrides.
            **kwargs: forwarded to :class:`LocalSequenceSource`.
        """
        parts = urlsplit(url)
        if parts.scheme != "smb" or not parts.hostname:
            raise ValueError(f"Expected an 'smb://host/...' URL, got {url!r}.")

        share, _, path = parts.path.lstrip("/").partition("/")
        if not share or not path:
            raise ValueError(
                f"smb URL must contain a share and a file path, e.g. "
                f"'smb://host/share/path/img.tif'. Got {url!r}."
            )

        return cls(
            host=parts.hostname,
            share=share,
            path=path,
            username=username or parts.username,
            password=password or parts.password,
            domain=domain,
            port=port or parts.port,
            **kwargs,
        )


class ImageJRoISource(RoISource):
    """Source fro ImageJ RoI file format"""

    def __init__(self, filename, range=None):
        self.overlay = RoiStorer.load(filename)
        self.range = range

    def __iter__(self):
        return self.overlay.timeIterator(frame_range=self.range)

    def __len__(self) -> int:
        if self.range:
            min(len(self.overlay), len(self.range))
        return len(self.overlay)


class RoiStorer:
    """
    Stores and loads overlay results in the roi format (readable by ImageJ)
    """

    @staticmethod
    def store(overlay: Overlay, filename: str, append=False):
        """
        Stores overlay results in the roi format (readable by fiji)

        overlay: the overlay to store
        filename: filename of the roi collection (e.g. rois.zip)
        append: appends the rois if the file already exists
        """

        # generate imagej rois from the overlay
        rois = [
            roifile.ImagejRoi.frompoints(contour.coordinates, t=contour.frame)
            for contour in overlay
        ]

        if not append and osp.isfile(filename):
            os.remove(filename)

        # write them to file
        roifile.roiwrite(filename, rois)

    @staticmethod
    def load(filename: str):
        # read the imagej rois from file
        rois = roifile.roiread(filename)

        # Ensure rois is a list
        if not isinstance(rois, list):
            rois = [rois]

        roi_id = -1
        # convert them into contours (recover time position)
        contours: list[Contour | Instance] = [
            Contour(np.array(roi.coordinates()), -1.0, roi.position - 1, id=roi_id)
            for roi in rois
        ]

        # return the overlay
        return Overlay(contours)
