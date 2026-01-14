"""Classes for OMERO storage interaction"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import omero
import tqdm.auto as tqdm
from omero.gateway import BlitzGateway
from omero.model import LengthI

from acia import ureg
from acia.base import BaseImage, Contour, ImageSequenceSource, Overlay, RoISource
from acia.notebook import JupyterVisualizationMixin
from acia.segm.local import LocalImage, THWCSequenceSource
from acia.segm.omero.shapeUtils import make_coordinates
from acia.segm.utils import compute_indices

from .shapeUtils import create_polygon


# We have a helper function for creating an ROI and linking it to new shapes
def create_roi(updateService, img, shapes):
    """
    Helper function to create the roi object
    updateService: omero update Service
    img: omero image object (not the id)
    shapes: list of omero.model shapes

    returns: RoI object
    """
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi)


# We have a helper function for creating an ROI and linking it to new shapes
def create_roi_fast(updateService, img, shapes):
    """
    Helper function to create the roi object (without waiting)
    updateService: omero update Service
    img: omero image object (not the id)
    shapes: list of omero.model shapes
    """
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    updateService.saveObject(roi)


class OmeroRoIStorer:
    """
    Stores and loads overlay results in the roi format (readable by ImageJ)
    """

    @staticmethod
    def storeWithConn(overlay, imageId: int, conn, force=False):
        # retrieve omero objects
        updateService = conn.getUpdateService()
        image = conn.getObject("Image", imageId)

        userId = conn.getUser().getId()
        imageOwnerId = image.getOwner().getId()

        if not force and userId != imageOwnerId:
            raise ValueError(
                "You try to write to non-owned data. Enable 'force' option if you are sure to do that."
            )

        # OmeroRoIStorer.clear(imageId=imageId, username=username, password=password, serverUrl=serverUrl, port=port, secure=secure)

        # size_t = image.getSizeT()
        size_z = image.getSizeZ()

        # this is a linearized overlay
        logging.info(
            "Using Linearized overlay: [t_1: z_0, z_1, ... z_Z, t_2: ,...] Use t and z"
        )
        shapes = [
            create_polygon(
                cont.coordinates,
                z=cont.frame % size_z,
                t=np.floor(cont.frame / size_z),
                description=f"Score: {cont.score:.2f}",
            )
            for cont in overlay
        ]

        for shape in tqdm.tqdm(shapes):
            create_roi_fast(updateService, image, [shape])

        logging.info(
            "Stored overlay with %d rois for image '%s'", len(overlay), image.getName()
        )

    @staticmethod
    def store(
        overlay: Overlay,
        imageId: int,
        username: str,
        password: str,
        serverUrl: str,
        port=4064,
        secure=True,
        force=False,
        conn=None,
    ):
        """
        Stores overlay results in OMERO. Uses existing connection if available.

        overlay: the overlay to store
        imageId: omero id of the image sequence
        username: omero username
        password: omero password
        serverUrl: omero web address
        port: omero port (default: 4064)
        conn: existing OMERO connection or None. (default: None)
        """

        with BlitzConn(username, password, serverUrl, port, secure, conn) as omero_conn:
            OmeroRoIStorer.storeWithConn(overlay, imageId, omero_conn, force)

    @staticmethod
    def load(
        imageId: int,
        username: str,
        password: str,
        serverUrl: str,
        port=4064,
        secure=True,
        roiId=None,
        conn=None,
    ) -> Overlay:
        """
        Loads overlay from omero. Only considers polygons.

        imageId: omero id of the image sequence
        username: omero username
        password: omero password
        serverUrl: omero web address
        port: omero port (default: 4064)
        """
        overlay = Overlay([])
        # open connection to omero
        # with BlitzGateway(
        #    username, password, host=serverUrl, port=port, secure=secure
        # ) as conn:
        with BlitzConn(username, password, serverUrl, port, secure, conn) as omero_conn:
            # get the roi service
            roi_service = omero_conn.getRoiService()
            result = roi_service.findByImage(imageId, None)

            image = omero_conn.getObject("Image", imageId)

            # size_t = image.getSizeT()
            size_z = image.getSizeZ()

            # loop rois
            for roi in result.rois:
                if (roiId is not None) and roi.getId() != roiId:
                    # if the roiId is specified, check whether we have the right one
                    continue

                # loop shapes inside roi
                for s in roi.copyShapes():
                    if isinstance(s, omero.model.PolygonI):
                        # extract important information
                        t = s.getTheT().getValue()
                        points = np.array(make_coordinates(s.getPoints().getValue()))
                        score = -1.0

                        if size_z > 1:
                            t = t * size_z + s.getTheZ().getValue()

                        id = s.getId().getValue()

                        label = s.getTextValue().getValue()

                        # add contour element to overlay
                        cont = Contour(points, score, t, id=id, label=label)
                        overlay.add_contour(cont)

        # return the overlay
        return overlay

    @staticmethod
    def clear(
        imageId: int,
        username: str | None = None,
        password: str | None = None,
        serverUrl: str | None = None,
        port=4064,
        secure=True,
        conn=None,
    ):
        # open connection to omero
        with BlitzConn(
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
        ) as omero_conn:
            # get the roi service
            roi_service = omero_conn.getRoiService()
            updateService = omero_conn.getUpdateService()
            result = roi_service.findByImage(imageId, None)

            print(f"Deleting {len(result.rois)} rois...")

            for roi in result.rois:
                shapes = roi.copyShapes()
                if len(shapes) > 1:
                    for s in roi.copyShapes():
                        roi.removeShape(s)
                    roi = updateService.saveAndReturnObject(roi)

            # delete all RoIs in the image
            if len(result.rois) > 0:
                omero_conn.deleteObjects(
                    "Roi",
                    [roi.getId().getValue() for roi in result.rois],
                    deleteAnns=True,
                    deleteChildren=True,
                    wait=True,
                )


class IngoreWithWrapper:
    """Wrapper to ignore context and do not exit after leaving. Is used for not starting/ending omero connections all the time!"""

    def __init__(self, object):
        self.object = object

    def __getattr__(self, attr):
        return getattr(self.object, attr)

    def __enter__(self):
        return self.object

    def __exit__(self, type, value, traceback):
        pass


class BlitzConn:
    """
    Encapsulates standard omero behavior
    """

    def __init__(
        self,
        username,
        password,
        serverUrl,
        port=4064,
        secure=True,
        conn=None,
        readonly=True,
    ):
        assert username is not None or conn is not None, "Please provide a username"
        assert password is not None or conn is not None, "Please provide a password"
        assert serverUrl is not None or conn is not None, (
            "Please provide a OMERO server"
        )

        self.username = username
        self.password = password
        self.serverUrl = serverUrl
        self.port = port
        self.secure = secure

        self.conn = conn
        self.readonly = readonly

    def make_connection(self):
        # try to keep connection alive
        if self.conn is not None and self.conn.keepAlive():
            return IngoreWithWrapper(self.conn)
        else:
            # return a new connection
            conn = BlitzGateway(
                self.username,
                self.password,
                host=self.serverUrl,
                port=self.port,
                secure=self.secure,
            )

            # establish connection
            connect_res = conn.connect()

            # check connection
            if connect_res is False:
                raise ConnectionError(
                    "Connection to OMERO failed! Please check your OMERO settings (username, password, omero host, ...)!"
                )

            # if readonly -> set group so that we can immediately access all data
            if self.readonly:
                conn.SERVICE_OPTS.setOmeroGroup("-1")
            # store the connection
            self.conn = conn
            return IngoreWithWrapper(self.conn)

    def __enter__(self):
        return self.make_connection()

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        # make sure the connection is always closed
        if self.conn:
            self.conn.close()
            self.conn = None


class OmeroSource(BlitzConn):
    """
    Base Class for omero image information. Bundles functionality for image and RoIs.
    """

    def __init__(
        self,
        imageId: float,
        username: str | None = None,
        password: str | None = None,
        serverUrl: str | None = None,
        port=4064,
        secure=True,
        conn=None,
        readonly=True,
    ):
        """
        Args:
            imageId (float): omero image id
            username (str, optional): omero username. Not needed when conn is provided. Defaults to None.
            password (str, optional): omero password. Not needed when conn is provided. Defaults to None.
            serverUrl (str, optional): omero url. Not needed when conn is provided. Defaults to None.
            port (int, optional): omero port. Not needed when conn is provided. Defaults to 4064.
            secure (bool, optional): Whether to choose secure connection. Defaults to True.
            conn ([type], optional): Existing omero connection. Defaults to None.
        """
        BlitzConn.__init__(
            self,
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
            readonly=readonly,
        )

        self.imageId = imageId

    @property
    def rawPixelSize(self) -> tuple[LengthI, LengthI]:
        """Return the pixel size in omero objects

        Returns:
            Tuple[LengthI,LengthI]: x and y pixel size in omero objects
        """
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)

            size_x_obj = image.getPixelSizeX(units="MICROMETER")
            size_y_obj = image.getPixelSizeY(units="MICROMETER")

            return size_x_obj, size_y_obj

    @property
    def pixelSize(self) -> tuple[float, float]:
        """Return the pixel size in micron

        Returns:
            Tuple[float,float]: x and y pixel size in micron
        """
        size_x_obj, size_y_obj = self.rawPixelSize

        return size_x_obj.getValue(), size_y_obj.getValue()

    @property
    def pixel_size(self) -> tuple[ureg.Quantity, ureg.Quantity]:
        """Return the pixel size in micrometer

        Returns:
            Tuple[float,float]: x and y pixel size in micrometer
        """
        size_x_obj, size_y_obj = self.rawPixelSize

        unit = "micrometer".upper()

        return (
            omero.model.LengthI(size_x_obj, unit).getValue() * ureg.micrometer,
            omero.model.LengthI(size_y_obj, unit).getValue() * ureg.micrometer,
        )

    def printPixelSize(self, unit="MICROMETER"):
        """Output pixel sizes

        Args:
            unit (str, optional): Name of the unit. Defaults to "MICROMETER".
        """
        # get raw
        size_x_obj, size_y_obj = self.rawPixelSize

        # convert to correct unit
        size_x_obj = omero.model.LengthI(size_x_obj, unit)
        size_y_obj = omero.model.LengthI(size_y_obj, unit)

        # output pixel sizes
        print(f" Pixel Size X: {size_x_obj.getValue()} ({size_x_obj.getSymbol()})")
        print(f" Pixel Size Y: {size_y_obj.getValue()} ({size_y_obj.getSymbol()})")


class OmeroSequenceSource(ImageSequenceSource, OmeroSource, JupyterVisualizationMixin):
    """
    Uses omero server as a source for images
    """

    def __init__(
        self,
        imageId: int,
        username: str | None = None,
        password: str | None = None,
        serverUrl: str | None = None,
        port=4064,
        channels=None,
        z=0,
        imageQuality=1.0,
        secure=True,
        colorList=None,
        range=None,
        conn=None,
    ):
        """
        imageId: id of the image sequence
        username: omero username
        password: omero password
        serverUrl: omero server url
        port: omero port
        channels: list of image channels to activate (e.g. include fluorescence channels)
        z: focus plane
        imageQuality: quality of the rendered images (1.0=no compression, 0.0=super compression)
        base_channel: id of the phase contrast channel (visualized over all rgb channels)
        """

        OmeroSource.__init__(
            self,
            imageId=imageId,
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
            readonly=True,
        )

        if channels is None:
            channels = [1]
        if colorList is None:
            colorList = ["FFFFFF", None, None]

        self.imageId = imageId
        self.channels = channels
        self.z = z
        self.imageQuality = imageQuality
        self.colorList = colorList
        self.range = range

        if self.range is not None:
            # we make it a list
            self.range = list(self.range)

            # we have a look that it is not tool long
            if np.max(self.range) > len(self):
                logging.warning(
                    "Range exceeds number of images! Truncate to %d images", len(self)
                )
                np_range = np.array(self.range)
                self.range = np_range[np_range < len(self)]

        assert len(self.channels) <= len(self.colorList), (
            f"you must specify a color for every channel! You have {len(self.channels)} channels ({self.channels}) but only {len(self.colorList)} color(s) ({self.colorList}). Please update your colorList!"
        )

    def imageName(self) -> str:
        """
        returns the name of the image
        """
        with self.make_connection() as conn:
            return str(conn.getObject("Image", self.imageId).getName())

    def datasetName(self) -> str:
        """
        returns the name of the dataset
        """
        with self.make_connection() as conn:
            return str(conn.getObject("Image", self.imageId).getParent().getName())

    def projectName(self) -> str:
        """
        returns the name of the associated project
        """
        with self.make_connection() as conn:
            return str(conn.getObject("Image", self.imageId).getProject().getName())

    def __get_omero_image(self):
        # open the connection
        with self.make_connection() as conn:
            return conn.getObject("Image", self.imageId)

    def __get_image(self, frame: int) -> BaseImage:
        # get the specified image
        image = self.__get_omero_image()

        size_t = image.getSizeT()
        size_z = image.getSizeZ()

        t, z = compute_indices(frame, size_t, size_z)

        # perform rendering
        image.setColorRenderingModel()
        image.setActiveChannels(self.channels, colors=self.colorList)
        rendered_image = image.renderImage(z, t, compression=self.imageQuality)

        # return local image
        return LocalImage(np.asarray(rendered_image, dtype=np.uint8), frame=frame)

    def __iter__(self):
        for frame in self.frame_list:
            if self.range is not None and (frame not in self.range):
                continue
            yield self.get_frame(frame)

    def get_frame(self, frame: int):
        return self.__get_image(frame)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def frame_list(self) -> list[int]:
        if self.range is not None:
            return list(self.range)
        else:
            return list(range(len(self)))

    def __len__(self):
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            if self.range is not None:
                return min(image.getSizeT() * image.getSizeZ(), len(self.range))
            return int(image.getSizeT() * image.getSizeZ())

    @property
    def size_t(self) -> int:
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeT())

    @property
    def size_h(self) -> int:
        """Returns the height of the image in pixels."""
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeY())

    @property
    def size_w(self) -> int:
        """Returns the width of the image in pixels."""
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeX())

    @property
    def size_c(self) -> int:
        """Returns the number of channels."""
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeC())

    def toTHWC(self) -> THWCSequenceSource:
        """Convert to THWCSequenceSource

        Returns:
            THWCSequenceSource: the same image sequence but as THWCSequenceSource
        """
        image_stack = np.stack([im.raw for im in self], axis=0)
        return THWCSequenceSource(image_stack)


class OmeroRawSource(ImageSequenceSource, OmeroSource, JupyterVisualizationMixin):
    """Raw OMERO source: Allows to easily access raw that is, e.g. 16-bit data of your OMERO images"""

    def __init__(
        self,
        imageId: int,
        username: str | None = None,
        password: str | None = None,
        serverUrl: str | None = None,
        port=4064,
        secure=True,
        conn=None,
        channels=None,
    ):
        OmeroSource.__init__(
            self,
            imageId=imageId,
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
        )

        if channels is None:
            channels = [0]

        self.channels = channels

    def __len__(self):
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)

            size_z = image.getSizeZ()
            size_t = image.getSizeT()

            return size_z * size_t

    def get_frame(self, frame):
        with self.make_connection() as conn:
            # Use the pixelswrapper to retrieve the plane as
            # a 2D numpy array see [https://github.com/scipy/scipy]
            #
            # Numpy array can be used for various analysis routines
            #
            image = conn.getObject("Image", self.imageId)
            size_z = image.getSizeZ()
            size_t = image.getSizeT()

            pixels = image.getPrimaryPixels()

            t, z = compute_indices(frame, size_t, size_z)

            planes = []
            for channel in self.channels:
                c = channel

                planes.append(pixels.getPlane(z, c, t))
            return LocalImage(np.stack(planes, axis=-1))

    def __iter__(self):
        with self.make_connection() as conn:
            # Use the pixelswrapper to retrieve the plane as
            # a 2D numpy array see [https://github.com/scipy/scipy]
            #
            # Numpy array can be used for various analysis routines
            #
            image = conn.getObject("Image", self.imageId)
            size_z = image.getSizeZ()
            size_c = image.getSizeC()
            size_t = image.getSizeT()

            pixels = image.getPrimaryPixels()

            for t in range(size_t):
                for z in range(size_z):
                    planes = []
                    for channel in self.channels:
                        assert channel < size_c, "Please specify a valid channel"
                        c = channel

                        planes.append(pixels.getPlane(z, c, t))
                    yield LocalImage(np.stack(planes, axis=-1))

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def size_t(self) -> int:
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeT())

    @property
    def size_h(self) -> int:
        """Returns the height of the image in pixels."""
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeY())

    @property
    def size_w(self) -> int:
        """Returns the width of the image in pixels."""
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeX())

    @property
    def size_c(self) -> int:
        """Returns the number of channels."""
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            return int(image.getSizeC())

    def toTHWC(self) -> THWCSequenceSource:
        """Convert to THWCSequenceSource

        Returns:
            THWCSequenceSource: the same image sequence but as THWCSequenceSource
        """
        image_stack = np.stack([im.raw for im in self], axis=0)
        return THWCSequenceSource(image_stack)


class OmeroRoISource(OmeroSource, RoISource):
    """Source for OMERO RoIs beloging to an OMERO image sequence"""

    def __init__(
        self,
        imageId: int,
        username: str,
        password: str,
        serverUrl: str,
        port=4064,
        secure=True,
        roiSelector=lambda rois: [rois[0]],
        range=None,
        scale=None,
        conn=None,
    ):
        OmeroSource.__init__(
            self,
            imageId=imageId,
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
        )

        self.imageId = imageId

        self.roiSelector = roiSelector
        self.range = range
        self.scale = scale
        if self.scale:
            # 1 pixel has the size of the returned value. To move to correct domain use that size as scale factor
            self.scaleFactor = omero.model.LengthI(
                self.rawPixelSize[0], self.scale
            ).getValue()

        self.overlay = None

    def __iter__(self):
        # return overlay iterator over time
        return self.get_overlay().timeIterator(frame_range=self.range)

    def get_overlay(self):
        if self.overlay is None:
            # compose an overlay from the rois
            self.overlay = OmeroRoIStorer.load(
                self.imageId,
                username=self.username,
                password=self.password,
                serverUrl=self.serverUrl,
                port=self.port,
                secure=self.secure,
                conn=self.make_connection(),
            )

            if self.scale:
                self.overlay.scale(self.scaleFactor)

        return self.overlay

    def __len__(self) -> int:
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            if self.range is not None:
                return int(min(image.getSizeT() * image.getSizeZ(), len(self.range)))
            return int(image.getSizeT()) * int(image.getSizeZ())


def upload_file(
    omero_type: str,
    omero_id: int,
    file_path: Path,
    conn: BlitzGateway,
    mime_type="text/plain",
    namespace: str | None = None,
):
    """Upload a file attachement for an OMERO object (Image, Datase, Project)

    Args:
        omero_type (str): Image, Dataset or Project
        omero_id (int): id of the omero object
        file_path (Path): file path of the file to upload
        conn (BlitzGateway): connection to OMERO
        mime_type (str, optional): mime type of the uploaded data. Defaults to "text/plain".
        namespace(str, optional): OMERO namespace of the uploaded file

    Returns:
        _type_: created annotation object
    """

    omero_object = conn.getObject(omero_type, omero_id)

    # create the original file and file annotation (uploads the file etc.)
    file_ann = conn.createFileAnnfromLocalFile(
        file_path, mimetype=mime_type, ns=namespace, desc=None
    )

    logging.info(
        "Attaching FileAnnotation to OMERO object: File ID: %d, %s Size: %d",
        file_ann.getId(),
        file_ann.getFile().getName(),
        file_ann.getFile().getSize(),
    )

    # link it to dataset.
    omero_object.linkAnnotation(file_ann)

    # return OMERO annotation
    return file_ann


def download_file_from_object(
    omero_type: str,
    omero_id: int,
    file_name: str,
    output_path: Path,
    conn,
    append_filename=False,
):
    """Download a file annotation with a given name from an OMERO object

    Args:
        omero_type (str): Image, Dataset or Project
        omero_id (int): unique id of the OMERO object
        file_name (str): the name of the file attachment to download
        output_path (Path): output path to save the file
        conn (_type_): OMERO connection
        append_filename (bool, optional): If true the filename is appended to the output path. Defaults to False.

    Raises:
        ValueError: _description_
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    anns = list_file_annotations(omero_type=omero_type, omero_id=omero_id, conn=conn)

    anns = list(filter(lambda ann: ann.getFile().getName() == file_name, anns))

    if len(anns) != 1:
        raise ValueError(f"Annotation count is not 1 but: {len(anns)}")

    download_file_from_annotation(anns[0], output_path, append_filename)


def download_file_from_annotation(ann, output_path: Path, append_filename=False):
    """Download OMERO file attachment

    Args:
        ann (_type_): OMERO annotation object
        output_path (Path): path to write the file to
        append_filename (bool, optional): If True treats output_path as a folder and appends the OMERO file name. Defaults to False.
    """
    assert isinstance(ann, omero.gateway.FileAnnotationWrapper)

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    logging.info(
        "File ID: %d %s Size: %d",
        ann.getFile().getId(),
        ann.getFile().getName(),
        ann.getFile().getSize(),
    )

    file_path = output_path

    if append_filename:
        file_path = file_path / ann.getFile().getName()

    with open(str(file_path), "wb") as f:
        logging.info("\nDownloading file to %s...", file_path)
        for chunk in ann.getFileInChunks():
            f.write(chunk)
    logging.info("File downloaded!")


def delete_file_annotation(annotation_id: int, conn: BlitzGateway):
    """Delete an OMERO file attachement

    Args:
        annotation_id (int): id of the annotation object
        conn (BlitzGateway): OMERO connection
    """
    # ann_obj = conn.getObject('Annotation', annotation_id)
    conn.deleteObjects("Annotation", [annotation_id], wait=True)


def list_file_annotations(omero_type: str, omero_id: int, conn: BlitzGateway):
    """List all file annotations of an OMERO object

    Args:
        omero_type (str): Image, Dataset or Project OMERO object type
        omero_id (int): unique OMERO id of the object
        conn (BlitzGateway): OMERO connection

    Returns:
        _type_: List of OMERO annotation objects for files
    """
    # get OMERO object
    omero_object = conn.getObject(omero_type, omero_id)

    annotations = []

    for ann in omero_object.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            # add file annotations
            annotations.append(ann)

    return annotations


def replace_file_annotation(
    omero_type: str, omero_id: int, file_path: Path, conn, mime_type: str = "text/plain"
):
    """Replace an OMERO file by a new version.

    Deletes existing file attachements on OMERO with the same name and uploads the new file.

    Args:
        omero_type (str): Image, Dataset or Project OMERO object type
        omero_id (int): unique OMERO id of the object
        file_path (Path): path of the file to upload
        conn (_type_): OMERO connection
        mime_type (str, optional): MIME type of the data file to upload. Defaults to "text/plain".

    Returns:
        _type_: new annotation object of the uploaded file
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # only get annotations with the same filename
    anns = filter(
        lambda ann: ann.getFile().getName() == file_path.name,
        list_file_annotations(omero_type, omero_id, conn),
    )

    # upload the new file version first
    new_ann = upload_file(omero_type, omero_id, file_path, conn, mime_type=mime_type)

    # delete every other annotation with the same name
    for ann in anns:
        delete_file_annotation(ann.getId(), conn)

    return new_ann


def print_file_annotations(omero_type: str, omero_id: int, conn: BlitzGateway):
    """List all file annotations

    Args:
        omero_type (str): Image, Dataset or Project OMERO object type
        omero_id (int): unique OMERO id of the object
        conn (BlitzGateway): OMERO connection
    """
    for ann in list_file_annotations(omero_type, omero_id, conn):
        print(
            "File ID:",
            ann.getFile().getId(),
            ann.getFile().getName(),
            "Size:",
            ann.getFile().getSize(),
            "Namespace:",
            ann.getNs(),
        )
