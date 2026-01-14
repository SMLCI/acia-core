"""Utils for OMERO segmenation data"""


import cv2
import numpy as np
import omero
from omero.gateway import BlitzGateway, DatasetWrapper, ImageWrapper, ProjectWrapper
from PIL import Image, ImageDraw, ImageFont

from acia.segm.omero.storer import OmeroSequenceSource
from acia.utils import ScaleBar


def getImage(conn: BlitzGateway, imageId: int) -> ImageWrapper:
    """Get omero image by id
        Note: only images in your current group are accessible

    Args:
        conn (BlitzGateway): current omero connection
        imageId (int): image id

    Returns:
        ImageWrapper: image object
    """
    return conn.getObject("Image", imageId)


def getDataset(conn: BlitzGateway, datasetId: int) -> DatasetWrapper:
    """Get omero dataset by id
        Note: only datasets in your current group are accessible

    Args:
        conn (BlitzGateway): active omero connection
        datasetId (int): dataset id

    Returns:
        DatasetWrapper: dataset object
    """
    return conn.getObject("Dataset", datasetId)


def getProject(conn: BlitzGateway, projectId: int) -> ProjectWrapper:
    """Get omero project by id
        Note: only projects in your current group are accessible

    Args:
        conn (BlitzGateway): active omero connection
        projectId (int): project id

    Returns:
        ProjectWrapper: project object
    """
    return conn.getObject("Project", projectId)


def list_projects(conn: BlitzGateway) -> list[ProjectWrapper]:
    """List projects in the current user group
        Note: only projects in your current group are accessible

    Args:
        conn (BlitzGateway): Current omero BlitzGateway connection

    Returns:
        List[ProjectWrapper]: List of project wrappers
    """
    return conn.getObjects("Project")  # type: ignore[no-any-return]


def list_image_ids_in_dataset(conn: BlitzGateway, datasetId: int) -> list[int]:
    """[summary]

    Args:
        conn (BlitzGateway): active omero connection
        datasetId (int): dataset id

    Returns:
        List[int]: array of all image ids of the dataset
    """
    return [
        image.getId() for image in conn.getObjects("Image", opts={"dataset": datasetId})
    ]


def list_images_in_dataset(conn: BlitzGateway, datasetId: int) -> list[ImageWrapper]:
    """List all images in the omero dataset

    Args:
        conn (BlitzGateway): active omero connection
        datasetId (int): dataset id

    Returns:
        List[ImageWrapper]: List of omero images
    """
    return conn.getObjects("Image", opts={"dataset": datasetId})  # type: ignore[no-any-return]


def list_datasets_in_project(
    conn: BlitzGateway, projectId: int
) -> list[DatasetWrapper]:
    return conn.getObjects("Dataset", opts={"project": projectId})  # type: ignore[no-any-return]


def list_images_in_project(conn: BlitzGateway, projectId: int) -> list[ImageWrapper]:
    return [
        image
        for dataset in list_datasets_in_project(conn, projectId=projectId)
        for image in dataset.listChildren()
    ]


def list_image_ids_in(omero_id: int, omero_type: str, conn: BlitzGateway) -> list[int]:
    """List all image sequences in an omero source (dataset, project or image)

    Args:
        omero_id (int): the omero id specifying the resource on the omero server
        omero_type (str): the type of the omero source. Can be 'project', 'dataset' or 'image'.
        conn (BlitzGateway): the connection to omero

    Raises:
        Exception: when wrong omero_type is specified

    Returns:
        List[int]: list of image ids contained in the OMERO resource
    """

    omero_type = omero_type.lower()
    func = None

    if omero_type == "project":
        func = list_images_in_project  # type: ignore[assignment]
    elif omero_type == "dataset":
        func = list_images_in_dataset  # type: ignore[assignment]
    elif omero_type == "image":
        return [omero_id]
    else:
        raise Exception(
            f"Wrong omero_type: '{omero_type}'! Please choose one of 'project', 'dataset' or 'image'!"
        )

    return list(map(lambda image: image.getId(), func(conn, omero_id)))  # type: ignore[misc]


def get_image_name(conn: BlitzGateway, imageId: int) -> str:
    return conn.getObject("Image", imageId).getName()  # type: ignore[no-any-return]


def get_project_name(conn: BlitzGateway, projectId: int) -> str:
    return conn.getObject("Project", projectId).getName()  # type: ignore[no-any-return]


def image_iterator(conn: BlitzGateway, object) -> ImageWrapper:
    if object.OMERO_CLASS == "Image":
        yield object
    if object.OMERO_CLASS == "Dataset":
        yield from list_images_in_dataset(conn, object.getId())
    if object.OMERO_CLASS == "Project":
        yield from list_images_in_project(conn, object.getId())


def create_project(conn: BlitzGateway, project_name: str) -> ProjectWrapper:
    new_project = ProjectWrapper(conn, omero.model.ProjectI())
    new_project.setName(project_name)
    new_project.save()
    return new_project


def create_dataset(
    conn: BlitzGateway, projectId: int, dataset_name: str
) -> DatasetWrapper:
    # Use omero.gateway.DatasetWrapper:
    new_dataset = DatasetWrapper(conn, omero.model.DatasetI())
    new_dataset.setName(dataset_name)
    new_dataset.save()
    # Can get the underlying omero.model.DatasetI with:
    dataset_obj = new_dataset._obj

    # Create link to project
    link = omero.model.ProjectDatasetLinkI()
    # We can use a 'loaded' object, but we might get an Exception
    # link.setChild(dataset_obj)
    # Better to use an 'unloaded' object (loaded = False)
    link.setChild(omero.model.DatasetI(dataset_obj.id.val, False))
    link.setParent(omero.model.ProjectI(projectId, False))
    conn.getUpdateService().saveObject(link)

    return new_dataset


class OmeroScaleBar(ScaleBar):
    """Renderer for scale bar to show metric size of pixels on image"""

    def __init__(
        self,
        oss: OmeroSequenceSource,
        width,
        unit="MICROMETER",
        short_title="μm",
        color=(255, 255, 255),
        font_size=25,
    ):
        self.width = width
        self.unit = unit
        self.color = color
        self.short_title = short_title
        self.font_size = font_size

        pixelSizes = oss.rawPixelSize

        pixelSize = omero.model.LengthI(pixelSizes[0], self.unit).getValue()

        self.pixelWidth = int(np.round(self.width / pixelSize))

        # TODO: make parameter
        self.pixelHeight = 10

        # TODO: no fixed font file
        self.font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf", self.font_size
        )

    def draw(self, image, xstart, ystart):
        # TODO: thickness parameter
        # draw line
        cv2.line(
            image, (xstart, ystart), (xstart + self.pixelWidth, ystart), self.color, 1
        )
        half_y = int(np.round(self.pixelHeight / 2))
        cv2.line(
            image, (xstart, ystart - half_y), (xstart, ystart + half_y), self.color
        )
        cv2.line(
            image,
            (xstart + self.pixelWidth, ystart - half_y),
            (xstart + self.pixelWidth, ystart + half_y),
            self.color,
        )

        # draw size
        unit_text = self.short_title
        text = f"{self.width} {unit_text}"
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        text_x, text_y = draw.textsize(text, font=self.font)
        draw.text(
            (xstart + self.pixelWidth / 2 - text_x / 2, ystart - text_y - 2),
            text,
            fill=self.color,
            font=self.font,
        )
        image = np.array(img_pil)

        return image


def has_all_tags(object, tag_list: list[str] | None = None):
    if tag_list is None:
        tag_list = []

    tag_list = tag_list.copy()
    for ann in object.listAnnotations():
        if omero.model.TagAnnotationI == ann.OMERO_TYPE and ann.getTextValue() in tag_list:
            del tag_list[tag_list.index(ann.getTextValue())]

        if len(tag_list) == 0:
            break

    return len(tag_list) == 0
