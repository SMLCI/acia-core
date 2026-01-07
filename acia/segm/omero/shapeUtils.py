"""Utilities for working with OMERO shape objects"""


import omero
from omero.rtypes import rdouble, rint, rstring

# from omero.model.enums import UnitsLength


# Another helper for generating the color integers for shapes
def rgba_to_int(red, green, blue, alpha=255):
    """Return the color as an Integer in RGBA encoding"""
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = r + g + b + a
    if rgba_int > (2**31 - 1):  # convert to signed 32-bit int
        rgba_int = rgba_int - 2**32
    return rgba_int


def create_rectangle(x: float, y: float, width: float, height: float, z: int, t: int):
    # create a rectangle shape (added to ROI below)
    print(
        f"Adding a rectangle at theZ: {z}, theT: {t}, X: {x}, Y: {y}, width: {width}, height: {height}"
    )
    rect = omero.model.RectangleI()
    rect.x = rdouble(x)
    rect.y = rdouble(y)
    rect.width = rdouble(width)
    rect.height = rdouble(height)
    rect.theZ = rint(z)
    rect.theT = rint(t)
    rect.textValue = rstring("test-Rectangle")
    rect.fillColor = rint(rgba_to_int(255, 255, 255, 255))
    rect.strokeColor = rint(rgba_to_int(255, 255, 0, 255))

    return rect


def create_ellipse(x: float, y: float, width: float, height: float, z: int, t: int):
    # create an Ellipse shape (added to ROI below)
    ellipse = omero.model.EllipseI()
    ellipse.x = rdouble(x)
    ellipse.y = rdouble(y)
    ellipse.radiusX = rdouble(width)
    ellipse.radiusY = rdouble(height)
    ellipse.theZ = rint(z)
    ellipse.theT = rint(t)
    ellipse.textValue = rstring("test-Ellipse")

    return ellipse


def make_polystr(points):
    return rstring(" ".join(f"{str(x)},{str(y)}" for x, y in points))


def make_coordinates(polystr: str) -> list[tuple[float]]:
    return [tuple(map(float, textCoord.split(","))) for textCoord in polystr.split(" ")]


def create_polygon(
    points: list,
    z: int,
    t: int,
    fillColor=(255, 0, 255, 50),
    strokeColor=(255, 255, 0),
    description="",
):
    # create an ROI with a single polygon, setting colors and lineWidth
    polygon = omero.model.PolygonI()
    polygon.theZ = rint(z)
    polygon.theT = rint(t)
    polygon.fillColor = rint(rgba_to_int(*fillColor))
    polygon.strokeColor = rint(rgba_to_int(*strokeColor))
    # polygon.strokeWidth = omero.model.LengthI(10, UnitsLength.PIXEL)
    polygon.points = make_polystr(points)
    polygon.textValue = rstring(description)

    return polygon
