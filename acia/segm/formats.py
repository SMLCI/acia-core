"""Functions for different segmentation formats"""

import gzip
import json
from pathlib import Path

import numpy as np
from tifffile import imread

from acia.base import Contour, ImageSequenceSource, Instance, Overlay
from acia.utils import multi_mask_to_polygons


def parse_simple_segmentation(file_content: str) -> Overlay:
    """Parse simple segmentation format from string (json)

    Args:
        file_content (str): simple segmentation file content as string

    Returns:
        Overlay: the overlay representation of the segmentation
    """

    file_data = json.loads(file_content)

    contours = []

    for frame in file_data:
        frame_id = frame["frame"]
        for det in frame["detections"]:
            contours.append(
                Contour(det["contour"], -1.0, frame_id, det["id"], det["label"])
            )

    return Overlay(contours)


def gen_simple_segmentation(overlay: Overlay) -> str:
    """Create a simple segmentation string from an Overlay

    Args:
        overlay (Overlay): the overlay to store

    Returns:
        str: string containing the stringified simple segmentation json format
    """

    frame_packages = []

    # loop over frames
    for frame_overlay in overlay.timeIterator():
        if len(frame_overlay) == 0:
            continue

        det_objects = []
        frame_id = -1

        # loop over all contours in a frame
        for cont in frame_overlay:
            # transform coordinates to list (otherwise not json serializable)
            coordinates = cont.coordinates
            if isinstance(coordinates, np.ndarray):
                coordinates = coordinates.tolist()

            # create detection object
            det_objects.append(
                dict(
                    label=cont.label,
                    contour=coordinates,
                    id=cont.id,
                )
            )

            frame_id = cont.frame

        # create the frame package
        frame_package = dict(
            frame=frame_id,
            detections=det_objects,
        )
        frame_packages.append(frame_package)

    # serialize into json format
    return json.dumps(frame_packages, separators=(",", ":"))


def _segmentation_artifact_path(path: str | Path) -> Path:
    """Normalize a segmentation artifact path to an ``.npz`` file name."""
    path = Path(path)
    if path.suffix == ".npz":
        return path
    return path.with_name(path.name + ".npz")


def _scalar(value):
    """Python scalar from a numpy scalar, keeping int/str as they were stored."""
    return value.item() if hasattr(value, "item") else value


def save_segmentation(path: str | Path, overlay: Overlay) -> Path:
    """Store a segmentation ``Overlay`` as a compressed binary polygon archive.

    The counterpart of :func:`load_segmentation`. Contours are written as one flat
    ``float32`` coordinate array plus per-detection offsets, alongside the ``id``,
    ``label`` and ``frame`` of each detection -- so ids and sub-pixel coordinates
    survive exactly and a reloaded overlay joins against a property table exported
    from the same segmentation.

    Why this rather than the alternatives, measured on a dense 150k-detection
    scene (1024x1024, 300 cells x 500 frames): gzipped
    :func:`gen_simple_segmentation` JSON is 106 MiB / 43 s to write, this format
    is 41 MiB / 1.7 s, and a zlib label-mask stack is 18 MiB / 2.0 s. The mask
    stack is smaller still, but rasterizing renumbers detection ids
    (:func:`overlay_from_masks`) and yields mask-backed
    :class:`~acia.base.Instance` objects whose geometry is derived from the full
    frame, making downstream property extraction ~20x slower. Keeping polygons
    keeps both the ids and the fast extraction path.

    What is **not** stored: the overlay's time model (attach it on load from the
    image source, see :func:`load_segmentation`), per-detection scores, and mask
    topology -- :class:`~acia.base.Instance` masks are serialized as polygons.

    Args:
        path: output file. A missing ``.npz`` suffix is appended, so
            ``.../segmentation`` and ``.../segmentation.npz`` are equivalent.
            Parent directories are created.
        overlay: the segmentation to store.

    Returns:
        The path actually written (with the normalized suffix).

    Raises:
        TypeError: if detection ids or labels are of mixed types, which numpy
            cannot store without pickling.
    """
    path = _segmentation_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    contours = list(overlay)

    coordinate_blocks = []
    offsets = [0]
    for cont in contours:
        coordinates = np.asarray(cont.coordinates, dtype=np.float32).reshape(-1, 2)
        coordinate_blocks.append(coordinates)
        offsets.append(offsets[-1] + len(coordinates))

    coordinates_array = (
        np.concatenate(coordinate_blocks)
        if coordinate_blocks
        else np.zeros((0, 2), dtype=np.float32)
    )

    ids = np.array([cont.id for cont in contours])
    labels = np.array([cont.label for cont in contours])
    for name, values, raw in (
        ("id", ids, [cont.id for cont in contours]),
        ("label", labels, [cont.label for cont in contours]),
    ):
        # numpy would silently coerce a mix of str and int to str (so an id of 1
        # reloads as "1"), which is worse than refusing to store it
        textual = [isinstance(value, str) for value in raw]
        if values.dtype == object or (any(textual) and not all(textual)):
            raise TypeError(
                f"Detection {name}s have mixed types and cannot be stored; keep "
                f"them homogeneous (all int or all str)."
            )

    # the frame extent is cheap to store here and lets an overlay reload at its
    # true length even without a source; `source` still wins when one is given
    frames = overlay.frames()
    num_frames = int(np.max(frames)) + 1 if len(frames) else 0

    np.savez_compressed(
        path,
        coordinates=coordinates_array,
        offsets=np.asarray(offsets, dtype=np.int64),
        ids=ids,
        labels=labels,
        frames=np.array([cont.frame for cont in contours], dtype=np.int64),
        num_frames=np.int64(num_frames),
    )

    return path


def _parse_segmentation_archive(path: Path) -> tuple[list[Contour], int]:
    """Contours and stored frame extent from an ``.npz`` segmentation archive."""
    with np.load(path, allow_pickle=False) as data:
        coordinates = data["coordinates"]
        offsets = data["offsets"]
        ids = data["ids"]
        labels = data["labels"]
        frames = data["frames"]
        num_frames = int(data["num_frames"])

    contours = [
        Contour(
            coordinates[offsets[i] : offsets[i + 1]],
            -1.0,
            int(frames[i]),
            _scalar(ids[i]),
            _scalar(labels[i]),
        )
        for i in range(len(ids))
    ]

    return contours, num_frames


def load_segmentation(
    path: str | Path, source: ImageSequenceSource | None = None
) -> Overlay:
    """Load a segmentation stored by :func:`save_segmentation`.

    The format is detected from the file's magic bytes, not its suffix, so the
    binary archive written by :func:`save_segmentation` and a plain or gzipped
    simple-segmentation JSON (:func:`gen_simple_segmentation`, the interchange
    format other tools read) all load through this one function.

    Passing ``source`` -- the image sequence the segmentation was computed on --
    restores what the artifact cannot carry, and takes precedence over anything
    stored in the file:

    * the **frame extent**: the returned overlay spans ``source.size_t``. This
      matters because a movie whose last frames hold no surviving cells would
      otherwise reload shorter than it really is (the JSON interchange format
      drops empty frames entirely).
    * the **time model**: ``source.timepoints`` is attached (stamping each
      detection's ``time``) when the source is time-calibrated. An uncalibrated
      source leaves the overlay uncalibrated -- no time is invented.

    Detection ids are stable across this round-trip (see :func:`save_segmentation`).

    Args:
        path: the artifact to read. If the literal path does not exist, the
            ``.npz``-normalized name is tried, so this mirrors whatever
            :func:`save_segmentation` accepted.
        source: image sequence used to restore frame extent and time calibration.
            ``None`` falls back to the extent stored in the archive (or, for JSON
            input, the last populated frame).

    Returns:
        The segmentation overlay.

    Raises:
        FileNotFoundError: if neither the literal nor the normalized path exists.
    """
    literal = Path(path)
    if literal.is_file():
        resolved = literal
    elif _segmentation_artifact_path(literal).is_file():
        resolved = _segmentation_artifact_path(literal)
    else:
        raise FileNotFoundError(
            f"No segmentation artifact at {literal} -- expected a file written by "
            "acia.segm.formats.save_segmentation()."
        )

    with open(resolved, "rb") as raw_file:
        magic = raw_file.read(2)

    if magic == b"PK":  # zip container -> numpy .npz archive
        contours, num_frames = _parse_segmentation_archive(resolved)
        overlay = Overlay(contours, frames=list(range(num_frames)))
    else:  # simple-segmentation JSON, gzipped (\x1f\x8b) or plain
        opener = gzip.open if magic == b"\x1f\x8b" else open
        with opener(resolved, "rt", encoding="utf-8") as input_file:  # type: ignore[operator]
            overlay = parse_simple_segmentation(input_file.read())

    if source is None:
        return overlay

    # the source is the authority on both the frame extent and the time model
    overlay = Overlay(overlay.contours, frames=list(range(source.size_t)))
    timepoints = source.timepoints
    if timepoints is not None:
        overlay = overlay.with_timepoints(timepoints)

    return overlay


def load_ctc_segmentation(segmentation_path: Path) -> Overlay:
    segmentation_path = Path(segmentation_path)

    segm_mask_files = sorted(segmentation_path.glob("*.tif"))

    overlay = Overlay([], frames=list(range(len(segm_mask_files))))

    c_id = 0

    for frame_id, segm_file in enumerate(segm_mask_files):
        polygons = multi_mask_to_polygons(imread(segm_file))

        for _, poly in polygons:
            points = np.array(poly.exterior.coords.xy)
            overlay.add_contour(Contour(points, -1, frame_id, c_id, "cell"))
            c_id += 1

    return overlay


def read_ctc_segmentation_native(segmentation_path: Path) -> Overlay:
    """Fast loading of CTC segmentation masks into an Overlay

    Args:
        segmentation_path (Path): Path to the folder containing all the *.tif masks

    Returns:
        Overlay: Overlay containing all masks
    """

    # List all the segmentation masks
    segmentation_path = Path(segmentation_path)
    segm_mask_files = sorted(segmentation_path.glob("*.tif"))

    segm_masks = [imread(segm_file) for segm_file in segm_mask_files]

    return overlay_from_masks(segm_masks)  # type: ignore[arg-type]


def overlay_from_masks(segm_masks: np.ndarray) -> Overlay:
    """Create a multi-frame overlay from an array of masks

    Args:
        segm_masks (np.ndarray): mask array [T x H x W]

    Returns:
        Overlay: returns the multi-frame overly with cell instances
    """
    overlay = Overlay([], frames=list(range(len(segm_masks))))

    # unique id for instances
    uid = 1

    # Iterate all the mask files
    for frame_id, mask in enumerate(segm_masks):
        # Find all cell labels (except 0)
        labels = np.unique(mask)[1:]

        # for every label create an instance and add it to the contour
        for label in labels:
            instance = Instance(mask=mask, frame=frame_id, label=label, id=uid)
            overlay.add_contour(instance)
            uid += 1

    return overlay
