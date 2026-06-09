=====
Usage
=====

To use AutomatedCellularImageAnalysis in a project::

    import acia


Remote image sources (SMB / SAMBA, S3, ...)
===========================================

Image sequences are read through `fsspec <https://filesystem-spec.readthedocs.io>`_,
so the same source classes work for local files and any fsspec-supported backend
(SMB/SAMBA shares, S3, HTTP, FTP, ...). For local files nothing changes::

    from acia.segm.local import LocalSequenceSource

    src = LocalSequenceSource("/data/experiments/pos1.tif")

To read from a SAMBA share, either pass a full ``smb://`` URL or use the
``SambaSequenceSource`` convenience class::

    from acia.segm.local import LocalSequenceSource, SambaSequenceSource

    # full URL (works on LocalSequenceSource too)
    src = LocalSequenceSource("smb://fileserver.lab/data/exp/pos1.tif")

    # convenience class from a URL ...
    src = SambaSequenceSource.from_url("smb://fileserver.lab/data/exp/pos1.tif")

    # ... or from explicit parts
    src = SambaSequenceSource(host="fileserver.lab", share="data",
                              path="exp/pos1.tif")

The SMB backend requires the optional ``remote`` extra::

    pip install acia[remote]


Storing credentials
-------------------

Rather than putting secrets in every script, credentials are stored once in a
per-user config file and resolved automatically by **host**. The file lives in
the OS-standard config directory (override with the ``ACIA_CONFIG`` environment
variable):

* Linux:   ``~/.config/acia/credentials.toml``
* macOS:   ``~/Library/Application Support/acia/credentials.toml``
* Windows: ``%APPDATA%\acia\credentials.toml``

It is TOML, keyed by ``[<protocol>."<host>"]`` (with an optional ``[<protocol>]``
table for protocol-wide defaults). Each entry provides one of three secret
mechanisms (precedence: keyring → env-var → plaintext):

.. code-block:: toml

    # OS keyring (most secure): password from the OS secret store,
    #   service "smb://<host>", username = the `username` value
    [smb."fileserver.lab"]
    username = "jdoe"
    domain   = "LAB"
    keyring  = true

    # env-var indirection: any *_env key is read from the environment
    [smb."other-host"]
    username     = "svc"
    password_env = "ACIA_OTHER_PW"

    # plaintext (discouraged; acia warns if the file is group/other-readable)
    [s3."my-bucket"]
    key    = "AKIA..."
    secret = "..."

With an entry in place, no secrets are needed in code -- they are matched by the
host in the URL::

    src = SambaSequenceSource.from_url("smb://fileserver.lab/data/exp/pos1.tif")

Values passed explicitly (as keyword arguments, in the URL as
``smb://user:pass@host/...``, or via ``storage_options``) always override the
config file.


Discovering all stacks in a folder
----------------------------------

``list_sequence_sources`` returns one source per matching file in a folder. It
uses the same fsspec + credential layer, so it works for a local directory or a
remote share::

    from acia.segm.local import list_sequence_sources

    # local
    for src in list_sequence_sources("/data/experiments", pattern="*.tif"):
        analyse(src)

    # SMB share -- credentials resolved by host, none in code
    sources = list_sequence_sources("smb://fileserver.lab/data/exp", pattern="*.tif")

Use ``recursive=True`` to walk sub-folders, ``storage_options=...`` to override
credentials, and pass extra keyword arguments (``normalize_image``, ``luts``,
``channel_index``) through to each source.


Scaling a notebook over many sources
====================================

:func:`acia.analysis.scale` runs an analysis notebook once per image source. It
is **source-aware**: each entry of ``image_ids`` may be an OMERO image id
(``int``), a path / ``smb://`` URL (``str``), or a parameter ``dict``. The entry
is injected into the notebook under ``parameter_name`` (default ``"image_id"``),
and the output folder is named by ``execution_naming``:

* ``int`` ids  -> ``execution_<id>``
* paths / URLs -> the file *stem* (``smb://srv/data/pos1.tif`` -> ``pos1``)

::

    from acia.analysis import scale

    # OMERO image ids (unchanged behaviour)
    scale("out", "analysis.ipynb", image_ids=[101, 102, 103])

    # SMB paths: folders named pos1/pos2/..., notebook receives image_id=<url>
    sources = list_sequence_sources("smb://fileserver.lab/data/exp")
    scale("out", "analysis.ipynb", image_ids=[s.filename for s in sources])

    # a clearer parameter name for path-based notebooks
    scale("out", "analysis.ipynb",
          image_ids=[s.filename for s in sources], parameter_name="image_path")

Inside the notebook (a papermill ``parameters`` cell), reconstruct the source
from the injected value::

    # parameters cell -> image_id = "smb://fileserver.lab/data/pos1.tif"
    from acia.segm.local import SambaSequenceSource
    source = SambaSequenceSource.from_url(image_id)

File stems are not globally unique (the same name in two folders collides); pass
a custom ``execution_naming`` if that is a concern -- ``scale`` warns when two
entries map to the same output folder. For full control, pass ``dict`` entries
(merged verbatim into the notebook parameters) together with an explicit
``execution_naming``.


Units in the extracted tables
=============================

The property extractors compute numeric values together with a physical unit per
column. :meth:`acia.analysis.ExtractorExecutor.execute` lets you pick how those
units appear in the returned DataFrame via ``units=``:

.. list-table::
   :header-rows: 1

   * - mode
     - columns
     - unit-safe math?
     - use for
   * - ``"none"`` (default)
     - plain floats (unit map in ``df.attrs["units"]``)
     - no
     - fast value access, plotting, existing code
   * - ``"header"``
     - floats; unit as a column-index level
     - no
     - CSV export, readable tables
   * - ``"pint"``
     - ``pint[...]`` extension dtype
     - **yes**
     - unit-correct computation

Only ``"pint"`` is **unit-safe**: arithmetic propagates units and raises on
dimensional mismatch. The ``"header"`` form and the ``df.attrs`` map are *inert*
carriers -- the unit is just a label and does **not** participate in
computation::

    from acia.analysis import ExtractorExecutor

    ex = ExtractorExecutor()

    df = ex.execute(overlay, images, extractors)               # "none" (floats)
    df["area"].iloc[0]          # 0.0294  (float)
    df.attrs["units"]["area"]   # "micrometer ** 2"

    q = ex.execute(overlay, images, extractors, units="pint")  # unit-safe
    q["area"].pint.magnitude    # plain float Series
    q["area"].pint.to("mm ** 2")
    q["area"] / q["length"]     # -> pint[micrometer]
    q["area"] + q["length"]     # raises pint.DimensionalityError

    h = ex.execute(overlay, images, extractors, units="header")  # export form
    h.columns                   # [("area", "micrometer ** 2"), ...]

Dimensionless columns (``id``, ``frame``, ``label``, ``circularity``,
fluorescence) stay plain numbers even in ``"pint"`` mode, so they remain index-
and merge-friendly.

The representations are convertible after the fact, so the choice is never a
dead end::

    from acia.analysis import attach_units, strip_units, from_header

    q = attach_units(df)            # floats (+ df.attrs) -> pint dtype
    floats, units = strip_units(q)  # pint dtype -> floats + {col: unit}
    q = from_header(h)              # header form -> pint dtype


Slicing & calibration (pixel size & frame interval)
===================================================

Image sequences support **numpy-style indexing** over the ``(T, H, W, C)`` axes,
implemented once on :class:`acia.base.ImageSequenceSource` so every source
(local, in-memory, THWC, OMERO, SMB) gets it::

    src[5]                      # the frame at index 5 (a BaseImage)
    src[::2]                    # a view: every second frame
    src[3:23]                   # a view: frames 3..22
    src[:, 100:200, 50:150]     # spatial crop (all frames)
    src[..., 0]                 # select channel 0 across all frames
    src[::2, 100:200, 50:150, 0]  # subsample + crop + channel, composed

An integer on the ``T`` axis returns that frame; a slice/list returns a lazy view
sequence. Views compose (``src[::2][1:]``) and never copy pixel data eagerly.

Physical calibration
--------------------

Define the **imaging interval** and **pixel size** in pint units once, at load.
They become metadata on the source and flow through slices and into extractors::

    from acia import ureg
    from acia.segm.local import LocalSequenceSource

    src = LocalSequenceSource(
        "exp.tif",
        pixel_size=0.065 * ureg.micrometer,   # space
        frame_interval=10 * ureg.minute,      # time
    )

    src.timepoints          # [0, 10, 20, ...] minute  (per frame)
    src.pixel_size          # 0.065 micrometer

Slicing transforms the calibration automatically:

* temporal subsampling scales the interval -- ``src[::2].timepoints`` is
  ``[0, 20, 40, ...]`` minute;
* a spatial **crop** keeps ``pixel_size``; a uniform spatial **step** scales it --
  ``src[:, ::2, ::2].pixel_size`` is ``0.13 micrometer``.

You can also tag an existing source/overlay fluently:
``src.with_frame_interval("10 minute")``, ``src.with_timepoints(...)``,
``src.with_pixel_size(0.065 * ureg.micrometer)``.

Overlays and detection timestamps
--------------------------------

Overlays support **temporal slicing** with slices/lists (``overlay[:20]`` cuts
after 20 frames; ``overlay[::2]`` subsamples), remapping frames to ``0..n-1``.
Indexing by a single id is unchanged (``overlay[contour_id]``). When an overlay
carries a time model, every detection gets a pint timestamp::

    overlay = overlay.with_frame_interval(10 * ureg.minute)
    overlay[:20]                 # first 20 frames, frames remapped
    overlay.timestamps          # pint array, one per contour
    contour.time                # pint Quantity for a single detection

Extractors pull the calibration
------------------------------

Spatial extractors derive their unit from ``images.pixel_size`` and the time
extractor reads ``images.timepoints`` (or the overlay's), so the common case
needs no per-extractor units::

    from acia.analysis import ExtractorExecutor, FrameEx, AreaEx, TimeEx

    df = ExtractorExecutor().execute(src, src, extractors=[
        FrameEx(), AreaEx(), TimeEx(),     # no input_unit needed
    ])

Precedence is **explicit ``input_unit`` > source calibration > default**, so the
classic style keeps working and overrides the source::

    AreaEx(input_unit=(0.065 * ureg.micrometer) ** 2)   # explicit wins
    TimeEx(input_unit="10 * minute")                    # legacy frame * interval

See also the "Units in the extracted tables" section above for how the resulting
DataFrame can expose those units (``units="none" | "header" | "pint"``).
