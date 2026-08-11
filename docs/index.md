# acia

**Automated single-cell image analysis for 2D+t live-cell imaging.**

`acia` turns a time-lapse microscopy file into quantitative single-cell results.
It gives you one API for reading ND2, CZI, OME-TIFF and folders of TIFFs; lazy,
numpy-style slicing over the `(T, H, W, C)` axes; physical units that travel with
the data from load to result; eight state-of-the-art segmentation and tracking
backends; and visualization from annotated videos to lineage trees.

```python
from acia.segm.open import open_sequence

src = open_sequence("experiment.nd2").position(0)
src = src[::20, 256:768, 256:768]   # every 20th frame, a 512x512 crop
src                                  # interactive viewer, right in Jupyter
```

Although it is built with microfluidic live-cell imaging in mind, nothing in the
library assumes cells — it works for any objects you can detect in images.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Getting started
:link: tutorials/01_open_your_first_file
:link-type: doc

New here? Five short tutorials take you from opening your first file to a
growth-rate curve. Every one runs in your browser on Colab — no install.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` User guide
:link: guide/index
:link-type: doc

Task-shaped recipes: reading from SMB/S3/OMERO, slicing and calibration, units in
the extracted tables, and scaling a notebook over hundreds of sequences.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API reference
:link: api/index
:link-type: doc

Every module, class and function, generated from the source.
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Glossary
:link: glossary
:link-type: doc

`Instance` or `Contour`? Tracklet graph or tracking graph? The vocabulary,
defined once.
:::
::::

## Installation

```bash
pip install acia
```

Optional features — ND2 and CZI readers, OMERO, remote shares, the interactive
widgets, and the segmentation backends — live behind extras. See
[Installation](installation.md), which also explains why the segmentation
backends are **mutually exclusive**.

## Applied examples

For complete, published analyses built on `acia` — growth-rate quantification,
fluorescence co-culture characterization, single-cell response to oxygen
alternation, and scaling those across hundreds of sequences — see the companion
[acia-workflows](https://github.com/JuBiotech/acia-workflows)
collection. The tutorials here teach the library; those notebooks show it applied
to real experiments.

```{toctree}
:hidden:

tutorials/index
guide/index
api/index
about
```
