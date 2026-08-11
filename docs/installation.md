# Installation

## Stable release

```bash
pip install acia
```

`acia` requires **Python 3.10 or newer** and is tested on 3.10–3.13.

## Optional features (extras)

The core install stays deliberately light. File-format readers, remote storage,
the interactive widgets and the deep-learning backends are installed on demand:

| Extra | Install | Gives you |
| --- | --- | --- |
| `nd2` | `pip install acia[nd2]` | Reading Nikon ND2 files ({class}`~acia.segm.nd2_source.ND2SequenceSource`) |
| `czi` | `pip install acia[czi]` | Reading Zeiss CZI files ({class}`~acia.segm.czi_source.CZISequenceSource`) |
| `omero` | `pip install acia[omero]` | Reading and writing OMERO images and ROIs |
| `remote` | `pip install acia[remote]` | SMB/SAMBA shares plus the OS-keyring credential store |
| `widget` | `pip install acia[widget]` | The interactive curation widgets (`ROICropper`, `SequenceDashboard`, …) |
| `notebook` | `pip install acia[notebook]` | Jupyter display helpers |
| `dev` | `pip install acia[dev]` | ruff, mypy, pytest, pre-commit — for contributors |
| `docs` | `pip install acia[docs]` | Sphinx and friends, to build this site |

Extras combine: `pip install "acia[nd2,czi,remote]"`.

Reading plain TIFF stacks, OME-TIFFs and folders of per-timepoint TIFFs needs no
extra — that support is in the core install.

## Segmentation backends

(mutually-exclusive-backends)=

:::{warning}
**The segmentation backends are mutually exclusive — install exactly one per
environment.**

They pin conflicting versions of `cellpose`, `torch` and `numpy`, so pip cannot
satisfy two of them at once:

* `cellpose` — classic Cellpose, pins `cellpose<4`
* `cellpose-sam` — Cellpose-SAM, pins `cellpose>=4`
* `omnipose` — pins `omnipose==1.0.6` **and** `scipy==1.11.4`

If you need to compare backends, use one virtual environment (or one Colab
runtime) per backend. That is exactly what the `acia-workflows` CI does — one
job per backend.
:::

| Extra | Install | Backend |
| --- | --- | --- |
| `cellpose` | `pip install acia[cellpose]` | {class}`~acia.segm.processor.cellpose.CellposeSegmenter` (Cellpose v3) |
| `cellpose-sam` | `pip install acia[cellpose-sam]` | {class}`~acia.segm.processor.cellpose_sam.CellposeSAMSegmenter` |
| `omnipose` | `pip install acia[omnipose]` | {class}`~acia.segm.processor.omnipose.OmniposeSegmenter` |
| `flowpose-rt` | see below | {class}`~acia.segm.processor.flowpose_rt.FlowposeRTSegmenter` |

`flowpose-rt` is a lightweight Omnipose-compatible backend that needs neither
cellpose nor omnipose nor numba at runtime. It is not on PyPI yet, so install it
from source:

```bash
pip install ../flowpose-rt
pip install "acia[flowpose-rt]"
```

**No GPU, or just trying things out?**
{class}`~acia.segm.processor.canny.CannySegmentationProcessor` needs no extra at
all — no torch, no model download. It is far less accurate than the deep-learning
backends, but it lets you run a complete pipeline end to end on any laptop.

Tracking backends (`trackastra`, `ultrack`, `laptrack`, `pyuat`) are installed
directly rather than through extras; see their own documentation.

## From source

```bash
git clone https://github.com/SMLCI/acia-core.git
cd acia-core
pip install -e ".[dev]"
```

Run the test suite and linters before opening a merge request:

```bash
pytest
ruff check acia tests
ruff format --check acia tests
```

## Building this documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The tutorials are executed as part of the build (see
{doc}`tutorials/01_open_your_first_file`), so the first run downloads a sample
dataset and takes a few minutes; subsequent builds reuse the execution cache.
