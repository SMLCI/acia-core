# **Acia**: Automated single-cell image analysis

[![CI](https://github.com/SMLCI/acia-core/actions/workflows/ci.yml/badge.svg)](https://github.com/SMLCI/acia-core/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/acia.svg)](https://pypi.org/project/acia/)
[![Python versions](https://img.shields.io/pypi/pyversions/acia.svg)](https://pypi.org/project/acia/)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://smlci.github.io/acia-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Accio** 🪄 - and your single-cell insights appear - Not quite but - `acia` - and your single-cell insights appear to become much easier 😉

`acia` turns a 2D+t time-lapse microscopy file into quantitative single-cell
results. One API reads ND2, CZI, OME-TIFF and folders of TIFFs; physical units
travel with the data from load to result; eight state-of-the-art segmentation and
tracking backends plug in behind one call.

Built for microfluidic live-cell imaging, but nothing in it assumes cells — it
works for any objects you can detect in images.

## The first five minutes

```python
from acia import ureg
from acia.segm.open import open_sequence
from acia.segm.processor.omnipose import OmniposeSegmenter
from acia.analysis import extract_growth

src = open_sequence("experiment.nd2").position(0)      # ND2, CZI, TIFF, folders
src = src[::10, 256:768, 256:768]                      # lazy: subsample + crop
src = src.with_pixel_size(0.072 * ureg.micrometer)     # calibration travels along

overlay = OmniposeSegmenter()(src)                     # -> detections
table, growth, figure = extract_growth(overlay, src)   # -> µm², hours, 1/hour
```

That is a complete pipeline. `growth.doubling_time` comes back as a pint
quantity, not a bare float.

## What you get

- **One reader for every format.** `open_sequence()` dispatches ND2, CZI, TIFF
  stacks and folders of per-timepoint TIFFs to a lazy handle that reads metadata
  without touching pixels — so opening a 100 GB acquisition costs nothing. OMERO,
  SMB/SAMBA shares and S3 work through the same interface.
- **numpy-style slicing that never copies.** `src[::2, 100:200, 50:150, 0]`
  composes subsampling, cropping and channel selection into a lazy view.
- **Units that do not get lost.** Declare pixel size and frame interval once;
  they survive slicing, flow into detections as timestamps, and are picked up
  automatically by the property extractors. Results come out in µm² and hours.
- **Segmentation and tracking, swappable.** Cellpose, Cellpose-SAM, Omnipose,
  Contour Proposal Network, YOLO — and trackastra, ultrack, PyUAT, laptrack —
  behind a uniform call signature, with lazy model loading and GPU autorelease.
- **Visualization all the way to publication.** Segmentation and tracking
  overlays, scale bars, timestamps, annotated videos, lineage trees, and an
  interactive viewer that appears when you put a source at the end of a Jupyter
  cell.

## Installation

```bash
pip install acia
```

Optional readers and backends live behind extras — `acia[nd2]`, `acia[czi]`,
`acia[omero]`, `acia[remote]`, `acia[widget]`.

> **Note:** the segmentation backends are **mutually exclusive** — `cellpose`,
> `cellpose-sam` and `omnipose` pin conflicting versions, so install exactly one
> per environment. See the
> [installation guide](https://smlci.github.io/acia-core/installation.html).

## Documentation

Full documentation: **<https://smlci.github.io/acia-core>**

New here? The getting-started tutorials are runnable notebooks — open them on
Colab and nothing needs installing:

1. [Open your first file](docs/tutorials/01_open_your_first_file.ipynb) — one API for every format
2. [The sequence model](docs/tutorials/02_the_sequence_model.ipynb) — THWC and lazy slicing
3. [Look at your data](docs/tutorials/03_look_at_your_data.ipynb) — viewers, scale bars, videos
4. [Calibration and units](docs/tutorials/04_calibration_and_units.ipynb) — µm² instead of px²
5. [Segment and quantify](docs/tutorials/05_segment_and_quantify.ipynb) — the full pipeline to a growth rate

For complete published analyses built on `acia` — growth-rate quantification,
fluorescence co-culture characterization, single-cell oxygen response, and
scaling those across hundreds of sequences — see the companion
[acia-workflows](https://github.com/JuBiotech/acia-workflows)
collection.

## Developers

```bash
git clone https://github.com/SMLCI/acia-core.git
cd acia-core
pip install -e ".[dev]"

pytest
ruff check acia tests
```

To build the documentation locally (the first build downloads a ~20 MB sample
dataset and executes the tutorials):

```bash
pip install -e ".[docs,omnipose]" --use-pep517
make docs
```

Contributions are welcome — see [CONTRIBUTING.rst](CONTRIBUTING.rst).

## License

MIT — see [LICENSE](LICENSE).
