# **Acia**: Automated single-cell image analysis

[![CI](https://github.com/SMLCI/acia-core/actions/workflows/ci.yml/badge.svg)](https://github.com/SMLCI/acia-core/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/acia.svg)](https://pypi.org/project/acia/)
[![Python versions](https://img.shields.io/pypi/pyversions/acia.svg)](https://pypi.org/project/acia/)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://smlci.github.io/acia-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Accio** 🪄 - and your single-cell insights appear - Not quite but - `acia` - and your single-cell insights appear to become much easier 😉

The `acia` library provides a modular image analysis pipeline utility functionality for analysing 2D+t time-lapse image sequences in microfluidic live-cell imaging experiments. It provides:
- Abstraction for various image sources (local, OMERO)
- automated image analysis for instance segmentation and tracking (eight SOTA AI approaches supported out of the box)
- automated and unit-aware single-object property extraction.
- extended visualization in videos, charts and interactive charts including segmentation masks and lineage trees

Although the funtionality is developed with microfluidic applications in mind, the library can be used for any objects detected in images.

**Note:** For examples of its usage please visit our application workflow collection including more than **10** real-world examples: https://github.com/JuBiotech/acia-workflows

## Installation

Install `acia` from pypi:

```bash
pip install acia
```


## Developers

1. Clone this repository
    ```bash
    git clone https://github.com/JuBiotech/acia-core.git
    cd acia-core
    ```

2.Install `acia` in development mode

    ```bash
    pip install -e .
    ```
