"""Sphinx configuration for the acia documentation.

The site is built by ``.github/workflows/docs.yml`` and published to
https://smlci.github.io/acia-core.

Two things about this setup are deliberate:

* **myst-nb, not myst-parser.** The tutorials under ``docs/tutorials/`` are real
  notebooks committed *without* outputs (enforced by the ``nbstripout``
  pre-commit hook), and executed here at build time. Nothing binary ever enters
  git, and every build is a proof that the tutorials still run against the
  current API.
* **autosummary, not sphinx-apidoc.** The API reference is generated from
  ``docs/api/index.rst`` during the normal ``sphinx-build``, so it cannot silently
  vanish the way the old ``make docs``-only ``modules.rst`` did.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

import acia  # noqa: E402

# -- Project information -----------------------------------------------------

project = "acia"
author = "Johannes Seiffarth"
copyright = f"2021-{datetime.now():%Y}, {author}"  # noqa: A001
version = acia.__version__
release = acia.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# myst-nb registers both .md and .ipynb; no explicit source_suffix needed.

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- Notebook execution ------------------------------------------------------
#
# "cache" executes a notebook only when its source changes, keyed on a hash, so
# incremental builds stay fast. Every tutorial is deliberately subsampled
# (``src[::20, 256:768, 256:768]``) to keep even the GPU one within a couple of
# CPU-minutes -- the reader turns the knob back up on Colab.

nb_execution_mode = "cache"
nb_execution_timeout = 900
nb_execution_raise_on_error = True
# Escape hatch: add a glob here if a notebook ever becomes too heavy for CI.
# Anything listed here ships whatever outputs it was committed with, which for a
# stripped notebook means *no* outputs -- so prefer subsampling over excluding.
nb_execution_excludepatterns: list[str] = []
nb_merge_streams = True

# acia uses tqdm.auto. Inside a kernel that resolves to the *ipywidget* bar,
# whose output is an inert <script type="...widget-view+json"> tag that a static
# page simply does not render -- harmless. Should ipywidgets ever be absent,
# though, tqdm falls back to the std bar, which writes to stderr and would show
# up as a red error block on every page (nb_output_stderr defaults to "show").
# This pins that fallback shut. Readers on Colab still get live progress bars.
os.environ.setdefault("TQDM_DISABLE", "1")

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Optional backends. acia deliberately keeps these behind extras -- and the
# segmentation ones are *mutually exclusive* (cellpose<4 vs cellpose>=4 vs
# omnipose's scipy==1.11.4 pin), so no single environment can import them all.
# Mocking lets the API reference document every module regardless of which
# extras the docs environment happens to have.
autodoc_mock_imports = [
    "aicspylibczi",
    "anywidget",
    "celldetection",
    "cellpose",
    "cellpose_omni",
    "flowpose_rt",
    "keyring",
    "laptrack",
    # mmcv/mmdet back the legacy offline predictors; neither is declared in any
    # extra, so they are never present in a docs environment.
    "mmcv",
    "mmdet",
    "nd2",
    "omero",
    "smbclient",
    "torch",
    "trackastra",
    "uatrack",
    "ultralytics",
    "ultrack",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
# Render a docstring's ``Attributes:`` section as inline :ivar: fields rather
# than standalone py:attribute objects. Without this, every documented dataclass
# field (RotatedCropSpec, FrameTransform, GrowthRateResult, StageContext, ...) is
# emitted twice -- once by napoleon and once by :undoc-members: -- which Sphinx
# reports as a duplicate object description.
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "pint": ("https://pint.readthedocs.io/en/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"acia {version}"
html_show_sourcelink = False

html_theme_options = {
    "github_url": "https://github.com/SMLCI/acia-core",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/acia/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_align": "left",
    "show_prev_next": True,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

html_context = {
    "github_user": "SMLCI",
    "github_repo": "acia-core",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}

htmlhelp_basename = "aciadoc"

# -- Link checking -----------------------------------------------------------

# Publisher sites reject the linkchecker's user agent even though the DOI itself
# resolves fine in a browser. Ignore those rather than let a 403 fail the build.
linkcheck_ignore = [
    r"https://doi\.org/10\.1093/bioinformatics/.*",  # OUP returns 403 to bots
]
linkcheck_anchors = False
linkcheck_timeout = 30
