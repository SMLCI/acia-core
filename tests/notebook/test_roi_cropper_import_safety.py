"""Import-safety test for ``acia.notebook`` when ``anywidget`` is absent.

``anywidget`` is an OPTIONAL dependency. ``acia.notebook`` is imported by
``acia.base`` / ``acia.segm.local`` / ``acia.segm.nd2_source``, so it MUST stay
importable without the extra; in that case ``ROICropper`` is a stub that raises a
clear ``pip install acia[widget]`` ImportError on instantiation.

This module deliberately does NOT use ``pytest.importorskip("anywidget")`` -- it
must run regardless of whether anywidget is installed, since it simulates the
missing-dependency case itself.
"""

import builtins
import importlib
import sys

import pytest


def test_import_safe_without_anywidget():
    real_import = builtins.__import__
    blocked = {"anywidget", "traitlets"}

    def fake_import(name, *args, **kwargs):
        if name in blocked or name.split(".")[0] in blocked:
            raise ImportError(f"simulated missing dependency: {name}")
        return real_import(name, *args, **kwargs)

    # Snapshot and drop acia.notebook plus its importers so a fresh import re-runs
    # the optional-dependency probe under the patched importer.
    affected = [
        "acia.notebook",
        "acia.base",
        "acia.segm.local",
        "acia.segm.nd2_source",
    ]
    saved = {name: sys.modules.get(name) for name in affected}
    for name in affected:
        sys.modules.pop(name, None)

    try:
        builtins.__import__ = fake_import
        notebook = importlib.import_module("acia.notebook")
        # Importing succeeded even though anywidget/traitlets are "missing".
        assert notebook._HAS_ANYWIDGET is False
        # The stubs raise a clear, actionable ImportError on instantiation.
        with pytest.raises(ImportError, match=r"acia\[widget\]"):
            notebook.ROICropper(object())
        with pytest.raises(ImportError, match=r"acia\[widget\]"):
            notebook.FilterExplorer(object(), object(), [])
        with pytest.raises(ImportError, match=r"acia\[widget\]"):
            notebook.SequenceDashboard(object())
    finally:
        builtins.__import__ = real_import
        # Restore the original modules so other tests see a normal import state.
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
        # Reload acia.notebook cleanly with the real importer in effect.
        importlib.reload(importlib.import_module("acia.notebook"))
