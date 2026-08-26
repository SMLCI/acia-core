=============
API reference
=============

Generated from the source. If you are new to ``acia``, start with the
:doc:`tutorials </tutorials/01_open_your_first_file>` instead — this page is for
looking things up.

``acia`` itself exports only the pint registry used throughout the library:
``ureg`` (the application registry), ``Q_`` (``Quantity``) and ``U_`` (``Unit``).
Everything else is imported from the submodules below.

Core data model
===============

Detections, overlays and the image-sequence abstraction that every reader
implements.

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.base
   acia.utils
   acia.colors
   acia.timing

Reading image sequences
=======================

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.segm.open
   acia.segm.local
   acia.segm.folder_source
   acia.segm.nd2_source
   acia.segm.czi_source
   acia.segm.tiff_metadata
   acia.segm.tiff_export
   acia.segm.formats
   acia.segm.utils
   acia.segm.omero.storer
   acia.segm.omero.utils
   acia.segm.omero.shapeUtils
   acia.config

Segmentation
============

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.segm.processor.canny
   acia.segm.processor.cellpose
   acia.segm.processor.cellpose_sam
   acia.segm.processor.omnipose
   acia.segm.processor.flowpose_rt
   acia.segm.processor.cpn
   acia.segm.processor.yolo
   acia.segm.processor.offline
   acia.segm.processor.online
   acia.segm.processor.predict
   acia.segm.filter

Tracking
========

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.tracking
   acia.tracking.formats
   acia.tracking.utils
   acia.tracking.output
   acia.tracking.processor.trackastra
   acia.tracking.processor.ultrack
   acia.tracking.processor.laptrack
   acia.tracking.processor.pyuat
   acia.tracking.processor.utils

Analysis
========

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.analysis
   acia.analysis.properties
   acia.analysis.growth_rate
   acia.analysis.doubling_time
   acia.analysis.units
   acia.analysis.stage

Visualization
=============

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.viz
   acia.viz.compose
   acia.viz.utils
   acia.segm.output

Preprocessing and curation
==========================

Drift correction, ROI selection and the provenance helpers.

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.registration
   acia.registration_persistence
   acia.selection
   acia.crop_capture
   acia.attribute

Notebook integration
====================

The Jupyter display mixin and the interactive curation widgets.

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   acia.notebook
