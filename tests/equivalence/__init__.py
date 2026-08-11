"""Golden-reference equivalence harness for the property/filter fast paths.

Property extraction and cell filtering are being optimised in stages (see
``scenes.py`` for the scenes and ``_generate.py`` for the snapshot). Every stage
must be a *pure speed* change, so the numbers this package pins are captured from
the pre-optimisation implementation and re-asserted after each stage.
"""
