# Getting started

Five short tutorials that take you from an unopened microscopy file to a
growth-rate curve. They are meant to be read **in order** — each one builds on
the last.

Every tutorial is a real Jupyter notebook. Click the Colab badge at the top of
any of them to run it in your browser with nothing installed, or find them under
`docs/tutorials/` in the repository.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1. Open your first file
:link: 01_open_your_first_file
:link-type: doc

One function for ND2, CZI, TIFF stacks and folders of TIFFs. Read metadata
without loading pixels, and pull out a position as a time series.
:::

:::{grid-item-card} 2. The sequence model
:link: 02_the_sequence_model
:link-type: doc

The `(T, H, W, C)` convention and numpy-style slicing — subsample, crop and pick
channels without copying a single pixel.
:::

:::{grid-item-card} 3. Look at your data
:link: 03_look_at_your_data
:link-type: doc

The interactive Jupyter viewer, contact sheets, scale bars, timestamps, and
exporting an annotated video.
:::

:::{grid-item-card} 4. Calibration and units
:link: 04_calibration_and_units
:link-type: doc

Declare pixel size and frame interval once, and get results in µm² and hours
instead of px² and frames.
:::

:::{grid-item-card} 5. Segment and quantify
:link: 05_segment_and_quantify
:link-type: doc

The payoff: deep-learning segmentation, measurement, artefact filtering and a
population growth rate — on real data, GPU optional.
:::
::::

Before you start, make sure `acia` is installed — see {doc}`/installation`. The
notebooks install it themselves if you run them on Colab.

```{toctree}
:hidden:

/installation
01_open_your_first_file
02_the_sequence_model
03_look_at_your_data
04_calibration_and_units
05_segment_and_quantify
```
