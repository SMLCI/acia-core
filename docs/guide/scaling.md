# Scaling a notebook over many sequences

{func}`acia.analysis.scale` runs an analysis notebook once per image source. It
is **source-aware**: each entry of `image_ids` may be an OMERO image id (`int`),
a path or `smb://` URL (`str`), or a parameter `dict`. The entry is injected into
the notebook under `parameter_name` (default `"image_id"`), and the output folder
is named by `execution_naming`:

* `int` ids → `execution_<id>`
* paths / URLs → the file *stem* (`smb://srv/data/pos1.tif` → `pos1`)

```python
from acia.analysis import scale

# OMERO image ids
scale("out", "analysis.ipynb", image_ids=[101, 102, 103])

# SMB paths: folders named pos1/pos2/..., notebook receives image_id=<url>
sources = list_sequence_sources("smb://fileserver.lab/data/exp")
scale("out", "analysis.ipynb", image_ids=[s.filename for s in sources])

# a clearer parameter name for path-based notebooks
scale("out", "analysis.ipynb",
      image_ids=[s.filename for s in sources], parameter_name="image_path")
```

Inside the notebook — in a papermill `parameters` cell — reconstruct the source
from the injected value:

```python
# parameters cell -> image_id = "smb://fileserver.lab/data/pos1.tif"
from acia.segm.local import SambaSequenceSource
source = SambaSequenceSource.from_url(image_id)
```

File stems are not globally unique (the same name in two folders collides); pass
a custom `execution_naming` if that is a concern — `scale` warns when two entries
map to the same output folder. For full control, pass `dict` entries (merged
verbatim into the notebook parameters) together with an explicit
`execution_naming`.

Use `max_workers` to run executions in parallel, and `exist_skip=True` to resume
a partially completed sweep without redoing finished sequences.

`analysis_script` also takes a **list** of notebooks, run in order per source — that is
how a chain of {doc}`stages <stages>` is scaled over a dataset. Each run then records what
it read and produced, so `stage_table()` summarises the whole batch afterwards.

## Progress output

There are always two levels: one bar counting **sources**, and per-notebook bars
counting **cells**. Sequentially, the source bar is labelled with what is running
right now, and each stage notebook gets its own bar underneath:

```text
01_Segment.ipynb | pos001_roi001.tiff:  33%|███| 1/3 [04:41<09:23, 281.63s/source]
  ↳ 01_Segment.ipynb | pos001_roi001.tiff: 100%|███| 21/21 [01:21<00:00, 3.88s/cell]
  ↳ 02_Track.ipynb | pos001_roi001.tiff:   100%|███| 15/15 [02:57<00:00, 10.11s/cell]
```

With `max_workers > 1` there is no single "current" source, so the source bar
reports the one that just finished (`done …` / `FAILED …`) and each worker gets a
bar of its own:

```text
Sources:  33%|███| 1/3 [04:41<09:23, 281.63s/source]
  [w1] pos001_roi001.tiff | 02_Track.ipynb:   57%|███| 12/21 [00:32<00:35, 3.91s/cell]
  [w2] pos002_roi001.tiff | 01_Segment.ipynb: 19%|█  |  4/21 [00:00<00:03, 4.88cell/s]
```

Those worker bars are drawn by the calling process from progress the workers
report over a queue. Workers deliberately never draw bars themselves: they share
one stderr with no shared cursor, so their output would overwrite itself into
unreadable fragments (particularly in a notebook, whose output area does not
emulate a terminal cursor).

`stage_progress` controls the per-notebook bars: `"keep"` (default) leaves the
finished ones on screen as a per-stage timing log, `"collapse"` removes each bar
once its stage is done, and `"off"` shows only the source bar.
