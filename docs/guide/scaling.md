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
