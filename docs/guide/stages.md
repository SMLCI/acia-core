# Splitting an analysis into stages

A long analysis is easier to work with as several notebooks — segment, then track, then
measure — than as one. {class}`acia.analysis.StageContext` is what holds such a chain
together: every stage of one imaged population shares an **output folder**, writes its
results there as files, and reads what the previous stage left.

```python
from acia.analysis import StageContext

ctx = StageContext.for_image(image_id, output_folder)
print(ctx)                                  # population pos001_roi002  ->  .../output
```

That one line answers the three questions every stage has: where results go, which
population this is, and what has already run here.

## Writing a stage

Name the stage when you build the context, then record as you go:

```python
ctx = StageContext.for_image(image_id, output_folder, stage='Segment')

source = open_sequence(image_id)
ctx.log_calibration(source)
ctx.log_params(backend='omnipose', diameter_px=12)

props = extract_properties(source)
write_units_csv(ctx.keyed(props), ctx.output_path('cell_properties.csv'))
np.savez(ctx.output_path('segmentation.npz'), **segmentation)

ctx.log_metrics(n_cells=len(props))
ctx.log_figure(fig, 'size_distribution', caption='cell area distribution')
ctx.finish()
```

Everything logged is written to `stage_manifest.json` **the moment it happens**, not at
the end. That matters more than it sounds: parameters and figures are the parts of a run
that re-running cannot recover, and a notebook that dies half-way used to record nothing
at all — while `scale(exist_skip=True)` then skips that stage on every later run. Until
{meth}`~acia.analysis.StageContext.finish`, the entry is marked `"running"`, so an
interrupted stage is never mistaken for a finished one.

{meth}`~acia.analysis.StageContext.output_path` joins a name onto the output folder and
creates its parent; {meth}`~acia.analysis.StageContext.input_path` does the same for a
file this stage reads, failing with a useful message instead of a confusing one further
down:

```python
segmentation = load_segmentation(ctx.input_path('segmentation.npz'))
```

The pair is about saying what you *meant* to read and write. Direction itself is still
observed rather than taken from these calls — see below — but a name declared and never
written shows up in the manifest as `io.missing`, which is usually a typo.

{meth}`~acia.analysis.StageContext.keyed` stamps the population's identity columns onto a
table so many populations concatenate cleanly later.

Under {func}`~acia.analysis.scale`, the parameters papermill injected are recorded on
their own — you do not have to repeat your parameter cell into `log_params`.

### Seeing what is there

Put the context at the end of a cell and it renders the folder: which stages ran and how
they ended, this stage's parameters, metrics and figures, and every file with the stage
that produced it.

```python
ctx
```

### Calibration

{meth}`~acia.analysis.StageContext.log_calibration` records the pixel size and frame
interval a stage actually resolved. The movie stays the single authority — this does not
become a second place to read calibration from — but it means "what interval did this run
use?" has an answer, and a later stage that resolves a different value **warns** instead
of quietly producing results that cannot be compared with the earlier ones.

### Re-running a stage

Re-running is normal and allowed — a stage is re-run because its settings changed, and
the folder should describe the run that produced the files now in it. Two things happen:

- the previous entry is **replaced, not merged**. Its metrics, figures and recorded
  outputs are dropped when the new run starts, so a re-run that dies half-way can never
  read as one complete run built from two;
- you get a warning naming what the re-run invalidates:

```text
stage 'Segment' already ran in this folder (finished 2026-08-25T08:00:24+00:00); this
run replaces its entry. Stages that read its results are now out of date until they are
re-run too: Track.
```

That last clause is the part you cannot see from the notebook you are in.
{func}`~acia.analysis.check_stale` reports it too, but only on *Track's* next run —
which is too late to be useful when the decision to re-run is being made here.

A previous run that never finished says so instead, and claims no staleness: nothing
downstream can have consumed it.

### `record()`

`ctx.record('Segment', n_cells=…)` still works and still writes exactly what it always
did; it is the explicit finalize, equivalent to logging those keyword arguments and
calling `finish()`. A context built without a `stage=` name behaves exactly as before:
nothing is written until `record()`.

A later stage does not need to be told the source again — the folder already records it:

```python
ctx = StageContext.for_image(output_folder=output_folder)   # image_id recovered
```

## What gets recorded, without being asked

While a stage runs, the context notes **which files it actually read and wrote**, and
records them with a cheap `(size, mtime)` fingerprint, alongside when it ran, how long it
took and which version of the notebook produced it:

```json
"Track": {
  "artifacts": [],
  "acia_version": "0.3.2",
  "started_at": "2026-08-17T09:10:41+00:00",
  "finished_at": "2026-08-17T09:12:03+00:00",
  "duration_s": 82.4,
  "code": {"notebook": "02_Track.ipynb", "mtime": "2026-08-16T17:02:55+00:00",
           "sha256": "9f2c…"},
  "env": {"python": "3.10.14", "platform": "Linux-5.15…", "host": "gpu-node-3"},
  "io": {
    "schema": "acia.stage_io/v1",
    "inputs": [
      {"path": "../pos001_roi001.tiff", "size": 4120233, "mtime": 1786951087},
      {"path": "segmentation.npz", "size": 88401, "mtime": 1786956912,
       "produced_by": "Segment"}
    ],
    "outputs": [{"path": "tracking/", "size": 1044, "mtime": 1786957923}]
  }
}
```

Nothing in the notebook asks for this. Two things follow from it:

**The dependency graph is derived, not declared.** `Track` read the file `Segment` wrote,
so there is an edge between them — nobody had to write that down, and it cannot go out of
date:

```python
from acia.analysis import stage_graph

stage_graph(output_folder)
# [('Segment', 'segmentation.npz', 'Track')]
```

**A result whose input moved says so.** Re-segment with a different filter and the
tracking output stays on disk looking perfectly current. The next context built in that
folder warns:

```text
UserWarning: Track may be stale -- its input 'segmentation.npz' changed after it ran
(recorded 2026-08-17T09:12:03+00:00, file modified 2026-08-17T10:41:55+00:00).
```

It only ever warns; you decide whether that matters. To actually redo the stage, remove
what it produced:

```python
ctx.clear('Track')          # deletes exactly Track's recorded outputs
```

This is also the fix for a stage that failed half-way: `scale(exist_skip=True)` keys on
the copied notebook existing, so a half-finished stage is skipped on every later run until
its traces are gone.

## Searching many runs

After a batch, {func}`~acia.analysis.stage_table` turns a whole folder of results into one
table — one row per stage run, with every setting a stage recorded as a column:

```python
from acia.analysis import stage_table

runs = stage_table('automated_executions_stages')

runs[runs.stale]                                          # what needs redoing
runs[runs.status != 'ok'][['population_id', 'stage', 'error_type', 'error_message']]
runs[runs.stage == 'Segment'].pixel_size.value_counts()   # settings drift in the batch
runs.groupby(['stage', 'code_sha256']).size()             # did it all run the same code?
runs.pivot_table(index='population_id', columns='stage',
                 values='status', aggfunc='first')        # coverage matrix
```

Every parameter and metric the stages logged is a column of its own name, so the headline
numbers of a whole fan-out — `n_detections_after`, `n_tracklets`, `growth_rate_CC` — are
answerable here without opening a single result CSV.

`status` reports how each stage *ended*, and `error_type`/`error_message` say why one did
not: across a batch that is what separates a single ROI running out of GPU memory from a
chain that is broken for every population. Counting the stages that left an entry cannot
tell those apart — a stage that ran and failed still left one.

The `code_sha256` column is worth knowing about: it is a digest of the notebook that ran,
so when two populations disagree you can tell whether they were produced by the same
analysis instead of guessing.

## Naming stages

Any string works as a stage name. Prefer one without a number — `Segment` rather than
`01_Segment` — and keep the numbering on the notebook *filename*, where it orders the
chain. A number in the stage name becomes a problem the day a stage is renumbered or one
is inserted before it: the manifest gets a new key and the previous records are orphaned,
even though it is the same stage. The ordering is already visible in `code.notebook` and
in the derived graph.

---

## Advanced

Everything below is optional. A stage chain works without any of it.

### Restricting what is captured

Capture is on by default and covers everything the stage does. `track_io=False` turns it
off, and {meth}`~acia.analysis.StageContext.track` turns it back on for a region — useful
in a notebook with exploratory cells whose reads should not count as dependencies:

```python
ctx = StageContext.for_image(image_id, output_folder, track_io=False)
...                                       # scratch work, not recorded
with ctx.track():
    overlay = segment(source)             # the real analysis
```

Regions are re-enterable, because a `with` block cannot span notebook cells: open one per
cell and everything accumulates into the next `record()`.

Reads are only noticed under the working directory, the output folder and the source.
Anything else — a model checkpoint under `/models`, say — needs `track_roots`:

```python
ctx = StageContext.for_image(image_id, output_folder, track_roots=['/models'])
```

### Reading an upstream stage's settings

```python
pixel_size = ctx.stage('Segment')['pixel_size']     # None if it never ran here
```

### Exporting lineage

{func}`acia.analysis.lineage.to_openlineage` writes the recorded runs as
[OpenLineage](https://openlineage.io) events, for handing a lineage graph to tooling
outside the project. Needs the optional extra, and writes to a file rather than a server:

```console
$ pip install "acia[lineage]"
```

```python
from acia.analysis.lineage import to_openlineage

to_openlineage('automated_executions_stages', 'lineage.jsonl')
```

### What capture cannot see

Worth knowing before trusting a record completely:

- Files read through a library's **C/C++ layer** (`cv2.imread`) raise no Python event.
  Writes are still caught, because the output folder is compared before and after; reads
  are not, so route them through `ctx.require()`.
- The same applies to **remote sources** (fsspec, SMB, OMERO), which never touch a local
  file. The source itself is always recorded regardless.
- Fingerprints are `(size, mtime)`, not content hashes — a move/change detector, not
  verification. Hashing a gigabyte image stack per stage would cost more than the
  analysis; the notebook, being kilobytes, *is* hashed.
- If capture fails for any reason, the `io` block is simply absent and the manifest is
  exactly what it would have been without this feature. Provenance never breaks a run.
