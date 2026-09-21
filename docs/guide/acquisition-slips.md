# Correcting acquisition slips

A multi-position timelapse (ND2 `P`, CZI `S`) is stored as a fixed grid of
*timepoint × position* slots, filled in the order the microscope visited them. Now
and then the microscope takes **two images in a row of the same position**. The
extra image takes the next slot, so from there to the end of the file every slot
holds the image that belongs one slot earlier, and the last real image is lost.

acia corrects this **in memory**. The raw file is never written.

## Declare the slip

Name the raw slot that holds the repeated image, i.e. the first slot that looks
wrong when you browse the file:

```python
from acia.segm.open import open_sequence

f = open_sequence("exp.nd2", acquisition_fixes=[(37, 5)])  # t=37, position 5
f.position(5).get_frame(37)   # now the real position 5, not a repeat of position 4
f.raw_position(5)             # the file's own, uncorrected order
```

Several slips accumulate. Positions the shift runs through have one frame fewer
than the file (`f.position(p).size_t`), because their last image was never recorded.

## Detect it

Most files record where the stage was for every image. A double shot is two
consecutive images at the same place:

```python
from acia.segm.acquisition_fixes import detect_acquisition_slips, save_fixes

raw = open_sequence("exp.nd2", acquisition_fixes=[])
slips = detect_acquisition_slips(raw, tol_um=2.0)  # suggestions only
```

Look at the suggested slots next to their neighbours before trusting them. The
`SequencePreprocessing` notebook shows them side by side.

## Make it stick

```python
save_fixes("exp.nd2", slips, note="double shot at t37")  # exp.nd2.acia-fixes.json
open_sequence("exp.nd2").acquisition_fixes               # picked up automatically
```

The sidecar stores the file's `T`/`P` and is refused for a file of a different
shape. Pass `acquisition_fixes=[]` to ignore it.

Selection and registration manifests record the fixes their frame indices were
made under. {func}`~acia.selection.load_selection` and
{func}`~acia.registration_persistence.load_registration` reopen the file with those
fixes, and warn if you hand them a source that uses different ones.
