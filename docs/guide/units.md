# Units in the extracted tables

The property extractors compute numeric values together with a physical unit per
column. {meth}`acia.analysis.ExtractorExecutor.execute` lets you pick how those
units appear in the returned DataFrame via `units=`:

| mode | columns | unit-safe math? | use for |
| --- | --- | --- | --- |
| `"none"` (default) | plain floats (unit map in `df.attrs["units"]`) | no | fast value access, plotting, existing code |
| `"header"` | floats; unit as a column-index level | no | CSV export, readable tables |
| `"pint"` | `pint[...]` extension dtype | **yes** | unit-correct computation |

Only `"pint"` is **unit-safe**: arithmetic propagates units and raises on
dimensional mismatch. The `"header"` form and the `df.attrs` map are *inert*
carriers — the unit is just a label and does **not** participate in computation:

```python
from acia.analysis import ExtractorExecutor

ex = ExtractorExecutor()

df = ex.execute(overlay, images, extractors)               # "none" (floats)
df["area"].iloc[0]          # 0.0294  (float)
df.attrs["units"]["area"]   # "micrometer ** 2"

q = ex.execute(overlay, images, extractors, units="pint")  # unit-safe
q["area"].pint.magnitude    # plain float Series
q["area"].pint.to("mm ** 2")
q["area"] / q["length"]     # -> pint[micrometer]
q["area"] + q["length"]     # raises pint.DimensionalityError

h = ex.execute(overlay, images, extractors, units="header")  # export form
h.columns                   # [("area", "micrometer ** 2"), ...]
```

Dimensionless columns (`id`, `frame`, `label`, `circularity`, fluorescence) stay
plain numbers even in `"pint"` mode, so they remain index- and merge-friendly.

The representations are convertible after the fact, so the choice is never a dead
end:

```python
from acia.analysis import attach_units, strip_units, from_header

q = attach_units(df)            # floats (+ df.attrs) -> pint dtype
floats, units = strip_units(q)  # pint dtype -> floats + {col: unit}
q = from_header(h)              # header form -> pint dtype
```

For round-tripping through CSV without losing the units, use
{func}`~acia.analysis.units.write_units_csv` and
{func}`~acia.analysis.units.read_units_csv`.
