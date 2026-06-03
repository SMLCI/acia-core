"""Unit representations for property-extractor tables.

The property extractors produce plain numeric values together with a mapping of
column -> physical unit. This module bridges between three representations of
that data:

* **plain floats** -- numeric columns, the unit map carried in
  ``df.attrs["units"]``. Fast and unsurprising, but *not* unit-safe: arithmetic
  ignores the units entirely.
* **header form** -- plain float values with the unit kept as an extra column
  index level (pint-pandas' "dequantified" form). Good for CSV export and
  readable tables, but still *not* unit-safe.
* **pint dtype** -- columns of dtype ``pint[<unit>]`` (pint-pandas extension
  arrays). This is the only **unit-safe** representation: arithmetic propagates
  units and raises on dimensional mismatch, ``.pint.to(...)`` converts, and
  ``.pint.magnitude`` drops back to plain floats.

The header form and the ``attrs`` map are inert carriers; call
:func:`attach_units` (or ``df.pint.quantify()``) to turn them back into the pint
dtype before doing unit-correct computation.
"""

from __future__ import annotations

import pandas as pd
import pint_pandas

from acia import ureg

# keep pint-pandas on the same registry as acia's Q_/U_ so columns and
# standalone Quantities interoperate without cross-registry errors
pint_pandas.PintType.ureg = ureg

#: key under which the column -> unit-string mapping is stored in ``df.attrs``
UNIT_ATTR = "units"

#: unit strings that denote "no physical dimension" and are left as plain numbers
_DIMENSIONLESS = {"", "1", "dimensionless", "none"}


def _is_dimensionless(unit) -> bool:
    if unit is None:
        return True
    text = str(unit).strip()
    if text in _DIMENSIONLESS:
        return True
    # robustly classify any remaining unit string (e.g. "1 dimensionless")
    try:
        return bool(ureg.Quantity(text).dimensionless)
    except Exception:
        return False


def _is_pint_column(series: pd.Series) -> bool:
    return isinstance(series.dtype, pint_pandas.PintType)


def attach_units(
    df: pd.DataFrame,
    units: dict | None = None,
    include_dimensionless: bool = False,
) -> pd.DataFrame:
    """Convert numeric columns into unit-safe ``pint[...]`` columns.

    Args:
        df: a DataFrame of plain numeric columns (e.g. the output of
            :meth:`acia.analysis.ExtractorExecutor.execute`).
        units: column -> unit-string mapping. Defaults to ``df.attrs["units"]``.
        include_dimensionless: if False (default), columns whose unit is
            dimensionless (e.g. ``id``, ``frame``, ``circularity``) are left as
            plain numbers so they stay index/merge friendly.

    Returns:
        A new DataFrame; dimensioned columns have dtype ``pint[<unit>]``. The
        index and any unmapped columns are preserved unchanged.
    """
    mapping = units if units is not None else df.attrs.get(UNIT_ATTR, {})

    result = df.copy()
    for column, unit in mapping.items():
        if column not in result.columns:
            continue
        if _is_pint_column(result[column]):
            continue
        if _is_dimensionless(unit) and not include_dimensionless:
            continue
        unit_str = "dimensionless" if _is_dimensionless(unit) else str(unit)
        result[column] = result[column].astype(f"pint[{unit_str}]")

    result.attrs[UNIT_ATTR] = dict(mapping)
    return result


def strip_units(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Inverse of :func:`attach_units`: turn ``pint`` columns back into floats.

    Returns:
        ``(plain_df, units)`` where ``plain_df`` has plain numeric columns (with
        the unit mapping stored in ``plain_df.attrs["units"]``) and ``units`` maps
        each column to its unit string.
    """
    known: dict[str, str] = dict(df.attrs.get(UNIT_ATTR, {}))

    result = df.copy()
    for column in result.columns:
        series = result[column]
        if _is_pint_column(series):
            known[column] = str(series.pint.units)
            result[column] = series.pint.magnitude

    result.attrs[UNIT_ATTR] = known
    return result, known


def units_in_header(df: pd.DataFrame, units: dict | None = None) -> pd.DataFrame:
    """Return the "header" form: plain floats with the unit as a column level.

    Equivalent to ``attach_units(df, units).pint.dequantify()``. The result is
    export-friendly (e.g. ``to_csv``) but **not** unit-safe; use
    :func:`from_header` to recover the pint dtype for computation.
    """
    return attach_units(df, units=units).pint.dequantify()


def from_header(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`units_in_header`: header form -> unit-safe ``pint`` dtype."""
    return df.pint.quantify()
