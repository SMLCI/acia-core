"""Analyse many runs at once: a catalog of results and composable analyses.

A batch (:func:`acia.analysis.scale`) leaves one output folder per ROI, each with
its tables (``cell_properties.csv``, ``doubling_times.csv``,
``cell_growth_rates.csv``, ``growth_estimates.csv``, fluorescence tables, ...), a
``summary.json`` and a ``stage_manifest.json``. :class:`ResultSet` turns one or
many such batches into tidy tables, so a question across positions, ROIs,
conditions or experiments is a few lines::

    rs = ResultSet.open(["automated_executions_stages", "../exp2/automated_executions_stages"])
    rs = rs.with_conditions({"JW_nd73-01": {"low": "1-10", "high": "11-20"}},
                            values={"low": 0.1, "high": 1.0}, value_name="substrate")

    dt = rs.table("doubling_times")                      # one row per cell cycle
    stats, fig = distribution(dt, "doubling_time", by="condition")

    cells = rs.table("cell_properties")                  # one row per detection
    curve, fig = dynamics(cells, "area", by="condition", agg="sum")

    mu = growth_rates(rs, source="fit")                  # per-cell growth rates
    fit = fit_monod(mu, S="substrate")                   # mu_max, K_s
    plot_monod(fit)

Every table carries the run keys (``experiment``, ``population_id``,
``position``, ``roi`` and, once mapped, ``condition``), so any of them can be
used for grouping. The analyses take and return plain DataFrames and draw either
a static matplotlib figure (``interactive=False``, for saving) or a plotly one
(``interactive=True``, hover shows the keys and ids).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from acia.analysis.cell_cycle import SUMMARY_NAME, read_summaries, summarize
from acia.analysis.conditions import assign_conditions
from acia.analysis.stage import MANIFEST_NAME, check_stale, read_manifest
from acia.analysis.units import UNIT_ATTR, _is_pint_column

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: the key columns a run contributes to every table it is read into
RUN_KEYS = ("experiment", "population_id", "position", "roi")


# --------------------------------------------------------------------------- #
# the catalog
# --------------------------------------------------------------------------- #
@dataclass
class ResultSet:
    """A selection of runs (one per ROI) across one or more batch folders.

    Build it with :meth:`open`; narrow it with :meth:`where`; add conditions
    with :meth:`with_conditions`. It is immutable: both return a new set.

    Attributes:
        runs: one row per run -- its keys, ``output_dir``, the stages recorded,
            ``status`` (``"ok"`` when every stage finished) and ``stale``
            (an input changed after a stage ran).
    """

    runs: pd.DataFrame
    conditions: Mapping | None = None
    condition_values: Mapping | None = None
    value_name: str = "condition_value"

    # --- building ---------------------------------------------------------

    @classmethod
    def open(cls, roots) -> ResultSet:
        """Every run below one or more folders, found by its ``stage_manifest.json``.

        The folder layout does not matter (``<root>/pos001_roi001/output``,
        ``<root>/<experiment>/pos001_roi001/output``, ...).

        Raises:
            ValueError: if no run is found.
        """
        if isinstance(roots, (str, Path)):
            roots = [roots]
        rows = []
        for root in roots:
            for manifest in sorted(Path(root).rglob(MANIFEST_NAME)):
                output_dir = manifest.parent
                data = read_manifest(output_dir)
                population = data.get("population") or {}
                stages = data.get("stages") or {}
                statuses = [str(e.get("status", "ok")) for e in stages.values()]
                rows.append(
                    {
                        "experiment": population.get("experiment")
                        or _experiment_from_path(output_dir, Path(root)),
                        "population_id": population.get("population_id")
                        or output_dir.parent.name,
                        "position": population.get("position"),
                        "roi": population.get("roi"),
                        "stages": list(stages),
                        "status": "ok"
                        if all(s == "ok" for s in statuses)
                        else "failed",
                        "stale": bool(check_stale(output_dir)),
                        "root": str(root),
                        "output_dir": str(output_dir),
                    }
                )
        if not rows:
            raise ValueError(f"no run (no {MANIFEST_NAME}) found below {list(roots)}")
        runs = pd.DataFrame(rows)
        for col in ("position", "roi"):
            runs[col] = pd.to_numeric(runs[col], errors="coerce").astype("Int64")
        return cls(runs=runs)

    def _derive(self, runs: pd.DataFrame | None = None, **changes) -> ResultSet:
        values = {
            "runs": self.runs if runs is None else runs,
            "conditions": self.conditions,
            "condition_values": self.condition_values,
            "value_name": self.value_name,
        }
        values.update(changes)
        return ResultSet(**values)

    def where(self, **criteria) -> ResultSet:
        """Keep the runs matching every criterion.

        Each value is a scalar, a list of accepted values, or a function of the
        column returning a boolean mask, e.g.
        ``rs.where(experiment="JW_nd73-01", position=range(1, 11), status="ok")``
        or ``rs.where(condition=["low", "high"])``.
        """
        runs = self.runs_with_conditions()
        mask = np.ones(len(runs), dtype=bool)
        for column, accepted in criteria.items():
            if column not in runs.columns:
                raise KeyError(f"no run column {column!r} (have {list(runs.columns)})")
            values = runs[column]
            if callable(accepted):
                mask &= np.asarray(accepted(values), dtype=bool)
            elif isinstance(accepted, (list, tuple, set, range, np.ndarray, pd.Series)):
                mask &= values.isin(list(accepted)).to_numpy()
            else:
                mask &= (values == accepted).to_numpy()
        return self._derive(runs=self.runs[mask].reset_index(drop=True))

    def with_conditions(
        self,
        conditions: Mapping,
        *,
        values: Mapping[str, Any] | None = None,
        value_name: str = "condition_value",
    ) -> ResultSet:
        """Map positions to conditions; see :func:`acia.analysis.conditions.assign_conditions`.

        Every table read afterwards gets a ``condition`` column (and
        ``value_name``, e.g. ``"substrate"``, with ``values``). Unmapped runs
        are reported once, as a warning.
        """
        derived = self._derive(
            conditions=conditions, condition_values=values, value_name=value_name
        )
        unmapped = derived.runs_with_conditions().attrs.get("unmapped") or []
        if unmapped:
            listed = ", ".join(f"{e} pos {p}" for e, p in unmapped[:8])
            more = f" (+{len(unmapped) - 8} more)" if len(unmapped) > 8 else ""
            warnings.warn(
                f"{len(unmapped)} run(s) have no condition: {listed}{more}",
                stacklevel=2,
            )
        return derived

    def runs_with_conditions(self) -> pd.DataFrame:
        """:attr:`runs`, plus ``condition`` (and its value) once mapped."""
        if self.conditions is None:
            return self.runs
        return assign_conditions(
            self.runs,
            self.conditions,
            values=self.condition_values,
            value_name=self.value_name,
        )

    # --- reading ----------------------------------------------------------

    def tables(self) -> pd.Series:
        """Which tables exist, and in how many runs."""
        counts: dict[str, int] = {}
        for folder in self.runs["output_dir"]:
            for csv in Path(folder).glob("*.csv"):
                counts[csv.stem] = counts.get(csv.stem, 0) + 1
        return pd.Series(counts, name="runs", dtype=int).sort_index()

    def table(self, name: str, *, units: bool = True) -> pd.DataFrame:
        """One table concatenated over all runs, tidy, with the run keys.

        Args:
            name: the file stem, e.g. ``"doubling_times"``, ``"cell_growth_rates"``,
                ``"cell_properties"``, ``"growth_estimates"`` -- any CSV the
                stages wrote.
            units: keep unit-aware (``pint``) columns where the CSV records
                units; ``False`` gives plain floats, with the units in
                ``attrs["units"]``.

        Returns:
            The rows of every run, with the table's own row id (``tracklet``,
            ``id``, ...) as a column and the run keys (and ``condition``) added.
            Runs without that table are listed in a warning.
        """
        runs = self.runs_with_conditions()
        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        unit_map: dict[str, str] = {}
        for run in runs.itertuples(index=False):
            path = Path(run.output_dir) / f"{name}.csv"
            if not path.is_file():
                missing.append(f"{run.experiment}/{run.population_id}")
                continue
            df = _to_plain(_read_table(path))
            unit_map.update(df.attrs.get(UNIT_ATTR) or {})
            keys = {
                k: getattr(run, k)
                for k in (*RUN_KEYS, "condition", self.value_name)
                if hasattr(run, k)
            }
            for key, value in keys.items():
                df[key] = value  # the run's key wins over a stale copy in the file
            frames.append(df)
        if missing:
            warnings.warn(
                f"{len(missing)} run(s) have no {name}.csv: {', '.join(missing[:6])}"
                + (" ..." if len(missing) > 6 else ""),
                stacklevel=2,
            )
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out.attrs[UNIT_ATTR] = unit_map
        if units and unit_map:
            out = _attach(out, unit_map)
        return out

    def summaries(self) -> pd.DataFrame:
        """One row per run from its ``summary.json`` (``section.field`` columns)."""
        runs = self.runs_with_conditions()
        paths, keys = [], {}
        for run in runs.itertuples(index=False):
            path = Path(run.output_dir) / SUMMARY_NAME
            if path.is_file():
                paths.append(path)
                keys[path] = {
                    k: getattr(run, k)
                    for k in (*RUN_KEYS, "condition", self.value_name)
                    if hasattr(run, k)
                }
        return read_summaries(paths, keys=keys)


def _experiment_from_path(output_dir: Path, root: Path) -> str:
    rel = output_dir.relative_to(root).parts
    return rel[0] if len(rel) > 2 else root.resolve().name


def _read_table(path: Path) -> pd.DataFrame:
    """A stage CSV -- read with its unit header row when it has one."""
    from acia.analysis.units import read_units_csv

    with open(path, encoding="utf-8") as fh:
        fh.readline()
        has_unit_row = fh.readline().startswith("unit,")
    df = read_units_csv(path) if has_unit_row else pd.read_csv(path)
    index_name = df.index.name
    if index_name and index_name not in df.columns:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)
    return df


def _to_plain(df: pd.DataFrame) -> pd.DataFrame:
    """Pint columns -> float magnitudes (units stay in ``attrs``)."""
    out = df.copy()
    units = dict(df.attrs.get(UNIT_ATTR) or {})
    for col in out.columns:
        if _is_pint_column(out[col]):
            units[col] = str(out[col].pint.units)
            out[col] = np.asarray(out[col].values.quantity.magnitude, dtype=float)
    out.attrs[UNIT_ATTR] = units
    return out


def _attach(df: pd.DataFrame, units: Mapping[str, str]) -> pd.DataFrame:
    from acia.analysis.units import _is_dimensionless

    out = df.copy()
    for col, unit in units.items():
        if (
            col in out.columns
            and not _is_dimensionless(unit)
            and pd.api.types.is_numeric_dtype(out[col])
        ):
            out[col] = pd.array(out[col].astype(float), dtype=f"pint[{unit}]")
    out.attrs[UNIT_ATTR] = dict(units)
    return out


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def _values(df: pd.DataFrame, col: str, unit: str | None) -> tuple[np.ndarray, str]:
    series = df[col]
    if _is_pint_column(series):
        q = series.values.quantity
        if unit:
            q = q.to(unit)
        return np.asarray(q.magnitude, dtype=float), str(q.units)
    recorded = str((df.attrs.get(UNIT_ATTR) or {}).get(col, ""))
    values = np.asarray(series, dtype=float)
    if unit and recorded and unit != recorded:
        from acia import Q_

        values = Q_(values, recorded).to(unit).magnitude
        recorded = unit
    return values, unit or recorded


def _only_valid(df: pd.DataFrame, valid_only: bool) -> pd.DataFrame:
    if valid_only and "valid" in df.columns:
        return df[df["valid"].astype(str).str.lower() == "true"]
    return df


def _hover_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in ("experiment", "position", "roi", "condition", "tracklet", "id")
        if c in df.columns
    ]


def distribution(
    df: pd.DataFrame,
    value: str,
    *,
    by: str | list[str] = "condition",
    kind: str = "violin",
    valid_only: bool = True,
    unit: str | None = None,
    interactive: bool = False,
    ax: Axes | None = None,
):
    """A value's distribution per group -- doubling time, μ, cell length, fluorescence.

    Args:
        df: a tidy table, e.g. ``rs.table("doubling_times")``.
        value: the column to show.
        by: the grouping column(s): ``"condition"``, ``"position"``,
            ``"experiment"``, ``["experiment", "condition"]``, ...
        kind: ``"violin"`` (default), ``"box"``, ``"strip"`` or ``"ecdf"``.
        valid_only: drop rows whose ``valid`` is False (tracking artifacts).
        unit: convert the value to this unit first (e.g. ``"1/hour"``).
        interactive: plotly figure (hover shows the keys and ids) instead of
            matplotlib.

    Returns:
        ``(stats, figure)``: n, mean, std, median and SEM per group, and the plot.
    """
    data = _only_valid(df, valid_only)
    v, v_unit = _values(data, value, unit)
    by_cols = [by] if isinstance(by, str) else list(by)
    plot_df = data[list(dict.fromkeys(by_cols + _hover_cols(data)))].copy()
    plot_df[value] = v
    plot_df = plot_df.dropna(subset=[value])
    plot_df["group"] = plot_df[by_cols].astype(str).agg(" · ".join, axis=1)

    stats = (
        plot_df.groupby(by_cols, dropna=False)[value]
        .apply(lambda s: pd.Series(summarize(s.to_numpy())))
        .unstack()
        .drop(columns="unit")
    )
    stats["unit"] = v_unit
    label = f"{value.replace('_', ' ')}" + (f" [{v_unit}]" if v_unit else "")
    groups = sorted(plot_df["group"].unique(), key=_natural_key)

    if interactive:
        import plotly.express as px

        hover = [c for c in _hover_cols(plot_df)]
        if kind == "ecdf":
            fig = px.ecdf(
                plot_df,
                x=value,
                color="group",
                hover_data=hover,
                category_orders={"group": groups},
            )
            fig.update_xaxes(title=label)
        elif kind == "box":
            fig = px.box(
                plot_df,
                x="group",
                y=value,
                points="all",
                hover_data=hover,
                category_orders={"group": groups},
            )
        elif kind == "strip":
            fig = px.strip(
                plot_df,
                x="group",
                y=value,
                hover_data=hover,
                category_orders={"group": groups},
            )
        else:
            fig = px.violin(
                plot_df,
                x="group",
                y=value,
                points="all",
                box=True,
                hover_data=hover,
                category_orders={"group": groups},
            )
        if kind != "ecdf":
            fig.update_yaxes(title=label)
            fig.update_xaxes(title=" · ".join(by_cols))
        fig.update_layout(template="plotly_white", height=420)
        return stats, fig

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(groups) + 1.5), 3.8))
    else:
        fig = ax.figure
    samples = [plot_df.loc[plot_df["group"] == g, value].to_numpy() for g in groups]
    positions = np.arange(1, len(groups) + 1)
    rng = np.random.default_rng(0)
    if kind == "ecdf":
        for g, s in zip(groups, samples, strict=True):
            xs = np.sort(s)
            ax.step(
                xs,
                np.arange(1, len(xs) + 1) / len(xs),
                where="post",
                label=f"{g} (n={len(xs)})",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("fraction of cells")
        ax.legend(fontsize=8)
    else:
        if kind == "violin":
            usable = [
                (p, s) for p, s in zip(positions, samples, strict=True) if len(s) > 1
            ]
            if usable:
                parts = ax.violinplot(
                    [s for _, s in usable],
                    positions=[p for p, _ in usable],
                    showmeans=True,
                    showextrema=False,
                )
                for body in list(parts["bodies"]):  # type: ignore[call-overload]
                    body.set_alpha(0.35)
        elif kind == "box":
            ax.boxplot(samples, positions=positions, widths=0.5, showfliers=False)
        for p, s in zip(positions, samples, strict=True):
            ax.scatter(
                p + rng.uniform(-0.08, 0.08, len(s)),
                s,
                s=8,
                alpha=0.5,
                color="tab:blue",
            )
        ax.set_xticks(
            positions,
            [f"{g}\n(n={len(s)})" for g, s in zip(groups, samples, strict=True)],
        )
        ax.set_ylabel(label)
    fig.tight_layout()
    return stats, fig


def dynamics(
    df: pd.DataFrame,
    value: str | None,
    *,
    time: str = "time",
    by: str | list[str] = "condition",
    agg: str = "mean",
    band: str = "sem",
    unit: str | None = None,
    time_unit: str | None = None,
    show_runs: bool = True,
    interactive: bool = False,
    ax: Axes | None = None,
):
    """A value over time per group -- cell count, total area, mean fluorescence, ...

    First every run is reduced to one curve (``agg`` over its rows per time
    point: ``"mean"``, ``"median"``, ``"sum"`` or ``"count"`` -- ``value`` is
    ignored for ``"count"``), then the runs of a group are averaged, with a band
    of ``"sem"``, ``"std"`` or ``"ci"`` (95%) across runs. Thin lines are the
    single runs.

    Returns:
        ``(curves, figure)``: per group and time point the mean over runs, the
        band and the number of runs.
    """
    by_cols = [by] if isinstance(by, str) else list(by)
    data = df.copy()
    t, t_unit = _values(data, time, time_unit)
    data["_t"] = t
    if agg == "count":
        data["_v"] = 1.0
        v_unit = ""
        label = "cells"
    else:
        if value is None:
            raise ValueError("value is required unless agg='count'")
        data["_v"], v_unit = _values(data, value, unit)
        label = f"{agg} {value.replace('_', ' ')}" + (f" [{v_unit}]" if v_unit else "")
    run_keys = [k for k in ("experiment", "population_id") if k in data.columns]
    reducer = {"count": "sum", "mean": "mean", "median": "median", "sum": "sum"}[agg]
    per_run = (
        data.groupby(list(dict.fromkeys([*by_cols, *run_keys, "_t"])), dropna=False)[
            "_v"
        ]
        .agg(reducer)
        .reset_index()
    )

    def _band(s: pd.Series) -> float:
        if len(s) < 2:
            return float("nan")
        sd = float(s.std(ddof=1))
        spread = {
            "std": sd,
            "sem": sd / np.sqrt(len(s)),
            "ci": 1.96 * sd / np.sqrt(len(s)),
        }
        return float(spread[band])

    curves = (
        per_run.groupby(by_cols + ["_t"], dropna=False)["_v"]
        .agg(mean="mean", band=_band, runs="count")
        .reset_index()
        .rename(columns={"_t": time})
    )
    curves.attrs[UNIT_ATTR] = {time: t_unit, "mean": v_unit}
    per_run["group"] = per_run[by_cols].astype(str).agg(" · ".join, axis=1)
    curves["group"] = curves[by_cols].astype(str).agg(" · ".join, axis=1)
    groups = sorted(curves["group"].unique(), key=_natural_key)
    x_label = f"{time} [{t_unit}]" if t_unit else time

    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure()
        palette = _palette(len(groups))
        for color, g in zip(palette, groups, strict=True):
            c = curves[curves["group"] == g]
            if show_runs:
                for _, run in per_run[per_run["group"] == g].groupby(run_keys):
                    name = " / ".join(str(run[k].iloc[0]) for k in run_keys)
                    fig.add_trace(
                        go.Scatter(
                            x=run["_t"],
                            y=run["_v"],
                            mode="lines",
                            line={"width": 0.7, "color": color},
                            opacity=0.35,
                            name=name,
                            legendgroup=g,
                            showlegend=False,
                            hovertemplate=f"{name}<br>%{{x}}, %{{y}}<extra></extra>",
                        )
                    )
            fig.add_trace(
                go.Scatter(
                    x=c[time],
                    y=c["mean"],
                    mode="lines",
                    name=f"{g}",
                    legendgroup=g,
                    line={"width": 3, "color": color},
                    error_y={
                        "type": "data",
                        "array": c["band"],
                        "visible": True,
                        "thickness": 0.8,
                    },
                )
            )
        fig.update_layout(
            template="plotly_white", height=420, xaxis_title=x_label, yaxis_title=label
        )
        return curves, fig

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    for color, g in zip(_palette(len(groups)), groups, strict=True):
        c = curves[curves["group"] == g]
        if show_runs:
            for _, run in per_run[per_run["group"] == g].groupby(run_keys):
                ax.plot(run["_t"], run["_v"], color=color, lw=0.6, alpha=0.3)
        ax.plot(
            c[time],
            c["mean"],
            color=color,
            lw=2,
            label=f"{g} ({int(c['runs'].max())} runs)",
        )
        ax.fill_between(
            c[time],
            c["mean"] - c["band"],
            c["mean"] + c["band"],
            color=color,
            alpha=0.2,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return curves, fig


def per_run(
    df: pd.DataFrame,
    value: str,
    *,
    stat: str = "median",
    valid_only: bool = True,
    unit: str | None = None,
    color: str | None = "condition",
    interactive: bool = False,
    ax: Axes | None = None,
):
    """One number per run (e.g. its median doubling time) against its position.

    Compares positions -- a gradient across the device, a bad chamber, an edge
    effect -- with the runs coloured by ``color`` (e.g. the condition).

    Returns:
        ``(table, figure)``: the run keys, the statistic and the run's n.
    """
    data = _only_valid(df, valid_only)
    v, v_unit = _values(data, value, unit)
    data = data.assign(_v=v)
    keys = [k for k in (*RUN_KEYS, "condition") if k in data.columns]
    table = (
        data.groupby(keys, dropna=False)["_v"]
        .agg(**{stat: stat, "n": "count"})
        .reset_index()
        .sort_values([k for k in ("experiment", "position", "roi") if k in keys])
    )
    label = f"{stat} {value.replace('_', ' ')}" + (f" [{v_unit}]" if v_unit else "")
    hue = color if color in table.columns else None

    if interactive:
        import plotly.express as px

        fig = px.scatter(
            table,
            x="position",
            y=stat,
            color=hue,
            symbol="experiment" if "experiment" in table else None,
            hover_data=[k for k in keys if k != hue] + ["n"],
        )
        fig.update_layout(template="plotly_white", height=400, yaxis_title=label)
        return table, fig

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure
    if hue:
        for (name, part), c in zip(
            table.groupby(hue, dropna=False),
            _palette(table[hue].nunique(dropna=False)),
            strict=True,
        ):
            ax.scatter(part["position"], part[stat], color=c, label=str(name), s=25)
        ax.legend(fontsize=8, title=hue)
    else:
        ax.scatter(table["position"], table[stat], s=25)
    ax.set_xlabel("position")
    ax.set_ylabel(label)
    fig.tight_layout()
    return table, fig


# --------------------------------------------------------------------------- #
# growth rates for cross-run fits
# --------------------------------------------------------------------------- #
def growth_rates(
    rs: ResultSet,
    source: str = "fit",
    *,
    valid_only: bool = True,
    exclude_flagged: bool = False,
) -> pd.DataFrame:
    """One growth rate per cell (or per run), in 1/h, with the run keys.

    Args:
        rs: the runs.
        source: where μ comes from:

            * ``"fit"`` -- each cell's exponential fit (``cell_growth_rates``);
            * ``"from_doubling_time"`` -- ``ln 2 / doubling time`` per cell;
            * ``"population:TSCA"`` (or ``CC`` / ``TCA``) -- the population fit
              of ``03_GrowthRate`` per run.
        valid_only: drop cells outside the tracklet filters.
        exclude_flagged: also drop cells whose growth fit was flagged.

    Returns:
        A table with a ``mu`` column (float, 1/h) plus keys and conditions.
    """
    if source.startswith("population:"):
        method = source.split(":", 1)[1]
        est = rs.table("growth_estimates", units=False)
        est = est[est["method"] == method].copy()
        if est.empty:
            raise ValueError(f"no {method!r} population fit in growth_estimates")
        units = (
            est["growth_rate_unit"].astype(str).str.replace("1/", "1 / ", regex=False)
        )
        from acia import Q_

        est["mu"] = [
            Q_(v, u).to("1/hour").magnitude
            for v, u in zip(est["growth_rate"], units, strict=True)
        ]
        est.attrs[UNIT_ATTR] = {"mu": "1 / hour"}
        return est
    rates = rs.table("cell_growth_rates", units=True)
    rates = _only_valid(rates, valid_only)
    if exclude_flagged and "fit_outlier" in rates.columns:
        rates = rates[rates["fit_outlier"].astype(str).str.lower() != "true"]
    column = {"fit": "mu_fit", "from_doubling_time": "mu_from_dt"}.get(source)
    if column is None:
        raise ValueError(
            "source must be 'fit', 'from_doubling_time' or 'population:<CC|TSCA|TCA>'"
        )
    out = rates.copy()
    out["mu"], _ = _values(rates, column, "1/hour")
    out.attrs[UNIT_ATTR] = {**(rates.attrs.get(UNIT_ATTR) or {}), "mu": "1 / hour"}
    return out.dropna(subset=["mu"])


# --------------------------------------------------------------------------- #
# Monod kinetics
# --------------------------------------------------------------------------- #
@dataclass
class MonodFit:
    """``μ(S) = μmax · S / (Ks + S)`` fitted to per-condition growth rates.

    Attributes:
        mu_max, ks: the parameters (μ in 1/h, Ks in the unit of S).
        mu_max_se, ks_se: their standard errors; ``*_ci`` the 95% intervals.
        r2: coefficient of determination of the fit to the condition means.
        points: per condition S, mean μ, its SEM/std and n (what was fitted).
        s_unit: the unit of S (for labels).
    """

    mu_max: float
    ks: float
    mu_max_se: float
    ks_se: float
    mu_max_ci: tuple[float, float]
    ks_ci: tuple[float, float]
    r2: float
    points: pd.DataFrame
    s_unit: str = ""

    def to_dict(self) -> dict:
        return {
            "mu_max_per_hour": self.mu_max,
            "mu_max_se": self.mu_max_se,
            "mu_max_ci": list(self.mu_max_ci),
            "ks": self.ks,
            "ks_se": self.ks_se,
            "ks_ci": list(self.ks_ci),
            "ks_unit": self.s_unit,
            "r2": self.r2,
            "points": self.points.to_dict(orient="records"),
        }

    def predict(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.asarray(self.mu_max * s / (self.ks + s), dtype=float)


def fit_monod(
    df: pd.DataFrame,
    *,
    S: str = "substrate",
    mu: str = "mu",
    group: str | list[str] = "condition",
    weights: str | None = "sem",
    s_unit: str = "",
) -> MonodFit:
    """Fit Monod kinetics to growth rates across conditions (and experiments).

    The rates are first averaged per ``group`` (every condition, or e.g.
    ``["experiment", "condition"]`` to keep experiments apart as replicates);
    the curve is fitted to those means, weighted by their ``"sem"`` or
    ``"std"`` (``None``: unweighted).

    Args:
        df: per-cell or per-run growth rates with an ``S`` column, e.g.
            :func:`growth_rates` after :meth:`ResultSet.with_conditions` with
            ``value_name="substrate"``.
        S: the substrate concentration column.
        mu: the growth rate column (1/h).
        group: what one fitted point is.
        weights: ``"sem"``, ``"std"`` or ``None``.
        s_unit: unit of S, for the result and the plot labels.

    Raises:
        ValueError: with fewer than three distinct concentrations.
    """
    from scipy.optimize import curve_fit

    group_cols = [group] if isinstance(group, str) else list(group)
    data = df.dropna(subset=[S, mu])
    points = (
        data.groupby(group_cols + [S], dropna=False)[mu]
        .agg(mu="mean", std="std", n="count")
        .reset_index()
        .rename(columns={S: "S"})
    )
    points["sem"] = points["std"] / np.sqrt(points["n"])
    if points["S"].nunique() < 3:
        raise ValueError("a Monod fit needs at least three distinct concentrations")

    def monod(s, mu_max, ks):
        return mu_max * s / (ks + s)

    sigma = None
    if weights in ("sem", "std"):
        sigma = points[weights].to_numpy(dtype=float)
        if np.any(~np.isfinite(sigma) | (sigma <= 0)):
            finite = sigma[np.isfinite(sigma) & (sigma > 0)]
            sigma = np.where(
                np.isfinite(sigma) & (sigma > 0),
                sigma,
                np.median(finite) if len(finite) else 1.0,
            )
    s = points["S"].to_numpy(dtype=float)
    y = points["mu"].to_numpy(dtype=float)
    guess = (float(np.nanmax(y)), float(np.nanmedian(s)) or 1.0)
    params, cov = curve_fit(
        monod,
        s,
        y,
        p0=guess,
        sigma=sigma,
        absolute_sigma=sigma is not None,
        bounds=(0, np.inf),
        maxfev=20000,
    )
    se = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(2, np.nan)
    residual = y - monod(s, *params)
    r2 = 1 - float(np.sum(residual**2)) / float(np.sum((y - y.mean()) ** 2) or np.nan)
    dof = max(1, len(s) - 2)
    from scipy.stats import t as student_t

    q = float(student_t.ppf(0.975, dof))
    return MonodFit(
        mu_max=float(params[0]),
        ks=float(params[1]),
        mu_max_se=float(se[0]),
        ks_se=float(se[1]),
        mu_max_ci=(float(params[0] - q * se[0]), float(params[0] + q * se[0])),
        ks_ci=(float(params[1] - q * se[1]), float(params[1] + q * se[1])),
        r2=r2,
        points=points,
        s_unit=s_unit,
    )


def plot_monod(
    fit: MonodFit,
    *,
    interactive: bool = False,
    ax: Axes | None = None,
    log_x: bool = False,
):
    """The fitted Monod curve over the condition means (± SEM), with μmax and Ks."""
    pts = fit.points
    s_max = float(pts["S"].max()) * 1.1 or 1.0
    grid = (
        np.geomspace(max(float(pts["S"][pts["S"] > 0].min()) / 3, 1e-9), s_max, 200)
        if log_x
        else np.linspace(0, s_max, 200)
    )
    s_label = "S" + (f" [{fit.s_unit}]" if fit.s_unit else "")
    text = (
        f"μmax = {fit.mu_max:.3g} ± {fit.mu_max_se:.2g} 1/h<br>"
        f"Ks = {fit.ks:.3g} ± {fit.ks_se:.2g} {fit.s_unit}<br>R² = {fit.r2:.3f}"
    )
    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure()
        hover = [c for c in pts.columns if c not in ("S", "mu", "std", "sem", "n")]
        fig.add_trace(
            go.Scatter(
                x=pts["S"],
                y=pts["mu"],
                mode="markers",
                name="conditions",
                error_y={"type": "data", "array": pts["sem"], "visible": True},
                customdata=pts[hover + ["n"]].astype(str).to_numpy(),
                hovertemplate="<br>".join(
                    f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover + ["n"])
                )
                + "<br>S %{x}, μ %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(x=grid, y=fit.predict(grid), mode="lines", name="Monod fit")
        )
        fig.add_annotation(
            x=0.98,
            y=0.05,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            align="right",
        )
        fig.update_layout(
            template="plotly_white",
            height=420,
            xaxis_title=s_label,
            yaxis_title="growth rate μ [1/h]",
        )
        if log_x:
            fig.update_xaxes(type="log")
        return fig

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.errorbar(
        pts["S"], pts["mu"], yerr=pts["sem"], fmt="o", capsize=3, label="conditions"
    )
    ax.plot(grid, fit.predict(grid), "-", label="Monod fit")
    ax.axhline(fit.mu_max, color="0.6", lw=0.8, ls=":")
    ax.axvline(fit.ks, color="0.6", lw=0.8, ls=":")
    ax.text(
        0.97,
        0.05,
        text.replace("<br>", "\n"),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(s_label)
    ax.set_ylabel("growth rate μ [1/h]")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _natural_key(text: str):
    import re

    # (0, number) / (1, text) pairs, so numbers and text never get compared:
    # "0.1 mM" < "0.3 mM" < "1 mM" < "10 mM", "pos2" < "pos10"
    parts = re.split(r"(\d+(?:\.\d+)?)", str(text))
    return [(0, float(p), "") if i % 2 else (1, 0.0, p) for i, p in enumerate(parts)]


def _palette(n: int) -> list:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [_hex(cmap(i % cmap.N)) for i in range(max(n, 1))][:n]


def _hex(rgba) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgba[:3])


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` (e.g. ``fit.to_dict()``) as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


__all__ = [
    "RUN_KEYS",
    "MonodFit",
    "ResultSet",
    "distribution",
    "dynamics",
    "fit_monod",
    "growth_rates",
    "per_run",
    "plot_monod",
    "save_json",
]
