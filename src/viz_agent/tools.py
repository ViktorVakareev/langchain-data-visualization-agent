"""The LangChain tools the agent can call - typed, read-only, errors returned as data.

The lab gave the LLM a Python REPL (``create_pandas_dataframe_agent``) and ran whatever
code it wrote. Here the LLM can only fill in typed arguments; every column, operator
and chart kind is validated against an allow-list before pandas touches the data.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool

from .charts import AGGS, KINDS, ChartSpec, render
from .datasets import STORE, DatasetError

OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "contains")


@tool
def list_datasets() -> list[str]:
    """List the CSV datasets you can chart. Call this first if the user doesn't name a file."""
    return STORE.list_files()


@tool
def describe_dataset(file_name: str = "") -> dict[str, Any]:
    """Columns of a dataset with dtype, number of distinct values and an example value.

    Use it to find the exact column names before charting.

    Args:
        file_name: CSV file name, e.g. "students.csv" ("" = the default dataset).
    """
    try:
        return STORE.summary(file_name)
    except DatasetError as exc:
        return {"error": str(exc)}


@tool
def aggregate(column: str, by: str = "", agg: str = "mean", file_name: str = "") -> dict[str, Any]:
    """Compute a statistic of a numeric column, optionally per group - the numbers behind a chart.

    Args:
        column: numeric column, e.g. "G3".
        by: optional grouping column, e.g. "internet" ("" = whole dataset).
        agg: one of mean, sum, median, count, min, max.
        file_name: CSV file name ("" = the default dataset).

    Returns:
        {"agg", "column", "by", "values": {group: value}, "n": {group: rows}} or {"error": ...}.
    """
    try:
        df = STORE.load(file_name)
        agg = agg.lower()
        if agg not in AGGS:
            raise DatasetError(f"Unknown aggregation '{agg}'. Use one of: {', '.join(AGGS)}.")
        col = STORE.column(df, column, numeric=agg != "count")
        if not by:
            return {"agg": agg, "column": col, "value": _num(df[col].agg(agg)), "n": int(df[col].notna().sum())}
        group = STORE.column(df, by)
        grouped = df.groupby(group)[col]
        return {"agg": agg, "column": col, "by": group,
                "values": {str(k): _num(v) for k, v in grouped.agg(agg).items()},
                "n": {str(k): int(v) for k, v in grouped.size().items()}}
    except DatasetError as exc:
        return {"error": str(exc)}


@tool
def correlation(x: str, y: str, file_name: str = "") -> dict[str, Any]:
    """Pearson correlation between two numeric columns (-1 strong negative ... 0 none ... +1 strong positive).

    Args:
        x: first numeric column, e.g. "absences".
        y: second numeric column, e.g. "G3".
        file_name: CSV file name ("" = the default dataset).
    """
    try:
        df = STORE.load(file_name)
        cx, cy = STORE.column(df, x, numeric=True), STORE.column(df, y, numeric=True)
        r = float(df[cx].corr(df[cy]))
        return {"x": cx, "y": cy, "pearson_r": round(r, 3), "strength": strength(r), "n": int(len(df))}
    except DatasetError as exc:
        return {"error": str(exc)}


@tool
def filter_rows(column: str, op: str, value: str, file_name: str = "", limit: int = 5) -> dict[str, Any]:
    """Count (and preview) the rows matching one condition, e.g. absences > 10.

    Args:
        column: column to test.
        op: one of ==, !=, >, >=, <, <=, in (comma-separated values), contains (text).
        value: the value to compare with, as text (numbers are converted for numeric columns).
        file_name: CSV file name ("" = the default dataset).
        limit: how many matching rows to preview (max 20).
    """
    try:
        df = STORE.load(file_name)
        col = STORE.column(df, column)
        if op not in OPS:
            raise DatasetError(f"Unknown operator '{op}'. Use one of: {', '.join(OPS)}.")
        series = df[col]
        numeric = pd.api.types.is_numeric_dtype(series)
        convert = (lambda v: float(v)) if numeric else (lambda v: str(v).strip())  # noqa: E731
        try:
            if op == "in":
                mask = series.isin([convert(v) for v in str(value).split(",")])
            elif op == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, regex=False)
            else:
                target = convert(value)
                mask = {"==": series == target, "!=": series != target, ">": series > target,
                        ">=": series >= target, "<": series < target, "<=": series <= target}[op]
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Can't compare '{col}' with {value!r}: {exc}") from exc
        rows = df[mask]
        return {"condition": f"{col} {op} {value}", "matching_rows": int(len(rows)), "total_rows": int(len(df)),
                "share_pct": round(100 * len(rows) / max(len(df), 1), 1),
                "preview": rows.head(max(0, min(int(limit), 20))).to_dict(orient="records")}
    except DatasetError as exc:
        return {"error": str(exc)}


@tool
def make_chart(kind: str, x: str = "", y: str = "", hue: str = "", agg: str = "mean", title: str = "",
               file_name: str = "") -> dict[str, Any]:
    """Draw a chart and save it as a PNG. Returns the file path, the numbers shown and the equivalent pandas code.

    Kinds and the columns they need:
      count   - x (categorical); optional hue: rows per category ("plot the gender count")
      bar     - x (categorical) + y (numeric) aggregated with agg; optional hue
      pie     - x (categorical); optional y (numeric) aggregated with agg, else row share
      box     - x (groups) + y (numeric): distribution per group
      scatter - x + y (both numeric): relationship / correlation
      hist    - x (numeric): distribution
      line    - x (ordered, e.g. month) + y aggregated with agg; optional hue (one line per level)
      heatmap - x + y + hue: agg of y per hue (rows) and x (columns); or no columns = correlation matrix

    Args:
        kind: one of count, bar, pie, box, scatter, hist, line, heatmap.
        x, y, hue: exact column names from describe_dataset.
        agg: mean, sum, median, count, min or max.
        title: optional chart title.
        file_name: CSV file name ("" = the default dataset).
    """
    try:
        return render(ChartSpec(kind, x, y, hue, agg, title), file_name).as_dict()
    except DatasetError as exc:
        return {"error": str(exc)}


def strength(r: float) -> str:
    a = abs(r)
    word = "no clear" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return word if a < 0.1 else f"{word} {'positive' if r > 0 else 'negative'}"


def _num(v: Any) -> Any:
    v = v.item() if hasattr(v, "item") else v
    return round(v, 3) if isinstance(v, float) else v


TOOLS = [list_datasets, describe_dataset, aggregate, correlation, filter_rows, make_chart]
TOOL_NAMES = [t.name for t in TOOLS]
CHART_KINDS = list(KINDS)
