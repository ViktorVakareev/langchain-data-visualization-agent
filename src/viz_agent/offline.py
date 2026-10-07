"""``OfflineVizModel`` - a deterministic, rule-based stand-in for an LLM.

It is **not an LLM**. It speaks the real tool-calling protocol (returns ``AIMessage``
with ``tool_calls``, reads ``ToolMessage`` results), so ``create_agent`` and the real
tools run without an API key or any cost. ``plan()`` maps a question to tool calls with
keyword rules; it understands the lab's questions and similar phrasing:

- "Generate a bar chart to plot the gender count."                    -> make_chart(count, x=sex)
- "pie chart ... average value of Walc for each Gender"                -> make_chart(pie, x=sex, y=Walc)
- "box plots ... 'freetime' and 'G3'"                                  -> make_chart(box, x=freetime, y=G3)
- "scatter plots ... 'Dalc' and 'G3', and ... 'Walc' and 'G3'"         -> two make_chart(scatter) calls in parallel
- "total revenue by month per category" (line), "... heatmap"          -> online-sales.csv
- "how many students are older than 18?", "average G3 by internet", "correlation between absences and G3"
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any, Optional

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .datasets import STORE, DatasetError
from .tools import strength

# phrase -> column name (used only if the dataset has that column)
SYNONYMS = [
    (r"\bgenders?\b|\bsex\b", "sex"), (r"final grades?|\bgrades?\b", "G3"), (r"free ?time", "freetime"),
    (r"internet", "internet"), (r"mother'?s? education", "Medu"), (r"father'?s? education", "Fedu"),
    (r"weekend alcohol", "Walc"), (r"(?:daily|workday) alcohol", "Dalc"), (r"\babsen(?:ces?|t)\b", "absences"),
    (r"study ?time", "studytime"), (r"going out", "goout"), (r"\bfail(?:ures?|ed)\b", "failures"),
    (r"\bages?\b|\bold(?:er)?\b|\byears\b", "age"), (r"\bsales\b", "revenue"), (r"\bcategor(?:y|ies)\b", "category"),
    (r"\bregions?\b", "region"), (r"\bmonth(?:ly|s)?\b|over time", "month"), (r"\borders?\b", "orders"),
]
TARGETS = ("G3", "revenue", "orders")
PLOT = r"\b(plot|chart|graph|visuali[sz]e|draw|pie|box ?plots?|scatter|histogram|heatmap|heat map|bars?)\b"
COMPARE = {"over": ">", "more than": ">", "above": ">", "greater than": ">", "older than": ">", "at least": ">=",
           "under": "<", "less than": "<", "below": "<", "younger than": "<", "at most": "<=",
           "equal to": "==", "exactly": "=="}


# --------------------------------------------------------------------- understanding
def find_columns(question: str) -> tuple[str, list[str]]:
    """Best-matching dataset and the columns mentioned, in order of appearance."""
    best: tuple[int, str, list[str]] = (-1, "", [])
    for file_name in STORE.list_files():
        columns = list(STORE.load(file_name).columns)
        hits: dict[str, int] = {}
        for col in columns:  # exact names: 'G3', Medu, ...
            m = re.search(rf"(?<![\w']){re.escape(col)}(?![\w])|'{re.escape(col)}'", question, re.IGNORECASE)
            if m:
                hits.setdefault(col, m.start())
        for pattern, col in SYNONYMS:
            m = re.search(pattern, question, re.IGNORECASE)
            if m and col in columns:
                hits[col] = min(hits.get(col, m.start()), m.start())
        found = sorted(hits, key=hits.get)
        stem = re.split(r"[-_.]", file_name.lower())[-2] if "." in file_name else file_name.lower()
        bonus = (0.75 if re.search(rf"\b{re.escape(stem.rstrip('s'))}", question, re.I) else 0) + \
            (0.5 if file_name == STORE.default else 0)
        if len(found) + bonus > best[0]:
            best = (len(found) + bonus, file_name, found)
    return best[1], best[2]


def chart_kind(q: str, cols: list[str], df: pd.DataFrame) -> str:
    for kind, pattern in [("pie", r"\bpie\b"), ("box", r"\bbox ?plots?\b|boxplot"), ("heatmap", r"heat ?map"),
                          ("hist", r"histogram|distribution of"), ("scatter", r"scatter"),
                          ("line", r"\bline\b|trend|over time|monthly|by month|per month")]:
        if re.search(pattern, q, re.I):
            return kind
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if re.search(r"correlat|relationship", q, re.I) and len(numeric) >= 2:
        return "scatter"
    if re.search(r"\bcount\b|number of", q, re.I) or len(cols) == 1:
        return "count"
    return "bar"


def aggregation(q: str) -> str:
    for agg, pattern in [("sum", r"\btotal\b|\bsum\b"), ("median", r"median"), ("max", r"\bmax|highest"),
                         ("min", r"\bmin|lowest"), ("count", r"\bcount\b|number of")]:
        if re.search(pattern, q, re.I):
            return agg
    return "mean"


def chart_specs(q: str, cols: list[str], df: pd.DataFrame) -> list[dict]:
    """Columns -> make_chart arguments (one dict per chart)."""
    kind = chart_kind(q, cols, df)
    is_num = lambda c: pd.api.types.is_numeric_dtype(df[c])  # noqa: E731
    target = next((c for c in TARGETS if c in cols), "")
    others = [c for c in cols if c != target]
    cats = [c for c in others if not is_num(c)]
    agg = aggregation(q)
    if kind == "heatmap":
        if re.search(r"correlat", q, re.I) or not target or len(others) < 2:
            return [{"kind": "heatmap"}]
        x = "month" if "month" in others else others[0]
        hue = next(c for c in others if c != x)
        return [{"kind": "heatmap", "x": x, "y": target, "hue": hue, "agg": "sum" if agg == "mean" else agg}]
    if kind == "count":
        return [{"kind": "count", "x": cols[0], **({"hue": cols[1]} if len(cols) > 1 else {})}]
    if kind == "hist":
        return [{"kind": "hist", "x": next((c for c in cols if is_num(c)), cols[0])}]
    if kind == "scatter":
        y = target or cols[-1]
        return [{"kind": "scatter", "x": x, "y": y} for x in cols if x != y and is_num(x)]
    if kind == "line":
        x = "month" if "month" in others else others[0]
        rest = [c for c in others if c != x]
        y = target or next(c for c in rest if is_num(c))
        hue = next((c for c in rest if c != y), "")
        return [{"kind": "line", "x": x, "y": y, **({"hue": hue} if hue else {}), "agg": agg}]
    # bar / pie / box: x = the grouping column, y = the measured one
    x = (cats or others or cols)[0]
    y = target or next((c for c in cols if c != x and is_num(c)), "")
    spec: dict[str, Any] = {"kind": kind, "x": x}
    if y:
        spec["y"] = y
        if kind != "box":
            spec["agg"] = agg
    elif kind == "bar":
        spec["kind"] = "count"
    rest = [c for c in cols if c not in (x, y)]
    if rest and kind == "bar":
        spec["hue"] = rest[0]
    return [spec]


def comparison(q: str, cols: list[str]) -> Optional[dict]:
    """"older than 18" / "absences over 10" -> filter_rows arguments."""
    phrases = "|".join(sorted(map(re.escape, COMPARE), key=len, reverse=True))
    m = re.search(rf"({phrases}|>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)", q, re.I)
    if not m or not cols:
        return None
    op = COMPARE.get(m.group(1).lower(), m.group(1))
    return {"column": cols[0], "op": op, "value": m.group(2)}


def plan(question: str) -> list[tuple[str, dict]]:
    """Question -> the tool calls to make (all in one response)."""
    q = question.strip()
    if re.search(r"which (data ?sets|files)|what (data ?sets|files)|list (the )?(data ?sets|files)", q, re.I):
        return [("list_datasets", {})]
    file_name, cols = find_columns(q)
    if not file_name:
        return [("list_datasets", {})]
    df = STORE.load(file_name)
    with_file = lambda args: {**args, "file_name": file_name}  # noqa: E731
    if re.search(PLOT, q, re.I):
        if not cols and not re.search(r"correlat", q, re.I):
            return [("describe_dataset", {"file_name": file_name})]
        return [("make_chart", with_file(spec)) for spec in chart_specs(q, cols, df)]
    if re.search(r"\b(how many rows|columns?|describe|structure|what('s| is) in)\b", q, re.I) and not \
            re.search(r"correlat|average|mean|total", q, re.I):
        return [("describe_dataset", {"file_name": file_name})]
    flt = comparison(q, cols)
    if flt:
        return [("filter_rows", with_file(flt))]
    if re.search(r"correlat|relationship", q, re.I):
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        y = next((c for c in TARGETS if c in numeric), numeric[-1] if numeric else "")
        return [("correlation", with_file({"x": x, "y": y})) for x in numeric if x != y] or \
            [("describe_dataset", {"file_name": file_name})]
    if re.search(r"average|mean|total|sum|median|highest|lowest", q, re.I) and cols:
        target = next((c for c in TARGETS if c in cols), cols[-1])
        by = next((c for c in cols if c != target), "")
        return [("aggregate", with_file({"column": target, **({"by": by} if by else {}), "agg": aggregation(q)}))]
    return [("describe_dataset", {"file_name": file_name})]


# --------------------------------------------------------------------- answering
def describe_result(name: str, args: dict, r: Any) -> str:
    if isinstance(r, dict) and "error" in r:
        return f"I couldn't do that: {r['error']}"
    if name == "list_datasets":
        return f"Available datasets: {', '.join(r)}." if r else "There are no CSV files in the data folder."
    if name == "describe_dataset":
        cols = ", ".join(f"{c['name']} ({c['dtype']})" for c in r["columns"])
        return f"{r['file_name']} has {r['rows']} rows and {len(r['columns'])} columns: {cols}."
    if name == "filter_rows":
        return f"{r['matching_rows']} of {r['total_rows']} rows ({r['share_pct']}%) match {r['condition']}."
    if name == "correlation":
        return f"Correlation between {r['x']} and {r['y']}: r = {r['pearson_r']} ({r['strength']} relationship)."
    if name == "aggregate":
        if "values" not in r:
            return f"The {r['agg']} of {r['column']} is {_fmt(r['value'])} (n={r['n']})."
        parts = ", ".join(f"{k}: {_fmt(v)} (n={r['n'][k]})" for k, v in r["values"].items())
        return f"{r['agg'].capitalize()} {r['column']} by {r['by']} - {parts}."
    if name == "make_chart":
        return _chart_sentence(args, r) + f" Chart saved to {r['path']}."
    return str(r)


def _chart_sentence(args: dict, r: dict) -> str:
    s, kind = r["summary"], r["kind"]
    if kind == "scatter":
        text = f"{r['title']}: r = {s['pearson_r']} ({strength(s['pearson_r'])} relationship)."
        if "mean_by_x" in s:
            means = list(s["mean_by_x"].items())
            text += f" Average {args['y']} goes from {_fmt(means[0][1])} at {args['x']}={means[0][0]} " \
                    f"to {_fmt(means[-1][1])} at {args['x']}={means[-1][0]}."
        return text
    if kind == "heatmap":
        values = s["values"]
        if "x" not in args:
            pairs = {tuple(sorted((a, b))): v for a, row in values.items() for b, v in row.items() if a != b}
            top = sorted(pairs.items(), key=lambda kv: -abs(kv[1]))[:3]
            return "Strongest correlations: " + ", ".join(f"{a}-{b} {v:+.2f}" for (a, b), v in top) + "."
        cell = max(((row, col, v) for row, cols in values.items() for col, v in cols.items()), key=lambda t: t[2])
        return f"{r['title']}: the highest cell is {cell[0]} in {cell[1]} ({_fmt(cell[2])})."
    if kind == "line" and "hue" in args:
        series: dict[str, tuple[str, float]] = {}
        for x, row in s["values"].items():
            for level, v in row.items():
                if level not in series or v > series[level][1]:
                    series[level] = (x, v)
        return f"{r['title']}: peaks - " + ", ".join(f"{k} {x} ({_fmt(v)})" for k, (x, v) in series.items()) + "."
    if kind == "box":
        return f"{r['title']}. Median {args['y']} by {args['x']}: " + \
            ", ".join(f"{k}: {_fmt(v)}" for k, v in s["median"].items()) + "."
    if kind == "hist":
        return f"{r['title']}: mean {_fmt(s['mean'])}, median {_fmt(s['median'])}, range {_fmt(s['min'])}-{_fmt(s['max'])}."
    key = "counts" if "counts" in s else "values"
    values = s[key]
    flat = all(not isinstance(v, dict) for v in values.values())
    if flat:
        share = s.get("share_pct", {})
        body = ", ".join(f"{k}: {_fmt(v)}" + (f" ({share[k]}%)" if k in share else "") for k, v in values.items())
        return f"{r['title']} - {body}."
    return f"{r['title']} - " + "; ".join(f"{k}: " + ", ".join(f"{h} {_fmt(v)}" for h, v in row.items())
                                         for k, row in values.items()) + "."


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.2f}".rstrip("0").rstrip(".")
    if isinstance(v, int) and abs(v) >= 10000:
        return f"{v:,}"
    return str(v)


def load(content: Any) -> Any:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except (TypeError, ValueError, SyntaxError):
            continue
    return content


def _call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


HELP = ("Offline demo model: I can chart the sample data - e.g. \"Plot the gender count\", \"pie chart of average "
        "Walc for each gender\", \"scatter plots of Dalc vs G3 and Walc vs G3\", \"total revenue by month per category\", "
        "\"correlation heatmap\" - or answer \"average G3 by internet\", \"how many students are older than 18?\".")


class OfflineVizModel(BaseChatModel):
    """Rule-based chat model that drives the real tools through real tool calls."""

    @property
    def _llm_type(self) -> str:
        return "offline-viz-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineVizModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        question = str(messages[start].content)
        turn = messages[start:]
        asked = [c for m in turn if isinstance(m, AIMessage) for c in m.tool_calls]
        results = {m.tool_call_id: load(m.content) for m in turn if isinstance(m, ToolMessage)}
        if not asked:
            try:
                calls = plan(question)
            except DatasetError as exc:
                return _result(AIMessage(content=f"I couldn't read the data: {exc}"))
            if not re.search(PLOT + r"|\w+ (by|for each|per|vs|versus|between|than|over|columns?)\b|datasets?|files?|"
                             r"rows|describe|correlat|average|mean|total", question, re.I):
                return _result(AIMessage(content=HELP))
            return _result(AIMessage(content="", tool_calls=[_call(n, a) for n, a in calls]))
        answer = " ".join(describe_result(c["name"], c["args"], results.get(c["id"])) for c in asked)
        return _result(AIMessage(content=answer))


def _result(message: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=message)])
