import pytest

from viz_agent import TOOLS, aggregate, correlation, describe_dataset, filter_rows, list_datasets, make_chart
from viz_agent.tools import strength


def test_tool_names_and_args_the_llm_sees():
    assert [t.name for t in TOOLS] == ["list_datasets", "describe_dataset", "aggregate", "correlation",
                                       "filter_rows", "make_chart"]
    assert list(make_chart.args) == ["kind", "x", "y", "hue", "agg", "title", "file_name"]
    assert "scatter - x + y (both numeric)" in make_chart.description
    for t in TOOLS:
        assert t.description and "file_name" in t.args or t.name == "list_datasets"


def test_list_and_describe():
    assert list_datasets.invoke({}) == ["online-sales.csv", "students.csv"]
    assert describe_dataset.invoke({"file_name": "online-sales.csv"})["rows"] == 144
    assert "error" in describe_dataset.invoke({"file_name": "../x.csv"})


def test_aggregate():
    r = aggregate.invoke({"column": "G3", "by": "internet"})
    assert set(r["values"]) == {"no", "yes"} and sum(r["n"].values()) == 400
    assert aggregate.invoke({"column": "revenue", "agg": "sum", "file_name": "online-sales.csv"})["value"] > 1e6
    assert aggregate.invoke({"column": "sex", "by": "school", "agg": "count"})["agg"] == "count"
    assert "Unknown aggregation" in aggregate.invoke({"column": "G3", "agg": "avg"})["error"]
    assert "not numeric" in aggregate.invoke({"column": "sex"})["error"]


def test_correlation():
    r = correlation.invoke({"x": "absences", "y": "G3"})
    assert r["pearson_r"] < 0 and r["strength"] == "no clear"
    assert correlation.invoke({"x": "G1", "y": "G3"})["strength"] == "strong positive"
    assert "error" in correlation.invoke({"x": "sex", "y": "G3"})


@pytest.mark.parametrize("args, matching", [
    ({"column": "age", "op": ">", "value": "18"}, 28), ({"column": "sex", "op": "==", "value": "F"}, None),
    ({"column": "Medu", "op": "in", "value": "0,1"}, None), ({"column": "school", "op": "contains", "value": "river"}, None),
])
def test_filter_rows(args, matching):
    r = filter_rows.invoke(args)
    assert 0 < r["matching_rows"] < 400 and len(r["preview"]) <= 5
    if matching is not None:
        assert r["matching_rows"] == matching


@pytest.mark.parametrize("args, message", [
    ({"column": "age", "op": "drop", "value": "1"}, "Unknown operator"),
    ({"column": "age", "op": ">", "value": "old"}, "Can't compare"),
    ({"column": "nope", "op": ">", "value": "1"}, "not found"),
])
def test_filter_rows_errors_are_data(args, message):
    assert message in filter_rows.invoke(args)["error"]


def test_make_chart_tool():
    r = make_chart.invoke({"kind": "count", "x": "sex"})
    assert r["path"].endswith("count-sex.png") and "value_counts" in r["code"]
    assert "Unknown chart kind" in make_chart.invoke({"kind": "radar", "x": "sex"})["error"]


@pytest.mark.parametrize("r, word", [(0.05, "no clear"), (-0.2, "weak negative"), (0.4, "moderate positive"), (-0.9, "strong negative")])
def test_strength(r, word):
    assert strength(r) == word
