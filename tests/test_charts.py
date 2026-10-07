import os
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from viz_agent import STORE, ChartSpec, DatasetError, render
from viz_agent.charts import KINDS, _fold, code_for, compact, default_title, validate

SPECS = [
    ChartSpec("count", x="sex"), ChartSpec("count", x="school", hue="sex"),
    ChartSpec("bar", x="internet", y="G3"), ChartSpec("bar", x="sex", y="G3", hue="address"),
    ChartSpec("pie", x="sex", y="Walc"), ChartSpec("pie", x="school"),
    ChartSpec("box", x="freetime", y="G3"), ChartSpec("scatter", x="Dalc", y="G3"),
    ChartSpec("scatter", x="absences", y="G3", hue="sex"), ChartSpec("hist", x="absences"),
    ChartSpec("heatmap"),
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.kind}-{s.x}-{s.y}-{s.hue}")
def test_every_kind_renders_a_png(spec):
    result = render(spec)
    path = Path(result.path)
    assert path.exists() and path.parent == Path(os.environ["CHART_DIR"])
    assert Image.open(path).size[0] > 800
    assert result.title and result.summary
    compile(result.code, "chart", "exec")           # the equivalent code is valid Python


def test_sales_line_and_heatmap():
    line = render(ChartSpec("line", x="month", y="revenue", hue="category", agg="sum"), "online-sales.csv")
    df = STORE.load("online-sales.csv")
    expected = df[df.category == "Electronics"].groupby("month")["revenue"].sum()["2025-11"]
    assert line.summary["values"]["2025-11"]["Electronics"] == pytest.approx(expected, rel=1e-3)
    heat = render(ChartSpec("heatmap", x="month", y="revenue", hue="region", agg="sum"), "online-sales.csv")
    assert set(heat.summary["values"]) == {"East", "North", "South", "West"}
    assert render(ChartSpec("line", x="month", y="orders", agg="sum"), "online-sales.csv").summary["values"]


def test_summary_numbers_match_pandas():
    df = STORE.load()
    bar = render(ChartSpec("bar", x="internet", y="G3")).summary
    assert bar["values"]["yes"] == round(df[df.internet == "yes"].G3.mean(), 3)
    assert bar["n"] == {"no": int((df.internet == "no").sum()), "yes": int((df.internet == "yes").sum())}
    scatter = render(ChartSpec("scatter", x="Medu", y="G3")).summary
    assert scatter["pearson_r"] == round(df.Medu.corr(df.G3), 3) and "mean_by_x" in scatter
    count = render(ChartSpec("count", x="sex")).summary["counts"]
    assert sum(count.values()) == 400 and all(isinstance(v, int) for v in count.values())
    pie = render(ChartSpec("pie", x="sex", y="Walc")).summary
    assert sum(pie["share_pct"].values()) == pytest.approx(100, abs=0.2)


def test_correlation_heatmap_is_symmetric_with_unit_diagonal():
    values = render(ChartSpec("heatmap")).summary["values"]
    assert values["G3"]["G3"] == 1 and values["Medu"]["G3"] == values["G3"]["Medu"]


def test_more_than_three_series_fold_into_other():
    s = pd.Series(list("aaaabbbccd"))
    assert sorted(_fold(s).unique()) == ["Other", "a", "b", "c"]
    assert sorted(_fold(pd.Series(list("aab"))).unique()) == ["a", "b"]


@pytest.mark.parametrize("spec, message", [
    (ChartSpec("radar", x="sex"), "Unknown chart kind"), (ChartSpec("bar", x="sex", y="G3", agg="avg"), "Unknown aggregation"),
    (ChartSpec("bar", x="sex"), "needs: x and y"), (ChartSpec("scatter", x="sex", y="G3"), "not numeric"),
    (ChartSpec("count", x="gender"), "not found"), (ChartSpec("heatmap", x="sex"), "needs"),
])
def test_validation_errors(spec, message):
    with pytest.raises(DatasetError, match=message):
        validate(spec, STORE.load())


def test_custom_title_and_slug():
    result = render(ChartSpec("count", x="sex", title="Students by gender"))
    assert result.title == "Students by gender" and result.path.endswith("count-sex.png")
    assert default_title(validate(ChartSpec("pie", x="school"), STORE.load())) == "Share of rows by school"


def test_equivalent_code_reproduces_the_numbers(tmp_path, monkeypatch, capsys):
    """Run the snippet the tool returns (it's our own template, not LLM output)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    STORE.load().to_csv(tmp_path / "data" / "students.csv", index=False)
    s = validate(ChartSpec("scatter", x="absences", y="G3"), STORE.load())
    exec(code_for(s, "students.csv"), {})
    assert float(capsys.readouterr().out) == pytest.approx(STORE.load().absences.corr(STORE.load().G3))
    assert (tmp_path / "scatter-absences-G3.png").exists()


@pytest.mark.parametrize("value, text", [(5, "5"), (2.5, "2.50"), (1234, "1.2k"), (101_234, "101k"), (2_100_000, "2.1M")])
def test_compact(value, text):
    assert compact(value) == text


def test_kinds_constant():
    assert KINDS == ("count", "bar", "pie", "box", "scatter", "hist", "line", "heatmap")
