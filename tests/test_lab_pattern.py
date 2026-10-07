"""The lab's original agent: create_pandas_dataframe_agent (langchain-experimental) runs LLM-written code."""

import pytest

pytest.importorskip("langchain_experimental")

from langchain_core.messages import AIMessage  # noqa: E402

from conftest import calls  # noqa: E402
from viz_agent import STORE, ChartSpec  # noqa: E402
from viz_agent.lab_pattern import OfflineCodeModel, build_pandas_agent, lab_llm, repl_code, show_code  # noqa: E402
from viz_agent.offline import OfflineVizModel  # noqa: E402


def test_pandas_agent_writes_and_runs_code_for_two_charts(tmp_path):
    agent = build_pandas_agent(OfflineCodeModel(file_name="students.csv"), STORE.load())
    response = agent.invoke({"input": "Generate scatter plots of 'Dalc' vs 'G3' and 'Walc' vs 'G3'."})
    steps = response["intermediate_steps"]
    assert [a.tool for a, _ in steps] == ["python_repl_ast", "python_repl_ast"]
    assert float(steps[0][1]) == pytest.approx(STORE.load().Dalc.corr(STORE.load().G3))
    assert "df.plot(kind='scatter', x='Dalc', y='G3')" in show_code(response)
    assert (tmp_path / "charts" / "lab-scatter-Walc-G3.png").exists()


def test_rows_question_runs_df_shape():
    response = build_pandas_agent(OfflineCodeModel(), STORE.load()).invoke({"input": "how many rows are there?"})
    assert show_code(response) == "df.shape" and "400" in response["output"]


def test_the_risk_any_code_the_model_writes_is_executed(scripted_model):
    """Why the typed tools exist: the REPL runs whatever the LLM sends (harmless here)."""
    model = scripted_model(calls(("python_repl_ast", {"query": "__import__('os').getcwd()"})),
                           AIMessage(content="done"), disable_streaming=True)  # AgentExecutor streams
    response = build_pandas_agent(model, STORE.load()).invoke({"input": "anything"})
    assert "/" in str(response["intermediate_steps"][0][1]) or "\\" in str(response["intermediate_steps"][0][1])


def test_show_code_handles_no_steps_and_string_input():
    class Action:
        tool_input = "df.head(); df.tail()"
    assert show_code({"intermediate_steps": []}) == ""        # the lab's [-1] would raise IndexError here
    assert show_code({"intermediate_steps": [(Action(), "")]}) == "df.head()\ndf.tail()"


@pytest.mark.parametrize("spec", [ChartSpec("count", x="sex"), ChartSpec("bar", x="internet", y="G3"),
                                  ChartSpec("pie", x="sex"), ChartSpec("pie", x="sex", y="Walc"),
                                  ChartSpec("box", x="freetime", y="G3"), ChartSpec("hist", x="age"),
                                  ChartSpec("line", x="age", y="G3"), ChartSpec("heatmap")])
def test_repl_code_is_valid_python(spec):
    code = repl_code(spec, STORE.load())
    compile(code, "repl", "exec")
    assert code.startswith("import matplotlib.pyplot as plt") and "print(" in code


def test_lab_llm():
    assert isinstance(lab_llm("offline", "students.csv"), OfflineCodeModel)
    assert isinstance(lab_llm("offline"), OfflineCodeModel) and not isinstance(lab_llm("offline"), OfflineVizModel)
