"""create_agent orchestration with a scripted fake model; provider wiring; schema conversion."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from conftest import calls
from viz_agent import TOOLS, ask, build_agent, charts, final_answer, get_llm, tool_calls
from viz_agent.agent import SYSTEM_PROMPT, parse
from viz_agent.offline import OfflineVizModel


def test_loop_message_order_and_parallel_charts(scripted_model):
    model = scripted_model(
        calls(("make_chart", {"kind": "scatter", "x": "Medu", "y": "G3"}),
              ("make_chart", {"kind": "scatter", "x": "Fedu", "y": "G3"})),
        AIMessage(content="Both are weak positive."))
    response = ask(build_agent(model), "Medu and Fedu vs G3")
    types = [type(m).__name__ for m in response["messages"]]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage", "ToolMessage", "AIMessage"]
    assert [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in charts(response)] == \
        ["scatter-Medu-G3.png", "scatter-Fedu-G3.png"]
    assert final_answer(response) == "Both are weak positive."
    assert SYSTEM_PROMPT in str(model.received[0][0].content)        # system prompt sent first


def test_tool_error_goes_back_to_the_model(scripted_model):
    model = scripted_model(calls(("make_chart", {"kind": "bar", "x": "gender", "y": "G3"})),
                           calls(("make_chart", {"kind": "bar", "x": "sex", "y": "G3"})),
                           AIMessage(content="Fixed the column name."))
    response = ask(build_agent(model), "average grade by gender")
    made = tool_calls(response)
    assert "not found" in made[0]["result"]["error"] and made[1]["result"]["path"].endswith("bar-sex-G3-mean.png")
    tool_msgs = [m for m in model.received[1] if isinstance(m, ToolMessage)]
    assert "Columns: school, sex" in tool_msgs[0].content           # the model saw the valid columns


def test_history_keeps_context(scripted_model):
    model = scripted_model(calls(("aggregate", {"column": "G3", "by": "internet"})), AIMessage(content="yes is higher"),
                           AIMessage(content="By 0.4 points."))
    agent = build_agent(model)
    first = ask(agent, "average G3 by internet")
    second = ask(agent, "by how much?", history=first["messages"])
    assert tool_calls(second) == [] and final_answer(second) == "By 0.4 points."
    assert len(tool_calls(second, since_last_human=False)) == 1


def test_helpers():
    assert final_answer({"messages": [AIMessage(content=[{"type": "text", "text": "a"}, "b"])]}) == "ab"
    assert parse('{"a": 1}') == {"a": 1} and parse("plain") == "plain"


def test_get_llm_providers(monkeypatch):
    assert isinstance(get_llm("offline"), OfflineVizModel)
    with pytest.raises(ValueError):
        get_llm("nope")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert type(get_llm("gemini")).__name__ == "ChatGoogleGenerativeAI"
    assert type(get_llm("openai")).__name__ == "ChatOpenAI"
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    assert isinstance(get_llm(), OfflineVizModel)


def test_tool_schemas_convert_for_gemini_and_openai():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations

    declared = [f.name for t in convert_to_genai_function_declarations(TOOLS) for f in (t.function_declarations or [])]
    assert declared == [t.name for t in TOOLS]
    params = convert_to_openai_tool(TOOLS[-1])["function"]["parameters"]
    assert params["required"] == ["kind"] and set(params["properties"]) >= {"x", "y", "hue", "agg"}
