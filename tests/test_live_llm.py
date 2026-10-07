"""Live evaluation with a real LLM - skipped unless an API key is set.  pytest -m live -v

Checks WHICH chart was made from WHICH columns (and that the PNG exists), not the wording.
"""

import os
from pathlib import Path

import pytest

from viz_agent import ask, build_agent, get_llm, tool_calls

PROVIDERS = [
    pytest.param("gemini", marks=pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")),
    pytest.param("openai", marks=pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")),
]
pytestmark = pytest.mark.live
CASES = [
    ("Generate a bar chart to plot the gender count.", [("count", "sex", "")]),
    ("Generate scatter plots showing the relationship between 'Medu' and 'G3', and between 'Fedu' and 'G3'.",
     [("scatter", "Medu", "G3"), ("scatter", "Fedu", "G3")]),
    ("Use bar plots to compare the average final grades ('G3') of students with and without internet access "
     "('internet' column).", [("bar", "internet", "G3")]),
]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("question, expected", CASES)
def test_lab_questions(provider, question, expected):
    response = ask(build_agent(get_llm(provider)), question)
    made = [c for c in tool_calls(response) if c["name"] == "make_chart" and "path" in (c["result"] or {})]
    got = {(c["args"]["kind"], c["args"].get("x", ""), c["args"].get("y", "")) for c in made}
    assert set(expected) <= got
    assert all(Path(c["result"]["path"]).exists() for c in made)
