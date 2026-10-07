import uuid
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from viz_agent import configure

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(autouse=True)
def sample_data(tmp_path, monkeypatch):
    """Every test: the sample CSVs, students.csv as default, charts written to a temp folder."""
    monkeypatch.setenv("CHART_DIR", str(tmp_path / "charts"))
    configure(DATA, "students.csv")
    yield
    configure(DATA, None)


class ScriptedToolModel(GenericFakeChatModel):
    """Fake chat model replaying scripted AIMessages (incl. tool calls); records what it was sent."""

    received: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def calls(*specs):
    """calls(("make_chart", {...}), ...) -> one AIMessage with these tool calls."""
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:6]}"} for n, a in specs])


@pytest.fixture
def scripted_model():
    def make(*messages, **kwargs):
        return ScriptedToolModel(messages=iter(messages), received=[], **kwargs)
    return make
