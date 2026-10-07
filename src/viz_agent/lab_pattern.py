"""The lab's original pattern: ``create_pandas_dataframe_agent`` (langchain-experimental).

That agent gives the LLM a Python REPL tool (``python_repl_ast``) with ``df`` preloaded,
and **executes whatever code the model writes** - which is why LangChain makes you pass
``allow_dangerous_code=True``. It's kept here so you can compare it with the typed tools
in ``tools.py``; don't point it at data or machines you care about with an untrusted model.

``OfflineCodeModel`` stands in for the LLM: it plans the chart with the same rules as the
offline model, then *writes pandas code* (as the lab's Llama model did) instead of calling
``make_chart``. So the classic agent, the REPL tool and ``intermediate_steps`` all run for real.

    pip install -e ".[lab]"
"""

from __future__ import annotations

import os
import re
import uuid
import warnings
from typing import Any, Optional

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .charts import ChartSpec, _slug, chart_dir, default_title, validate
from .datasets import STORE
from .offline import chart_specs, find_columns

LAB_PREFIX = ("You are working with a pandas dataframe called `df`. Use the python_repl_ast tool to run Python code. "
              "Use matplotlib for charts, save them with plt.savefig and print the numbers you used.")


def build_pandas_agent(llm: BaseChatModel, df: pd.DataFrame, **kwargs: Any):
    """The lab's agent, with a tool-calling LLM (works with Gemini, OpenAI and the offline stand-in)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # langchain-experimental is being sunset
        from langchain_experimental.agents import create_pandas_dataframe_agent

    return create_pandas_dataframe_agent(
        llm, df, agent_type="tool-calling", prefix=LAB_PREFIX, verbose=kwargs.pop("verbose", False),
        return_intermediate_steps=True,
        # the lab passed handle_parsing_errors=True directly; newer versions ignore that (with a warning)
        agent_executor_kwargs={"handle_parsing_errors": True, **kwargs.pop("agent_executor_kwargs", {})},
        allow_dangerous_code=True,  # the agent runs LLM-written Python - see the module docstring
        **kwargs)


def show_code(response: dict) -> str:
    """All code the agent ran, one block per step (the lab used [-1], which fails if there were no steps)."""
    blocks = []
    for action, _ in response.get("intermediate_steps", []):
        code = action.tool_input.get("query", "") if isinstance(action.tool_input, dict) else str(action.tool_input)
        blocks.append(code.replace("; ", "\n"))
    return "\n\n".join(blocks)


def repl_code(spec: ChartSpec, df: pd.DataFrame) -> str:
    """Code the lab's LLM would write for this chart (``df`` already exists in the REPL)."""
    s = validate(spec, df)
    path = (chart_dir() / f"lab-{_slug(s)}.png").as_posix()
    plot, numbers = {
        "count": (f"df['{s.x}'].value_counts().sort_index().plot(kind='bar')", f"df['{s.x}'].value_counts().sort_index()"),
        "bar": (f"df.groupby('{s.x}')['{s.y}'].{s.agg}().plot(kind='bar')", f"df.groupby('{s.x}')['{s.y}'].{s.agg}()"),
        "pie": (f"df.groupby('{s.x}')['{s.y}'].{s.agg}().plot(kind='pie', autopct='%1.1f%%')" if s.y
                else f"df['{s.x}'].value_counts().plot(kind='pie', autopct='%1.1f%%')",
                f"df.groupby('{s.x}')['{s.y}'].{s.agg}()" if s.y else f"df['{s.x}'].value_counts()"),
        "box": (f"df.boxplot(column='{s.y}', by='{s.x}')", f"df.groupby('{s.x}')['{s.y}'].median()"),
        "scatter": (f"df.plot(kind='scatter', x='{s.x}', y='{s.y}')", f"df['{s.x}'].corr(df['{s.y}'])"),
        "hist": (f"df['{s.x}'].plot(kind='hist', bins=20)", f"df['{s.x}'].describe()"),
        "line": (f"df.pivot_table(index='{s.x}', columns='{s.hue}', values='{s.y}', aggfunc='{s.agg}').plot()" if s.hue
                 else f"df.groupby('{s.x}')['{s.y}'].{s.agg}().plot()",
                 f"df.groupby('{s.x}')['{s.y}'].{s.agg}().tail()"),
        "heatmap": ("plt.imshow(df.select_dtypes('number').corr(), cmap='RdBu', vmin=-1, vmax=1); plt.colorbar()",
                    "df.select_dtypes('number').corr().round(2).iloc[:3, :3]"),
    }[s.kind]
    return "\n".join(["import matplotlib.pyplot as plt", plot, f"plt.title({default_title(s)!r})",
                      f"plt.savefig('{path}')", "plt.close('all')", f"print({numbers})"])


class OfflineCodeModel(BaseChatModel):
    """Rule-based stand-in that writes pandas code for the python_repl_ast tool. Not an LLM."""

    file_name: str = ""

    @property
    def _llm_type(self) -> str:
        return "offline-code-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineCodeModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        question = str([m for m in messages if isinstance(m, HumanMessage)][-1].content)
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if observations:
            text = "\n".join(str(o.content).strip() for o in observations)
            n = len(observations)
            return _result(AIMessage(content=f"I wrote and ran {n} code block{'s' * (n > 1)}. It printed:\n{text}"))
        file_name, cols = find_columns(question)
        df = STORE.load(self.file_name or file_name)
        cols = [c for c in cols if c in df.columns]
        if not cols and not re.search(r"correlat", question, re.I):
            return _result(AIMessage(content="", tool_calls=[_call("df.shape")]))
        specs = chart_specs(question, cols, df)
        return _result(AIMessage(content="", tool_calls=[_call(repl_code(ChartSpec(**s), df)) for s in specs]))


def _call(code: str) -> dict:
    return {"name": "python_repl_ast", "args": {"query": code}, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def _result(message: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=message)])


def lab_llm(provider: Optional[str] = None, file_name: str = "") -> BaseChatModel:
    """Like get_llm(), but the offline provider writes code instead of calling make_chart."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "offline":
        return OfflineCodeModel(file_name=file_name)
    from .agent import get_llm

    return get_llm(provider)
