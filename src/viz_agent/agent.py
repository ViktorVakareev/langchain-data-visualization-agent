"""Provider switch, the LangChain 1.x agent (create_agent) and response helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage

from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are a data visualization assistant. You answer questions about CSV datasets by calling tools - "
    "never invent numbers and never write code yourself. "
    "If you don't know the exact column names, call describe_dataset first (and list_datasets if the file is unclear). "
    "To plot, call make_chart with exact column names; when the user asks for several relationships "
    "(e.g. 'A vs G3 and B vs G3'), make one chart per pair. "
    "Pick the chart kind that fits: count/bar for categories, pie for shares, box for distributions per group, "
    "scatter for two numeric columns, line for trends over time, heatmap for two categories or correlations. "
    "After charting, explain what the numbers in the tool result show in 2-4 plain sentences "
    "(e.g. which group is higher, the correlation and whether it is weak or strong) and give the chart path."
)

PROVIDERS = ("gemini", "openai", "offline")
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
    """Build the chat model. ``provider`` defaults to $LLM_PROVIDER, then "gemini".

    - gemini:  free API key from Google AI Studio in $GOOGLE_API_KEY ($GEMINI_MODEL)
    - openai:  $OPENAI_API_KEY ($OPENAI_MODEL), billed per token
    - offline: rule-based stand-in, no key, no network, no cost
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODELS["gemini"]),
                                      temperature=0, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"]), temperature=0, **kwargs)
    if provider == "offline":
        from .offline import OfflineVizModel

        return OfflineVizModel()
    raise ValueError(f"Unknown provider {provider!r}. Choose one of {PROVIDERS}.")


def build_agent(llm: Optional[BaseChatModel] = None, tools: Optional[Sequence] = None,
                system_prompt: str = SYSTEM_PROMPT):
    """Create the tool-calling agent: a LangGraph loop of model -> tools -> model ..."""
    return create_agent(model=llm or get_llm(), tools=list(tools or TOOLS), system_prompt=system_prompt)


def ask(agent, question: str, history: Optional[list] = None) -> dict:
    """Run one user turn. Pass the previous ``response["messages"]`` as history to keep context."""
    return agent.invoke({"messages": [*(history or []), ("human", question)]})


def final_answer(response: dict) -> str:
    content = response["messages"][-1].content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def tool_calls(response: dict, since_last_human: bool = True) -> list[dict]:
    """Tool calls made in the response: [{"name", "args", "result"}, ...]."""
    messages = response["messages"]
    if since_last_human:
        last_human = max(i for i, m in enumerate(messages) if m.type == "human")
        messages = messages[last_human:]
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    calls = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_msg = results.get(call["id"])
            calls.append({"name": call["name"], "args": call["args"],
                          "result": parse(tool_msg.content) if tool_msg else None})
    return calls


def charts(response: dict) -> list[str]:
    """Paths of the PNG files the agent created in the last turn."""
    return [c["result"]["path"] for c in tool_calls(response)
            if c["name"] == "make_chart" and isinstance(c["result"], dict) and "path" in c["result"]]


def parse(content: Any) -> Any:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content
