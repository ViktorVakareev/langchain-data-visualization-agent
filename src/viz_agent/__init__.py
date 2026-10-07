"""Data visualization agent - chat with your CSV files and get charts, built on LangChain tool calling."""

from .agent import SYSTEM_PROMPT, ask, build_agent, charts, final_answer, get_llm, tool_calls
from .charts import ChartResult, ChartSpec, render
from .datasets import STORE, DatasetError, DatasetStore, configure
from .tools import TOOLS, aggregate, correlation, describe_dataset, filter_rows, list_datasets, make_chart

__version__ = "1.0.0"

__all__ = [
    "STORE", "SYSTEM_PROMPT", "TOOLS", "ChartResult", "ChartSpec", "DatasetError", "DatasetStore",
    "aggregate", "ask", "build_agent", "charts", "configure", "correlation", "describe_dataset", "filter_rows",
    "final_answer", "get_llm", "list_datasets", "make_chart", "render", "tool_calls",
]
