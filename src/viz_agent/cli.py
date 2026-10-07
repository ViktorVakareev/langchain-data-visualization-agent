"""Command line: chat with your CSV files and get charts.

    viz-agent "Plot the gender count" -p offline --trace
    viz-agent "scatter plots of Dalc vs G3 and Walc vs G3" -p offline -t --code
    viz-agent "Plot total revenue by month per category" -d online-sales.csv
    viz-agent "pie chart of average Walc for each gender" -e pandas-agent -t   # the lab's original agent
    viz-agent -p gemini                                  # interactive chat ('exit' to quit)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .agent import PROVIDERS, ask, build_agent, final_answer, get_llm, parse
from .datasets import STORE, configure

_COLOR = sys.stdout.isatty() or bool(os.getenv("FORCE_COLOR"))
PAD = " " * 16


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _short(text: str, limit: int = 300) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f" … (+{len(text) - limit} chars)"


def _tool_body(content: str, show_code: bool) -> str:
    result = parse(content)
    if isinstance(result, dict) and "path" in result and "code" in result:  # a chart
        body = f"🖼  {result['path']}\n{PAD}{_short(json.dumps(result['summary']), 220)}"
        if show_code:
            body += "\n" + textwrap.indent(result["code"], PAD + _c("2", "│ "), lambda _: True)
        return body
    return textwrap.indent(_short(content), PAD).lstrip()


def print_trace(messages: list, start: int = 0, show_code: bool = False) -> None:
    """Print each step of the agent loop: question, tool calls, tool results, answer."""
    for i, msg in enumerate(messages[start:], start=start + 1):
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            print(f"{_c('1;36', f'[{i}] Human')}       {msg.content}")
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            label = "AI → tool" if len(msg.tool_calls) == 1 else f"AI → {len(msg.tool_calls)} tools"
            for call in msg.tool_calls:
                print(f"{_c('1;33', f'[{i}] {label:<10}')}  {_c('33', call['name'])}({call['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"{_c('1;35', f'[{i}] Tool')}        {_c('35', msg.name or '')} → {_tool_body(msg.content, show_code)}")
        elif isinstance(msg, AIMessage):
            print(f"{_c('1;32', f'[{i}] AI answer')}   {final_answer({'messages': [msg]})}")


def print_lab_steps(response: dict) -> None:
    """The lab's view: every code block the LLM wrote and what running it printed."""
    for i, (action, observation) in enumerate(response.get("intermediate_steps", []), start=1):
        code = action.tool_input.get("query", "") if isinstance(action.tool_input, dict) else str(action.tool_input)
        print(_c("1;33", f"[step {i}] LLM-written code → {action.tool}"))
        print(textwrap.indent(code.replace("; ", "\n"), _c("2", "  │ "), lambda _: True))
        print(_c("1;35", "  output: ") + textwrap.indent(_short(str(observation).strip()), " " * 10).lstrip())


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="viz-agent", description="Data visualization agent (LangChain tool calling)")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive chat")
    parser.add_argument("-p", "--provider", choices=PROVIDERS, default=os.getenv("LLM_PROVIDER", "gemini"))
    parser.add_argument("-m", "--model", default=None, help="override the model name")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"), help="folder with CSV files")
    parser.add_argument("-d", "--dataset", default=os.getenv("DEFAULT_DATASET", "students.csv"),
                        help="default CSV file (the agent can still use the others)")
    parser.add_argument("-e", "--engine", choices=("agent", "pandas-agent"), default="agent",
                        help="agent = typed tools + create_agent; pandas-agent = the lab's code-executing agent")
    parser.add_argument("-t", "--trace", action="store_true", help="show every tool call and result")
    parser.add_argument("-c", "--code", action="store_true", help="with --trace: show the pandas code for each chart")
    args = parser.parse_args(argv)

    configure(args.data_dir)
    STORE.default = args.dataset if args.dataset in STORE.list_files() else None

    if args.engine == "pandas-agent":
        from .lab_pattern import build_pandas_agent, lab_llm

        file_name = STORE.resolve(args.dataset or None)
        lab_agent = build_pandas_agent(lab_llm(args.provider, file_name), STORE.load(file_name))

        def run(question: str) -> None:
            response = lab_agent.invoke({"input": question})
            if args.trace:
                print(f"{_c('1;36', 'Human')}  {question}")
                print_lab_steps(response)
            print(_c("1;32", "Agent: ") + str(response["output"]))
    else:
        agent = build_agent(get_llm(args.provider, args.model))
        history: list = []

        def run(question: str) -> None:
            nonlocal history
            response = ask(agent, question, history)
            if args.trace:
                print_trace(response["messages"], start=len(history), show_code=args.code)
            else:
                print(_c("1;32", "Agent: ") + final_answer(response))
            history = response["messages"]

    if args.question:
        run(" ".join(args.question))
        return 0

    files = ", ".join(STORE.list_files())
    print(f"📊 Data visualization agent {_c('2', f'(provider: {args.provider}, engine: {args.engine}, data: {files})')}"
          " - type 'exit' to quit")
    while True:
        try:
            question = input(_c("1;36", "\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        run(question)
    print("bye")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
