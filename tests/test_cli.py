import subprocess
import sys

import pytest

from viz_agent.cli import main


@pytest.fixture(autouse=True)
def no_color(monkeypatch):
    monkeypatch.setattr("viz_agent.cli._COLOR", False)  # plain text even when FORCE_COLOR is set

SCATTERS = "Generate scatter plots of 'Dalc' vs 'G3' and 'Walc' vs 'G3'."


def test_cli_answer(capsys):
    assert main(["Plot the gender count", "-p", "offline"]) == 0
    assert "Number of rows by sex - F: 202, M: 198" in capsys.readouterr().out


def test_cli_trace_with_code(capsys):
    main([SCATTERS, "-p", "offline", "-t", "-c"])
    out = capsys.readouterr().out
    for expected in ["Human", "AI → 2 tools", "make_chart", "scatter-Walc-G3.png", "│ df.plot(kind='scatter'", "AI answer"]:
        assert expected in out


def test_cli_other_dataset(capsys):
    main(["Plot total revenue by month per category", "-p", "offline", "-d", "online-sales.csv"])
    assert "peaks - Electronics" in capsys.readouterr().out


def test_cli_pandas_agent_engine(capsys):
    pytest.importorskip("langchain_experimental")
    main(["pie chart of average Walc for each gender", "-p", "offline", "-e", "pandas-agent", "-t"])
    out = capsys.readouterr().out
    assert "LLM-written code → python_repl_ast" in out and "groupby('sex')['Walc'].mean()" in out


def test_cli_interactive_with_memory(monkeypatch, capsys):
    answers = iter(["Which datasets do I have?", "histogram of absences", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["-p", "offline", "-t"]) == 0
    out = capsys.readouterr().out
    assert "online-sales.csv, students.csv" in out and "hist-absences.png" in out and "bye" in out
    assert "[5] Human" in out                                          # second turn continues the history


def test_cli_unknown_default_dataset_falls_back(capsys):
    main(["Which datasets do I have?", "-p", "offline", "-d", "missing.csv"])
    assert "students.csv" in capsys.readouterr().out


def test_python_dash_m_entry_point():
    out = subprocess.run([sys.executable, "-m", "viz_agent", "--help"], capture_output=True, text=True).stdout
    assert "--engine {agent,pandas-agent}" in out
