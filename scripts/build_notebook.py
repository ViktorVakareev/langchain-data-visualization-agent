"""Generate and execute notebooks/walkthrough.ipynb (offline provider by default).

    python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Data visualization agent — walkthrough\n\n"
       "Chat with a CSV file and get charts. The LLM picks a **typed chart tool** and its columns; Python draws the "
       "chart. No LLM-written code is executed. Runs **offline by default** (rule-based stand-in model: no API key, "
       "no cost). Set `PROVIDER = \"gemini\"` (free key) or `\"openai\"` for a real LLM."),
    code("PROVIDER = \"offline\"  # \"gemini\" | \"openai\" | \"offline\"\n\n"
         "import os, json\nfrom pathlib import Path\nfrom dotenv import load_dotenv\nfrom IPython.display import Image, display\n"
         "load_dotenv()\nos.environ[\"CHART_DIR\"] = \"charts\"\n\n"
         "from viz_agent import (TOOLS, STORE, configure, make_chart, describe_dataset, aggregate,\n"
         "                       build_agent, get_llm, ask, final_answer, tool_calls, charts)\n"
         "from viz_agent.cli import print_trace\n"
         "configure(Path(\"..\") / \"data\" if Path.cwd().name == \"notebooks\" else \"data\", \"students.csv\")\n"
         "df = STORE.load()\nprint(df.shape)\ndf.head(3)"),
    md("## 1. The tools — what the LLM sees\nName + docstring + typed arguments. The model can only *fill in* these "
       "arguments; it can't run arbitrary code."),
    code("for t in TOOLS:\n    print(f\"{t.name:17} args={list(t.args)}\")"),
    md("## 2. Call a chart tool directly (no LLM)\nIt returns the PNG path, the numbers shown, and the equivalent pandas code."),
    code("result = make_chart.invoke({\"kind\": \"bar\", \"x\": \"internet\", \"y\": \"G3\"})\n"
         "print(json.dumps(result[\"summary\"]))\nprint(result[\"code\"])\ndisplay(Image(result[\"path\"], width=520))"),
    md("## 3. The lab's four tasks, asked in plain English"),
    code("agent = build_agent(get_llm(PROVIDER))\n"
         "LAB = [\"Generate a bar chart to plot the gender count.\",\n"
         "       \"Generate a pie chart to display average value of Walc for each Gender.\",\n"
         "       \"Create box plots to analyze the relationship between 'freetime' and 'G3' across different levels of free time.\",\n"
         "       \"Generate scatter plots to examine the correlation between 'Dalc' and 'G3', and between 'Walc' and 'G3'.\"]\n"
         "for q in LAB:\n    response = ask(agent, q)\n    print(\"🤖\", final_answer(response))\n"
         "    for path in charts(response):\n        display(Image(path, width=440))"),
    md("## 4. Exercises\n**Exercise 1** — mother's and father's education vs final grade (two charts from one question: "
       "the model makes **parallel tool calls**)."),
    code("ex1 = ask(agent, \"Generate scatter plots showing the relationship between 'Medu' (mother's education level) and \"\n"
         "                 \"'G3' (final grade), and between 'Fedu' (father's education level) and 'G3'.\")\n"
         "print_trace(ex1[\"messages\"])\nfor path in charts(ex1):\n    display(Image(path, width=440))\n"
         "print(df.groupby(\"Medu\")[\"G3\"].mean().round(2).to_dict())  # check with plain pandas"),
    md("**Exercise 2** — internet access at home vs average final grade."),
    code("ex2 = ask(agent, \"Use bar plots to compare the average final grades ('G3') of students with internet access \"\n"
         "                 \"at home versus those without ('internet' column).\")\n"
         "print(final_answer(ex2))\ndisplay(Image(charts(ex2)[0], width=440))\n"
         "print(aggregate.invoke({\"column\": \"G3\", \"by\": \"internet\"}))  # the same numbers, no chart"),
    md("**Exercise 3** — absences vs final grade, and *the code behind the chart*. In the lab you read the LLM's "
       "generated code from `intermediate_steps`; here every chart tool returns its equivalent code."),
    code("ex3 = ask(agent, \"Plot a scatter plot showing the correlation between the number of absences ('absences') \"\n"
         "                 \"and final grades ('G3') of students.\")\n"
         "print(final_answer(ex3))\nchart = [c for c in tool_calls(ex3) if c[\"name\"] == \"make_chart\"][0]\n"
         "print(chart[\"result\"][\"code\"])\ndisplay(Image(chart[\"result\"][\"path\"], width=440))"),
    md("## 5. Another dataset: online sales over time"),
    code("sales = ask(agent, \"Plot total revenue by month per category\")\nprint(final_answer(sales))\n"
         "display(Image(charts(sales)[0], width=520))"),
    md("## 6. Evaluate the agent: check the charts it chose, not the wording"),
    code("expected = {(\"count\", \"sex\", \"\"), (\"scatter\", \"Medu\", \"G3\"), (\"scatter\", \"Fedu\", \"G3\"), (\"bar\", \"internet\", \"G3\")}\n"
         "first = ask(agent, LAB[0])\n"
         "actual = {(c[\"args\"][\"kind\"], c[\"args\"].get(\"x\", \"\"), c[\"args\"].get(\"y\", \"\"))\n"
         "          for r in (first, ex1, ex2) for c in tool_calls(r) if c[\"name\"] == \"make_chart\"}\n"
         "for item in sorted(expected):\n    print(\"✅\" if item in actual else \"❌\", item)\n"
         "print(f\"\\n{len(expected & actual)}/{len(expected)} checks passed\")"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=240, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, ROOT / "notebooks" / "walkthrough.ipynb")
print("wrote notebooks/walkthrough.ipynb")
