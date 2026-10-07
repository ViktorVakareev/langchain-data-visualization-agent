"""Offline rule-based model: question -> tool calls, and the full create_agent loop."""

import pytest

from viz_agent import ask, build_agent, charts, final_answer, tool_calls
from viz_agent.offline import HELP, OfflineVizModel, describe_result, find_columns, plan

LAB = [
    ("Generate a bar chart to plot the gender count.", [{"kind": "count", "x": "sex"}]),
    ("Generate a pie chart to display average value of Walc for each Gender.",
     [{"kind": "pie", "x": "sex", "y": "Walc", "agg": "mean"}]),
    ("Create box plots to analyze the relationship between 'freetime' (amount of free time) and 'G3' (final grade) "
     "across different levels of free time.", [{"kind": "box", "x": "freetime", "y": "G3"}]),
    ("Generate scatter plots to examine the correlation between 'Dalc' (daily alcohol consumption) and 'G3', "
     "and between 'Walc' (weekend alcohol consumption) and 'G3'.",
     [{"kind": "scatter", "x": "Dalc", "y": "G3"}, {"kind": "scatter", "x": "Walc", "y": "G3"}]),
    # Exercises 1-3
    ("Generate scatter plots showing the relationship between 'Medu' (mother's education level) and 'G3' (final grade), "
     "and between 'Fedu' (father's education level) and 'G3'.",
     [{"kind": "scatter", "x": "Medu", "y": "G3"}, {"kind": "scatter", "x": "Fedu", "y": "G3"}]),
    ("Use bar plots to compare the average final grades ('G3') of students with internet access at home versus "
     "those without ('internet' column).", [{"kind": "bar", "x": "internet", "y": "G3", "agg": "mean"}]),
    ("Plot a scatter plot showing the correlation between the number of absences ('absences') and final grades "
     "('G3') of students.", [{"kind": "scatter", "x": "absences", "y": "G3"}]),
]


@pytest.mark.parametrize("question, specs", LAB, ids=[f"lab-{i}" for i in range(len(LAB))])
def test_plan_lab_questions(question, specs):
    assert plan(question) == [("make_chart", {**s, "file_name": "students.csv"}) for s in specs]


@pytest.mark.parametrize("question, expected", [
    ("Which datasets do I have?", ("list_datasets", {})),
    ("how many rows of data are in this file?", ("describe_dataset", {"file_name": "students.csv"})),
    ("Give me all the data where student's age is over 18 years old.",
     ("filter_rows", {"column": "age", "op": ">", "value": "18", "file_name": "students.csv"})),
    ("What is the average G3 by internet?",
     ("aggregate", {"column": "G3", "by": "internet", "agg": "mean", "file_name": "students.csv"})),
    ("total revenue", ("aggregate", {"column": "revenue", "agg": "sum", "file_name": "online-sales.csv"})),
    ("correlation between absences and G3", ("correlation", {"x": "absences", "y": "G3", "file_name": "students.csv"})),
    ("Plot total revenue by month per category", ("make_chart", {
        "kind": "line", "x": "month", "y": "revenue", "hue": "category", "agg": "sum", "file_name": "online-sales.csv"})),
    ("Show a heatmap of total revenue by region and month", ("make_chart", {
        "kind": "heatmap", "x": "month", "y": "revenue", "hue": "region", "agg": "sum", "file_name": "online-sales.csv"})),
    ("Draw a correlation heatmap of the student data", ("make_chart", {"kind": "heatmap", "file_name": "students.csv"})),
    ("histogram of absences", ("make_chart", {"kind": "hist", "x": "absences", "file_name": "students.csv"})),
    ("bar chart of average G3 by sex and address", ("make_chart", {
        "kind": "bar", "x": "sex", "y": "G3", "agg": "mean", "hue": "address", "file_name": "students.csv"})),
    ("plot something nice", ("describe_dataset", {"file_name": "students.csv"})),
])
def test_plan_other_questions(question, expected):
    assert plan(question)[0] == expected


def test_find_columns_uses_synonyms_and_order():
    assert find_columns("final grade by gender") == ("students.csv", ["G3", "sex"])
    assert find_columns("orders per region")[0] == "online-sales.csv"


@pytest.mark.parametrize("question, n_charts, answer_part", [
    (LAB[3][0], 2, "G3 vs Dalc: r = -0.151 (weak negative relationship)"),
    (LAB[4][0], 2, "Average G3 goes from 8.44 at Medu=0 to 12.05 at Medu=4"),
    (LAB[5][0], 1, "Average G3 by internet - no: 10.55, yes: 10.95"),
    (LAB[1][0], 1, "%)"),
    (LAB[2][0], 1, "Median G3 by freetime"),
    ("Plot total revenue by month per category", 1, "peaks - Electronics 2025-11"),
    ("Show a heatmap of total revenue by region and month", 1, "highest cell is North in 2025-11"),
    ("correlation heatmap", 1, "Strongest correlations: G1-G2 +0.94, G2-G3 +0.94"),
    ("histogram of absences", 1, "median 5"),
    ("plot the count of students by school and sex", 1, "River Academy: F"),
])
def test_offline_agent_end_to_end(question, n_charts, answer_part):
    response = ask(build_agent(OfflineVizModel()), question)
    assert len(charts(response)) == n_charts
    assert answer_part in final_answer(response)


@pytest.mark.parametrize("question, answer_part", [
    ("Which datasets do I have?", "online-sales.csv, students.csv"),
    ("describe the students data", "400 rows and 17 columns"),
    ("How many students are older than 18?", "28 of 400 rows (7.0%) match age > 18"),
    ("correlation between Medu and G3", "weak positive relationship"),
    ("What is the average G3?", "The mean of G3 is"),
    ("hello", "Offline demo model"),
])
def test_offline_answers_without_charts(question, answer_part):
    response = ask(build_agent(OfflineVizModel()), question)
    assert answer_part in final_answer(response) and charts(response) == []


def test_tool_errors_are_explained():
    assert describe_result("make_chart", {}, {"error": "Column 'x' not found."}) == "I couldn't do that: Column 'x' not found."
    assert "no CSV files" in describe_result("list_datasets", {}, [])


def test_help_mentions_lab_examples():
    assert "gender count" in HELP and "correlation heatmap" in HELP


def test_parallel_calls_share_one_ai_message():
    response = ask(build_agent(OfflineVizModel()), LAB[3][0])
    ai_with_calls = [m for m in response["messages"] if getattr(m, "tool_calls", None)]
    assert len(ai_with_calls) == 1 and len(ai_with_calls[0].tool_calls) == 2
    assert [c["name"] for c in tool_calls(response)] == ["make_chart", "make_chart"]
