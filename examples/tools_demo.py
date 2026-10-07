"""Call the tools directly - no LLM, no API key. This is exactly what the agent can do.

    python examples/tools_demo.py
"""

import json

from viz_agent import aggregate, configure, correlation, describe_dataset, filter_rows, list_datasets, make_chart

configure("data", "students.csv")


def show(label, result):
    print(f"▶ {label}\n  {json.dumps(result, default=str)[:300]}")


show("list_datasets()", list_datasets.invoke({}))
cols = describe_dataset.invoke({"file_name": "students.csv"})
show("describe_dataset('students.csv') - columns", [c["name"] for c in cols["columns"]])
show("aggregate(G3 by internet)", aggregate.invoke({"column": "G3", "by": "internet"}))
show("correlation(absences, G3)", correlation.invoke({"x": "absences", "y": "G3"}))
show("filter_rows(age > 18)", {k: v for k, v in filter_rows.invoke({"column": "age", "op": ">", "value": "18"}).items()
                                if k != "preview"})
chart = make_chart.invoke({"kind": "bar", "x": "internet", "y": "G3"})
show("make_chart(bar, internet, G3)", {"path": chart["path"], "summary": chart["summary"]})
print("  code:\n    " + chart["code"].replace("\n", "\n    "))
show("make_chart(bar, gender, G3)  <- wrong column: an error message, not an exception",
     make_chart.invoke({"kind": "bar", "x": "gender", "y": "G3"}))
show("filter_rows(age, 'drop', ...)  <- not on the allow-list",
     filter_rows.invoke({"column": "age", "op": "drop", "value": "1"}))
