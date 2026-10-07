"""The lab's three exercises with the typed-tools agent - and a pandas check of every answer.

    python examples/lab_exercises.py offline      # or gemini / openai
"""

import sys

from dotenv import load_dotenv

from viz_agent import STORE, ask, build_agent, configure, final_answer, get_llm, tool_calls

load_dotenv()
configure("data", "students.csv")
agent = build_agent(get_llm(sys.argv[1] if len(sys.argv) > 1 else None))
df = STORE.load()

EXERCISES = {
    "Exercise 1 - parents' education vs final grade":
        "Generate scatter plots showing the relationship between 'Medu' (mother's education level) and 'G3' "
        "(final grade), and between 'Fedu' (father's education level) and 'G3'.",
    "Exercise 2 - internet access vs average final grade":
        "Use bar plots to compare the average final grades ('G3') of students with internet access at home "
        "versus those without ('internet' column).",
    "Exercise 3 - absences vs final grade (and the code behind the chart)":
        "Plot a scatter plot showing the correlation between the number of absences ('absences') and final "
        "grades ('G3') of students.",
}

for title, question in EXERCISES.items():
    print(f"\n=== {title}")
    response = ask(agent, question)
    print("🤖", final_answer(response))
    for call in tool_calls(response):
        if call["name"] == "make_chart" and "code" in (call["result"] or {}):
            print(f"🖼  {call['result']['path']}")
            if title.startswith("Exercise 3"):
                print("   equivalent code:\n   " + call["result"]["code"].replace("\n", "\n   "))

print("\n=== Check with plain pandas")
print("mean G3 by Medu:", df.groupby("Medu")["G3"].mean().round(2).to_dict())
print("mean G3 by Fedu:", df.groupby("Fedu")["G3"].mean().round(2).to_dict())
print("mean G3 by internet:", df.groupby("internet")["G3"].mean().round(2).to_dict())
print("corr(absences, G3):", round(df["absences"].corr(df["G3"]), 3))
