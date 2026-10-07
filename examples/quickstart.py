"""Ask the lab's four chart questions and print the answers + chart files.

    python examples/quickstart.py offline        # or gemini / openai (default: $LLM_PROVIDER)
"""

import sys

from dotenv import load_dotenv

from viz_agent import ask, build_agent, charts, configure, final_answer, get_llm

load_dotenv()
configure("data", "students.csv")
agent = build_agent(get_llm(sys.argv[1] if len(sys.argv) > 1 else None))

QUESTIONS = [
    "Generate a bar chart to plot the gender count.",
    "Generate a pie chart to display average value of Walc for each Gender.",
    "Create box plots to analyze the relationship between 'freetime' and 'G3' across different levels of free time.",
    "Generate scatter plots to examine the correlation between 'Dalc' and 'G3', and between 'Walc' and 'G3'.",
]
for question in QUESTIONS:
    response = ask(agent, question)
    print(f"❓ {question}\n🤖 {final_answer(response)}\n🖼  {', '.join(charts(response)) or '-'}\n")
