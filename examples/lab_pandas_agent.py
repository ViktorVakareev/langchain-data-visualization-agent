"""The lab's original agent: create_pandas_dataframe_agent (langchain-experimental).

The model writes Python, and the agent EXECUTES it in a REPL with `df` loaded
(hence allow_dangerous_code=True). Compare with the typed tools used by viz-agent.

    pip install -e ".[lab]"
    python examples/lab_pandas_agent.py offline      # or gemini / openai
"""

import sys

from dotenv import load_dotenv

from viz_agent import STORE, configure
from viz_agent.lab_pattern import build_pandas_agent, lab_llm, show_code

load_dotenv()
configure("data", "students.csv")
agent = build_pandas_agent(lab_llm(sys.argv[1] if len(sys.argv) > 1 else None, "students.csv"), STORE.load())

for question in ["how many rows of data are in this file?",
                 "Generate a pie chart to display average value of Walc for each Gender.",
                 "Generate scatter plots to examine the correlation between 'Dalc' and 'G3', and between 'Walc' and 'G3'."]:
    response = agent.invoke({"input": question})
    print(f"❓ {question}")
    print("🧑‍💻 code the LLM wrote (and the agent executed):")
    print("   " + show_code(response).replace("\n", "\n   "))
    print(f"🤖 {response['output']}\n")
