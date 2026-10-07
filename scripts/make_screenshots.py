"""Regenerate the README images in docs/images/ (offline, reproducible).

    pip install -e ".[all]" ansi2html nbconvert playwright
    python scripts/make_screenshots.py

Every terminal screenshot is the *real* output of the command shown in its title bar,
run with the offline provider so no API key is needed. The chart gallery is rendered
by the same make_chart engine the agent uses.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ansi2html import Ansi2HTMLConverter
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
GALLERY = OUT / "charts"
GALLERY.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "FORCE_COLOR": "1", "PY_COLORS": "1", "LLM_PROVIDER": "offline", "DATA_DIR": "data",
       "CHART_DIR": "charts", "COLUMNS": "120", "PYTHONWARNINGS": "ignore"}

TERMINAL_CSS = """
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.win{display:inline-block;min-width:860px;max-width:1180px;border-radius:10px;overflow:hidden;
     box-shadow:0 12px 32px rgba(15,23,42,.28);background:#0f172a}
.bar{background:#1e293b;padding:10px 14px;display:flex;align-items:center;gap:8px}
.dot{width:12px;height:12px;border-radius:50%}
.title{color:#94a3b8;font-size:13px;margin-left:10px;font-family:ui-monospace,Consolas,monospace}
pre{margin:0;padding:18px 22px;color:#e2e8f0;font:14px/1.55 ui-monospace,'Cascadia Code',Consolas,monospace;
    white-space:pre-wrap;word-break:break-word}
.prompt{color:#22c55e}
"""


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def terminal_html(command: str, output: str, title: str) -> str:
    body = Ansi2HTMLConverter(inline=True, dark_bg=True).convert(output, full=False)
    return f"""<html><head><meta charset="utf-8"><style>{TERMINAL_CSS}</style></head><body>
<div class="win"><div class="bar"><span class="dot" style="background:#ef4444"></span>
<span class="dot" style="background:#f59e0b"></span><span class="dot" style="background:#22c55e"></span>
<span class="title">{html.escape(title)}</span></div>
<pre><span class="prompt">$</span> {html.escape(command)}\n{body}</pre></div></body></html>"""


def shoot(page, html_doc: str, name: str, selector: str = ".win") -> None:
    page.set_content(html_doc, wait_until="networkidle")
    page.locator(selector).first.screenshot(path=str(OUT / name))
    print("  ✓", name)


ARCHITECTURE = """<html><head><meta charset="utf-8"><style>
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.card{display:inline-block;background:#fff;border-radius:12px;padding:22px 26px;box-shadow:0 12px 32px rgba(15,23,42,.18)}
h2{margin:0 0 4px;font-size:20px;color:#0f172a}p{margin:0 0 14px;color:#475569;font-size:14px}
text{font-family:-apple-system,Segoe UI,sans-serif} .mono text{font-family:ui-monospace,Consolas,monospace}
</style></head><body><div class="card">
<h2>How the agent turns a question into a chart</h2>
<p>The LLM only fills in typed tool arguments (chart kind + column names). Python validates them and draws the chart. No LLM-written code is executed.</p>
<svg width="1080" height="430" viewBox="0 0 1080 430" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>
<rect x="10" y="40" width="190" height="96" rx="12" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>
<text x="105" y="74" text-anchor="middle" font-size="16" font-weight="700" fill="#075985">User question</text>
<text x="105" y="98" text-anchor="middle" font-size="12.5" fill="#0c4a6e">"Scatter plots of Dalc</text>
<text x="105" y="115" text-anchor="middle" font-size="12.5" fill="#0c4a6e">and Walc vs G3"</text>
<rect x="260" y="20" width="250" height="140" rx="12" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
<text x="385" y="52" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">Agent loop (create_agent)</text>
<text x="385" y="78" text-anchor="middle" font-size="13" fill="#78350f">LLM: Gemini · OpenAI · offline</text>
<text x="385" y="100" text-anchor="middle" font-size="12.5" fill="#78350f">reason → call tools → observe</text>
<text x="385" y="120" text-anchor="middle" font-size="12.5" fill="#78350f">→ explain the numbers</text>
<text x="385" y="144" text-anchor="middle" font-size="11.5" fill="#92400e">(2 charts = 2 parallel tool calls)</text>
<rect x="600" y="10" width="470" height="232" rx="12" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="835" y="40" text-anchor="middle" font-size="16" font-weight="700" fill="#6b21a8">6 typed tools (@tool)</text>
<g class="mono" font-size="12.5" fill="#581c87">
<text x="620" y="70">1 list_datasets</text><text x="830" y="70" fill="#7e22ce">which CSV files?</text>
<text x="620" y="96">2 describe_dataset</text><text x="830" y="96" fill="#7e22ce">exact column names, dtypes</text>
<text x="620" y="122">3 aggregate</text><text x="830" y="122" fill="#7e22ce">mean/sum/... per group</text>
<text x="620" y="148">4 correlation</text><text x="830" y="148" fill="#7e22ce">Pearson r + strength</text>
<text x="620" y="174">5 filter_rows</text><text x="830" y="174" fill="#7e22ce">allow-listed operators</text>
<text x="620" y="200">6 make_chart</text><text x="830" y="200" fill="#7e22ce">8 kinds → PNG + numbers + code</text>
<text x="620" y="228" font-size="11.5" fill="#7e22ce">read-only · columns validated · errors returned as data</text></g>
<rect x="600" y="290" width="220" height="120" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="710" y="322" text-anchor="middle" font-size="15" font-weight="700" fill="#166534">Chart engine</text>
<text x="710" y="346" text-anchor="middle" font-size="12.5" fill="#14532d">ChartSpec → validate →</text>
<text x="710" y="364" text-anchor="middle" font-size="12.5" fill="#14532d">pandas + matplotlib</text>
<text x="710" y="390" text-anchor="middle" font-size="11.5" fill="#166534">colorblind-safe palette</text>
<rect x="850" y="290" width="220" height="120" rx="12" fill="#e2e8f0" stroke="#475569" stroke-width="2"/>
<text x="960" y="322" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">Outputs</text>
<text x="960" y="346" text-anchor="middle" font-size="12.5" fill="#334155">charts/scatter-Dalc-G3.png</text>
<text x="960" y="366" text-anchor="middle" font-size="12.5" fill="#334155">{"pearson_r": -0.151, ...}</text>
<text x="960" y="386" text-anchor="middle" font-size="12.5" fill="#334155">equivalent pandas code</text>
<rect x="260" y="290" width="250" height="120" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="385" y="322" text-anchor="middle" font-size="16" font-weight="700" fill="#166534">Answer + chart files</text>
<text x="385" y="348" text-anchor="middle" font-size="12.5" fill="#14532d">"Dalc vs G3: r = -0.15, weak</text>
<text x="385" y="366" text-anchor="middle" font-size="12.5" fill="#14532d">negative; Walc: no clear link"</text>
<line x1="200" y1="88" x2="256" y2="88" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<path d="M510,70 C550,60 565,60 596,62" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="522" y="54" font-size="11.5" fill="#334155">tool calls</text>
<path d="M600,150 C560,150 545,130 514,128" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="524" y="168" font-size="11.5" fill="#334155">results (JSON)</text>
<line x1="710" y1="242" x2="710" y2="286" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<line x1="820" y1="350" x2="846" y2="350" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<line x1="385" y1="160" x2="385" y2="286" stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>
<text x="395" y="230" font-size="11.5" fill="#166534">no more tool calls</text>
</svg></div></body></html>"""


def gallery() -> None:
    """The README chart gallery, drawn by the same engine the agent calls."""
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["CHART_DIR"] = str(GALLERY)
    from viz_agent import ChartSpec, configure, render

    configure(ROOT / "data", "students.csv")
    for spec, file_name in [
        (ChartSpec("count", x="sex"), None), (ChartSpec("pie", x="sex", y="Walc"), None),
        (ChartSpec("box", x="freetime", y="G3"), None), (ChartSpec("scatter", x="Dalc", y="G3"), None),
        (ChartSpec("scatter", x="Medu", y="G3"), None), (ChartSpec("bar", x="internet", y="G3"), None),
        (ChartSpec("scatter", x="absences", y="G3"), None), (ChartSpec("heatmap"), None),
        (ChartSpec("line", x="month", y="revenue", hue="category", agg="sum"), "online-sales.csv"),
        (ChartSpec("heatmap", x="month", y="revenue", hue="region", agg="sum"), "online-sales.csv"),
    ]:
        print("  ✓ charts/" + Path(render(spec, file_name).path).name)


def main() -> None:
    gallery()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        shoot(page, ARCHITECTURE, "architecture.png", ".card")

        py = sys.executable
        ex1 = ("viz-agent \"Generate scatter plots showing the relationship between 'Medu' (mother's education level) "
               "and 'G3' (final grade), and between 'Fedu' (father's education level) and 'G3'.\" -p offline -t")
        ex2 = ("viz-agent \"Use bar plots to compare the average final grades ('G3') of students with internet "
               "access at home versus those without ('internet' column).\" -p offline -t")
        ex3 = ("viz-agent \"Plot a scatter plot showing the correlation between the number of absences ('absences') "
               "and final grades ('G3') of students.\" -p offline -t --code")
        shots = [
            ("agent-trace-lab.png", "Agent trace: the lab's scatter-plot task (two charts, parallel tool calls)",
             "viz-agent \"Generate scatter plots to examine the correlation between 'Dalc' and 'G3', "
             "and between 'Walc' and 'G3'.\" -p offline -t", None),
            ("agent-trace-exercises.png", "Exercises 1 and 2", f"{ex1} && {ex2}", None),
            ("agent-trace-code.png", "Exercise 3: the chart plus the equivalent pandas code", ex3, None),
            ("agent-trace-sales.png", "Second dataset: sales trends and a heatmap",
             "viz-agent \"Plot total revenue by month per category\" -p offline -t && "
             "viz-agent \"Show a heatmap of total revenue by region and month\" -p offline", None),
            ("tools-direct.png", "Tools called directly (no LLM)", f"{py} examples/tools_demo.py", "python examples/tools_demo.py"),
            ("lab-pandas-agent.png", "The lab's pattern: create_pandas_dataframe_agent runs LLM-written code",
             f"{py} examples/lab_pandas_agent.py offline", "python examples/lab_pandas_agent.py offline"),
            ("pytest-results.png", "Test suite", f"{py} -m pytest -v --color=yes -p no:cacheprovider", "pytest -v"),
            ("pytest-coverage.png", "Coverage",
             f"{py} -m pytest -q --color=yes -p no:cacheprovider --cov=viz_agent --cov-report=term-missing",
             "pytest --cov=viz_agent --cov-report=term-missing"),
        ]
        for name, title, cmd, shown in shots:
            shoot(page, terminal_html(shown or cmd, run(cmd), title), name)

        nb_html = run(f"{py} -m jupyter nbconvert --to html --stdout notebooks/walkthrough.ipynb 2>/dev/null")
        page.set_viewport_size({"width": 1100, "height": 900})
        page.set_content(nb_html, wait_until="networkidle")
        page.screenshot(path=str(OUT / "notebook-walkthrough.png"), full_page=True)
        print("  ✓ notebook-walkthrough.png")
        browser.close()
    shutil.rmtree(ROOT / "charts", ignore_errors=True)


if __name__ == "__main__":
    main()
