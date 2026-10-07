"""Generate the bundled sample datasets in data/ (synthetic, reproducible, license-free).

    python scripts/make_datasets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# --- students.csv: same idea as the lab's student table, but made up ------------
n = 400
medu = rng.choice([0, 1, 2, 3, 4], n, p=[.03, .15, .27, .25, .30])
fedu = np.clip(medu + rng.integers(-1, 2, n), 0, 4)
studytime = rng.choice([1, 2, 3, 4], n, p=[.27, .5, .16, .07])
failures = rng.choice([0, 1, 2, 3], n, p=[.78, .13, .05, .04])
internet = np.where(rng.random(n) < 0.6 + 0.07 * (medu - 2), "yes", "no")
freetime = rng.choice([1, 2, 3, 4, 5], n, p=[.05, .16, .40, .29, .10])
goout = np.clip(freetime + rng.integers(-1, 2, n), 1, 5)
dalc = rng.choice([1, 2, 3, 4, 5], n, p=[.70, .19, .06, .03, .02])
walc = np.clip(dalc + rng.choice([0, 1, 2], n, p=[.45, .35, .20]), 1, 5)
absences = rng.poisson(5.5, n) + rng.choice([0, 10], n, p=[.92, .08])
sex = rng.choice(["F", "M"], n, p=[.52, .48])
ability = rng.normal(0, 2.2, n)
g1 = 9 + 0.55 * medu + 0.25 * fedu + 0.6 * studytime - 1.4 * failures + 0.35 * (internet == "yes") \
    - 0.35 * dalc - 0.12 * walc - 0.04 * absences + ability + rng.normal(0, 1.2, n)
g2 = g1 + rng.normal(0, 1.1, n)
g3 = g2 + rng.normal(0, 1.0, n)
students = pd.DataFrame({
    "school": rng.choice(["North High", "River Academy"], n, p=[.88, .12]),
    "sex": sex, "age": rng.choice([15, 16, 17, 18, 19, 20], n, p=[.21, .26, .25, .21, .06, .01]),
    "address": rng.choice(["urban", "rural"], n, p=[.78, .22]),
    "Medu": medu, "Fedu": fedu, "studytime": studytime, "failures": failures, "internet": internet,
    "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc, "absences": absences,
    "G1": np.clip(g1, 0, 20).round().astype(int), "G2": np.clip(g2, 0, 20).round().astype(int),
    "G3": np.clip(g3, 0, 20).round().astype(int),
})
students.to_csv(DATA / "students.csv", index=False)

# --- online-sales.csv: monthly orders/revenue by category and region -------------
rows = []
months = pd.period_range("2025-01", "2025-12", freq="M")
base = {"Electronics": 420, "Home": 260, "Fashion": 310}
price = {"Electronics": 185.0, "Home": 64.0, "Fashion": 48.0}
region_factor = {"North": 1.15, "South": 0.85, "East": 1.0, "West": 1.05}
for i, month in enumerate(months):
    season = 1 + 0.08 * np.sin(2 * np.pi * (i - 2) / 12) + (0.45 if month.month in (11, 12) else 0)
    for category, b in base.items():
        trend = 1 + (0.03 * i if category == "Fashion" else 0.01 * i)
        for region, f in region_factor.items():
            orders = int(b * f * season * trend * rng.normal(1, .06))
            rows.append({"month": str(month), "category": category, "region": region, "orders": orders,
                         "revenue": round(orders * price[category] * rng.normal(1, .03), 2)})
pd.DataFrame(rows).to_csv(DATA / "online-sales.csv", index=False)
print(students.shape, len(rows), "sales rows; mean G3", round(students.G3.mean(), 2))
