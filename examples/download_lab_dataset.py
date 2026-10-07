"""Optional: download the original lab's student-mat.csv into data/lab/ (free, no API).

    python examples/download_lab_dataset.py
    viz-agent "Plot the gender count" --data-dir data/lab -d student-mat.csv -p offline
"""

import urllib.request
from pathlib import Path

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv"
target = Path(__file__).resolve().parents[1] / "data" / "lab" / "student-mat.csv"
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    print(f"✔ {target} already present")
else:
    urllib.request.urlretrieve(URL, target)
    print(f"⬇ {target}")
