import csv, json, sqlite3
from pathlib import Path

from settings import DB_FILE, CATEGORIES

CATS = list(CATEGORIES)
OUT = Path("descriptions.csv")

rows = sqlite3.connect(DB_FILE).execute(
    "select path, result_json from images order by path").fetchall()

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["folder", "file"] + CATS)
    for path, result_json in rows:
        p = Path(path)
        r = json.loads(result_json)
        w.writerow([p.parent.name, p.name] + [r.get(c, "") for c in CATS])

print(f"Saved {OUT} ({len(rows)} images)")