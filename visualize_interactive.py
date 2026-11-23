import base64, io, json, sqlite3
from pathlib import Path

import pandas as pd
from PIL import Image

from settings import DB_FILE, CATEGORIES

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

df = pd.read_csv("transfer.csv", encoding="utf-8-sig").fillna("")

desc = {}
for p, r in sqlite3.connect(DB_FILE).execute("select path, result_json from images"):
    p = Path(p)
    desc[(p.parent.name, p.name.upper())] = json.loads(r)


def data_url(path, size):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


data = {}
for part, g in df.groupby("participant"):
    folder = Path("images") / part
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT]
    ai = sorted(p for p in files if p.stem.upper().startswith("AI_"))
    web = sorted(p for p in files if p.stem.upper().startswith("WEB_"))
    final = [p for p in files if p not in ai + web][0]

    def item(p, size):
        rows = g[g["inspiration"].str.upper() == p.name.upper()]
        rel = {r["category"]: {"label": r["label"], "evidence": r["evidence"]}
               for _, r in rows.iterrows()}
        return {"name": p.stem, "img": data_url(p, size), "rel": rel,
                "desc": desc.get((part, p.name.upper()), {})}

    data[part] = {"final": item(final, 600),
                  "ai": [item(p, 320) for p in ai],
                  "web": [item(p, 320) for p in web]}

payload = json.dumps({"data": data, "cats": CATEGORIES}, ensure_ascii=False)
Path("data.js").write_text("const P = " + payload + ";", encoding="utf-8")
print("Saved data.js")