import csv, json, re, sqlite3
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from settings import DB_FILE, CATEGORIES

CATS = list(CATEGORIES)
if not Path(DB_FILE).exists():
    raise SystemExit(f"Database not found: {DB_FILE}")

ABSENT = re.compile(
    r"\b(no|not|cannot|without|neither)\b[^.]*\b(visible|shown|identifiable|"
    r"determined|depicted|explicit|apparent|clear)", re.I)

def parts(text):
    text = re.sub(r"Inferred:\s*", "", text)
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip() and not ABSENT.search(s)]

items = []
for p, r in sqlite3.connect(DB_FILE).execute("select path, result_json from images"):
    p = Path(p); part = p.parent.name
    role = "inspiration" if p.stem.lower().startswith(("ai_", "web_")) else "final"
    items.append(dict(file=p.name, part=part, role=role, res=json.loads(r)))

flat = {c: [] for c in CATS}          # (item index, sentence)
for i, it in enumerate(items):
    for c in CATS:
        for s in parts(it["res"][c]):
            flat[c].append((i, s))

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb, idx = {}, {c: {} for c in CATS}
for c in CATS:
    if not flat[c]:
        continue
    e = model.encode([s for _, s in flat[c]], normalize_embeddings=True)
    e = e - e.mean(axis=0)                                   # centering
    emb[c] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    for k, (i, _) in enumerate(flat[c]):
        idx[c].setdefault(i, []).append(k)

def coverage(i, j, c):
    a, b = idx[c].get(i, []), idx[c].get(j, [])
    if not a or not b:
        return None, []
    sim = emb[c][a] @ emb[c][b].T
    best = sim.max(axis=1)
    matches = [[flat[c][a[r]][1], flat[c][b[int(sim[r].argmax())]][1], round(float(best[r]), 3)]
               for r in range(len(a))]
    return float(best.mean()), matches

insp = [i for i, it in enumerate(items) if it["role"] == "inspiration"]
fin = [j for j, it in enumerate(items) if it["role"] == "final"]
scores = {c: {"real": [], "random": []} for c in CATS}

with open("parts_pairs.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["participant_insp", "inspiration", "participant_final", "final",
                "same_participant", "category", "coverage", "best_matches"])
    for i in insp:
        for j in fin:
            same = items[i]["part"] == items[j]["part"]
            for c in CATS:
                cov, matches = coverage(i, j, c)
                if cov is not None:
                    scores[c]["real" if same else "random"].append(cov)
                w.writerow([items[i]["part"], items[i]["file"], items[j]["part"],
                            items[j]["file"], same, c,
                            None if cov is None else round(cov, 3),
                            json.dumps(matches, ensure_ascii=False)])

def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

print(f"{'category':22} {'real':>7} {'random':>7} {'gap':>7} {'AUC':>6}")
for c in CATS:
    r, n = scores[c]["real"], scores[c]["random"]
    if r and n:
        print(f"{c:22} {np.mean(r):7.3f} {np.mean(n):7.3f} {np.mean(r)-np.mean(n):7.3f} {auc(r, n):6.2f}")
print("Saved parts_pairs.csv")