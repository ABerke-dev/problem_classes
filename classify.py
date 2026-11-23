import csv, os, re, sqlite3, time
from pathlib import Path
from openai import OpenAI

from settings import DB_FILE, JUDGE_MODEL, CATEGORIES
   # keep the model you want to use for labelling

CATS = list(CATEGORIES)
LABELS = ["use", "transform", "reference", "none"]
OUT = Path("transfer.csv")
if not Path(DB_FILE).exists():
    raise SystemExit(f"Database not found: {DB_FILE}")

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                timeout=120, max_retries=0)

ABSENT = re.compile(
    r"\b(no|not|cannot|without|neither)\b[^.]*\b(visible|shown|identifiable|"
    r"determined|depicted|explicit|apparent|clear)", re.I)

def parts(text):
    text = re.sub(r"Inferred:\s*", "", text)
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip() and not ABSENT.search(s)]

RULES = """You compare a design inspiration (A) with a final project (B), category by category.
For each category decide which relation holds between what A shows and what B shows:
- use: the same property appears in B essentially unchanged
- transform: the property from A appears in B but changed to fit a new context (different scale, form, material, medium or function)
- reference: only an abstract idea, principle or relation from A is echoed in B; the surface properties differ
- none: nothing in B corresponds to this category of A
Use ONLY the two texts."""
SYSTEM = (RULES
          + "\nAnswer with exactly six lines, one per category, in this exact format:\n"
          + "category_name | label | evidence\n"
          + "category_name is one of: " + ", ".join(CATS) + "\n"
          + "label is one of: " + ", ".join(LABELS) + "\n"
          + "evidence is a short phrase of at most 20 words and must not contain the | character.\n"
          + "No JSON, no markdown, no extra text.")

def parse_answer(text):
    found = {}
    for line in text.splitlines():
        line = line.strip().lstrip("-*•0123456789.) ").replace("**", "").replace("`", "")
        cols = [p.strip() for p in line.split("|")]
        if len(cols) < 2:
            continue
        cat = re.sub(r"[^a-z_]", "", cols[0].lower().replace(" ", "_"))
        if cat not in CATS:
            continue
        label = re.sub(r"[^a-z]", "", cols[1].lower())
        found[cat] = (label, " ".join(cols[2:]).strip())
    missing = [c for c in CATS if c not in found]
    if missing:
        raise ValueError("missing " + ", ".join(missing) + " | answer: "
                         + text[:200].replace("\n", " / "))
    return found

items = []
for p, r in sqlite3.connect(DB_FILE).execute("select path, result_json from images"):
    import json
    p = Path(p); part = p.parent.name
    role = "inspiration" if p.stem.lower().startswith(("ai_", "web_")) else "final"
    items.append(dict(file=p.name, part=part, role=role, res=json.loads(r)))

def block(res):
    return "\n".join(f"{c}: " + (" ".join(parts(res[c])) or "(not identifiable)") for c in CATS)

done = set()
if OUT.exists():
    with open(OUT, encoding="utf-8-sig") as f:
        done = {(r["participant"], r["inspiration"], r["final"]) for r in csv.DictReader(f)}

new = not OUT.exists()
fails = 0
with open(OUT, "a", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["participant", "inspiration", "final", "category", "label",
                    "evidence", "judge_model"])
    for a in [x for x in items if x["role"] == "inspiration"]:
        for b in [x for x in items if x["role"] == "final" and x["part"] == a["part"]]:
            if (a["part"], a["file"], b["file"]) in done:
                continue
            msg = f"INSPIRATION A:\n{block(a['res'])}\n\nFINAL PROJECT B:\n{block(b['res'])}"
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL, temperature=0, max_tokens=1500,
                    extra_body={"reasoning": {"enabled": False}},  # delete this line if you get a 400 error about reasoning
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": msg}])
                ch = resp.choices[0]
                t = (ch.message.content or "").strip()
                if not t:
                    rz = len(getattr(ch.message, "reasoning", None) or "")
                    raise ValueError(f"empty answer (finish_reason={ch.finish_reason}, reasoning_chars={rz})")
                found = parse_answer(t)

                rows_out = []
                for c in CATS:
                    na = not parts(a["res"][c]) or not parts(b["res"][c])
                    label = "n/a" if na else found[c][0]
                    if label not in LABELS + ["n/a"]:
                        label = "invalid"
                    rows_out.append([a["part"], a["file"], b["file"], c, label,
                                     found[c][1], JUDGE_MODEL])
                w.writerows(rows_out)
                f.flush()
                fails = 0
                print("done", a["part"], a["file"], b["file"])
            except Exception as e:
                err = str(e)
                if "Error code: 402" in err or ("429" in err and "day" in err.lower()):
                    print("Daily or credit limit reached. Stopping. Rerun later.")
                    raise SystemExit
                print("Failed", a["file"], b["file"], err[:300])
                fails += 1
                if fails >= 3:
                    print("3 failures in a row. Stopping. Try another model or retry later.")
                    raise SystemExit
            time.sleep(2)
print("Saved", OUT)