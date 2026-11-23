import base64, hashlib, io, json, os, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from settings import MODEL, DB_FILE, IMAGE_DIR, CATEGORIES

DB_PATH = Path(DB_FILE)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SLEEP_BETWEEN_CALLS = 3

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=90,
    max_retries=0,
)

PROMPT = f"""You analyse design and architecture images.
For EACH category below write exactly 2 short factual sentences (max 40 words in total).
Describe only what is visible. If a category cannot be judged from the image,
write exactly "not identifiable".
For function_ergonomics, context_meaning and structure_technology, start with
"Inferred:" when you are not describing something directly visible.
Do not invent brands, exact models, dimensions, hidden materials or construction methods.
If the image is a sketch, drawing, collage or mockup, describe the design it depicts,
not the drawing medium, paper, ink or presentation format. Example: write
"A bed and a central light fixture in a rectangular room", not "magenta ink on dotted paper".
Use the same neutral style for every image.

Categories (key: what to cover):
{json.dumps(CATEGORIES, indent=1)}

Return a single JSON object with exactly these keys and string values.
No markdown, no extra text.
"""
PROMPT_ID = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]
TAG = f"{MODEL}|{PROMPT_ID}"   # a changed prompt makes the script reprocess images


class OutOfCredits(Exception):
    pass


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_data_url(path, max_side=1280):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS images (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            model TEXT NOT NULL,
            result_json TEXT NOT NULL,
            search_text TEXT NOT NULL,
            processed_at TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS image_search
        USING fts5(path UNINDEXED, search_text)
    """)
    con.commit()
    return con


def analyze_image(path, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=3000,
                response_format={"type": "json_object"},   # delete this line if the API rejects it
                extra_body={"reasoning": {"effort": "low"}},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": to_data_url(path)}},
                    ],
                }],
            )
            msg = resp.choices[0].message
            text = msg.content
            if not text:
                finish = resp.choices[0].finish_reason
                refusal = getattr(msg, "refusal", None)
                reasoning = getattr(msg, "reasoning", None)
                raise ValueError(
                    f"Empty content (finish_reason={finish}, refusal={refusal}, "
                    f"has_reasoning={bool(reasoning)}, usage={resp.usage})"
                )
            text = text.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(text)
            missing = [k for k in CATEGORIES if k not in result]
            if missing:
                raise ValueError(f"Missing categories: {missing}")
            result = {k: str(result[k]) for k in CATEGORIES}
            return result, " ".join(result.values())
        except Exception as exc:
            msg = str(exc)
            if "Error code: 402" in msg or ("429" in msg and "day" in msg.lower()):
                raise OutOfCredits(msg[:300])
            last_err = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{path.name}: {last_err}")


def main():
    if not IMAGE_DIR.exists():
        raise SystemExit(f"Image folder not found: {IMAGE_DIR}")
    files = sorted(p for p in IMAGE_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit("No supported images found.")

    con = connect_db()
    completed = failed = 0
    try:
        for path in tqdm(files, desc="Analyzing images"):
            path_key = str(path.resolve())
            try:
                digest = file_hash(path)
                existing = con.execute(
                    "SELECT sha256, model FROM images WHERE path = ?", (path_key,)
                ).fetchone()
                if existing == (digest, TAG):
                    continue

                result, search_text = analyze_image(path)

                with con:
                    con.execute(
                        """INSERT OR REPLACE INTO images
                           (path, sha256, model, result_json, search_text, processed_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (path_key, digest, TAG,
                         json.dumps(result, ensure_ascii=False), search_text,
                         datetime.now(timezone.utc).isoformat()),
                    )
                    con.execute("DELETE FROM image_search WHERE path = ?", (path_key,))
                    con.execute("INSERT INTO image_search (path, search_text) VALUES (?, ?)",
                                (path_key, search_text))
                completed += 1
                time.sleep(SLEEP_BETWEEN_CALLS)

            except OutOfCredits as exc:
                print("\nOpenRouter says your credits are used up. Stopping.", exc)
                break
            except Exception as exc:
                failed += 1
                print(f"\nFailed: {path}\n  {exc}")
    finally:
        con.close()

    print(f"New or updated images: {completed}")
    print(f"Failed images: {failed}")
    print(f"Database: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()