"""Auto-caption crops through a local vLLM server (OpenAI-compatible API, structured output).

  # 1. start the server (see captions/README.md), 2. held-out check, 3. full run
  ~/.venvs/vllm/bin/python captions/auto_caption.py --eval --tag 2b          # 50 held-out hand labels
  ~/.venvs/vllm/bin/python captions/auto_caption.py --run                    # all non-dropped crops, resumable

Writes dataset/auto_labels/<crop>.json with `auto: true`. Hand labels are never touched: for the
hand-labeled crops the hand fields are copied and only `kind`/`condition` come from the model
(`hand: true`, model prediction kept under `pred`).
"""
import argparse, base64, io, json, os, random, sys, time, csv, difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import schema as S

DATA = Path(os.path.expanduser("~/Development/graff-data/dataset"))
HERE = Path(__file__).parent
SEED = 360

SYSTEM = """You label photos of graffiti for a training dataset. Look at the piece in the photo and answer with one JSON object.

kind: tag = a quick one-colour signature or scribble, usually small; throwup = fast bubbly letters with a fill and an outline, usually two colours; piece = a large, elaborate multi-colour lettering work; character = a painted cartoon figure or face; streetart = a mural or figurative picture, not lettering; paste_up = a paper poster, sticker or wheat-paste; stencil = sprayed through a stencil.
condition: fresh = bright, crisp paint; faded = sun-bleached, washed out, worn; buffed = partly painted over by the city (flat grey or beige patches); gone_over = other graffiti painted on top of it.
text: the word the piece spells, lowercase letters only; "" if you cannot read it.
background: the surface it is painted on.
style: 1 to 3 words describing the letter shapes.
style: pick the words that really fit; most pieces are "block" or "simple", add "bubble" only for puffy inflated letters, "wildstyle" only for interlocking arrows and extensions.
background: use "wall" for a plain painted or plastered wall, "concrete wall" only if clearly raw concrete.
fill: the colour INSIDE the letters, normally ONE colour (white and chrome are the most common; chrome = silver/metallic). outline: the line around the letters, usually black. highlights: only for clearly separate small accent strokes or shine marks; most pieces have none, so answer [].

These are labels written by hand for other photos, to show the vocabulary and level of detail:
{examples}"""


def load_hand() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((DATA / "text_labels").glob("*.json"))}


def split_heldout(hand: dict) -> tuple:
    f = HERE / "heldout.txt"
    if f.exists():
        held = [l.strip() for l in f.read_text().split() if l.strip()]
    else:
        held = sorted(random.Random(SEED).sample(sorted(hand), 50))
        f.write_text("\n".join(held) + "\n")
    return held, [k for k in sorted(hand) if k not in set(held)]


def example_line(d: dict) -> str:
    ex = {"text": S.norm_text(d["text"]), "background": S.norm_background(d["background"]) or "wall",
          "style": S.norm_style(d["style"]) or ["simple"], "fill": S.norm_colors(d["fill"]) or ["white"],
          "outline": S.norm_colors(d["outline"]), "highlights": S.norm_colors(d["highlights"])}
    return json.dumps(ex)


def pick_examples(pool: list, hand: dict, n: int) -> list:
    """Seeded, stratified a little: spread over fill colours so one colour does not dominate."""
    rng = random.Random(SEED)
    pool = [k for k in pool if hand[k]["text"] and S.norm_colors(hand[k]["fill"])]
    rng.shuffle(pool)
    seen, out = {}, []
    for k in pool:
        c = S.norm_colors(hand[k]["fill"])[0]
        if seen.get(c, 0) < 2 or len(seen) > 8 and len(out) < n and seen.get(c, 0) < 4:
            seen[c] = seen.get(c, 0) + 1
            out.append(k)
        if len(out) >= n:
            break
    return out


def encode(path: Path, maxside: int) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((maxside, maxside), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def ask(a, system: str, path: Path) -> dict:
    body = {"model": a.model, "temperature": 0.0, "seed": SEED, "max_tokens": 220,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": encode(path, a.maxside)}},
                             {"type": "text", "text": "Label this graffiti."}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "graffiti_label", "schema": S.json_schema(), "strict": True}}}
    last = None
    for attempt in range(3):
        try:
            r = requests.post(a.api.rstrip("/") + "/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"{path.name}: {last}")


def run_many(a, system: str, names: list, on_done) -> int:
    fails = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ask, a, system, DATA / "crops" / (n + ".jpg")): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            n = futs[f]
            try:
                on_done(n, f.result())
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", e, flush=True)
            if i % 100 == 0:
                print(f"{i}/{len(names)}", flush=True)
    return fails


def agreement(preds: dict, hand: dict) -> dict:
    cnt = {}
    def add(field, ok):
        c = cnt.setdefault(field, [0, 0]); c[0] += int(ok); c[1] += 1
    for n, p in preds.items():
        h = hand[n]
        ht, pt = S.norm_text(h["text"]), S.norm_text(p["text"])
        if ht:
            add("text (exact)", ht == pt)
            add("text (fuzzy>=0.6)", difflib.SequenceMatcher(None, ht, pt).ratio() >= 0.6)
        hb = S.norm_background(h["background"])
        if hb: add("background (exact)", hb == p["background"])
        hs = S.norm_style(h["style"])
        if hs:
            add("style (first token)", hs[0] == p["style"][0])
            add("style (any overlap)", bool(set(hs) & set(p["style"])))
        for f in ("fill", "outline"):
            hc = S.norm_colors(h[f])
            if hc:
                add(f"{f} (primary)", hc[0] == p[f][0] if p[f] else False)
                add(f"{f} (any overlap)", bool(set(hc) & set(p[f])))
        hh = S.norm_colors(h["highlights"])
        add("highlights (empty/overlap)", (not hh and not p["highlights"]) or bool(set(hh) & set(p["highlights"])))
    return {k: {"agree": v[0], "n": v[1], "pct": round(100 * v[0] / v[1], 1)} for k, v in cnt.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--eval", action="store_true", help="held-out check on 50 hand-labeled crops")
    ap.add_argument("--run", action="store_true", help="caption all non-dropped crops (resumable)")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--shots", type=int, default=12)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--maxside", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "auto_labels"))
    a = ap.parse_args()
    hand = load_hand()
    held, pool = split_heldout(hand)
    shots = pick_examples(pool, hand, a.shots)
    system = SYSTEM.format(examples="\n".join(example_line(hand[k]) for k in shots))
    t0 = time.time()

    if a.eval:
        preds = {}
        fails = run_many(a, system, held, lambda n, p: preds.__setitem__(n, p))
        res = agreement(preds, hand)
        res["_meta"] = {"model": a.model, "held_out": len(held), "ok": len(preds), "fails": fails,
                        "shots": shots, "seconds": round(time.time() - t0, 1)}
        (HERE / f"eval_{a.tag}.json").write_text(json.dumps(res, indent=1))
        (HERE / f"eval_{a.tag}_preds.json").write_text(json.dumps(preds, indent=1))
        for k, v in res.items():
            print(k, v)
        return

    if a.run:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        dropped = set()
        if (DATA / "dupes.csv").exists():
            dropped = {Path(r["dropped"]).stem for r in csv.DictReader(open(DATA / "dupes.csv"))}
        names = sorted(p.stem for p in (DATA / "crops").glob("*.jpg"))
        todo = [n for n in names if (n in hand or n not in dropped) and not (out / f"{n}.json").exists()]
        if a.limit: todo = todo[:a.limit]
        print(f"{len(names)} crops, {len(dropped)} dropped as dupes, {len(todo)} to do", flush=True)
        def save(n, pred):
            lab = S.to_label(pred)
            if n in hand:  # hand fields win, model only backfills kind/condition
                h = hand[n]
                lab = {**{k: h[k] for k in ("text", "background", "fill", "outline", "style", "highlights")},
                       "kind": lab["kind"], "condition": lab["condition"], "hand": True, "pred": S.to_label(pred)}
            lab["auto"] = True
            lab["model"] = a.model
            tmp = out / f".{n}.tmp"
            tmp.write_text(json.dumps(lab, ensure_ascii=False))
            tmp.rename(out / f"{n}.json")
        fails = run_many(a, system, todo, save)
        print(f"done in {time.time() - t0:.0f}s, {fails} failures", flush=True)
        (HERE / "run_time.txt").write_text(f"{time.time() - t0:.0f}s for {len(todo)} crops, {fails} failures\n")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
