"""Turn label JSON into training captions: dataset/train/<crop>.jpg (symlink) + <crop>.txt.

  python3 create_text_dataset.py                       # auto_labels -> ~/Development/graff-data/dataset/train
  python3 create_text_dataset.py --reviewed-only       # only labels the director accepted / fixed in the editor

Templates are the original one plus `kind` and `condition`. In about half of the captions (seeded per crop)
the real word is left out ("phrasing without it"), so the model does not only learn real names; crops
without a readable word always use the no-word phrasing. Crops dropped by dedupe (dupes.csv) and labels
rejected in the editor are skipped. Nothing is copied: crops are symlinked.
"""
import argparse, csv, json, os, random, re, zlib
from pathlib import Path

BASE = ("a CONDITION KIND graffiti on a BACKGROUND with the word TEXT written in FILL letters "
        "with a OUTLINE outline and HIGHLIGHTS details in a STYLE style")
NOTEXT = ("a CONDITION KIND graffiti on a BACKGROUND in STYLE style, FILL letters "
          "with a OUTLINE outline and HIGHLIGHTS details")
KIND_WORDS = {"tag": "tag", "throwup": "throw-up", "piece": "piece", "character": "character",
              "streetart": "street art", "paste_up": "paste-up", "stencil": "stencil"}
CONDITION_WORDS = {"fresh": "fresh", "faded": "faded", "buffed": "buffed", "gone_over": "gone-over"}
SEED = 360


def fill(tpl: str, d: dict) -> str:
    p = tpl
    if not d.get("highlights"):
        p = p.replace(" and HIGHLIGHTS details", "").replace(" HIGHLIGHTS details", "")
    if not d.get("fill"):
        p = p.replace("written in FILL letters", "written").replace("FILL letters", "letters")
    if not d.get("outline"):
        p = p.replace(" with a OUTLINE outline", "")
    if not d.get("style"):
        p = p.replace(" in a STYLE style", "").replace(" in STYLE style,", ",")
    if not d.get("background"):
        p = p.replace(" on a BACKGROUND", "")
    vals = {"CONDITION": CONDITION_WORDS.get(d.get("condition", ""), ""), "KIND": KIND_WORDS.get(d.get("kind", ""), ""),
            "STYLE": d.get("style", ""), "TEXT": d.get("text", ""), "HIGHLIGHTS": d.get("highlights", ""),
            "OUTLINE": d.get("outline", ""), "FILL": d.get("fill", ""), "BACKGROUND": d.get("background", "")}
    if vals["BACKGROUND"] in ("brick", "concrete", "metal", "stone", "wood"):
        vals["BACKGROUND"] += " wall"
    for k, v in vals.items():
        p = p.replace(k, v)
    p = re.sub(r"\s+", " ", p).replace(" ,", ",").strip()
    return re.sub(r"^a (?=[aeiou])", "an ", p)


def caption(name: str, d: dict) -> str:
    use_text = bool(d.get("text")) and random.Random(zlib.crc32(name.encode()) ^ SEED).random() >= 0.5
    return fill(BASE if use_text else NOTEXT, d)


def main() -> None:
    root = Path(os.path.expanduser("~/Development/graff-data/dataset"))
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputdir", default=str(root / "auto_labels"))
    ap.add_argument("-c", "--crops", default=str(root / "crops"))
    ap.add_argument("-o", "--outputdir", default=str(root / "train"))
    ap.add_argument("--dupes", default=str(root / "dupes.csv"))
    ap.add_argument("--reviewed-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.outputdir)
    out.mkdir(parents=True, exist_ok=True)
    dropped = set()
    if Path(a.dupes).exists():
        dropped = {Path(r["dropped"]).stem for r in csv.DictReader(open(a.dupes))}
    n = skipped = 0
    for js in sorted(Path(a.inputdir).glob("*.json")):
        d = json.loads(js.read_text())
        img = Path(a.crops) / (js.stem + ".jpg")
        if js.stem in dropped or d.get("rejected") or not img.exists() or (a.reviewed_only and not d.get("reviewed")):
            skipped += 1
            continue
        link = out / img.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(img)
        (out / (js.stem + ".txt")).write_text(caption(js.stem, d) + "\n", encoding="utf-8")
        n += 1
    print(f"{n} pairs -> {out} ({skipped} skipped: dupes, rejected, missing or unreviewed)")


if __name__ == "__main__":
    main()
