"""Build phase-1 caption pairs from the hand labels.

Uses the template of create_text_dataset.py. Every crop gets two captions:
  <name>.txt      the original template (with the real word)
  <name>.alt.txt  placeholder-free variant (no real word), so the model does not only learn real names
Training picks one of the two at random (~50 %). Crops are symlinked, never copied.
"""
import argparse, json, os, re
from pathlib import Path

BASE = ("a graffiti on a BACKGROUND with the word TEXT written in FILL letters "
        "with a OUTLINE outline and HIGHLIGHTS details in a STYLE style")
NOTEXT = ("a graffiti on a BACKGROUND in STYLE style, FILL letters "
          "with a OUTLINE outline and HIGHLIGHTS details")


def fill(tpl: str, d: dict) -> str:
    p = tpl
    if d.get("highlights", "") == "":
        p = re.sub(r" (?:and )?\w+ HIGHLIGHTS \w+", "", p).replace(" and HIGHLIGHTS details", "")
        p = p.replace(" HIGHLIGHTS details", "")
    if d.get("fill", "") == "":
        p = p.replace("written in FILL letters", "written").replace("FILL letters", "letters")
    if d.get("outline", "") == "":
        p = re.sub(r" with a OUTLINE outline", "", p)
    for k in ["STYLE", "TEXT", "HIGHLIGHTS", "OUTLINE", "FILL", "BACKGROUND"]:
        v = d.get(k.lower(), "")
        if k == "BACKGROUND" and v == "brick":
            v = "brick wall"
        p = p.replace(k, v)
    return re.sub(r"\s+", " ", p).replace(" ,", ",").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Development/graff-data/dataset"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data)
    out = Path(a.out or data / "train_phase1")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for js in sorted((data / "text_labels").glob("*.json")):
        img = data / "crops" / (js.stem + ".jpg")
        if not img.exists():
            print("missing crop", img)
            continue
        d = json.loads(js.read_text())
        link = out / img.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(img)
        (out / (js.stem + ".txt")).write_text(fill(BASE, d) + "\n")
        (out / (js.stem + ".alt.txt")).write_text(fill(NOTEXT, d) + "\n")
        n += 1
    print(n, "pairs ->", out)


if __name__ == "__main__":
    main()
