"""Phase 3 captions: kind first and concrete, condition second, colours / style / wall after (common.caption3).

Writes dataset/train_phase3/<crop>.jpg (symlink), .txt (hand labels with a word: the word included) and .alt.txt
(never the word). Same selection as phase 2 (create_text_dataset.py --pairs): dupes, rejected labels skipped.
"""
import csv, json, os
from pathlib import Path
os.environ.setdefault("GRAFF_PHASE", "phase3")
from common import *

dropped = {Path(r["dropped"]).stem for r in csv.DictReader(open(ROOT / "dupes.csv"))}
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
n = 0
import collections
kinds = collections.Counter()
for js in sorted((ROOT / "auto_labels").glob("*.json")):
    d = json.loads(js.read_text())
    img = ROOT / "crops" / (js.stem + ".jpg")
    if js.stem in dropped or d.get("rejected") or not img.exists():
        continue
    link = TRAIN_DIR / img.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(img)
    hand = bool(d.get("hand")) and bool(d.get("text"))
    (TRAIN_DIR / f"{js.stem}.txt").write_text(caption3(d, hand) + "\n", encoding="utf-8")
    (TRAIN_DIR / f"{js.stem}.alt.txt").write_text(caption3(d, False) + "\n", encoding="utf-8")
    kinds[d.get("kind")] += 1
    n += 1
print(n, "pairs ->", TRAIN_DIR, dict(kinds))
