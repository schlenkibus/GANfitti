"""Write data/blocklist.txt: every distinct `text` value of the hand labels and auto labels, lowercase.

The graffgen bake OCRs each generated texture and drops it if it reads as a blocklisted word (edit distance <= 1).
Only words go in, no images.   python3 captions/make_blocklist.py [labels_dir ...]
"""
import json, os, sys
from pathlib import Path

root = Path(os.path.expanduser("~/Development/graff-data/dataset"))
dirs = [Path(d) for d in sys.argv[1:]] or [root / "text_labels", root / "auto_labels"]
words = set()
for d in dirs:
    for p in d.glob("*.json") if d.exists() else []:
        t = (json.loads(p.read_text()).get("text") or "").strip().lower()
        if t:
            words.add(t)
out = Path(__file__).resolve().parent.parent / "data" / "blocklist.txt"
out.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
print(len(words), "words ->", out)
