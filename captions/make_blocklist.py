"""Write the two blocklists in data/ (lowercase words only, no images).

  data/blocklist_hand.txt   words of the hand labels (538 crops): real, so the bake matches them with edit distance <= 1
  data/blocklist_auto.txt   OCR guesses of the auto labels (+ the model's guesses on hand crops): mostly wrong,
                            so the bake matches them EXACTLY only (edit distance 1 would eat most 3-4 letter words)
Words in the hand list are removed from the auto list.   python3 captions/make_blocklist.py
"""
import json, os
from pathlib import Path

root = Path(os.path.expanduser("~/Development/graff-data/dataset"))
hand, auto = set(), set()
for p in (root / "text_labels").glob("*.json"):
    t = (json.loads(p.read_text()).get("text") or "").strip().lower()
    if t:
        hand.add(t)
for p in (root / "auto_labels").glob("*.json"):
    d = json.loads(p.read_text())
    if d.get("hand"):
        hand_t = (d.get("text") or "").strip().lower()
        if hand_t:
            hand.add(hand_t)
        t = ((d.get("pred") or {}).get("text") or "").strip().lower()
    else:
        t = (d.get("text") or "").strip().lower()
    if t:
        auto.add(t)
auto -= hand
data = Path(__file__).resolve().parent.parent / "data"
for name, words in (("blocklist_hand.txt", hand), ("blocklist_auto.txt", auto)):
    (data / name).write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    print(len(words), "words ->", data / name)
