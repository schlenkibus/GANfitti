"""Distribution of kind/condition/style/background/fill over the auto labels -> captions/stats.json (small, committed)."""
import collections, json, os
from pathlib import Path

root = Path(os.path.expanduser("~/Development/graff-data/dataset"))
dropped = set()
if (root / "dupes.csv").exists():
    import csv
    dropped = {Path(r["dropped"]).stem for r in csv.DictReader(open(root / "dupes.csv"))}
L = {p.stem: json.loads(p.read_text()) for p in (root / "auto_labels").glob("*.json")}
use = {n: l for n, l in L.items() if n not in dropped and not l.get("rejected")}
out = {"labels": len(L), "after_dedupe": len(use)}
split = lambda s: [t.strip() for t in s.split(",") if t.strip()]
for f in ("kind", "condition", "background"):
    out[f] = dict(collections.Counter(l.get(f, "") for l in use.values()).most_common())
for f in ("style", "fill", "outline", "highlights"):
    c = collections.Counter(t for l in use.values() for t in (split(l.get(f, "")) or ["(none)"]))
    out[f] = dict(c.most_common())
out["kind_x_condition"] = {k: v for k, v in sorted(collections.Counter(f"{l.get('kind')}/{l.get('condition')}" for l in use.values()).items())}
Path(__file__).with_name("stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
