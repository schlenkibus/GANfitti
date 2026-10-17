"""Review queue for auto labels (used by create_text_labels_editor.py --auto).

Keyboard flow (one label per screen, no mouse needed):
  Enter / a   accept the label as shown
  e           edit (focus the fields; Tab between them, Enter saves as "fixed", Esc cancels)
  x           reject (label stays on disk with rejected: true, training skips it)
  s           skip for now        z   undo the last decision
Decisions are written into the label JSON: reviewed = accepted | fixed | rejected, fixed_fields = [...],
and orig = the model's original values (so agreement can be counted: /stats or --stats).
"""
import json, random
from pathlib import Path

import schema as S

FIELDS = ["text", "kind", "condition", "background", "style", "fill", "outline", "highlights"]


def load_all(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def build_queue(labels: dict, sample: int, include_hand: bool, seed: int = 360) -> list:
    names = sorted(n for n, l in labels.items() if l.get("auto") and (include_hand or not l.get("hand")))
    random.Random(seed).shuffle(names)
    if sample:
        names = names[:sample]  # stable: same sample on every start, reviewed ones stay in it
    return names


def stats(labels: dict, names: list) -> dict:
    rev = [labels[n] for n in names if labels[n].get("reviewed")]
    out = {"queue": len(names), "reviewed": len(rev),
           "accepted": sum(l["reviewed"] == "accepted" for l in rev),
           "fixed": sum(l["reviewed"] == "fixed" for l in rev),
           "rejected": sum(l["reviewed"] == "rejected" for l in rev), "field_agreement_pct": {}}
    kept = [l for l in rev if l["reviewed"] != "rejected"]
    for f in FIELDS:
        if kept:
            ok = sum(f not in l.get("fixed_fields", []) for l in kept)
            out["field_agreement_pct"][f] = round(100 * ok / len(kept), 1)
    return out


def run(labels_dir: str, images_dir: str, sample: int = 0, include_hand: bool = False, port: int = 5000,
        stats_only: bool = False) -> None:
    from flask import Flask, jsonify, request, send_from_directory, render_template
    ldir, idir = Path(labels_dir), Path(images_dir)
    labels = load_all(ldir)
    queue = build_queue(labels, sample, include_hand)
    if stats_only:
        print(json.dumps(stats(labels, queue), indent=1))
        return
    skipped: set = set()
    history: list = []
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent.parent / "templates"))

    def current():
        for n in queue:
            if n not in skipped and not labels[n].get("reviewed"):
                return n
        return None

    def save(n):
        (ldir / f"{n}.json").write_text(json.dumps(labels[n], ensure_ascii=False))

    @app.route("/")
    def index():
        return render_template("graff-auto-review.html", enums={"kind": S.KINDS, "condition": S.CONDITIONS})

    @app.route("/api/current")
    def api_current():
        n = current()
        st = stats(labels, queue)
        if n is None:
            return jsonify({"done": True, "stats": st})
        return jsonify({"name": n, "label": {f: labels[n].get(f, "") for f in FIELDS}, "stats": st,
                        "left": sum(1 for m in queue if not labels[m].get("reviewed"))})

    @app.route("/img/<name>.jpg")
    def img(name):
        return send_from_directory(idir, f"{name}.jpg")

    @app.route("/api/decide", methods=["POST"])
    def decide():
        body = request.get_json()
        n, action = body["name"], body["action"]
        if action == "skip":
            skipped.add(n)
            return jsonify({"ok": True})
        lab = labels[n]
        lab["orig"] = {f: lab.get(f, "") for f in FIELDS}
        if action == "accept":
            lab["reviewed"] = "accepted"
        elif action == "reject":
            lab["reviewed"] = "rejected"
            lab["rejected"] = True
        else:
            new = body["label"]
            changed = [f for f in FIELDS if (new.get(f, "") or "").strip().lower() != str(lab.get(f, "")).strip().lower()]
            for f in changed:
                lab[f] = new[f].strip().lower()
            lab["reviewed"] = "fixed" if changed else "accepted"
            lab["fixed_fields"] = changed
        save(n)
        history.append(n)
        return jsonify({"ok": True})

    @app.route("/api/undo", methods=["POST"])
    def undo():
        if not history:
            return jsonify({"ok": False})
        n = history.pop()
        lab = labels[n]
        lab.update(lab.pop("orig", {}))
        for k in ("reviewed", "fixed_fields", "rejected"):
            lab.pop(k, None)
        save(n)
        skipped.discard(n)
        skipped.difference_update(skipped)  # show the undone crop next
        return jsonify({"ok": True})

    print(f"auto review queue: {len(queue)} labels, open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port)
