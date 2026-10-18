"""Shared helpers for the Sana LoRA phase-1 tooling."""
import os, json, re
from pathlib import Path

ROOT = Path(os.environ.get("GRAFF_DATA", os.path.expanduser("~/Development/graff-data/dataset")))
PHASE = os.environ.get("GRAFF_PHASE", "phase1")  # phase1 (538 hand crops) or phase2 (full captioned set)
TRAIN_DIR = ROOT / f"train_{PHASE}"
CACHE_DIR = ROOT / "cache" / PHASE
LORA_DIR = ROOT / "lora" / PHASE
# phase 3 switches to the 1.6B base (same Gemma-2 encoder + DC-AE VAE as 600M: identical file hashes)
MODEL_ID = ("Efficient-Large-Model/Sana_1600M_512px_diffusers" if PHASE == "phase3"
            else "Efficient-Large-Model/Sana_600M_512px_diffusers")
VARIANT = "fp16" if PHASE == "phase3" else None  # fp16 shards are half the download; we load bf16 anyway

# aspect buckets (w, h), all multiples of 32 (DC-AE f32), ~512*512 pixels
BUCKETS = [(512, 512), (640, 384), (384, 640), (768, 320), (320, 768), (576, 448), (448, 576)]

MAX_SEQ = 300  # Gemma token budget used by Sana


def nearest_bucket(w: int, h: int):
    import math
    r = math.log(w / h)
    return min(BUCKETS, key=lambda b: abs(math.log(b[0] / b[1]) - r))


# ---- prompt grid -----------------------------------------------------------
KINDS = ["tag", "throw-up", "piece", "cartoon character", "street art mural",
         "paste-up poster", "stencil", "wildstyle piece"]
WORDS = ["kroz", "vemt", "duna", "plok", "shiv", "brax", "tolu", "zeke"]  # invented, one per kind
STYLES = ["block", "bubble", "semi wildstyle", "simple, rounded"]
COLORS = [("white", "black"), ("chrome", "red")]  # (fill, outline)
NEGATIVE = ""


def grid_prompts():
    """Rows = style x color (8), columns = kind (8). Returns list of (prompt, seed)."""
    out = []
    for si, style in enumerate(STYLES):
        for ci, (fill, outline) in enumerate(COLORS):
            for ki, kind in enumerate(KINDS):
                p = (f"a {kind} graffiti on a brick wall with the word {WORDS[ki]} written in "
                     f"{fill} letters with a {outline} outline in a {style} style")
                out.append((p, 1000 + (si * 2 + ci) * 8 + ki))
    return out


# ---- phase-2 grid: rows = kind/condition (8), columns = fill/outline/background combos that are NOT white on red brick
ROWS2 = [  # (condition, kind, style, invented word)
    ("fresh", "tag", "simple", "kroz"), ("fresh", "throw-up", "bubble", "vemt"),
    ("fresh", "piece", "block", "duna"), ("fresh", "character", "clean", "plok"),
    ("fresh", "wildstyle piece", "wildstyle", "shiv"), ("buffed", "piece", "block", "brax"),
    ("faded", "tag", "simple", "tolu"), ("gone-over", "piece", "block", "zeke")]
COLS2 = [("blue", "orange", "concrete wall"), ("chrome", "black", "shutters"),
         ("pink", "green", "wall"), ("yellow", "black", "grey concrete wall"),
         ("green", "purple", "brick wall"), ("purple", "yellow", "concrete wall"),
         ("orange", "dark blue", "metal"), ("red", "white", "concrete wall")]


def grid_prompts2():
    """Rows = kind/condition (8), columns = color/background combos (8). Same training-caption template with the word."""
    out = []
    for ri, (cond, kind, style, word) in enumerate(ROWS2):
        for ci, (fill, outline, bg) in enumerate(COLS2):
            k = kind if kind != "wildstyle piece" else "piece"
            out.append((f"a {cond} {k} graffiti on a {bg} with the word {word} written in {fill} letters "
                        f"with a {outline} outline in a {style} style".replace(" a orange", " an orange"),
                        2000 + ri * 8 + ci))
    return out


def default_chi():
    """Sana's default 'complex human instruction' prefix, as used by the pipeline at inference."""
    import inspect
    from diffusers import SanaPipeline
    return inspect.signature(SanaPipeline.__call__).parameters["complex_human_instruction"].default


# ---- phase 3: kind-first captions -------------------------------------------------------------------------
KIND3 = {  # kind label -> concrete description of what that kind looks like
    "tag": "a quick single-colour tag scrawl, thin marker-like lines, small, among other tags",
    "throwup": "a throw-up: two-colour bubble letters, fill and outline",
    "piece": "a full-colour graffiti piece with 3D blocks and highlights",
    "character": "a cartoon character painted next to letters",
    "streetart": "a street art mural",
    "paste_up": "a paste-up poster glued to the wall",
    "stencil": "a spray stencil",
}
COND3 = {
    "fresh": "fresh, bright, crisp paint",
    "faded": "old faded weathered paint",
    "buffed": "buffed, painted over with a patch of mismatched grey paint",
    "gone_over": "gone over, covered by newer tags and layers of paint",
}


def caption3(d: dict, with_text: bool) -> str:
    """Kind first (concrete), condition second, then colours / style / wall, the word last (only if with_text)."""
    kind = d.get("kind") or "piece"
    parts = [KIND3.get(kind, "a graffiti piece")]
    if d.get("condition") in COND3:
        parts.append(COND3[d["condition"]])
    fill, outline, hl = d.get("fill") or "", d.get("outline") or "", d.get("highlights") or ""
    if fill:
        c = f"{fill} letters"
        if outline and kind != "tag":  # tags are single-colour by definition
            c += f" with a {outline} outline"
        if hl and kind in ("piece", "character"):
            c += f" and {hl} highlights"
        parts.append(c)
    if d.get("style") and kind != "tag":
        parts.append(f"{d['style']} style")
    bg = d.get("background") or ""
    if bg in ("brick", "concrete", "metal", "stone", "wood"):
        bg += " wall"
    if bg:
        parts.append(f"on {bg}" if bg.endswith("shutters") else f"on a {bg}")
    if with_text and d.get("text"):
        parts.append(f"the word {d['text']}")
    return re.sub(r"\ba (?=[aeiou])", "an ", re.sub(r"\s+", " ", ", ".join(parts)).strip())


# phase-3 grids: "new" = rows kind/condition x 8 colour/wall combos (kind-first captions), "kinds" = 4 kinds x 8 seeds, same words/colours
def grid_prompts3():
    out = []
    for ri, (cond, kind, style, word) in enumerate(ROWS2):
        k = {"character": "character", "throw-up": "throwup", "wildstyle piece": "piece"}.get(kind, kind)
        for ci, (fill, outline, bg) in enumerate(COLS2):
            d = {"kind": k, "condition": cond, "fill": fill, "outline": outline,
                 "style": {"simple": "simple", "clean": "clean"}.get(style, style),
                 "background": bg, "text": word, "highlights": ""}
            out.append((caption3(d, True), 2000 + ri * 8 + ci))
    return out


def grid_prompts_kinds():
    out = []
    for ki, k in enumerate(["tag", "throwup", "piece", "character"]):
        for si in range(8):
            fill, outline = [("blue", "orange"), ("pink", "green")][si // 4]
            d = {"kind": k, "condition": "fresh", "fill": fill, "outline": outline, "style": "block",
                 "background": "concrete wall", "text": "kroz", "highlights": ""}
            out.append((caption3(d, True), 3000 + si % 4))  # 4 seeds x 2 colour sets, identical across kinds
    return out
