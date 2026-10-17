"""Shared label vocabulary and helpers for the auto-caption tooling.

The label JSON keeps the original GANfitti schema ({text, background, fill, outline, style, highlights},
comma-joined strings) and adds `kind` and `condition`. Auto labels also carry `auto: true`.
"""
import re

KINDS = ["tag", "throwup", "piece", "character", "streetart", "paste_up", "stencil"]
CONDITIONS = ["fresh", "faded", "buffed", "gone_over"]
BACKGROUNDS = ["brick wall", "wall", "concrete wall", "stone wall", "metal", "shutters", "garage",
               "container", "electric box", "fence", "pipe", "train", "truck", "wood", "other"]
STYLES = ["block", "simple", "rounded", "bubble", "toy", "semi wildstyle", "wildstyle", "hollow",
          "clean", "tag", "roll", "thick", "readable", "abstract", "character"]  # most frequent first (enum order biases small models)
COLORS = ["white", "black", "chrome", "grey", "red", "orange", "yellow", "green", "light green",
          "dark green", "blue", "light blue", "dark blue", "purple", "pink", "light pink", "brown",
          "beige", "bronze", "gold", "turquoise"]

# keyword -> enum, first hit wins (used to normalise the free-text hand labels)
_BG_KEYS = [("brick", "brick wall"), ("concrete", "concrete wall"), ("stone", "stone wall"),
            ("metal", "metal"), ("shutter", "shutters"), ("garage", "garage"), ("container", "container"),
            ("electric", "electric box"), ("fence", "fence"), ("pipe", "pipe"), ("train", "train"),
            ("ubahn", "train"), ("truck", "truck"), ("wood", "wood"), ("wall", "wall")]
_COLOR_ALIAS = {"silver": "chrome", "gray": "grey", "light grey": "grey", "dark grey": "grey",
                "bronze": "bronze", "cyan": "turquoise", "teal": "turquoise", "violet": "purple"}


def norm_background(s: str) -> str:
    s = (s or "").lower()
    for k, v in _BG_KEYS:
        if k in s:
            return v
    return "other" if s.strip() else ""


def norm_style(s: str) -> list:
    s = (s or "").lower()
    found = []
    if "semi wildstyle" in s or "semi-wildstyle" in s:
        found.append("semi wildstyle")
        s = s.replace("semi wildstyle", "").replace("semi-wildstyle", "")
    for t in re.split(r"[,/;]| and ", s):
        t = t.strip()
        if t == "bubbly":
            t = "bubble"
        if t == "wild style":
            t = "wildstyle"
        if t in STYLES and t not in found:
            found.append(t)
    return found


def norm_colors(s: str) -> list:
    out = []
    for t in re.split(r"[,/;]| and ", (s or "").lower()):
        t = t.strip()
        t = _COLOR_ALIAS.get(t, t)
        if t in COLORS and t not in out:
            out.append(t)
    return out


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def json_schema() -> dict:
    colors = {"type": "array", "items": {"type": "string", "enum": COLORS}, "maxItems": 2}
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS},
            "condition": {"type": "string", "enum": CONDITIONS},
            "text": {"type": "string", "pattern": "^[a-z0-9 .]{0,12}$"},
            "background": {"type": "string", "enum": BACKGROUNDS},
            "style": {"type": "array", "items": {"type": "string", "enum": STYLES}, "minItems": 1, "maxItems": 3},
            "fill": {**colors, "minItems": 1},
            "outline": colors,
            "highlights": colors,
        },
        "required": ["kind", "condition", "text", "background", "style", "fill", "outline", "highlights"],
        "additionalProperties": False,
    }


def to_label(pred: dict) -> dict:
    """Model output (arrays) -> label JSON (comma-joined strings, original schema order + kind/condition)."""
    j = lambda v: ", ".join(v) if isinstance(v, list) else (v or "")
    return {"text": (pred.get("text") or "").strip().lower(), "background": pred.get("background", ""),
            "fill": j(pred.get("fill")), "outline": j(pred.get("outline")), "style": j(pred.get("style")),
            "highlights": j(pred.get("highlights")), "kind": pred.get("kind", ""),
            "condition": pred.get("condition", "")}
