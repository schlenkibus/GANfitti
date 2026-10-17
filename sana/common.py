"""Shared helpers for the Sana LoRA phase-1 tooling."""
import os, json
from pathlib import Path

MODEL_ID = "Efficient-Large-Model/Sana_600M_512px_diffusers"
ROOT = Path(os.environ.get("GRAFF_DATA", os.path.expanduser("~/Development/graff-data/dataset")))
TRAIN_DIR = ROOT / "train_phase1"
CACHE_DIR = ROOT / "cache" / "phase1"
LORA_DIR = ROOT / "lora" / "phase1"

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


def default_chi():
    """Sana's default 'complex human instruction' prefix, as used by the pipeline at inference."""
    import inspect
    from diffusers import SanaPipeline
    return inspect.signature(SanaPipeline.__call__).parameters["complex_human_instruction"].default
