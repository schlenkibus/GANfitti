"""Render fixed prompt grids for the base model and each LoRA checkpoint.

usage: grid.py [--base] [--grid old|new|kinds] [ckpt_dir ...]
  -> sana/grids/<name>.jpg (phase1) or sana/grids/<phase>/[new_|kinds_]<name>.jpg (contact sheet, 192 px tiles)
  old grid: columns = kind, rows = style x color (white/chrome on brick)
  new grid: rows = kind/condition, columns = color/background combos (phase 3: kind-first captions)
  kinds grid (phase 3): rows = tag / throw-up / piece / character, columns = 4 seeds x 2 color sets, same word everywhere
"""
import argparse, sys
import torch
from PIL import Image, ImageDraw
from diffusers import SanaPipeline
from common import *

TILE = 192
HERE = Path(__file__).parent
GRID_FILE = {"old": "grid_text.pt", "new": "grid_text_new.pt", "kinds": "grid_text_kinds.pt"}


@torch.no_grad()
def render(pipe, grid: str, out: Path, steps: int = 20, cfg: float = 4.5, device="cuda"):
    """Render grid `grid` with `pipe` (text encoder-less) into the contact sheet `out`."""
    g = torch.load(CACHE_DIR / GRID_FILE[grid])
    emb, msk, prompts = g["embeds"], g["mask"], g["prompts"]
    tiles = []
    for i, (p, seed) in enumerate(prompts):
        img = pipe(prompt_embeds=emb[i:i+1].to(device), prompt_attention_mask=msk[i:i+1].to(device),
                   negative_prompt=None, negative_prompt_embeds=emb[-1:].to(device), negative_prompt_attention_mask=msk[-1:].to(device),
                   height=512, width=512, num_inference_steps=steps, guidance_scale=cfg,
                   generator=torch.Generator(device).manual_seed(seed)).images[0]
        tiles.append(img.resize((TILE, TILE), Image.LANCZOS))
    rows = (len(tiles) + 7) // 8
    sheet = Image.new("RGB", (8 * TILE, rows * TILE))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 8) * TILE, (i // 8) * TILE))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=60, optimize=True)


def sheet_path(grid: str, name: str) -> Path:
    gdir = HERE / "grids" / ("" if PHASE == "phase1" else PHASE)
    return gdir / f"{grid + '_' if grid in ('new', 'kinds') else ''}{name}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--grid", default="old", help="old, new, kinds or a comma list")
    ap.add_argument("--cfg", type=float, default=4.5)
    a = ap.parse_args()
    pipe = SanaPipeline.from_pretrained(MODEL_ID, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16, variant=VARIANT)
    pipe.to("cuda"); pipe.vae.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    jobs = ([("base_step00000", None)] if a.base else []) + [(Path(c).name, c) for c in a.ckpts]
    for name, ck in jobs:
        pipe.unload_lora_weights()
        if ck:
            pipe.load_lora_weights(ck)
        for grid in a.grid.split(","):
            render(pipe, grid, sheet_path(grid, name), a.steps, a.cfg)
            print("wrote", sheet_path(grid, name), "peakMB", torch.cuda.max_memory_allocated() / 2**20, flush=True)


if __name__ == "__main__":
    main()
