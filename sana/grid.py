"""Render the fixed prompt grid (8 kinds x 4 styles x 2 colors) for the base model and each LoRA checkpoint.

usage: grid.py [--base] [--grid old|new] [ckpt_dir ...]
  -> sana/grids/<name>.jpg (phase1) or sana/grids/<phase>/[new_]<name>.jpg (contact sheet, 192 px tiles)
  old grid: columns = kind, rows = style x color (white/chrome on brick); new grid (phase 2): rows = kind/condition,
  columns = non-white-on-red-brick color/background combos
"""
import argparse, sys
import torch
from PIL import Image, ImageDraw
from diffusers import SanaPipeline
from common import *

TILE = 192
HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--grid", default="old", choices=["old", "new"])
    ap.add_argument("--cfg", type=float, default=4.5)
    a = ap.parse_args()
    g = torch.load(CACHE_DIR / ("grid_text.pt" if a.grid == "old" else "grid_text_new.pt"))
    emb, msk, prompts = g["embeds"], g["mask"], g["prompts"]
    pipe = SanaPipeline.from_pretrained(MODEL_ID, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.vae.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    jobs = ([("base_step00000", None)] if a.base else []) + [(Path(c).name, c) for c in a.ckpts]
    gdir = HERE / "grids" / ("" if PHASE == "phase1" else PHASE)
    gdir.mkdir(parents=True, exist_ok=True)
    for name, ck in jobs:
        pipe.unload_lora_weights()
        if ck:
            pipe.load_lora_weights(ck)
        tiles = []
        for i, (p, seed) in enumerate(prompts):
            img = pipe(prompt_embeds=emb[i:i+1].cuda(), prompt_attention_mask=msk[i:i+1].cuda(),
                       negative_prompt=None, negative_prompt_embeds=emb[-1:].cuda(), negative_prompt_attention_mask=msk[-1:].cuda(),
                       height=512, width=512, num_inference_steps=a.steps, guidance_scale=a.cfg,
                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            tiles.append(img.resize((TILE, TILE), Image.LANCZOS))
        sheet = Image.new("RGB", (8 * TILE, 8 * TILE))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 8) * TILE, (i // 8) * TILE))
        sheet.save(gdir / f"{'new_' if a.grid == 'new' else ''}{name}.jpg", quality=60, optimize=True)
        print("wrote", name, "peakMB", torch.cuda.max_memory_allocated() / 2**20, flush=True)


if __name__ == "__main__":
    main()
