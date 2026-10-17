# Sana LoRA (graffgen phase 1)

Sana 600M 512 px LoRA on the 538 hand-labeled crops, tuned to fit an 8 GB GPU (RTX 3070 Ti).
Design: graf-sim `docs/design/graffgen.md` §2. Photos, crops, captions, caches and weights are **not** in this repo.

Data layout (`$GRAFF_DATA`, default `~/Development/graff-data/dataset`): `crops/`, `text_labels/`, then generated
`train_phase1/` (symlinks + captions), `cache/phase1/`, `lora/phase1/`.

| file | what |
|---|---|
| `make_captions.py` | labels -> two captions per crop: the `create_text_dataset.py` template (real word) and a placeholder-free variant; training samples each ~50 % |
| `cache.py` | pass 1: Gemma-2 embeddings + DC-AE latents (aspect buckets, 4 crop/scale-jitter variants, no flips), then frees the encoders |
| `train.py` | pass 2: LoRA on attention projections, bf16 + fp32 LoRA master weights, 8-bit AdamW, gradient checkpointing, batch 1 x accumulation |
| `grid.py` | fixed prompt grid (8 kinds x 4 styles x 2 colors, invented words, fixed seeds) -> `grids/<ckpt>.jpg` |
| `train_sana_lora.sh`, `config.phase1.json` | wrapper + settings |

Setup: `python3 -m venv ~/.venvs/sana && ~/.venvs/sana/bin/pip install torch torchvision diffusers transformers peft accelerate bitsandbytes sentencepiece protobuf`.
Grid layout: columns = kind (tag, throw-up, piece, character, street art, paste-up, stencil, wildstyle piece), rows = style (block, bubble, semi wildstyle, simple rounded) x color set (white/black, chrome/red), two rows per style.

## Phase 1 result (2000 steps, rank 32, alpha 32, lr 1e-4, 4x accumulation, batch 1)
Peak VRAM 1.37 GB allocated / 1.46 GB reserved (training), ~2.2 GB for grid rendering; text encoder caching peaks at 5.7 GB. 1.36 s per optimizer step (4 micro-batches), 45 min total.
Grids in `grids/`: `base_step00000.jpg` is the untouched base model, `stepNNNNN.jpg` the checkpoints. Same seed and prompt per tile across all sheets.
