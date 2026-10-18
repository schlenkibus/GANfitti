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

## Phase 2 (full captioned set, 5740 crops)
`GRAFF_PHASE=phase2 sana/train_phase2.sh`: captions (`create_text_dataset.py --pairs`: auto labels never carry a word, only the hand labels do, ~50 % via `p_alt`)
-> `cache.py` (Gemma embeddings trimmed to real length, ~0.7 GB instead of 16 GB; VAE in bf16, bucketed batches, DataLoader, resumable `latents.part.pt`)
-> `train.py --config config.phase2.json` (rank 32, alpha 32, lr 5e-5, 3000 steps, 4x accumulation, **balanced sampling**: weight = min(5, kind x fill x background rarity), seeded)
-> grids. Grids in `grids/phase2/`: `new_*.jpg` = new grid (rows kind/condition, columns non-white-on-red-brick color/background combos, invented words); `*.jpg` = old grid (white/chrome on brick).
Result: 1.47 s/step, 74 min for 3000 steps, peak 1.36 GB allocated / 1.46 GB reserved (training); VAE caching peak 5.0 GB; text caching ~5.7 GB.

## Phase 3 (Sana 1.6B, kinds): `train_phase3.sh`, `config.phase3.json`, `make_captions3.py`
Base `Sana_1600M_512px_diffusers` (Apache 2.0; same Gemma-2 encoder and DC-AE VAE as 600M, identical file hashes). Captions are kind-first and concrete
(`common.caption3`: tag = "quick single-colour tag scrawl, thin marker-like lines, small, among other tags", throw-up = "two-colour bubble letters, fill and outline",
piece = "full-colour piece with 3D blocks and highlights", character = "cartoon character painted next to letters"), condition second, colours / style / wall after.
Kind weight cap 8x (kinds with < 30 crops are not boosted), tag latents re-encoded from the crop without the 3 % AOI margin and with up to 1.5x zoom jitter,
other latents reused from phase 2. lr 4e-5, rank/alpha 32, 2000 steps, 4x accumulation, checkpoint + two grids every 250 steps (rendered in-process by `train.py --grids new,kinds`,
because the 1.6B pipeline does not fit next to a training process in 8 GB). Grids in `grids/phase3/` (`new_*` = rows kind/condition, `kinds_*` = tag/throw-up/piece/character x 8 seeds/colours).
Training 0.97 s/step, 32 min for 2000 steps (+23 min of grids); peak VRAM 3.45 GB allocated / 3.38 GB reserved in training, 4.3 GB while rendering grids; text caching 5.7 GB, VAE 4.2 GB.
