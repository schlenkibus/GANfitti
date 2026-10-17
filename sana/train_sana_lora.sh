#!/usr/bin/env bash
# Sana 600M LoRA, phase 1 (538 hand-labeled crops). Fits 8 GB: bf16, batch 1 + grad accum,
# gradient checkpointing, 8-bit AdamW, cached Gemma-2 embeddings and VAE latents.
# usage: sana/train_sana_lora.sh [train.py flags]    (steps: captions -> cache -> train -> grids)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.venvs/sana/bin/python}
python3 make_captions.py
[ -f "${GRAFF_DATA:-$HOME/Development/graff-data/dataset}/cache/phase1/latents.pt" ] || $PY cache.py
D=${GRAFF_DATA:-$HOME/Development/graff-data/dataset}
$PY train.py --config config.phase1.json "$@"
$PY grid.py --base "$D"/lora/phase1/step*/
