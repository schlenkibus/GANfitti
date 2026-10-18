#!/usr/bin/env bash
# Sana 1.6B LoRA, phase 3: kind-first captions, stronger kind balancing (cap 8x), tight tag crops, lr 4e-5, 2000 steps.
# usage: sana/train_phase3.sh   (captions -> cache -> base grids -> train; grids rendered in-process at every checkpoint)
set -euo pipefail
cd "$(dirname "$0")"
export GRAFF_PHASE=phase3
PY=${PY:-$HOME/.venvs/sana/bin/python}
D=${GRAFF_DATA:-$HOME/Development/graff-data/dataset}
L=$D/lora/phase3
python3 make_captions3.py
[ -f "$D/cache/phase3/latents.pt" ] || $PY cache.py
mkdir -p "$L"
[ -f grids/phase3/kinds_base_step00000.jpg ] || $PY grid.py --base --grid new,kinds
$PY train.py --config config.phase3.json > "$L/train.log" 2>&1
