#!/usr/bin/env bash
# Sana 600M LoRA, phase 2: full captioned set (5740 crops), balanced sampling, lr 5e-5, 3000 steps.
# usage: sana/train_phase2.sh      (captions -> cache -> train -> grids; each step is one process)
set -euo pipefail
cd "$(dirname "$0")"
export GRAFF_PHASE=phase2
PY=${PY:-$HOME/.venvs/sana/bin/python}
D=${GRAFF_DATA:-$HOME/Development/graff-data/dataset}
python3 ../create_text_dataset.py --pairs -o "$D/train_phase2"
[ -f "$D/cache/phase2/latents.pt" ] || $PY cache.py
$PY train.py --config config.phase2.json
$PY grid.py --base --grid new "$D"/lora/phase2/step*/
$PY grid.py --base --grid old "$D"/lora/phase2/step*/
