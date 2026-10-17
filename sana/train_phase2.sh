#!/usr/bin/env bash
# Sana 600M LoRA, phase 2: full captioned set (5740 crops), balanced sampling, lr 5e-5, 3000 steps.
# usage: sana/train_phase2.sh      (captions -> cache -> train, with the new grid rendered after every checkpoint)
set -euo pipefail
cd "$(dirname "$0")"
export GRAFF_PHASE=phase2
PY=${PY:-$HOME/.venvs/sana/bin/python}
D=${GRAFF_DATA:-$HOME/Development/graff-data/dataset}
L=$D/lora/phase2
python3 ../create_text_dataset.py --pairs -o "$D/train_phase2"
[ -f "$D/cache/phase2/latents.pt" ] || $PY cache.py
mkdir -p "$L"
[ -f grids/phase2/new_base_step00000.jpg ] || $PY grid.py --base --grid new   # base model sheet (training has not started: no GPU contention)
$PY train.py --config config.phase2.json > "$L/train.log" 2>&1 &
TP=$!
for s in 00500 01000 01500 02000 02500 03000; do   # grid worker: shares the GPU with training (~2 GB)
	until [ -f "$L/step$s/pytorch_lora_weights.safetensors" ] || ! kill -0 $TP 2>/dev/null; do sleep 15; done
	sleep 5
	[ -f "$L/step$s/pytorch_lora_weights.safetensors" ] && $PY grid.py --grid new "$L/step$s" || echo "skip missing step$s"
done
wait $TP
$PY grid.py --grid old "$L"/step00500 "$L"/step01500 "$L"/step03000   # old grid, for the phase-1 comparison
$PY grid.py --base --grid old
