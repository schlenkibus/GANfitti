# Auto-captions (graf-sim #360)

Tooling that labels the GANfitti crops with a local vision-language model and lets a human review them.
Nothing here ships photos, crops or labels: only scripts, the blocklist and small stats.

Data lives outside the repo in `~/Development/graff-data/dataset/` (`crops/`, `text_labels/` hand labels,
`auto_labels/` machine labels, `dupes.csv`, `train/`).

| Step | Command |
|---|---|
| 1. Dedupe (pHash + CLIP + ORB check; deletes nothing) | `~/.venvs/sana/bin/python captions/dedupe.py` -> `dataset/dupes.csv` |
| 2. Serve the model | `~/.venvs/vllm/bin/vllm serve Qwen/Qwen3-VL-2B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.7 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":589824}' --max-num-seqs 8 --max-num-batched-tokens 2048 --enforce-eager --port 8000` |
| 3. Held-out check (50 hand labels never used as few-shot) | `~/.venvs/vllm/bin/python captions/auto_caption.py --eval --tag qwen3vl2b_v2` -> `captions/eval_<tag>.json` |
| 4. Full run (resumable: skips existing JSON) | `~/.venvs/vllm/bin/python captions/auto_caption.py --run` -> `dataset/auto_labels/` |
| 5. Review queue | `~/.venvs/vllm/bin/python create_text_labels_editor.py --auto ~/Development/graff-data/dataset/auto_labels -i ~/Development/graff-data/dataset/crops --sample 100` |
| 6. Training captions | `python3 create_text_dataset.py` -> `dataset/train/` (add `--reviewed-only` for checked labels) |
| Blocklist | `python3 captions/make_blocklist.py` -> `data/blocklist.txt` |
| Stats | `python3 captions/stats.py` -> `captions/stats.json` |

Stop the server when you are done (the GPU is shared; never run it next to training).

## Label schema
`{text, background, fill, outline, style, highlights}` as in the hand labels (comma-joined strings), plus
`kind` (tag, throwup, piece, character, streetart, paste_up, stencil) and `condition` (fresh, faded, buffed,
gone_over). Auto labels carry `auto: true`; for the 538 hand-labeled crops the hand fields are kept and only
`kind`/`condition` come from the model (`hand: true`). Review writes `reviewed`, `fixed_fields`, `rejected`.
Enums for background, style and colors are in `captions/schema.py`; vLLM structured output keeps every
enum field valid.

## Review keys
Enter/a accept, e edit (Enter saves, Esc cancels), x reject, s skip, z undo. `--stats` prints agreement
per field for the sample.

## Results (Qwen3-VL-2B-Instruct, Apache 2.0)
Held-out check on 50 hand labels never used as few-shot (`heldout.txt`; numbers in `eval_*.json`):
`eval_qwen3vl2b.json` is the first prompt, `eval_qwen3vl2b_v2.json` the final prompt and enum order,
`eval_qwen3vl4b_awq.json` the 4-bit 4B comparison (cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit, Apache 2.0).
`kind` and `condition` have no hand labels, so they are only checked by the review queue.
Distribution of the whole set: `stats.json`. The 2B model over-predicts `piece` and `faded`; streetart,
paste_up and stencil are almost absent, so those kinds need other data.
