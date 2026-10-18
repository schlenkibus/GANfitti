"""Pass 1: precompute Gemma-2 text embeddings and DC-AE VAE latents, then free the encoders.

Writes CACHE_DIR/text.pt (name -> {text, alt} embeds+masks) and CACHE_DIR/latents.pt
(name -> list of K latent variants: 0 = centred crop, others = scale/offset jitter; no flips).
"""
import argparse, gc, random
import torch
from PIL import Image
from diffusers import SanaPipeline
from common import *


def trim_entry(d):
    """Cut right padding (mask is 1...1 0...0): 300 -> ~40 tokens, 8x smaller cache; identical text/alt share tensors."""
    out = {}
    for k, (e, m) in d.items():
        n = int(m.nonzero().max()) + 1
        assert int(m[:n].sum()) == n, "mask not contiguous"
        out[k] = (e[:n].clone(), m[:n].clone())
    if torch.equal(out["text"][0], out["alt"][0]) and out["text"][0].shape == out["alt"][0].shape:
        out["alt"] = out["text"]
    return out


def free():
    gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    names = sorted(p.stem for p in TRAIN_DIR.glob("*.jpg"))
    pipe = SanaPipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, torch_dtype=torch.bfloat16, variant=VARIANT)
    pipe.text_encoder.to(dev)

    def enc(prompts):
        if len(prompts) > 8:  # chunk: the 300-token CHI prefix makes big batches OOM on 8 GB
            parts = [enc(prompts[i:i + 8]) for i in range(0, len(prompts), 8)]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        e, m, _, _ = pipe.encode_prompt(prompts, do_classifier_free_guidance=False, device=dev,
                                        clean_caption=False, max_sequence_length=MAX_SEQ,
                                        complex_human_instruction=default_chi())
        return e.cpu(), m.cpu()

    if (CACHE_DIR / "text.pt").exists() and (CACHE_DIR / ("grid_text_kinds.pt" if PHASE == "phase3" else "grid_text_new.pt")).exists():
        print("text.pt exists, skipping the text pass", flush=True)
        del pipe; free()
    else:
        text = {}
        for i, n in enumerate(names):
            t = (TRAIN_DIR / f"{n}.txt").read_text().strip()
            al = (TRAIN_DIR / f"{n}.alt.txt").read_text().strip()
            e, m = enc([t, al] if t != al else [t])
            if t == al:
                e, m = e.repeat(2, 1, 1), m.repeat(2, 1)
            text[n] = trim_entry({"text": (e[0], m[0]), "alt": (e[1], m[1])})
            if i % 100 == 0:
                print("text", i, len(names), flush=True)
        # prompt grid embeddings (+ empty negative) so grid rendering never needs the encoder again
        if PHASE == "phase3":
            for gname, gp in (("new", grid_prompts3()), ("kinds", grid_prompts_kinds())):
                e, m = enc([p for p, _ in gp] + [NEGATIVE])
                torch.save({"embeds": e, "mask": m, "prompts": gp}, CACHE_DIR / f"grid_text_{gname}.pt")
        else:
            gp = grid_prompts()
            e, m = enc([p for p, _ in gp] + [NEGATIVE])
            torch.save({"embeds": e, "mask": m, "prompts": gp}, CACHE_DIR / "grid_text.pt")
            gp2 = grid_prompts2()
            e, m = enc([p for p, _ in gp2] + [NEGATIVE])
            torch.save({"embeds": e, "mask": m, "prompts": gp2}, CACHE_DIR / "grid_text_new.pt")
        torch.save(text, CACHE_DIR / "text.pt")
        print("text encoder peak MB", torch.cuda.max_memory_allocated() / 2**20, flush=True)
        del pipe, text; free(); torch.cuda.reset_peak_memory_stats()


    from diffusers import AutoencoderDC
    import numpy as np
    vae = AutoencoderDC.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, variant=VARIANT).to(dev)
    sf = vae.config.scaling_factor
    part_f = CACHE_DIR / "latents.part.pt"
    lat = torch.load(part_f) if part_f.exists() else {}
    tags = set()
    if PHASE == "phase3":  # same VAE as phase 2 -> reuse its latents; only tags are re-encoded (tight crop, stronger zoom)
        import json
        tags = {n for n in names if json.load(open(ROOT / "auto_labels" / f"{n}.json")).get("kind") == "tag"}
        if not lat:
            old = torch.load(ROOT / "cache" / "phase2" / "latents.pt")
            lat = {n: old[n] for n in names if n in old and n not in tags}
            del old
    todo = [n for n in names if n not in lat]
    print("vae: done", len(lat), "todo", len(todo), flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            n = todo[i]
            rng = random.Random(f"{a.seed}-{n}")  # per-image seed: independent of order / resume
            img = Image.open(TRAIN_DIR / f"{n}.jpg").convert("RGB")
            w, h = img.size
            bw, bh = nearest_bucket(w, h)
            vs = []
            tight = n in tags
            if tight:  # drop the 3 % AOI margin that recrop.py added on each side: inner 1/1.06 of the crop
                mx, my = round(w * 0.03 / 1.06), round(h * 0.03 / 1.06)
                img = img.crop((mx, my, w - mx, h - my)); w, h = img.size
                bw, bh = nearest_bucket(w, h)
            for v in range(a.variants):
                jit = 1.0 if v == 0 else rng.uniform(1.0, 1.5 if tight else 1.25)
                s = max(bw / w, bh / h) * jit
                rw, rh = max(bw, round(w * s)), max(bh, round(h * s))
                im = img.resize((rw, rh), Image.LANCZOS)
                if v == 0:
                    x0, y0 = (rw - bw) // 2, (rh - bh) // 2
                else:
                    x0, y0 = rng.randint(0, rw - bw), rng.randint(0, rh - bh)
                im = im.crop((x0, y0, x0 + bw, y0 + bh))
                vs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1)
            return n, torch.stack(vs)

    # batches of up to 8 images of the same bucket
    by_b = {}
    for i, n in enumerate(todo):
        w, h = Image.open(TRAIN_DIR / f"{n}.jpg").size
        if n in tags:
            w, h = w - 2 * round(w * 0.03 / 1.06), h - 2 * round(h * 0.03 / 1.06)
        by_b.setdefault(nearest_bucket(w, h), []).append(i)
    batches = [idx[j:j + 8] for idx in by_b.values() for j in range(0, len(idx), 8)]
    dl = torch.utils.data.DataLoader(DS(), batch_sampler=batches, num_workers=4, collate_fn=lambda b: b)
    done = 0
    for batch in dl:
        x = torch.cat([t for _, t in batch]).to(dev, torch.bfloat16)  # (B*variants, 3, H, W)
        with torch.no_grad():
            z = torch.cat([(vae.encode(x[j:j + 8]).latent * sf).to(torch.bfloat16).cpu() for j in range(0, len(x), 8)])  # 8 tensors per forward (8 GB)
        for k, (n, _) in enumerate(batch):
            lat[n] = [z[k * a.variants + v] for v in range(a.variants)]
        done += len(batch)
        if done % 500 < len(batch):
            torch.save(lat, part_f)
            print("vae", len(lat), len(names), flush=True)
    torch.save(lat, CACHE_DIR / "latents.pt")
    part_f.unlink(missing_ok=True)
    print("vae peak MB", torch.cuda.max_memory_allocated() / 2**20, "scaling", sf)


if __name__ == "__main__":
    main()
