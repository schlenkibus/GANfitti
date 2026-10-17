"""Pass 2: Sana LoRA training from cached embeddings/latents (no text encoder / VAE in VRAM)."""
import argparse, json, random, time, math
import torch
from diffusers import SanaTransformer2DModel, FlowMatchEulerDiscreteScheduler, SanaPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling
from peft import LoraConfig, get_peft_model_state_dict
import bitsandbytes as bnb
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON file; CLI flags override")
    ap.add_argument("--steps", type=int, default=1750)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--p_alt", type=float, default=0.5, help="prob. of the placeholder-free caption")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--out", default=str(LORA_DIR))
    args = ap.parse_args()
    if args.config:
        for k, v in json.load(open(args.config)).items():
            if getattr(args, k) == ap.get_default(k):
                setattr(args, k, v)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    rng = random.Random(args.seed); torch.manual_seed(args.seed)

    text = torch.load(CACHE_DIR / "text.pt")
    lat = torch.load(CACHE_DIR / "latents.pt")
    names = sorted(lat)
    print(len(names), "images", flush=True)

    tr = SanaTransformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16).to(dev)
    tr.requires_grad_(False)
    if not args.no_grad_ckpt:
        tr.enable_gradient_checkpointing()
    tr.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian",
                              target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in tr.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # fp32 master weights for the LoRA
    print("trainable params", sum(p.numel() for p in params), flush=True)
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=1e-2)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    sigmas_all = sched.sigmas.to(dev); ts_all = sched.timesteps.to(dev)
    tr.train()

    def save(tag):
        d = out / tag; d.mkdir(exist_ok=True)
        SanaPipeline.save_lora_weights(d, transformer_lora_layers=get_peft_model_state_dict(tr))
        print("saved", d, flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); tl = time.time(); losses = []
    order = []
    for step in range(1, args.steps + 1):
        for _ in range(args.grad_accum):
            if not order:
                order = names[:]; rng.shuffle(order)
            n = order.pop()
            z = rng.choice(lat[n])[None].to(dev, torch.bfloat16)
            e, m = text[n]["alt" if rng.random() < args.p_alt else "text"]
            e = e[None].to(dev, torch.bfloat16); m = m[None].to(dev)
            noise = torch.randn_like(z)
            u = compute_density_for_timestep_sampling("logit_normal", 1, 0.0, 1.0)
            idx = (u * sched.config.num_train_timesteps).long().clamp(0, len(ts_all) - 1)
            sigma = sigmas_all[idx].view(1, 1, 1, 1).to(torch.bfloat16)
            noisy = (1 - sigma) * z + sigma * noise
            pred = tr(noisy, encoder_hidden_states=e, encoder_attention_mask=m,
                      timestep=ts_all[idx].to(torch.bfloat16)).sample
            loss = ((pred.float() - (noise - z).float()) ** 2).mean() / args.grad_accum
            loss.backward()
            losses.append(loss.item() * args.grad_accum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            now = time.time()
            print(f"step {step} loss {sum(losses[-100:])/len(losses[-100:]):.4f} "
                  f"{(now-tl)/25:.2f}s/step({args.grad_accum} micro) peakMB {torch.cuda.max_memory_allocated()/2**20:.0f}", flush=True)
            tl = now
        if step % args.ckpt_every == 0 or step == args.steps:
            save(f"step{step:05d}")
    tot = time.time() - t0
    stats = {"steps": args.steps, "grad_accum": args.grad_accum, "rank": args.rank,
             "total_s": tot, "s_per_step": tot / args.steps,
             "peak_alloc_MB": torch.cuda.max_memory_allocated() / 2**20,
             "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20, "args": vars(args)}
    json.dump(stats, open(out / "train_stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
