"""One-off: shrink an untrimmed text.pt (300-token padded, ~16 GB for 5740 crops) in place via mmap. See cache.trim_entry."""
import torch
from common import *
from cache import trim_entry

f = CACHE_DIR / "text.pt"
t = torch.load(f, mmap=True)
out = {n: trim_entry(d) for n, d in t.items()}
del t
torch.save(out, CACHE_DIR / "text_trim.pt")
f.unlink()
(CACHE_DIR / "text_trim.pt").rename(f)
print("trimmed", len(out), "max len", max(d["text"][0].shape[0] for d in out.values()))
