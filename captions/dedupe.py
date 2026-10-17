"""Group near-duplicate crops (same piece shot several times). Deletes nothing.

Signals: perceptual hash (DCT pHash, 64 bit) and CLIP ViT-B/32 image similarity (MIT weights).
Two crops are linked when
  * pHash hamming distance <= --phash-max (anywhere in the set), or
  * CLIP cosine >= --clip-min, the photos are neighbours (same folder, file number within --window)
    and ORB+RANSAC finds >= --inliers matching local features (or CLIP >= 0.985).
Linked crops form a group (union-find); the sharpest crop (variance of Laplacian) is kept.
Output: dupes.csv with rows group,kept,dropped (one row per dropped crop).
Run with the sana venv:  ~/.venvs/sana/bin/python captions/dedupe.py
"""
import argparse, csv, os, re, sys
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(os.path.expanduser("~/Development/graff-data/dataset"))


def phash(img: Image.Image) -> int:
    a = np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float64)
    n = 32
    k = np.arange(n)
    dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))  # rows: freq
    f = dct @ a @ dct.T
    low = f[:8, :8].ravel()
    bits = low > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def sharpness(img: Image.Image) -> float:
    g = np.asarray(img.convert("L").resize((512, 512), Image.BILINEAR), dtype=np.float64)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--phash-max", type=int, default=6)
    ap.add_argument("--clip-min", type=float, default=0.90)
    ap.add_argument("--inliers", type=int, default=25, help="min ORB+RANSAC inliers to confirm a CLIP candidate")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--cache", default=None, help="npz cache of features")
    ap.add_argument("--stats", action="store_true", help="print similarity stats only")
    a = ap.parse_args()
    data = Path(a.data)
    names = sorted(p.name for p in (data / "crops").glob("*.jpg"))
    cache = Path(a.cache) if a.cache else data / "cache" / "dedupe_features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        assert list(z["names"]) == names
        hashes, sharp, emb = z["hashes"], z["sharp"], z["emb"]
    else:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().half()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        hashes, sharp, embs = [], [], []
        batch = []
        def flush():
            if not batch: return
            inp = proc(images=batch, return_tensors="pt")["pixel_values"].to(dev).half()
            with torch.no_grad():
                e = model.get_image_features(pixel_values=inp)
                e = e if isinstance(e, torch.Tensor) else e.pooler_output
                e = torch.nn.functional.normalize(e.float(), dim=-1)
            embs.append(e.cpu().numpy()); batch.clear()
        for i, n in enumerate(names):
            im = Image.open(data / "crops" / n).convert("RGB")
            im.thumbnail((512, 512))
            hashes.append(str(phash(im))); sharp.append(sharpness(im)); batch.append(im.copy())
            if len(batch) == 64: flush()
        flush()
        hashes = np.array(hashes, dtype=object); sharp = np.array(sharp); emb = np.concatenate(embs)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, names=np.array(names), hashes=hashes, sharp=sharp, emb=emb)
    hv = [int(h) for h in hashes]
    n = len(names)
    sim = emb @ emb.T
    m = [re.match(r"(.*?)_(\d+)\.jpg$", x) for x in names]
    photo = [mm.group(1) for mm in m]
    def num(p):
        mm = re.search(r"(\d+)$", p); return int(mm.group(1)) if mm else -10**9
    pn = [num(p) for p in photo]
    pre = [re.sub(r"\d+$", "", p) for p in photo]
    uf = UF(n)
    ham = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        ham[i] = [bin(hv[i] ^ hv[j]).count("1") for j in range(n)]
    np.fill_diagonal(ham, 99)
    if a.stats:
        iu = np.triu_indices(n, 1)
        for t in (0.85, 0.9, 0.93, 0.95, 0.97):
            print("clip>=", t, int((sim[iu] >= t).sum()))
        for t in (2, 4, 6, 8, 10):
            print("phash<=", t, int((ham[iu] <= t).sum()))
        return
    import cv2
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    feats = {}

    def feat(i):
        if i not in feats:
            g = cv2.imread(str(data / "crops" / names[i]), cv2.IMREAD_GRAYSCALE)
            s = 640 / max(g.shape)
            if s < 1: g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            feats[i] = orb.detectAndCompute(g, None)
        return feats[i]

    def inliers(i, j) -> int:
        (ka, da), (kb, db) = feat(i), feat(j)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8: return 0
        good = [m for m, n2 in (p for p in bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n2.distance]
        if len(good) < 8: return 0
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return 0 if mask is None else int(mask.sum())

    links = 0
    cand = 0
    for i in range(n):
        for j in range(i + 1, n):
            if photo[i] == photo[j]:
                continue  # different pieces of one photo are different pieces
            near = pre[i] == pre[j] and abs(pn[i] - pn[j]) <= a.window
            if ham[i, j] <= a.phash_max:
                uf.union(i, j); links += 1
            elif near and sim[i, j] >= a.clip_min:
                cand += 1
                # CLIP alone cannot tell "same piece" from "same writer, same wall": verify with local features
                if sim[i, j] >= 0.985 or inliers(i, j) >= a.inliers:
                    uf.union(i, j); links += 1
    print(f"clip candidates {cand}")
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    multi = sorted((sorted(v) for v in groups.values() if len(v) > 1), key=lambda v: names[v[0]])
    rows = 0
    with open(data / "dupes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "kept", "dropped"])
        for gi, g in enumerate(multi):
            keep = max(g, key=lambda i: sharp[i])
            for i in g:
                if i != keep:
                    w.writerow([gi, names[keep], names[i]]); rows += 1
    print(f"crops {n}, links {links}, groups {len(multi)}, dropped {rows}, kept {n - rows}")


if __name__ == "__main__":
    main()
