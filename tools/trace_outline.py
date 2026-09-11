#!/usr/bin/env python3
"""
Trace a car side-profile image into a T-normalised outline for the proportion tool.
  python tools/trace_outline.py image.png --name "BMW 3 Series" --key D [--out outlines/D.json] [--preview]
Assumes a straight side view on a plain background. Output: JSON with top/bottom outlines in
tire-diameter units, axle positions, and overall length; plus an optional preview PNG.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

def mask_car(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB")).astype(np.int32)
    h, w, _ = a.shape
    # background colour = median of the border pixels
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.abs(a - bg).sum(axis=2)
    m = diff > 60
    # keep the largest blob
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n == 0: raise SystemExit("no foreground found")
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m)
    return m

def wheels(mask: np.ndarray):
    """Wheel centres and radius from the two lowest 'bumps' of the mask: columns where the mask
    reaches the ground line. Returns (fa_x, ra_x, radius_px, ground_y)."""
    h, w = mask.shape
    bottom = np.array([np.max(np.nonzero(mask[:, x])[0]) if mask[:, x].any() else -1 for x in range(w)])
    ground = int(np.percentile(bottom[bottom > 0], 99))
    on_ground = bottom >= ground - max(2, h // 300)
    # runs of ground contact = tyres
    runs, start = [], None
    for x in range(w):
        if on_ground[x] and start is None: start = x
        if (not on_ground[x] or x == w - 1) and start is not None:
            runs.append((start, x)); start = None
    runs = sorted(runs, key=lambda r: -(r[1] - r[0]))[:2]
    runs = sorted(runs)
    if len(runs) < 2: raise SystemExit("could not find two tyres on the ground line")
    centres = [(r[0] + r[1]) / 2 for r in runs]
    # tyre diameter: height of the mask directly above the contact patch until the arch opens (first gap) -> use chord: contact width ~ 0.55*D
    widths = [r[1] - r[0] for r in runs]
    # better: measure the tyre as the vertical extent of the foreground at the centre column that is
    # contiguous from the ground up until the first background pixel (the arch gap) or the body
    diam = []
    for cx in centres:
        col = mask[:, int(cx)]
        y = ground
        while y > 0 and col[y]: y -= 1
        diam.append(ground - y)
    D = float(np.median(diam))
    return centres[0], centres[1], D / 2, ground

def outline(mask: np.ndarray):
    h, w = mask.shape
    cols = [x for x in range(w) if mask[:, x].any()]
    top = [(x, np.min(np.nonzero(mask[:, x])[0])) for x in cols]
    bot = [(x, np.max(np.nonzero(mask[:, x])[0])) for x in cols]
    return top, bot

def resample(pts, n):
    p = np.array(pts, dtype=float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--name", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--preview", action="store_true"); ap.add_argument("--n", type=int, default=160)
    a = ap.parse_args()
    img = Image.open(a.image)
    m = mask_car(img)
    fa, ra, r, ground = wheels(m)
    T = 2 * r
    top, bot = outline(m)
    x0 = top[0][0]
    tx = lambda x: (x - x0) / T
    ty = lambda y: (ground - y) / T
    topT = [[round(tx(x), 4), round(ty(y), 4)] for x, y in resample(top, a.n)]
    botT = [[round(tx(x), 4), round(ty(y), 4)] for x, y in resample(bot, a.n)]
    out = {"key": a.key, "name": a.name, "source": Path(a.image).name, "T_px": round(T, 1),
           "L": round(tx(top[-1][0]), 4), "fa": round(tx(fa), 4), "ra": round(tx(ra), 4), "H": round(max(p[1] for p in topT), 4),
           "top": topT, "bottom": botT}
    p = Path(a.out) if a.out else Path("outlines") / f"{a.key}.json"
    p.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(p, "w"))
    print(f"{a.name}: T={T:.0f}px  L={out['L']:.2f} T  wheelbase={out['ra']-out['fa']:.2f} T  H={out['H']:.2f} T  -> {p}")
    if a.preview:
        from PIL import ImageDraw
        pv = img.convert("RGB"); dr = ImageDraw.Draw(pv)
        dr.line([tuple(map(int, q)) for q in resample(top, 400)], fill=(255, 106, 61), width=3)
        dr.line([tuple(map(int, q)) for q in resample(bot, 400)], fill=(125, 211, 252), width=2)
        for cx in (fa, ra): dr.ellipse([cx - r, ground - 2 * r, cx + r, ground], outline=(255, 122, 144), width=3)
        pv.save(p.with_suffix(".preview.png")); print("preview ->", p.with_suffix('.preview.png'))

if __name__ == "__main__":
    main()
