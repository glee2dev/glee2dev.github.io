#!/usr/bin/env python3
"""
Trace a car side-profile photo into a T-normalised top outline, using the car's known package
(length, wheelbase, front overhang, tyre OD in mm) for scale and axle placement.
  python tools/trace_outline.py img.png --key D --name "BMW 3 Series" --L 4713 --WB 2851 --FO 831 --tire 660 [--flip] [--preview]
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

def largest(m):
    lab, n = ndimage.label(m)
    if n == 0: return m
    sizes = ndimage.sum(m, lab, range(1, n + 1)); return lab == (1 + int(np.argmax(sizes)))

def trace(img, L, WB, FO, tire, n=160):
    a = np.asarray(img.convert("RGB")).astype(np.int32); h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]); bg = np.median(border, axis=0)
    lum = a.mean(axis=2)
    fg = largest(ndimage.binary_fill_holes(ndimage.binary_closing(np.abs(a - bg).sum(axis=2) > 45, iterations=2)))
    # remove antenna / mirror / shadow fringes with an opening sized to the car
    est_T = np.ptp(np.nonzero(fg.any(axis=0))[0]) / (L / tire); k = max(5, int(est_T * 0.16))
    body = ndimage.binary_opening(fg, structure=np.ones((1, k))); body = ndimage.binary_opening(body, structure=np.ones((k // 2 + 1, 1)))
    xs = np.nonzero(body.any(axis=0))[0]; x0, x1 = xs.min(), xs.max()
    Tpx = (x1 - x0) / (L / tire)                                   # scale from the known length
    fa = x0 + FO / tire * Tpx; ra = fa + WB / tire * Tpx
    # ground: lowest dark pixel in the columns around each axle (tyre bottoms; shadows are grey, tyres are black)
    def tyre_bottom(cx):
        best = 0
        for x in range(int(cx - Tpx * 0.15), int(cx + Tpx * 0.15)):
            col = np.nonzero(fg[:, x] & (lum[:, x] < 80))[0]
            if col.size: best = max(best, col.max())
        return best
    ground = float(np.mean([tyre_bottom(fa), tyre_bottom(ra)]))
    cols = [x for x in range(w) if body[:, x].any()]
    top = np.array([(x, np.min(np.nonzero(body[:, x])[0])) for x in cols], dtype=float)
    top[:, 1] = ndimage.maximum_filter1d(top[:, 1], size=max(5, int(Tpx * 0.14)))   # drop upward spikes (antenna, fin)
    top[:, 1] = ndimage.uniform_filter1d(top[:, 1], size=max(3, int(Tpx * 0.04)))
    # reduce the scan to its real control points, then draw one smooth curve through them
    def rdp(pts, eps):
        if len(pts) < 3: return pts
        a, b = pts[0], pts[-1]; ab = b - a; nrm = np.hypot(*ab) or 1
        dist = np.abs(np.cross(ab, pts - a)) / nrm; i = int(np.argmax(dist))
        if dist[i] > eps: return np.vstack([rdp(pts[:i + 1], eps)[:-1], rdp(pts[i:], eps)])
        return np.vstack([a, b])
    ctrl = rdp(top, Tpx * 0.018)
    # drop tiny features: merge vertices closer than 0.12 T
    keep = [ctrl[0]]
    for q in ctrl[1:]:
        if np.hypot(*(q - keep[-1])) >= Tpx * 0.12: keep.append(q)
    if np.any(keep[-1] != ctrl[-1]): keep.append(ctrl[-1])
    ctrl = np.array(keep)
    def catmull(P, per=14):
        out = []
        Pp = np.vstack([P[0], P, P[-1]])
        for i in range(1, len(Pp) - 2):
            p0, p1, p2, p3 = Pp[i - 1], Pp[i], Pp[i + 1], Pp[i + 2]
            for k in range(per):
                u = k / per; u2, u3 = u * u, u * u * u
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
        out.append(P[-1]); return np.array(out)
    smooth = catmull(ctrl)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(smooth, axis=0).T))]; t = np.linspace(0, d[-1], n)
    rs = np.c_[np.interp(t, d, smooth[:, 0]), np.interp(t, d, smooth[:, 1])]
    T = lambda v: v / Tpx
    out = {"T_px": round(Tpx, 1), "L": round(T(x1 - x0), 4), "fa": round(T(fa - x0), 4), "ra": round(T(ra - x0), 4),
           "H": round(T(ground - top[:, 1].min()), 4), "top": [[round(T(x - x0), 4), round(T(ground - y), 4)] for x, y in rs], "ctrl": [[round(T(x - x0), 4), round(T(ground - y), 4)] for x, y in ctrl]}
    return out, {"top": top, "fa": fa, "ra": ra, "ground": ground, "Tpx": Tpx}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--key", required=True); ap.add_argument("--name", required=True)
    for k in ("L", "WB", "FO", "tire"): ap.add_argument("--" + k, type=float, required=True)
    ap.add_argument("--flip", action="store_true"); ap.add_argument("--out", default=None); ap.add_argument("--preview", action="store_true"); ap.add_argument("--n", type=int, default=160)
    a = ap.parse_args()
    im0 = Image.open(a.image)
    if im0.mode in ("RGBA", "LA", "P"):
        im0 = im0.convert("RGBA"); bgw = Image.new("RGBA", im0.size, (255, 255, 255, 255)); bgw.alpha_composite(im0); im0 = bgw
    img = im0.convert("RGB")
    if a.flip: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    out, dbg = trace(img, a.L, a.WB, a.FO, a.tire, a.n); out.update({"key": a.key, "name": a.name, "source": Path(a.image).name, "flipped": a.flip})
    p = Path(a.out) if a.out else Path("outlines") / f"{a.key}.json"; p.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(p, "w"))
    print(f"{a.key:<3} {a.name:<18} T={out['T_px']:>5.0f}px  L={out['L']:.2f}  H={out['H']:.2f}  ctrl={len(out['ctrl'])}")
    if a.preview:
        pv = img.copy(); dr = ImageDraw.Draw(pv); r = dbg["Tpx"] / 2
        dr.line([tuple(map(float, q)) for q in dbg["top"]], fill=(255, 106, 61), width=3)
        for cx in (dbg["fa"], dbg["ra"]): dr.ellipse([cx - r, dbg["ground"] - 2 * r, cx + r, dbg["ground"]], outline=(255, 122, 144), width=3)
        dr.line([(0, dbg["ground"]), (pv.width, dbg["ground"])], fill=(125, 211, 252), width=1)
        pv.save(p.with_suffix(".preview.png"))

if __name__ == "__main__":
    main()
