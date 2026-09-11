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
    est_T = np.ptp(np.nonzero(fg.any(axis=0))[0]) / (L / tire); k = max(5, int(est_T * 0.20))
    body = ndimage.binary_opening(fg, structure=np.ones((1, k))); body = ndimage.binary_opening(body, structure=np.ones((3, 1)))   # thin spoilers survive
    # extent from the body above the shadow band (the lowest 8% of the car's height), so a cast shadow can't widen it
    ys_all = np.nonzero(body.any(axis=1))[0]; cut = int(ys_all.max() - 0.08 * (ys_all.max() - ys_all.min()))
    xs = np.nonzero(body[:cut].any(axis=0))[0]; x0, x1 = xs.min(), xs.max()
    Tpx = (x1 - x0) / (L / tire)
    # second pass: a car column has body at bumper height (a quarter tyre above the lowest body row); shadow columns don't
    cut2 = int(ys_all.max() - 0.25 * Tpx); xs = np.nonzero(body[:cut2].any(axis=0))[0]; x0, x1 = xs.min(), xs.max()
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
    cols = [x for x in range(x0, x1 + 1) if body[:, x].any()]
    top = np.array([(x, np.min(np.nonzero(body[:, x])[0])) for x in cols], dtype=float)
    # the real edge, cleaned once: a small Gaussian along the line removes compression noise without moving the shape
    top[:, 1] = ndimage.gaussian_filter1d(top[:, 1], sigma=max(1.0, Tpx * 0.02))
    T = lambda v: v / Tpx
    X = (top[:, 0] - x0) / Tpx; Y = (ground - top[:, 1]) / Tpx
    Lt = X[-1]; slope = np.gradient(ndimage.uniform_filter1d(Y, size=max(5, int(Tpx * 0.10))), X)
    peak = int(np.argmax(Y)); yPeak = float(Y[peak])
    def first(cond, lo, hi):
        idx = np.nonzero(cond & (X >= lo) & (X <= hi))[0]; return int(idx[0]) if idx.size else None
    steep = slope > np.tan(np.radians(22))
    ia = first(steep, 0.9, X[peak]); ia = ia if ia is not None else int(np.argmax(slope[:peak])); cowl = ia
    ir = ia
    while ir < peak and slope[ir] > np.tan(np.radians(10)): ir += 1
    roofStart = ir
    fall = slope < -np.tan(np.radians(9))
    ie = first(fall, X[peak] + 0.15, Lt - 0.15); ie = ie if ie is not None else min(len(X) - 2, peak + 5); roofEnd = ie
    idk = ie
    while idk < len(X) - 1 and slope[idk] < -np.tan(np.radians(8)): idk += 1
    deck = idk if X[idk] < Lt - 0.25 else None
    K = lambda i: [round(float(X[i]), 3), round(float(Y[i]), 3)]
    landmarks = {"cowl": K(cowl), "roofStart": K(roofStart), "peak": [round(float(X[peak]), 3), round(yPeak, 3)], "roofEnd": K(roofEnd), "deck": (K(deck) if deck is not None else None),
                 "rearForm": 1.0 if deck is None else (0.0 if (Y[deck] < yPeak - 0.35) else 1.0)}
    # DLO: the glass is the dark region above the belt; pillars painted black join it, which is the one-band DLO we want
    beltY = ground - 0.95 * Tpx; hoodY = ground - 1.35 * Tpx
    bodyBand = fg & (np.arange(h)[:, None] > ground - 0.9 * Tpx) & (np.arange(h)[:, None] < ground - 0.4 * Tpx)
    bodyLum = float(np.median(lum[bodyBand])) if bodyBand.any() else 128.0
    thr = min(105.0, 0.62 * bodyLum)
    glass = fg & (lum < thr) & (np.arange(h)[:, None] < beltY)
    glass = ndimage.binary_opening(glass, structure=np.ones((3, max(3, int(Tpx * 0.04)))))
    lab2, n2 = ndimage.label(glass); dlo = None
    if n2:
        sizes2 = ndimage.sum(glass, lab2, range(1, n2 + 1)); keep = [i + 1 for i, sz in enumerate(sizes2) if sz > (Tpx * 0.35) ** 2]
        if keep:
            gm = np.isin(lab2, keep); gm = ndimage.binary_closing(gm, structure=np.ones((3, int(Tpx * 0.25))))
            gc = [x for x in range(w) if gm[:, x].any()]
            if len(gc) > Tpx * 1.2:
                gt = np.array([(x, np.min(np.nonzero(gm[:, x])[0])) for x in gc], dtype=float); gb = np.array([(x, np.max(np.nonzero(gm[:, x])[0])) for x in gc], dtype=float)
                sg = max(1.0, Tpx * 0.02); gt[:, 1] = ndimage.gaussian_filter1d(gt[:, 1], sg); gb[:, 1] = ndimage.gaussian_filter1d(gb[:, 1], sg)
                poly = np.vstack([gt, gb[::-1]])
                dd = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]; tt = np.linspace(0, dd[-1], 140)
                pr = np.c_[np.interp(tt, dd, poly[:, 0]), np.interp(tt, dd, poly[:, 1])]
                dlo = [[round(float((x - x0) / Tpx), 4), round(float((ground - y) / Tpx), 4)] for x, y in pr]
    # the outline itself: the cleaned edge, resampled by arc length
    d = np.r_[0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]; t = np.linspace(0, d[-1], n)
    rs = np.c_[np.interp(t, d, X), np.interp(t, d, Y)]
    T = lambda v: v / Tpx
    out = {"T_px": round(Tpx, 1), "L": round(T(x1 - x0), 4), "fa": round(T(fa - x0), 4), "ra": round(T(ra - x0), 4),
           "H": round(float(yPeak), 4), "top": [[round(float(x), 4), round(float(y), 4)] for x, y in rs], "landmarks": landmarks, "dlo": dlo}
    return out, {"top": top, "fa": fa, "ra": ra, "ground": ground, "Tpx": Tpx, "x0": x0}

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
    lm = out["landmarks"]; print(f"{a.key:<3} {a.name:<18} L={out['L']:.2f} H={out['H']:.2f} dlo={'traced' if out['dlo'] else 'parametric'}")
    if a.preview:
        pv = img.copy(); dr = ImageDraw.Draw(pv); r = dbg["Tpx"] / 2
        dr.line([tuple(map(float, q)) for q in dbg["top"]], fill=(255, 106, 61), width=2)
        if out.get("dlo"): dr.polygon([(dbg["x0"] + q[0] * dbg["Tpx"], dbg["ground"] - q[1] * dbg["Tpx"]) for q in out["dlo"]], outline=(95, 224, 200), width=2)
        for cx in (dbg["fa"], dbg["ra"]): dr.ellipse([cx - r, dbg["ground"] - 2 * r, cx + r, dbg["ground"]], outline=(255, 122, 144), width=3)
        dr.line([(0, dbg["ground"]), (pv.width, dbg["ground"])], fill=(125, 211, 252), width=1)
        pv.save(p.with_suffix(".preview.png"))

if __name__ == "__main__":
    main()
