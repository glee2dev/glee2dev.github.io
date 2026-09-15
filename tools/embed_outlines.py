#!/usr/bin/env python3
"""
Embed outlines/*.json into proportion/index.html as the `const OUTLINES={...}` literal.

The tool does not fetch the JSON files at runtime; it reads this literal. Editing a file
under outlines/ changes nothing on the page until this runs.

  python tools/embed_outlines.py            # rewrite proportion/index.html in place
  python tools/embed_outlines.py --check    # exit 1 if the literal is stale

Embedded schema per key: name, L, fa, ra, H, top, lm (= landmarks). dlo, source, T_px and
flipped stay in the files only.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "proportion" / "index.html"
ORDER = ['A', 'B', 'B2', 'C', 'C2', 'D', 'D2', 'E', 'E2', 'F', 'J', 'S', 'P', 'M']   # SEG_ORDER in the tool
MARK = "const OUTLINES="

def build():
    out = {}
    for k in ORDER:
        d = json.load(open(ROOT / "outlines" / f"{k}.json"))
        out[k] = {"name": d["name"], "L": d["L"], "fa": d["fa"], "ra": d["ra"], "H": d["H"],
                  "top": d["top"], "lm": d["landmarks"]}
    return json.dumps(out, separators=(",", ":"))

def span(html):
    i = html.index(MARK) + len(MARK)
    depth = 0
    for j in range(i, len(html)):
        if html[j] == "{": depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0: return i, j + 1
    raise ValueError("unbalanced OUTLINES literal")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    html = PAGE.read_text(encoding="utf-8"); i, j = span(html)
    fresh = build(); current = html[i:j]
    if a.check:
        same = json.loads(current) == json.loads(fresh)
        print("OUTLINES literal is", "current" if same else "STALE"); sys.exit(0 if same else 1)
    PAGE.write_text(html[:i] + fresh + html[j:], encoding="utf-8")
    print(f"embedded {len(ORDER)} outlines into {PAGE.relative_to(ROOT)} ({len(fresh):,} chars, was {len(current):,})")

if __name__ == "__main__":
    main()
