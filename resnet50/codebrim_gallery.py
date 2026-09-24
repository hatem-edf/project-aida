#!/usr/bin/env python3
"""
Pick CODEBRIM example images for the slides.

On the HPC (login node is fine, this is light):
    source ~/work/.venv-bikit/bin/activate
    python codebrim_gallery.py              # 10 per class; --seed 1 for a new draw

On your laptop, in a second terminal:
    ssh -N -L 8765:localhost:8765 hpc

Then open http://localhost:8765 and click any image to download the full-res file.
"""
import argparse, csv, html, os, random
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image

HOME = Path.home()
CSV = HOME / "work/project-aida/building-inspection-toolkit/bikit/data/codebrim-classif-balanced.csv"
DATA = HOME / ".cache/bikit"                      # img_path in the CSV is relative to this
OUT = HOME / "work/project-aida/gallery"
CLASSES = ["Crack", "Spalling", "Rust", "BarsExposed", "Efflorescence", "NoDamage"]
SLIDE_NAME = {"Crack": "Crack", "Spalling": "Spallation", "Rust": "Corrosion stain",
              "BarsExposed": "Exposed bars", "Efflorescence": "Efflorescence", "NoDamage": "Background"}

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--min-side", type=int, default=250, help="skip images smaller than this (px)")
args = ap.parse_args()

# Originals only: "_copyNN" files are augmented duplicates in the balanced train split.
# They keep the original crop's labels even when the augmentation cut the defect out.
rows = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if "_copy" not in r["img_name"]:
            rows.setdefault(r["img_name"], r)
labels = {n: {c for c in CLASSES if r[c] == "1"} for n, r in rows.items()}
paths = {n: DATA / r["img_path"] for n, r in rows.items()}
print(f"{len(rows)} unique images in the CSV")

(OUT / "thumbs").mkdir(parents=True, exist_ok=True)
(OUT / "full").mkdir(exist_ok=True)
rng = random.Random(args.seed)

def tile(name):
    thumb = OUT / "thumbs" / (Path(name).stem + ".jpg")
    if not thumb.exists():
        im = Image.open(paths[name]).convert("RGB")
        im.thumbnail((600, 600))
        im.save(thumb, quality=88)
    full = OUT / "full" / name
    if not full.exists():
        full.symlink_to(paths[name])
    return (f'<a href="full/{name}" download><img loading=lazy src="thumbs/{thumb.name}"></a>'
            f'<p>{name}<br><span>{", ".join(sorted(labels[name]))}</span></p>')

def draw(pool):
    """Random images from pool, skipping ones too small for a slide (reads headers only)."""
    pool = pool[:]
    rng.shuffle(pool)
    picks = []
    for n in pool:
        if paths[n].exists() and min(Image.open(paths[n]).size) >= args.min_side:
            picks.append(n)
            if len(picks) == args.n:
                break
    return picks

sections = []
for cls in CLASSES:
    pure = [n for n, l in labels.items() if l == {cls}]
    sections.append((f"{SLIDE_NAME[cls]} <small>{len(pure)} single-label originals</small>", draw(pure)))

multi = [n for n, l in labels.items() if len(l) >= 3]
sections.append((f"Several defects at once, for slide 7 <small>{len(multi)} originals</small>", draw(multi)))

body = "".join(f"<h2>{title}</h2><div class=g>" + "".join(f"<div>{tile(n)}</div>" for n in picks) + "</div>"
               for title, picks in sections)
(OUT / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>CODEBRIM picker</title>
<style>body{{font:15px system-ui;margin:24px;background:#f6f4f0;color:#16232e}}
h2{{margin:32px 0 10px}}small{{font-weight:400;color:#6a7179;font-size:14px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}
.g div{{background:#fff;border-radius:8px;overflow:hidden}}
img{{width:100%;aspect-ratio:2.8;object-fit:cover;display:block}}
body.full img{{aspect-ratio:auto}}
p{{margin:6px 8px;font-size:12px;color:#55606b}}span{{color:#b0491f}}</style>
<p>Seed {args.seed}. Thumbnails are cropped to the slide's frame
(<label><input type=checkbox onchange="document.body.classList.toggle('full')"> show uncropped</label>).
Click an image to download the full-resolution file.</p>{body}""")

handler = partial(SimpleHTTPRequestHandler, directory=str(OUT))
handler.log_message = lambda *a: None
print(f"""
Ready. On your LAPTOP run:   ssh -N -L {args.port}:localhost:{args.port} hpc
Then open:                  http://localhost:{args.port}
Ctrl+C here when done.""")
try:
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
except KeyboardInterrupt:
    pass
