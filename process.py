"""Nero product images -> pure white background, square, centered.

Two methods:
- "whiten": for studio shots on an even light background. Divides the image by the
  background brightness so the backdrop becomes exactly 255 while the jewelry keeps
  its full contrast (no cut-out, so thin or out-of-focus chains don't fade).
- "cutout": AI background removal (rembg) for photos whose background isn't even.
"""
import json, os, io, sys, urllib.request
import numpy as np
from PIL import Image

OUT = os.environ.get("OUT_DIR", "out")
src = json.load(open("sources.json"))
only = sys.argv[1:]
os.makedirs(OUT, exist_ok=True)
_session = None


def border_stats(a):
    b = np.concatenate([a[:20].reshape(-1, 3), a[-20:].reshape(-1, 3),
                        a[:, :20].reshape(-1, 3), a[:, -20:].reshape(-1, 3)])
    return b.mean(axis=0), b.std(axis=0).mean()


def whiten(img):
    a = np.asarray(img).astype(np.float32)
    bg, _ = border_stats(a)
    out = np.clip(a * (255.0 / np.maximum(bg, 1.0)), 0, 255)
    out[out.min(axis=2) >= 247] = 255  # flatten residual backdrop noise to pure white
    out = Image.fromarray(out.astype(np.uint8))
    mask = Image.fromarray(((np.asarray(out).min(axis=2) < 245) * 255).astype(np.uint8))
    return out, mask.getbbox()


def cutout(img):
    global _session
    from rembg import remove, new_session
    _session = _session or new_session("isnet-general-use")
    cut = remove(img, session=_session)
    bbox = cut.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    white = Image.new("RGB", cut.size, (255, 255, 255))
    white.paste(cut, (0, 0), cut)
    return white, bbox


for url, name in src:
    if only and not any(o in name for o in only):
        continue
    dst = os.path.join(OUT, name.rsplit(".", 1)[0] + ".jpg")
    if os.path.exists(dst):
        continue
    img = Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=60).read())).convert("RGB")
    bg, noise = border_stats(np.asarray(img).astype(np.float32))
    method = "whiten" if bg.min() > 200 and noise < 6 else "cutout"
    im, bbox = (whiten if method == "whiten" else cutout)(img)
    if bbox:
        im = im.crop(bbox)
    side = int(max(im.size) * 1.15)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    if side != 1200:
        canvas = canvas.resize((1200, 1200), Image.LANCZOS)
    canvas.save(dst, "JPEG", quality=92)
    print(method, f"bg={bg.round()} noise={noise:.1f}", name)
