"""Download Nero product images, replace background with pure white, square-pad, save to out/."""
import json, os, io, sys, urllib.request
from PIL import Image
from rembg import remove, new_session

session = new_session("isnet-general-use")
src = json.load(open("sources.json"))
only = sys.argv[1:]  # optional filename filters for a sample run
os.makedirs("out", exist_ok=True)
for url, name in src:
    if only and not any(o in name for o in only):
        continue
    dst = os.path.join("out", name.rsplit(".", 1)[0] + ".jpg")
    if os.path.exists(dst):
        continue
    raw = urllib.request.urlopen(url, timeout=60).read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    cut = remove(img, session=session, alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10)
    bbox = cut.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    if bbox:
        cut = cut.crop(bbox)
    side = int(max(cut.size) * 1.15)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(cut, ((side - cut.width) // 2, (side - cut.height) // 2), cut)
    canvas = canvas.resize((1200, 1200), Image.LANCZOS) if side > 1200 else canvas
    canvas.save(dst, "JPEG", quality=92)
    print("ok", name)
