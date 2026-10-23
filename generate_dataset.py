"""Generate a standalone synthetic Persian glyph dataset using local system fonts.
For better font diversity, put 3+ Arabic/Persian-capable .ttf/.otf files in fonts/.
"""
from pathlib import Path
import random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from labels import LETTERS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dataset"
FONTS = ROOT / "fonts"
SAMPLES_PER_FONT = 128
CANVAS = 160
SEED = 501
random.seed(SEED)
np.random.seed(SEED)

def find_fonts():
    candidates = list(FONTS.glob("*.ttf")) + list(FONTS.glob("*.otf")) + list(FONTS.glob("*.ttc"))
    # Common Windows fonts; choose those that exist. Arabic glyph coverage varies by machine.
    win = Path("C:/Windows/Fonts")
    for name in ["tahoma.ttf", "arial.ttf", "segoeui.ttf", "times.ttf", "trado.ttf"]:
        p = win / name
        if p.exists(): candidates.append(p)
    unique = []
    for p in candidates:
        if p.exists() and p not in unique: unique.append(p)
    return unique[:8]

def main():
    fonts = find_fonts()
    if len(fonts) < 3:
        raise SystemExit(
            "Need at least 3 Arabic/Persian-capable fonts. Add .ttf/.otf files to the included fonts folder, "
            "or install fonts such as Noto Naskh Arabic, Noto Sans Arabic, and Amiri."
        )
    OUT.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_dir():
            for f in p.iterdir():
                if f.is_file(): f.unlink()
            p.rmdir()
    (ROOT / "dataset_labels.txt").write_text(
        "".join(f"{i}|{ch}|{name}\n" for i,(ch,name) in enumerate(LETTERS)), encoding="utf-8"
    )
    for cid, (char, name) in enumerate(LETTERS):
        folder = OUT / f"{cid:02d}_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        for fid, font_path in enumerate(fonts):
            for n in range(SAMPLES_PER_FONT):
                canvas = Image.new("L", (CANVAS, CANVAS), 255)
                draw = ImageDraw.Draw(canvas)
                size = random.randint(48, 82)
                try:
                    font = ImageFont.truetype(str(font_path), size=size, layout_engine=ImageFont.Layout.RAQM)
                except Exception:
                    font = ImageFont.truetype(str(font_path), size=size)
                bbox = draw.textbbox((0, 0), char, font=font)
                w, h = max(1,bbox[2]-bbox[0]), max(1,bbox[3]-bbox[1])
                x = (CANVAS-w)//2 - bbox[0] + random.randint(-5,5)
                y = (CANVAS-h)//2 - bbox[1] + random.randint(-5,5)
                draw.text((x, y), char, fill=random.randint(0, 55), font=font)
                angle = random.uniform(-14,14)
                canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
                if random.random() < .25: canvas = canvas.filter(ImageFilter.GaussianBlur(random.uniform(.2,.7)))
                if random.random() < .5:
                    canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(.75,1.25))
                arr = np.asarray(canvas).astype(np.float32)
                if random.random() < .3:
                    arr += np.random.normal(0, random.uniform(1,5), arr.shape)
                arr = np.clip(arr,0,255).astype("uint8")
                Image.fromarray(arr).resize((160,160), Image.Resampling.LANCZOS).save(
                    folder / f"font{fid}_{n:03d}.png"
                )
    print(f"Dataset generated: {len(LETTERS)} classes x {len(fonts)} fonts x {SAMPLES_PER_FONT} samples.")

if __name__ == "__main__":
    main()
