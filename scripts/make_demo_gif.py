"""Render docs/assets/demo.gif from a real run of examples/guard_demo.py.

    python scripts/make_demo_gif.py [--font PATH] [--out docs/assets/demo.gif]

The GIF shows the demo's source excerpt, then its actual output, typed out like a terminal session.
Nothing is staged: the output lines come from running the demo, so regenerate this after behaviour changes.
Needs Pillow and a monospace TrueType font (Consolas on Windows, DejaVu Sans Mono on Linux).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "guard_demo.py"
FONTS = ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/System/Library/Fonts/Menlo.ttc"]

BG, BAR, FG, DIM = (22, 24, 29), (40, 43, 51), (220, 223, 228), (128, 134, 145)
COLORS = {"APPROVE": (80, 200, 120), "RESUMED": (80, 200, 120), "BLOCK": (240, 90, 90), "REVIEW": (240, 190, 70)}
KEYWORD = (120, 170, 250)
W, PAD, SIZE, LINE = 1000, 22, 15, 22


def demo_output() -> list[str]:
    out = subprocess.run([sys.executable, str(DEMO), "--fast"], capture_output=True, text=True, check=True,
                         cwd=ROOT)
    return out.stdout.rstrip("\n").split("\n")


def source_excerpt() -> list[str]:
    lines = DEMO.read_text(encoding="utf-8").split("\n")
    start = next(i for i, x in enumerate(lines) if x.startswith("guard = Guard("))
    end = next(i for i, x in enumerate(lines) if "dropped" in x)
    return ["# examples/guard_demo.py"] + lines[start:end + 1]


class Term:
    def __init__(self, font: ImageFont.FreeTypeFont, rows: int, title: str):
        self.font, self.rows, self.title = font, rows, title
        self.h = 40 + PAD + rows * LINE + PAD

    def frame(self, lines: list[str], cursor: bool = False) -> Image.Image:
        im = Image.new("RGB", (W, self.h), BG)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 34], fill=BAR)
        for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse([16 + 22 * i, 11, 28 + 22 * i, 23], fill=c)
        d.text((W // 2, 17), self.title, font=self.font, fill=DIM, anchor="mm")
        y = 40 + PAD
        for n, line in enumerate(lines[-self.rows:]):
            self._line(d, PAD, y, line)
            if cursor and n == len(lines[-self.rows:]) - 1:
                x = PAD + d.textlength(line, font=self.font)
                d.rectangle([x + 2, y + 2, x + 10, y + LINE - 4], fill=FG)
            y += LINE
        return im

    def _line(self, d: ImageDraw.ImageDraw, x: int, y: int, line: str) -> None:
        stripped = line.strip()
        word = stripped.split(" ")[0] if stripped else ""
        if line.startswith("#"):
            d.text((x, y), line, font=self.font, fill=DIM)
        elif word in COLORS:
            lead = line[: len(line) - len(stripped)] + word
            d.text((x, y), lead, font=self.font, fill=COLORS[word])
            d.text((x + d.textlength(lead, font=self.font), y), line[len(lead):], font=self.font, fill=FG)
        elif line.startswith("$"):
            d.text((x, y), "$", font=self.font, fill=KEYWORD)
            d.text((x + d.textlength("$", font=self.font), y), line[1:], font=self.font, fill=FG)
        elif line.lstrip().startswith(("@guard", "def ", "guard =")):
            d.text((x, y), line, font=self.font, fill=KEYWORD)
        else:
            d.text((x, y), line, font=self.font, fill=FG if not line.startswith("audit") else DIM)


def build(font_path: str, out: Path) -> None:
    font = ImageFont.truetype(font_path, SIZE)
    code, run = source_excerpt(), demo_output()
    rows = max(len(code), len(run) + 1)
    frames: list[Image.Image] = []
    durations: list[int] = []

    def add(im: Image.Image, ms: int) -> None:
        frames.append(im)
        durations.append(ms)

    term = Term(font, rows, "action-sentinel: @guard.tool")
    shown: list[str] = []
    for line in code:  # the source appears line by line, then holds
        shown.append(line)
        add(term.frame(shown), 70)
    durations[-1] = 4200

    term.title = "python examples/guard_demo.py"
    shown = []
    for line in run:
        if line.startswith("$"):  # commands are typed
            for i in range(2, len(line) + 1, 3):
                add(term.frame(shown + [line[:i]], cursor=True), 35)
            shown.append(line)
            add(term.frame(shown), 450)
        else:
            shown.append(line)
            add(term.frame(shown), 900 if line.strip() else 150)
    durations[-1] = 5000

    out.parent.mkdir(parents=True, exist_ok=True)
    # One palette for every frame, taken from the frames with the most colour, so the labels stay true.
    sample = Image.new("RGB", (W, frames[0].height * 2))
    sample.paste(frames[len(code) - 1], (0, 0))
    sample.paste(frames[-1], (0, frames[0].height))
    palette = sample.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print("wrote %s (%d frames, %.0f KB)" % (out, len(frames), out.stat().st_size / 1024))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--font", default=next((f for f in FONTS if Path(f).is_file()), None))
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets" / "demo.gif"))
    args = ap.parse_args()
    if not args.font:
        print("no monospace font found; pass --font", file=sys.stderr)
        return 1
    build(args.font, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
