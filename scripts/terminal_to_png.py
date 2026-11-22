#!/usr/bin/env python3
"""Turn saved terminal output (with or without colour codes) into a PNG.

    python scripts/terminal_to_png.py results/demo_seed0.txt
    python scripts/terminal_to_png.py run.txt --out docs/run.png --light

For slides and the thesis appendix: the image looks like the terminal, colours
included, without a screenshot tool. Colour codes are kept when the run was
captured with colour forced on (see --record below); a --save file has no
colour and renders in plain ink, which is still clean.

    python scripts/terminal_to_png.py --record results/demo_seed0_color.txt -- \\
        scripts/run_experiment.py --dataset locomo --path data/locomo10.json --runs 1

runs the command with colour forced on and saves its output, codes included,
then renders it.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from make_doc_png import find_browser, SCALE  # noqa: E402

# xterm-256 values for the indices ocmg/console.py uses, plus the 16 basics.
XTERM = {25: "#005faf", 111: "#87afff", 245: "#8a8a8a", 240: "#585858",
         34: "#00af00", 214: "#ffaf00", 167: "#d75f5f"}
BASIC = {30: "#000000", 31: "#cd3131", 32: "#0dbc79", 33: "#e5e510",
         34: "#2472c8", 35: "#bc3fbc", 36: "#11a8cd", 37: "#e5e5e5",
         90: "#666666", 91: "#f14c4c", 92: "#23d18b", 93: "#f5f543",
         94: "#3b8eea", 95: "#d670d6", 96: "#29b8db", 97: "#ffffff"}
THEMES = {
    # VS Code's own terminal colours. On the dark theme the deep blue used for
    # emphasis is lifted one step, as VS Code itself does for low contrast.
    "dark": {"bg": "#1e1e1e", "fg": "#cccccc", 25: "#3b8eea", 240: "#6e6e6e"},
    "light": {"bg": "#ffffff", "fg": "#1f1f1f", 111: "#3a6fc4", 214: "#b07800",
              245: "#6e6e6e", 240: "#a0a0a0"},
}
SGR = re.compile(r"\x1b\[([0-9;]*)m")


# Symbols a terminal draws one cell wide but a browser's fallback font draws
# wider; each is boxed to exactly one character so table borders line up.
_NARROW = "✔✘⚠▲▼●√"


def _cells(escaped: str) -> str:
    for ch in _NARROW:
        escaped = escaped.replace(
            ch, f'<span style="display:inline-block;width:1ch;'
                f'overflow:hidden;text-align:center">{ch}</span>')
    return escaped


def ansi_to_html(text: str, theme: dict) -> str:
    out, open_span = [], False
    pos = 0
    for m in SGR.finditer(text):
        out.append(_cells(html.escape(text[pos:m.start()])))
        pos = m.end()
        codes = [int(c) for c in m.group(1).split(";") if c] or [0]
        if open_span:
            out.append("</span>")
            open_span = False
        if codes == [0]:
            continue
        style, i = [], 0
        while i < len(codes):
            c = codes[i]
            if c == 1:
                style.append("font-weight:700")
            elif c == 38 and i + 2 < len(codes) and codes[i + 1] == 5:
                n = codes[i + 2]
                style.append(f"color:{theme.get(n, XTERM.get(n, '#cccccc'))}")
                i += 2
            elif c in BASIC:
                style.append(f"color:{BASIC[c]}")
            i += 1
        if style:
            out.append(f'<span style="{";".join(style)}">')
            open_span = True
    out.append(_cells(html.escape(text[pos:])))
    if open_span:
        out.append("</span>")
    return "".join(out)


def render(txt_path: str, png_path: str, light: bool) -> None:
    theme = THEMES["light" if light else "dark"]
    with open(txt_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read().rstrip() + "\n"
    cols = max((len(SGR.sub("", line)) for line in text.splitlines()), default=80)
    width_px = int(cols * 8.0) + 60
    page = (f'<!doctype html><html><head><meta charset="utf-8"><style>'
            f'body{{margin:0;background:{theme["bg"]}}}'
            f'pre{{margin:0;padding:22px 28px;color:{theme["fg"]};'
            f'font:13.3px/1.22 "Cascadia Mono",Consolas,monospace;'
            f'white-space:pre}}</style></head><body><pre>'
            f'{ansi_to_html(text, theme)}</pre></body></html>')
    browser = find_browser()
    if not browser:
        raise SystemExit("needs Microsoft Edge or Google Chrome")
    lines = text.count("\n") + 4
    height = min(16000, int(lines * 17) + 80)
    with tempfile.TemporaryDirectory() as tmp:
        page_path = os.path.join(tmp, "t.html")
        raw = os.path.join(tmp, "t.png")
        with open(page_path, "w", encoding="utf-8") as fh:
            fh.write(page)
        subprocess.run([
            browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--no-first-run", f"--user-data-dir={os.path.join(tmp, 'p')}",
            f"--force-device-scale-factor={SCALE}",
            f"--window-size={width_px},{height}",
            f"--screenshot={raw}",
            "file:///" + page_path.replace("\\", "/"),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120)
        img = trim_bg(raw, theme["bg"])
        img.save(png_path, optimize=True)


def trim_bg(path: str, bg_hex: str):
    from PIL import Image, ImageChops  # noqa: PLC0415
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, tuple(int(bg_hex[i:i + 2], 16) for i in (1, 3, 5)))
    box = ImageChops.difference(img, bg).getbbox()
    if box:
        pad = int(22 * SCALE)
        img = img.crop((0, 0, min(img.width, box[2] + pad),
                        min(img.height, box[3] + pad)))
    return img


def record(out_txt: str, command: list) -> None:
    """Run a project script with colour forced on, saving its raw output."""
    script, args = command[0], command[1:]
    code = (
        "import sys, runpy; sys.path.insert(0, '.'); "
        "from ocmg import console as C; C.set_color(True); "
        f"sys.argv = {[script] + args!r}; "
        f"runpy.run_path({script!r}, run_name='__main__')"
    )
    env = dict(os.environ, OCMG_UTF8="1", OCMG_COLOR="1", PYTHONIOENCODING="utf-8",
               COLUMNS=str(shutil.get_terminal_size((120, 24)).columns))
    with open(out_txt, "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, stdout=fh,
                       stderr=subprocess.STDOUT, env=env, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="terminal output -> PNG")
    ap.add_argument("text", help="saved output file (with --record: where to save it)")
    ap.add_argument("--out", default=None, help="PNG path (default: same name .png)")
    ap.add_argument("--light", action="store_true", help="white background")
    ap.add_argument("--record", action="store_true",
                    help="first run the command after -- with colour on")
    # Everything after "--" is the command to record, so its own flags are
    # never mistaken for this script's.
    argv = sys.argv[1:]
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    if args.record:
        if not cmd:
            ap.error("--record needs a command after --")
        record(args.text, cmd)
    out = args.out or os.path.splitext(args.text)[0] + ".png"
    render(args.text, out, args.light)
    print(f"image    {out}   open it, or drop it into a slide")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
