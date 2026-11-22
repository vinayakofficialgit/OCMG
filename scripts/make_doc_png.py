#!/usr/bin/env python3
"""Render a Markdown document to PNG images, as it looks in a preview.

    python scripts/make_doc_png.py                       # docs/PIPELINE_FLOW.md
    python scripts/make_doc_png.py --doc docs/OTHER.md

Writes docs/PIPELINE_FLOW.png, the whole document as one tall image.
With --pages it also writes docs/pipeline_flow_pages/NN_*.png, one image per
"## " section, for slides.

How: markdown-it turns the Markdown into HTML with a plain document style, and
headless Microsoft Edge (or Chrome) screenshots each section; PIL trims the
blank space and stitches the sections together. Everything runs locally; no
page is published anywhere.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

from markdown_it import MarkdownIt
from PIL import Image, ImageChops

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
]
WIDTH = 1040          # CSS pixels of the rendered page
SCALE = 1.4           # device scale, so text stays sharp when zoomed
MAX_H = 15000         # tallest section the browser is asked to render

CSS = """
body { margin: 0; background: #ffffff; }
main { width: 960px; margin: 0 auto; padding: 28px 40px 36px;
       font: 15px/1.55 "Segoe UI", system-ui, sans-serif; color: #1f2328; }
h1 { font-size: 30px; border-bottom: 1px solid #d0d7de; padding-bottom: 8px; }
h2 { font-size: 23px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px;
     margin-top: 8px; color: #1f3a5f; }
h3 { font-size: 18px; margin-top: 26px; color: #1f3a5f; }
p, li { margin: 6px 0; }
code { font: 13px Consolas, "Cascadia Mono", monospace; background: #eff2f6;
       padding: 1px 5px; border-radius: 4px; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
      padding: 12px 14px; overflow: hidden; }
pre code { background: none; padding: 0; font-size: 12px; line-height: 1.4;
           white-space: pre; }
table { border-collapse: collapse; margin: 10px 0 14px; font-size: 13.5px;
        width: 100%; }
th, td { border: 1px solid #d0d7de; padding: 5px 9px; vertical-align: top;
         text-align: left; }
th { background: #f0f3f6; }
tr:nth-child(even) td { background: #fafbfc; }
img { max-width: 100%; display: block; margin: 12px auto; }
blockquote { border-left: 4px solid #d0d7de; margin: 8px 0; padding: 2px 14px;
             color: #57606a; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 22px 0; }
.foot { color: #8c959f; font-size: 12px; margin-top: 18px; }
"""


def find_browser() -> str | None:
    for b in BROWSERS:
        if os.path.exists(b):
            return b
    return shutil.which("msedge") or shutil.which("chrome")


def split_sections(md: str) -> list[tuple[str, str]]:
    """[(title, markdown)] split at each '## ' heading outside code fences."""
    parts, cur, title, fence = [], [], "intro", False
    for line in md.split("\n"):
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            if any(x.strip() for x in cur):
                parts.append((title, "\n".join(cur)))
            cur, title = [line], line[3:].strip()
        else:
            cur.append(line)
    if any(x.strip() for x in cur):
        parts.append((title, "\n".join(cur)))
    return parts


def slug(text: str) -> str:
    text = re.sub(r"^\d+\.\s*", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "section"


def render_html(md_text: str, base_dir: str, footer: str) -> str:
    body = MarkdownIt("commonmark").enable("table").render(md_text)
    base = "file:///" + base_dir.replace("\\", "/").rstrip("/") + "/"
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<base href="{base}"><style>{CSS}</style></head><body><main>'
            f'{body}<div class="foot">{footer}</div></main></body></html>')


def screenshot(browser: str, html_path: str, png_path: str, profile: str) -> None:
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    subprocess.run([
        browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--no-first-run", "--disable-extensions",
        f"--user-data-dir={profile}",
        f"--force-device-scale-factor={SCALE}",
        f"--window-size={WIDTH},{MAX_H}",
        "--virtual-time-budget=3000",       # let the images load first
        f"--screenshot={os.path.abspath(png_path)}", url,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=120)


def trim(path: str) -> Image.Image:
    """Cut the blank space under the content."""
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img, bg).getbbox()
    if box:
        pad = int(24 * SCALE)
        img = img.crop((0, 0, img.width, min(img.height, box[3] + pad)))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="render a Markdown doc to PNG")
    ap.add_argument("--doc", default=os.path.join("docs", "PIPELINE_FLOW.md"))
    ap.add_argument("--out", default=None,
                    help="tall PNG path (default: next to the doc, same name)")
    ap.add_argument("--pages", action="store_true",
                    help="also save one PNG per section")
    args = ap.parse_args()

    doc = os.path.abspath(args.doc)
    if not os.path.exists(doc):
        print(f"not found: {args.doc}")
        return 1
    browser = find_browser()
    if not browser:
        print("needs Microsoft Edge or Google Chrome for the headless render")
        return 1

    base_dir = os.path.dirname(doc)
    name = os.path.splitext(os.path.basename(doc))[0]
    out_png = os.path.abspath(args.out or os.path.join(base_dir, f"{name}.png"))
    pages_dir = os.path.join(base_dir, f"{name.lower()}_pages")
    if args.pages:
        os.makedirs(pages_dir, exist_ok=True)
        for old in os.listdir(pages_dir):
            if old.endswith(".png"):
                os.remove(os.path.join(pages_dir, old))

    with open(doc, encoding="utf8") as fh:
        sections = split_sections(fh.read())

    images = []
    with tempfile.TemporaryDirectory() as tmp:
        profile = os.path.join(tmp, "profile")
        for i, (title, md) in enumerate(sections, 1):
            html = os.path.join(tmp, f"s{i:02d}.html")
            raw = os.path.join(tmp, f"s{i:02d}.png")
            with open(html, "w", encoding="utf8") as fh:
                fh.write(render_html(md, base_dir, f"{name}.md · part {i} of "
                                     f"{len(sections)}"))
            screenshot(browser, html, raw, profile)
            img = trim(raw)
            if img.height >= MAX_H * SCALE - 2:
                print(f"warning: section '{title}' may be cut off; "
                      "raise MAX_H in this script")
            images.append(img)
            if args.pages:
                page = os.path.join(pages_dir, f"{i:02d}_{slug(title)}.png")
                img.save(page, optimize=True)
                print(f"page {i:2d}/{len(sections)}  "
                      f"{os.path.relpath(page, ROOT)}")

    total = sum(im.height for im in images)
    sheet = Image.new("RGB", (images[0].width, total), (255, 255, 255))
    y = 0
    for im in images:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(out_png, optimize=True)
    print(f"whole    {os.path.relpath(out_png, ROOT)}   "
          f"{sheet.width} x {sheet.height} px, "
          f"{os.path.getsize(out_png) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
