"""Terminal presentation for experiment output.

The console output is a figure, not a log. It is what gets shown in the viva
and in the recorded presentation, so it follows the same encoding discipline as
the thesis figures rather than printing whatever is convenient.

Four rules are enforced here and are worth stating, because the obvious
implementation breaks each one.

Nominal categories get one hue, not one hue each. The six conditions have no
natural order, so colouring each a different colour would burn the only free
visual channel on information the numbers already carry, and nothing would
stand out. Instead every bar takes the same hue and the condition under
discussion is emphasised while the rest recede. The story is one row, so one
row is highlighted.

Status colour never carries meaning alone. Green and red appear only where the
colour *means* pass or fail, and always alongside a glyph and a word, so the
output survives a projector, a colourblind reader, a monochrome print and a
redirect into a file.

Accuracy and token cost never share a chart. They have different units and
different scales; putting them on one axis would invent a relationship that is
not in the data. They sit in separate columns, and only the derived ratio
(accuracy per thousand tokens) is plotted.

Numbers wear text colour, not series colour. The bar carries identity; the
figures beside it stay in ordinary ink so the table remains readable.

Colour degrades in three steps: 256-colour, then the 16-colour terminal
defaults, then none at all. It is switched off automatically when the stream is
redirected, when NO_COLOR is set, or on a dumb terminal, so piping a run into a
file produces clean text.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence

# ----------------------------------------------------------------- capability

_FORCE: Optional[bool] = None


def _enable_windows_vt() -> bool:
    """Turn on ANSI escape handling in the Windows console.

    Windows Terminal and the VS Code terminal handle escapes natively, but a
    plain conhost window does not until this flag is set.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32
        # -11 is STD_OUTPUT_HANDLE; 0x0004 is ENABLE_VIRTUAL_TERMINAL_PROCESSING.
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def set_color(enabled: Optional[bool]) -> None:
    """Force colour on or off; None restores automatic detection."""
    global _FORCE
    _FORCE = enabled


def use_color() -> bool:
    if _FORCE is not None:
        return _FORCE
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    return _enable_windows_vt()


def _unicode_ok() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


# --------------------------------------------------------------------- colour
# 256-colour indices chosen as the nearest cube steps to the design system's
# tokens, so the terminal and the thesis figures read as the same palette.
_C = {
    "emphasis": "38;5;25",     # sequential blue 500  #256abf
    "emphasis_lo": "38;5;111",  # sequential blue 250  #86b6ef
    "muted": "38;5;245",       # muted ink            #898781
    "dim": "38;5;240",         # chart chrome         #52514e
    "good": "38;5;34",         # status good          #0ca30c
    "warning": "38;5;214",     # status warning       #fab219
    "critical": "38;5;167",    # status critical      #d03b3b
    "bold": "1",
}
# Fallback for terminals that only honour the original sixteen.
_C16 = {
    "emphasis": "34", "emphasis_lo": "94", "muted": "37", "dim": "90",
    "good": "32", "warning": "33", "critical": "31", "bold": "1",
}


def paint(text: str, *roles: str) -> str:
    if not roles or not use_color():
        return text
    table = _C if os.environ.get("TERM") != "linux" else _C16
    codes = ";".join(table.get(r, _C16.get(r, "")) for r in roles if r)
    if not codes:
        return text
    return f"\033[{codes}m{text}\033[0m"


def _w(text: str) -> int:
    """Printable width, ignoring escape sequences."""
    out, in_esc = 0, False
    for ch in text:
        if in_esc:
            in_esc = ch != "m"
            continue
        if ch == "\033":
            in_esc = True
            continue
        out += 1
    return out


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - _w(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


# ---------------------------------------------------------------- glyph sets

class Glyphs:
    def __init__(self, unicode_ok: bool):
        if unicode_ok:
            self.ok, self.bad, self.warn = "✔", "✘", "⚠"
            self.up, self.down, self.flat = "▲", "▼", "·"
            self.full, self.light = "█", "░"
            self.parts = ["", "▏", "▎", "▍", "▌",
                          "▋", "▊", "▉"]
            self.rule, self.vrule, self.dot = "─", "│", "·"
        else:
            self.ok, self.bad, self.warn = "OK", "X", "!"
            self.up, self.down, self.flat = "^", "v", "-"
            self.full, self.light = "#", "."
            self.parts = [""]
            self.rule, self.vrule, self.dot = "-", "|", "-"


G = Glyphs(_unicode_ok())


def refresh_glyphs() -> None:
    """Re-detect the output encoding (after a stream is reconfigured)."""
    global G
    G = Glyphs(_unicode_ok())


# ------------------------------------------------------------------ structure

WIDTH = 78


def banner(title: str, subtitle: str = "") -> str:
    lines = [paint(G.rule * WIDTH, "dim"), paint(title, "bold", "emphasis")]
    if subtitle:
        lines.append(paint(subtitle, "muted"))
    lines.append(paint(G.rule * WIDTH, "dim"))
    return "\n".join(lines)


def section(number: str, title: str) -> str:
    """A numbered section rule: solid hairline, one shade off the surface."""
    stem = f"{number} {G.dot} " if number else ""
    head = f"{G.rule}{G.rule}  {stem}{title}  "
    tail = G.rule * max(0, WIDTH - _w(head))
    return "\n" + paint(head, "dim") + paint(tail, "dim")


def note(text: str, role: str = "muted", glyph: str = "") -> str:
    """A prose line, wrapped to the block width and hanging-indented.

    Wrapping happens before painting. Wrapping already-painted text would
    measure the escape sequences as printable characters and break the line
    far too early, mid-word.
    """
    import textwrap  # noqa: PLC0415

    prefix = f"{glyph} " if glyph else ""
    if "\033" in text:
        # Already carries its own colour; wrapping it would count escapes.
        return f"  {prefix}{text}"
    body = textwrap.wrap(text, width=WIDTH - 2 - len(prefix)) or [""]
    pad_to = " " * (2 + len(prefix))
    lines = [f"  {prefix}{body[0]}"] + [f"{pad_to}{b}" for b in body[1:]]
    return "\n".join(paint(l, role) for l in lines)


def kv(label: str, value: str, width: int = 24, hint: str = "") -> str:
    line = f"  {pad(label, width)}{value}"
    if hint:
        line += paint(f"   {hint}", "muted")
    return line


def status(ok: bool, met: str = "MET", unmet: str = "NOT MET") -> str:
    """A pass/fail badge: colour, glyph and word together, never colour alone."""
    if ok:
        return paint(f"{G.ok} {met}", "good")
    return paint(f"{G.bad} {unmet}", "critical")


def delta(
    value: float,
    higher_is_better: bool = True,
    fmt: str = "{:+.3f}",
    neutral: float = 0.0,
) -> str:
    """A signed change, coloured and glyphed by whether it is an improvement.

    The arrow encodes *valence*, not direction: the sign of the number already
    says which way the value moved, so spending the glyph on direction too
    would leave the colour as the only carrier of good-versus-bad. On a cost
    column, where an increase is bad, that would print a red up-arrow - and a
    reader who cannot see the red would take the arrow for good news. Arrow up
    always means better.
    """
    if abs(value) <= neutral:
        return paint(G.flat, "muted")
    better = (value > 0) if higher_is_better else (value < 0)
    glyph = G.up if better else G.down
    return paint(f"{fmt.format(value)} {glyph}", "good" if better else "critical")


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]] = None,
    emphasis_row: Optional[int] = None,
) -> str:
    """An aligned table with a recessive header rule.

    The emphasised row is brightened rather than recoloured, so the column
    semantics stay intact.
    """
    aligns = list(aligns or ["left"] + ["right"] * (len(headers) - 1))
    widths = [
        max(_w(str(headers[i])), max((_w(str(r[i])) for r in rows), default=0))
        for i in range(len(headers))
    ]
    out = [
        "  " + "  ".join(
            paint(pad(str(h), widths[i], aligns[i]), "muted")
            for i, h in enumerate(headers)
        ),
        "  " + paint(G.rule * (sum(widths) + 2 * (len(widths) - 1)), "dim"),
    ]
    for n, row in enumerate(rows):
        cells = [pad(str(c), widths[i], aligns[i]) for i, c in enumerate(row)]
        line = "  " + "  ".join(cells)
        if n == emphasis_row:
            line = "  " + "  ".join(
                paint(cells[0], "bold", "emphasis") if i == 0 else cells[i]
                for i in range(len(cells))
            )
        out.append(line)
    return "\n".join(out)


def hbar(
    label: str,
    value: float,
    vmax: float,
    width: int = 28,
    label_width: int = 18,
    emphasis: bool = False,
    value_fmt: str = "{:.3f}",
    suffix: str = "",
) -> str:
    """One horizontal bar.

    Sub-character eighths are used so that small differences remain visible at
    a narrow width instead of collapsing to the same number of blocks.
    """
    frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    filled = frac * width
    whole = int(filled)
    rest = filled - whole
    bar = G.full * whole
    if len(G.parts) > 1 and whole < width:
        bar += G.parts[int(rest * 8)]
    role = "emphasis" if emphasis else "dim"
    body = paint(pad(bar, width), role)
    val = value_fmt.format(value)
    val = paint(val, "bold") if emphasis else val
    return (
        f"  {pad(label, label_width)}{pad(val, 7, 'right')}  {body}"
        + (paint(f"  {suffix}", "muted") if suffix else "")
    )


def bar_chart(
    items: Sequence[tuple],
    title: str = "",
    width: int = 28,
    label_width: int = 18,
    value_fmt: str = "{:.3f}",
    emphasise: Optional[str] = None,
) -> str:
    """A horizontal bar chart over nominal categories.

    One hue for every bar with a single emphasised category, which is the
    correct encoding when the categories have no order and the point of the
    chart is one of them. A per-category palette would double-encode bar
    length as hue and leave nothing emphasised.

    `items` is a sequence of (label, value) or (label, value, suffix).
    """
    vmax = max((i[1] for i in items), default=0.0)
    lines = []
    if title:
        lines.append(note(title, "muted"))
    for item in items:
        label, value = item[0], item[1]
        suffix = item[2] if len(item) > 2 else ""
        lines.append(
            hbar(
                label, value, vmax, width=width, label_width=label_width,
                emphasis=(label == emphasise), value_fmt=value_fmt, suffix=suffix,
            )
        )
    return "\n".join(lines)


def before_after(
    label_before: str,
    before: float,
    label_after: str,
    after: float,
    width: int = 28,
    label_width: int = 22,
    higher_is_better: bool = True,
    value_fmt: str = "{:.3f}",
    reference: Optional[tuple] = None,
) -> str:
    """Two bars on one scale, with the change called out.

    The 'before' bar recedes and the 'after' bar is emphasised: this is a
    change, so the comparison is the subject and the two states are not
    competing categories.
    """
    vmax = max(before, after, (reference[1] if reference else 0.0))
    lines = [
        hbar(label_before, before, vmax, width, label_width, False, value_fmt),
        hbar(label_after, after, vmax, width, label_width, True, value_fmt),
    ]
    if reference:
        # Aligned to the same value column as the bars above it, so the eye
        # reads it as the same scale rather than a stray line.
        lines.append(
            paint(
                "  "
                + pad(reference[0], label_width)
                + pad(value_fmt.format(reference[1]), 7, "right"),
                "muted",
            )
        )
    change = after - before
    better = (change > 0) if higher_is_better else (change < 0)
    glyph = G.up if change > 0 else G.down
    lines.append(
        "  "
        + pad("", label_width)
        + paint(
            f"{change:+.3f} {glyph}  "
            + ("improvement" if better else "regression"),
            "good" if better else "critical",
        )
    )
    return "\n".join(lines)


def legend(entries: Sequence[tuple]) -> str:
    """A legend is always present where colour carries meaning.

    Laid out by printable width, so an entry moves to the next line only when
    it genuinely does not fit. Each entry stays intact: a legend broken
    mid-phrase is worse than one that runs to two lines.
    """
    lines: List[str] = []
    current: List[str] = []
    used = 2
    for glyph, text, role in entries:
        entry = f"{glyph} {text}"
        need = _w(entry) + (3 if current else 0)
        if current and used + need > WIDTH:
            lines.append("  " + "   ".join(current))
            current, used = [], 2
            need = _w(entry)
        current.append(paint(entry, role))
        used += need
    if current:
        lines.append("  " + "   ".join(current))
    return "\n".join(lines)
