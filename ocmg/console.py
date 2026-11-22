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

import atexit
import os
import re
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
    # OCMG_COLOR=1 forces colour on, so child processes of run_all.py keep it
    # when their output is being recorded (scripts/terminal_to_png.py).
    if os.environ.get("OCMG_COLOR") == "1":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    return _enable_windows_vt()


def _piped_windows() -> bool:
    """True when stdout is a Windows pipe (| Tee-Object, | more, ...).

    PowerShell decodes a native program's pipe with the console code page
    (437 on most machines), not UTF-8, so UTF-8 output arrives as mojibake.
    OCMG_UTF8=1 declares a pipe that does read UTF-8.
    """
    try:
        piped = not sys.stdout.isatty()
    except Exception:
        piped = True
    return os.name == "nt" and piped and os.environ.get("OCMG_UTF8") != "1"


def _console_codec() -> str:
    """The code page PowerShell uses to decode this process's pipe."""
    try:
        import codecs  # noqa: PLC0415
        import ctypes  # noqa: PLC0415
        cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        name = "utf-8" if cp == 65001 else f"cp{cp}"
        codecs.lookup(name)
        return name
    except Exception:
        return "ascii"


def _stream_encoding() -> str:
    if _piped_windows():
        return _console_codec()
    return (getattr(sys.stdout, "encoding", None) or "ascii").lower()


def _unicode_ok() -> bool:
    return "utf" in _stream_encoding()


def _encodable(text: str, enc: str) -> bool:
    try:
        text.encode(enc)
        return True
    except (UnicodeError, LookupError):
        return False


def _pick(candidates, enc: str) -> str:
    for c in candidates:
        if _encodable(c, enc):
            return c
    return candidates[-1]


# Best-to-plainest stand-ins for every symbol the output uses. Code page 437
# has the single-line box set, the full block and the middle dot, so tables
# keep their borders through | Tee-Object; the rest fall back to ASCII.
_FALLBACK = {
    "✔": ("√", "OK"), "✘": ("X",), "⚠": ("!",), "▲": ("^",), "▼": ("v",),
    "·": ("-",), "●": ("*",), "━": ("═", "="), "─": ("-",), "│": ("|",),
    "╭": ("┌", "+"), "╮": ("┐", "+"), "╰": ("└", "+"), "╯": ("┘", "+"),
    "┌": ("+",), "┐": ("+",), "└": ("+",), "┘": ("+",), "├": ("+",),
    "┤": ("+",), "┬": ("+",), "┴": ("+",), "┼": ("+",),
    "█": ("#",), "░": (".",), "▏": ("",), "▎": ("",), "▍": ("",),
    "▌": ("",), "▋": ("",), "▊": ("",), "▉": ("",),
    "±": ("+/-",), "\u00a0": (" ",), "≥": (">=",), "≤": ("<=",),
    "−": ("-",), "→": ("->",), "×": ("x",),
}


class _Translate:
    """A console stream that swaps symbols its code page cannot show."""

    def __init__(self, stream, enc: str):
        self._stream = stream
        self._map = str.maketrans({
            ch: _pick(alts, enc) for ch, alts in _FALLBACK.items()
            if not _encodable(ch, enc)})

    def write(self, text):
        return self._stream.write(text.translate(self._map))

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _Tee:
    """stdout that also writes a colour-free UTF-8 copy to a file."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, text):
        self._stream.write(text)
        self._fh.write(_ANSI.sub("", text))
        return len(text)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def save_to(path: Optional[str], append: bool = False) -> None:
    """Copy everything printed from here on into `path` (UTF-8, no colour).

    The replacement for `| Tee-Object`: the console stays a console, so it
    keeps its colour and glyphs, and the file is UTF-8 whatever the shell.
    `append` is for child processes adding to a log their parent started.
    """
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fh = open(path, "a" if append else "w", encoding="utf-8", buffering=1)
    original = sys.stdout
    sys.stdout = _Tee(original, fh)

    def _close() -> None:
        # Hand stdout back first: the interpreter flushes it once more after
        # atexit handlers run, and a closed file under it raises then.
        sys.stdout = original
        fh.close()

    atexit.register(_close)


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
    """Every symbol, chosen as the best the output encoding can show."""

    def __init__(self, enc="utf-8"):
        if isinstance(enc, bool):            # older callers passed unicode_ok
            enc = "utf-8" if enc else "ascii"
        p = lambda *c: _pick(c, enc)  # noqa: E731
        self.ok, self.bad, self.warn = p("✔", "√", "OK"), p("✘", "X"), p("⚠", "!")
        self.up, self.down, self.flat = p("▲", "^"), p("▼", "v"), p("·", "-")
        self.full, self.light = p("█", "#"), p("░", ".")
        eighths = ["▏", "▎", "▍", "▌", "▋", "▊", "▉"]
        self.parts = ([""] + eighths if all(_encodable(e, enc) for e in eighths)
                      else [""])
        self.rule, self.vrule, self.dot = p("─", "-"), p("│", "|"), p("·", "-")
        self.heavy = p("━", "═", "=")
        self.tl, self.tr = p("╭", "┌", "+"), p("╮", "┐", "+")
        self.bl, self.br = p("╰", "└", "+"), p("╯", "┘", "+")
        # square corners and junctions for tables
        self.ctl, self.ctr = p("┌", "+"), p("┐", "+")
        self.cbl, self.cbr = p("└", "+"), p("┘", "+")
        self.tee_d, self.tee_u = p("┬", "+"), p("┴", "+")
        self.tee_r, self.tee_l, self.cross = p("├", "+"), p("┤", "+"), p("┼", "+")
        # outside UTF-8 a plain space, so copied text has no hidden NBSP
        self.nbsp = "\u00a0" if "utf" in str(enc).lower() else " "
        self.bullet = p("●", "*")


G = Glyphs(_stream_encoding())


def refresh_glyphs() -> None:
    """Re-detect the output encoding (call after reconfiguring stdout).

    On a Windows pipe the console side is switched to the console code page
    and wrapped so any symbol it cannot show is replaced, never garbled. A
    --save file sits outside that wrapper and keeps full UTF-8.
    """
    global G
    if _piped_windows():
        enc = _console_codec()
        holder = sys.stdout
        target = holder._stream if isinstance(holder, _Tee) else holder
        if not isinstance(target, _Translate):
            try:
                target.reconfigure(encoding=enc, errors="replace")
            except Exception:
                pass
            wrapped = _Translate(target, enc)
            if isinstance(holder, _Tee):
                holder._stream = wrapped
            else:
                sys.stdout = wrapped
    G = Glyphs(_stream_encoding())


# ------------------------------------------------------------------ structure

def _term_cols() -> int:
    import shutil  # noqa: PLC0415
    return shutil.get_terminal_size((120, 24)).columns


# The block width follows the terminal, between a readable minimum and a
# maximum past which lines get too long to scan. Every helper below wraps to
# it, so no line is ever wider than the window and nothing wraps by accident.
WIDTH = max(78, min(_term_cols() - 2, 112))


def _wrap_words(text: str, width: int) -> List[str]:
    """Word-wrap by printable width; escape sequences are never split.

    Splits on plain spaces only, so a non-breaking space keeps a phrase such
    as a status badge on one line.
    """
    lines: List[str] = []
    cur: List[str] = []
    used = 0
    for word in text.split(" "):
        if not cur and word == "":
            continue                      # no leading blanks on a new line
        ww = _w(word)
        add = ww + (1 if cur else 0)
        if cur and used + add > width:
            lines.append(" ".join(cur).rstrip())
            cur, used = ([word], ww) if word else ([], 0)
        else:
            cur.append(word)
            used += add
    if cur:
        lines.append(" ".join(cur).rstrip())
    return lines or [""]


_ONE_PAINT = re.compile(r"^(\x1b\[[0-9;]*m)([^\x1b]*)\x1b\[0m$", re.S)


def _wrap_cell(text: str, width: int) -> List[str]:
    """Wrap a table cell; a cell painted in one colour stays that colour on
    every line instead of bleeding into the next column."""
    if _w(text) <= width:
        return [text]
    m = _ONE_PAINT.match(text)
    if m:
        code, inner = m.groups()
        return [f"{code}{line}\x1b[0m" for line in _wrap_words(inner, width)]
    lines = _wrap_words(text, width)
    if "\x1b" in text:
        lines = [line + "\x1b[0m" for line in lines]
    return lines


def banner(title: str, subtitle: str = "") -> str:
    """The run's title in a rounded box."""
    inner = WIDTH - 2

    def row(text: str, *roles: str) -> str:
        text = text if _w(text) <= inner - 2 else text[: inner - 5] + "..."
        return (paint(G.vrule, "dim") + " " + paint(pad(text, inner - 2), *roles)
                + " " + paint(G.vrule, "dim"))

    lines = [paint(G.tl + G.rule * inner + G.tr, "dim"),
             row(title, "bold", "emphasis")]
    if subtitle:
        lines.append(row(subtitle, "muted"))
    lines.append(paint(G.bl + G.rule * inner + G.br, "dim"))
    return "\n".join(lines)


def section(number: str, title: str) -> str:
    """A numbered section header: bold title, muted subtitle, hairline rest.

    A title written as "NAME   subtitle" (three spaces) is split so the
    name stands out and the explanation recedes.
    """
    name, _, sub = title.partition("   ")
    name, sub = name.strip(), sub.strip()
    parts = [paint(G.heavy * 2, "emphasis"), " "]
    plain = G.heavy * 2 + " "
    if number:
        parts += [paint(number, "bold", "emphasis"), paint(f" {G.dot} ", "dim")]
        plain += f"{number} {G.dot} "
    parts.append(paint(name, "bold"))
    plain += name
    if sub and _w(plain) + 2 + _w(sub) + 4 <= WIDTH:
        parts.append("  " + paint(sub, "muted"))
        plain += "  " + sub
    tail = max(2, WIDTH - _w(plain) - 1)
    out = "\n" + "".join(parts) + " " + paint(G.rule * tail, "dim")
    if sub and sub not in plain:
        out += "\n" + note(sub, "muted")
    return out


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
    """label, bold value, muted explanation on the same line.

    An explanation too long for the window wraps under its own column, or
    under the value when that column would be too narrow to read.
    """
    head = f"  {pad(label, width)}{paint(value, 'bold')}"
    if not hint:
        return head
    col = _w(head) + 3
    avail = WIDTH - col
    if _w(hint) <= avail:
        return head + "   " + paint(hint, "muted")
    if avail >= 32:
        lines = _wrap_words(hint, avail)
        return head + "   " + paint(lines[0], "muted") + "".join(
            "\n" + " " * col + paint(line, "muted") for line in lines[1:])
    indent = 2 + width
    return head + "".join(
        "\n" + " " * indent + paint(line, "muted")
        for line in _wrap_words(hint, WIDTH - indent))


def badge(text: str, role: str, glyph: str = "") -> str:
    """A coloured status word with its glyph, never split across lines."""
    body = f"{glyph} {text}" if glyph else text
    return paint(body.replace(" ", G.nbsp), role)


def status(ok: bool, met: str = "MET", unmet: str = "NOT MET") -> str:
    """A pass/fail badge: colour, glyph and word together, never colour alone."""
    if ok:
        return badge(met, "good", G.ok)
    return badge(unmet, "critical", G.bad)


def grade(value: float, good_at: float, fair_at: float,
          words=("GOOD", "FAIR", "WEAK"), higher_is_better: bool = True) -> str:
    """Three-level badge: good / caution / bad by two thresholds."""
    v = value if higher_is_better else -value
    g, f = (good_at, fair_at) if higher_is_better else (-good_at, -fair_at)
    if v >= g:
        return badge(words[0], "good", G.ok)
    if v >= f:
        return badge(words[1], "warning", G.warn)
    return badge(words[2], "critical", G.bad)


def rank_paint(values, fmt: str = "{:.3f}", higher_is_better: bool = True):
    """Format a column, the best value green and the worst red.

    Colour carries rank only; the number itself is always printed, so the
    column reads the same without colour.
    """
    nums = [v for v in values if v is not None]
    if len(set(nums)) < 2:
        return ["-" if v is None else fmt.format(v) for v in values]
    best = max(nums) if higher_is_better else min(nums)
    worst = min(nums) if higher_is_better else max(nums)
    out = []
    for v in values:
        if v is None:
            out.append("-")
            continue
        text = fmt.format(v)
        out.append(paint(text, "good") if v == best
                   else paint(text, "critical") if v == worst else text)
    return out


def color_key() -> str:
    """One line saying what the colours mean, or why there are none."""
    if use_color():
        b = G.bullet
        return ("  " + paint("colour key", "muted") + "   "
                + paint(f"{b} good / best / target met", "good") + "   "
                + paint(f"{b} caution", "warning") + "   "
                + paint(f"{b} bad / worst / target missed", "critical") + "   "
                + paint("OCMG", "bold", "emphasis"))
    if _piped_windows():
        return note("colours are off because the output is piped (| Tee-Object). "
                    "For colours on screen and a clean text file, run the same "
                    "command with --save FILE instead of the pipe.", "muted")
    return ""


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
    """A bordered table: one column per field, a rule under the header.

    Never wider than the block width: when it would be, the last column
    (a text column) wraps inside its own cell. The emphasised row has its
    name in the emphasis colour and its numbers bold.
    """
    aligns = list(aligns or ["left"] + ["right"] * (len(headers) - 1))
    n = len(headers)
    widths = [
        max(_w(str(headers[i])), max((_w(str(r[i])) for r in rows), default=0))
        for i in range(n)
    ]

    def total(ws):                       # indent + borders + padding
        return 2 + 1 + sum(w + 3 for w in ws)

    wrap_last = False
    if n > 1 and aligns[-1] == "left" and total(widths) > WIDTH:
        room = WIDTH - (total(widths) - widths[-1])
        if room >= 12:
            widths[-1], wrap_last = room, True

    v = paint(G.vrule, "dim")

    def hline(left: str, mid: str, right: str) -> str:
        return "  " + paint(left + mid.join(G.rule * (w + 2) for w in widths)
                            + right, "dim")

    def line(cells) -> str:
        return "  " + v + v.join(f" {c} " for c in cells) + v

    out = [hline(G.ctl, G.tee_d, G.ctr),
           line([paint(pad(str(h), widths[i], aligns[i]), "bold")
                 for i, h in enumerate(headers)]),
           hline(G.tee_r, G.cross, G.tee_l)]
    for k, row in enumerate(rows):
        last = (_wrap_cell(str(row[-1]), widths[-1]) if wrap_last
                else [str(row[-1])])
        cells = ([pad(str(c), widths[i], aligns[i]) for i, c in enumerate(row[:-1])]
                 + [pad(last[0], widths[-1], aligns[-1])])
        if k == emphasis_row:
            cells = ([paint(cells[0], "bold", "emphasis")]
                     + [paint(c, "bold") for c in cells[1:-1]] + [cells[-1]])
        out.append(line(cells))
        for extra in last[1:]:
            out.append(line([" " * widths[i] for i in range(n - 1)]
                            + [pad(extra, widths[-1], aligns[-1])]))
    out.append(hline(G.cbl, G.tee_u, G.cbr))
    return "\n".join(out)


def facts(rows: Sequence[tuple]) -> str:
    """Key results as a table.

    Each row is (measure, value, meaning) or (measure, value, status,
    meaning). The status column appears only when some row has one.
    """
    has_status = any(len(r) > 3 and r[2] for r in rows)
    if has_status:
        data = [[r[0], paint(str(r[1]), "bold"), r[2] if len(r) > 3 else "",
                 paint(r[-1], "muted") if len(r) > 2 else ""] for r in rows]
        return table(["measure", "value", "status", "meaning"], data,
                     ["left", "left", "left", "left"])
    data = [[r[0], paint(str(r[1]), "bold"),
             paint(r[-1], "muted") if len(r) > 2 else ""] for r in rows]
    return table(["measure", "value", "meaning"], data, ["left", "left", "left"])


def card(title: str, rows: Sequence[tuple]) -> str:
    """A boxed summary: (label, value[, explanation]) per row.

    The explanation follows the value when it fits and wraps beneath it,
    inside the box, when it does not.
    """
    inner = WIDTH - 4                    # between "  |" and "|"
    lab_w = max([_w(r[0]) for r in rows] + [10]) + 2
    vbar = paint(G.vrule, "dim")

    def boxed(text: str) -> str:
        return "  " + vbar + " " + pad(text, inner - 2) + " " + vbar

    # tl + rule + space + title + space + fill + tr = inner + 2, like the rows
    lines = ["  " + paint(G.tl + G.rule, "dim") + " " + paint(title, "bold", "emphasis")
             + " " + paint(G.rule * max(2, inner - _w(title) - 3) + G.tr, "dim")]
    for r in rows:
        label, value = r[0], r[1]
        expl = r[2] if len(r) > 2 else ""
        first = pad(label, lab_w) + paint(value, "bold")
        room = inner - 2 - _w(first) - 3
        if expl and _w(expl) <= room:
            lines.append(boxed(first + "   " + paint(expl, "muted")))
            continue
        lines.append(boxed(first))
        if expl:
            for line in _wrap_words(expl, inner - 2 - lab_w):
                lines.append(boxed(" " * lab_w + paint(line, "muted")))
    lines.append("  " + paint(G.bl + G.rule * inner + G.br, "dim"))
    return "\n".join(lines)


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
