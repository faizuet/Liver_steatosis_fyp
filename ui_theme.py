"""Visual design system for the Liver Diagnosis System UI.

Tokens, iconography, and small motion helpers live here so every screen
shares one look. No ML / prediction logic belongs in this module.
"""

from __future__ import annotations

from functools import lru_cache

import customtkinter as ctk
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------
BG = "#0B1016"
SIDEBAR = "#0E1620"
TOPBAR = "#101820"
SURFACE = "#151D27"
CARD = "#1B2430"
INSET = "#111820"
ELEVATED = "#222C38"
BORDER = "#2C3644"
BORDER_FOCUS = "#3BA89A"
TEXT = "#F2F5F8"
TEXT_SECONDARY = "#C5CDD6"
MUTED = "#8A95A3"
FAINT = "#5E6A78"
ACCENT = "#2F9E8F"
ACCENT_HOVER = "#3CB3A3"
ACCENT_SOFT = "#16332F"
OK = "#3DDC97"
OK_SOFT = "#14352C"
DANGER = "#E06B63"
DANGER_SOFT = "#3A1E1C"
WARN = "#E0A25A"
WHITE = "#FFFFFF"

FONT = "Segoe UI"

BREAKPOINT = 1140
SIDEBAR_WIDE = 228
SIDEBAR_NARROW = 76


def f(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=FONT, size=size, weight=weight)


def center(window, width: int, height: int) -> None:
    window.update_idletasks()
    x = max((window.winfo_screenwidth() - width) // 2, 0)
    y = max((window.winfo_screenheight() - height) // 2, 32)
    window.geometry(f"{width}x{height}+{x}+{y}")


def _hex(color: str) -> tuple[int, int, int, int]:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return r, g, b, 255


def _canvas(size: int, fill: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0) if fill is None else _hex(fill))
    return img, ImageDraw.Draw(img)


def _line(draw: ImageDraw.ImageDraw, xy, color: str, width: int = 2) -> None:
    draw.line(xy, fill=_hex(color), width=width, joint="curve")


@lru_cache(maxsize=64)
def _icon_pil(name: str, size: int, color: str) -> Image.Image:
    s = size * 2  # 2x for sharpness
    img, d = _canvas(s)
    c = color
    w = max(2, s // 12)
    pad = s // 5
    if name == "mark":
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=s // 4, fill=_hex(ACCENT))
        d.ellipse((s * 0.28, s * 0.22, s * 0.72, s * 0.62), outline=_hex(WHITE), width=w)
        d.polygon(
            [(s * 0.50, s * 0.40), (s * 0.34, s * 0.78), (s * 0.66, s * 0.78)],
            outline=_hex(WHITE),
        )
        _line(d, [(s * 0.50, s * 0.48), (s * 0.50, s * 0.70)], WHITE, w)
    elif name == "scan":
        L, R, T, B = pad, s - pad, pad, s - pad
        arm = s // 4
        _line(d, [(L, T + arm), (L, T), (L + arm, T)], c, w)
        _line(d, [(R - arm, T), (R, T), (R, T + arm)], c, w)
        _line(d, [(L, B - arm), (L, B), (L + arm, B)], c, w)
        _line(d, [(R - arm, B), (R, B), (R, B - arm)], c, w)
        d.ellipse((s * 0.32, s * 0.32, s * 0.68, s * 0.68), outline=_hex(c), width=w)
    elif name == "labs":
        # flask
        _line(d, [(s * 0.38, pad), (s * 0.38, s * 0.42), (s * 0.26, s * 0.78), (s * 0.74, s * 0.78), (s * 0.62, s * 0.42), (s * 0.62, pad)], c, w)
        _line(d, [(s * 0.34, pad), (s * 0.66, pad)], c, w)
        d.ellipse((s * 0.44, s * 0.52, s * 0.56, s * 0.64), fill=_hex(c))
    elif name == "image":
        d.rounded_rectangle((pad, pad, s - pad, s - pad), radius=s // 8, outline=_hex(c), width=w)
        d.ellipse((s * 0.30, s * 0.30, s * 0.48, s * 0.48), outline=_hex(c), width=w)
        _line(d, [(pad + w, s - pad - w), (s * 0.42, s * 0.52), (s * 0.58, s * 0.68), (s - pad - w, s * 0.46)], c, w)
    elif name == "check":
        d.ellipse((2, 2, s - 3, s - 3), fill=_hex(OK_SOFT), outline=_hex(OK), width=w)
        _line(d, [(s * 0.28, s * 0.52), (s * 0.44, s * 0.68), (s * 0.72, s * 0.34)], OK, w + 1)
    elif name == "alert":
        d.polygon(
            [(s * 0.50, pad), (s - pad, s - pad), (pad, s - pad)],
            outline=_hex(DANGER),
        )
        _line(d, [(s * 0.50, s * 0.38), (s * 0.50, s * 0.58)], DANGER, w)
        d.ellipse((s * 0.45, s * 0.66, s * 0.55, s * 0.76), fill=_hex(DANGER))
    elif name == "dot":
        d.ellipse((s * 0.30, s * 0.30, s * 0.70, s * 0.70), fill=_hex(c))
    elif name == "spark":
        _line(d, [(s * 0.50, pad), (s * 0.50, s - pad)], c, w)
        _line(d, [(pad, s * 0.50), (s - pad, s * 0.50)], c, w)
        _line(d, [(s * 0.28, s * 0.28), (s * 0.72, s * 0.72)], c, w)
        _line(d, [(s * 0.72, s * 0.28), (s * 0.28, s * 0.72)], c, w)
    else:
        d.ellipse((pad, pad, s - pad, s - pad), outline=_hex(c), width=w)

    return img.resize((size, size), Image.Resampling.LANCZOS)


def icon(name: str, size: int = 22, color: str = ACCENT) -> ctk.CTkImage:
    """Crisp flat icons. CTkImage is created per call so it stays bound to the live Tk root."""
    out = _icon_pil(name, size, color)
    return ctk.CTkImage(light_image=out, dark_image=out, size=(size, size))


@lru_cache(maxsize=8)
def _empty_art_pil(kind: str, size: int) -> Image.Image:
    """Soft empty-state illustration (exact palette, no stock clipart)."""
    s = size * 2
    img, d = _canvas(s)
    cx = cy = s // 2
    d.ellipse((cx - s * 0.46, cy - s * 0.46, cx + s * 0.46, cy + s * 0.46), fill=_hex(INSET))
    d.ellipse((cx - s * 0.34, cy - s * 0.34, cx + s * 0.34, cy + s * 0.34), outline=_hex(BORDER), width=3)
    if kind == "scan":
        m = s // 5
        arm = s // 7
        for pts in (
            [(m, m + arm), (m, m), (m + arm, m)],
            [(s - m - arm, m), (s - m, m), (s - m, m + arm)],
            [(m, s - m - arm), (m, s - m), (m + arm, s - m)],
            [(s - m - arm, s - m), (s - m, s - m), (s - m, s - m - arm)],
        ):
            _line(d, pts, ACCENT, 4)
        d.ellipse((cx - s * 0.16, cy - s * 0.16, cx + s * 0.16, cy + s * 0.16), outline=_hex(ACCENT), width=4)
    else:
        _line(
            d,
            [
                (s * 0.40, s * 0.22),
                (s * 0.40, s * 0.46),
                (s * 0.30, s * 0.74),
                (s * 0.70, s * 0.74),
                (s * 0.60, s * 0.46),
                (s * 0.60, s * 0.22),
            ],
            ACCENT,
            4,
        )
        _line(d, [(s * 0.36, s * 0.22), (s * 0.64, s * 0.22)], ACCENT, 4)
        d.ellipse((s * 0.46, s * 0.54, s * 0.54, s * 0.62), fill=_hex(ACCENT))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def empty_art(kind: str, size: int = 120) -> ctk.CTkImage:
    """Soft empty-state illustration (exact palette, no stock clipart)."""
    out = _empty_art_pil(kind, size)
    return ctk.CTkImage(light_image=out, dark_image=out, size=(size, size))
