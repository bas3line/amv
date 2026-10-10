"""End credit row: X logo + inlovewithgo, Instagram logo + rword_shubham, GitHub logo + bas3line.

One centred row in the bottom 78 px letterbox bar (y 1002-1080) under the Hokage hero shot; it stays
put through the black end card (the card is black there too) and fades with the song.

  credits_rgba(t) -> uint8 (1080, 1920, 4), white RGB + straight alpha; t = seconds since T0
  composite(frame_bgr, t) -> frame_bgr with the row laid over it (returns the frame untouched outside the span)

Logos: simple-icons@16.30.0 (cdn.jsdelivr.net/npm/simple-icons@16.30.0/icons/{x,instagram,github}.svg),
rasterised by resvg at 4x and area-downsampled. Text: Montserrat SemiBold (wght 600), also drawn at 4x.
Animation: each item fades in and rises 14 px, one beat (0.41098 s) apart, ease-out over 8 frames;
then holds; everything fades 89.082 -> 89.932 with the end card (same curve as titles.birthday_card).
"""
import functools
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import resvg_py

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_DIR = os.path.join(WORK, "fx", "logos")
FONT = os.path.join(WORK, "fonts", "Montserrat[wght].ttf")
WGHT = 600
W, H = 1920, 1080
FPS = 24000 / 1001
SS = 4                          # supersample factor for logos and text

T0 = 85.905                     # edit time the first item starts (one beat after the 85.494 hero cut)
BEAT = 0.41098
IN_DUR = 8 / FPS                # 8-frame ease-out per item
RISE = 14.0
FADE0, FADE1 = 89.082, 89.932   # titles.FADE0 / FADE1: the end card's fade with the song
END = 89.964                    # last edit time drawn

CAP = 34                        # cap height (px at 1080p)
LOGO = round(1.15 * CAP)        # logo box (39 px)
GAP_LT = round(0.45 * CAP)      # logo -> handle
GAP_IT = round(2.0 * CAP)       # item -> item
CY = 1041                       # vertical centre of the row = centre of the bottom letterbox bar

ITEMS = (("x", "inlovewithgo"), ("instagram", "rword_shubham"), ("github", "bas3line"))


def _clamp(u):
    return 0.0 if u < 0 else 1.0 if u > 1 else u


def _ease_out_cubic(u):
    u = _clamp(u)
    return 1 - (1 - u) ** 3


def _ease_in_out_sine(u):
    u = _clamp(u)
    return -(math.cos(math.pi * u) - 1) / 2


@functools.lru_cache(maxsize=None)
def _font():
    f = ImageFont.truetype(FONT, round(CAP * SS / 0.70))     # Montserrat cap height = 0.70 em
    f.set_variation_by_axes([WGHT])
    return f


@functools.lru_cache(maxsize=None)
def _logo_ss(name):
    """-> float32 alpha (LOGO*SS, LOGO*SS) of the white logo at supersampled size."""
    with open(os.path.join(LOGO_DIR, f"{name}.svg")) as fh:
        svg = fh.read().replace("<path ", '<path fill="#ffffff" ', 1)
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=LOGO * SS, height=LOGO * SS))
    rgba = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    return rgba[..., 3].astype(np.float32) / 255.0


@functools.lru_cache(maxsize=None)
def _items():
    """-> list of (alpha patch float32 at 1x, x0, y0): one per item, laid out as a centred row."""
    f = _font()
    widths = [LOGO + GAP_LT + f.getlength(t) / SS for _, t in ITEMS]
    total = sum(widths) + GAP_IT * (len(ITEMS) - 1)
    x = (W - total) / 2
    pad = 8
    out = []
    for (name, text), wd in zip(ITEMS, widths):
        x0 = int(math.floor(x)) - pad
        y0 = CY - 40
        pw, ph = int(math.ceil(wd)) + 2 * pad + 2, min(80, H - y0)
        cvs = np.zeros((ph * SS, pw * SS), np.float32)
        # logo, vertically centred on CY
        lx = int(round((x - x0) * SS))
        ly = int(round((CY - LOGO / 2 - y0) * SS))
        cvs[ly:ly + LOGO * SS, lx:lx + LOGO * SS] = np.maximum(cvs[ly:ly + LOGO * SS, lx:lx + LOGO * SS],
                                                               _logo_ss(name))
        # handle: caps centred on CY (baseline = CY + CAP/2)
        im = Image.new("L", (pw * SS, ph * SS), 0)
        ImageDraw.Draw(im).text(((x - x0 + LOGO + GAP_LT) * SS, (CY + CAP / 2 - y0) * SS), text,
                                font=f, fill=255, anchor="ls")
        cvs = np.maximum(cvs, np.asarray(im, np.float32) / 255.0)
        patch = cv2.resize(cvs, (pw, ph), interpolation=cv2.INTER_AREA)
        out.append((patch, x0, y0))
        x += wd + GAP_IT
    return out


def _alpha(t):
    """-> float32 (H, W) alpha of the whole row at local time t (seconds since T0)."""
    a = np.zeros((H, W), np.float32)
    fade = 1.0 - _ease_in_out_sine((T0 + t - FADE0) / (FADE1 - FADE0))
    if t < 0 or fade <= 0:
        return a
    for i, (patch, x0, y0) in enumerate(_items()):
        e = _ease_out_cubic((t - i * BEAT) / IN_DUR)
        if e <= 0:
            continue
        dy = RISE * (1 - e)
        p = patch
        if dy > 1e-3:
            p = cv2.warpAffine(patch, np.float32([[1, 0, 0], [0, 1, dy]]), (patch.shape[1], patch.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ph, pw = p.shape
        a[y0:y0 + ph, x0:x0 + pw] = np.maximum(a[y0:y0 + ph, x0:x0 + pw], p * (e * fade))
    return a


def credits_rgba(t):
    """t = seconds since T0 -> uint8 (1080, 1920, 4): white, straight alpha."""
    out = np.full((H, W, 4), 255, np.uint8)
    out[..., 3] = np.clip(_alpha(t) * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def composite(frame_bgr, t):
    """Lay the row over a BGR uint8 frame. t = seconds since T0."""
    if t < 0 or T0 + t >= FADE1:
        return frame_bgr
    a = _alpha(t)[..., None]
    f = frame_bgr.astype(np.float32)
    return np.clip(f + (255.0 - f) * a + 0.5, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    for (patch, x0, y0), (n, txt) in zip(_items(), ITEMS):
        print(n, txt, "x0", x0, "y0", y0, "w", patch.shape[1])
