"""refpack_A -- the effects the reference edits use and DRAFT_v0 does not (refgap pass A).

Refs studied frame by frame (sheets in work/refgap_A/):
  G3GC3YhGuSo  (23.976 fps, so ref frame counts map 1:1 to our edit)
  fUx4vgbpPi4  (30 fps; frame counts below already converted x0.8 to 23.976)
  MoxZ7YVRO0w  (30 fps; same conversion)

Contract (same as fxpack): uint8 BGR (1080, 1920, 3) in and out, pure, deterministic
(randomness only from explicit seeds). `n` = frames since the beat frame (n = 0 is the
first clean frame of the new shot), `k` = frame index inside a transition, `u` = 0..1
progress inside a hold. Heavy blurs run at half or quarter resolution: they only ever
appear on the 1-3 transition frames where the picture is meant to be smeared.
Held frames never exceed 1.10x and are re-sharpened, so the footage stays readable.

fxpack.py and director.py are imported, never edited.
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if f"{_W}/fx" not in sys.path:
    sys.path.insert(0, f"{_W}/fx")
import fxpack as fx  # noqa: E402

W, H = 1920, 1080
FPS = 24000 / 1001
FONT_DIR = f"{_W}/fonts"


# ----------------------------------------------------------------------------- helpers
def _f32(img):
    return img.astype(np.float32)


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _screen(a, b):
    a, b = _f32(a) / 255.0, _f32(b) / 255.0
    return _u8((1 - (1 - a) * (1 - b)) * 255)


def _luma(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


@lru_cache(maxsize=64)
def _line_kernel(length, angle_deg):
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), np.float32)
    c = length // 2
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    for t in np.linspace(-c, c, length * 2):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        k[y, x] = 1.0
    return k / k.sum()


def dir_blur_fast(img, length_px, angle_deg, down=4):
    """Directional (motion) blur of `length_px` at full res, computed at 1/down res."""
    if length_px < 2:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // down, h // down), interpolation=cv2.INTER_AREA)
    k = _line_kernel(max(3, int(length_px / down)), float(angle_deg))
    small = cv2.filter2D(small, -1, k, borderType=cv2.BORDER_REFLECT101)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def zoom_blur(img, strength, center=(0.5, 0.5), taps=10, down=2):
    """Radial zoom blur: average of `taps` copies scaled 1 .. 1+strength about center."""
    if strength <= 0.002:
        return img
    h, w = img.shape[:2]
    sw, sh = w // down, h // down
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    cx, cy = center[0] * sw, center[1] * sh
    acc = np.zeros_like(small)
    for i in range(taps):
        s = 1.0 + strength * i / (taps - 1)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return cv2.resize(_u8(acc / taps), (w, h), interpolation=cv2.INTER_LINEAR)


def _scale_about(img, s, center=(0.5, 0.5), rot=0.0, dx=0.0, dy=0.0):
    return fx.xform(img, s=s, rot=rot, dx=dx, dy=dy, center=center)


def _sharpen(img, amount=0.35, sigma=1.0):
    return fx.sharpen(img, amount=amount, sigma=sigma)


# ============================================================================ 1. SMEAR WHIP
# G3GC 7.90-7.98, 8.40, 8.90, 9.40, 9.90, 12.87-12.91, 13.37-13.41, 13.87-13.91, 15.80, 16.80,
# 17.30, 17.80, 19.24-19.28; MoxZ7 68.0, 68.4, 69.1, 69.5, 69.9, 70.2, 71.0. Two frames per cut,
# the picture is an almost unreadable colour smear (Laplacian var 2-31 vs 300-9000 on held frames),
# luma kept (no flash). Blur length measured on the 400 px sheet: 40-60 px -> 190-290 px at 1920.
SMEAR_LEN = (190.0, 280.0)     # k=0 (outgoing, b-2), k=1 (incoming, b-1)
SMEAR_SCALE = (1.05, 1.09)


def smear_whip(img, k, angle=-18.0, outgoing=None, center=(0.5, 0.5)):
    """2-frame motion-blur whip.  k=0: outgoing shot on b-2; k=1: incoming shot on b-1.
    Frame b (n=0) is the incoming shot clean. `outgoing` defaults to k==0."""
    out = (k == 0) if outgoing is None else outgoing
    ln = SMEAR_LEN[0 if out else 1]
    s = SMEAR_SCALE[0 if out else 1]
    a = math.radians(angle)
    sign = 1.0 if out else -1.0     # outgoing travels forward, incoming arrives from behind
    dx, dy = sign * 0.06 * W * math.cos(a), sign * 0.06 * W * math.sin(a)
    moved = _scale_about(img, s, center, dx=dx, dy=dy)
    return dir_blur_fast(moved, ln, angle)


def white_smear(img, k, angle=0.0, lift=0.62, desat=0.55):
    """G3GC 7.40-7.48 and 15.30: the 2-frame bridge into the manga panels is a smear lifted
    toward warm white (luma 184-192, sat ~25%). Use it into a bright or B/W shot."""
    sm = smear_whip(img, k, angle=angle)
    f = _f32(sm)
    g = _luma(sm)[..., None]
    f = f * (1 - desat) + g * desat
    f = f * (1 - lift) + np.float32([236, 242, 246]) * lift
    return _u8(f)


# ======================================================================= 2. ZOOM-BLUR ENTRY
# G3GC 6.98 (into the eye), 14.80, 8.03-8.07; fUx4 46.57-46.63 (into the blue Susanoo);
# the new shot's first 1-2 frames are radially zoom-blurred toward the subject, then sharp.
ZOOM_ENTRY = (0.32, 0.12)      # strength on n = 0, 1 (0 from n = 2 on)


def zoom_entry(img, n, center=(0.5, 0.5), strengths=ZOOM_ENTRY, s0=1.10):
    """Radial zoom-blur entry. n=0 heaviest, n=1 light, n>=2 untouched. Scale 1.10->1.0."""
    if n >= len(strengths) or n < 0:
        return img
    s = 1.0 + (s0 - 1.0) * (1.0 - n / len(strengths))
    return zoom_blur(_scale_about(img, s, center), strengths[n], center)


def zoom_exit(img, k, frames=2, center=(0.5, 0.5)):
    """fUx4 47.33-47.37: the shot collapses into a radial zoom blur on its last 2 frames
    (k = 0..frames-1), scale 1.0 -> 1.25 with blur 0.15 -> 0.45."""
    u = (k + 1) / frames
    return zoom_blur(_scale_about(img, 1.0 + 0.25 * u, center), 0.15 + 0.30 * u, center)


def zoom_through(img, k, frames=3, center=(0.5, 0.5)):
    """G3GC 4.25-4.38: last 3 frames of a shot punch INTO the subject until it is texture
    (1.35x, 1.85x, 2.6x) with rising zoom blur. Transition-only extreme (<=3 frames)."""
    scales = (1.35, 1.85, 2.6)[-frames:]
    blur = (0.10, 0.22, 0.38)[-frames:]
    return zoom_blur(_scale_about(img, scales[k], center), blur[k], center)


# ============================================================================ 3. ECHO TRAIL
# G3GC 9.00-9.30, 9.60-9.70, 16.90-17.20, 17.40-17.60: the current frame plus 2-3 earlier
# frames, each a little bigger, layered at ~50/30/18% -> the action leaves ghost copies.
ECHO_SCALES = (1.035, 1.07, 1.105)
ECHO_ALPHA = (0.42, 0.26, 0.15)


def echo_trail(cur, history, center=(0.5, 0.5), scales=ECHO_SCALES, alphas=ECHO_ALPHA, lighten=True):
    """cur: this frame. history: [frame t-2, t-4, t-6] (newest first; pass what exists).
    The current frame stays dominant (>= ~0.55) and is sharpened after the mix."""
    acc = _f32(cur)
    for p, s, a in zip(history, scales, alphas):
        g = _f32(_scale_about(p, s, center))
        if lighten:   # ghosts read on dark areas, never mud the bright subject
            g = np.maximum(g, acc * 0.85)
        acc = acc * (1 - a) + g * a
    return _sharpen(_u8(acc), 0.4)


# ======================================================================= 4. IMPACT SHAKE+ROLL
# G3GC 3.96-4.17 (the drop): rms 184 px @1920 with roll, 4.42-4.63 139 px, 7.92 210 px,
# 13.85 219 px, 18.81 234 px, 22.27 360 px; flips 50-80% of frames -> violent alternating
# jolts for ~5 frames. We keep the vocabulary at ~half the amplitude and always cover the
# frame (no black wedges), because the plate has to stay readable.
SHAKE_BIG = dict(amp=110.0, rot=4.5, tau=2.2, frames=7)
SHAKE_MED = dict(amp=60.0, rot=2.0, tau=2.0, frames=5)


def _jolts(seed, n):
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.55, 1.0, size=(n, 3))
    sx = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * rng.choice([1, -1])
    sy = np.where(rng.random(n) < 0.6, -sx, sx)
    return v, sx, sy


def impact_shake(img, n, seed=0, amp=110.0, rot=4.5, tau=2.2, frames=7, center=(0.5, 0.5)):
    """Big alternating jolt on frames n = 0..frames-1 after an impact. Overscales by exactly
    the cover scale for that frame (<= ~1.14 at n=0, back to 1.0 by n=frames)."""
    if n < 0 or n >= frames:
        return img
    v, sx, sy = _jolts(seed, frames)
    d = math.exp(-n / tau)
    dx, dy = sx[n] * amp * d * v[n, 0], sy[n] * amp * 0.6 * d * v[n, 1]
    r = sx[n] * rot * d * v[n, 2]
    s = fx.cover_scale(rot=r, dx=dx, dy=dy, center=center)
    return _scale_about(img, s, center, rot=r, dx=dx, dy=dy)


# =========================================================================== 5. TILTED HOLD
# MoxZ7 68.5-71.0 (the Baryon fight run, ~0.3-0.5 s shots): every held shot sits on a rolled
# frame (2-3 deg, sign alternating shot to shot) that keeps rotating slowly through the hold.
# MoxZ7 lets black wedges show; we cover instead (overscale ~1.06-1.08, re-sharpened).
def tilted_hold(img, u, shot_idx=0, roll=(2.6, -0.8), center=(0.5, 0.5), drift_px=24.0):
    sgn = 1.0 if shot_idx % 2 == 0 else -1.0
    r = sgn * (roll[0] + (roll[1] - roll[0]) * fx.ease_in_out(u))
    dx = sgn * drift_px * (u - 0.5)
    s = fx.cover_scale(rot=r, dx=dx, center=center)
    return _sharpen(_scale_about(img, s, center, rot=r, dx=dx), 0.30)


# ============================================================================== 6. SOFT GLOW
# fUx4 45.2-46.5 (luma 190-220 while the Six-Paths plate stays legible), G3GC chorus (every
# warm frame carries a halo; blacks still crushed). Glow = highlights blurred wide and screened
# over an UNTOUCHED copy, so line art keeps its edge.
def soft_glow(img, strength=0.45, threshold=150.0, sigma=26.0, tint=(200, 230, 255)):
    g = _luma(img)
    m = np.clip((g - threshold) / (255 - threshold), 0, 1)[..., None]
    hi = _f32(img) * m
    small = cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    halo = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    halo = halo * (np.float32(tint) / 255.0) * strength
    return _screen(img, _u8(halo))


# ======================================================================= 7. DUOTONE WASHES
# fUx4 46.00-46.50: the whole frame mapped to a 2-colour yellow ramp (Baryon/KCM gold);
# 46.50-46.57 the ramp swings to blue in 2 frames; 46.60-47.43 the Susanoo shot plays as
# a blue ramp. A colour hit that unifies a run of shots, not a filter on everything.
DUO = {
    "gold": ((20, 70, 120), (150, 245, 255)),    # BGR dark, light
    "blue": ((90, 30, 8), (255, 225, 170)),
    "red": ((20, 20, 90), (190, 210, 255)),
    "navy": ((60, 22, 10), (235, 225, 215)),
}


def duotone(img, preset="gold", amount=0.85, contrast=1.15):
    dark, light = DUO[preset] if isinstance(preset, str) else preset
    g = np.clip((_luma(img) / 255.0 - 0.5) * contrast + 0.5, 0, 1)[..., None]
    tone = np.float32(dark) * (1 - g) + np.float32(light) * g
    return _u8(_f32(img) * (1 - amount) + tone * amount)


def duotone_swing(img, k, frames=2, a="gold", b="blue", amount=0.85):
    """fUx4 46.50-46.57: the wash flips preset over `frames` frames (k = 0..frames-1)."""
    u = (k + 1) / (frames + 1)
    return _u8(_f32(duotone(img, a, amount)) * (1 - u) + _f32(duotone(img, b, amount)) * u)


# ============================================================================ 8. LIGHT SWEEP
# fUx4 45.40-45.90: a soft white diagonal band (about 0.2 W wide, ~-20 deg) sweeps left->right
# across the glowing plate over ~12 frames (30 fps) -> 10 frames at 23.976.
def light_sweep(img, u, angle=-20.0, width=0.20, strength=0.85, tint=(235, 245, 255)):
    yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
    xx, yy = xx * 2 / W, yy * 2 / H
    a = math.radians(angle)
    proj = xx * math.cos(a) - (yy - 0.5) * math.sin(a) * (H / W)
    pos = -0.3 + 1.6 * u
    band = np.exp(-((proj - pos) / (width / 2.2)) ** 2) * strength
    band = cv2.resize(band, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    return _screen(img, _u8(band * np.float32(tint)))


# =============================================================================== 9. HALFTONE
# G3GC 14.03-14.70: a comic dot screen over the warm frame (Laplacian var jumps 1.4k -> 24-27k
# purely from the dots). Dots are multiplied in at ~45% so the drawing reads through.
def halftone(img, cell=9, amount=0.45, angle=22.0):
    g = _luma(img) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = math.radians(angle)
    xr = xx * math.cos(a) + yy * math.sin(a)
    yr = -xx * math.sin(a) + yy * math.cos(a)
    fx_ = (xr % cell) / cell - 0.5
    fy_ = (yr % cell) / cell - 0.5
    d = np.sqrt(fx_ ** 2 + fy_ ** 2)
    rad = 0.62 * np.sqrt(1 - g)                  # dark areas -> big dots
    dots = np.clip((rad - d) * cell * 1.5, 0, 1)
    mul = 1 - amount * dots
    return _u8(_f32(img) * mul[..., None])


# ============================================================================= 10. DARK INK
# G3GC 10.40-10.70 and 18.33-18.74 (12 frames on an eye close-up): everything below mid-grey
# drops to deep navy, skin keeps its colour, contours glow white. A "power-up" look for a
# 0.5 s eye shot (Sage / Kurama / Rinnegan eyes).
def dark_ink(img, thresh=110.0, navy=(48, 22, 12), edge_gain=1.0):
    g = _luma(img)
    keep = np.clip((g - thresh) / 40.0, 0, 1)[..., None]
    base = _f32(img) * keep + np.float32(navy) * (1 - keep)
    edges = _f32(fx.neon_edges(img, glow=(255, 235, 210))) * edge_gain
    return _screen(_u8(base), _u8(edges))


# ============================================================================ 11. SUBTITLES
# fUx4 0.0-8.3 and MoxZ7 11-66: the story half plays the ORIGINAL dialogue (ASR on fUx4 0-12 s:
# "イルカ先生のおかげだ" / "え?" / "あの日のこと覚えてる" / "ああ") with English subtitles.
# Measured on fUx4 5.5 s (fux_sub_5.5.png): small-caps serif, cap height ~35 px, line 1
# "ALL OF THIS IS THANKS TO YOU," spans x 405-1513 (1108 px, wide tracking), baselines y ~893 and
# ~962 (0.827 H / 0.891 H, inside the picture rows 81-999), soft white glow, the trailing "..."
# in gold. Words type in one per ~3 frames @30 (= 2 @23.976), fade out ~8 frames.
SUB_FONT = os.path.join(FONT_DIR, "MarcellusSC-Regular.ttf")
SPEAKER = {"naruto": (255, 255, 255), "iruka": (226, 236, 246), "accent": (74, 195, 232)}


@lru_cache(maxsize=4)
def _font(size):
    return ImageFont.truetype(SUB_FONT, size)


def subtitle(img, lines, n, n_end=None, word_every=2, fade_in=2, fade_out=8, size=50,
             tracking=11, y=(0.827, 0.891), color=SPEAKER["naruto"], line2_alpha=1.0, glow=0.55):
    """lines: ['ALL OF THIS IS THANKS TO YOU,', 'IRUKA-SENSEI...']; n = frames since the line's
    first word (its vocal onset); n_end = frame (same origin) where it starts fading out."""
    if n < 0:
        return img
    fade = 1.0 if n_end is None else float(np.clip(1 - (n - n_end) / fade_out, 0, 1))
    if fade <= 0:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    shadow = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    dr, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    font = _font(size)
    widx = 0
    for li, line in enumerate(lines):
        words = line.split(" ")
        widths = [font.getlength(w) + tracking * (len(w) - 1) for w in words]
        space = font.getlength(" ") + tracking
        total = sum(widths) + space * (len(words) - 1)
        x = (W - total) / 2
        yb = y[min(li, len(y) - 1)] * H
        for w, ww in zip(words, widths):
            a_in = np.clip((n - widx * word_every) / fade_in, 0, 1) if n >= widx * word_every else 0.0
            a = a_in * fade * (1.0 if li == 0 else line2_alpha)
            if a > 0:
                cx = x
                for ch in w:
                    rgb = (SPEAKER["accent"] if ch in ".!?" and w.endswith("..") else color)[::-1]
                    dr.text((cx, yb), ch, font=font, fill=(*rgb, int(240 * a)), anchor="ls")
                    ds.text((cx + 2, yb + 2), ch, font=font, fill=(0, 0, 0, int(170 * a)), anchor="ls")
                    cx += font.getlength(ch) + tracking
            x += ww + space
            widx += 1
    from PIL import ImageFilter
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    halo = layer.filter(ImageFilter.GaussianBlur(7))
    if glow > 0:
        hr = np.array(halo)
        hr[..., 3] = np.clip(hr[..., 3].astype(np.float32) * glow, 0, 255).astype(np.uint8)
        halo = Image.fromarray(hr)
    pil = pil.convert("RGBA")
    pil.alpha_composite(shadow)
    pil.alpha_composite(halo)
    pil.alpha_composite(layer)
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# ========================================================================= 12. DROP RECIPE
def drop_frame(img, n, seed=19, history=None, center=(0.5, 0.5)):
    """The draft's drop (19.737) punches once then holds a dead-static plate for 0.8 s
    (checked: d_drop.jpg, 19.95-20.53 identical framing). Ref drops (G3GC 3.96, fUx4 46.0) keep
    the plate alive. n = frames since 19.737:
      n 0      exposure pop + zoom entry blur 0.32 + big shake
      n 1-6    big shake decaying + soft glow 0.55 -> 0.25
      n 2-12   echo trail if history given (the debris leaves ghosts)
    """
    out = img
    if n <= 1:
        out = zoom_entry(out, n, center)
    out = impact_shake(out, n, seed=seed, center=center, **SHAKE_BIG)
    if history and 2 <= n <= 12:
        out = echo_trail(out, history, center)
    g = 0.55 - 0.30 * min(1.0, n / 12)
    out = soft_glow(out, strength=g)
    if n == 0:
        out = fx.exposure_pop(out, target=190.0)
    return out
