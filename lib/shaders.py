"""shaders.py -- shader-grade frame effects for the Jane! AMV.

Contract: frames are uint8 BGR (H, W, 3), usually 1080x1920. Every function is
pure (never mutates its input) and deterministic (randomness comes from an
explicit seed / time). Heavy grids, LUTs and noise banks are cached per size.

Implementation notes
- Blurs that are "directional sums of shifted/scaled copies" (zoom blur,
  motion blur) use the log2 doubling trick: n passes of avg(img, warp(img, d_k))
  with d_k doubling give 2^n evenly spaced taps for n warps. They run at half
  resolution and are recomposited with the sharp frame where blur is ~0.
- Geometric distortions (lens, shockwave, vhs jitter) use cv2.remap with
  fixed-point (CV_16SC2) maps where the map is cacheable.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

__all__ = [
    "camera", "zoom_blur", "motion_blur", "lens", "shockwave", "bloom",
    "rgb_split", "chroma_radial", "light_leak", "film_burn", "grade", "glitch",
    "pixel_sort", "vhs", "zoom_through", "whip", "flash", "spin", "luma_wipe",
    "vignette", "grain", "LOOKS",
]

BORDER = cv2.BORDER_REFLECT101


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _u8(x: np.ndarray) -> np.ndarray:
    np.clip(x, 0, 255, out=x)
    return x.astype(np.uint8)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ss(e0, e1, x):  # scalar smoothstep
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@lru_cache(maxsize=8)
def _grid(h: int, w: int):
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return gx, gy


@lru_cache(maxsize=16)
def _polar(h: int, w: int, cx: float, cy: float):
    """radius normalised to half-diagonal, plus unit direction vectors."""
    gx, gy = _grid(h, w)
    vx = gx - cx * w
    vy = gy - cy * h
    hd = 0.5 * math.hypot(w, h)
    r = np.sqrt(vx * vx + vy * vy) / hd
    inv = 1.0 / np.maximum(r * hd, 1e-3)
    return r.astype(np.float32), (vx * inv).astype(np.float32), (vy * inv).astype(np.float32)


def _scale_about(img, s, cx, cy, dsize=None, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = np.float32([[s, 0, (1 - s) * cx * w], [0, s, (1 - s) * cy * h]])
    return cv2.warpAffine(img, M, dsize or (w, h), flags=interp, borderMode=BORDER)


def _shift(img, dx, dy):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=BORDER)


def _half(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def _up(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _screen(base, add):
    """screen blend of two uint8 images."""
    inv = cv2.multiply(255 - base, 255 - add, scale=1 / 255.0)
    return 255 - inv


# --------------------------------------------------------------------------
# camera with real shutter motion blur
# --------------------------------------------------------------------------
def _cam_matrix(w, h, zoom, rot, dx, dy, scale=1.0):
    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), rot, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    if scale != 1.0:
        M = M * scale
    return M


def camera(frame, zoom=1.0, rot_deg=0.0, dx=0.0, dy=0.0, prev=None, shutter=0.5,
           max_samples=16):
    """Scale/rotate about centre then translate (dx, dy in px).

    prev: (zoom, rot_deg, dx, dy) of the previous frame. When given, the
    camera path between prev and current is integrated over the last
    `shutter` fraction of the frame interval (shutter=0.5 is a 180deg
    shutter). Sample count adapts to on-screen travel; big moves accumulate
    at half resolution (they are smeared anyway).
    """
    h, w = frame.shape[:2]
    cur = (zoom, rot_deg, dx, dy)
    Mc = _cam_matrix(w, h, *cur)
    if prev is None or shutter <= 0:
        return cv2.warpAffine(frame, Mc, (w, h), flags=cv2.INTER_LINEAR, borderMode=BORDER)

    z0, r0, x0, y0 = prev
    s0 = 1.0 - shutter

    def lerp_state(s):
        z = math.exp(math.log(max(z0, 1e-3)) * (1 - s) + math.log(max(zoom, 1e-3)) * s)
        return z, r0 + (rot_deg - r0) * s, x0 + (dx - x0) * s, y0 + (dy - y0) * s

    Ms = _cam_matrix(w, h, *lerp_state(s0))
    corners = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1], [w / 2, h / 2, 1]]).T
    travel = float(np.max(np.linalg.norm(Mc @ corners - Ms @ corners, axis=0)))
    n = int(min(max_samples, max(1, math.ceil(travel / 3.0))))
    if n <= 1:
        return cv2.warpAffine(frame, Mc, (w, h), flags=cv2.INTER_LINEAR, borderMode=BORDER)

    half = n > 5
    sc = 0.5 if half else 1.0
    ow, oh = (w // 2, h // 2) if half else (w, h)
    src = _half(frame) if half else frame
    acc = np.zeros((oh, ow, 3), np.float32)
    for s in np.linspace(s0, 1.0, n):
        M = _cam_matrix(w, h, *lerp_state(float(s)))
        if half:
            # source is half-res too: conjugate by 0.5 scale
            M = M.copy()
            M[:, 2] *= 0.5
        cv2.accumulate(cv2.warpAffine(src, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=BORDER), acc)
    acc *= 1.0 / n
    out = acc.astype(np.uint8)
    return _up(out, w, h) if half else out


# --------------------------------------------------------------------------
# blurs
# --------------------------------------------------------------------------
def zoom_blur(frame, strength=0.25, center=(0.5, 0.5), passes=5, boost=0.35):
    """Radial (zoom) blur. strength = fractional scale range of the smear
    (0.25 -> taps spread from 1.0x to 1.25x). Centre stays razor sharp,
    streak highlights get a bloom-ish lift so it reads punchy, not muddy."""
    if strength <= 1e-4:
        return frame.copy()
    h, w = frame.shape[:2]
    cx, cy = center
    small = _half(frame).astype(np.float32)
    total = 1.0 + strength
    for k in range(passes):
        f = total ** (2 ** k / 2 ** passes)
        small = cv2.addWeighted(small, 0.5, _scale_about(small, f, cx, cy), 0.5, 0)
    blur = _up(small, w, h)
    # sharp core: blur amount ~ r * strength, fade in from the centre
    r, _, _ = _polar(h, w, round(cx, 3), round(cy, 3))
    m = _mask_cache_zoom(h, w, round(cx, 3), round(cy, 3))
    base = frame.astype(np.float32)
    out = base + (blur - base) * m[..., None]
    if boost > 0:  # lift the streaks: highlights of the blur screen'd back in
        hi = cv2.max(blur - 150.0, 0) * (boost * 1.6)
        out += hi
    return _u8(out)


@lru_cache(maxsize=8)
def _mask_cache_zoom(h, w, cx, cy):
    r, _, _ = _polar(h, w, cx, cy)
    return _smoothstep(0.02, 0.35, r).astype(np.float32)


def motion_blur(frame, angle_deg=0.0, length=60.0, passes=5):
    """Directional shutter blur of `length` px along angle (0 = horizontal)."""
    if length < 1.0:
        return frame.copy()
    h, w = frame.shape[:2]
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    small = _half(frame).astype(np.float32)
    L = length * 0.5  # half-res px
    small = _shift(small, -ux * L * 0.5, -uy * L * 0.5)  # centre the kernel
    for k in range(passes):
        d = L * (2 ** k) / (2 ** passes)
        small = cv2.addWeighted(small, 0.5, _shift(small, ux * d, uy * d), 0.5, 0)
    return _u8(_up(small, w, h))


# --------------------------------------------------------------------------
# lens distortion + chromatic aberration
# --------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _lens_maps(h, w, kq, caq, cx, cy):
    k = kq / 100.0
    ca = caq / 1000.0
    gx, gy = _grid(h, w)
    ccx, ccy = cx * w, cy * h
    hd = 0.5 * math.hypot(w, h)
    vx = (gx - ccx) / hd
    vy = (gy - ccy) / hd
    r2 = vx * vx + vy * vy
    norm = (1.0 + k) if k > 0 else 1.0
    maps = []
    for ch_s in (-1.0, 0.0, 1.0):  # B, G, R
        f = (1.0 + k * r2) / norm * (1.0 + ch_s * ca * (0.25 + r2))
        mx = (ccx + vx * hd * f).astype(np.float32)
        my = (ccy + vy * hd * f).astype(np.float32)
        maps.append(cv2.convertMaps(mx, my, cv2.CV_16SC2))
    return maps


def lens(frame, k=0.15, center=(0.5, 0.5), ca=0.006):
    """Barrel (k>0, bulge) / pincushion (k<0) distortion with radial chromatic
    aberration (R pushed out, B pulled in). Maps cached per 0.01 k-bucket."""
    h, w = frame.shape[:2]
    maps = _lens_maps(h, w, int(round(k * 100)), int(round(ca * 1000)),
                      round(center[0], 2), round(center[1], 2))
    chans = cv2.split(frame)
    out = [cv2.remap(c, m[0], m[1], cv2.INTER_LINEAR, borderMode=BORDER) for c, m in zip(chans, maps)]
    return cv2.merge(out)


# --------------------------------------------------------------------------
# shockwave
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _polar_half(h, w, cx, cy):
    hh, hw = h // 2, w // 2
    gx, gy = np.meshgrid(np.arange(hw, dtype=np.float32) * 2, np.arange(hh, dtype=np.float32) * 2)
    return np.sqrt((gx - cx * w) ** 2 + (gy - cy * h) ** 2).astype(np.float32)


def shockwave(frame, t, center=(0.5, 0.5), speed=1400.0, width=60.0, amp=40.0,
              chroma=True, glow=0.5):
    """Expanding refraction ring. t seconds since impact, speed px/s,
    width px (ring thickness), amp px (peak displacement). Ring fades as it
    grows. Adds per-channel refraction split and a faint rim light."""
    h, w = frame.shape[:2]
    R = speed * t
    maxR = math.hypot(w, h) * 0.75
    fade = max(0.0, 1.0 - R / maxR) ** 1.5
    if fade <= 0.0 or amp * fade < 0.3:
        return frame.copy()
    cx, cy = round(center[0], 3), round(center[1], 3)
    rh = _polar_half(h, w, cx, cy)
    x = (rh - R) / width
    prof = (x * np.exp(-x * x)).astype(np.float32)  # derivative-of-gaussian lens profile
    disp_h = prof * (amp * fade * 2.33)
    disp = _up(disp_h, w, h)
    _, ux, uy = _polar(h, w, cx, cy)
    gx, gy = _grid(h, w)
    dx = ux * disp
    dy = uy * disp
    mx = gx - dx
    my = gy - dy
    if chroma:
        chans = cv2.split(frame)
        out = []
        for c, s in zip(chans, (0.6, 1.0, 1.4)):
            out.append(cv2.remap(c, gx - dx * s, gy - dy * s, cv2.INTER_LINEAR, borderMode=BORDER))
        res = cv2.merge(out)
    else:
        res = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=BORDER)
    if glow > 0:
        ring = _up(np.exp(-x * x * 2.5).astype(np.float32), w, h) * (glow * 150.0 * fade)
        res = cv2.add(res, cv2.merge([ring, ring, ring]).astype(np.uint8))
    return res


# --------------------------------------------------------------------------
# bloom
# --------------------------------------------------------------------------
def bloom(frame, threshold=170, strength=1.0, radius=1.0, tint=None):
    """Highlight glow: soft-knee threshold, 5-level downsample/blur/upsample
    chain (wide + tight lobes), additive. radius scales the lobe sizes."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    knee = 30.0
    soft = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
    soft = soft * soft
    hard = np.maximum(lum - threshold, 0) / np.maximum(lum, 1)
    wgt = np.maximum(soft * 0.35, hard * 1.6)
    bright = small * wgt[..., None]
    if tint is not None:
        bright = bright * (np.float32(tint) / 255.0)
    acc = np.zeros_like(bright)
    lvl = bright
    weights = (1.0, 0.9, 0.8, 0.7, 0.6)
    sigma0 = 2.0 * radius
    for i, wt in enumerate(weights):
        bl = cv2.GaussianBlur(lvl, (0, 0), sigma0)
        acc += _up(bl, acc.shape[1], acc.shape[0]) * wt
        if i < len(weights) - 1:
            lvl = cv2.pyrDown(lvl)
    glow = _up(acc, w, h) * (strength * 0.55)
    return _u8(frame.astype(np.float32) + glow)


# --------------------------------------------------------------------------
# chromatic splits
# --------------------------------------------------------------------------
def rgb_split(frame, amount=12.0, angle=0.0):
    """Linear channel split: R pushed +amount px along angle, B -amount."""
    a = math.radians(angle)
    dx, dy = math.cos(a) * amount, math.sin(a) * amount
    b, g, r = cv2.split(frame)
    return cv2.merge([_shift(b, -dx, -dy), g, _shift(r, dx, dy)])


def chroma_radial(frame, amount=0.012, center=(0.5, 0.5)):
    """Radial CA: R scaled up, B scaled down about centre (edges fringe)."""
    cx, cy = center
    b, g, r = cv2.split(frame)
    return cv2.merge([_scale_about(b, 1.0 - amount, cx, cy), g, _scale_about(r, 1.0 + amount, cx, cy)])


# --------------------------------------------------------------------------
# procedural noise (low-res, upsampled)
# --------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _noise_bank(seed: int, size: int = 64):
    rng = np.random.default_rng(seed)
    return rng.random((size, size)).astype(np.float32)


def _fbm(seed, w, h, t, octaves=4, base=4.0, drift=(0.15, 0.07)):
    """Smooth value-noise fBm at (h, w), animated by scrolling each octave."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        tex = _noise_bank(seed * 31 + o)
        n = tex.shape[0]
        freq = base * (2 ** o)
        # sample a freq x (freq*aspect) window that scrolls with t
        ox = (t * drift[0] * (1 + o * 0.6) * n) % n
        oy = (t * drift[1] * (1 + o * 0.4) * n) % n
        fx = freq * (w / h) / w
        fy = freq / h
        M = np.float32([[fx, 0, ox], [0, fy, oy]])
        samp = cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_WRAP)
        out += samp * amp
        tot += amp
        amp *= 0.5
    return out / tot


def light_leak(frame, t, seed=0, color=None, strength=0.85):
    """Animated organic light leak (drifting coloured blobs entering from an
    edge), screen-blended. color: BGR tuple or None for warm amber/magenta."""
    h, w = frame.shape[:2]
    rng = np.random.default_rng(seed)
    lw, lh = 192, 108
    gx, gy = np.meshgrid(np.linspace(0, 1, lw, dtype=np.float32), np.linspace(0, 1, lh, dtype=np.float32))
    side = rng.integers(0, 4)
    field = np.zeros((lh, lw, 3), np.float32)
    pal = [np.float32(color)] if color is not None else [
        np.float32((40, 120, 255)), np.float32((60, 60, 255)), np.float32((150, 60, 255)),
        np.float32((90, 200, 255))]
    for i in range(5):
        ph = rng.random() * 6.28
        sp = 0.3 + rng.random() * 0.5
        along = (rng.random() + 0.25 * math.sin(t * sp + ph)) % 1.0
        depth = 0.05 + 0.25 * rng.random() + 0.08 * math.sin(t * sp * 1.3 + ph * 2)
        if side == 0:
            px, py = depth, along
        elif side == 1:
            px, py = 1 - depth, along
        elif side == 2:
            px, py = along, depth
        else:
            px, py = along, 1 - depth
        sx = 0.12 + 0.25 * rng.random()
        sy = sx * (1.0 + rng.random())
        g = np.exp(-(((gx - px) / sx) ** 2 + ((gy - py) / sy * 0.5625) ** 2))
        c = pal[i % len(pal)]
        field += g[..., None] * c
    # organic breakup
    nz = _fbm(seed + 7, lw, lh, t * 0.4, octaves=3, base=3.0)
    field *= (0.55 + 0.9 * nz)[..., None]
    breath = 0.75 + 0.25 * math.sin(t * 1.7 + seed)
    field *= strength * breath
    leak = cv2.resize(np.clip(field, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    return _screen(frame, leak)


@lru_cache(maxsize=4)
def _fire_lut():
    stops = [(0.0, (0, 0, 0)), (0.25, (10, 10, 120)), (0.45, (0, 60, 230)),
             (0.65, (40, 170, 255)), (0.82, (150, 235, 255)), (1.0, (255, 255, 255))]
    lut = np.zeros((256, 3), np.float32)
    xs = np.linspace(0, 1, 256)
    for c in range(3):
        lut[:, c] = np.interp(xs, [s[0] for s in stops], [s[1][c] for s in stops])
    return lut.astype(np.uint8)


def film_burn(frame, t, seed=0):
    """Film burn transition overlay. t in [0,1]: burn blooms in from an edge,
    peaks (near white-out) at 0.5, recedes. Additive fire-ramp colour."""
    h, w = frame.shape[:2]
    lw, lh = 240, 135
    rng = np.random.default_rng(seed)
    nz = _fbm(seed, lw, lh, t * 3.0, octaves=5, base=2.5, drift=(0.25, 0.12))
    gx, gy = np.meshgrid(np.linspace(0, 1, lw, dtype=np.float32), np.linspace(0, 1, lh, dtype=np.float32))
    ex, ey = rng.random() < 0.5, rng.random()
    edge = (1 - gx if ex else gx) * 0.8 + np.abs(gy - ey) * 0.35  # distance from the burning edge
    inten = math.sin(math.pi * min(max(t, 0.0), 1.0))
    front = inten * 1.05 - 0.12
    v = np.clip((front - edge + (nz - 0.5) * 1.1) * 2.2, 0, 1)
    v = v ** 1.6
    idx = (v * 255).astype(np.uint8)
    col = _fire_lut()[idx]
    col = cv2.resize(col, (w, h), interpolation=cv2.INTER_CUBIC)
    # darken the emulsion slightly ahead of the burn so the fire reads
    dark = cv2.resize((255 - (np.clip(v * 3, 0, 1) * 90 * inten)).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_LINEAR)
    out = cv2.multiply(frame, cv2.merge([dark, dark, dark]), scale=1 / 255.0)
    out = cv2.add(out, col)
    if inten > 0.85:  # brief over-exposure at the very peak
        k = (inten - 0.85) / 0.15
        out = cv2.addWeighted(out, 1.0, np.full_like(out, 255), 0.4 * k, 0)
    return out


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------
def _curve(points):
    xs, ys = zip(*points)
    return np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.uint8)


def _scurve(c=0.6):
    x = np.linspace(0, 1, 256)
    y = x + c * (x - 0.5) * (1 - np.abs(2 * x - 1)) * 0.9 * np.sign(1)  # cheap s-curve
    y = 0.5 + (y - 0.5) * (1 + c * 0.35)
    return np.clip(y * 255, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _luts():
    L = {}
    # teal/orange: R up in highs, down in lows; B up in lows, down in highs
    L["teal_orange"] = (
        _curve([(0, 6), (40, 40), (90, 92), (150, 128), (210, 170), (255, 215)]),   # B
        _curve([(0, 2), (40, 26), (90, 80), (150, 148), (210, 205), (255, 245)]),   # G
        _curve([(0, 0), (40, 12), (90, 64), (150, 166), (210, 236), (255, 255)]),   # R
    )
    L["cold_steel"] = (
        _curve([(0, 4), (50, 52), (120, 140), (200, 225), (255, 255)]),
        _curve([(0, 0), (50, 32), (120, 118), (200, 205), (255, 245)]),
        _curve([(0, 0), (50, 18), (120, 96), (200, 186), (255, 228)]),
    )
    L["bleach"] = _curve([(0, 0), (45, 10), (110, 96), (170, 190), (225, 248), (255, 255)])
    L["contrast"] = _curve([(0, 0), (40, 18), (128, 128), (215, 240), (255, 255)])
    L["noir"] = _curve([(0, 0), (50, 14), (128, 120), (200, 230), (255, 255)])
    return L


def _saturate(img_f, s):
    gray = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)[..., None]
    return gray + (img_f - gray) * s


@lru_cache(maxsize=4)
def _halftone_screen(h, w, cell=7, angle=45.0):
    gx, gy = _grid(h, w)
    a = math.radians(angle)
    u = (gx * math.cos(a) + gy * math.sin(a)) * (2 * math.pi / cell)
    v = (-gx * math.sin(a) + gy * math.cos(a)) * (2 * math.pi / cell)
    s = (np.cos(u) + np.cos(v)) * 0.25 + 0.5  # 0..1 dot field
    return (s * 255).astype(np.uint8)


LOOKS = ("teal_orange", "cold_steel", "red_noir", "manga_bw", "bleach")


def grade(frame, look="teal_orange", amount=1.0):
    """Strong film looks. amount mixes with the original (0..1)."""
    h, w = frame.shape[:2]
    luts = _luts()
    if look in ("teal_orange", "cold_steel"):
        b, g, r = cv2.split(frame)
        lb, lg, lr = luts[look]
        out = cv2.merge([cv2.LUT(b, lb), cv2.LUT(g, lg), cv2.LUT(r, lr)])
        sat = 1.25 if look == "teal_orange" else 0.55
        f = _saturate(out.astype(np.float32), sat)
        out = _u8(f)
    elif look == "red_noir":
        hsv = cv2.cvtColor(cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        hch, s, v = cv2.split(hsv)
        hue_d = np.minimum(hch, 180 - hch).astype(np.float32)  # distance from red (0/180)
        m = (1 - _smoothstep(6, 14, hue_d)) * _smoothstep(70, 140, s.astype(np.float32)) \
            * _smoothstep(40, 90, v.astype(np.float32))
        m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
        m = _up(m, w, h)[..., None]
        gray = cv2.LUT(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), luts["noir"])
        g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        red = frame.astype(np.float32)
        red = _saturate(red, 1.5) * np.float32([0.55, 0.55, 1.15])
        out = _u8(g3 + (red - g3) * m)
    elif look == "manga_bw":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.LUT(gray, luts["contrast"])
        # screentone: mids become dots, deep shadows solid, highs paper white
        scr = _halftone_screen(h, w)
        tone = np.where(gray > scr, 255, 0).astype(np.uint8)
        tone = np.where(gray < 38, 0, tone)
        tone = np.where(gray > 200, 255, tone).astype(np.uint8)
        # ink lines from smoothed image
        sm = cv2.bilateralFilter(cv2.resize(gray, (w // 2, h // 2)), 7, 40, 7)
        edges = cv2.adaptiveThreshold(sm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)
        edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
        ink = cv2.min(tone, edges)
        # paper tint
        paper = np.float32([228, 240, 246])
        out = _u8(ink.astype(np.float32)[..., None] / 255.0 * paper + 8)
    elif look == "bleach":
        # bleach bypass: silver retained -> desaturate ~65%, hard contrast,
        # metallic highlights. All uint8 (LUT + addWeighted) for speed.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mix = cv2.addWeighted(frame, 0.38, g3, 0.62, 0)
        lb = luts["bleach"]
        b, g, r = cv2.split(mix)
        out = cv2.merge([cv2.LUT(b, lb), cv2.LUT(g, lb), cv2.LUT(r, lb)])
    else:
        raise ValueError(f"unknown look {look!r}; choose from {LOOKS}")
    if amount >= 0.999:
        return out
    return cv2.addWeighted(frame, 1 - amount, out, amount, 0)


# --------------------------------------------------------------------------
# glitch / datamosh
# --------------------------------------------------------------------------
def glitch(frame, t, seed=0, amount=0.7, fps=30):
    """Slice displacement, datamosh block smears and per-band RGB offsets.
    Deterministic in (seed, frame index)."""
    h, w = frame.shape[:2]
    fi = int(round(t * fps))
    rng = np.random.default_rng(seed * 100003 + fi)
    out = frame.copy()
    a = amount
    # 1. horizontal slice shifts
    for _ in range(int(4 + 14 * a)):
        y = rng.integers(0, h - 8)
        hh = int(rng.integers(3, int(10 + 90 * a)))
        sx = int(rng.normal(0, 140 * a))
        out[y:y + hh] = np.roll(out[y:y + hh], sx, axis=1)
    # 2. datamosh block smears (repeat an edge row/col across the block)
    bs = 32
    for _ in range(int(3 + 10 * a)):
        bw = bs * int(rng.integers(2, 9))
        bh = bs * int(rng.integers(1, 5))
        x = int(rng.integers(0, max(1, w - bw))) // bs * bs
        y = int(rng.integers(0, max(1, h - bh))) // bs * bs
        mode = rng.integers(0, 3)
        if mode == 0:
            out[y:y + bh, x:x + bw] = out[y:y + 1, x:x + bw]
        elif mode == 1:
            out[y:y + bh, x:x + bw] = out[y:y + bh, x:x + 1]
        else:
            sx = int(rng.integers(-3, 4)) * bs
            sy = int(rng.integers(-2, 3)) * bs
            ys, xs = np.clip(y + sy, 0, h - bh), np.clip(x + sx, 0, w - bw)
            out[y:y + bh, x:x + bw] = frame[ys:ys + bh, xs:xs + bw]
    # 3. RGB offset bands
    for _ in range(int(2 + 6 * a)):
        y = int(rng.integers(0, h - 20))
        hh = int(rng.integers(20, int(40 + 200 * a)))
        ch = int(rng.integers(0, 3))
        out[y:y + hh, :, ch] = np.roll(out[y:y + hh, :, ch], int(rng.integers(-60, 60) * a + 1), axis=1)
    # 4. a couple of posterised / channel-swapped bands
    for _ in range(int(1 + 3 * a)):
        y = int(rng.integers(0, h - 10))
        hh = int(rng.integers(6, 60))
        band = out[y:y + hh]
        if rng.random() < 0.5:
            out[y:y + hh] = (band // 64) * 64 + 32
        else:
            out[y:y + hh] = band[:, :, ::-1]
    # global rgb split jitter
    return rgb_split(out, 6 + 14 * a * rng.random(), 0)


# --------------------------------------------------------------------------
# pixel sort
# --------------------------------------------------------------------------
def pixel_sort(frame, threshold=140, direction="h", upper=255, reverse=False):
    """Sort pixels by luma inside every contiguous run whose luma is in
    [threshold, upper], per row ('h') or column ('v'). Fully vectorised:
    one argsort per image of a composite key (run_id << 8 | luma)."""
    if direction == "h":
        img = frame[::2]  # fast path: sort every other row, duplicate (streaks are 1-D anyway)
    else:
        img = np.ascontiguousarray(frame[:, ::2].transpose(1, 0, 2))
    h, w = img.shape[:2]
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (lum >= threshold) & (lum <= upper)
    # each masked run shares one id; each unmasked pixel its own id
    start = np.ones((h, w), np.bool_)
    start[:, 1:] = ~(mask[:, 1:] & mask[:, :-1])
    rid = np.cumsum(start, axis=1, dtype=np.int32)
    keyl = lum.astype(np.int32)
    if reverse:
        keyl = 255 - keyl
    key = (rid << 8) | np.where(mask, keyl, 0)
    order = np.argsort(key, axis=1, kind="stable")
    out = np.take_along_axis(img, order[..., None], axis=1)
    if direction == "h":
        return np.repeat(out, 2, axis=0)[:frame.shape[0]]
    return cv2.resize(np.ascontiguousarray(out.transpose(1, 0, 2)), (frame.shape[1], frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# VHS
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _scanlines(h, w):
    rows = np.where((np.arange(h) % 3) == 0, 205, 255).astype(np.uint8)
    return np.repeat(np.repeat(rows[:, None], w, axis=1)[..., None], 3, axis=2)


def vhs(frame, t, seed=0, fps=30, strength=1.0):
    """VHS: horizontal luma softening, smeared and offset chroma, tracking
    wobble + a rolling tracking-noise band, scanlines, tape noise."""
    h, w = frame.shape[:2]
    fi = int(round(t * fps))
    rng = np.random.default_rng(seed * 7919 + fi)
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.blur(y, (5, 1))
    cs = (w // 4, h // 2)
    cr = cv2.resize(cv2.resize(cr, cs, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    cb = cv2.resize(cv2.resize(cb, cs, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    cr = cv2.blur(_shift(cr, 10 * strength, 0), (25, 1))
    cb = cv2.blur(_shift(cb, -4 * strength, 0), (25, 1))
    # saturate a touch (VHS reds bloom)
    cr = cv2.addWeighted(cr, 1.25, cr, 0, -0.25 * 128)
    img = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    # tracking wobble: per-row shift
    rows = np.arange(h, dtype=np.float32)
    shift = 1.5 * np.sin(rows * 0.021 + t * 7.0) * strength
    band_y = (t * 0.37 * h + seed * 211) % (h * 1.4) - 0.2 * h
    band = np.exp(-((rows - band_y) / 26.0) ** 2)
    shift += band * rng.normal(0, 18, h).astype(np.float32) * strength
    if rng.random() < 0.15:  # occasional head-switch tear at the bottom
        shift[-int(h * 0.04):] += 40 * strength
    gx, gy = _grid(h, w)
    img = cv2.remap(img, gx - shift[:, None].astype(np.float32), gy, cv2.INTER_LINEAR, borderMode=BORDER)
    # noise streaks in the tracking band
    nz = (rng.random((h, w // 8)) ** 6 * 255 * band[:, None]).astype(np.uint8)
    nz = cv2.resize(nz, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.add(img, cv2.merge([nz, nz, nz]))
    img = cv2.multiply(img, _scanlines(h, w), scale=1 / 255.0)
    # lifted blacks, slightly warm, soft highs
    img = cv2.convertScaleAbs(img, alpha=0.9, beta=14)
    return img


# --------------------------------------------------------------------------
# finishing: vignette + grain
# --------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _vig_mask(h, w, sq, rq):
    s = sq / 100.0
    rnd = rq / 100.0
    gx, gy = _grid(h, w)
    nx = (gx / w - 0.5) * 2
    ny = (gy / h - 0.5) * 2
    nx = nx * (1 - rnd) + nx * (w / h) * 0.5625 * rnd
    d = np.sqrt(nx * nx + ny * ny) / math.sqrt(2)
    m = 1.0 - s * _smoothstep(0.35, 1.05, d) ** 1.2
    m8 = (m * 255).astype(np.uint8)
    return cv2.merge([m8, m8, m8])


def vignette(frame, strength=0.5, roundness=0.0):
    h, w = frame.shape[:2]
    return cv2.multiply(frame, _vig_mask(h, w, int(round(strength * 100)), int(round(roundness * 100))),
                        scale=1 / 255.0)


@lru_cache(maxsize=4)
def _grain_bank(h, w, n=6, seed=1234):
    rng = np.random.default_rng(seed)
    bank = []
    for _ in range(n):
        g = rng.standard_normal((h, w)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.75)
        g /= g.std() + 1e-6
        bank.append(np.clip(g * 40, -127, 127).astype(np.int8))
    return bank


@lru_cache(maxsize=1)
def _grain_wlut():
    x = np.arange(256) / 255.0
    return np.clip(0.35 + 0.65 * (1 - (2 * x - 1) ** 2), 0, 1).astype(np.float32)


def grain(frame, t, amount=0.5, fps=30):
    """Animated luma-only film grain, strongest in midtones. amount ~0.3-1."""
    h, w = frame.shape[:2]
    bank = _grain_bank(h, w)
    fi = int(round(t * fps))
    g = bank[fi % len(bank)]
    if fi // len(bank) % 2:  # break up the 6-frame cycle
        g = g[::-1, ::-1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wgt = _grain_wlut()[gray]
    d = (g.astype(np.float32) * wgt * (amount * 0.6)).astype(np.int16)
    out = frame.astype(np.int16)
    out += d[..., None]
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# --------------------------------------------------------------------------
# transitions (t in [0, 1])
# --------------------------------------------------------------------------
def _ease_in(u, p=3.0):
    return u ** p


def zoom_through(a, b, t, frames=12, peak_zoom=2.6, blur=0.55):
    """Punch into A with an accelerating zoom + zoom blur, cut at t=0.5,
    B arrives over-zoomed and decelerates to rest. Exposure kick at the cut."""
    if t < 0.5:
        u = t / 0.5
        e = _ease_in(u)
        z = 1 + (peak_zoom - 1) * e
        img = camera(a, z)
        img = zoom_blur(img, blur * e)
        kick = 1 + 0.6 * u ** 6
    else:
        u = (t - 0.5) / 0.5
        e = (1 - u) ** 3
        z = 1 + (peak_zoom - 1) * 0.6 * e
        img = camera(b, z)
        img = zoom_blur(img, blur * e)
        kick = 1 + 0.6 * e ** 2
    if kick > 1.01:
        img = cv2.convertScaleAbs(img, alpha=kick, beta=0)
    return img


def whip(a, b, t, direction="left", frames=10, max_blur=420):
    """Whip pan: A slides out, B in, with velocity-driven motion blur.
    direction: left/right/up/down = where the content travels."""
    h, w = a.shape[:2]

    def pos(u):  # smootherstep, fast in the middle
        u = min(max(u, 0.0), 1.0)
        return u * u * u * (u * (u * 6 - 15) + 10)

    p = pos(t)
    vel = (pos(t + 0.5 / frames) - pos(t - 0.5 / frames))  # fraction per frame
    horiz = direction in ("left", "right")
    span = w if horiz else h
    off = int(round(p * span))
    sign = -1 if direction in ("left", "up") else 1
    canvas = np.empty_like(a)
    if horiz:
        if sign < 0:  # content moves left: A exits left, B enters from right
            canvas[:, :w - off] = a[:, off:]
            canvas[:, w - off:] = b[:, :off]
        else:
            canvas[:, off:] = a[:, :w - off]
            canvas[:, :off] = b[:, w - off:]
    else:
        if sign < 0:
            canvas[:h - off] = a[off:]
            canvas[h - off:] = b[:off]
        else:
            canvas[off:] = a[:h - off]
            canvas[:off] = b[h - off:]
    L = min(max_blur, vel * span * 0.9)
    ang = 0.0 if horiz else 90.0
    img = motion_blur(canvas, ang, L) if L > 2 else canvas
    return img


def flash(a, t, color=(255, 255, 255), power=1.0):
    """Exposure flash peaking at t=0, decaying out (with bloom)."""
    i = (1 - min(max(t, 0.0), 1.0)) ** 3.0 * power
    if i < 0.01:
        return a.copy()
    img = cv2.convertScaleAbs(a, alpha=1 + 0.9 * i, beta=0)
    img = bloom(img, threshold=140, strength=1.5 * i, radius=1.5)
    col = np.empty_like(a)
    col[:] = color
    return cv2.addWeighted(img, 1.0, col, 0.75 * i ** 1.5, 0)


def spin(a, b, t, frames=12, turns=0.5, zoom=1.25):
    """Spin transition: A accelerates into a rotation with rotational shutter
    blur, cut at t=0.5, B decelerates out of the spin."""
    deg = 360.0 * turns
    dt = 1.0 / frames

    def state(tt):
        if tt < 0.5:
            u = max(tt, 0.0) / 0.5
            return 1 + (zoom - 1) * u ** 2, deg * u ** 3
        u = (min(tt, 1.0) - 0.5) / 0.5
        return 1 + (zoom - 1) * (1 - u) ** 2, -deg * (1 - u) ** 3

    src = a if t < 0.5 else b
    z, r = state(t)
    tp = t - dt
    if (tp < 0.5) != (t < 0.5):
        tp = 0.5 if t >= 0.5 else t
    zp, rp = state(tp)
    return camera(src, z, r, prev=(zp, rp, 0, 0), shutter=0.6)


@lru_cache(maxsize=16)
def _wipe_pattern(h, w, pattern):
    gx, gy = _grid(h, w)
    nx, ny = gx / w, gy / h
    if pattern == "linear":
        p = nx * 0.85 + ny * 0.15
    elif pattern == "radial":
        p = np.sqrt((nx - 0.5) ** 2 + ((ny - 0.5) * h / w) ** 2) / 0.58
    elif pattern == "diamond":
        p = (np.abs(nx - 0.5) + np.abs(ny - 0.5) * h / w) / 0.78
    elif pattern == "clock":
        p = (np.arctan2(ny - 0.5, nx - 0.5) + math.pi) / (2 * math.pi)
    elif pattern == "blinds":
        p = ((ny * 9) % 1.0) * 0.6 + nx * 0.4
    elif pattern == "noise":
        p = _fbm(99, w // 4, h // 4, 0.0, octaves=4, base=3.0)
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_CUBIC)
        p = (p - p.min()) / (p.max() - p.min())
    else:
        raise ValueError(pattern)
    p = (p - p.min()) / (p.max() - p.min() + 1e-6)
    return p.astype(np.float32)


def luma_wipe(a, b, t, pattern="noise", softness=0.06, edge_color=(120, 210, 255), edge=0.8):
    """Gradient (luma-matte) wipe from A to B with a glowing burn edge.
    pattern: linear | radial | diamond | clock | blinds | noise | 'luma'
    ('luma' uses B's own brightness as the matte: highlights arrive first)."""
    h, w = a.shape[:2]
    hw, hh = w // 2, h // 2  # matte computed at half res (smooth anyway)
    if pattern == "luma":
        small = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (hw, hh), interpolation=cv2.INTER_AREA)
        p = 1.0 - cv2.GaussianBlur(small, (0, 0), 3).astype(np.float32) / 255.0
    else:
        p = _wipe_pattern(hh, hw, pattern)
    s = softness
    thr = t * (1 + 2 * s) - s
    m = _smoothstep(thr + s, thr - s, p).astype(np.float32)
    m = _up(m, w, h)
    out = cv2.blendLinear(b, a, m, 1.0 - m)
    if edge > 0 and 0.0 < t < 1.0:
        band = np.exp(-((p - thr) / (s * 0.45)) ** 2).astype(np.float32) * edge
        band = _up(band, w, h)
        glow = cv2.merge([(band * c).astype(np.uint8) for c in edge_color])
        out = cv2.add(out, glow)
    return out
