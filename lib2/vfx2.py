"""vfx2 — premium procedural atmospheric + graphic VFX for the Jane AMV (v2).

All frame functions take/return uint8 BGR (H, W, 3) and are deterministic in
(t, seed). Heavy one-off textures (rain tiles, lens dirt, ink fields, big
Sharingan) are built once and memoised in-process (lru_cache).

Atmosphere   rain, embers, ash, dust, halation, lens_dirt, god_rays, heat_distortion
Graphics     sharingan (BGRA), raikiri, electric_rim, chakra_aura
Transitions  iris_transition, ink_splash_transition   (t = progress 0..1)
"""
from __future__ import annotations

import math
import zlib
from functools import lru_cache

import cv2
import numpy as np

FPS = 30


# ----------------------------------------------------------------- helpers
def _clip01(x):
    return min(1.0, max(0.0, float(x)))


def _ss(e0, e1, x):
    u = _clip01((x - e0) / (e1 - e0))
    return u * u * (3 - 2 * u)


def _smooth(e0, e1, x):  # array smoothstep
    u = np.clip((x - e0) / (e1 - e0), 0, 1)
    return u * u * (3 - 2 * u)


def _fidx(t):
    return int(round(t * FPS))


def _half(img):
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)


def _up(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _screen(base, add):
    """Screen blend, uint8 in/out (fast path)."""
    inv = cv2.multiply(255 - base, 255 - add, scale=1 / 255.0)
    return 255 - inv


def _luma(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


@lru_cache(maxsize=32)
def _noise_tex(h, w, cells, seed):
    """Smooth tileable-ish value noise 0..1 float32 (h, w)."""
    rng = np.random.default_rng(seed)
    g = rng.random((cells, cells)).astype(np.float32)
    g = np.vstack([g, g[:1]])
    g = np.hstack([g, g[:, :1]])
    big = cv2.resize(g, (w + w // cells, h + h // cells), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(big[:h, :w])


@lru_cache(maxsize=16)
def _fbm_tex(h, w, seed, octaves=4, base=4):
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        acc += amp * _noise_tex(h, w, base * 2 ** o, seed * 31 + o)
        tot += amp
        amp *= 0.5
    acc /= tot
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    return acc


def _scroll(tex, dx, dy, w=None, h=None):
    """Wrap-scroll a texture by (dx, dy) px, optional output size (tiles)."""
    th, tw = tex.shape[:2]
    w = w or tw
    h = h or th
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_WRAP)


# ===================================================================== RAIN
# layer: (tile, n_streaks, len_px(lo,hi), thick, bright, blur_x, speed px/s, scale)
_RAIN_LAYERS = (
    (384, 900, (10, 26), 1, 0.30, 0.6, 900.0, 0.55),   # far: dense, thin, dim
    (512, 380, (26, 60), 1, 0.55, 0.8, 1500.0, 0.85),  # mid
    (640, 70, (70, 150), 2, 0.85, 1.6, 2300.0, 1.25),  # near: few, long, soft
)


@lru_cache(maxsize=8)
def _rain_tile(li, seed):
    ts, n, (l0, l1), th, br, bx, _, _ = _RAIN_LAYERS[li]
    rng = np.random.default_rng(seed * 101 + li)
    canvas = np.zeros((ts, ts), np.float32)
    for _ in range(n):
        x = rng.uniform(0, ts)
        y = rng.uniform(0, ts)
        L = rng.uniform(l0, l1)
        b = br * rng.uniform(0.35, 1.0)
        # tapered streak: 3 nested segments (soft tails, bright body)
        for frac, k in ((1.0, 0.35), (0.7, 0.4), (0.4, 0.25)):
            y0 = y + L * (1 - frac) / 2
            for ox in (-ts, 0, ts):
                for oy in (-ts, 0, ts):
                    p0 = (int((x + ox) * 16), int((y0 + oy) * 16))
                    p1 = (int((x + ox) * 16), int((y0 + L * frac + oy) * 16))
                    cv2.line(canvas, p0, p1, float(b * k), th, cv2.LINE_AA, shift=4)
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=bx, sigmaY=bx * 2.5)
    return canvas


@lru_cache(maxsize=4)
def _mist_tex(h, w, seed):
    return _fbm_tex(h, w, seed + 7, octaves=4, base=3)


def rain(frame, t, intensity=1.0, angle=12.0, seed=0, color=(235, 225, 215),
         splashes=True, mist=0.35, wet=0.25):
    """Layered cinematic rain.

    intensity 0..1.5 (density/brightness); angle = slant in degrees (positive
    leans right-to-left fall); mist 0..1 low fog; wet 0..1 darken+cool the plate
    (rain needs a dimmer, cooler scene to read). color = streak tint (BGR).
    """
    h, w = frame.shape[:2]
    hw, hh = w // 2, h // 2
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    acc = np.zeros((hh, hw), np.float32)
    for li, L in enumerate(_RAIN_LAYERS):
        tile = _rain_tile(li, seed)
        ts, speed, sc = L[0], L[6], L[7]
        # dst(half-res) -> tile coords: rotate by angle, scale, scroll along fall
        k = sc  # >1 = bigger (nearer) streaks
        off = (speed * t / k) * 0.5  # half-res px
        phase = (li * 211.7 + seed * 37.3)
        M = np.float32([[ca / k, sa / k, phase],
                        [-sa / k, ca / k, -off + phase * 0.7]])
        lay = cv2.warpAffine(tile, M, (hw, hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_WRAP)
        acc += lay
    acc *= intensity
    # depth: near rain visible everywhere, slight fade toward top (sky glare)
    # backlight: streaks read more over dark areas (and catch highlights)
    lum = cv2.resize(_luma(frame), (hw, hh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    vis = 0.75 + 0.5 * (1 - lum)
    acc *= vis
    rain_rgb = np.empty((hh, hw, 3), np.float32)
    for c in range(3):
        rain_rgb[..., c] = acc * color[c]
    if splashes and intensity > 0:
        f = _fidx(t)
        rng = np.random.default_rng(seed * 7919 + f)
        n = int(22 * intensity)
        sp = np.zeros((hh, hw), np.float32)
        for _ in range(n):
            yy = rng.uniform(0.78, 1.0)
            x = rng.uniform(0, hw)
            y = yy * hh
            s = 1.0 + 3.5 * (yy - 0.78) / 0.22
            b = rng.uniform(0.25, 0.6)
            cv2.ellipse(sp, (int(x * 4), int(y * 4)), (int(s * 4 * 1.6), int(s * 4 * 0.35)), 0, 180, 360,
                        b, 1, cv2.LINE_AA, shift=2)
            for _k in range(rng.integers(2, 5)):
                ang = rng.uniform(-150, -30)
                ln = s * rng.uniform(0.8, 2.0)
                p1 = (x + math.cos(math.radians(ang)) * ln, y + math.sin(math.radians(ang)) * ln)
                cv2.line(sp, (int(x * 4), int(y * 4)), (int(p1[0] * 4), int(p1[1] * 4)), b * 0.7, 1,
                         cv2.LINE_AA, shift=2)
        sp = cv2.GaussianBlur(sp, (0, 0), 0.7)
        for c in range(3):
            rain_rgb[..., c] += sp * color[c] * 0.8
    if mist > 0:
        m = _mist_tex(hh // 2, hw // 2, seed)
        m = _scroll(m, -t * 18.0, 0)
        m = _up(m, hw, hh)
        grad = np.linspace(0.35, 1.0, hh, dtype=np.float32)[:, None]
        fog = m * grad * mist * 70
        for c in range(3):
            rain_rgb[..., c] += fog * (color[c] / 255)
    out = frame
    if wet > 0:
        k = 1 - 0.35 * wet
        cool = np.diag([k * (1 + 0.12 * wet), k, k * (1 - 0.10 * wet)]).astype(np.float32)
        out = cv2.transform(out, cool)
    lay = _up(np.clip(rain_rgb, 0, 255).astype(np.uint8), w, h)
    return _screen(out, lay)


# ============================================================ PARTICLES
@lru_cache(maxsize=16)
def _particles(kind, n, seed):
    rng = np.random.default_rng(seed * 4099 + zlib.crc32(kind.encode()) % 997)   # not hash(): it changes per process
    z = rng.random(n) ** 1.6  # mostly far
    return dict(
        x0=rng.random(n), y0=rng.random(n), z=z,
        ph=rng.random(n) * 6.283, ph2=rng.random(n) * 6.283,
        fq=rng.uniform(0.4, 1.6, n), fq2=rng.uniform(1.5, 4.0, n),
        sp=rng.uniform(0.7, 1.3, n), heat=rng.random(n),
        rot=rng.uniform(-6, 6, n), asp=rng.uniform(0.25, 0.8, n),
        tw=rng.uniform(3, 11, n),
    )


def _particle_pos(P, t, w, h, vy, vx, sway, margin=60):
    z = P['z']
    depth_speed = 0.25 + 1.1 * z
    x = (P['x0'] * (w + 2 * margin) + vx * depth_speed * t
         + sway * (0.4 + z) * np.sin(P['fq'] * t + P['ph'])
         + 0.35 * sway * np.sin(P['fq2'] * t + P['ph2']))
    y = P['y0'] * (h + 2 * margin) + vy * depth_speed * P['sp'] * t
    x = np.mod(x, w + 2 * margin) - margin
    y = np.mod(y, h + 2 * margin) - margin
    dx = vx * depth_speed + sway * (0.4 + z) * P['fq'] * np.cos(P['fq'] * t + P['ph'])
    dy = vy * depth_speed * P['sp']
    return x, y, dx, dy


def embers(frame, t, density=1.0, seed=0, wind=(60.0, -170.0), glow=1.0, heat=1.0):
    """Rising embers with depth-of-field + additive glow. wind=(vx, vy) px/s at 1080p
    (negative vy = rising). heat 0..1 shifts palette toward white-hot."""
    h, w = frame.shape[:2]
    hw, hh = w // 2, h // 2
    n = int(260 * density)
    P = _particles('ember', max(n, 1), seed)
    x, y, dx, dy = _particle_pos(P, t, w, h, wind[1], wind[0], 45.0)
    x, y, dx, dy = x / 2, y / 2, dx / 2, dy / 2
    z = P['z']
    f = _fidx(t)
    layers = [np.zeros((hh, hw, 3), np.float32) for _ in range(3)]
    for i in range(n):
        li = 0 if z[i] < 0.45 else (1 if z[i] < 0.8 else 2)
        tw = 0.55 + 0.45 * math.sin(P['tw'][i] * t + P['ph'][i] * 3)
        hot = min(1.0, P['heat'][i] * 0.8 + 0.25 * heat)
        # BGR: deep orange -> yellow-white
        col = (40 + 150 * hot ** 2, 90 + 150 * hot, 255)
        b = (0.6 + 0.9 * z[i]) * tw
        r = 0.6 + 2.6 * z[i] ** 2
        sh = 1 / 22.0  # motion streak length (shutter)
        p0 = (int(x[i] * 16), int(y[i] * 16))
        p1 = (int((x[i] - dx[i] * sh) * 16), int((y[i] - dy[i] * sh) * 16))
        c = tuple(float(v * b) for v in col)
        cv2.line(layers[li], p0, p1, c, max(1, int(round(r))), cv2.LINE_AA, shift=4)
    out = layers[0] * 1.3 + cv2.GaussianBlur(layers[1], (0, 0), 0.9) * 1.6 \
        + cv2.GaussianBlur(layers[2], (0, 0), 3.5) * 2.6
    gl = cv2.GaussianBlur(_half(out), (0, 0), 6) * (2.2 * glow)
    out = out + _up(gl, hw, hh)
    lay = _up(np.clip(out, 0, 255).astype(np.uint8), w, h)
    return cv2.add(frame, lay)


def ash(frame, t, density=1.0, seed=0, tone='light', wind=(-40.0, 70.0), opacity=0.85):
    """Falling ash / snow-like flakes, tumbling, with DOF. tone 'light' (snowy ash,
    pale grey) or 'dark' (burnt black flakes). Positive wind[1] = falling."""
    h, w = frame.shape[:2]
    hw, hh = w // 2, h // 2
    n = int(320 * density)
    P = _particles('ash', max(n, 1), seed + 11)
    x, y, _, _ = _particle_pos(P, t, w, h, wind[1], wind[0], 60.0)
    x, y = x / 2, y / 2
    z = P['z']
    layers = [np.zeros((hh, hw), np.float32) for _ in range(3)]
    for i in range(n):
        li = 0 if z[i] < 0.45 else (1 if z[i] < 0.8 else 2)
        s = 0.8 + 4.5 * z[i] ** 1.5
        rot = P['rot'][i] * t + P['ph'][i]
        sq = 0.25 + 0.75 * abs(math.cos(rot))  # tumbling
        a = math.degrees(P['ph2'][i] + 0.3 * rot)
        cv2.ellipse(layers[li], (int(x[i] * 16), int(y[i] * 16)),
                    (max(1, int(s * 16)), max(1, int(s * sq * P['asp'][i] * 16 + 8))), a, 0, 360,
                    float(0.55 + 0.45 * z[i]), -1, cv2.LINE_AA, shift=4)
    al = layers[0] * 0.55 + cv2.GaussianBlur(layers[1], (0, 0), 0.9) \
        + cv2.GaussianBlur(layers[2], (0, 0), 4.0) * 1.3
    al = np.clip(_up(al, w, h) * opacity, 0, 1)[..., None]
    col = np.array((215, 214, 210) if tone == 'light' else (28, 26, 26), np.float32)
    out = frame.astype(np.float32) * (1 - al) + col * al
    return out.astype(np.uint8)


def dust(frame, t, density=1.0, seed=0, color=(170, 210, 255), light=None):
    """Floating dust motes catching light (slow drift, soft, additive). If `light`
    (uint8/float map HxW) is given, motes glow only where it's bright."""
    h, w = frame.shape[:2]
    hw, hh = w // 2, h // 2
    n = int(180 * density)
    P = _particles('dust', max(n, 1), seed + 5)
    x, y, _, _ = _particle_pos(P, t, w, h, -8.0, 14.0, 30.0)
    x, y = x / 2, y / 2
    z = P['z']
    lay = [np.zeros((hh, hw), np.float32), np.zeros((hh, hw), np.float32)]
    for i in range(n):
        tw = 0.5 + 0.5 * math.sin(P['tw'][i] * 0.4 * t + P['ph'][i])
        li = 0 if z[i] < 0.7 else 1
        cv2.circle(lay[li], (int(x[i] * 16), int(y[i] * 16)), int((0.6 + 3.5 * z[i] ** 2) * 16),
                   float(tw * (0.4 + 0.6 * z[i])), -1, cv2.LINE_AA, shift=4)
    a = cv2.GaussianBlur(lay[0], (0, 0), 0.8) + cv2.GaussianBlur(lay[1], (0, 0), 4.5) * 2.0
    if light is None:
        sm = cv2.resize(_luma(frame), (hw // 4, hh // 4), interpolation=cv2.INTER_AREA)
        light = _up(cv2.GaussianBlur(sm, (0, 0), 3).astype(np.float32) / 255, hw, hh)
    else:
        light = cv2.resize(light.astype(np.float32), (hw, hh), interpolation=cv2.INTER_AREA)
        if light.max() > 1.5:
            light = light / 255
    a = a * (0.25 + light)
    lay3 = np.clip(a[..., None] * np.array(color, np.float32), 0, 255).astype(np.uint8)
    return cv2.add(frame, _up(lay3, w, h))


# ==================================================================== LENS
def halation(frame, strength=0.6, threshold=200, tint=(30, 80, 255)):
    """Film halation: red-orange bleed hugging highlight edges (two radii).
    The highlight core itself is subtracted so bright plates don't wash pink."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    l = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hl = np.clip((l - threshold) / (256 - threshold), 0, 1)
    g = cv2.GaussianBlur(hl, (0, 0), 2.5) * 0.9 + cv2.GaussianBlur(hl, (0, 0), 9) * 0.7
    g = np.clip(g - hl * 0.9, 0, 1)  # bleed lives *around* highlights
    lay = g[..., None] * np.array(tint, np.float32) * strength * 1.6
    lay = _up(np.clip(lay, 0, 255).astype(np.uint8), w, h)
    return _screen(frame, lay)


@lru_cache(maxsize=4)
def _dirt_tex(h, w, seed):
    rng = np.random.default_rng(seed + 900)
    d = np.zeros((h // 2, w // 2), np.float32)
    hh, hw = d.shape
    for _ in range(70):  # soft smudges
        c = (int(rng.uniform(0, hw)), int(rng.uniform(0, hh)))
        ax = (int(rng.uniform(10, 120)), int(rng.uniform(8, 90)))
        cv2.ellipse(d, c, ax, rng.uniform(0, 180), 0, 360, float(rng.uniform(0.15, 0.6)), -1, cv2.LINE_AA)
    d = cv2.GaussianBlur(d, (0, 0), 18)
    sm = np.zeros_like(d)
    for _ in range(9):  # finger smears (curved strokes)
        x, y = rng.uniform(0, hw), rng.uniform(0, hh)
        ang = rng.uniform(0, 6.28)
        pts = []
        for k in range(12):
            ang += rng.uniform(-0.25, 0.25)
            x += math.cos(ang) * 14
            y += math.sin(ang) * 14
            pts.append((int(x), int(y)))
        cv2.polylines(sm, [np.array(pts, np.int32)], False, float(rng.uniform(0.3, 0.7)),
                      int(rng.uniform(6, 16)), cv2.LINE_AA)
    d += cv2.GaussianBlur(sm, (0, 0), 6)
    sp = np.zeros_like(d)
    for _ in range(260):  # dust specks + a few rings
        c = (int(rng.uniform(0, hw)), int(rng.uniform(0, hh)))
        r = int(rng.uniform(1, 4))
        if rng.random() < 0.08:
            cv2.circle(sp, c, r + 3, float(rng.uniform(0.2, 0.5)), 1, cv2.LINE_AA)
        else:
            cv2.circle(sp, c, r, float(rng.uniform(0.2, 0.7)), -1, cv2.LINE_AA)
    d += cv2.GaussianBlur(sp, (0, 0), 1.2) * 0.6
    d *= 0.6 + 0.8 * _fbm_tex(hh, hw, seed + 3, 3, 3)
    d = np.clip(d / (np.percentile(d, 99.7) + 1e-6), 0, 1)
    return cv2.resize((d * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)


def lens_dirt(frame, strength=0.8, seed=0, threshold=200):
    """Procedural lens smudges/dust revealed only where bright light hits.
    Threshold adapts upward on bright plates so only true highlights light it."""
    h, w = frame.shape[:2]
    dirt = _dirt_tex(h, w, seed)
    small = cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    l = small.max(axis=2)
    thr = min(250.0, max(threshold, float(np.percentile(l, 90))))
    m = np.clip((l - thr) / (256 - thr), 0, 1)[..., None]
    light = small * m
    light = cv2.GaussianBlur(light, (0, 0), 10) * 0.9 + cv2.GaussianBlur(light, (0, 0), 30) * 1.4
    light = _up(np.clip(light * strength, 0, 255).astype(np.uint8), w, h)
    lay = cv2.multiply(light, cv2.merge([dirt, dirt, dirt]), scale=1 / 255.0)
    return cv2.add(frame, lay)


def god_rays(frame, center=(0.5, 0.3), strength=0.8, threshold=170, decay=0.9,
             tint=(170, 215, 255), streaks=0.6, seed=0):
    """Volumetric light shafts radiating from `center` (normalized) out of bright areas."""
    h, w = frame.shape[:2]
    sw, sh = w // 4, h // 4
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    l = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    thr = min(250.0, max(threshold, float(np.percentile(l, 88))))  # adapt on bright plates
    m = np.clip((l - thr) / (256 - thr), 0, 1)
    cx, cy = center[0] * sw, center[1] * sh
    # light source falls off away from the centre so rays have a clear origin
    yy0, xx0 = np.mgrid[0:sh, 0:sw].astype(np.float32)
    m = m * np.exp(-((xx0 - cx) ** 2 + (yy0 - cy) ** 2) / (2 * (0.35 * sw) ** 2))
    if streaks > 0:  # angular occlusion so shafts break into beams
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        ang = np.arctan2(yy - cy, xx - cx)
        n = _noise_tex(1, 256, 48, seed + 1)[0]
        idx = ((ang + np.pi) / (2 * np.pi) * 255).astype(np.int32)
        m = m * (1 - streaks + streaks * n[idx] * 1.6)
    acc = m
    step = 0.08 * decay
    for k in range(6):  # log-step radial blur
        s = 1 / (1 + step * (2 ** k))  # sample toward the centre -> rays point outward
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        z = cv2.warpAffine(acc, M, (sw, sh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT)
        acc = (acc + z) * 0.5
    acc = cv2.GaussianBlur(acc, (0, 0), 1.5)
    lay = acc[..., None] * np.array(tint, np.float32) * strength * 4.0
    lay = _up(np.clip(lay, 0, 255).astype(np.uint8), w, h)
    return _screen(frame, lay)


def heat_distortion(frame, t, strength=6.0, mask=None, scale=1.0, seed=0):
    """Rising heat shimmer via animated noise displacement. mask (HxW 0..1 or
    uint8) limits where it applies."""
    h, w = frame.shape[:2]
    sw, sh = w // 4, h // 4
    n1 = _fbm_tex(sh, sw, seed + 21, 3, int(6 * scale) or 1)
    n2 = _fbm_tex(sh, sw, seed + 22, 3, int(6 * scale) or 1)
    d1 = _scroll(n1, 0, t * 55.0) - 0.5
    d2 = _scroll(n2, t * 6.0, t * 40.0) - 0.5
    dx = _up(d1, w, h) * strength * 2
    dy = _up(d2, w, h) * strength * 2
    if mask is not None:
        mk = mask.astype(np.float32)
        if mk.max() > 1.5:
            mk = mk / 255
        dx *= mk
        dy *= mk
    xx, yy = _grid(h, w)
    return cv2.remap(frame, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@lru_cache(maxsize=4)
def _grid(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


# =============================================================== SHARINGAN
_RED_IN = np.array((38, 30, 215), np.float32)   # BGR inner red
_RED_OUT = np.array((14, 6, 120), np.float32)   # BGR rim red


def _poly_tomoe(cx, cy, R, theta, head, tail_deg, ss):
    """Tomoe = head disc + tail tapering along the ring (pointing backward)."""
    pts_out, pts_in = [], []
    N = 28
    for k in range(N + 1):
        s = k / N
        ang = theta - math.radians(tail_deg) * s
        wdt = head * (1 - s) ** 1.05
        rr = R + head * 0.75 * s ** 0.8  # tail sweeps outward off the head (comma)
        pts_out.append((cx + (rr + wdt) * math.cos(ang), cy + (rr + wdt) * math.sin(ang)))
        pts_in.append((cx + (rr - wdt * 0.15) * math.cos(ang), cy + (rr - wdt * 0.15) * math.sin(ang)))
    poly = np.array(pts_out + pts_in[::-1], np.float64) * ss
    hx, hy = cx + R * math.cos(theta), cy + R * math.sin(theta)
    return poly, (hx * ss, hy * ss, head * ss)


def _poly_blade(cx, cy, R, theta, curve, r0, r1, wmax, ss):
    """Curved scythe blade from the centre to the rim (Mangekyo pinwheel)."""
    lead, trail = [], []
    N = 40
    for k in range(N + 1):
        s = k / N
        r = r0 + (r1 - r0) * s
        ang = theta + curve * s ** 1.25
        wd = wmax * (math.sin(math.pi * min(1.0, s * 1.15 + 0.02)) ** 0.8) * (1 - s) ** 0.25
        lead.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        a2 = ang - wd / max(r, 1e-3)
        trail.append((cx + r * math.cos(a2), cy + r * math.sin(a2)))
    return np.array(lead + trail[::-1], np.float64) * ss


def _iris_base(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    R = size / 2 * 0.98
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / R
    ang = np.arctan2(yy - c, xx - c)
    k = np.clip(r, 0, 1)[..., None]
    col = _RED_IN * (1 - k ** 1.8) + _RED_OUT * k ** 1.8
    # subtle radial fibres
    fib = 1 + 0.07 * np.sin(ang * 61 + 3 * np.sin(ang * 7)) * np.clip(r - 0.25, 0, 1)
    col = col * fib[..., None]
    alpha = np.clip((1 - r) * R, 0, 1)  # 1px AA edge
    return col, alpha, r, R, c


def _render_eye(size, kind, rot, morph=0.0, highlight=True, ss=3):
    col, alpha, r, R, c = _iris_base(size)
    S = size * ss
    blk = np.zeros((S, S), np.uint8)
    redcut = np.zeros((S, S), np.uint8)
    C, RR = c, R
    rotr = math.radians(rot)
    sh = 4  # cv2 subpixel shift bits
    f = 2 ** sh

    def fpoly(m, poly, v=255):
        cv2.fillPoly(m, [np.round(poly * f).astype(np.int32)], v, cv2.LINE_AA, shift=sh)

    def fcirc(m, x, y, rad, v=255, th=-1):
        cv2.circle(m, (int(round(x * f)), int(round(y * f))), int(round(rad * f)), v, th, cv2.LINE_AA, shift=sh)

    # outer black ring
    fcirc(blk, C * ss, C * ss, RR * ss * 0.965, 255, int(RR * ss * 0.07))
    tomoe_w = 1.0 - _ss(0.5, 0.68, morph) if kind == '3tomoe' else 0.0
    ms_w = 1.0 - tomoe_w if kind == '3tomoe' else 1.0
    tom = np.zeros((S, S), np.uint8)
    ms = np.zeros((S, S), np.uint8)
    if tomoe_w > 0:
        fcirc(tom, C * ss, C * ss, RR * ss * 0.58, 200, max(1, int(RR * ss * 0.022)))  # thin ring
        fcirc(tom, C * ss, C * ss, RR * ss * 0.21)  # pupil
        tail = 52 + 120 * _ss(0.0, 0.7, morph)
        head = RR * (0.12 - 0.03 * _ss(0.2, 0.7, morph))
        for k in range(3):
            th = rotr + k * 2 * math.pi / 3
            poly, (hx, hy, hr) = _poly_tomoe(C, C, RR * 0.58, th, head, tail, ss)
            fpoly(tom, poly)
            fcirc(tom, hx, hy, hr)
    if ms_w > 0:
        k_ = 'kakashi_mangekyo' if kind == '3tomoe' else kind
        if k_ == 'kakashi_mangekyo':
            fcirc(ms, C * ss, C * ss, RR * ss * 0.30)
            for k in range(3):
                th = rotr + k * 2 * math.pi / 3
                fpoly(ms, _poly_blade(C, C, RR, th, 1.35, RR * 0.18, RR * 0.95, RR * 0.36, ss))
            fcirc(redcut, C * ss, C * ss, RR * ss * 0.075)
        else:  # obito: thicker blades + inner red ring cut
            fcirc(ms, C * ss, C * ss, RR * ss * 0.34)
            for k in range(3):
                th = rotr + k * 2 * math.pi / 3 + math.pi / 3
                fpoly(ms, _poly_blade(C, C, RR, th, -1.15, RR * 0.2, RR * 0.95, RR * 0.46, ss))
            fcirc(redcut, C * ss, C * ss, RR * ss * 0.24, 255, max(1, int(RR * ss * 0.035)))
            fcirc(redcut, C * ss, C * ss, RR * ss * 0.07)
    ds = lambda m: cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    a_blk = ds(blk)
    a_tom = ds(tom) * tomoe_w
    a_ms = np.clip(ds(ms) - ds(redcut), 0, 1) * ms_w
    ink = np.clip(a_blk + a_tom + a_ms, 0, 1)
    ink_col = np.array((12, 8, 10), np.float32)
    out = col * (1 - ink[..., None]) + ink_col * ink[..., None]
    if highlight:  # glossy specular + rim shade -> reads as an eye, not a sticker
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        hx, hy = C - RR * 0.38, C - RR * 0.42
        sp = np.exp(-(((xx - hx) / (RR * 0.16)) ** 2 + ((yy - hy) / (RR * 0.10)) ** 2))
        sp2 = np.exp(-(((xx - (C + RR * 0.3)) / (RR * 0.06)) ** 2 + ((yy - (C + RR * 0.36)) / (RR * 0.04)) ** 2))
        out = out + (sp * 0.75 + sp2 * 0.4)[..., None] * (255 - out)
        shade = 1 - 0.35 * np.clip((r - 0.7) / 0.3, 0, 1) * (0.5 + 0.5 * np.clip((yy - C) / RR + 0.3, 0, 1))
        out = out * shade[..., None]
    bgra = np.dstack([np.clip(out, 0, 255), alpha * 255]).astype(np.uint8)
    return bgra


def sharingan(size=512, t=0.0, kind='3tomoe', spin=0.6, morph=None, highlight=True, angle=0.0):
    """Procedural Sharingan graphic -> BGRA uint8 (size, size, 4).

    kind: '3tomoe' | 'kakashi_mangekyo' | 'obito_mangekyo'. spin = rev/s
    (tomoe spin clockwise-on-screen). morph (0..1, only for kind='3tomoe'):
    tails stretch + the pattern crossfades into Kakashi's Mangekyo pinwheel
    while the spin accelerates. angle = extra rotation in degrees."""
    m = 0.0 if morph is None else _clip01(morph)
    rot = angle + 360.0 * spin * t + (540.0 * m ** 2 if morph is not None else 0.0)
    return _render_eye(int(size), kind, rot, m, highlight)


@lru_cache(maxsize=6)
def _eye_cached(kind, size=1400):
    return _render_eye(size, kind, 0.0, 0.0, True, ss=2)


def _over_bgra(dst_f, bgra_u8_full):
    a = bgra_u8_full[..., 3:4].astype(np.float32) / 255
    return dst_f * (1 - a) + bgra_u8_full[..., :3].astype(np.float32) * a


def _scale_about(img, s, cx, cy, out_center=None):
    h, w = img.shape[:2]
    ox, oy = out_center if out_center else (cx, cy)
    M = np.float32([[s, 0, ox - s * cx], [0, s, oy - s * cy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def iris_transition(a, b, t, center=(0.5, 0.5), graphic=True, kind='3tomoe',
                    center_b=(0.5, 0.5), turns=1.25, zoom_a=True):
    """Iris wipe A->B. graphic=True: a spinning Sharingan grows out of the eye at
    `center` (normalized, in A), the camera dives through its pupil and B opens
    from inside the pupil (match cut). t = progress 0..1 (~14-20 frames)."""
    h, w = a.shape[:2]
    t = _clip01(t)
    diag = math.hypot(w, h)
    cA = (center[0] * w, center[1] * h)
    cB = (center_b[0] * w, center_b[1] * h)
    k = _ss(0.25, 0.95, t)
    cx, cy = cA[0] + (cB[0] - cA[0]) * k, cA[1] + (cB[1] - cA[1]) * k
    xx, yy = _grid(h, w)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if not graphic:
        e = t * t * (3 - 2 * t)
        rad = e * diag * 0.6
        ring = max(4.0, h * 0.012)
        A = _scale_about(a, 1 + 0.25 * e, cA[0], cA[1]) if zoom_a else a
        B = _scale_about(b, 1.25 - 0.25 * e, cB[0], cB[1], (cx, cy))
        m_in = np.clip(rad - dist, 0, 1)[..., None]
        m_ring = (np.clip(rad + ring - dist, 0, 1) - m_in[..., 0])[..., None]
        out = A.astype(np.float32) * (1 - m_in - m_ring) + B.astype(np.float32) * m_in
        return np.clip(out, 0, 255).astype(np.uint8)
    # exponential dolly: graphic radius from r0 to r1 (pupil > screen at t=1)
    r0, r1 = h * 0.035, diag * 0.62 / 0.21
    e = t ** 1.6
    rg = r0 * (r1 / r0) ** e
    grow = rg / r0
    A = _scale_about(a, min(1 + (grow - 1) * 0.08, 6.0), cA[0], cA[1], (cx, cy)) if zoom_a else a
    fade = _ss(0.0, 0.12, t)
    # red bloom around the eye during the approach (quarter-res, screen)
    gk = (1 - _ss(0.5, 0.8, t)) * fade
    if gk > 0:
        q = 4
        xq, yq = _grid(h // q, w // q)
        dq = np.sqrt((xq - cx / q) ** 2 + (yq - cy / q) ** 2)
        gl = np.exp(-np.maximum(dq - rg * 0.9 / q, 0) ** 2 / (2 * (rg * 0.45 / q) ** 2)) * 0.6 * gk
        glow = np.clip(gl[..., None] * np.array((20, 20, 255), np.float32), 0, 255).astype(np.uint8)
        A = _screen(A, _up(glow, w, h))
    # Sharingan fades in fast, spins (pre-rendered hi-res, warped per frame)
    eye = _eye_cached(kind)
    es = eye.shape[0]
    rot = 360.0 * turns * (t ** 0.7)
    s = rg / (es / 2 * 0.98)
    M = cv2.getRotationMatrix2D((es / 2, es / 2), -rot, s)
    M[0, 2] += cx - es / 2
    M[1, 2] += cy - es / 2
    g = cv2.warpAffine(eye, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ga = g[..., 3].astype(np.float32) * (fade / 255.0)
    out = cv2.blendLinear(A, np.ascontiguousarray(g[..., :3]), 1 - ga, ga)
    # pupil hole reveals B (B dollies out from 1.35 -> 1.0)
    rp = rg * 0.21
    reveal = _ss(h * 0.06, h * 0.32, rp)
    if reveal > 0:
        sb = 1.35 - 0.35 * _ss(0.5, 1.0, t)
        B = _scale_about(b, sb, cB[0], cB[1], (cx, cy))
        if rp > diag:
            return B
        m = np.clip(rp - dist, 0, 1) * reveal
        out = cv2.blendLinear(out, B, 1 - m, m)
    return out


# ============================================================ INK SPLASH
@lru_cache(maxsize=8)
def _ink_field(h, w, seed):
    """Arrival-time field 0..1: where/when ink reaches each pixel."""
    rng = np.random.default_rng(seed + 333)
    sw, sh = w // 2, h // 2
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    ang = rng.uniform(-0.5, 0.5) + (0 if rng.random() < 0.5 else math.pi)
    dx, dy = math.cos(ang), math.sin(ang)
    proj = (xx - sw / 2) * dx + (yy - sh / 2) * dy
    proj = (proj - proj.min()) / (proj.max() - proj.min())
    n = _fbm_tex(sh, sw, seed + 40, 5, 3)
    field = 0.12 + proj * 0.78 + (n - 0.5) * 0.35
    clear = field.copy()  # B is revealed only by the flood + big splats (not spray/drips)
    # big splats with spiky (polar-noise) edges
    for i in range(7):
        px = rng.uniform(0.1, 0.9) * sw
        py = rng.uniform(0.1, 0.9) * sh
        R = rng.uniform(0.10, 0.24) * sh
        t0 = rng.uniform(0.0, 0.5)
        th = np.arctan2(yy - py, xx - px)
        spikes = np.zeros_like(th)
        for f_ in (5, 11, 23, 37):
            spikes += np.maximum(np.sin(th * f_ + rng.uniform(0, 6.28)), 0) ** 6 * rng.uniform(0.2, 0.7)
        rad = R * (1 + 0.15 * np.sin(th * 3 + rng.uniform(0, 6)) + spikes)
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2) / rad
        field = np.minimum(field, t0 + d * 0.35)
        clear = np.minimum(clear, t0 + 0.2 + d * 0.5)
        # drips running down from the splat
        for _ in range(int(rng.integers(2, 5))):
            x0 = px + rng.uniform(-0.8, 0.8) * R
            y0 = py + R * rng.uniform(0.3, 0.8)
            L = rng.uniform(0.15, 0.45) * sh
            wd = rng.uniform(3, 9)
            tt = (yy - y0) / L
            prof = wd * (1 - 0.6 * np.clip(tt, 0, 1)) + 1.5
            inside = (np.abs(xx - x0) < prof) & (tt > -0.2)
            bulb = np.sqrt((xx - x0) ** 2 + (yy - (y0 + L)) ** 2) < wd * 1.4
            dr = np.where(inside, t0 + 0.2 + np.clip(tt, 0, 1) * 0.35, 9.0)
            dr = np.minimum(dr, np.where(bulb, t0 + 0.55, 9.0))
            field = np.minimum(field, dr)
    # spray droplets flying ahead of the wave
    for _ in range(260):
        px, py = rng.uniform(0, sw), rng.uniform(0, sh)
        r = rng.uniform(1, 7) ** 1.3
        dd = np.zeros((1,))
        y0, y1 = int(max(0, py - r - 2)), int(min(sh, py + r + 2))
        x0, x1 = int(max(0, px - r - 2)), int(min(sw, px + r + 2))
        if y1 <= y0 or x1 <= x0:
            continue
        sub = np.sqrt((xx[y0:y1, x0:x1] - px) ** 2 + (yy[y0:y1, x0:x1] - py) ** 2) < r
        tval = rng.uniform(0.0, 0.7)
        field[y0:y1, x0:x1] = np.where(sub, np.minimum(field[y0:y1, x0:x1], tval), field[y0:y1, x0:x1])
    field = cv2.GaussianBlur(field, (0, 0), 0.8)
    clear = np.maximum(cv2.GaussianBlur(clear, (0, 0), 0.8), field)
    rs = lambda f: cv2.resize(f.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return rs(field), rs(clear)


def ink_splash_transition(a, b, t, seed=0, band=0.13, ink=(14, 12, 16), paper_flash=False):
    """Manga ink-splatter wipe A->B: splats + drips + spray flood the frame in
    black ink, B is revealed behind the ink front. t = progress 0..1."""
    h, w = a.shape[:2]
    F, Fc = _ink_field(h, w, seed)
    tt = -0.05 + _clip01(t) * (1.0 + 0.35 + band + 0.05)
    soft = 0.006
    m_ink = np.clip((tt - F) / soft * 0.5 + 0.5, 0, 1)            # ink arrived
    m_b = np.clip((tt - band - Fc) / soft * 0.5 + 0.5, 0, 1)      # ink cleared -> B
    if m_ink.max() <= 0:
        return a.copy()
    # wet sheen: faint highlight just behind the ink front
    sheen = np.clip(1 - np.abs(tt - F - 0.02) * 50, 0, 1) * 30
    ink_img = cv2.merge([cv2.convertScaleAbs(sheen, alpha=1, beta=float(c)) for c in ink])
    out = cv2.blendLinear(a, ink_img, 1 - m_ink, m_ink)
    if m_b.max() > 0:
        out = cv2.blendLinear(out, b, 1 - m_b, m_b)
    return out


# ================================================================ RAIKIRI
def _bolt(rng, p0, p1, disp=0.22, levels=6, branch=0.35, depth=0, out=None, wgt=1.0):
    if out is None:
        out = []
    pts = [np.array(p0, np.float64), np.array(p1, np.float64)]
    d = disp
    for _ in range(levels):
        new = [pts[0]]
        for i in range(len(pts) - 1):
            a_, b_ = pts[i], pts[i + 1]
            seg = b_ - a_
            L = np.hypot(*seg) + 1e-6
            nrm = np.array((-seg[1], seg[0])) / L
            mid = (a_ + b_) / 2 + nrm * rng.normal() * d * L
            new += [mid, b_]
        pts = new
        d *= 0.58
    out.append((np.array(pts), wgt))
    if depth < 2:
        n = len(pts)
        for i in range(2, n - 2):
            if rng.random() < branch / (n / 8):
                rem = pts[-1] - pts[i]
                L = np.hypot(*rem) * rng.uniform(0.25, 0.6)
                ang = math.atan2(rem[1], rem[0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 0.9)
                q = pts[i] + L * np.array((math.cos(ang), math.sin(ang)))
                _bolt(rng, pts[i], q, disp * 1.1, max(3, levels - 2), branch * 0.6, depth + 1, out, wgt * 0.55)
    return out


def raikiri(frame, t, origin=(0.5, 0.6), target=None, seed=0, color=(255, 205, 120),
            radius=0.32, intensity=1.0, n_bolts=9, hold=1, ambient=0.6):
    """Raikiri / Chidori lightning from a hand. origin/target normalized.
    color BGR (default blue-white; purple lightning ~ (255, 90, 190)).
    hold = frames each bolt pattern persists (1 = new every frame, 2 = anime-ish)."""
    h, w = frame.shape[:2]
    ox, oy = origin[0] * w, origin[1] * h
    f = _fidx(t) // max(1, hold)
    rng = np.random.default_rng(seed * 100003 + f)
    R = radius * h
    bolts = []
    for i in range(n_bolts):
        ang = rng.uniform(0, 2 * math.pi)
        L = R * rng.uniform(0.45, 1.25)
        p1 = (ox + L * math.cos(ang), oy + L * math.sin(ang))
        _bolt(rng, (ox, oy), p1, 0.22, 6, 0.5, 0, bolts, rng.uniform(0.6, 1.0))
    if target is not None:
        tx, ty = target[0] * w, target[1] * h
        for _ in range(2):
            _bolt(rng, (ox, oy), (tx + rng.normal() * 12, ty + rng.normal() * 12), 0.16, 7, 0.45, 0, bolts, 1.3)
    chirps = []
    for _ in range(14):  # crackle arcs hugging the ball
        a0 = rng.uniform(0, 2 * math.pi)
        r_ = R * rng.uniform(0.08, 0.22)
        p0 = (ox + r_ * math.cos(a0), oy + r_ * math.sin(a0))
        a1 = a0 + rng.uniform(0.5, 1.5)
        r2 = R * rng.uniform(0.08, 0.3)
        p1 = (ox + r2 * math.cos(a1), oy + r2 * math.sin(a1))
        _bolt(rng, p0, p1, 0.35, 4, 0.0, 2, chirps, 0.7)
    hw, hh = w // 2, h // 2
    glow = np.zeros((hh, hw), np.float32)
    core = np.zeros((h, w), np.uint8)
    for pts, wg in bolts + chirps:
        P = np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(core, [P], False, int(255 * min(1, wg + 0.25)), max(1, int(round(2.2 * wg))),
                      cv2.LINE_AA, shift=4)
        Ph = np.round(pts * 8).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(glow, [Ph], False, float(wg), max(2, int(5 * wg)), cv2.LINE_AA, shift=4)
    fl = (0.75 + 0.25 * rng.random()) * intensity
    g1 = cv2.GaussianBlur(glow, (0, 0), 4)
    g2 = cv2.GaussianBlur(cv2.resize(glow, (hw // 2, hh // 2), interpolation=cv2.INTER_AREA), (0, 0), 12)
    g = g1 * 1.6 + _up(g2, hw, hh) * 3.0
    # ball of chakra + ambient scene light
    xx, yy = _grid(hh, hw)
    d2 = ((xx - ox / 2) ** 2 + (yy - oy / 2) ** 2)
    rb = R * 0.10 / 2 * (0.85 + 0.3 * rng.random())
    ball = np.exp(-d2 / (2 * rb ** 2)) * 2.5 + np.exp(-d2 / (2 * (rb * 2.6) ** 2)) * 0.9
    amb = np.exp(-d2 / (2 * (R * 0.9 / 2) ** 2)) * ambient
    col = np.array(color, np.float32) / 255
    lay = (g * 1.5 + amb)[..., None] * col * 150 * fl + ball[..., None] * (col * 0.4 + 0.6) * 255 * fl
    lay = _up(np.clip(lay, 0, 255).astype(np.uint8), w, h)
    white = np.clip(col * 0.35 + 0.65, 0, 1) * min(fl, 1.0)
    core3 = cv2.merge([cv2.convertScaleAbs(core, alpha=float(white[c])) for c in range(3)])
    out = _screen(frame, lay)
    return cv2.add(out, core3)


def electric_rim(frame, t, mask=None, color=(255, 205, 120), strength=1.0, seed=0, thresh=(60, 140)):
    """Flickering electric glow along edges (mask edge if given, else Canny)."""
    h, w = frame.shape[:2]
    hw, hh = w // 2, h // 2
    if mask is not None:
        mk = cv2.resize(mask.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_AREA)
        if mk.max() <= 1:
            mk = mk * 255
        e = cv2.morphologyEx(mk, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    else:
        e = cv2.Canny(cv2.GaussianBlur(_half(_luma(frame)), (0, 0), 1.2), *thresh)
    f = _fidx(t)
    n = _noise_tex(hh, hw, 40, seed + f % 997)
    brk = (n > 0.55).astype(np.float32)
    ef = e.astype(np.float32) / 255 * brk
    # jitter: displace edges by a pixel or two per frame
    dx = (f * 7 % 5) - 2
    ef = np.roll(ef, dx, axis=1)
    g = ef * 1.2 + cv2.GaussianBlur(ef, (0, 0), 3) * 3 + cv2.GaussianBlur(ef, (0, 0), 10) * 4
    fl = 0.7 + 0.3 * ((f * 2654435761) % 1000) / 1000
    lay = g[..., None] * np.array(color, np.float32) * strength * fl
    lay = _up(np.clip(lay, 0, 255).astype(np.uint8), w, h)
    return _screen(frame, lay)


def chakra_aura(frame, t, mask, color=(255, 120, 60), strength=1.0, seed=0, height=0.25):
    """Flame-like chakra aura rising off a subject mask (HxW uint8/float).
    Rendered outside the mask only (subject stays clean)."""
    h, w = frame.shape[:2]
    sw, sh = w // 4, h // 4
    mk = cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    if mk.max() > 1.5:
        mk /= 255
    pad = max(3, int(height * sh * 0.35)) | 1
    dil = cv2.GaussianBlur(cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))), (0, 0), pad * 0.4)
    n1 = _fbm_tex(sh, sw, seed + 61, 4, 6)
    n2 = _fbm_tex(sh, sw, seed + 62, 4, 10)
    nA = _scroll(n1, 0, t * 90) - 0.5
    nB = _scroll(n2, t * 8, t * 140)
    xx, yy = _grid(sh, sw)
    up = height * sh * (0.3 + 0.9 * nB)  # tongues stretch upward
    src = cv2.remap(dil, xx + nA * 18, yy + up, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    tongues = _smooth(0.2, 0.6, nB * 0.7 + src * 0.6)
    flame = np.clip(src * tongues * 1.6, 0, 1.5) * (1 - mk)
    flame = cv2.GaussianBlur(flame, (0, 0), 1.0)
    lay = (flame[..., None] * np.array(color, np.float32) * strength
           + (flame ** 3)[..., None] * 255 * 0.5 * strength)
    lay = _up(np.clip(lay, 0, 255).astype(np.uint8), w, h)
    return _screen(frame, lay)
