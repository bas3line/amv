"""fxpack -- effects & transitions for the UZUMAKI Naruto 10.10 edit (style_bible.md).

Contract
- Frames are uint8 BGR (1080, 1920, 3). Every function is pure (never mutates its
  input) and deterministic (randomness only from explicit seeds).
- Beat-relative functions take `n` = output frame index relative to the beat frame
  b (n = 0 is the first clean frame of the new shot, n = -1 is b-1, ...).
- Progress-based transitions take `k` = frame index inside the transition.
- Every geometric op reports its scale to a per-frame scale log:
      fx.reset_scale(); img = ...; fx.used_scale()  -> max scale used on that frame
  so the director can enforce the 1.12 (held) / 1.35 (absolute) caps.

Reuses lib/shaders.py (bloom, zoom_blur, motion_blur, camera, rgb_split,
chroma_radial, vignette) and lib2/vfx2.py (embers, dust, halation, god_rays,
lens_dirt, raikiri, electric_rim, chakra_aura, sharingan eyes) and lib2/motion.py
(flow speed ramps). Nothing in those files is modified.

Bible map (section 10): T0 slit_open | T1 hard cut (no fx) | T2 t2_frame |
T3 radial_collapse / radial_enter | T4 blur_dissolve | T5 hue_flash | T6 exposure_pop |
T7 white_flash (+ from_white) | T8 glitch_wave | T9 iris | T10 strobe |
T11 pip / pip_arrive | T12 neon_edges + ink_burst + settle | T13 tilt_plane |
T14 cross_dissolve. Camera: entry_punch (Z1), push_in (Z2), camera_move (shake /
drift / handheld), pulse (q-DyE envelope), slow_push. Speed: speed_ramp, conform.
Grade: fit_grade / apply_grade / grade (per act, closed-loop to the bible targets).
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache

import cv2
import numpy as np

_W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (f"{_W}/lib", f"{_W}/lib2"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import shaders as S  # noqa: E402
import vfx2 as V  # noqa: E402
import motion as MO  # noqa: E402

W, H, FPS = 1920, 1080, 30
HOLD_CAP = 1.12   # max scale on held frames
MAX_SCALE = 1.35  # max scale ever
BAR = 78          # letterbox bar height (px at 1080)
REFLECT = cv2.BORDER_REFLECT101

GOLD = (120, 205, 255)        # BGR warm gold
GOLD_WHITE = (200, 235, 255)  # BGR white-gold
PALE_BLUE = (255, 225, 175)   # BGR

# ----------------------------------------------------------------------------- scale log
_SCALE = {"max": 1.0}


def reset_scale():
    _SCALE["max"] = 1.0


def used_scale():
    return _SCALE["max"]


def _note(s):
    if s > _SCALE["max"]:
        _SCALE["max"] = float(s)


# ----------------------------------------------------------------------------- helpers
def _clip01(x):
    return min(1.0, max(0.0, float(x)))


def ease_out_cubic(x):
    x = _clip01(x)
    return 1 - (1 - x) ** 3


def ease_in_quad(x):
    x = _clip01(x)
    return x * x


def ease_in_out(x):
    x = _clip01(x)
    return x * x * (3 - 2 * x)


def ease_in_out_sine(x):
    x = _clip01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out_back(x, c1=1.70158):
    x = _clip01(x)
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def _thumb(img, w=320):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def luma(img):
    """Mean Rec.601 luma (0-255)."""
    return float(cv2.cvtColor(_thumb(img), cv2.COLOR_BGR2GRAY).mean())


def stats(img):
    """Mean luma, mean HSV saturation (0-255) and mean(R)-mean(B) -- the units the bible uses."""
    t = _thumb(img)
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    m = t.reshape(-1, 3).mean(0)
    return dict(luma=float(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).mean()), sat=float(hsv[..., 1].mean()),
                rb=float(m[2] - m[0]))


def sharpness(img):
    """Laplacian variance at 480 px wide (bible sec. 0.1 threshold: > 100 on held frames)."""
    g = cv2.cvtColor(cv2.resize(img, (480, int(round(img.shape[0] * 480 / img.shape[1]))),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _screen(a, b):
    return 255 - cv2.multiply(255 - a, 255 - b, scale=1 / 255.0)


def sharpen(img, amount=0.5, sigma=1.0):
    """Unsharp mask (bible: grade per shot *after* sharpening)."""
    if amount <= 0:
        return img.copy()
    bl = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, bl, -amount, 0)


# ----------------------------------------------------------------------------- geometry
def _affine(s, rot, dx, dy, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), rot, s)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def _covers(M, w, h, eps=0.5):
    """True when every output corner maps inside the source (no edge/wedge shows)."""
    Mi = cv2.invertAffineTransform(M)
    c = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    p = Mi @ c
    return bool((p[0] >= -eps).all() and (p[0] <= w + eps).all() and (p[1] >= -eps).all() and (p[1] <= h + eps).all())


def cover_scale(rot=0.0, dx=0.0, dy=0.0, center=(0.5, 0.5), w=W, h=H):
    """Minimum scale so a frame rotated by rot deg and shifted by (dx, dy) about `center`
    still fills the canvas (no black wedges, no mirrored edges)."""
    cx, cy = center[0] * w, center[1] * h
    if abs(rot) < 1e-6 and abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 1.0
    lo, hi = 1.0, 3.0
    if _covers(_affine(lo, rot, dx, dy, cx, cy), w, h):
        return 1.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if _covers(_affine(mid, rot, dx, dy, cx, cy), w, h):
            hi = mid
        else:
            lo = mid
    return hi


def xform(img, s=1.0, rot=0.0, dx=0.0, dy=0.0, center=(0.5, 0.5), border=REFLECT, interp=cv2.INTER_CUBIC):
    """Scale/rotate about `center` (normalised) then translate. Logs the scale."""
    _note(s)
    if abs(s - 1) < 1e-6 and abs(rot) < 1e-6 and abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return img.copy()
    h, w = img.shape[:2]
    M = _affine(s, rot, dx, dy, center[0] * w, center[1] * h)
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=border)


def dir_blur(img, angle_deg, length):
    """Directional (shutter) blur of `length` px along angle (0 = +x, 90 = +y)."""
    return S.motion_blur(img, angle_deg, length) if length >= 1.0 else img.copy()


def radial_blur(img, px, center=(0.5, 0.5)):
    """Radial zoom blur whose streak is `px` long at the frame's side edge (r = W/2)."""
    if px < 0.5:
        return img.copy()
    return S.zoom_blur(img, strength=px / (0.5 * img.shape[1]), center=center, boost=0.2)


def chroma(img, px=6.0, mode="linear", angle=0.0):
    """Chromatic aberration: 'linear' = R +px / B -px along angle; 'radial' = px at the edge."""
    if px < 0.3:
        return img.copy()
    if mode == "radial":
        return S.chroma_radial(img, amount=px / (0.5 * img.shape[1]))
    return S.rgb_split(img, px, angle)


# ----------------------------------------------------------------------------- Z1 / camera
def punch_scale(n, s0=1.22, tau=4.0, hold_cap=HOLD_CAP, trans=3):
    """Z1 zoom-out entry: s(n) = 1 + (s0-1) exp(-n/tau). For s0 above 1.22 tau is shortened so
    frame n=trans (the first held frame) is already <= hold_cap (1.30 -> tau 3.27)."""
    a = s0 - 1.0
    if a <= 0:
        return 1.0
    if a > hold_cap - 1.0:
        tau = min(tau, trans / math.log(a / (hold_cap - 1.0)))
    if n <= 0:
        return s0
    return 1.0 + a * math.exp(-n / tau)


def _punch_tau(s0, tau=4.0, hold_cap=HOLD_CAP, trans=3):
    a = s0 - 1.0
    if a > hold_cap - 1.0:
        tau = min(tau, trans / math.log(a / (hold_cap - 1.0)))
    return tau


SHAKE = {  # amp px at 1920, rot deg, length frames, translation decay tau, rotation decay tau
    "hit": (36.0, 1.5, 7, 2.5, 3.0),
    "big": (60.0, 3.0, 8, 2.5, 3.0),
    "micro": (10.0, 0.0, 2, 2.5, 3.0),
}


@lru_cache(maxsize=256)
def _shake_signs(seed, n):
    rng = np.random.default_rng(seed * 7919 + 17)
    u = rng.uniform(0.55, 1.0, size=(n, 3))
    ang = rng.uniform(-0.6, 0.6, size=n)  # direction jitter so it's not a pure diagonal
    sg = np.array([(-1) ** k for k in range(n)], np.float64)
    ph = 1 if rng.random() < 0.5 else -1
    return u, ang, sg * ph


def shake_offsets(n, preset="hit", seed=0):
    """(dx, dy, rot) of an impact shake n frames after the hit (alternating random jolts)."""
    if preset is None or n < 0:
        return 0.0, 0.0, 0.0
    amp, rot, length, tt, tr = SHAKE[preset] if isinstance(preset, str) else preset
    if n >= length:
        return 0.0, 0.0, 0.0
    u, ang, sg = _shake_signs(int(seed), 16)
    et, er = math.exp(-n / tt), math.exp(-n / tr)
    a = math.pi / 4 + ang[n]
    dx = amp * et * sg[n] * u[n, 0] * math.cos(a) * 1.25
    dy = amp * et * -sg[n] * u[n, 1] * math.sin(a)
    r = rot * er * -sg[n] * u[n, 2]
    return dx, dy, r


def handheld_offsets(t, seed=0, amp=15.0, rot=0.4, speed=0.35):
    """Soft handheld drift (bible: translation <= 15 px, rotation <= 0.4 deg)."""
    def nz(k, f):
        return (math.sin(t * f * 6.283 + seed * 1.7 + k) * 0.6 + math.sin(t * f * 2.31 * 6.283 + seed * 3.1 + 2 * k) * 0.4)
    return amp * nz(0, speed), amp * 0.6 * nz(5, speed * 0.8), rot * nz(9, speed * 0.6)


def pose(n, punch=None, tau=4.0, drift=0.0, drift_dir=(1.0, -1.0), shake=None, seed=0, handheld=False, t=0.0,
         center=(0.5, 0.5), max_scale=MAX_SCALE):
    """Camera pose (scale, rot, dx, dy) for frame n of a shot. Scale = max(punch, cover) so the
    punch zoom itself hides the shake edges; shake is attenuated if it would need > max_scale."""
    s_p = punch_scale(n, punch, tau) if punch else 1.0
    tp = _punch_tau(punch, tau) if punch else tau
    dx = dy = rot = 0.0
    if drift and n >= 0:
        L = math.hypot(*drift_dir) or 1.0
        e = math.exp(-n / tp)
        dx += drift_dir[0] / L * drift * e
        dy += drift_dir[1] / L * drift * e
    sx = sy = sr = 0.0
    if shake:
        sx, sy, sr = shake_offsets(n, shake, seed)
    if handheld:
        hx, hy, hr = handheld_offsets(t, seed)
        sx, sy, sr = sx + hx, sy + hy, sr + hr
    k = 1.0
    for _ in range(12):
        s_c = cover_scale(rot + sr * k, dx + sx * k, dy + sy * k, center)
        if s_c <= max_scale:
            break
        k *= 0.85
    return max(s_p, s_c), rot + sr * k, dx + sx * k, dy + sy * k


def camera_move(img, n, punch=None, tau=4.0, drift=0.0, drift_dir=(1.0, -1.0), shake=None, seed=0,
                handheld=False, t=0.0, center=(0.5, 0.5), blur=True, blur_frames=3, shutter=0.5):
    """Z1 entry punch + impact shake + handheld drift in one warp. Shutter motion blur between
    the previous and current pose only on the first `blur_frames` frames (bible: <= 3 blurred)."""
    cur = pose(n, punch, tau, drift, drift_dir, shake, seed, handheld, t, center)
    _note(cur[0])
    if blur and 0 <= n < blur_frames and (shake or punch):
        prv = pose(n - 1, None if n == 0 else punch, tau, 0.0 if n == 0 else drift, drift_dir,
                   shake, seed, handheld, t - 1 / FPS, center)
        if n == 0 and punch:  # the new shot has no previous frame: blur only the shake jolt
            prv = (cur[0], prv[1], prv[2], prv[3])
        if center == (0.5, 0.5):
            return S.camera(img, cur[0], cur[1], cur[2], cur[3], prev=prv, shutter=shutter)
    s, r, dx, dy = cur
    return xform(img, s, r, dx, dy, center)


def entry_punch(img, n, s0=1.22, drift=40.0, drift_dir=(1.0, -1.0), tau=4.0, center=(0.5, 0.5)):
    """Z1: zoom-out entry punch with a decaying diagonal drift. n = frames since the beat."""
    s, r, dx, dy = pose(n, s0, tau, drift, drift_dir, None, 0, False, 0.0, center)
    return xform(img, s, r, dx, dy, center)


def impact(img, n, big=False, seed=0, punch=None, pop=True, ca=True, center=(0.5, 0.5)):
    """Impact recipe (hits/explosions): shake (36 px/1.5 deg, or 60 px/3 deg when big),
    1-frame exposure pop on n=0, RGB split that decays over 3 frames."""
    out = camera_move(img, n, punch=punch, shake="big" if big else "hit", seed=seed, center=center)
    if pop and n == 0:
        out = exposure_pop(out, 170 if not big else 185)
    if ca and 0 <= n < 3:
        out = chroma(out, (8.0 if big else 5.0) * (1 - n / 3), angle=20)
    return out


def push_in(img, k, n=5, steps=(1.02, 1.03, 1.05, 1.07, 1.09), center=(0.5, 0.5)):
    """Z2 push-in ramp into a cut: k = 0..n-1 over the shot's last n frames, per-frame scale
    multiplies by `steps` (+28.6% total). Follow it with radial_collapse (T3)."""
    s = 1.0
    for i in range(min(max(k + 1, 0), n)):
        s *= steps[i]
    return xform(img, s, center=center)


def slow_push(img, u, s0=1.0, s1=1.03, center=(0.5, 0.5)):
    """Slow linear-in-time push (A1 shot 1: 1.00 -> 1.03; Hokage rock 1.00 -> 1.04)."""
    return xform(img, s0 + (s1 - s0) * _clip01(u), center=center)


def pulse(img, n, n_total, s0=1.15, ease_frames=20, floor=0.60, center=(0.5, 0.5)):
    """q-DyE pre-drop envelope: enter at s0, ease to 1.00 over `ease_frames`, exposure decays
    100% -> floor before the next cut."""
    s = s0 - (s0 - 1.0) * ease_out_cubic(n / max(ease_frames, 1))
    out = xform(img, s, center=center)
    g = 1.0 - (1.0 - floor) * _clip01(n / max(n_total - 1, 1))
    return cv2.convertScaleAbs(out, alpha=g) if g < 0.999 else out


# ----------------------------------------------------------------------------- exposure
def _gamma_lut(g):
    return np.clip(255.0 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8)


def match_luma(img, target):
    """Bring mean luma to `target`: gain when darkening, gamma lift (then white screen) when brightening."""
    t = cv2.cvtColor(_thumb(img), cv2.COLOR_BGR2GRAY)
    m = float(t.mean())
    if abs(m - target) < 0.5:
        return img.copy()
    if target < m:
        return cv2.convertScaleAbs(img, alpha=target / max(m, 1e-3))
    lo, hi = 0.08, 1.0
    if float(cv2.LUT(t, _gamma_lut(lo)).mean()) < target:  # gamma alone cannot reach it
        out = cv2.LUT(img, _gamma_lut(lo))
        m2 = luma(out)
        a = _clip01((target - m2) / max(255 - m2, 1e-3))
        return cv2.addWeighted(out, 1 - a, np.full_like(out, 255), a, 0)
    for _ in range(18):
        mid = math.sqrt(lo * hi)
        if float(cv2.LUT(t, _gamma_lut(mid)).mean()) > target:
            lo = mid
        else:
            hi = mid
    return cv2.LUT(img, _gamma_lut(math.sqrt(lo * hi)))


def exposure_pop(img, target=175.0, tint=GOLD_WHITE):
    """T6: screen-add a white-gold copy of the frame itself until mean luma ~ target (150-195)."""
    th = _thumb(img)
    tf = np.float32(tint) / 255.0

    def run(src, k):
        add = cv2.multiply(src, (float(k * tf[0]), float(k * tf[1]), float(k * tf[2]), 0.0))  # saturating u8
        return _screen(src, add)

    if luma(th) >= target:
        return img.copy()
    lo, hi = 0.0, 3.0
    if luma(run(th, hi)) < target:
        return match_luma(run(img, hi), target)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if luma(run(th, mid)) < target:
            lo = mid
        else:
            hi = mid
    return run(img, hi)


def white_flash(img, min_luma=222.0, gold=False, keep=0.12):
    """T7: solid-ish white flash frame (faint structure kept). gold=True for the final hit."""
    col = np.empty_like(img)
    col[:] = (205, 236, 255) if gold else (250, 252, 255)
    out = cv2.addWeighted(img, keep, col, 1 - keep, 0)
    m = luma(out)
    if m < min_luma:
        out = cv2.add(out, np.full_like(out, int(math.ceil(min_luma - m))))
    return out


def from_white(img, k, amounts=(0.7, 0.3, 0.0)):
    """Bloom out of white after a T7 flash: screen-add white 0.7 -> 0.3 -> 0 (k = 0, 1, 2)."""
    a = amounts[k] if 0 <= k < len(amounts) else 0.0
    if a <= 0:
        return img.copy()
    return cv2.addWeighted(img, 1 - a, np.full_like(img, 255), a, 0)


def bloom(img, threshold=170, strength=0.6, radius=1.2, tint=None):
    """Highlight bloom (shaders.bloom)."""
    return S.bloom(img, threshold, strength, radius, tint)


def gold_bloom(img, u, threshold=150, strength=1.2, target=220.0, rays_from=None, rays=0.9):
    """Build-into-drop overexposure (19.00-19.67): gold bloom 0 -> strength, mean luma climbs
    toward `target`, optional god rays from `rays_from` (normalised, e.g. the eyes)."""
    u = _clip01(u)
    m0 = luma(img)
    out = S.bloom(img, threshold, strength * u, 1.6, GOLD) if u > 0 else img.copy()
    if rays_from is not None and u > 0:
        out = V.god_rays(out, center=rays_from, strength=rays * u, threshold=150, tint=GOLD_WHITE, streaks=0.5)
    tgt = m0 + (target - m0) * u ** 1.4
    if luma(out) < tgt:
        out = match_luma(out, tgt)
    return out


# ----------------------------------------------------------------------------- letterbox
SLIT = (171, 291, 384, 459, 525, 579, 627, 669, 705, 735, 762, 789, 810, 831, 849, 861, 873, 885, 897, 903,
        909, 915, 918)
SLIT_SETTLED = H - 2 * BAR  # 924


def slit_height(f):
    """Visible band height for opening frame f (0 = black, 1..23 = measured fUx4 table)."""
    if f <= 0:
        return 0.0
    if f <= len(SLIT):
        return float(SLIT[f - 1])
    return float(SLIT_SETTLED)


def band(img, height):
    """Keep a vertically centred band of `height` px, black elsewhere, 1 px AA edges."""
    h = img.shape[0]
    top = (h - height) / 2.0
    if abs(height - round(height)) < 1e-6:  # integer band: snap to whole rows so it is exactly `height` rows
        top = math.floor(top + 0.5)
    bot = top + height
    y = np.arange(h, dtype=np.float32) + 0.5
    m = np.clip(y - top + 0.5, 0, 1) * np.clip(bot - y + 0.5, 0, 1)
    out = img.copy()
    full = m >= 0.999
    zero = m <= 0.001
    out[zero] = 0
    edge = ~(full | zero)
    if edge.any():
        out[edge] = (out[edge].astype(np.float32) * m[edge][:, None, None]).astype(np.uint8)
    return out


def slit_open(img, f):
    """T0: frame 0 black, frames 1-23 the centre slit-open, then the 78 px letterbox."""
    return band(img, slit_height(f))


def letterbox(img, bar=BAR):
    return band(img, img.shape[0] - 2 * bar) if bar > 0.01 else img.copy()


def letterbox_retract(img, f, f0=577, f1=592, bar=BAR):
    """Bars 78 -> 0 with ease-in-quad between frames f0 and f1 (fully open on f1)."""
    u = (f - f0) / max(f1 - f0, 1)
    return letterbox(img, bar * (1 - ease_in_quad(u)))


def letterbox_close(img, f, f0=2551, f1=2565, bar=BAR):
    """Bars 0 -> 78 ease-in-out over f0..f1 (the ending bookend)."""
    u = (f - f0) / max(f1 - f0, 1)
    return letterbox(img, bar * ease_in_out(u))


# ----------------------------------------------------------------------------- T2 whip
WHIP_OUT = ((0.945, 60.0), (0.924, 110.0))  # (scale, translation px) on b-3, b-2


def _dir(direction):
    L = math.hypot(*direction) or 1.0
    return direction[0] / L, direction[1] / L


def whip_out(a, k, direction=(1.0, -1.0), smear=1.0):
    """T2 out frames (k=0 -> b-3, k=1 -> b-2): zoom out 0.945 / 0.924, slide 60 / 110 px along
    the diagonal, directional blur = displacement x smear."""
    s, d = WHIP_OUT[min(max(k, 0), 1)]
    ux, uy = _dir(direction)
    out = xform(a, s, 0.0, ux * d, uy * d)
    return dir_blur(out, math.degrees(math.atan2(uy, ux)), d * smear)


def whip_in(b, direction=(1.0, -1.0), s=1.28, blur=40.0):
    """T2 b-1: incoming shot at 1.28x with a 40 px directional blur in the whip direction."""
    ux, uy = _dir(direction)
    out = xform(b, s)
    return dir_blur(out, math.degrees(math.atan2(uy, ux)), blur)


def t2_frame(a, b, n, direction=(1.0, -1.0), s0=1.22, drift=40.0, smear=1.0, center=(0.5, 0.5)):
    """Whole T2 whip cut as one call. n = frame relative to the beat: n <= -4 returns `a`
    untouched, -3/-2 whip-out of a, -1 whip-in of b, n >= 0 the sharp Z1 entry punch of b.
    Pass the frames that play at that output frame (a = outgoing shot, b = incoming shot)."""
    if n <= -4:
        return a.copy()
    if n <= -2:
        return whip_out(a, n + 3, direction, smear)
    if n == -1:
        return whip_in(b, direction)
    return entry_punch(b, n, s0, drift, direction, center=center)


# ----------------------------------------------------------------------------- T3 radial
T3_BLUR = (90.0, 200.0)  # see radial_collapse: the bible's 30-50 px reads as no blur at all


def radial_collapse(img, k, frames=2, s_end=1.3, blur_px=T3_BLUR, center=(0.5, 0.5), s_from=1.0):
    """T3 out: k = 0..frames-1, scale s_from -> s_end (capped at MAX_SCALE) with radial zoom blur
    blur_px[0] -> blur_px[1] (streak length at the frame side edge) from the subject centre.
    Pass the *unscaled* frame and s_from = the scale it was already at (e.g. 1.287 after a Z2
    push) so the scales do not compound. Default blur is stronger than the bible's 30-50 px:
    rendered at 30-50 px the streaks were invisible on the demo, while the q-DyE 17.867 / 18.533
    reference frames show streaks across most of the frame; pass blur_px=(30, 50) for the
    bible value."""
    u = (k + 1) / frames
    s = min(s_from + (s_end - s_from) * u, MAX_SCALE) if s_end > s_from else s_from
    px = blur_px[0] + (blur_px[1] - blur_px[0]) * (k / max(frames - 1, 1))
    return radial_blur(xform(img, s, center=center), px, center)


def radial_enter(img, n, s0=1.25, blur_px=T3_BLUR[0], center=(0.5, 0.5), tau=4.0):
    """T3 in (e.g. the name plate on f1431): enters at s0 with radial blur, decays per Z1;
    blur only on n = 0 and n = 1 (40%). Bible says 30 px; default is T3_BLUR[0] for the
    same visibility reason as radial_collapse."""
    s = punch_scale(n, s0, tau)
    out = xform(img, s, center=center)
    px = blur_px if n <= 0 else (blur_px * 0.4 if n == 1 else 0.0)
    return radial_blur(out, px, center) if px > 0 else out


# ----------------------------------------------------------------------------- T4 blur-dissolve
def _defocus(img, sigma):
    if sigma < 0.4:
        return img.copy()
    h, w = img.shape[:2]
    if sigma < 6:
        sm = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.GaussianBlur(sm, (0, 0), sigma / 2), (w, h), interpolation=cv2.INTER_LINEAR)
    sm = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(sm, (0, 0), sigma / 4), (w, h), interpolation=cv2.INTER_LINEAR)


def blur_dissolve(a, b, k, frames=14, sigma_peak=18.0, luma_from=17.0, luma_to=140.0):
    """T4 (18.093 only): defocus peaks mid-way (sigma 18), A dissolves into B, mean luma ramps
    linearly luma_from -> luma_to. Sharp again on the last frame."""
    u = k / max(frames - 1, 1)
    sigma = sigma_peak * math.sin(math.pi * u)
    mix = ease_in_out((u - 0.2) / 0.6)
    img = cv2.addWeighted(a, 1 - mix, b, mix, 0) if 0 < mix < 1 else (a if mix <= 0 else b)
    img = _defocus(img, sigma)
    return match_luma(img, luma_from + (luma_to - luma_from) * u)


def cross_dissolve(a, b, k, frames=7):
    """T14 (A6 only): plain 6-9 frame cross-dissolve. k = 0..frames-1."""
    u = (k + 1) / (frames + 1)
    return cv2.addWeighted(a, 1 - u, b, u, 0)


# ----------------------------------------------------------------------------- T5 hue flash
def hue_flash(img, k, wash=40):
    """T5: k = 0, 1, 2 -> hue +60, +120, +180 deg with a +40 luma wash. Then cut with T3."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    hsv[..., 0] = ((h + 30 * (min(max(k, 0), 2) + 1)) % 180).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.add(out, np.full_like(out, wash))


# ----------------------------------------------------------------------------- T8 glitch
@lru_cache(maxsize=4)
def _grid(h, w):
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return gx, gy


def glitch_wave(img, k, amp=40.0, wavelength=22.0, split=12.0, dark=0.25, blown=200.0, ca=8.0, seed=0):
    """T8 (f1429-1430 only). k=0: horizontal sine wave-warp (40 px, 22 px wavelength, phase
    stepping per row band) + RGB split R +12 / B -12 on the frame darkened to 25%.
    k=1: blown-out frame at mean luma ~200 with 8 px chromatic aberration."""
    h, w = img.shape[:2]
    if k >= 1:
        return chroma(chroma(match_luma(img, blown), ca), ca * 0.6, mode="radial")
    rng = np.random.default_rng(seed + 991)
    y = np.arange(h, dtype=np.float32)
    phase = np.repeat(rng.uniform(0, 6.283, h // 24 + 1), 24)[:h].astype(np.float32)  # steps per row band
    env = np.repeat(rng.uniform(0.35, 1.0, h // 40 + 1), 40)[:h].astype(np.float32)
    disp = amp * env * np.sin(2 * np.pi * y / wavelength + phase)
    gx, gy = _grid(h, w)
    src = cv2.convertScaleAbs(img, alpha=dark)
    b_, g_, r_ = cv2.split(src)
    mx = gx + disp[:, None]
    out = cv2.merge([cv2.remap(b_, mx + split, gy, cv2.INTER_LINEAR, borderMode=REFLECT),
                     cv2.remap(g_, mx, gy, cv2.INTER_LINEAR, borderMode=REFLECT),
                     cv2.remap(r_, mx - split, gy, cv2.INTER_LINEAR, borderMode=REFLECT)])
    return out


# ----------------------------------------------------------------------------- T9 iris
@lru_cache(maxsize=2)
def _rinnegan(size=1400):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    R = size / 2 * 0.98
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / R
    k = np.clip(r, 0, 1)[..., None]
    col = np.float32((225, 185, 200)) * (1 - k ** 1.6) + np.float32((150, 95, 125)) * k ** 1.6  # BGR lavender
    ink = np.zeros_like(r)
    lw = 0.016
    for rk in (0.30, 0.48, 0.66, 0.84):
        ink = np.maximum(ink, np.clip(1 - np.abs(r - rk) * R / (lw * R), 0, 1))
    ink = np.maximum(ink, np.clip((0.115 - r) * R, 0, 1))          # pupil
    ink = np.maximum(ink, np.clip(1 - np.abs(r - 0.975) * R / (0.03 * R), 0, 1))  # rim
    out = col * (1 - ink[..., None]) + np.float32((40, 18, 34)) * ink[..., None]
    hx, hy = c - R * 0.38, c - R * 0.42
    sp = np.exp(-(((xx - hx) / (R * 0.16)) ** 2 + ((yy - hy) / (R * 0.10)) ** 2))
    out = out + (sp * 0.7)[..., None] * (255 - out)
    shade = 1 - 0.3 * np.clip((r - 0.7) / 0.3, 0, 1)
    out = out * shade[..., None]
    alpha = np.clip((1 - r) * R, 0, 1)
    return np.dstack([np.clip(out, 0, 255), alpha * 255]).astype(np.uint8)


def _eye(kind):
    if kind == "rinnegan":
        return _rinnegan()
    return V._eye_cached(kind)


def iris(a, b, k, frames=9, center=(0.5, 0.5), center_b=(0.5, 0.5), eye="3tomoe", turns=1.0, zoom_a=True):
    """T9: the eye graphic (3tomoe | kakashi_mangekyo | obito_mangekyo | rinnegan) grows out of
    the real eye at `center` in A, the camera dives through the pupil, B opens from inside it.
    Adapted from vfx2.iris_transition (which only knows Sharingan kinds). k = 0..frames-1.
    A's dolly is capped at MAX_SCALE so the source never goes soft past the bible limit."""
    h, w = a.shape[:2]
    t = (k + 1) / frames
    diag = math.hypot(w, h)
    cA = (center[0] * w, center[1] * h)
    cB = (center_b[0] * w, center_b[1] * h)
    kk = V._ss(0.25, 0.95, t)
    cx, cy = cA[0] + (cB[0] - cA[0]) * kk, cA[1] + (cB[1] - cA[1]) * kk
    xx, yy = _grid(h, w)
    r0, r1 = h * 0.035, diag * 0.62 / 0.21
    rg = r0 * (r1 / r0) ** (t ** 1.6)
    grow = rg / r0
    sa = min(1 + (grow - 1) * 0.08, MAX_SCALE)
    if zoom_a:
        _note(sa)
        A = V._scale_about(a, sa, cA[0], cA[1], (cx, cy))
    else:
        A = a
    fade = V._ss(0.0, 0.12, t)
    eye_img = _eye(eye)
    es = eye_img.shape[0]
    rot = 360.0 * turns * (t ** 0.7)
    s = rg / (es / 2 * 0.98)
    M = cv2.getRotationMatrix2D((es / 2, es / 2), -rot, s)
    M[0, 2] += cx - es / 2
    M[1, 2] += cy - es / 2
    g = cv2.warpAffine(eye_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ga = g[..., 3].astype(np.float32) * (fade / 255.0)
    out = cv2.blendLinear(A, np.ascontiguousarray(g[..., :3]), 1 - ga, ga)
    rp = rg * 0.21 if eye != "rinnegan" else rg * 0.115
    reveal = V._ss(h * 0.06, h * 0.32, rp)
    if reveal > 0:
        sb = 1.30 - 0.30 * V._ss(0.5, 1.0, t)
        _note(sb)
        B = V._scale_about(b, sb, cB[0], cB[1], (cx, cy))
        if rp > diag:
            return B
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        m = np.clip(rp - dist, 0, 1) * reveal
        out = cv2.blendLinear(out, B, 1 - m, m)
    return out


# ----------------------------------------------------------------------------- T10 strobe
def strobe(img, k, jolt=6.0, seed=0):
    """T10: per beat a 5-frame burst B/I/B/I/B (k = 0..4); image frames jolted 6 px."""
    if k % 2 == 0:
        return np.zeros_like(img)
    rng = np.random.default_rng(seed * 31 + k)
    a = rng.uniform(0, 2 * math.pi)
    dx, dy = jolt * math.cos(a), jolt * math.sin(a)
    return xform(img, cover_scale(0, dx, dy), 0, dx, dy)


# ----------------------------------------------------------------------------- T11 PIP
@lru_cache(maxsize=8)
def _rrect_mask(w, h, r, ss=3):
    m = np.zeros((h * ss, w * ss), np.uint8)
    R = r * ss
    cv2.rectangle(m, (R, 0), (w * ss - R - 1, h * ss - 1), 255, -1)
    cv2.rectangle(m, (0, R), (w * ss - 1, h * ss - R - 1), 255, -1)
    for cx, cy in ((R, R), (w * ss - R - 1, R), (R, h * ss - R - 1), (w * ss - R - 1, h * ss - R - 1)):
        cv2.circle(m, (cx, cy), R, 255, -1, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)


def _cover_crop(img, aspect, focus=(0.5, 0.5)):
    h, w = img.shape[:2]
    if w / h > aspect:
        cw, ch = int(round(h * aspect)), h
    else:
        cw, ch = w, int(round(w / aspect))
    x0 = int(round(min(max(focus[0] * w - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(focus[1] * h - ch / 2, 0), h - ch)))
    return img[y0:y0 + ch, x0:x0 + cw]


def _pip_tile(inner, pw, ph, radius, focus, stroke=3, stroke_col=(26, 22, 24)):
    crop = _cover_crop(inner, pw / ph, focus)
    tile = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA)
    m = _rrect_mask(pw, ph, radius)
    inner_m = cv2.erode(m, np.ones((2 * stroke + 1, 2 * stroke + 1), np.uint8))
    a = inner_m.astype(np.float32) / 255
    col = np.empty_like(tile)
    col[:] = stroke_col
    tile = cv2.blendLinear(tile, col, a, 1 - a)
    return np.dstack([tile, m])


def pip(bg, inner, k, frames=15, grow_frames=7, width=0.40, aspect=2.3, radius=28, rest=(0.5, 0.62),
        focus=(0.5, 0.45), grow=0.40, wobble=2.0):
    """T11 rise: a rounded rectangle (40% W, 2.3:1, r 28) showing `inner` rises from the bottom
    over `frames` (easeOutBack, +-2 deg wobble); over the last `grow_frames` it grows +40% while
    the background defocuses (the zoom-through starts). k = 0..frames-1; at the beat call
    pip_arrive(inner, n)."""
    h, w = bg.shape[:2]
    pw, ph = int(round(w * width)), int(round(w * width / aspect))
    u = (k + 1) / frames
    e = ease_out_back(min(1.0, (k + 1) / max(frames - grow_frames + 2, 1)))
    y0 = h + ph * 0.6
    cy = y0 + (rest[1] * h - y0) * e
    cx = rest[0] * w
    gu = _clip01((k - (frames - grow_frames) + 1) / grow_frames)
    sc = 1.0 + grow * gu ** 2
    rot = wobble * math.sin(k * 0.95 + 0.6) * math.exp(-k / 6.0)
    tile = _pip_tile(inner, pw, ph, radius, focus)
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), rot, sc)
    M[0, 2] += cx - pw / 2
    M[1, 2] += cy - ph / 2
    lay = cv2.warpAffine(tile, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    base = _defocus(bg, 7.0 * gu) if gu > 0 else bg
    # soft drop shadow
    sh = cv2.GaussianBlur(cv2.resize(lay[..., 3], (w // 4, h // 4), interpolation=cv2.INTER_AREA), (0, 0), 4)
    sh = cv2.resize(sh, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    sh = np.roll(sh, (10, 6), axis=(0, 1)) * 0.45
    base = cv2.blendLinear(base, np.zeros_like(base), 1 - sh, sh)
    a = lay[..., 3].astype(np.float32) / 255
    return cv2.blendLinear(np.ascontiguousarray(lay[..., :3]), base, a, 1 - a)


def double_exposure(img, u, ghost_scale=1.22, opacity=0.5, center=(0.5, 0.45)):
    """Same shot overlaid on itself at ghost_scale (q-DyE 13.6-13.9 double-eye morph)."""
    if u <= 0:
        return img.copy()
    g = V._scale_about(img, ghost_scale, center[0] * img.shape[1], center[1] * img.shape[0])
    a = opacity * u
    return _screen(cv2.addWeighted(img, 1 - a * 0.5, g, a * 0.5, 0), cv2.convertScaleAbs(g, alpha=a * 0.35))


def pip_arrive(inner, n, s0=HOLD_CAP, blur_px=60.0, morph=3, center=(0.5, 0.45)):
    """T11 arrival at the beat: full screen at 1.12 (hold cap) with a radial blur on n=0,
    decaying per Z1, plus a `morph`-frame double exposure (n = 0..morph-1)."""
    out = radial_enter(inner, n, s0, blur_px, center)
    if 0 <= n < morph:
        out = double_exposure(out, 1 - n / morph, center=center)
    return out


# ----------------------------------------------------------------------------- T12 neon + ink
def neon_edges(img, lo=45, hi=120, core=(255, 255, 255), glow=PALE_BLUE):
    """T12 frame 1: inverted Canny edges glowing white / pale blue on black (MBoD 28.33)."""
    h, w = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                                      cv2.COLOR_BGR2GRAY), (0, 0), 1.3)
    e = cv2.Canny(g, lo, hi).astype(np.float32) / 255
    g1 = cv2.GaussianBlur(e, (0, 0), 2.5)
    g2 = cv2.GaussianBlur(cv2.resize(e, (w // 8, h // 8), interpolation=cv2.INTER_AREA), (0, 0), 3)
    g2 = cv2.resize(g2, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    lay = e[..., None] * np.float32(core) + (g1 * 2.2 + g2 * 5.0)[..., None] * np.float32(glow)
    return cv2.resize(np.clip(lay, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=8)
def _ink_burst_field(h, w, seed, center):
    rng = np.random.default_rng(seed + 4242)
    cx, cy = center[0] * w, center[1] * h
    m = np.zeros((h, w), np.float32)
    for _ in range(70):  # ink seeds: small blots scattered around the centre
        a = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(0.08, 0.45) * w
        cv2.circle(m, (int(cx + r0 * math.cos(a)), int(cy + r0 * math.sin(a) * 0.8)), int(rng.uniform(1, 5)),
                   float(rng.uniform(0.6, 1.0)), -1, cv2.LINE_AA)
    acc = m
    for kk in range(7):  # log-step radial smear outward: blots become tapered spokes
        s = 1 + 1.2 * 2 ** kk / 2 ** 6
        M = np.float32([[1 / s, 0, cx * (1 - 1 / s)], [0, 1 / s, cy * (1 - 1 / s)]])
        acc = np.maximum(acc, cv2.warpAffine(acc, M, (w, h), flags=cv2.INTER_LINEAR) * 0.9)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / w) ** 2)
    acc = acc * np.clip((r - 0.06) / 0.10, 0, 1)  # clean white core like the reference
    return np.clip(cv2.GaussianBlur(acc, (0, 0), 0.7) * 1.1, 0, 1)


def ink_burst(b, seed=0, center=(0.5, 0.5), ghost=0.12, paper=(240, 243, 245), ink=(20, 16, 18)):
    """T12 frame 2: white paper with black ink spokes bursting from the centre (MBoD 28.37),
    a faint ghost of the incoming shot underneath."""
    h, w = b.shape[:2]
    f = _ink_burst_field(h // 4, w // 4, seed, (round(center[0], 3), round(center[1], 3)))
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
    pap = np.empty_like(b)
    pap[:] = paper
    pap = cv2.addWeighted(pap, 1 - ghost, b, ghost, 0)
    col = np.empty_like(b)
    col[:] = ink
    return cv2.blendLinear(col, pap, f, 1 - f)


def settle(b, k, frames=4, s0=1.12, blur=(0.30, 0.16, 0.07, 0.02), center=(0.5, 0.5)):
    """T12 frames 3-6: the incoming shot settles out of a heavy zoom blur (k = 0..frames-1)."""
    s = 1.0 + (s0 - 1.0) * (1 - k / frames) ** 2
    out = xform(b, s, center=center)
    st = blur[k] if k < len(blur) else 0.0
    return S.zoom_blur(out, st, center) if st > 0 else out


def t12_frame(a, b, k, seed=0, center=(0.5, 0.5)):
    """Whole T12 stack: k=0 neon edges of a, k=1 ink burst over b, k=2..5 settle of b."""
    if k <= 0:
        return neon_edges(a)
    if k == 1:
        return ink_burst(b, seed, center)
    return settle(b, k - 2, center=center)


# ----------------------------------------------------------------------------- T13 3D plane
def _plane_H(w, h, roll, tilt_y, tilt_x=0.0, focal=1.25):
    f = focal * w
    ry, rx, rz = math.radians(tilt_y), math.radians(tilt_x), math.radians(roll)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = []
    for x, y in src:
        p = R @ np.array([x - w / 2, y - h / 2, 0.0])
        z = f + p[2]
        dst.append([w / 2 + f * p[0] / z, h / 2 + f * p[1] / z])
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def _scaled_H(Hm, s, w, h):
    T = np.array([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2], [0, 0, 1]])
    return T @ Hm


def _H_covers(Hm, w, h, eps=0.5):
    Hi = np.linalg.inv(Hm)
    c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    p = Hi @ c
    p = p[:2] / p[2]
    return bool((p[0] >= -eps).all() and (p[0] <= w + eps).all() and (p[1] >= -eps).all() and (p[1] <= h + eps).all())


def _max_magnification(Hm, w, h, n=5):
    """Largest local linear magnification of output px vs source px over an n x n grid
    (sqrt of the forward Jacobian determinant, from finite differences of the inverse map)."""
    Hi = np.linalg.inv(Hm)
    best = 0.0
    for x in np.linspace(0, w, n):
        for y in np.linspace(0, h, n):
            pts = np.array([[x, y, 1], [x + 1, y, 1], [x, y + 1, 1]], np.float64).T
            q = Hi @ pts
            q = q[:2] / q[2]
            J = np.column_stack([q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]])
            best = max(best, 1.0 / math.sqrt(abs(np.linalg.det(J)) + 1e-12))
    return best


def tilt_plane(img, u=0.0, roll=(3.0, 6.0), tilt=(6.0, 10.0), tilt_x=0.0, fill="mirror", cap=HOLD_CAP,
               dark=0.40, focal=1.25):
    """T13: footage as a plane with roll and Y-tilt, swinging from (roll[0], tilt[0]) to
    (roll[1], tilt[1]) over shot progress u. fill='overscale' scales until no corner shows
    (needs ~1.2-1.35 local magnification, over the hold cap); fill='mirror' keeps the largest
    local magnification <= `cap` and fills what is left with a 40%-darkened mirrored edge
    extension. Subject is never cut."""
    h, w = img.shape[:2]
    e = ease_in_out_sine(u)
    r = roll[0] + (roll[1] - roll[0]) * e
    ty = tilt[0] + (tilt[1] - tilt[0]) * e
    Hm = _plane_H(w, h, r, ty, tilt_x, focal)
    lo, hi = 0.5, 3.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _H_covers(_scaled_H(Hm, mid, w, h), w, h):
            hi = mid
        else:
            lo = mid
    if fill == "overscale":
        s = hi
    else:  # cap the *local* magnification (near side of the tilt is magnified more than s)
        s = min(hi, cap / max(_max_magnification(Hm, w, h), 1e-6))
    Hs = _scaled_H(Hm, s, w, h)
    _note(_max_magnification(Hs, w, h))
    out = cv2.warpPerspective(img, Hs, (w, h), flags=cv2.INTER_CUBIC, borderMode=REFLECT)
    if fill == "overscale" or s >= hi - 1e-4:
        return out
    m = cv2.warpPerspective(np.full((h // 2, w // 2), 255, np.uint8),
                            np.diag([0.5, 0.5, 1.0]) @ Hs @ np.diag([2.0, 2.0, 1.0]), (w // 2, h // 2),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    g = (1 - dark) + dark * m
    return cv2.blendLinear(out, np.zeros_like(out), g, 1 - g)


# ----------------------------------------------------------------------------- speed ramps
def conform_indices(n_out, src_fps=24000 / 1001, fps=FPS, start=0):
    """23.976 -> 30 conform by nearest source frame (no blending)."""
    return [int(start + math.floor(k * src_fps / fps + 0.5)) for k in range(n_out)]


def hit_curve(n_out, n_fast, n_crawl, fast=1.5, crawl=0.15, out=3.0):
    """Bible velocity: action at `fast` (1.3-1.6x), crawl at 0.15x for `n_crawl` (3-6) frames on
    the key pose, then whip out at `out` for the rest. Returns motion.Curve keys (u, speed)."""
    def u(k):
        return k / n_out
    a, b = n_fast, n_fast + n_crawl
    keys = [(0.0, fast), (u(a - 1), fast), (u(a), crawl), (u(b - 1), crawl)]
    if b < n_out:
        keys += [(u(b), out), (1.0, out)]
    else:
        keys += [(1.0, crawl)]
    # strictly increasing u
    clean = []
    for k in keys:
        if not clean or k[0] > clean[-1][0] + 1e-9:
            clean.append(k)
    return clean


def ramp_frames_needed(n_out, n_fast, n_crawl, **kw):
    return MO.source_frames_needed(n_out, hit_curve(n_out, n_fast, n_crawl, **kw))


def speed_ramp(frames, n_out, n_fast, n_crawl, fast=1.5, crawl=0.15, out=3.0, interp="flow", blur_from=2.0,
               shutter=0.5, src_scale=24000 / 1001 / FPS):
    """lib2.motion RampClip on the bible velocity curve. Speeds are in source frames per output
    frame at 30 fps, so 1.0x real time for 23.976 footage is src_scale (0.799) -- the curve is
    multiplied by it. Motion blur only above `blur_from` (the whip-out), so held action stays sharp."""
    keys = [(uu, v * src_scale) for uu, v in hit_curve(n_out, n_fast, n_crawl, fast, crawl, out)]
    return MO.RampClip(frames, n_out, keys, interp=interp, shutter=shutter, blur_from=blur_from * src_scale)


# ----------------------------------------------------------------------------- grades
def _curve(points):
    xs, ys = zip(*points)
    return np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.float32)


@lru_cache(maxsize=1)
def _bases():
    I = np.arange(256, dtype=np.float32)
    sc = _curve([(0, 0), (40, 30), (128, 128), (215, 228), (255, 255)])
    to = S._luts()["teal_orange"]
    return {  # (B, G, R) float LUTs
        "neutral": (I, I, I),
        "night": (_curve([(0, 14), (60, 70), (128, 140), (255, 250)]), _curve([(0, 6), (60, 58), (128, 124), (255, 246)]),
                  _curve([(0, 0), (60, 44), (128, 108), (255, 238)])),
        "amber": (_curve([(0, 0), (90, 70), (180, 160), (255, 225)]), _curve([(0, 2), (90, 88), (180, 182), (255, 248)]),
                  _curve([(0, 8), (90, 108), (180, 205), (255, 255)])),
        "contrast": (sc, sc, sc),
        "faded": (_curve([(0, 20), (128, 116), (255, 228)]), _curve([(0, 22), (128, 124), (255, 236)]),
                  _curve([(0, 26), (128, 134), (255, 242)])),
        "teal_orange": tuple(x.astype(np.float32) for x in to),
        "violet": (_curve([(0, 18), (128, 150), (255, 255)]), _curve([(0, 2), (128, 120), (255, 240)]),
                   _curve([(0, 4), (128, 112), (255, 236)])),
        "sepia": (_curve([(0, 10), (128, 104), (255, 222)]), _curve([(0, 12), (128, 124), (255, 242)]),
                  _curve([(0, 18), (128, 144), (255, 255)])),
        "golden": (_curve([(0, 4), (128, 104), (255, 225)]), _curve([(0, 4), (128, 126), (255, 245)]),
                   _curve([(0, 8), (128, 146), (255, 255)])),
    }


# luma / sat (HSV 0-255) / R-B ranges from bible sec. 8; extras applied after the fit.
ACTS = {
    "A1": dict(luma=(45, 75), sat=(65, 100), rb=(-25, -10), base="night", lift=12, vignette=0.25, skin=0.7),
    "A2_iruka": dict(luma=(55, 80), sat=(70, 100), rb=(-15, -6), base="night", lift=10, vignette=0.2, skin=0.7),
    "A2_ramen": dict(luma=(85, 110), sat=(80, 100), rb=(15, 25), base="amber", lift=4, vignette=0.15),
    "A3_hot": dict(luma=(98, 115), sat=(90, 110), rb=(22, 28), base="contrast", halation=0.5),
    "A3_cool": dict(luma=(98, 115), sat=(90, 110), rb=(-13, -11), base="contrast", halation=0.5, skin=0.5),
    "NAME": dict(luma=(132, 148), sat=(90, 110), rb=(40, 50), base="contrast", halation=0.4),
    "A4": dict(luma=(80, 100), sat=(45, 60), rb=(6, 16), base="faded", keep_red=True),
    "A5": dict(luma=(95, 120), sat=(115, 140), rb=(-5, 15), base="teal_orange", halation=0.5),
    # white-gold bloom shots are bright by nature (bible sec. 8): luma ceiling raised to 175
    "A5_gold": dict(luma=(105, 175), sat=(100, 140), rb=(25, 40), base="golden", bloom=0.5),
    "A5_purple": dict(luma=(95, 115), sat=(115, 140), rb=(-25, -8), base="violet", halation=0.3, skin=0.5),
    "A6": dict(luma=(110, 140), sat=(80, 95), rb=(25, 35), base="sepia", bloom=0.3),
    "A7": dict(luma=(120, 150), sat=(80, 110), rb=(20, 30), base="golden", god_rays=0.5, bloom=0.25),
}


def _lut3(base, gamma, d, lift):
    B, G, R = base
    out = []
    for ch, off in ((B, -d / 2), (G, 0.0), (R, d / 2)):
        y = 255.0 * (ch / 255.0) ** gamma
        # mid-weighted offset: blacks/whites shift only 15%, so a big R-B correction never
        # paints the shadows blue (or the highlights pink)
        y = y + off * (0.15 + 0.85 * np.maximum(np.sin(np.pi * np.clip(y, 0, 255) / 255.0), 0.0) ** 0.7)
        y = lift + np.clip(y, 0, 255) * (255.0 - lift) / 255.0
        out.append(np.clip(y, 0, 255))
    return np.dstack(out).astype(np.uint8)  # (1, 256, 3)


def _skin_mask(img):
    h, w = img.shape[:2]
    sm = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.clip(1 - np.abs(hue - 14.0) / 9.0, 0, 1) * np.clip((s - 30) / 30, 0, 1) * np.clip((215 - s) / 40, 0, 1) \
        * np.clip((v - 70) / 40, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def _red_mask(img):
    h, w = img.shape[:2]
    sm = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype(np.float32)
    hd = np.minimum(hsv[..., 0], 180 - hsv[..., 0])
    m = np.clip((10 - hd) / 5, 0, 1) * np.clip((hsv[..., 1] - 110) / 50, 0, 1) * np.clip((hsv[..., 2] - 60) / 40, 0, 1)
    return cv2.resize(cv2.GaussianBlur(m, (0, 0), 1.2), (w, h), interpolation=cv2.INTER_LINEAR)


def _sat(img, k):
    if abs(k - 1) < 1e-3:
        return img
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, k, g, 1 - k, 0)


def apply_grade(img, p, extras=True, t=0.0):
    """Apply a fitted grade (from fit_grade) to a frame. extras = halation/bloom/god rays."""
    cfg = ACTS[p["act"]]
    base = _bases()[cfg["base"]]
    amt = p.get("amt", 1.0)
    if amt < 0.999:
        ident = _bases()["neutral"]
        base = tuple(i + (b - i) * amt for b, i in zip(base, ident))
    out = cv2.LUT(img, _lut3(base, p["gamma"], p["d"], cfg.get("lift", 0)))
    out = _sat(out, p["k"])
    if cfg.get("skin"):
        warm = cv2.LUT(img, _lut3(_bases()["neutral"], p["gamma"], 0.0, cfg.get("lift", 0)))
        m = _skin_mask(img) * cfg["skin"]
        out = cv2.blendLinear(warm, out, m, 1 - m)
    if cfg.get("keep_red"):
        m = _red_mask(img)
        vivid = _sat(cv2.LUT(img, _lut3(_bases()["neutral"], p["gamma"], 0.0, 0)), 1.15)
        out = cv2.blendLinear(vivid, out, m, 1 - m)
    if cfg.get("vignette"):
        out = S.vignette(out, cfg["vignette"])
    if extras:
        if cfg.get("halation"):
            out = V.halation(out, cfg["halation"], threshold=200)
        if cfg.get("bloom"):
            out = S.bloom(out, 185, cfg["bloom"], 1.4, GOLD_WHITE)
        if cfg.get("god_rays"):
            out = V.god_rays(out, center=p.get("rays_center", (0.5, 0.2)), strength=cfg["god_rays"], tint=GOLD_WHITE)
    return out


SAT_K_MAX = 1.8  # saturation gain cap: past this flat anime colour posterises
D_MAX = 90.0     # R-B correction cap; past it the source fights the look and we accept a miss


def fit_grade(img, act, iters=6, rays_center=(0.5, 0.2)):
    """Fit gamma / R-B offset / saturation so the graded frame lands inside the act's bible
    ranges. Fit once per shot (e.g. on its middle frame) and apply to every frame of the shot
    so the grade never flickers. Returns params incl. the achieved stats ('got').
    Deliberate deviations: a warm act never makes a frame cooler than it already was (the R-B
    upper bound becomes max(hi, raw R-B)), a cool act never makes it warmer; saturation gain
    is capped at SAT_K_MAX, so a very desaturated source can land below the sat range."""
    cfg = ACTS[act]
    th = _thumb(img, 480)
    p = dict(act=act, gamma=1.0, d=0.0, k=1.0, rays_center=rays_center)
    raw_rb = stats(th)["rb"]
    lo, hi = cfg["rb"]
    if lo > 0:
        rb_rng = (lo, max(hi, raw_rb))
    elif hi < 0:
        rb_rng = (min(lo, raw_rb), hi)
    else:
        rb_rng = (lo, hi)

    def clampr(v, r):
        return min(max(v, r[0]), r[1])

    for _ in range(iters):
        # luma via gamma (bisection in log space)
        st = stats(apply_grade(th, p, extras=False))
        tgt = clampr(st["luma"], cfg["luma"])
        if abs(tgt - st["luma"]) > 0.5:
            lo, hi = math.log(0.4), math.log(3.0)
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                p["gamma"] = math.exp(mid)
                if stats(apply_grade(th, p, extras=False))["luma"] > tgt:
                    lo = mid
                else:
                    hi = mid
            p["gamma"] = math.exp(0.5 * (lo + hi))
        st = stats(apply_grade(th, p, extras=False))
        tgt = clampr(st["rb"], rb_rng)
        want = p["d"] + (tgt - st["rb"]) * 1.1
        if abs(want) > D_MAX and p.get("amt", 1.0) > 0.2:  # the base look itself fights the source:
            p["amt"] = p.get("amt", 1.0) * 0.6            # weaken it instead of piling on correction
        p["d"] = float(np.clip(want, -D_MAX, D_MAX))
        st = stats(apply_grade(th, p, extras=False))
        tgt = clampr(st["sat"], cfg["sat"])
        if st["sat"] > 1:
            p["k"] = float(np.clip(p["k"] * (tgt / st["sat"]) ** 1.1, 0.1, SAT_K_MAX))
    p["got"] = stats(apply_grade(th, p, extras=False))
    return p


def grade(img, act, fit=None, extras=True):
    """Convenience: fit (if not given) + apply."""
    return apply_grade(img, fit or fit_grade(img, act), extras)


# ----------------------------------------------------------------------------- overlays (presets)
def embers(img, t, density=0.6, seed=0, **kw):
    """Rising embers (fire / Kurama shots, name plate at density 0.6)."""
    return V.embers(img, t, density=density, seed=seed, **kw)


def dust(img, t, density=0.8, seed=0, **kw):
    return V.dust(img, t, density=density, seed=seed, **kw)


def raikiri(img, t, origin, target=None, seed=0, purple=False, **kw):
    """Chidori / Kirin lightning from `origin` (normalised). purple=True for Sasuke's later arcs."""
    col = (255, 90, 190) if purple else (255, 205, 120)
    return V.raikiri(img, t, origin=origin, target=target, seed=seed, color=col, hold=2, **kw)


def electric_rim(img, t, mask=None, strength=0.6, seed=0):
    return V.electric_rim(img, t, mask=mask, color=(150, 225, 255), strength=min(strength, 0.6), seed=seed)


def chakra_aura(img, t, mask, strength=0.6, seed=0, color=(80, 190, 255)):
    """KCM / Six Paths gold aura off a subject mask (strength capped at 0.6 per bible)."""
    return V.chakra_aura(img, t, mask, color=color, strength=min(strength, 0.6), seed=seed)


def glow_mask(img, thr=200):
    """Cheap subject mask for glowing KCM bodies (bright, warm) when no matte exists."""
    h, w = img.shape[:2]
    sm = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    m = (cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY) > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def halation(img, strength=0.5, threshold=200):
    return V.halation(img, strength, threshold)


def god_rays(img, center=(0.5, 0.2), strength=0.5):
    return V.god_rays(img, center=center, strength=strength, tint=GOLD_WHITE)


def lens_dirt(img, strength=0.3, seed=0):
    """Flash frames only, strength <= 0.3."""
    return V.lens_dirt(img, min(strength, 0.3), seed)


def eye_glint(img, pos, strength=1.0, length=0.55, color=(90, 190, 255), t=0.0):
    """Anamorphic light streak + core flare on an eye (fUx4 56.6 / 57.9 look). pos normalised.
    Not in the bible's overlay list: use only on the 'Jane!' eye close-up if the lead approves."""
    h, w = img.shape[:2]
    sw, sh = w // 4, h // 4
    xx, yy = _grid(sh, sw)
    px, py = pos[0] * sw, pos[1] * sh
    fl = 0.85 + 0.15 * math.sin(t * 37.0)
    streak = np.exp(-((yy - py) / 1.2) ** 2) * np.exp(-np.abs(xx - px) / (length * sw * 0.35))
    halo = np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * 7.0 ** 2)))
    core = np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.6 ** 2)))
    lay = (streak * 1.3 + halo * 0.8)[..., None] * np.float32(color) + core[..., None] * 255.0
    lay = cv2.GaussianBlur(lay, (0, 0), 0.8) * strength * fl
    lay = cv2.resize(np.clip(lay, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    return _screen(img, lay)


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("annotations", "math", "sys", "cv2", "np",
                                                                   "lru_cache", "S", "V", "MO")]
