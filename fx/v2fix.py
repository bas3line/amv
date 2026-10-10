"""v2 QA fixes: mid-luma ink frame (replaces dark_neg in the impact packages), eye light-trail bolts
(G3GC 10.0-10.3), visible RGB split on the impact cuts (dMWE)."""
import math

import cv2
import numpy as np

import fxpack as fx
import refpack_B as RB


def lin_mean(img):
    """Whole-frame mean of linearised luminance (0..1), same measure as the flash counter."""
    sm = cv2.resize(img, (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lin = np.where(sm <= 0.04045, sm / 12.92, ((sm + 0.055) / 1.055) ** 2.4)
    return float((lin @ np.float32([0.0722, 0.7152, 0.2126])).mean())    # BGR weights


def scale_to_lin(img, target):
    """Multiply the frame so its mean linear luminance is ~target (3 Newton-ish steps on the gain)."""
    f = img.astype(np.float32)
    g = 1.0
    for _ in range(3):
        cur = lin_mean(np.clip(f * g, 0, 255).astype(np.uint8))
        if cur <= 1e-4:
            break
        g *= (target / cur) ** (1 / 2.2)
    return np.clip(f * g, 0, 255).astype(np.uint8)


def ink_dim(img, seed=0, center=(0.5, 0.5), target=0.40):
    """The impact package's ink frame, held one more frame at ~0.40 linear luminance (a grey page)
    instead of the near-black dark_neg: keeps the manga beat, drops one bright/dark flash pair."""
    return scale_to_lin(RB.ink(img, seed, center=center), target)


# ----------------------------------------------------------------------------- eye light trails
def _bolt(p0, ang, length, rng, seg=14, jag=0.22):
    """Jagged polyline from p0 heading `ang` (radians) for `length` px."""
    pts = [np.float32(p0)]
    d = np.float32((math.cos(ang), math.sin(ang)))
    nrm = np.float32((-d[1], d[0]))
    step = length / seg
    a = ang
    for i in range(seg):
        a += rng.normal(0, 0.28)
        a = ang + float(np.clip(a - ang, -0.6, 0.6))
        dd = np.float32((math.cos(a), math.sin(a)))
        p = pts[-1] + dd * step * rng.uniform(0.7, 1.3) + nrm * rng.normal(0, jag * step)
        pts.append(p)
    return np.array(pts)


def eye_trails(img, k, eyes, color=(60, 170, 255), seed=0, n_frames=6, scale=1.0):
    """G3GC-style glowing light trails out of the eyes. eyes = [(x, y, dir)] in frame px, dir = -1 (bolt
    heads left) / +1 (right). k = 0..n_frames-1; the bolts re-strike every frame and fade over the
    last two frames. Additive (screen) light over a small area: not a full-frame flash."""
    if k < 0 or k >= n_frames:
        return img
    amp = 1.0 if k < n_frames - 2 else (0.65 if k == n_frames - 2 else 0.35)
    rng = np.random.default_rng(seed * 101 + k * 7)
    h, w = img.shape[:2]
    core = np.zeros((h, w), np.float32)
    for (x, y, dr) in eyes:
        base = math.pi if dr < 0 else 0.0
        tilt = -0.32 * (1 if dr > 0 else -1)          # rises away from the face, as in the reference
        for b in range(2):
            ang = base + tilt + rng.normal(0, 0.12) + (0.0 if b == 0 else 0.25 * (1 if dr > 0 else -1))
            L = rng.uniform(520, 820) * scale * (1.0 if b == 0 else 0.55)
            pts = _bolt((x, y), ang, L, rng)
            cv2.polylines(core, [np.int32(pts)], False, 1.0, int(round(3 * scale)) + 1, cv2.LINE_AA)
            for q in range(2):     # short branches
                i0 = int(rng.integers(3, len(pts) - 4))
                ba = ang + rng.choice((-1, 1)) * rng.uniform(0.4, 0.9)
                bp = _bolt(pts[i0], ba, L * rng.uniform(0.15, 0.3), rng, seg=6)
                cv2.polylines(core, [np.int32(bp)], False, 0.8, max(1, int(round(2 * scale))), cv2.LINE_AA)
        cv2.circle(core, (int(x), int(y)), int(18 * scale), 1.0, -1, cv2.LINE_AA)
    col = np.float32(color) / 255.0
    g1 = cv2.GaussianBlur(core, (0, 0), 5 * scale)
    g2 = cv2.GaussianBlur(core, (0, 0), 18 * scale)
    eye_glow = np.zeros((h, w), np.float32)
    for (x, y, _) in eyes:
        cv2.circle(eye_glow, (int(x), int(y)), int(60 * scale), 1.0, -1)
    eye_glow = cv2.GaussianBlur(eye_glow, (0, 0), 30 * scale)
    light = (core[..., None] * np.float32((0.9, 0.97, 1.0)) * 1.0 +
             (g1 * 1.6)[..., None] * col + (g2 * 1.4 + eye_glow * 0.55)[..., None] * col)
    light = np.clip(light * amp, 0, 1.5)
    f = img.astype(np.float32) / 255.0
    out = 1.0 - (1.0 - f) * (1.0 - np.clip(light, 0, 1))      # screen
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def rgb_hit(img, n, px0=22.0, frames=4, angle=0.0):
    """Visible RGB split decaying over `frames` frames (22 / 16.5 / 11 / 5.5 px), then cropped in a
    hair so the shifted channel never shows a coloured border."""
    if n < 0 or n >= frames:
        return img
    px = px0 * (1 - n / frames)
    out = fx.chroma(img, px, angle=angle)
    s = 1.0 + (2 * px + 4) / img.shape[1]
    return fx.xform(out, s)
