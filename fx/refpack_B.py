"""refpack_B.py -- editing grammar measured frame-by-frame in the "that's what I call an edit"
references q-DyE7MSPEs (Pain arc, 720p30), dMWE_jtVkJU (Naruto, 720p30) and MBoDXrcNS6M
(Blue Lock, 1080p30).

Every function takes / returns uint8 BGR 1920x1080 frames. All randomness is seeded.
`k` = frame index inside an effect (0 = first frame of it); `n` = frame relative to a beat.
Frame counts are given in OUTPUT frames (23.976 fps). The refs are 30 fps; 1-3 frame events are
kept at the same frame count (a single frame is a single frame), longer holds are scaled x0.8.

What the draft lacks (measured, see refgap_B/*.txt): the refs put 4-6 changed frames on EVERY
beat of a chorus (MBoD 22-39 s: 7-12 frame-events/s; the draft's chorus 1: 2-4/s, i.e. only the
cuts themselves). Those extra frames are the impact package, the in-shot smear, colour hits and
white/glow entries below -- all 1-5 frames long, so the held footage stays sharp.
"""
import math, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fxpack as fx  # noqa: E402  (reuse: xform, dir_blur, radial_blur, chroma, bloom, _defocus,
#                       cover_scale, radial_enter, whip_out, from_white, ease_*)

W, H = 1920, 1080


def _f(img):
    return img.astype(np.float32)


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _rng(seed, salt=0):
    return np.random.default_rng(int(seed) * 1000003 + salt)


# ============================================================================ M1 impact package
# MBoD f693-698 (23.10-23.27 s, every accent of the chorus; the same 5-frame shape repeats at
# 25.20, 28.33, 30.37, 31.57, 33.57 ...):
#   693  outgoing shot, zoom-smeared (radial blur, ~1.15x)
#   694  NEW shot as a high-key NEGATIVE: white page, subject in dark lines + coloured glow
#        (blue there; mean luma 200+)
#   695  NEW shot as INK: black/white threshold line art with radial streaks
#   696  NEW shot DARK: near-black negative, only the edges glint (luma ~13)
#   697  NEW shot enters radial-blurred, 698 clean.
# q-DyE f491-495 (16.38-16.52, Pain arc) is the same idea in grey: 1 dark negative + 3 desaturated
# high-contrast radial-blurred frames. The draft has ONE inverted frame (the drop, f473) and no
# other impact frames.

def neg_glow(img, tint=(255, 170, 40), glow=1.0):
    """High-key negative impact frame: inverted luma (white page, dark lines) + tinted glow
    around the subject's edges. tint is BGR (default electric blue, as MBoD f694)."""
    g = _gray(img)
    inv = 255 - g
    inv = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)
    # push toward white page: gamma 0.55 lifts mids, lines stay dark
    page = 255.0 * (inv.astype(np.float32) / 255.0) ** 0.40
    out = np.dstack([page] * 3)
    # glow: dark structures in the negative, blurred, tinted, multiplied in
    dark = (255.0 - page)
    halo = cv2.GaussianBlur(dark, (0, 0), 14) * 1.3 * glow
    t = np.array(tint, np.float32) / 255.0
    out = out - halo[..., None] * (1.0 - t)[None, None, :]
    return _u8(out)


def ink(img, seed=0, streak=140.0, center=(0.5, 0.5)):
    """Black-on-white ink impact frame (MBoD f695 / q-DyE f493): threshold line art + radial
    streaks. Pure B/W."""
    g = _gray(img)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 9)
    solid = (g < np.percentile(g, 22)).astype(np.uint8) * 255  # deepest shadows go solid black
    out = cv2.bitwise_and(edges, 255 - solid)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return fx.radial_blur(out, streak, center)


def dark_neg(img, keep=0.10):
    """Near-black negative (MBoD f696, luma ~13): inverted, crushed, only edges glint."""
    g = _gray(img)
    e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.5), 60, 140)
    e = cv2.GaussianBlur(e, (0, 0), 2.0).astype(np.float32)
    base = (255.0 - g.astype(np.float32)) * keep
    v = base + e * 1.4
    return _u8(np.dstack([v * 0.95, v, v * 1.05]))


def impact_cut(a, b, k, tint=(255, 170, 40), seed=0, center=(0.5, 0.5), short=False):
    """The 5-frame impact transition a -> b. Put k = 1 (the white frame) ON the beat frame:
    k=0 at b-1 (a zoom-smeared), k=1 at b (neg_glow of b), k=2 ink(b), k=3 dark_neg(b),
    k=4 radial entry of b (fxpack.radial_enter n=0), k>=5 clean b (continue radial_enter n=k-4
    if the director wants the Z1 decay).
    short=True (for 1-beat shots, ~10 frames at 23.976): drops the dark frame, so b is
    affected on 3 frames (neg, ink, radial) and clean from k=4."""
    if short and k >= 3:
        k += 1
    if k <= 0:
        return fx.radial_collapse(a, 0, frames=2, s_end=1.18, blur_px=(110.0, 110.0), center=center)
    if k == 1:
        return neg_glow(b, tint)
    if k == 2:
        return ink(b, seed, center=center)
    if k == 3:
        return dark_neg(b)
    if k == 4:
        return fx.radial_enter(b, 0, s0=1.18, blur_px=90.0, center=center)
    return fx.radial_enter(b, k - 4, s0=1.18, blur_px=90.0, center=center)


def mono_hit(img, k, center=(0.5, 0.5)):
    """q-DyE grey variant (f491-495), 4 frames ON a hit inside a held shot (no cut needed):
    k=0 dark negative, k=1..3 desaturated, contrast x1.6, radial blur 120/80/40 px."""
    if k == 0:
        return dark_neg(img, keep=0.22)
    g = _gray(img).astype(np.float32)
    g = (g - 128.0) * 1.6 + 128.0
    out = cv2.cvtColor(_u8(g), cv2.COLOR_GRAY2BGR)
    return fx.radial_blur(fx.xform(out, 1.0 + 0.04 * (4 - k), center=center), (160, 120, 80, 40)[min(k, 3)], center)


# ============================================================================ M2 beat smear
# q-DyE f1055-1069 (35.20-35.67 s): ONE held shot (Naruto vs Pain) gets hit on every beat:
# frame 1 = offset + rotate + directional smear, frame 2 = heavier smear, frame 3 = re-enters
# rotated ~4 deg, then 3-4 clean frames. Measured global jumps between consecutive frames
# (phase-corr, 720p scaled to 1080): 250-600 px on smear frames (q 23.72-24.0 s: -250, 143/307,
# -443, 608 px). The draft holds 2-beat shots dead still between cuts.

def beat_smear(img, n, seed=0, amp=170.0, rot=4.0, blur=110.0):
    """In-shot beat hit. n = frame relative to the beat: n=0 big jolt + smear, n=1 half,
    n=2 small re-entry rotation, n>=3 untouched (sharp). Direction alternates per seed."""
    if n < 0 or n >= 3:
        return img.copy()
    r = _rng(seed, 11)
    ang = float(r.uniform(0, 2 * math.pi))
    sgn = 1 if r.random() < 0.5 else -1
    k = (1.0, 0.45, 0.12)[n]
    dx, dy = amp * k * math.cos(ang), amp * k * math.sin(ang) * 0.6
    rt = sgn * rot * (1.0, 0.6, 0.35)[n]
    s = fx.cover_scale(rt, dx, dy)
    out = fx.xform(img, s, rt, dx, dy)
    bl = blur * (1.0, 0.5, 0.0)[n]
    if bl > 0:
        out = fx.dir_blur(out, math.degrees(math.atan2(dy, dx)), bl)
    if n == 0:
        out = fx.chroma(out, 6.0, angle=math.degrees(ang))
    return out


# ============================================================================ M3 duotone colour hit
# dMWE f348-358 (11.61-11.95 s): radial-blur in, then 9 frames of a RED/CYAN split duotone with
# RGB split (shadows RGB(120,40,46), highlights cyan), radial-blur out. Repeats every 5.5 s on
# the Sharingan / Obito shots (11.645, 17.150, 22.656, 28.161). The draft has 3 tinted frames in
# chorus 1 (21.73-21.81) and none in the final chorus.

def duotone(img, shadow=(46, 40, 120), high=(235, 228, 120), split=8.0, contrast=1.35):
    """Two-colour grade: luma -> lerp(shadow, high) (BGR), contrast boost, RGB split px."""
    g = _gray(img).astype(np.float32) / 255.0
    g = np.clip((g - 0.5) * contrast + 0.5, 0, 1)
    s = np.array(shadow, np.float32)
    h = np.array(high, np.float32)
    out = s[None, None, :] * (1 - g[..., None]) + h[None, None, :] * g[..., None]
    out = _u8(out)
    return fx.chroma(out, split) if split > 0 else out


def duotone_hit(img, k, frames=7, shadow=(46, 40, 120), high=(235, 228, 120)):
    """k=0 radial-blurred duotone, 1..frames-3 full duotone, last 2 frames blend back to the
    original (0.5, 0.2). 7 output frames = 0.29 s (ref 9 frames @30)."""
    d = duotone(img, shadow, high)
    if k == 0:
        return fx.radial_blur(fx.xform(d, 1.12), 120.0)
    if k >= frames:
        return img.copy()
    rem = frames - k
    if rem <= 2:
        a = (0.5, 0.2)[2 - rem]
        return cv2.addWeighted(d, a, img, 1 - a, 0)
    return d


# Palette per subject (BGR): use the ref's red/cyan on Sharingan/Rinnegan, purple/white on Pain,
# orange/black on Kurama.
DUO = {
    "sharingan": ((46, 40, 120), (235, 228, 120)),
    "rinnegan": ((70, 20, 40), (255, 215, 235)),
    "kurama": ((10, 20, 60), (90, 200, 255)),
}


# ============================================================================ M4 white whip-in + glow
# MBoD f786-801 (26.20-26.73): whip out (2 frames smeared), 2 frames of WHITE horizontal smear,
# then the new close-up enters overexposed and decays over 4 frames into a held high-key look
# (white bloom in the lights, mean luma ~205 for 16 frames). Same at 30.47, 34.63, 45.50.
# q-DyE f1070-1076: radial star-streak out, white-blown blurred entry, halo glow decays 5 frames.
# MBoD f1752-1763: focus rack-in then the same diffused glow hold.

def white_whip_in(img, k, frames=5, direction=(1.0, 0.0), smear=260.0):
    """k=0: near-white with a horizontal smear of the incoming shot; k=1..frames-1: exposure
    (screen-white 0.85 -> 0) and smear (260 -> 0 px) decay with ease-out. k>=frames: img."""
    if k >= frames:
        return img.copy()
    u = k / max(frames - 1, 1)
    e = 1.0 - fx.ease_out_cubic(u)
    white = 0.85 * e if k > 0 else 0.88
    out = img
    L = smear * e
    if L >= 1:
        out = fx.dir_blur(fx.xform(img, 1.0 + 0.10 * e), math.degrees(math.atan2(direction[1], direction[0])), L)
    return cv2.addWeighted(out, 1 - white, np.full_like(out, 255), white, 0)


def glow_hold(img, strength=0.55, threshold=150, lift=0.10):
    """High-key diffusion glow for a held hero shot (MBoD f794-801 / q f1073-1075): soft bloom
    of the highlights + small lift. Detail stays sharp (bloom is added, not a blur of the plate)."""
    m0 = fx.luma(img)
    # already-bright plates (Six Paths gold, mean luma 170+) blow out under a fixed bloom:
    # scale strength down with brightness and cap the result at max(m0+18, 205) mean luma.
    k = float(np.clip((215.0 - m0) / 110.0, 0.15, 1.0))
    out = fx.bloom(img, threshold=max(threshold, int(m0 + 25)), strength=strength * k, radius=1.6)
    if lift > 0:
        out = cv2.addWeighted(out, 1 - lift * k, np.full_like(out, 255), lift * k, 0)
    cap = max(m0 + 18.0, 205.0) if m0 < 205 else m0 + 4.0
    if fx.luma(out) > cap:
        out = fx.match_luma(out, cap)
    return out


def focus_rack_in(img, k, frames=5, sigma0=14.0):
    """MBoD f1752-1757: the new shot enters defocused and racks sharp over `frames`."""
    if k >= frames:
        return img.copy()
    u = k / max(frames - 1, 1)
    return fx._defocus(img, sigma0 * (1 - fx.ease_out_cubic(u)) + 0.0)


# ============================================================================ M5 3D card swing
# q-DyE f705-728 (23.52-24.29, Pain vs Naruto taijutsu): the shot plays as a CARD in 3D --
# shrunk to ~0.88 with black around it, top edge sloped (perspective yaw/roll a few degrees), and
# it swings between beats; a radial streak takes it out (f722). The draft never shows the frame
# as an object.

def card3d(img, yaw=0.0, pitch=0.0, roll=0.0, scale=1.0, dx=0.0, dy=0.0, bg=None, f=1800.0):
    """Render img as a plane rotated yaw/pitch/roll (deg) seen through a pinhole (focal f px),
    scaled, offset. bg: None = black, 'blur' = darkened blurred copy, or an image."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    hw, hh = w / 2.0 * scale, h / 2.0 * scale
    P = np.float32([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]])
    ry, rx, rz = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    Q = (Rz @ Rx @ Ry @ P.T).T
    Q[:, 2] += f
    dst = np.float32([[cx + dx + q[0] * f / q[2], cy + dy + q[1] * f / q[2]] for q in Q])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    if bg is None:
        base = np.zeros_like(img)
    elif isinstance(bg, str) and bg == "blur":
        base = _u8(_f(fx._defocus(img, 24.0)) * 0.35)
    else:
        base = bg
    card = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    return _u8(_f(card) * m + _f(base) * (1 - m))


def card_swing(img, n, n_total, seed=0, yaw0=14.0, pitch0=6.0, roll0=-3.5, s0=0.84, s1=0.92, bg="blur"):
    """A held shot as a swinging card: starts at yaw0/pitch0/roll0, scale s0, eases to a residual
    (25%) tilt at scale s1 by the end of the shot, with a 2-frame whip-in on n=0/1.
    Sign of the swing alternates with seed so consecutive cards swing opposite ways."""
    sg = 1 if (seed % 2 == 0) else -1
    u = fx.ease_out_cubic(min(max(n / max(n_total - 1, 1), 0.0), 1.0))
    k = 1.0 - 0.75 * u
    out = card3d(img, yaw=sg * yaw0 * k, pitch=pitch0 * k, roll=sg * roll0 * k,
                 scale=s0 + (s1 - s0) * u, bg=bg)
    if n <= 1:
        out = fx.dir_blur(out, 0.0 if sg > 0 else 180.0, (90.0, 35.0)[max(n, 0)])
    return out


# ============================================================================ S1 highlight isolate
# MBoD f1740-1749 (58.00-58.30): the frame darkens over 4 frames until ONLY the eye highlights
# remain on black (5 frames), they shrink, 2 near-black frames, then the next shot racks in.

def highlight_isolate(img, k, frames=8, keep_pct=99.3):
    """k=0..frames-1: everything below a rising threshold sinks to black; the top highlights
    (eyes, chakra) survive alone, then fade. Returns black for k >= frames."""
    if k >= frames:
        return np.zeros_like(img)
    u = k / max(frames - 1, 1)
    g = _gray(img).astype(np.float32)
    thr_hi = np.percentile(g, keep_pct)
    thr = np.percentile(g, 50) + (thr_hi - np.percentile(g, 50)) * min(u * 1.6, 1.0)
    soft = 18.0
    m = np.clip((g - thr) / soft, 0, 1)
    dim = max(0.0, 1.0 - u * 2.2)  # base image dims to 0 by ~45%
    a = np.maximum(m, dim)[..., None]
    fade = 1.0 if u < 0.75 else max(0.0, 1.0 - (u - 0.75) / 0.25)
    out = _f(img) * a * fade
    return _u8(out)


# ============================================================================ S2 dip to black
# MBoD f657-662 (21.90-22.07): the held shot dims 1.0 -> 0.75 -> 0.5 -> 0.3 over 4 frames, then
# ONE pure black frame, then the hit (white page + text there; for us: the impact package).

def dip_black(img, k, levels=(0.78, 0.55, 0.32, 0.0)):
    """k=0..3 before the hit frame: luma x levels[k] (last = pure black)."""
    if k < 0 or k >= len(levels):
        return img.copy()
    return _u8(_f(img) * levels[k])


# ============================================================================ S3 dutch drift (dialogue)
# MBoD f1450-1494 (48.33-49.80): a sync-dialogue line plays over a 1.5 s hold that is tilted
# ~8 deg (dutch) and drifts steadily sideways; the line is subtitled lower-centre in condensed
# caps with ONE keyword coloured. It is the only long hold in the chorus section.

def dutch_drift(img, u, angle=-4.0, pan=60.0):
    """u in 0..1 across the hold: constant dutch angle, linear pan of `pan` px.
    Scale is the minimum that covers (about 1.07 at 4 deg)."""
    dx = pan * (u - 0.5)
    s = fx.cover_scale(angle, dx, 0.0)
    return fx.xform(img, s, angle, dx, 0.0)


# ============================================================================ schedule helper
FPS = 24000 / 1001
BEAT = 0.4109797


def frame_of(t):
    return int(round(t * FPS))


def beat_frames(times):
    return [frame_of(t) for t in times]


# Where each effect goes (edit times in s; EDL indices from work/edl.json, 129 shots). The
# director owns the final call; these avoid the moments that already carry a flash in the draft
# (19.737 inverted drop frame, 38.62-38.91 white, 48.917 white, 75.630 white, 59.19-60.42 strobe).
PLAN = {
    # impact_cut: k=1 (neg_glow) ON the beat frame, k=0 on the frame before. Every bar downbeat
    # that is a cut in chorus 1 and the final chorus. tint per subject (BGR). Use short=True when
    # the incoming shot is 1 beat long. 65.767 (EDL 96, a 5-frame micro-cut) is left out on
    # purpose: the package would cover the whole shot; give it neg_glow on its first frame only.
    "impact_cut": [
        (21.381, 22, "blue"), (23.025, 26, "blue"), (24.669, 29, "blue"), (26.313, 32, "orange"),
        (27.957, 36, "orange"), (29.601, 40, "orange"), (31.244, 42, "purple"), (32.888, 44, "purple"),
        (34.532, 47, "purple"), (36.176, 50, "blue"), (39.464, 54, "purple"), (41.108, 56, "blue"),
        (42.752, 59, "purple"), (44.396, 61, "blue"), (46.040, 63, "blue"),
        (49.328, 67, "orange"), (55.081, 76, "orange"), (57.547, 79, "orange"),
        (62.479, 89, "purple"), (64.123, 93, "blue"), (67.411, 102, "purple"),
        (69.055, 106, "blue"), (70.699, 110, "blue"), (72.342, 112, "orange"), (73.986, 115, "blue"),
        (77.274, 121, "blue"), (78.918, 123, "blue"),
    ],
    # mono_hit inside a held shot on a bar downbeat that is NOT a cut
    "mono_hit": [(37.820, 51)],
    # beat_smear: every beat strictly inside a shot held >= 2 beats (n=0 on the beat frame)
    "beat_smear": [
        (23.436, 26), (25.902, 31), (30.012, 40), (30.833, 41), (31.655, 42), (32.477, 43), (33.299, 44),
        (35.765, 49), (36.587, 50), (37.409, 51), (39.053, 53), (39.875, 54), (40.697, 55), (42.341, 58),
        (43.163, 59), (43.985, 60), (44.807, 61), (45.629, 62), (46.451, 63), (47.273, 64),
        (61.246, 87), (62.068, 88), (65.356, 95), (71.109, 110), (71.931, 111), (72.753, 112),
        (76.863, 120), (77.685, 121), (78.507, 122),
    ],
    # duotone_hit (7 frames from the given time); palette key from DUO
    "duotone_hit": [(22.203, 24, "kurama"), (29.190, 39, "sharingan"), (31.655, 42, "sharingan"),
                    (40.697, 55, "rinnegan"), (52.615, 71, "kurama"), (58.780, 82, "kurama"),
                    (62.890, 90, "rinnegan")],
    # white_whip_in k=0.. from the shot start, then glow_hold for the next 8-12 frames only
    "white_whip_in+glow_hold": [(51.793, 70), (61.657, 88), (66.589, 100), (80.562, 124), (85.494, 128)],
    # card_swing for the whole shot, seed alternates (Pain arc 2-beat holds, q-DyE 23.5-24.3)
    "card_swing": [(39.464, 54, 0), (40.286, 55, 1), (41.930, 58, 0), (42.752, 59, 1), (44.396, 61, 0),
                   (45.218, 62, 1)],
    # THE STOP 47.684-48.917: dip_black on the
    # 4 frames before 47.684, focus_rack_in on the new plate, dutch_drift across it, then
    # highlight_isolate over its last 8 frames into the existing 48.917 white flash.
    "stop": {"dip_black_frames": [frame_of(47.684) - 4 + i for i in range(4)],
             "plate_start": frame_of(47.684), "isolate_from": frame_of(48.917) - 8,
             "suggested_plate": ("kurama_war_1", 79.8, "Kurama's eyes; proven in refgap_B/P6_misc.jpg")},
    # dialogue holds (story part 0-19.7 and EDL 125 Iruka 82.206): dutch_drift on the COMPOSED
    # 16:9 frame (after the pillar fill), focus_rack_in on entry
    "dutch_drift": "every sync-dialogue hold (Iruka / Naruto lines) and EDL 125",
}
TINT = {"blue": (255, 170, 40), "orange": (40, 140, 255), "purple": (255, 110, 190)}
