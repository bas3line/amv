#!/usr/bin/env python3
"""UZUMAKI -- Naruto birthday edit (10.10), cut to "Jane!" (The Long Faces). v2: original voices,
reference-edit effects (fx/refpack_A.py, fx/refpack_B.py), credit row with logos (fx/credits.py).

Frame-exact compositor. Every output frame (1920x1080, 24000/1001 fps) is built in numpy/OpenCV
from the normalised 1080p footage, then piped to ffmpeg. The song (song/edit.wav) is muxed
untouched as AAC 256k. Timeline = style_bible.md laid on the song.json beat grid; every cut
starts on a beat frame round(t * 24000/1001). Effects come from fx/fxpack.py, text from
fx/titles.py (both imported, not modified).

Story: alone (0-8.23) -> Iruka saves him (8.23-13.16) -> ramen, "the next Hokage" (13.16-16.45)
-> build into the drop (16.45-19.74) -> JANE: Sage Naruto arrives, then the cooking from Haku,
Neji, Gaara, Sasuke I, Kakuzu, Pain (19.74-47.68) -> THE STOP: the Rasengan lands on Pain, gold
duotone + light sweep, no text (47.68-48.92) -> Kurama, Minato, Obito (48.92-60.84) -> Six Paths, Madara,
Kurama vs Susanoo, the final Valley of the End (60.84-80.56) -> Hero of the Leaf, Iruka,
Kakashi, the monument (80.56-85.49) -> Seventh Hokage + HAPPY BIRTHDAY, NARUTO / 10.10 (85.49-end).

Final-fix pass (QA round 2): soft SD / low-bitrate plates go through Real-ESRGAN
(realesr-animevideov3, tools/sr.py) into sr_cache/ before the render; Part I 4:3 plates are
cropped full-bleed to 16:9 with a per-shot vertical offset (no more window-box); T13 tilts stay
inside the plate; T5 hue flashes became gold-white pops; the strobe uses dark (not black) frames
at <= 3 flashes/s on sharp plates; the vector-eye iris became a dive into the real eye.

usage:
  director.py --sr               Real-ESRGAN every source frame the SR shots read -> sr_cache/ (run first)
  director.py                    render all (parallel lossless chunks) -> out/naruto_bday_edit.mp4
  director.py --out PATH         ... or render to PATH
  director.py --stills 0,40,473  PNG stills of edit frames -> dir_tmp/stills/
  director.py --edl              write edl.json only
  director.py --chunk A B OUT    (internal) render frames [A, B) losslessly + a per-frame log

Paths resolve from the repo root (this file's directory), or from $AMV_WORK when it is set.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

WORK = os.environ.get("AMV_WORK") or os.path.dirname(os.path.abspath(__file__))
for _p in (f"{WORK}/fx", f"{WORK}/lib", f"{WORK}/lib2"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import fxpack as fx  # noqa: E402
import titles as TT  # noqa: E402
import shaders as SH  # noqa: E402
import vfx2 as V  # noqa: E402
import refpack_A as RA  # noqa: E402   v2: effects measured on G3GC / fUx4 / MoxZ7
import refpack_B as RB  # noqa: E402   v2: effects measured on q-DyE / dMWE / MBoD
import credits as CR  # noqa: E402     v2: X / IG / GitHub credit row
import v2fix as V2  # noqa: E402       v2 QA: mid-luma ink frame, eye light trails, RGB split
import functools  # noqa: E402

# ---- final-fix overrides of the shared fx / titles modules (patched here; the module files are untouched)
_story_words_orig = TT._story_words


def _story_words_fix(moment, pos=None):
    """'UZUMAKI NARUTO' 1.5x bigger, ~55 px higher (clear of the magatama row) and gone before the bloom peaks."""
    if moment != "uzumaki":
        return _story_words_orig(moment, pos)
    cap = 68
    base = 0.79 * 1080 + cap / 2
    lay, _ = TT._line_layout("UZUMAKI NARUTO".split(), TT.MARC, cap, TT.T1_TRACK, 1920 / 2, base)
    out = [dict(text=w, cap=cap, x=x, base=b, k0=k0, g=0) for (w, x, b), k0 in zip(lay, (549, 554))]
    return out, {0: (570, 578)}


_bday_lines_orig = TT._bday_lines


def _bday_lines_fix(cx, top_base, card=False):
    out = _bday_lines_orig(cx, top_base, card)
    if not card:      # 10.10 over the grin: cap 40 -> 52 (it was a thin dim line on the white collar)
        for wd in out:
            if wd["text"] == "10.10":
                lay, _ = TT._line_layout(["10.10"], TT.CINZ, 52, 0.6, cx, wd["base"] + 12)
                wd.update(cap=52, x=lay[0][1], base=wd["base"] + 12)
    return out


TT._story_words = _story_words_fix
TT._bday_lines = _bday_lines_fix
# A5 purple: lower the violet look's black point so Susanoo shots keep their blacks (it read milky)
fx._bases()["violet"] = (fx._curve([(0, 2), (40, 30), (128, 150), (255, 255)]),
                         fx._curve([(0, 0), (40, 22), (128, 120), (255, 240)]),
                         fx._curve([(0, 0), (40, 22), (128, 112), (255, 236)]))

OUT = f"{WORK}/out/naruto_bday_edit.mp4"
SONG_WAV = f"{WORK}/voices/mix.wav"     # v2: song + the original Naruto/Iruka lines, ducked (voices/mix.py)
FOOT = f"{WORK}/footage"
TMP = f"{WORK}/dir_tmp"
EDL_PATH = f"{WORK}/edl.json"
W, H = 1920, 1080
FPS_N, FPS_D = 24000, 1001
FPS = FPS_N / FPS_D
SONG = json.load(open(f"{WORK}/song/song.json"))
BEATS = [b["t"] for b in SONG["beats"]]
DUR = float(SONG["duration"])            # 89.9316
NF = int(math.ceil(DUR * FPS))           # 2157 frames -> 89.965 s
PRE, POST = 4, 5    # extra decoded frames before/after each shot (xd / bd read 3 past the cut, the card 1)


def fr(t):
    return int(round(t * FPS))


def bt(i):
    """Edit time of beat i (song.json grid); fractional i = part-beats (half-beat micro-cuts)."""
    i0 = int(math.floor(i))
    f = i - i0
    if f < 1e-9:
        return BEATS[i0]
    return BEATS[i0] + (BEATS[i0 + 1] - BEATS[i0]) * f


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


# ----------------------------------------------------------------------------- footage
# id -> (file, frame format, native fps, sharpen amount for soft sources)
CLIPS = {
    "C1": ("childhood_1", "169", 24.0, 0.0),
    "C2": ("childhood_2", "169", 30.0, 0.0),
    "C4": ("childhood_4", "169", 30.0, 0.0),
    "I1": ("iruka_ramen_1", "43", FPS, 0.45),
    "I2": ("iruka_ramen_2", "43", FPS, 0.45),
    "I3": ("iruka_ramen_3", "169", FPS, 0.0),
    "I5": ("iruka_ramen_5", "169", FPS, 0.0),
    "I6": ("iruka_ramen_6", "169", FPS, 0.0),
    "F1": ("early_fights_1", "43", FPS, 0.45),
    "I4": ("iruka_ramen_4", "169", FPS, 0.0),
    "F2": ("early_fights_2", "43", FPS, 0.15),
    "F3": ("early_fights_3", "43", FPS, 0.35),
    "F4": ("early_fights_4", "43", FPS, 0.15),
    "F5": ("early_fights_5", "43", FPS, 0.40),
    "P1": ("pain_arc_1", "169", FPS, 0.30),
    "P2": ("pain_arc_2", "169", 24.0, 0.0),
    "P3": ("pain_arc_3", "169", 24.0, 0.0),
    "P4": ("pain_arc_4", "169", 24.0, 0.0),
    "K1": ("kurama_war_1", "169", FPS, 0.0),
    "K2": ("kurama_war_2", "169", FPS, 0.35),
    "K3": ("kurama_war_3", "169", FPS, 0.30),
    "K4": ("kurama_war_4", "169", FPS, 0.0),
    "M1": ("madara_kaguya_sasuke_1", "169", FPS, 0.0),
    "M2": ("madara_kaguya_sasuke_2", "169", FPS, 0.40),
    "M3": ("madara_kaguya_sasuke_3", "169", FPS, 0.40),
    "M4": ("madara_kaguya_sasuke_4", "169", FPS, 0.0),
    "H1": ("hokage_1", "185", FPS, 0.0),
    "H2": ("hokage_2", "169", FPS, 0.0),
    "H4": ("hokage_4", "185", FPS, 0.0),
}
LB185 = {"hokage_1": (21, 1059), "hokage_4": (21, 1059)}   # content rows [y0, y1) of 1.85:1 sources
# Real-ESRGAN plates (sr_cache/<clip>/<source frame>.png): 4:3 clips are stored as the SR'd picture at
# 1920x1440 (cropped to 16:9 per shot by `yoff`), 16:9 clips at 1920x1080. Sharpen is off for them.
SR_CLIPS = {"I1", "I2", "F1", "F2", "F3", "F4", "F5", "C2", "I3", "I5", "P1", "K2", "K3", "M2", "M3"}
SR_DIR = f"{WORK}/sr_cache"
# px of source border trimmed off every edge (thin dark / orange edge lines in these uploads)
CLIP_TRIM = {"M1": 4, "P1": 16}   # v2: P1 6 -> 16 (a 2 px orange/dark border still showed on the dome at 38.64)


def fill43(img, fit=1.0, yoff=0.5):
    """4:3 picture (x 240-1680) at `fit` scale over a blurred, darkened cover of itself.
    No crop of the picture at fit 1.0 -- the fill only replaces the black pillars."""
    c = img[:, 240:1680]
    sm = cv2.resize(c, (240, 180), interpolation=cv2.INTER_AREA)[22:157]
    sm = cv2.GaussianBlur(sm, (0, 0), 6)
    bg = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = cv2.convertScaleAbs(bg, alpha=0.40)
    if abs(fit - 1.0) < 1e-3:
        fg = c
    else:
        fw, fh = int(round(1440 * fit)), int(round(1080 * fit))
        fg = cv2.resize(c, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
        y0 = int(round((fh - H) * yoff))
        fg = fg[y0:y0 + H]
    x0 = (W - fg.shape[1]) // 2
    x1 = x0 + fg.shape[1]
    out = bg
    out[:, x0:x1] = fg
    # a soft shadow on the fill next to the picture edge so the seam reads as a frame
    ramp = (0.55 + 0.45 * np.arange(1, 19, dtype=np.float32) / 18)[None, :, None]
    if x0 >= 18:
        out[:, x0 - 18:x0] = (out[:, x0 - 18:x0].astype(np.float32) * ramp[:, ::-1]).astype(np.uint8)
    if x1 + 18 <= W:
        out[:, x1:x1 + 18] = (out[:, x1:x1 + 18].astype(np.float32) * ramp).astype(np.uint8)
    return out


def fill185(img, y0, y1):
    """1.85:1 letterboxed source -> fill the frame (scale ~1.04, trims the sides)."""
    c = img[y0:y1]
    s = H / c.shape[0]
    w = int(round(W * s))
    c = cv2.resize(c, (w, H), interpolation=cv2.INTER_LANCZOS4)
    x0 = (w - W) // 2
    return np.ascontiguousarray(c[:, x0:x0 + W])


def base_scale(cid, d):
    fmt = CLIPS[cid][1]
    if d.get("zoom"):
        return d["zoom"][2]
    if fmt == "43":
        return d.get("fit43", 4 / 3 if cid in SR_CLIPS else 1.0)
    if fmt == "185":
        y0, y1 = LB185[CLIPS[cid][0]]
        return H / (y1 - y0)
    return 1.0


# ----------------------------------------------------------------------------- EDL
E = []


def S(b, clip, src, act, **kw):
    E.append(dict(b=b, clip=clip, src=src, act=act, **kw))


# A1 ALONE (0 - 8.230): slit-open, 78 px bars, night teal-navy, hard cuts on the bar, no camera moves
S(0, "C1", 29.62, "A1", rev=True, slow=(1.0, 1.03), note="ECU kid Naruto: eyes closed, open, looking up")
# v2: the ORIGINAL voices (voices/dialogue_plan.json). SYNC shots run at 1.0x (no slow/ramp/speed/rev).
S(4, "I1", 3.95, "A1", yoff=0.40, sync="L1",
  note="SYNC L1 Naruto scrubbing the Hokage faces: 'It's not like there's anyone waiting at home for me!'")
S(10.5, "I6", 0.0, "A1", note="alone on the academy swing (official HD)")
S(12.5, "C2", 8.12, "A1", note="fists clenched, crying among grey villagers")
# A2 IRUKA + RAMEN
S(14.5, "I2", 17.12, "A2_iruka", yoff=0.50, note="Mizuki hurls the Fuma shuriken")
S(17, "I2", 19.85, "A2_iruka", yoff=0.50, note="Iruka dives in to shield him")
S(18, "I2", 21.08, "A2_iruka", yoff=0.50, note="the shuriken in Iruka's back, Naruto under him")
S(19, "I2", 32.82, "A2_iruka", yoff=0.50, sync="L2",
  note="SYNC L2 Iruka: 'You feel lonely and it hurts inside.' (flashback, cut 34.284 to Iruka crying)")
S(26, "I2", 87.98, "A2_iruka", yoff=0.45, sync="L3a", note="SYNC L3a Iruka: 'But that's not who Naruto is.'")
S(31.5, "I2", 95.25, "A2_iruka", yoff=0.40, sync="L3b", note="SYNC L3b Iruka: 'He's Naruto Uzumaki.'")
S(36.5, "I1", 30.12, "A2_ramen", yoff=0.35, sync="L4", note="SYNC L4 Naruto, chopsticks up: 'The next Hokage!'")
S(40, "I3", 27.60, "A2_ramen", pulse=True, note="the Hokage rock")
S(41, "I2", 112.20, "A2_ramen", yoff=0.50, pulse=True, note="shadow-clone hand sign")
S(42, "C4", 30.20, "A2_ramen", pulse=True, note="goggles, sunset, wide-eyed")
S(43, "I2", 98.65, "A2_iruka", yoff=0.50, pulse=True, pop=165, note="crying behind the tree (17.682 accent)")
S(44, "K1", 109.30, "A5_gold", kind="build", tin="bd", rays=(0.5, 0.36),
  note="T4 blur-dissolve into KCM Naruto; UZUMAKI NARUTO; gold bloom; bars retract; whip")
# DROP + chorus 1 (A3, 19.737 - 47.684)
S(48, "P2", 4.25, "A3_hot", wdir=(1, 1),
  note="DROP: Sage Naruto crashes down on Pain's summons (v2: refpack_A.drop_frame, live shake + echo + glow)")
S(50, "P2", 0.05, "A3_hot", pop=175, note="'Jane!' Sage eyes (1 beat; the 2nd beat went to Zabuza)")
S(51, "F5", 0.00, "A3_cool", yoff=0.45, note="Part I begins: Zabuza, kunai in his teeth (VIZ Blu-ray)")
S(52, "F5", 2.40, "A3_hot", yoff=0.50, note="Haku arc: the hand sign")
S(53, "F5", 4.35, "A3_hot", yoff=0.50, tin="whip", wdir=(-1, -1), note="first Nine-Tails chakra erupts")
S(54, "F5", 11.50, "A3_hot", yoff=0.50, note="feral red eyes")
S(55, "F1", 20.80, "A3_hot", yoff=0.50, tin="whip", wdir=(1, -1),
  note="the fox-chakra fist into Haku's mask (v2 QA: no body shake, the effect stays on the cut frames)")
S(56, "F2", 2.50, "A3_cool", yoff=0.50, note="Neji: Kaiten")
S(58, "F2", 9.55, "A3_hot", yoff=0.50, tin="whip", wdir=(1, 1), pop=175, note="Naruto launches over the wall (23.847)")
S(59, "F2", 11.60, "A3_cool", yoff=0.50, note="Neji screams")
S(60, "F2", 21.90, "A3_hot", yoff=0.50, tin="whip", wdir=(-1, 1), hits=[(0, "hit")], note="blitz")
S(61, "F2", 25.40, "A3_hot", yoff=0.50, note="kunai, close")
S(62, "F2", 27.85, "A3_hot", yoff=0.50, tin="whip", wdir=(1, -1), note="split-screen charge")
S(64, "F3", 6.55, "A3_cool", yoff=0.50, pop=175, note="Gaara, cracked face (26.313)")
S(65, "F3", 23.30, "A3_cool", yoff=0.50, punch=1.22, note="the clone swarm")
S(66, "F3", 59.20, "A3_cool", yoff=0.50, tin="whip", wdir=(-1, -1), note="dives at the camera")
S(67, "F3", 66.75, "A3_hot", yoff=0.50, hits=[(0, "hit")], note="Uzumaki Barrage")
S(68, "F3", 76.20, "A3_cool", yoff=0.50, tin="whip", wdir=(1, 1), note="Shukaku")
S(69, "F3", 80.60, "A3_hot", yoff=0.50, note="Gamabunta")
S(70, "F3", 19.98, "A3_hot", yoff=0.50, tin="whip", wdir=(-1, 1), note="Naruto, hand sign, determined")
S(71, "F3", 108.85, "A3_cool", yoff=0.50, z2=True, note="Gaara's eyes in Shukaku's forehead")
S(72, "F3", 105.60, "A3_hot", yoff=0.50, tin="radial", hits=[(0, "big")], embers=0.6, pop=180,
  note="the Nine-Tails bursts out of the smoke")
S(74, "F3", 109.60, "A3_hot", yoff=0.50, slow=(1.0, 1.05), note="Nine-Tails grin")
S(76, "F4", 24.75, "A3_cool", yoff=0.50, tin="whip", wdir=(1, -1), eye=(0.52, 0.42),
  note="Valley of the End: Sasuke's Sharingan (2 beats; the soft Naruto-eye ECU is gone)")
S(78, "F4", 8.50, "A3_hot", yoff=0.50, tin="eyedive", note="Rasengan in the red chakra")
S(80, "F4", 17.80, "A3_cool", yoff=0.50, tin="whip", wdir=(-1, -1), pop=175, note="Curse Mark Chidori (32.888)")
S(82, "F4", 31.40, "A3_cool", yoff=0.50, note="over the waterfall")
S(83, "F4", 37.20, "A3_hot", yoff=0.50, tin="whip", wdir=(1, 1), note="Naruto + Rasengan, close")
S(84, "F4", 38.30, "A3_cool", yoff=0.50, tin="whip", wdir=(-1, 1), note="Sasuke, Curse Mark 2")
S(85, "F4", 43.20, "A3_cool", yoff=0.50, hits=[(0, "big")], note="they collide")
S(86, "F4", 45.20, "A3_hot", yoff=0.50, slow=(1.0, 1.05), note="blue and orange streaks")
S(88, "P1", 8.40, "A3_cool", tin="whip", wdir=(1, -1), speed=0.82,
  note="Kakuzu: the Rasenshuriken charge (starts at 8.40: the 'Eat this!' subtitle is burned in until ~8.3)")
S(90, "P1", 9.12, "A3_cool", ramp=[(3, 1.0), (5, 0.15), (22, 1.3)], hits=[(3, "big")], pop=175,
  note="Rasenshuriken slams into Kakuzu (36.998; crawl on contact)")
S(93, "P1", 62.80, "A3_cool", tin="whip", wdir=(-1, -1), note="inside the jutsu: the mask shatters")
S(94, "P1", 21.90, "A3_cool", note="the Rasenshuriken dome on the ground")
S(96, "P2", 1.62, "A3_cool", slow=(1.0, 1.05), note="Pain stares")
S(98, "P2", 36.65, "A3_hot", tin="whip", wdir=(1, 1), hits=[(2, "hit")], note="Frog Kumite: fist into Pain")
S(100, "P2", 37.70, "A3_hot", hits=[(0, "hit")], pop=175, note="the kick (41.108)")
S(101, "P3", 7.40, "A3_cool", tin="whip", wdir=(-1, 1), note="clones in the smoke")
S(102, "P2", 90.30, "A3_cool", tin="whip", wdir=(1, -1), note="the Sage Rasenshuriken spins into the ground")
S(104, "P2", 94.80, "A3_cool", note="two Pains swallowed")
S(106, "P2", 105.10, "A3_hot", hits=[(0, "big")], pop=180, note="the blast spikes (43.574)")
S(108, "P3", 20.30, "A3_cool", tin="whip", wdir=(-1, -1), note="the clones form the Rasengan")
S(110, "P3", 22.50, "A3_cool", punch=1.15, note="three clones, Rasengan")
S(112, "P3", 37.00, "A3_cool", tin="whip", wdir=(1, 1), note="Naruto leaps at Pain")
S(114, "P3", 44.05, "A3_hot", hits=[(0, "big")], pop=180, note="Rasengan into Pain on 'grass'")
# THE STOP (47.684 - 48.917): the music stops for 3 beats. v2: no text -- the Rasengan drives on
# into the anime's own impact freeze, gold duotone + light sweep, white smear out (fUx4 45.2-46.0)
S(116, "P3", 44.842, "A3_hot", stop=True, note="THE STOP: the Rasengan drives into Pain, lands on the impact freeze")
# A4 BUILD (48.917 - 60.835): war, desaturated, exposure pulse per cut
S(119, "K1", 0.20, "A4", xp=0.8, eye=(0.5, 0.45), note="Kurama's eye opens (out of the white smear)")
S(120, "K1", 45.30, "A4", punch=1.30, hits=[(0, "big")], pop=185, embers=0.5, note="crash 49.328: Kurama's jaw")
S(122, "K1", 52.20, "A4", tin="whip", wdir=(1, -1), xp=0.8, note="Kurama grins over Naruto")
S(124, "K1", 61.60, "A4", hand=True, xp=0.8, note="the hatred chakra")
S(126, "K1", 65.50, "A4", xp=0.8, note="Naruto in the red aura")
S(128, "K1", 79.80, "A4", tin="whip", wdir=(-1, -1), pop=170, note="Kurama's eyes (52.615)")
S(129, "K1", 86.05, "A4", punch=1.15, xp=0.8, note="Naruto, close")
S(130, "K1", 92.30, "A4", tin="whip", wdir=(1, 1), xp=0.8, note="the clones pull the chakra out")
S(132, "K1", 29.20, "A4", xp=0.8, note="the seal opens")
S(133, "K2", 38.40, "A4", xp=0.8, note="fist bump with Kurama")
S(134, "K2", 73.00, "A4", tin="whip", wdir=(-1, 1), pop=170, slow=(1.0, 1.04), note="Kurama grins (55.081)")
S(136, "K3", 3.60, "A4", xp=0.8, note="Minato and Naruto touch fists")
S(138, "K3", 66.50, "A4", tin="whip", wdir=(1, -1), xp=0.8, note="two Kuramas")
S(140, "K4", 9.90, "A4", punch=1.22, hits=[(0, "big")], pop=180, note="KCM + Sasuke vs Obito (57.547)")
S(141, "K4", 44.00, "A4", tin="whip", wdir=(-1, -1), note="Obito")
S(142, "K4", 47.55, "A4", hits=[(0, "hit")], note="the flaming arm smash (impact pose, Obito in frame)")
S(143, "K4", 13.20, "A4", note="Naruto's red eye")
S(144, "K4", 7.30, "A4", tin="strobe", note="strobe: the Rinnegan eye (v2: sharp HD plate, was an ink-smear frame)")
S(145, "K4", 26.50, "A4", tin="strobe", note="strobe: Six Paths Naruto glowing (v2: sharp, was the soft M2 smear)")
S(146, "K4", 5.30, "A4", tin="strobe", note="strobe: Sasuke's Rinnegan")
S(147, "M2", 12.90, "A4", note="Susanoo; the Six Paths PIP rises over it (the 4th strobe beat gives way to the PIP)")
# A5 COOKING II (60.835 - 80.562): final chorus
S(148, "M2", 6.00, "A5_gold", tin="pip", note="PIP zoom-through: Six Paths Naruto")
S(150, "M4", 18.00, "A5_gold", tin="whip", wdir=(1, 1), tilt=(0.75, 1.5, 1.5, 3.0), embers=0.6,
  note="Six Paths Naruto bursts out of the fire")
S(152, "M2", 22.20, "A5_purple", tin="radial", punch=1.22, pop=175, note="Madara (62.479)")
S(153, "M2", 17.80, "A5_purple", note="Sasuke's Rinnegan")
S(154, "M2", 2.20, "A5", tin="whip", wdir=(-1, -1), note="Madara, the Ten-Tails jinchuriki")
S(155, "M2", 84.80, "A5_gold", punch=1.22, note="Truth-Seeking Rasengan")
S(156, "M2", 93.00, "A5", tin="radial", punch=1.22, hits=[(0, "hit")], note="the combined strike")
S(157, "M3", 4.20, "A5", tin="whip", wdir=(1, -1), note="Team 7 strike")
S(158, "M3", 43.70, "A5_purple", pop=175, slow=(1.0, 1.05), note="Susanoo's eyes flare (64.945)")
S(160, "M4", 40.40, "A5_purple", tin="t12", note="PEAK 65.767: Susanoo on the statue")
S(160.5, "M4", 42.90, "A5", note="Six Paths on the water")
S(161, "M4", 49.90, "A5", hits=[(0, "hit")], note="Kurama vs Susanoo")
S(161.5, "M4", 52.40, "A5_gold", note="the tail")
S(162, "M4", 70.00, "A5_gold", tin="whip", wdir=(-1, 1), pop=175, note="Six Paths, close (66.589)")
S(163, "M4", 75.00, "A5_purple", tin="goldpop", note="Sasuke (gold-white exposure pop in)")
S(164, "M1", 0.10, "A5_purple", tin="t12", win=(1, 4), eye=(0.5, 0.40), eyekind="rinnegan",
  note="Valley of the End II: the Rinnegan (dive into the eye)")
S(165, "M1", 1.30, "A5", tin="eyedive", note="crouch, then the punch")
S(166, "M1", 16.00, "A5_purple", tin="radial", punch=1.22, note="the Chidori rips through the clones")
S(167, "M1", 43.80, "A5", hits=[(0, "hit")], note="fists")
S(168, "M1", 44.60, "A5", hits=[(0, "hit")], note="fists")
S(169, "M1", 46.93, "A5", hits=[(0, "hit")], note="fists")
S(170, "M1", 48.43, "A5", hits=[(0, "hit")], note="blood")
S(171, "M1", 54.90, "A5", tin="radial", punch=1.22, note="the bloody hit")
S(172, "M1", 74.50, "A5_purple", tin="whip", wdir=(1, 1), note="Chidori hit")
S(174, "M4", 24.80, "A5_gold", tin="whip", wdir=(-1, -1), tilt=(-0.75, -1.5, -1.5, -3.0), lumacap=150,
  note="Kurama arm")
S(176, "M4", 64.70, "A5", tilt=(1.0, 1.5, 2.0, 3.0), hits=[(0, "big")], pop=180,
  note="the fists collide mid-lake (72.342)")
S(178, "M4", 72.30, "A5_gold", tin="whip", wdir=(1, -1), note="fists in the rain")
S(179, "M1", 79.10, "A5_purple", tin="goldpop", note="Rinnegan Chidori (gold-white exposure pop in)")
S(180, "M1", 93.45, "A5", tin="whip", wdir=(-1, 1), note="Team 7 hands on the Rasengan")
S(181, "M1", 98.10, "A5", note="Naruto, the Rasengan in his hand")
S(182, "M1", 101.85, "A5", tin="whip", wdir=(1, 1), note="red chakra, he screams")
S(183, "M1", 102.55, "A5", tin="radial", punch=1.15, z2=True, note="he charges with the Rasengan")
S(184, "M1", 106.55, "A5", tin="white", ramp=[(4, 1.0), (6, 0.2), (10, 1.5)], hits=[(0, "big")],
  note="75.630 THE CLASH: Rasengan meets Chidori (white flash, crawl on contact)")
S(186, "M1", 109.20, "A5", tilt=(-0.75, -1.25, -1.5, -2.5), note="the lightning storm")
S(188, "M1", 112.36, "A5", speed=0.5, hits=[(0, "big")], pop=185,
  note="77.274 the sphere erupts (half speed; starts after the source's 3 black flicker frames, ends before its white-out)")
S(190, "M1", 113.50, "A5", slow=(1.0, 1.04), note="between the statues")
S(192, "M1", 121.40, "A5", slow=(1.0, 1.03), note="aftermath: side by side")
# A6 ACKNOWLEDGED (80.562 - 85.494): sepia gold, hard cuts and dissolves, no punches
S(196, "P4", 116.60, "A6", note="the village tosses Naruto into the sun")
S(200, "P4", 114.00, "A6", tin="xd", note="Iruka cries and smiles")
S(202, "P4", 16.30, "A6", tin="xd", note="Kakashi carries Naruto home")
S(205, "H4", 2.00, "A6", tin="xd", slow=(1.0, 1.04),
  note="the Hokage monument over the city (84.261; v2 QA: the draft's crisp wide + slow push, no 2.5x SR crop)")
# A7 HOKAGE + BIRTHDAY (85.494 - end)
S(208, "H1", 56.40, "A7", rays=(0.5, 0.15), note="the Seventh Hokage cloak")
S(210, "H1", 66.88, "A7", rays=(0.5, 0.12), note="the grin (HAPPY BIRTHDAY, NARUTO / 10.10)")
S(216, None, None, None, kind="card", note="final hit 88.782: gold flash + end card")

# placement of the text blocks (decided on stills)
IRUKA_POS = (1010, 300)
BDAY_CX = 0.5
BDAY_TOP = 760
TRANS_WIN = {"whip": (3, 0), "radial": (2, 0), "hue": (3, 0), "goldpop": (3, 0), "eyedive": (3, 0),
             "iris": (7, 0), "t12": (1, 5), "xd": (3, 3), "bd": (0, 11), "pip": (10, 0), "white": (0, 3),
             "strobe": (0, 2), "smear": (2, 0), "wsmear": (2, 0)}
# v2: refpack_A SHAKE_MED / SHAKE_BIG amplitudes (G3GC drop rms 184 px; kept at ~half, frame always covered)
SHAKE = {"hit": (60.0, 2.0, 5), "big": (110.0, 4.5, 7), "micro": (10.0, 0.0, 2)}


# ----------------------------------------------------------------------------- shots
class Shot:
    def __init__(self, i, d, t0, t1):
        d = dict(d)
        if d.get("tin") == "whip" and "punch" not in d:
            d["punch"] = 1.22          # T2: the incoming shot lands sharp at 1.22x, then Z1 decay
        self.i, self.d = i, d
        self.t0, self.t1 = t0, t1
        self.f0, self.f1 = fr(t0), fr(t1)
        self.n = self.f1 - self.f0
        self.kind = d.get("kind", "shot")
        self.cid = d.get("clip")
        self.tin = d.get("tin", "cut")
        self.frames = None
        self.cache = {}
        self.gp = None
        self.wdir = d.get("wdir", (1, -1))
        self.hits = [(int(k), p) for k, p in d.get("hits", [])]
        self.bscale = base_scale(self.cid, d) if self.cid else 1.0

    # -- source mapping
    def src_off(self, k):
        d = self.d
        if "ramp" in d:
            off, left = 0.0, k
            if k < 0:
                return k / FPS * d["ramp"][0][1]
            for n, sp in d["ramp"]:
                step = min(left, n)
                off += step / FPS * sp
                left -= step
                if left <= 0:
                    break
            if left > 0:
                off += left / FPS * d["ramp"][-1][1]
            return off
        return k / FPS * d.get("speed", 1.0)

    def src_time(self, k):
        o = self.src_off(k)
        return self.d["src"] - o if self.d.get("rev") else self.d["src"] + o

    def src_range(self):
        """Source frame indices [m0, m1] this shot reads (its frames plus PRE/POST for transitions)."""
        sfps = CLIPS[self.cid][2]
        ts = [self.src_time(k) for k in (-PRE, self.n + POST)]
        lo, hi = max(0.0, min(ts)), max(ts)
        return int(math.floor(lo * sfps)), int(math.ceil(hi * sfps)) + 1

    def sr_key(self):
        """sr_cache sub-directory for this shot's Real-ESRGAN plates, or None."""
        if self.cid is None:
            return None
        name = CLIPS[self.cid][0]
        if self.d.get("zoom"):
            cx, cy, z = self.d["zoom"]
            return f"{name}_z{cx}_{cy}_{z}"
        return name if self.cid in SR_CLIPS else None

    def decode(self):
        if self.frames is not None or self.cid is None:
            return
        name, fmt, sfps, _ = CLIPS[self.cid]
        m0, m1 = self.src_range()
        key = self.sr_key()
        if key:
            self.frames = load_sr(key, m0, m1)
        else:
            self.frames = decode_raw(name, sfps, m0, m1)
        if len(self.frames) == 0:
            raise RuntimeError(f"no frames: shot {self.i} {name}@{m0}")
        self.m0 = m0
        self.sfps = sfps

    def release(self):
        self.frames = None
        self.cache = {}

    def raw(self, k):
        self.decode()
        idx = int(math.floor(self.src_time(k) * self.sfps + 0.5)) - self.m0
        return self.frames[min(max(idx, 0), len(self.frames) - 1)]

    def prep(self, img):
        name, fmt, _, shp = CLIPS[self.cid]
        if self.sr_key():
            if img.shape[0] > H:      # SR'd 4:3 picture at 1920x1440 -> full-bleed 16:9 crop at yoff
                y0 = int(round((img.shape[0] - H) * self.d.get("yoff", 0.5)))
                img = img[y0:y0 + H]
            shp = self.d.get("sharpen", 0.0)
        elif fmt == "43":
            img = fill43(img, self.d.get("fit43", 1.0), self.d.get("yoff", 0.5))
        elif fmt == "185":
            img = fill185(img, *LB185[name])
        tr = CLIP_TRIM.get(self.cid, 0)
        if tr:
            img = cv2.resize(img[tr:H - tr, tr:W - tr], (W, H), interpolation=cv2.INTER_LANCZOS4)
        if shp > 0:
            img = fx.sharpen(img, shp, 1.1)
        return np.ascontiguousarray(img)

    def fit(self):
        if self.gp is None:
            mid = self.prep(self.raw(max(0, self.n // 2)))
            self.gp = fx.fit_grade(mid, self.d["act"], rays_center=self.d.get("rays", (0.5, 0.2)))
        return self.gp

    def plate(self, k):
        if k in self.cache:
            return self.cache[k]
        src = self.prep(self.raw(k))
        gp = self.fit()
        img = fx.apply_grade(src, gp, extras=True)
        pa = PROTECT.get(self.d["act"], 0.0)
        if pa > 0:   # keep Naruto's yellow hair yellow under the teal / violet looks (they turn it green)
            warm = fx.apply_grade(src, dict(gp, d=0.0, amt=0.0), extras=True)
            m = yellow_mask(src) * pa
            img = cv2.blendLinear(warm, img, m, 1 - m)
        if self.d["act"] == "A4":   # selective colour: Kurama orange / KCM gold stay vivid, the war stays grey
            vivid = fx._sat(fx.apply_grade(src, dict(gp, d=0.0, amt=0.0, k=1.0), extras=False), 1.12)
            m = orange_mask(src) * 0.9
            img = cv2.blendLinear(vivid, img, m, 1 - m)
        if self.d.get("lumacap") and fx.luma(img) > self.d["lumacap"]:
            img = fx.match_luma(img, float(self.d["lumacap"]))
        if len(self.cache) > 24:
            for kk in sorted(self.cache)[:8]:
                del self.cache[kk]
        self.cache[k] = img
        return img

    # -- camera
    def pose(self, k):
        d = self.d
        t = k / FPS
        s, rot, dx, dy = 1.0, 0.0, 0.0, 0.0
        if d.get("punch") and k >= 0:
            s0 = d["punch"]
            n30 = t * 30
            s *= fx.punch_scale(n30, s0)
            e = math.exp(-n30 / fx._punch_tau(s0))
            L = math.hypot(*self.wdir) or 1
            dx += self.wdir[0] / L * 40 * e
            dy += self.wdir[1] / L * 40 * e
        if d.get("slow"):
            a, b = d["slow"]
            s *= a + (b - a) * clamp(k / max(self.n - 1, 1))
        if d.get("pulse"):
            s *= 1.15 - 0.15 * fx.ease_out_cubic(k / 16.0)
        if d.get("z2"):
            kk = k - (self.n - 5)
            if kk >= 0:
                for st in (1.02, 1.03, 1.05, 1.07, 1.09)[: min(kk + 1, 5)]:
                    s *= st
        for kh, preset in self.hits:
            jx, jy, jr = shake(k - kh, preset, self.i * 13 + kh)
            dx, dy, rot = dx + jx, dy + jy, rot + jr
        if d.get("hand"):
            hx, hy, hr = fx.handheld_offsets(t, self.i)
            dx, dy, rot = dx + hx, dy + hy, rot + hr
        if abs(rot) > 1e-6 or abs(dx) > 1e-3 or abs(dy) > 1e-3:
            s = max(s, fx.cover_scale(rot, dx, dy))
        return min(s, 1.35), rot, dx, dy

    def in_shake(self, k):
        return any(0 <= k - kh < SHAKE[p][2] for kh, p in self.hits)

    def render(self, k):
        """Graded plate + camera + per-shot fx for frame k of this shot -> (img, scale)."""
        d = self.d
        img = self.plate(k)
        fx.reset_scale()
        # v2: the tilt_plane perspective tilt showed mirror seams (EDL 86/109/110/118); those shots now get
        # refpack_A.tilted_hold in the post layer (pure roll + drift at the covering scale, never any fill)
        sc_fx = fx.used_scale()
        s, rot, dx, dy = self.pose(k)
        if abs(s - 1) > 1e-4 or abs(rot) > 1e-6 or abs(dx) > 1e-3 or abs(dy) > 1e-3:
            blur_kk = [k - kh for kh, p in self.hits if 1 <= k - kh <= 2]
            if blur_kk:
                img = SH.camera(img, s, rot, dx, dy, prev=self.pose(k - 1), shutter=0.5)
            else:
                img = fx.xform(img, s, rot, dx, dy)
        if d.get("pulse"):
            g = 1.0 - 0.40 * clamp(k / max(self.n - 1, 1))
            img = cv2.convertScaleAbs(img, alpha=g)
        if d.get("xp"):
            g = 1.0 - (1.0 - d["xp"]) * clamp(k / max(self.n - 1, 1))
            img = cv2.convertScaleAbs(img, alpha=g)
        for kh, p in self.hits:
            kk = k - kh
            if 0 <= kk < 3:
                img = fx.chroma(img, (8.0 if p == "big" else 5.0) * (1 - kk / 3), angle=20)
        if d.get("pop") and k == 0:
            img = fx.exposure_pop(img, float(d["pop"]))
        if d.get("embers"):
            img = V.embers(img, (self.f0 + k) / FPS, density=d["embers"], seed=self.i)
        if self.tin == "pip" and 0 <= k < 8:      # T11 arrival: full screen at 1.12, radial blur, double exposure
            fx.reset_scale()
            img = fx.pip_arrive(img, k)
            sc_fx = max(sc_fx, fx.used_scale())
        if self.tin in ("hue", "goldpop", "eyedive") and 0 <= k < 2:   # followed by a T3 radial entry
            img = fx.radial_blur(img, 90.0 if k == 0 else 36.0)
        scale = self.bscale * max(s, sc_fx)
        return img, scale

    # -- transitions owned by this (incoming) shot
    def window(self):
        a, c = self.d.get("win", TRANS_WIN.get(self.tin, (0, 0)))
        return self.f0 - a, self.f0 + c


PROTECT = {"A1": 0.9, "A2_iruka": 0.85, "A3_cool": 0.7, "A5_purple": 0.6, "A4": 0.5}


def yellow_mask(img):
    """Soft mask of saturated yellow (hair, Kurama-chakra yellow) in the ungraded frame."""
    sm = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.clip(1 - np.abs(hh - 27.0) / 11.0, 0, 1) * np.clip((ss - 55) / 40, 0, 1) * np.clip((vv - 70) / 50, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)


def orange_mask(img):
    """Soft mask of saturated orange-to-gold (Kurama fur, KCM cloak, fire) in the ungraded frame."""
    sm = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.clip(1 - np.abs(hh - 17.0) / 13.0, 0, 1) * np.clip((ss - 70) / 50, 0, 1) * np.clip((vv - 60) / 50, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)


GOLD_WHITE = np.float32((200, 235, 255))   # BGR


def gold_wash(img, u):
    """Screen the frame toward gold-white by u (0..1); held under luma 200 so it never counts as a white flash."""
    f = img.astype(np.float32)
    out = np.clip(255 - (255 - f) * (1 - u * GOLD_WHITE / 255), 0, 255).astype(np.uint8)
    return fx.match_luma(out, 200.0) if fx.luma(out) > 200 else out


def decode_raw(name, sfps, m0, m1):
    """Source frames m0..m1 of footage/<name>.mp4 as (n, 1080, 1920, 3) BGR (fewer at the clip end)."""
    cmd = ["ffmpeg", "-v", "error"]
    if m0 > 0:
        cmd += ["-ss", f"{(m0 - 0.5) / sfps:.6f}"]
    # passthrough: the default cfr mode duplicates the first frame after a seek (checked against
    # select=eq(n,..)), which shifted every shot by one source frame
    cmd += ["-i", f"{FOOT}/{name}.mp4", "-frames:v", str(m1 - m0 + 1), "-an", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    cnt = len(raw) // (W * H * 3)
    return np.frombuffer(raw, np.uint8)[: cnt * W * H * 3].reshape(cnt, H, W, 3)


def sr_path(key, idx):
    return f"{SR_DIR}/{key}/{idx:06d}.png"


def load_sr(key, m0, m1):
    """Real-ESRGAN plates m0..m1 from sr_cache (stops at the first missing index past m0)."""
    out = []
    for idx in range(m0, m1 + 1):
        img = cv2.imread(sr_path(key, idx), cv2.IMREAD_COLOR)
        if img is None:
            break
        out.append(img)
    if not out:
        raise RuntimeError(f"sr_cache missing {key} @ {m0}: run director.py --sr first")
    return out


def shake(kk, preset, seed):
    if kk < 0:
        return 0.0, 0.0, 0.0
    amp, rot, L = SHAKE[preset]
    if kk >= L:
        return 0.0, 0.0, 0.0
    u, ang, _ = fx._shake_signs(int(seed), 16)
    t = kk / FPS
    et, er = math.exp(-t / (2.5 / 30)), math.exp(-t / (3.0 / 30))
    sg = -1.0 if kk % 2 else 1.0
    a = math.pi / 4 + ang[kk]
    return (amp * et * sg * u[kk, 0] * math.cos(a) * 1.25, amp * et * -sg * u[kk, 1] * math.sin(a),
            rot * er * -sg * u[kk, 2])


def build_shots():
    shots = []
    for i, d in enumerate(E):
        t0 = 0.0 if d["b"] == 0 else bt(d["b"])
        t1 = bt(E[i + 1]["b"]) if i + 1 < len(E) else NF / FPS
        shots.append(Shot(i, d, t0, t1))
    shots[-1].f1 = NF
    shots[-1].n = NF - shots[-1].f0
    return shots


SHOTS = build_shots()
OWNER = np.zeros(NF, np.int32)
for _s in SHOTS:
    OWNER[_s.f0:_s.f1] = _s.i


# ----------------------------------------------------------------------------- v2 reference-effect plan
# Every effect below is a per-frame post layer (FX[j]) applied in frame() after the shot / transition and
# before bars + text, or a transition change (tin) made here before the transition windows are built.
# Times are edit seconds on the song grid; everything from the DROP (19.737) on is the draft's timing.
FX = {}
FXLOG = []


_STARTS = {s.f0 for s in SHOTS}


def fs(t):
    """Frame of edit time t, snapped to a shot start within +-1 frame (plan times are rounded to 1 ms)."""
    f = fr(t)
    for d in (0, 1, -1):
        if f + d in _STARTS:
            return f + d
    return f


def shot_at(t):
    return SHOTS[int(OWNER[fs(t)])]


def addfx(j, order, tag, fn):
    if 0 <= j < NF:
        FX.setdefault(j, []).append((order, tag, fn))


A3, A5 = (19.737, 47.684), (60.835, 80.562)
ACCENT_MED = [21.381, 23.847, 26.313, 32.888, 36.998, 41.108, 43.574, 52.615, 55.081, 57.547, 62.479,
              64.945, 66.589, 72.342]
ACCENT_BIG = [49.328, 75.630, 77.274]
IMPACT_OK_TINS = ("cut", "whip", "radial")


def _in(t, rng):
    return rng[0] - 1e-3 <= t < rng[1] - 1e-3


def _drop_auto_punch(s):
    if s.tin == "whip" and "punch" not in E[s.i]:
        s.d.pop("punch", None)


def v2_plan():
    impact_frames = set()
    glow_frames = set()
    impact_cuts = {}
    # -- 1. impact_cut (refpack_B M1) on the bar-downbeat cuts of chorus 1 + the final chorus
    for t, _, tint in RB.PLAN["impact_cut"]:
        b = fs(t)
        s = SHOTS[int(OWNER[b])]
        if s.f0 != b or s.tin not in IMPACT_OK_TINS:
            FXLOG.append(f"impact_cut skipped at {t} (shot {s.i} tin {s.tin}, f0 {s.f0} vs {b})")
            continue
        _drop_auto_punch(s)
        if s.tin == "radial":
            s.d.pop("punch", None)
        s.tin = "cut"
        short = s.n < 14
        impact_cuts[b] = s.i
        addfx(b - 1, 40, "impact0", lambda img: RB.impact_cut(img, None, 0))
        ks = (1, 2, 3, 4) if not short else (1, 2, 3)
        for kk in ks:
            if kk == 3 and not short:
                # QA flash fix: the near-black dark_neg made a 4th bright/dark pair per package (8-9 transitions
                # in 1 s at 36.2 / 77.3); hold the ink frame at ~0.40 linear luminance instead
                addfx(b + kk - 1, 40, "impact3_inkdim", lambda img, sd=s.i: V2.ink_dim(img, seed=sd))
            else:
                addfx(b + kk - 1, 40, f"impact{kk}",
                      lambda img, kk=kk, tint=RB.TINT[tint], sd=s.i, sh=short: RB.impact_cut(None, img, kk, tint=tint,
                                                                                              seed=sd, short=sh))
            impact_frames.add(b + kk - 1)
        impact_frames.add(b - 1)
        # dMWE RGB split, visible (22 px decaying over 4 frames) on the first clean frames after the package
        r0 = b + (2 if short else 3)
        for n in range(4):
            if r0 + n < s.f1 - 2:
                addfx(r0 + n, 42, "rgb_hit", lambda img, n=n, a=(0.0 if len(impact_cuts) % 2 else 20.0): V2.rgb_hit(img, n, angle=a))
    # (65.767 micro-cut already carries the t12 neon + ink-burst impact frames; no extra neg_glow there)
    # -- 2. accent beats: impact shake (pose hits, SHAKE upgraded to refpack_A MED/BIG) + zoom-blur entry
    for t in ACCENT_MED + ACCENT_BIG:
        f = fs(t)
        s = SHOTS[int(OWNER[f])]
        kh = f - s.f0
        if not any(abs(k - kh) <= 3 for k, _ in s.hits):
            s.hits.append((kh, "big" if t in ACCENT_BIG else "hit"))
        if kh == 0 and f not in impact_cuts and s.tin in ("cut", "whip"):
            _drop_auto_punch(s)
            s.tin = "cut"
            c = s.d.get("eye", (0.5, 0.5))
            for n in (0, 1):
                addfx(f + n, 35, "zoom_entry", lambda img, n=n, c=c: RA.zoom_entry(img, n, c))
    # -- 3. zoom-blur entries on the eye close-ups
    for t in (20.559, 22.203, 48.917, 62.890):
        f = fs(t)
        c = SHOTS[int(OWNER[f])].d.get("eye", (0.5, 0.45))
        if f in impact_frames:
            continue
        for n in (0, 1):
            addfx(f + n, 35, "zoom_entry", lambda img, n=n, c=c: RA.zoom_entry(img, n, c))
    # -- 3b. G3GC eye light trails (ref 10.0-10.3) on the three eye close-ups, 6 frames from k=2 (after the
    #        zoom-blur entry). Eye centres read off gridded v2 frames: (x, y, bolt direction) per k.
    def _lerp_eyes(keys, k):
        ks = sorted(keys)
        if k <= ks[0]:
            return keys[ks[0]]
        for a_, b_ in zip(ks, ks[1:]):
            if a_ <= k <= b_:
                u = (k - a_) / (b_ - a_)
                return [(xa + (xb - xa) * u, ya + (yb - ya) * u, da) for (xa, ya, da), (xb, yb, _) in
                        zip(keys[a_], keys[b_])]
        return keys[ks[-1]]
    trails = [(20.559, {2: [(744, 290, -1), (1289, 373, 1)], 4: [(738, 300, -1), (1282, 400, 1)],
                        6: [(755, 311, -1), (1289, 404, 1)], 7: [(749, 293, -1), (1278, 411, 1)]},
               (60, 175, 255), 1.0),                                   # Sage Naruto: gold-orange
              (48.917, {2: [(905, 605, -1), (905, 605, 1)]}, (40, 80, 255), 1.4),   # Kurama: red-orange
              (62.890, {2: [(1208, 525, 1), (705, 510, -1)]}, (255, 110, 200), 1.0)]  # Sasuke: Rinnegan purple
    for t, keys, col, scl in trails:
        f0 = fs(t)
        for k in range(2, 8):
            addfx(f0 + k, 38, "eye_trails", lambda img, k=k, e=_lerp_eyes(keys, k), c=col, sc=scl, sd=f0:
                  V2.eye_trails(img, k - 2, e, color=c, seed=sd, scale=sc))
    # -- 4. smear whips on every other cut in A3 / A5 (rotating angle)
    angles = (-18.0, 162.0, 72.0, -108.0)
    na = 0
    accents = {fs(t) for t in ACCENT_MED + ACCENT_BIG}
    for s in SHOTS[1:]:
        if not (_in(s.t0, A3) or _in(s.t0, A5)) or s.f0 in impact_cuts or s.f0 in accents:
            continue
        if s.tin not in ("cut", "whip") or s.d.get("stop") or s.n < 6 or SHOTS[s.i - 1].n < 6:
            continue
        if s.f0 == fr(19.737) or s.f0 == fr(48.917):
            continue
        _drop_auto_punch(s)
        s.tin = "smear"
        s.d["smear_angle"] = angles[na % 4]
        na += 1
    # -- 5. the DROP: live drop (zoom entry + BIG shake + echo + decaying glow), n = 0..19
    sd = shot_at(19.737)
    for n in range(20):
        addfx(sd.f0 + n, 2, "drop", lambda img, n=n, s=sd: RA.drop_frame(
            img, n, seed=19, history=[s.render(n - q)[0] for q in (2, 4, 6)] if 2 <= n <= 12 else None))
    # -- 6. THE STOP 47.684-48.917: dip to black, focus rack in, gold duotone + light sweep, white smear out
    st = shot_at(47.684)
    for k in range(4):
        addfx(st.f0 - 4 + k, 50, "dip_black", lambda img, k=k: RB.dip_black(img, k))
    for k in range(st.n - 2):
        addfx(st.f0 + k, 5, "duotone_gold", lambda img: RA.duotone(img, "gold", 0.85))
    for k in range(5):
        addfx(st.f0 + k, 36, "focus_rack", lambda img, k=k: RB.focus_rack_in(img, k, frames=5, sigma0=14.0))
    n_sw = 12
    k_sw = 8
    for q in range(n_sw):
        addfx(st.f0 + k_sw + q, 10, "light_sweep", lambda img, u=q / (n_sw - 1): RA.light_sweep(img, u))
    ku = shot_at(48.917)
    ku.tin = "wsmear"
    ku.d["smear_angle"] = 0.0
    # -- 7. colour hits: refpack_B duotone_hit on the eye / dojutsu shots, dark_ink once, halftone once
    for t, _, pal in RB.PLAN["duotone_hit"]:
        if abs(t - 22.203) < 1e-3:
            continue            # that shot gets the dark-ink eye look instead
        f = fr(t)
        sh_, hi_ = RB.DUO[pal]
        for k in range(7):
            addfx(f + k, 6, "duotone_hit", lambda img, k=k, a=sh_, b=hi_: RB.duotone_hit(img, k, 7, a, b))
    sdk = shot_at(22.203)
    for k in range(sdk.n):
        addfx(sdk.f0 + k, 5, "dark_ink", lambda img: RA.dark_ink(img, 110.0))
    sn = shot_at(23.025)       # Neji Kaiten 2-beat hold
    for k in range(sn.n):
        addfx(sn.f0 + k, 5, "halftone", lambda img: RA.halftone(img, 9, 0.35))
    # -- 8. gold build 18.093-19.737: gold duotone (lighter) + one light sweep 18.6 -> 19.4
    sb_ = shot_at(18.2)
    for j in range(sb_.f0 - 0, fr(19.737)):
        addfx(j, 5, "duotone_gold", lambda img: RA.duotone(img, "gold", 0.55, 1.10))
    f_sw0, f_sw1 = fr(18.6), fr(19.4)
    for j in range(f_sw0, f_sw1 + 1):
        addfx(j, 10, "light_sweep", lambda img, u=(j - f_sw0) / (f_sw1 - f_sw0): RA.light_sweep(img, u))
    # -- 9. high-key hero entries: white whip-in (5 frames) + glow hold (10 frames)
    for t, _ in RB.PLAN["white_whip_in+glow_hold"]:
        f = fs(t)
        s = SHOTS[int(OWNER[f])]
        if s.f0 != f:
            continue
        if s.tin in ("whip", "smear"):
            _drop_auto_punch(s)
            s.tin = "cut"
        s.d.pop("pop", None)
        for k in range(5):
            addfx(f + k, 35, "white_whip_in", lambda img, k=k: RB.white_whip_in(img, k))
        for k in range(5, 15):
            if k < s.n:
                addfx(f + k, 6, "glow_hold", lambda img: RB.glow_hold(img))
                glow_frames.add(f + k)
    # -- 10. 3D cards on the Pain-arc 2-beat holds (q-DyE 23.5-24.3)
    carded = set()
    for t, _, seed in RB.PLAN["card_swing"]:
        s = shot_at(t)
        if s.f0 != fs(t):
            FXLOG.append(f"card skipped at {t}")
            continue
        carded.add(s.i)
        for k in range(s.n):
            addfx(s.f0 + k, 20, "card", lambda img, k=k, n=s.n, sd=seed: RB.card_swing(img, k, n, seed=sd))
    # -- 11. tilted (rolled) holds: every 2+ beat shot in A5 and the Pain stretch 41.1-47.684, plus the shots
    #        that carried tilt_plane (it showed mirror seams). Pure roll + drift at the covering scale.
    for s in SHOTS:
        if s.cid is None or s.i in carded or s.d.get("stop") or s.d.get("ramp"):
            continue
        if s.d.get("tilt") or ((_in(s.t0, A5) or _in(s.t0, (41.1, 47.684))) and s.n >= 19):
            for k in range(s.n):
                addfx(s.f0 + k, 20, "tilted_hold",
                      lambda img, u=k / max(s.n - 1, 1), si=s.i: RA.tilted_hold(img, u, si))
    # -- 12. in-shot beat smears on the beats inside 2+ beat holds (not on crawl / stop shots)
    for q, (t, _) in enumerate(RB.PLAN["beat_smear"]):
        f = fr(t)
        s = SHOTS[int(OWNER[f])]
        if s.d.get("ramp") or s.d.get("stop") or f - s.f0 < 3 or s.f1 - f < 3:
            continue
        if any(f + n in impact_frames for n in range(3)):
            continue
        if f in glow_frames:     # v2 QA: no beat smear inside the white-whip glow hold (7 transitions at 61.6-62.1)
            continue
        for n in range(3):
            addfx(f + n, 30, "beat_smear", lambda img, n=n, sd=q + 7: RB.beat_smear(img, n, seed=sd))
    # -- 13. mono hit inside the Rasenshuriken slam (bar downbeat, not a cut)
    f = fr(37.820)
    for k in range(4):
        addfx(f + k, 30, "mono_hit", lambda img, k=k: RB.mono_hit(img, k))
    # -- 14. zoom-through on the last 3 frames into the clash (75.630) and Neji -> the launch (23.847)
    for t in (75.630, 23.847):
        sb2 = shot_at(t)
        sa2 = SHOTS[sb2.i - 1]
        sa2.d.pop("z2", None)
        if sb2.tin in ("whip", "smear"):
            _drop_auto_punch(sb2)
            sb2.tin = "cut"
        for k in range(3):
            addfx(sb2.f0 - 3 + k, 37, "zoom_through", lambda img, k=k: RA.zoom_through(img, k))
    # -- 15. echo trails: only inside the drop (drop_frame). Tried on the fist into Haku (22.614) and the
    #        launch (23.847): both plates turned to mush on stills, so they were taken out.
    # -- 16. soft glow on the Six Paths PIP hold (60.835)
    s = shot_at(60.9)
    for k in range(8, s.n):
        addfx(s.f0 + k, 6, "soft_glow", lambda img: RA.soft_glow(img, 0.40))
    # -- 16b. frame 1075 (card + beat smear) showed a reflected strip ~13 px in from the left edge: crop in 2%
    addfx(1075, 45, "edge_crop", lambda img: fx.xform(img, 1.02))
    # -- 17. dialogue holds: light dutch drift (MBoD 48.3) on the sync shots and Iruka at 82.206
    for q, s in enumerate([x for x in SHOTS if x.d.get("sync")] + [shot_at(82.3)]):
        ang = -2.5 if q % 2 == 0 else 2.5
        for k in range(s.n):
            addfx(s.f0 + k, 20, "dutch_drift", lambda img, u=k / max(s.n - 1, 1), a=ang: RB.dutch_drift(img, u, a, 40.0))


_ECHO_OK = {}


def _echo(s, k, img):
    """refpack_A echo trail, only on plates with raw Laplacian > 600 (soft plates go to mush under it)."""
    if s.i not in _ECHO_OK:
        g = cv2.cvtColor(cv2.resize(s.plate(max(0, s.n // 2)), (480, 270), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        _ECHO_OK[s.i] = float(cv2.Laplacian(g, cv2.CV_64F).var()) > 600
    if not _ECHO_OK[s.i]:
        return img
    return RA.echo_trail(img, [s.render(k - q)[0] for q in (2, 4, 6)])


v2_plan()
WINDOWS = {}
for _s in SHOTS[1:]:
    a, c = _s.window()
    for j in range(a, c):
        assert j not in WINDOWS, f"transition windows overlap at frame {j}"
        WINDOWS[j] = _s.i


# ----------------------------------------------------------------------------- frame assembly
def transition(B, j):
    """Frame j inside the transition window of incoming shot B. -> (img, scale, tag)"""
    A = SHOTS[B.i - 1]
    b = B.f0
    ka, kb = j - A.f0, j - B.f0
    t = B.tin
    if t == "whip":
        kk = j - (b - 3)
        if kk < 2:
            a, sa = A.render(ka)
            return fx.whip_out(a, kk, B.wdir), sa, "whip-out"
        return fx.whip_in(B.plate(0), B.wdir), B.bscale * 1.28, "whip-in"
    if t == "radial":
        kk = j - (b - 2)
        a, sa = A.render(ka)
        if A.d.get("z2"):
            return fx.radial_blur(a, 90.0 if kk == 0 else 160.0), sa, "radial-out"
        fx.reset_scale()
        out = fx.radial_collapse(a, kk, frames=2, s_end=1.3)
        return out, A.bscale * max(sa / A.bscale, fx.used_scale()), "radial-out"
    if t == "hue":
        kk = j - (b - 3)
        a, sa = A.render(ka)
        return fx.hue_flash(a, kk), sa, "hue-flash"
    if t == "goldpop":   # replaces the T5 hue flash (it turned fists pink / neon green): gold-white wash ramp
        kk = j - (b - 3)
        a, sa = A.render(ka)
        return gold_wash(a, (0.22, 0.40, 0.58)[kk]), sa, "gold-pop"
    if t == "eyedive":   # replaces the vector-eye iris: the camera dives into the real eye, B enters radially
        kk = j - (b - 3)
        a, sa = A.render(ka)
        c = A.d.get("eye", (0.5, 0.5))
        s = (1.10, 1.22, 1.35)[kk]
        out = fx.radial_blur(fx.xform(a, s, center=c), (30.0, 80.0, 160.0)[kk], center=c)
        return out, sa * s, "eye-dive"
    if t == "iris":
        n_ir = B.f0 - B.window()[0]
        kk = j - (b - n_ir)
        a, sa = A.render(ka)
        bb, sb = B.render(kb)
        fx.reset_scale()
        out = fx.iris(a, bb, kk, frames=n_ir, center=A.d.get("eye", (0.5, 0.5)), center_b=(0.5, 0.5),
                      eye=A.d.get("eyekind", "3tomoe"))
        return out, max(sa, sb, fx.used_scale()), "iris"
    if t == "t12":
        kk = j - (b - 1)
        if kk == 0:
            a, sa = A.render(ka)
            return fx.neon_edges(a), sa, "t12-neon"
        if kk == 1:   # paper a shade off-white so the burst frame never counts as a white flash
            return fx.ink_burst(B.plate(0), seed=B.i, paper=(218, 223, 228)), B.bscale, "t12-ink"
        n_settle = B.window()[1] - 1
        bb, sb = B.render(kb)
        fx.reset_scale()
        out = fx.settle(bb, kk - 2, frames=n_settle, blur=(0.30, 0.16, 0.07, 0.02)[-n_settle:] if n_settle < 4
                        else (0.30, 0.16, 0.07, 0.02))
        return out, sb * fx.used_scale(), "t12-settle"
    if t == "xd":
        kk = j - (b - 3)
        a, sa = A.render(ka)
        bb, sb = B.render(kb)
        return fx.cross_dissolve(a, bb, kk, frames=6), max(sa, sb), "dissolve"
    if t == "bd":
        kk = j - b
        a, sa = A.render(ka)
        bb, sb = build_frame(B, kb)
        return fx.blur_dissolve(a, bb, kk, frames=11), max(sa, sb), "blur-dissolve"
    if t == "pip":
        kk = j - (b - 10)
        a, sa = A.render(ka)
        out = fx.pip(a, B.plate(0), kk, frames=10, grow_frames=5, focus=(0.5, 0.42))
        return out, sa, "pip-rise"
    if t == "white":
        kk = j - b
        bb, sb = B.render(kb)
        if kk == 0:
            return fx.lens_dirt(fx.white_flash(bb), 0.3, seed=B.i), sb, "white-flash"
        return fx.from_white(bb, kk - 1, amounts=(0.7, 0.3)), sb, "from-white"
    if t == "strobe":   # one dark frame per beat (35% of the image, never black) -> <= 3 flashes/s
        kk = j - b
        bb, sb = B.render(kb)
        fx.reset_scale()
        if kk == 0:
            out = cv2.convertScaleAbs(fx.xform(bb, 1.04), alpha=0.35)
            return out, sb * 1.04, "strobe-dark"
        out = fx.chroma(fx.strobe(bb, 1, jolt=6.0, seed=B.i), 5.0, angle=20)
        return out, sb * fx.used_scale(), "strobe"
    if t in ("smear", "wsmear"):   # v2 refpack_A: 2-frame smear whip, outgoing on b-2, incoming on b-1, b clean
        kk = j - (b - 2)
        ang = B.d.get("smear_angle", -18.0)
        if kk == 0:
            img, sc = A.render(ka)
        else:
            img, sc = B.render(-1)
        if t == "wsmear":
            return RA.white_smear(img, kk, angle=ang), sc * RA.SMEAR_SCALE[kk], "white-smear"
        return RA.smear_whip(img, kk, angle=ang), sc * RA.SMEAR_SCALE[kk], "smear"
    raise ValueError(t)


def build_frame(B, k):
    """The build plate (KCM Naruto, 18.093 -> 19.737): gold bloom 19.00-19.67, 2-frame whip-zoom."""
    img, s = B.render(k)
    t = (B.f0 + k) / FPS
    if t >= 19.0:   # overexposure build; held under luma ~195 so it never reads as a white flash frame
        u = clamp((t - 19.0) / 0.67)
        img = fx.gold_bloom(img, u, strength=0.55, target=185.0, rays_from=B.d.get("rays", (0.5, 0.36)), rays=0.4)
    kk = k - (B.n - 2)
    if kk >= 0:
        fx.reset_scale()
        img = fx.radial_collapse(img, kk, frames=2, s_end=1.3, center=(0.5, 0.42))
        img = fx.dir_blur(img, -45.0, 30.0 + 30.0 * kk)
        s = s * fx.used_scale()
        if fx.luma(img) > 196:     # the whip smear of the bloom must not read as a white flash frame
            img = fx.match_luma(img, 196.0)
    return img, s


def card_frame(C, j):
    t = j / FPS - 88.782
    if j == C.f0:
        P = SHOTS[C.i - 1]
        img, _ = P.render(j - P.f0)
        img = apply_bars(img, j)
        return birthday_card_v2(img, max(t, 0.0)), "card-flash"
    img = birthday_card_v2(np.zeros((H, W, 3), np.uint8), max(t, 1.0 / 30))
    return CR.composite(img, j / FPS - CR.T0), "card"


def birthday_card_v2(frame, t):
    """titles.birthday_card without the old handle lockup + IG glyph echo (v2: the credit row replaces it)."""
    k = t * FPS
    if int(round(k)) <= 0:
        f = frame.astype(np.float32)
        gold = np.array((215, 240, 255), np.float32)
        return TT._u8(f * 0.12 + gold * 0.88 + 8)
    edit_t = 88.782 + t
    fade = 1 - TT.ease_in_out_sine((edit_t - TT.FADE0) / (TT.FADE1 - TT.FADE0))
    base = np.zeros((H, W, 3), np.float32)
    if fade <= 0:
        return TT._u8(base)
    cv = TT._Canvas()
    for wd in TT._bday_lines(W / 2, 460, card=True):
        TT._draw_word(cv, wd['face'], wd['text'], wd['cap'], wd['tr'], wd['x'], wd['base'],
                      TT.T3_COL, TT.T3_RED, '!', 1.0)
    u = clamp((k - 2) / 14)
    if 0 < u < 1:
        xs = np.arange(W, dtype=np.float32)[None, :]
        ys = np.arange(H, dtype=np.float32)[:, None]
        pos = -300 + u * (W + 600)
        band = np.exp(-((xs + 0.35 * (ys - 540) - pos) / 70.0) ** 2)
        cv.P *= (1 + 0.9 * band)[..., None]
    return TT._u8(cv.flatten(base, glow=TT.T3_GLOW, opacity=fade))


SLIT = np.array((0,) + fx.SLIT, np.float32)


def bars_for(j):
    t = j / FPS
    f30 = max(t, 1.0 / FPS) * 30      # v2 QA: frame 0 opens like frame 1 (was pure black; IG cover frame)
    if f30 < 23:
        return "slit", float(np.interp(f30, np.arange(len(SLIT)), SLIT))
    if t < 19.24:
        return "bar", 78.0
    if t < 19.737:   # fully retracted one frame before the drop frame (f473 kept 2 px lines before)
        u = (t - 19.24) / (19.737 - 1.0 / FPS - 19.24)
        return "bar", 78.0 * (1 - fx.ease_in_quad(u))
    if 85.022 <= t < 88.782:
        u = (t - 85.022) / (85.494 - 85.022)
        return "bar", 78.0 * fx.ease_in_out(u)
    return "bar", 0.0


def apply_bars(img, j):
    kind, v = bars_for(j)
    if kind == "slit":
        return fx.band(img, v) if v > 0 else np.zeros_like(img)
    return fx.letterbox(img, v) if v > 0.01 else img


def post(j, img, info):
    """v2 reference-effect layer (FX built by v2_plan), lowest order first."""
    tags = []
    for order, tag, fn in sorted(FX.get(j, []), key=lambda x: x[0]):
        img = fn(img)
        tags.append(tag)
    if tags:
        info["fx"] = tags
    return img


# v2 subtitles for the original voice lines (refpack_A.subtitle, fUx4 4.3-8.3). Onset = speech_t0 from
# voices/mix_placement.json; each line fades when its speech ends (or just before the next line types in).
_PLACED = {p["id"]: p for p in json.load(open(f"{WORK}/voices/mix_placement.json"))["placed"]}
SUBS = [("L1", ["IT'S NOT LIKE THERE'S ANYONE", "WAITING AT HOME FOR ME!"], "naruto"),
        ("L2", ["YOU FEEL LONELY", "AND IT HURTS INSIDE."], "iruka"),
        ("L3a", ["BUT THAT'S NOT WHO NARUTO IS."], "iruka"),
        ("L3b", ["HE'S NARUTO UZUMAKI."], "iruka"),
        ("L4", ["THE NEXT HOKAGE!"], "naruto")]


def _sub_spans():
    out = []
    for q, (lid, lines, spk) in enumerate(SUBS):
        p = _PLACED[lid]
        j0 = fr(p["speech_t0"])
        j_end = fr(p["speech_t1"]) + 4
        fo = 8
        if q + 1 < len(SUBS):
            nxt = fr(_PLACED[SUBS[q + 1][0]]["speech_t0"])
            if j_end + fo > nxt:
                fo = 4
                j_end = min(j_end, nxt - fo)
        out.append((j0, j_end, fo, lines, RA.SPEAKER[spk]))
    return out


SUB_SPANS = _sub_spans()


def finish(j, img, sh):
    """Global layers: letterbox / slit, dialogue subtitles, birthday lower third, credit row.
    v2: the typographic story moments (alone / iruka / ramen / uzumaki) are gone -- the voices say them."""
    img = apply_bars(img, j)
    t = j / FPS
    for j0, j_end, fo, lines, col in SUB_SPANS:
        if j0 <= j <= j_end + fo:
            img = RA.subtitle(img, lines, j - j0, j_end - j0, fade_out=fo, color=col)
    if 86.316 - 1e-6 <= t < 88.782 - 1e-6:
        tb = t - 86.316
        if j == fr(87.138):
            tb = 25.0 / 30.0      # land NARUTO (and its 1-frame pop) exactly on this frame
        img = TT.birthday_lower(img, tb, cx=BDAY_CX, top_base=BDAY_TOP, backing=1.5)
    if CR.T0 <= t < CR.FADE1:
        img = CR.composite(img, t - CR.T0)
    return img


def frame(j):
    """-> (BGR uint8 frame, info dict) for edit frame j."""
    sh = SHOTS[OWNER[j]]
    info = dict(j=j, shot=int(sh.i), tag="held")
    if j in WINDOWS:
        B = SHOTS[WINDOWS[j]]
        img, sc, tag = transition(B, j)
        info["tag"] = tag
        info["shot"] = int(B.i) if j >= B.f0 else int(B.i - 1)
    elif sh.kind == "card":
        img, tag = card_frame(sh, j)
        info["tag"] = tag
        return img, dict(info, scale=1.0)
    elif sh.kind == "build":
        img, sc = build_frame(sh, j - sh.f0)
        info["tag"] = "build"
    else:
        k = j - sh.f0
        img, sc = sh.render(k)
        if (sh.d.get("punch") and k < 3) or sh.in_shake(k) or (sh.d.get("pulse")) or (sh.d.get("pop") and k == 0):
            info["tag"] = "decay" if not sh.d.get("pulse") else "pulse"
        if sh.tin == "pip" and k < 3:
            info["tag"] = "pip-arrive"
        if sh.d.get("z2") and k >= sh.n - 5:
            info["tag"] = "z2-push"
    img = post(j, img, info)
    img = finish(j, img, sh)
    info["scale"] = round(float(sc), 4)
    return np.ascontiguousarray(img), info


# ----------------------------------------------------------------------------- rendering
def needs(j):
    """Shot ids whose source frames frame j reads."""
    ids = {int(OWNER[j])}
    if j in WINDOWS:
        ids |= {WINDOWS[j], WINDOWS[j] - 1}
    s = SHOTS[int(OWNER[j])]
    if s.kind == "card" and j == s.f0:
        ids.add(s.i - 1)
    return {i for i in ids if SHOTS[i].cid}


def render_chunk(a, b, out):
    t0 = time.time()
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", f"{FPS_N}/{FPS_D}", "-i", "-", "-c:v", "libx264rgb", "-preset", "ultrafast", "-qp", "0", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    log = []
    first_use, last_use = {}, {}
    for j in range(a, b):
        for i in needs(j):
            first_use.setdefault(i, j)
            last_use[i] = j
    ex = ThreadPoolExecutor(2)
    futs = {}
    for j in range(a, b):
        for i in sorted(first_use):
            if i not in futs and first_use[i] <= j + 24 and last_use[i] >= j:
                futs[i] = ex.submit(SHOTS[i].decode)
        for i in needs(j):
            futs[i].result()
        img, info = frame(j)
        sm = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        info["luma"] = round(float(g.mean()), 1)
        info["lap"] = round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)
        log.append(info)
        p.stdin.write(img.tobytes())
        for i in list(futs):
            if last_use[i] <= j:
                futs.pop(i).result()
                SHOTS[i].release()
                last_use[i] = -1
        if (j - a) % 48 == 0:
            print(f"[{a}-{b}] frame {j} {time.time() - t0:.0f}s", flush=True)
    p.stdin.close()
    rc = p.wait()
    json.dump(log, open(out + ".json", "w"))
    print(f"[{a}-{b}] done in {time.time() - t0:.0f}s rc={rc}", flush=True)
    return rc


def write_edl():
    rows = []
    for s in SHOTS:
        d = s.d
        cname = CLIPS[s.cid][0] if s.cid else None
        row = dict(i=s.i, kind=s.kind, t0=round(s.t0, 4), t1=round(s.t1, 4), f0=s.f0, f1=s.f1, frames=s.n,
                   beat=d["b"], clip=cname, src_in=d.get("src"), src_out=(round(s.src_time(s.n), 3) if s.cid else None),
                   reverse=bool(d.get("rev")), act=d.get("act"), transition_in=s.tin, note=d.get("note", ""))
        for key in ("punch", "hits", "pop", "slow", "pulse", "xp", "tilt", "embers", "ramp", "z2", "hand", "wdir",
                    "yoff", "zoom", "lumacap", "speed", "eye"):
            if key in d:
                row[key] = d[key]
        if s.cid:
            row["real_esrgan"] = bool(s.sr_key())
            row["base_scale"] = round(s.bscale, 4)
        rows.append(row)
    meta = dict(fps=f"{FPS_N}/{FPS_D}", frames=NF, duration=round(NF / FPS, 4), audio=SONG_WAV,
                frame_of_time="round(t * 24000/1001)", shots=len(rows), cuts=len(rows) - 1)
    json.dump(dict(meta=meta, shots=rows), open(EDL_PATH, "w"), indent=1, ensure_ascii=False)
    return EDL_PATH


def sr_zoom(img, zoom, SRM):
    """(cx, cy, z): the W/z x H/z source window around (cx, cy), Real-ESRGAN x4, back to 1920x1080."""
    cx, cy, z = zoom
    w, h = int(round(W / z)), int(round(H / z))
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    return cv2.resize(SRM.x4(np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])), (W, H), interpolation=cv2.INTER_AREA)


def precompute_sr():
    """Real-ESRGAN every source frame the SR shots read -> sr_cache/<key>/<idx>.png (existing files are kept)."""
    sys.path.insert(0, f"{WORK}/tools")
    import sr as SRM
    todo = {}
    for s in SHOTS:
        key = s.sr_key()
        if key:
            todo.setdefault(key, (s, []))[1].append(s.src_range())
    t0, done = time.time(), 0
    for key, (s, ranges) in todo.items():
        name, fmt, sfps, _ = CLIPS[s.cid]
        os.makedirs(f"{SR_DIR}/{key}", exist_ok=True)
        for m0, m1 in sorted(ranges):
            if all(os.path.exists(sr_path(key, i)) for i in range(m0, m1 + 1)):
                continue
            for i, img in enumerate(decode_raw(name, sfps, m0, m1)):
                p = sr_path(key, m0 + i)
                if os.path.exists(p):
                    continue
                if s.d.get("zoom"):
                    out = sr_zoom(img, s.d["zoom"], SRM)
                elif fmt == "43":
                    out = SRM.sr43(img)
                else:
                    out = SRM.sr169(img)
                cv2.imwrite(p, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                done += 1
        print(f"sr {key}: {len(ranges)} shots, {done} new frames in total, {time.time() - t0:.0f}s", flush=True)
    return 0


def render_all(jobs=6):
    write_edl()
    os.makedirs(f"{TMP}/chunks", exist_ok=True)
    cuts = [s.f0 for s in SHOTS]
    bounds = [0]
    for q in range(1, jobs):
        target = NF * q / jobs
        bounds.append(min(cuts, key=lambda c: abs(c - target)))
    bounds.append(NF)
    bounds = sorted(set(bounds))
    procs = []
    t0 = time.time()
    for a, b in zip(bounds, bounds[1:]):
        out = f"{TMP}/chunks/c_{a:05d}_{b:05d}.mkv"
        procs.append((out, subprocess.Popen([sys.executable, os.path.abspath(__file__), "--chunk", str(a), str(b), out])))
    rcs = [p.wait() for _, p in procs]
    print("chunks rc", rcs, f"{time.time() - t0:.0f}s", flush=True)
    if any(rcs):
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    lst = f"{TMP}/chunks/list.txt"
    with open(lst, "w") as f:
        for out, _ in procs:
            f.write(f"file '{out}'\n")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", SONG_WAV,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-r", f"{FPS_N}/{FPS_D}", "-af", "volume=-0.5dB",   # v2 QA: margin for the AAC true-peak overshoot
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", OUT]
    rc = subprocess.run(cmd).returncode
    logs = []
    for out, _ in procs:
        logs += json.load(open(out + ".json"))
    json.dump(logs, open(f"{TMP}/frame_log.json", "w"))
    print(f"final rc={rc} -> {OUT} total {time.time() - t0:.0f}s", flush=True)
    return rc


def stills(js):
    os.makedirs(f"{TMP}/stills", exist_ok=True)
    for j in js:
        img, info = frame(j)
        cv2.imwrite(f"{TMP}/stills/f{j:05d}.png", img)
        print(j, info, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", nargs=3)
    ap.add_argument("--stills")
    ap.add_argument("--edl", action="store_true")
    ap.add_argument("--sr", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    if a.sr:
        sys.exit(precompute_sr())
    if a.chunk:
        sys.exit(render_chunk(int(a.chunk[0]), int(a.chunk[1]), a.chunk[2]))
    if a.edl:
        print(write_edl())
        sys.exit(0)
    if a.stills:
        stills([int(x) for x in a.stills.split(",")])
        sys.exit(0)
    sys.exit(render_all(a.jobs))
