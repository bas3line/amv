"""Flash counter on a rendered mp4 (or a frame range of it).

Global: whole-frame mean of linearised luminance per frame. Consecutive frame deltas are merged into
monotonic runs; a run is a transition when |change| >= 0.10 and the darker end < 0.80. Count the
transitions starting inside every 1 s window (24 frames). >6 transitions = >3 flashes.
Area: per-pixel linear luminance; a frame step is an UP (DOWN) transition when >= 25% of the pixels rise
(fall) by >= 0.10 with the darker value < 0.80; consecutive same-direction steps merge.
usage: flash.py video.mp4 [t0 t1]
"""
import os
import subprocess
import sys

import numpy as np

FPS = 24000 / 1001
w, h = 240, 135
src = sys.argv[1]
cmd = [os.environ.get("FFMPEG", "ffmpeg"), "-v", "error", "-i", src, "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo",
       "-pix_fmt", "rgb24", "-"]
raw = subprocess.run(cmd, capture_output=True, check=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255.0
lin = np.where(fr <= 0.04045, fr / 12.92, ((fr + 0.055) / 1.055) ** 2.4)
Y = lin @ np.float32([0.2126, 0.7152, 0.0722])          # (n, h, w)
L = Y.reshape(len(Y), -1).mean(1)
n = len(L)


def runs(series_dir):
    """series_dir: list of (frame, direction, ok) per step -> transition start frames (merged runs)."""
    out = []
    last = 0
    for f, d, ok in series_dir:
        if not ok:
            last = 0
            continue
        if d != last:
            out.append((f, d))
        last = d
    return out


# global: strict mode merges strictly monotonic runs only (any tiny reversal splits a transition);
# zigzag mode (default) ignores reversals smaller than 0.10 (extrema with 0.10 hysteresis).
STRICT = "--strict" in sys.argv
if STRICT:
    sys.argv.remove("--strict")
tr_g = []
if STRICT:
    i = 0
    while i < n - 1:
        d = np.sign(L[i + 1] - L[i])
        if d == 0:
            i += 1
            continue
        j = i
        while j < n - 1 and np.sign(L[j + 1] - L[j]) == d:
            j += 1
        lo, hi = min(L[i], L[j]), max(L[i], L[j])
        if hi - lo >= 0.10 and lo < 0.80:
            tr_g.append((i + 1, int(d)))
        i = j
else:
    ext_i, d = 0, 0           # last confirmed extremum index, current leg direction
    cand = 0                  # running extreme of the current leg
    for i in range(1, n):
        if d == 0:
            if abs(L[i] - L[ext_i]) >= 0.10:
                d = 1 if L[i] > L[ext_i] else -1
                cand = i
            continue
        if (L[i] - L[cand]) * d > 0:
            cand = i
        elif abs(L[i] - L[cand]) >= 0.10:     # reversal: the leg ext_i -> cand is complete
            lo = min(L[ext_i], L[cand])
            if lo < 0.80:
                tr_g.append((ext_i + 1, d))
            ext_i, cand, d = cand, i, -d
    if d and abs(L[cand] - L[ext_i]) >= 0.10 and min(L[ext_i], L[cand]) < 0.80:
        tr_g.append((ext_i + 1, d))
# area
tr_a = []
last = 0
for i in range(n - 1):
    a, b = Y[i], Y[i + 1]
    dk = np.minimum(a, b) < 0.80
    up = ((b - a >= 0.10) & dk).mean() >= 0.25
    dn = ((a - b >= 0.10) & dk).mean() >= 0.25
    d = 1 if up and not dn else (-1 if dn and not up else (0 if not up else 2))
    if d == 2:       # both: count as one opposite of the last
        d = -last if last else 1
    if d and d != last:
        tr_a.append((i + 1, d))
    last = d if d else 0


def windows(tr, label):
    fs = np.array([f for f, _ in tr])
    worst = []
    for s in range(n):
        c = int(((fs >= s) & (fs < s + 24)).sum())
        worst.append(c)
    worst = np.array(worst)
    bad = np.where(worst > 6)[0]
    print(f"{label}: transitions={len(tr)} max/1s={worst.max()} at {worst.argmax() / FPS:.3f}s; "
          f"windows>6: {len(bad)} starts", end="")
    if len(bad):
        groups = np.split(bad, np.where(np.diff(bad) > 1)[0] + 1)
        print(" ->", ", ".join(f"{g[0] / FPS:.2f}-{g[-1] / FPS:.2f}(max {worst[g].max()})" for g in groups))
    else:
        print()
    return worst


print(f"frames {n}")
wg = windows(tr_g, "global")
wa = windows(tr_a, "area")
if len(sys.argv) > 3:
    t0, t1 = float(sys.argv[2]), float(sys.argv[3])
    for f in range(int(t0 * FPS), int(t1 * FPS) + 1):
        print(f"{f} {f / FPS:.3f} L={L[f]:.2f} winG={wg[f]} winA={wa[f]}")
