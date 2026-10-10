"""Velocity edits: speed ramps, smooth slow motion and vector motion blur.

    import motion
    n_src = motion.source_frames_needed(n_out, "velocity")      # decode this many
    out = motion.remap(src_frames, n_out, "velocity", interp="flow")

or lazily, one output frame at a time (for the director's per-frame loop):

    rc = motion.RampClip(src_frames, n_out, "ramp_in")
    img = rc.frame(k)

Frames are uint8 BGR (H, W, 3); everything is deterministic.

Design notes
- A curve is a speed profile v(u) over output time u in [0, 1] (source frames
  advanced per output frame); positions are its running sum. Presets live in
  PRESETS; a curve can also be a list of (u, speed) keys or a callable. Keys are
  interpolated in log-speed with smoothstep, so 3x -> 0.25x decelerates the way
  the eye expects instead of linearly.
- Anime is drawn on 2s/3s and the 24->30 pulldown in these sources adds exact
  duplicates. Retimer finds the distinct drawings and interpolates between
  drawings, never between a drawing and its own duplicate, so slow motion moves
  on every output frame instead of stuttering.
- interp='flow': DIS optical flow at reduced resolution, both directions, the
  Super-SloMo intermediate-flow approximation, forward/backward consistency +
  photometric checks as per-pixel occlusion weights, and weight sharpening so
  mis-registered pixels resolve to one source instead of a ghost. A pair whose
  flow is not trustworthy (cut, flash, huge motion) falls back to a crisp
  nearest-drawing switch.
- Fast parts get vector motion blur: the frame is smeared along its own flow at
  half resolution, with length = per-frame motion x source frames exposed by
  the shutter, and composited back only where things actually move so static
  areas stay full-res sharp. Accumulation is in (approximately) linear light so
  lightning streaks stay hot instead of greying out.
"""
from collections import OrderedDict
import bisect
import math

import cv2
import numpy as np

__all__ = ["PRESETS", "Curve", "make_curve", "positions", "source_frames_needed", "Retimer", "RampClip",
           "remap"]

# ----------------------------------------------------------------------------- curves

# (u, speed) keys; speed is source frames per output frame (negative = reverse).
PRESETS = {
    # fast 3x, braking hard into 0.25x for the last third: lands on the hit.
    "ramp_in": [(0.0, 3.0), (0.30, 3.0), (0.62, 0.25), (1.0, 0.25)],
    # slow-mo release that whips away into speed.
    "ramp_out": [(0.0, 0.25), (0.38, 0.25), (0.70, 3.0), (1.0, 3.0)],
    # classic AMV velocity: fast - slow - fast.
    "velocity": [(0.0, 3.0), (0.22, 3.0), (0.40, 0.25), (0.60, 0.25), (0.78, 3.0), (1.0, 3.0)],
    # fast, then a dead freeze on the last frame with a micro push-in.
    "hit_freeze": [(0.0, 3.0), (0.50, 3.0), (0.60, 0.0), (1.0, 0.0)],
    # drift forward in slow-mo, then rewind hard to the in-point (the reverse
    # speed is rescaled so the rewind lands exactly on frame 0; ~-3x here).
    "reverse_snap": [(0.0, 0.6), (0.76, 0.6), (0.84, -4.0), (1.0, -4.0)],
}
PUSH = {"hit_freeze": (0.60, 1.045)}  # preset -> (u where the push starts, end scale)
FLOOR = 1e-3  # log-space floor for zero speed


def _smooth(x):
    return x * x * (3 - 2 * x)


class Curve:
    """Speed profile v(u) plus an optional push-in scale(u)."""

    def __init__(self, keys=None, fn=None, push=None, name="custom"):
        self.keys, self.fn, self.push, self.name = keys, fn, push, name

    def speed(self, u):
        if self.fn is not None:
            return float(self.fn(u))
        ks = self.keys
        if u <= ks[0][0]:
            return ks[0][1]
        for (u0, v0), (u1, v1) in zip(ks, ks[1:]):
            if u <= u1:
                a = _smooth((u - u0) / max(u1 - u0, 1e-9))
                if v0 * v1 > 0:  # same sign: interpolate in log space
                    s = math.copysign(1, v0)
                    return s * math.exp((1 - a) * math.log(abs(v0)) + a * math.log(abs(v1)))
                if v0 == 0 and v1 == 0:
                    return 0.0
                if v0 == 0 or v1 == 0:  # to/from a freeze: log ramp to the floor
                    lo, hi = (v1, v0) if v1 == 0 else (v0, v1)
                    b = a if v1 == 0 else 1 - a
                    s = math.copysign(1, hi)
                    v = s * math.exp((1 - b) * math.log(abs(hi)) + b * math.log(FLOOR))
                    return 0.0 if abs(v) <= FLOOR * 1.0001 else v
                return (1 - a) * v0 + a * v1  # sign change: linear through zero
        return ks[-1][1]

    def scale(self, u):
        if not self.push:
            return 1.0
        u0, s1 = self.push
        if u <= u0:
            return 1.0
        x = (u - u0) / max(1 - u0, 1e-9)
        return 1.0 + (s1 - 1.0) * (1 - (1 - x) ** 3)


def make_curve(curve, **kw):
    """curve: preset name | list of (u, speed) | callable u->speed | Curve.
    kw for presets: fast=, slow= rescale the preset's fast (>1) / slow (<1) keys;
    push=(u0, scale) overrides the push-in."""
    if isinstance(curve, Curve):
        return curve
    if callable(curve):
        return Curve(fn=curve, push=kw.get("push"))
    if isinstance(curve, str):
        keys = [list(k) for k in PRESETS[curve]]
        fast, slow = kw.get("fast"), kw.get("slow")
        for k in keys:
            if fast and abs(k[1]) > 1:  # presets are authored around 3x; scale proportionally
                k[1] = k[1] * fast / 3.0
            elif slow and 0 < abs(k[1]) < 1:
                k[1] = math.copysign(slow, k[1])
        return Curve([tuple(k) for k in keys], push=kw.get("push", PUSH.get(curve)), name=curve)
    return Curve(sorted(tuple(k) for k in curve), push=kw.get("push"))


def _speeds(n_out, c):
    return np.array([c.speed((k + 0.5) / n_out) for k in range(n_out)], np.float64)


def positions(n_out, curve, n_src=None, start=0.0, **kw):
    """Fractional source index for each output frame (and the speed there).
    With n_src, the curve is scaled to consume exactly the clip: its furthest
    position lands on frame n_src-1 (so speeds become relative)."""
    c = make_curve(curve, **kw)
    v = _speeds(n_out, c)
    neg = v < 0
    if neg.any():  # rewinds land exactly back on the in-point, never before it
        fwd, back = v[~neg].sum(), -v[neg].sum()
        v = np.where(neg, v * (fwd / max(back, 1e-9)), v)
    p = np.concatenate([[0.0], np.cumsum(v)[:-1]])
    p = np.maximum(p, 0.0)
    if n_src is not None:
        reach = max(p.max(), 1e-9)
        k = (n_src - 1 - start) / reach
        p, v = p * k, v * k
    return p + start, v


def source_frames_needed(n_out, curve, **kw):
    """Frames to decode from the in-point so every output frame (and its motion
    blur window) has real source to read."""
    p, v = positions(n_out, curve, **kw)
    return int(math.ceil(p.max() + np.abs(v).max())) + 2


# ----------------------------------------------------------------------------- retimer

class Retimer:
    """Samples a decoded clip at fractional source times.

    frames: list/array of uint8 BGR frames (all the same size).
    flow_w: width of the flow field (height follows aspect); 480 is ~4x down at 1080p.
    """

    def __init__(self, frames, flow_w=480, hold_thresh=0.25, cut_ratio=4.0, cut_min=20.0, max_span=3,
                 conf_min=0.55, cache=24):
        self.f = frames
        self.n = len(frames)
        H, W = frames[0].shape[:2]
        self.H, self.W = H, W
        self.fw, self.fh = flow_w, int(round(flow_w * H / W))
        self.sx, self.sy = W / self.fw, H / self.fh
        self.g = [cv2.resize(cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2GRAY), (self.fw, self.fh),
                             interpolation=cv2.INTER_AREA) for x in frames]
        d = np.array([float(cv2.norm(self.g[i + 1], self.g[i], cv2.NORM_L1)) / (self.fw * self.fh)
                      for i in range(self.n - 1)])
        self.diff = d
        moving = d[d > hold_thresh]
        med = float(np.median(moving)) if len(moving) else 1.0
        # global median is fine for a single shot; a local one would follow speed changes better
        self.cut = d > max(cut_min, cut_ratio * med)
        self.keys = [0] + [i + 1 for i in range(self.n - 1) if d[i] > hold_thresh]  # first frame of each drawing
        self.max_span, self.conf_min = max_span, conf_min
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._cache, self._cap = OrderedDict(), cache
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self._gx, self._gy = gx, gy
        lx, ly = np.meshgrid(np.arange(self.fw, dtype=np.float32), np.arange(self.fh, dtype=np.float32))
        self._lx, self._ly = lx, ly
        self._lin = (np.arange(256, dtype=np.float32) / 255.0) ** 2  # ~linear light (gamma 2)
        self._hg = None
        self.stats = {"flow": 0, "nearest": 0, "exact": 0}

    # -- structure ------------------------------------------------------------
    def locate(self, t):
        """-> (a, b, alpha): interpolate frames a->b by alpha; alpha None = exact frame a."""
        t = min(max(t, 0.0), self.n - 1)
        i = int(math.floor(t + 1e-9))
        k = bisect.bisect_right(self.keys, i) - 1
        if k + 1 >= len(self.keys):
            return min(i, self.n - 1), None, None
        nxt = self.keys[k + 1]
        span = min(nxt - self.keys[k], self.max_span)
        start = nxt - span
        if t < start + 1e-6:
            return i, None, None
        return nxt - 1, nxt, (t - start) / span

    def _pair(self, a, b):
        key = (a, b)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        g0, g1 = self.g[a], self.g[b]
        f01 = self.dis.calc(g0, g1, None)
        f10 = self.dis.calc(g1, g0, None)

        def check(fa, fb, ga, gb):
            mx, my = self._lx + fa[..., 0], self._ly + fa[..., 1]
            fb_w = cv2.remap(fb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fbe = np.sqrt(((fa + fb_w) ** 2).sum(-1))  # px at flow res
            gb_w = cv2.remap(gb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ph = cv2.absdiff(ga, gb_w).astype(np.float32)
            occ = np.exp(-(fbe / 1.5) ** 2) * np.exp(-(ph / 24.0) ** 2)  # 1 = trustworthy
            return occ.astype(np.float32), ph

        o0, ph0 = check(f01, f10, g0, g1)
        o1, ph1 = check(f10, f01, g1, g0)
        changed = cv2.absdiff(g0, g1) > 10
        nch = int(changed.sum())
        if self.cut[a:b].any():
            conf = 0.0
        elif nch < 0.003 * changed.size:
            conf = 1.0
        else:  # share of changed pixels the flow explains from both sides
            conf = float(((o0 > 0.5) & (o1 > 0.5) & changed).sum()) / nch
        rec = dict(f01=f01, f10=f10, o0=o0, o1=o1, conf=conf)
        self._cache[key] = rec
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return rec

    # -- sampling ---------------------------------------------------------------
    def sample(self, t, interp="flow"):
        a, b, al = self.locate(t)
        if b is None:
            self.stats["exact"] += 1
            return np.asarray(self.f[a])
        if interp == "nearest" or al < 0.02 or al > 0.98:
            self.stats["nearest"] += 1
            return np.asarray(self.f[a if al < 0.5 else b])
        if interp == "blend":
            return cv2.addWeighted(np.asarray(self.f[a]), 1 - al, np.asarray(self.f[b]), al, 0)
        rec = self._pair(a, b)
        if rec["conf"] < self.conf_min:
            self.stats["nearest"] += 1
            return np.asarray(self.f[a if al < 0.5 else b])
        self.stats["flow"] += 1
        return self._flow_interp(a, b, al, rec)

    def _flow_interp(self, a, b, t, rec):
        f01, f10 = rec["f01"], rec["f10"]
        ft0 = -(1 - t) * t * f01 + t * t * f10
        ft1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
        # occlusion weights at time t, at flow res
        w0 = cv2.remap(rec["o0"], self._lx + ft0[..., 0], self._ly + ft0[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE) * (1 - t)
        w1 = cv2.remap(rec["o1"], self._lx + ft1[..., 0], self._ly + ft1[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE) * t
        # anti-ghost: sharpen toward the more reliable / nearer source
        w0, w1 = w0 * w0 + 1e-4 * (1 - t), w1 * w1 + 1e-4 * t
        s = w0 + w1
        w0 /= s
        W, H = self.W, self.H
        up = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)
        F0, F1 = up(ft0), up(ft1)
        m0x = cv2.scaleAdd(F0[..., 0], self.sx, self._gx)
        m0y = cv2.scaleAdd(F0[..., 1], self.sy, self._gy)
        m1x = cv2.scaleAdd(F1[..., 0], self.sx, self._gx)
        m1y = cv2.scaleAdd(F1[..., 1], self.sy, self._gy)
        i0 = cv2.remap(np.asarray(self.f[a]), m0x, m0y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        i1 = cv2.remap(np.asarray(self.f[b]), m1x, m1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        W0 = up(w0)
        return cv2.blendLinear(i0, i1, W0, 1.0 - W0)

    # -- motion blur --------------------------------------------------------------
    def blur_pair(self, t):
        """Drawing pair whose motion describes source time t (None at the clip end)."""
        a, b, _ = self.locate(t)
        if b is None:  # inside a hold: use the motion toward the next drawing
            k = bisect.bisect_right(self.keys, a)
            if k >= len(self.keys):
                return None
            b = self.keys[k]
            a = b - 1
        return a, b

    def motion_blur(self, img, t, exposure, taps=7, min_px=1.5):
        """Smear img along the local flow by `exposure` source frames (speed x
        shutter). Computed at half res; composited only where motion > min_px."""
        pb = self.blur_pair(t)
        if pb is None:
            return img
        a, b = pb
        rec = self._pair(a, b)
        if rec["conf"] < self.conf_min:
            return img
        span = b - self.keys[bisect.bisect_right(self.keys, a) - 1]  # source frames between the two drawings
        vel = (rec["f01"] - rec["f10"]) * 0.5 / max(span, 1)  # px/source frame at flow res
        L = vel * exposure
        mlen = np.sqrt((L ** 2).sum(-1)) * self.sx  # full-res px
        if float(mlen.max()) < min_px:
            return img
        hw, hh = self.W // 2, self.H // 2
        if self._hg is None:
            self._hg = np.meshgrid(np.arange(hw, dtype=np.float32), np.arange(hh, dtype=np.float32))
        gx, gy = self._hg
        Lh = cv2.resize(L, (hw, hh), interpolation=cv2.INTER_LINEAR) * np.float32(self.sx / 2)
        lin = cv2.LUT(cv2.resize(img, (hw, hh), interpolation=cv2.INTER_AREA), self._lin)
        acc = np.zeros_like(lin)
        for s in np.linspace(-0.5, 0.5, taps, dtype=np.float32):
            acc += cv2.remap(lin, cv2.scaleAdd(Lh[..., 0], float(s), gx), cv2.scaleAdd(Lh[..., 1], float(s), gy),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blur = cv2.convertScaleAbs(cv2.sqrt(acc * (1.0 / taps)), alpha=255.0)
        blur = cv2.resize(blur, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        m = np.clip((mlen - min_px) / (3 * min_px), 0, 1).astype(np.float32)
        m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 1.5), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return cv2.blendLinear(blur, np.asarray(img), m, 1.0 - m)


# ----------------------------------------------------------------------------- clip API

class RampClip:
    """A time-remapped clip: frame(k) for k in [0, n_out).

    frames: decoded source (decode source_frames_needed(n_out, curve) frames, or
    pass fit=True to stretch the curve over whatever you decoded).
    shutter: fraction of the output frame the virtual shutter is open (0.5 = 180deg);
    motion blur kicks in where |speed| > blur_from.
    """

    def __init__(self, frames, n_out, curve, interp="flow", fit=False, start=0.0, shutter=0.75, blur_from=1.3,
                 retimer=None, **kw):
        self.c = make_curve(curve, **kw)
        self.n_out, self.interp = n_out, interp
        self.p, self.v = positions(n_out, self.c, n_src=len(frames) if fit else None, start=start)
        self.rt = retimer or Retimer(frames)
        self.shutter, self.blur_from = shutter, blur_from
        self._ready = False

    def _blurs(self, k):
        return self.shutter > 0 and abs(float(self.v[k])) > self.blur_from and self.interp != "nearest"

    def prefetch(self):
        """Precompute every flow pair this clip touches (the per-shot offline
        step, ~20 ms/pair). Afterwards frame(k) is pure warping."""
        if self._ready:
            return
        need = []
        for k in range(self.n_out):
            t = float(self.p[k])
            a, b, al = self.rt.locate(t)
            if b is not None and self.interp == "flow" and 0.02 <= al <= 0.98:
                need.append((a, b))
            if self._blurs(k):
                pb = self.rt.blur_pair(t)
                if pb:
                    need.append(pb)
        need = list(dict.fromkeys(need))
        self.rt._cap = max(self.rt._cap, len(need) + 4)
        for a, b in need:
            self.rt._pair(a, b)
        self.pairs = len(need)
        self._ready = True

    def frame(self, k):
        self.prefetch()
        k = min(max(int(k), 0), self.n_out - 1)
        t, v = float(self.p[k]), float(self.v[k])
        img = self.rt.sample(t, self.interp)
        if self._blurs(k):
            img = self.rt.motion_blur(img, t, (abs(v) - 1.0) * self.shutter)
        s = self.c.scale((k + 0.5) / self.n_out)
        if s != 1.0:
            H, W = img.shape[:2]
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return img


def remap(frames, n_out, curve, interp="flow", **kw):
    """Time-remap `frames` to n_out frames along `curve` (see RampClip)."""
    rc = RampClip(frames, n_out, curve, interp=interp, **kw)
    return [rc.frame(k) for k in range(n_out)]
