"""titles.py -- the type of the UZUMAKI birthday edit.

All public functions take a uint8 BGR (1080, 1920, 3) frame and return a NEW uint8 BGR frame.
Nothing here touches audio. Times are seconds; "k" is a frame index at 30 fps.

  birthday_lower(frame, t, ...)     HAPPY BIRTHDAY, / NARUTO / 10.10 word build over the hero shot
                                    t = seconds since 86.316 (edit f2589)
  story_text(frame, moment, f, ...) kinetic story words ('alone', 'iruka', 'ramen', 'uzumaki') at
                                    ABSOLUTE edit frame f
  _Canvas, _draw_word, _line_layout, _bday_lines: the building blocks director.py composes the
  end card from (director.birthday_card_v2).
"""
from __future__ import annotations

import functools
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import uharfbuzz as hb

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORK, 'lib2'))
sys.path.insert(0, os.path.join(WORK, 'lib'))

W, H, FPS = 1920, 1080, 30
FONT_DIR = os.path.join(WORK, 'fonts')
T1_EMBOLDEN = 0.5     # px of faux-bold stroke at cap 40 (scales with cap) for Marcellus SC


def _bgr(hexs):
    hexs = hexs.lstrip('#')
    r, g, b = int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16)
    return np.array((b, g, r), np.float32)


def _clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out_cubic(u):
    u = _clamp(u)
    return 1 - (1 - u) ** 3


def ease_in_out_sine(u):
    u = _clamp(u)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def ease_in_quad(u):
    u = _clamp(u)
    return u * u


# =============================================================== fonts / shaping
class Face:
    """A font file (+ optional wght) shaped with HarfBuzz (GPOS kerning) and rasterised with Pillow."""

    def __init__(self, file, wght=None):
        self.path = os.path.join(FONT_DIR, file)
        self.wght = wght
        self._hbface = hb.Face(hb.Blob.from_file_path(self.path))
        self.upem = self._hbface.upem
        self.cap_ratio = -self.pil(1000.0).getbbox('H', anchor='ls')[1] / 1000.0

    @functools.lru_cache(maxsize=64)
    def pil(self, size):
        f = ImageFont.truetype(self.path, size)
        if self.wght:
            f.set_variation_by_axes([self.wght])
        return f

    def size_for_cap(self, cap):
        return cap / self.cap_ratio

    @functools.lru_cache(maxsize=512)
    def shape(self, text, size, tracking=0.0):
        """-> ((char, pen_x, advance), ...), total advance.  tracking in em (fraction of size)."""
        font = hb.Font(self._hbface)
        if self.wght:
            font.set_variations({'wght': float(self.wght)})
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf, {'liga': False, 'clig': False, 'kern': True})
        sc = size / self.upem
        x, out = 0.0, []
        for info, p in zip(buf.glyph_infos, buf.glyph_positions):
            adv = p.x_advance * sc
            out.append((text[info.cluster], x + p.x_offset * sc, adv))
            x += adv + tracking * size
        return tuple(out), x - tracking * size


MONT = Face('Montserrat[wght].ttf', 900)          # T2 name
MARC = Face('MarcellusSC-Regular.ttf')             # T1 / T3 words
CINZ = Face('Cinzel[wght].ttf', 900)               # T3 numerals "10.10"


# =============================================================== flat word rendering (T1 / T3)
@functools.lru_cache(maxsize=1024)
def _word_masks(face, text, cap, tracking, accent):
    """Supersampled (2x -> INTER_AREA) masks of one word. -> (main, acc, (ox, oy), advance)
    main/acc: float32 0..1; (ox, oy) = baseline-left inside the mask; acc = chars in `accent`."""
    S = 2
    size = face.size_for_cap(cap)
    glyphs, adv = face.shape(text, size, tracking)
    f = face.pil(size * S)
    asc, desc = f.getmetrics()
    pad = int(math.ceil(size * 0.4)) * S
    w = int(math.ceil(adv * S)) + 2 * pad
    h = asc + desc + 2 * pad
    w += w % 2
    h += h % 2
    main = Image.new('L', (w, h))
    acc = Image.new('L', (w, h))
    dm, da = ImageDraw.Draw(main), ImageDraw.Draw(acc)
    by = pad + asc
    sw = int(round(T1_EMBOLDEN * cap / 40 * S)) if face is MARC else 0   # Marcellus is light; fUx4's serif is medium
    for ch, x, _ in glyphs:
        (da if ch in accent else dm).text((pad + x * S, by), ch, font=f, fill=255, anchor='ls',
                                          stroke_width=sw, stroke_fill=255)
    m = cv2.resize(np.asarray(main, np.float32) / 255.0, (w // S, h // S), interpolation=cv2.INTER_AREA)
    a = cv2.resize(np.asarray(acc, np.float32) / 255.0, (w // S, h // S), interpolation=cv2.INTER_AREA)
    m.setflags(write=False)
    a.setflags(write=False)
    return m, a, (pad / S, by / S), adv


class _Canvas:
    """Premultiplied text accumulation layer (full frame)."""

    def __init__(self):
        self.P = np.zeros((H, W, 3), np.float32)
        self.A = np.zeros((H, W), np.float32)
        self.any = False

    def over(self, mask, x0, y0, color, alpha):
        """Composite a float mask at integer top-left (x0, y0) with straight colour."""
        if alpha <= 0.002:
            return
        mh, mw = mask.shape
        x0, y0 = int(round(x0)), int(round(y0))
        ax0, ay0 = max(0, x0), max(0, y0)
        ax1, ay1 = min(W, x0 + mw), min(H, y0 + mh)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        m = mask[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] * alpha
        inv = 1 - m
        self.P[ay0:ay1, ax0:ax1] = self.P[ay0:ay1, ax0:ax1] * inv[..., None] + m[..., None] * color
        self.A[ay0:ay1, ax0:ax1] = m + self.A[ay0:ay1, ax0:ax1] * inv
        self.any = True

    def over_rgba(self, P, A, x0, y0, alpha=1.0):
        """Composite a premultiplied (P, A) patch whose top-left is (x0, y0)."""
        mh, mw = A.shape
        ax0, ay0 = max(0, x0), max(0, y0)
        ax1, ay1 = min(W, x0 + mw), min(H, y0 + mh)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        p = P[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] * alpha
        a = A[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] * alpha
        inv = 1 - a
        self.P[ay0:ay1, ax0:ax1] = p + self.P[ay0:ay1, ax0:ax1] * inv[..., None]
        self.A[ay0:ay1, ax0:ax1] = a + self.A[ay0:ay1, ax0:ax1] * inv
        self.any = True

    def flatten(self, frame, glow=None, shadow=None, opacity=1.0, backing=None):
        """frame uint8/float -> float32.  glow=(bgr, sigma, strength); shadow=(bgr, dx, dy, sigma, strength);
        backing=max strength of an adaptive soft dark halo that only appears when the plate behind the
        words is bright (keeps off-white serif legible on daylight plates, invisible on night plates)."""
        out = frame.astype(np.float32)
        if not self.any or opacity <= 0.002:
            return out
        A = self.A * opacity
        P = self.P * opacity
        if backing:
            m = self.A > 0.3
            if m.any():
                lum = cv2.cvtColor(_u8(out), cv2.COLOR_BGR2GRAY)
                bg = float(np.percentile(lum[m], 60))
                st = backing * _clamp((bg - 80) / 110)
                if st > 0.01:
                    halo = np.clip(_blur_half(A, 14) * 1.6, 0, 1) * st * 0.55
                    drop = _shift(cv2.GaussianBlur(A, (0, 0), 2.5), 3, 4) * st * 0.9   # fUx4-style tight drop
                    a = np.clip(1 - (1 - halo) * (1 - drop), 0, 0.92)
                    out = out * (1 - a[..., None])
        if shadow is not None:
            col, dx, dy, sig, st = shadow
            a = _blur_half(A, sig)
            a = _shift(a, dx, dy) * st
            out = out * (1 - a[..., None]) + col * a[..., None]
        if glow is not None:
            col, sig, st = glow
            g = _blur_half(A, sig) * st
            add = g[..., None] * col
            out = 255 - (255 - out) * (255 - np.minimum(add, 255)) / 255  # screen
        return P + out * (1 - A[..., None])


def _blur_half(a, sigma):
    """Gaussian blur at half resolution (fast, for glows/shadows)."""
    h, w = a.shape[:2]
    s = cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), max(0.5, sigma / 2))
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def _shift(a, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def _u8(x):
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def _screen_white(img_f, amt):
    return 255 - (255 - img_f) * (1 - amt)


# --------------------------------------------------------------- T1 / T3 style tokens
T1_COL = _bgr('#EDEDE6')
T1_ACC = _bgr('#CDE6A0')
T1_GLOW = (_bgr('#FFFFFF'), 6.0, 0.35)
T1_GLOW_WARM = (_bgr('#FFE7A8'), 6.0, 0.40)
T1_TRACK = 0.16
T1_PUNCT = '.,!'
T3_COL = _bgr('#FFF4DC')
T3_RED = _bgr('#E0332E')
T3_GLOW = (_bgr('#FFC860'), 10.0, 0.45)


def _line_layout(words, face, cap, tracking, x, baseline, align='center'):
    """-> list of (word, x_left, baseline) and total line width (word spaces = shaped space + tracking)."""
    size = face.size_for_cap(cap)
    space = face.shape(' ', size, 0.0)[1] + tracking * size * 2
    widths = [face.shape(w, size, tracking)[1] for w in words]
    total = sum(widths) + space * (len(words) - 1)
    if align == 'center':
        xl = x - total / 2
    elif align == 'right':
        xl = x - total
    else:
        xl = x
    out = []
    for w, wd in zip(words, widths):
        out.append((w, xl, baseline))
        xl += wd + space
    return out, total


def _draw_word(cv, face, text, cap, tracking, x_left, baseline, col, acc_col, accent, alpha):
    m, a, (ox, oy), _ = _word_masks(face, text, cap, tracking, accent)
    x0, y0 = x_left - ox, baseline - oy
    cv.over(m, x0, y0, col, alpha)
    if accent:
        cv.over(a, x0, y0, acc_col, alpha)


def _fade_in(k, k0, n=3):
    """word fade 0->1 over n frames; first visible frame is k0 (alpha 1/n)."""
    if k < k0:
        return 0.0
    return _clamp((k - k0 + 1) / n)


def _fade_out(k, k0, k1):
    """1 until k0, linear to 0 at k1."""
    if k <= k0:
        return 1.0
    if k >= k1:
        return 0.0
    return 1 - (k - k0) / (k1 - k0)


# =============================================================== T1 story moments (bible 2, 5, 7)
def _story_words(moment, pos=None):
    """-> list of dicts: text, face, cap, x, base, k0 (edit frame), group, accent colour."""
    out = []
    if moment == 'alone':                           # moment 1, f99 .. dissolve f192-198
        cap = 40
        base = 0.82 * H + cap / 2
        lay, _ = _line_layout("THE PAIN OF BEING ALONE..".split(), MARC, cap, T1_TRACK, W / 2, base)
        for i, (w, x, b) in enumerate(lay):
            out.append(dict(text=w, cap=cap, x=x, base=b, k0=99 + 3 * i, g=0))
        groups = {0: (192, 198)}
    elif moment == 'iruka':                         # moment 2, f296, ghost lines 1-2 at f346, clear f395
        x0, top = pos or (1010, 300)
        caps = (44, 84, 44, 84)
        lines = ("HE'S NOT THE", "NINE-TAILS..", "HE'S UZUMAKI NARUTO", "OF THE LEAF!")
        trk = (T1_TRACK, T1_TRACK, T1_TRACK + 0.25, T1_TRACK)
        gaps = (0, 30, 40, 30)
        widest = max(_line_layout(t.split(), MARC, c, r, 0, 0, align='left')[1] for t, c, r in zip(lines, caps, trk))
        x0 = min(x0, W - 70 - widest)               # keep the whole block inside the frame
        y = top
        kk = {0: 296, 1: 346}                       # words every 3 frames within each pair
        for li, (txt, cap, tr, gp) in enumerate(zip(lines, caps, trk, gaps)):
            y += gp + cap
            lay, _ = _line_layout(txt.split(), MARC, cap, tr, x0, y, align='left')
            grp = 0 if li < 2 else 1
            for w, x, b in lay:
                out.append(dict(text=w, cap=cap, x=x, base=b, k0=kk[grp], g=grp, tr=tr))
                kk[grp] += 3
        groups = {0: ('ghost', 346), 1: (389, 395)}
    elif moment == 'ramen':                         # moment 3, f395 / line 2 on the 14.805 bar f444, clear f494
        cap = 40
        lh = 74
        c = 0.82 * H
        b1 = c - lh / 2 + cap / 2
        b2 = c + lh / 2 + cap / 2
        lay1, _ = _line_layout("ONE DAY I'M GONNA BE HOKAGE..".split(), MARC, cap, T1_TRACK, W / 2, b1)
        lay2, _ = _line_layout("BELIEVE IT!".split(), MARC, cap, T1_TRACK, W / 2, b2)
        for i, (w, x, b) in enumerate(lay1):
            out.append(dict(text=w, cap=cap, x=x, base=b, k0=395 + 3 * i, g=0))
        for i, (w, x, b) in enumerate(lay2):
            out.append(dict(text=w, cap=cap, x=x, base=b, k0=444 + 3 * i, g=0))
        groups = {0: (488, 494)}
    elif moment == 'uzumaki':                       # moment 4, f549 / f555, rides the bloom out
        cap = 46
        base = 0.84 * H + cap / 2
        lay, _ = _line_layout("UZUMAKI NARUTO".split(), MARC, cap, T1_TRACK, W / 2, base)
        for (w, x, b), k0 in zip(lay, (549, 555)):
            out.append(dict(text=w, cap=cap, x=x, base=b, k0=k0, g=0))
        groups = {0: (578, 588)}
    else:
        raise ValueError(moment)
    return out, groups


STORY_SPAN = {'alone': (99, 198), 'iruka': (296, 395), 'ramen': (395, 494), 'uzumaki': (549, 588)}


def story_text(frame, moment, f, pos=None, backing=0.8):
    """T1 kinetic words for moments 1-4 at absolute edit frame f (float ok).
    pos: (x_left, top) of the Iruka block (place it in the negative space beside Iruka)."""
    words, groups = _story_words(moment, pos)
    cv = _Canvas()
    for wd in words:
        a = _fade_in(f, wd['k0'])
        g = groups[wd['g']]
        if g[0] == 'ghost':                         # 1 -> 0.25 over 4 frames, -> 0 by 8
            d = f - g[1]
            if d >= 0:
                a *= (1 - 0.75 * d / 4) if d < 4 else max(0.0, 0.25 * (1 - (d - 4) / 4))
        else:
            a *= _fade_out(f, g[0], g[1])
        if a <= 0:
            continue
        _draw_word(cv, MARC, wd['text'], wd['cap'], wd.get('tr', T1_TRACK), wd['x'], wd['base'],
                   T1_COL, T1_ACC, T1_PUNCT, a)
    glow = T1_GLOW_WARM if moment == 'uzumaki' else T1_GLOW
    return _u8(cv.flatten(frame, glow=glow, backing=backing))


# =============================================================== birthday (bible 9)
BDAY0_FRAME = 2589            # 86.316
CARD0_FRAME = 2663            # 88.782 final hit
FADE0, FADE1 = 89.082, 89.932
END_FRAME = 2698


def _bday_lines(cx, top_base, card=False):
    """-> list of word dicts for HAPPY BIRTHDAY, / NARUTO / 10.10 laid out centred on cx."""
    out = []
    b1 = top_base
    lay1, _ = _line_layout(['HAPPY', 'BIRTHDAY,'], MARC, 52, T1_TRACK, cx, b1)
    b2 = b1 + 30 + 96
    word2 = 'NARUTO!' if card else 'NARUTO'
    lay2, _ = _line_layout([word2], MARC, 96, 0.10, cx, b2)
    b3 = b2 + 34 + 40
    lay3, _ = _line_layout(['10.10'], CINZ, 40, 0.6, cx, b3)
    for (w, x, b), k0 in zip(lay1, (0, 3)):
        out.append(dict(text=w, face=MARC, cap=52, tr=T1_TRACK, x=x, base=b, k0=k0))
    out.append(dict(text=lay2[0][0], face=MARC, cap=96, tr=0.10, x=lay2[0][1], base=b2, k0=25))
    out.append(dict(text='10.10', face=CINZ, cap=40, tr=0.6, x=lay3[0][1], base=b3, k0=50))
    return out


def birthday_lower(frame, t, cx=0.5, top_base=760, backing=0.8):
    """Moment 5a over the Hokage hero shot.  t = seconds since 86.316 (k=0 at edit f2589).
    HAPPY (k0) BIRTHDAY, (k3)  NARUTO lands k25 = 87.138 (1-frame exposure pop)  10.10 lands k50 = 87.960."""
    k = t * FPS
    ki = int(round(k))
    img = frame.astype(np.float32)
    pop = (ki == 25)
    if pop:                                                     # 1-frame exposure pop of the plate itself
        img = 255 - (255 - img) * (255 - img * 0.40) / 255
    cv = _Canvas()
    for wd in _bday_lines(cx * W, top_base):
        a = _fade_in(k, wd['k0'])
        if a <= 0:
            continue
        _draw_word(cv, wd['face'], wd['text'], wd['cap'], wd['tr'], wd['x'], wd['base'],
                   T3_COL, T3_RED, '!', a)
    gc, gs, gst = T3_GLOW
    return _u8(cv.flatten(img, glow=(gc, gs, gst * (1.8 if pop else 1.0)), backing=backing))
