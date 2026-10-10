"""Smoke tests that need no footage or song: every effect runs on a synthetic 1080p frame, returns a
1920x1080 BGR uint8 frame, and is deterministic. Run with `python tests/test_smoke.py` or pytest."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("fx", "lib", "lib2"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import credits as CR  # noqa: E402
import fxpack as fx  # noqa: E402
import refpack_A as RA  # noqa: E402
import refpack_B as RB  # noqa: E402
import v2fix as V2  # noqa: E402

H, W = 1080, 1920


def frame(seed=0):
    """A deterministic test card: colour gradients, rings and hard edges, so blurs and shakes have detail."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.default_rng(seed)
    img = np.stack([x / W * 255, y / H * 255, (np.sin(x / 40) * np.cos(y / 40) * 0.5 + 0.5) * 255], -1)
    for _ in range(12):
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(30, 220)
        d = np.hypot(x - cx, y - cy)
        img[(d > r) & (d < r + 12)] = rng.uniform(0, 255, 3)
    return np.ascontiguousarray(img.astype(np.uint8))


A, B = frame(0), frame(1)
EFFECTS = {
    "fx.exposure_pop": lambda: fx.exposure_pop(A, target=190.0),
    "fx.white_flash": lambda: fx.white_flash(A),
    "fx.radial_blur": lambda: fx.radial_blur(A, 24),
    "fx.chroma": lambda: fx.chroma(A, px=8.0),
    "fx.slit_open": lambda: fx.slit_open(A, 5),
    "RA.smear_whip": lambda: RA.smear_whip(A, 0),
    "RA.zoom_entry": lambda: RA.zoom_entry(A, 0),
    "RA.zoom_through": lambda: RA.zoom_through(A, 1),
    "RA.impact_shake": lambda: RA.impact_shake(A, 2, seed=3),
    "RA.tilted_hold": lambda: RA.tilted_hold(A, 0.5, shot_idx=2),
    "RA.soft_glow": lambda: RA.soft_glow(A),
    "RA.duotone": lambda: RA.duotone(A, "gold"),
    "RA.light_sweep": lambda: RA.light_sweep(A, 0.5),
    "RA.halftone": lambda: RA.halftone(A),
    "RA.dark_ink": lambda: RA.dark_ink(A),
    "RA.subtitle": lambda: RA.subtitle(A, ["HE'S NARUTO UZUMAKI."], 12),
    "RA.drop_frame": lambda: RA.drop_frame(A, 0),
    "RB.neg_glow": lambda: RB.neg_glow(A),
    "RB.ink": lambda: RB.ink(A, seed=2),
    "RB.impact_cut": lambda: RB.impact_cut(A, B, 2, seed=4),
    "RB.mono_hit": lambda: RB.mono_hit(A, 1),
    "RB.beat_smear": lambda: RB.beat_smear(A, 1, seed=5),
    "RB.duotone_hit": lambda: RB.duotone_hit(A, 1),
    "RB.white_whip_in": lambda: RB.white_whip_in(A, 1),
    "RB.glow_hold": lambda: RB.glow_hold(A),
    "RB.focus_rack_in": lambda: RB.focus_rack_in(A, 1),
    "RB.card_swing": lambda: RB.card_swing(A, 4, 20, seed=6),
    "RB.dip_black": lambda: RB.dip_black(A, 1),
    "RB.dutch_drift": lambda: RB.dutch_drift(A, 0.5),
    "V2.ink_dim": lambda: V2.ink_dim(A, seed=1),
    "V2.rgb_hit": lambda: V2.rgb_hit(A, 0),
}


def test_effects_shape_and_determinism():
    for name, f in EFFECTS.items():
        a, b = f(), f()
        assert a.shape == (H, W, 3) and a.dtype == np.uint8, (name, a.shape, a.dtype)
        assert np.array_equal(a, b), f"{name} is not deterministic"


def test_effects_change_the_frame():
    unchanged = [n for n, f in EFFECTS.items() if np.array_equal(f(), A)]
    assert not unchanged, f"no-op effects: {unchanged}"


def test_credits_row():
    rgba = CR.credits_rgba(2.0)
    assert rgba.shape == (H, W, 4) and rgba.dtype == np.uint8
    alpha = rgba[..., 3]
    assert alpha[1002:].max() == 255, "the credit row should be fully opaque in the bottom bar"
    assert alpha[:1000].max() == 0, "the credit row should stay inside the bottom bar"


if __name__ == "__main__":
    for t in (test_effects_shape_and_determinism, test_effects_change_the_frame, test_credits_row):
        t()
        print("ok", t.__name__)
    print(f"{len(EFFECTS)} effects ok")
