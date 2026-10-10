"""dialogue_plan.json + song/edit.wav -> voices/mix.wav

Each line's audio is cut from the htdemucs vocals stem of its source clip (BGM/SFX removed), high-passed,
faded, levelled, and placed so that it stays lip-synced to the shot that shows it:

    edit time of source time s = fr(bt(video.b0)) / FPS + (s - video.src)

(the same frame math as director.py's build_shots/src_time at speed 1.0), unless the line sets "edit_at".

Song ducking: out = song + (gv - 1) * vocals_stem + (ga - 1) * accomp_stem, gv/ga ramp 1 -> duck -> 1.
Where gv = ga = 1 the output IS the original edit.wav sample (bit-exact); inside a duck it equals
gv*vocals + ga*accomp + (demucs residual, about -47 dB), so the edges blend with no seam.

Run:  python voices/mix.py [--plan voices/dialogue_plan.json] [--out voices/mix.wav]
Needs ffmpeg and demucs on PATH (or $FFMPEG / $DEMUCS); paths in the plan are relative to the repo root.
"""
import json, math, os, subprocess, sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly

W = os.environ.get("AMV_WORK") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = f"{W}/voices"
FPS = 24000 / 1001
FF = os.environ.get("FFMPEG", "ffmpeg")
DEMUCS = os.environ.get("DEMUCS", "demucs")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rel(p):
    return p if os.path.isabs(p) else os.path.join(W, p)


PLAN = arg("--plan", f"{V}/dialogue_plan.json")
OUT = arg("--out", f"{V}/mix.wav")
plan = json.load(open(PLAN))
plan["song"], plan["song_stems_dir"] = rel(plan["song"]), rel(plan["song_stems_dir"])
for _st in plan["stems"].values():
    _st["window_wav"], _st["vocals"] = rel(_st["window_wav"]), rel(_st["vocals"])
BEATS = [b["t"] for b in json.load(open(f"{W}/song/song.json"))["beats"]]


def bt(i):
    i0 = int(math.floor(i))
    f = i - i0
    return BEATS[i0] if f < 1e-9 else BEATS[i0] + (BEATS[i0 + 1] - BEATS[i0]) * f


def fr(t):
    return int(round(t * FPS))


def shot_t0(b):
    return 0.0 if b == 0 else fr(bt(b)) / FPS


def ensure_stem(st):
    """htdemucs vocals stem of a source window (extract + separate if missing)."""
    if os.path.exists(st["vocals"]):
        return
    src = st["window_wav"]
    if not os.path.exists(src):
        os.makedirs(os.path.dirname(src), exist_ok=True)
        subprocess.run([FF, "-v", "error", "-y", "-ss", str(st["offset"]), "-t", str(st["dur"]), "-i",
                        f"{W}/footage/{st['clip']}.mp4", "-vn", "-ac", "2", "-ar", "44100", src], check=True)
    subprocess.run([DEMUCS, "-n", "htdemucs", "--two-stems", "vocals", "-o",
                    f"{V}/stems_dlg", src], check=True)


def ensure_song_stems():
    d = plan["song_stems_dir"]
    if not all(os.path.exists(f"{d}/{n}.wav") for n in ("vocals", "drums", "bass", "other")):
        subprocess.run([DEMUCS, "-n", "htdemucs", "-o", f"{V}/stems_song",
                        plan["song"]], check=True)


song, sr = sf.read(plan["song"], always_2d=True)
info = sf.info(plan["song"])
ensure_song_stems()
d = plan["song_stems_dir"]
voc = sf.read(f"{d}/vocals.wav", always_2d=True)[0]
acc = sum(sf.read(f"{d}/{n}.wav", always_2d=True)[0] for n in ("drums", "bass", "other"))
assert voc.shape == song.shape == acc.shape, (voc.shape, song.shape)
N = len(song)

hp = butter(4, plan["dialogue"]["highpass_hz"], "highpass", fs=sr, output="sos")
dlg = np.zeros_like(song)
placed = []
for L in plan["lines"]:
    st = plan["stems"][L["stem"]]
    ensure_stem(st)
    x, ssr = sf.read(st["vocals"], always_2d=True)
    assert ssr == sr
    a0, a1 = L["audio_src_in"], L["audio_src_out"]
    # "align": measured lead of the stem vs a full decode of the clip (i1_0_36.wav came out one AAC frame early)
    o = st["offset"] + st.get("align", 0.0)
    seg = x[int(round((a0 - o) * sr)):int(round((a1 - o) * sr))].copy()
    seg = sosfiltfilt(hp, seg, axis=0)
    seg = np.repeat(seg.mean(1, keepdims=True), 2, axis=1)          # centre the voice
    # level: active-speech RMS (frames within 25 dB of the loudest 20 ms) -> target
    hop = sr // 50
    fr_rms = np.array([np.sqrt(np.mean(seg[i:i + hop] ** 2) + 1e-12) for i in range(0, len(seg) - hop, hop)])
    act = fr_rms[20 * np.log10(fr_rms) > 20 * np.log10(fr_rms.max()) - 25]
    cur = 20 * np.log10(np.sqrt(np.mean(act ** 2)))
    g = 10 ** ((plan["dialogue"]["target_active_rms_db"] + L.get("gain_db", 0.0) - cur) / 20)
    seg *= g
    fi, fo = int(L.get("fade_in", 0.04) * sr), int(L.get("fade_out", 0.05) * sr)
    seg[:fi] *= np.linspace(0, 1, fi)[:, None]
    seg[-fo:] *= np.linspace(1, 0, fo)[:, None]
    t_at = L["edit_at"] if "edit_at" in L else shot_t0(L["video"]["b0"]) + (a0 - L["video"]["src"])
    i0 = int(round(t_at * sr))
    dlg[i0:i0 + len(seg)] += seg
    placed.append(dict(id=L["id"], edit_t0=round(t_at, 3), edit_t1=round(t_at + (a1 - a0), 3),
                       speech_t0=round(t_at + (L["speech"][0] - a0), 3), speech_t1=round(t_at + (L["speech"][1] - a0), 3),
                       gain_db=round(20 * np.log10(g), 1)))

# duck envelope: windows = each line's audio span, merged when the gap is shorter than down+up ramps
dk = plan["duck"]
win = sorted((p["edit_t0"], p["edit_t1"]) for p in placed)
merged = []
for a, b in win:
    if merged and a - merged[-1][1] < dk["merge_gap"]:
        merged[-1][1] = max(merged[-1][1], b)
    else:
        merged.append([a, b])
depth = np.zeros(N)                                                   # 0 = no duck, 1 = full duck
t = np.arange(N) / sr
for a, b in merged:
    a0, b1 = a - dk["lead"], b + dk["tail"]
    down = np.clip((t - (a0 - dk["down"])) / dk["down"], 0, 1)
    up = np.clip(((b1 + dk["up"]) - t) / dk["up"], 0, 1)
    env = np.minimum(down, up)
    env = 0.5 - 0.5 * np.cos(np.pi * env)                              # S-curve ramps
    depth = np.maximum(depth, env)
gv = 10 ** (dk["vocals_db"] * depth / 20)
ga = 10 ** (dk["accomp_db"] * depth / 20)
gv[depth == 0] = 1.0
ga[depth == 0] = 1.0
out = song + (gv - 1)[:, None] * voc + (ga - 1)[:, None] * acc + dlg

# true-peak guard, applied only inside the duck spans (outside them the song is left untouched)
lim = 10 ** (plan["true_peak_ceiling_db"] / 20)
dlg_final = dlg.copy()
for p in placed:                                                     # per line, so one hot line does not pull the others
    a, b = p["edit_t0"], p["edit_t1"]
    i0, i1 = int(a * sr), int(b * sr)
    bed, sp = out[i0:i1] - dlg_final[i0:i1], dlg_final[i0:i1].copy()
    k = 1.0
    while np.abs(resample_poly(bed + k * sp, 4, 1, axis=0)).max() > lim and k > 0.3:
        k *= 0.97                                                     # pull only the dialogue down
    out[i0:i1] = bed + k * sp
    dlg_final[i0:i1] = k * sp
    if k < 1.0:
        print(f"true-peak guard: dialogue x{k:.3f} in {a:.2f}-{b:.2f}")
assert len(out) == N
sf.write(OUT, out, sr, subtype=info.subtype)
sf.write(OUT.replace(".wav", "_bed.wav"), out - dlg_final, sr, subtype="FLOAT")      # ducked song only (for QA)
json.dump(dict(placed=placed, duck_windows=[[round(a, 3), round(b, 3)] for a, b in merged]),
          open(OUT.replace(".wav", "_placement.json"), "w"), indent=1)
print(json.dumps(dict(placed=placed, duck_windows=merged), indent=1))
