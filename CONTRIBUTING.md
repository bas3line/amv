# Contributing

Thanks for wanting to add to amv. New effects are the best contribution, and fixes, new edits and docs are welcome too.

## Setup

You need Python 3.12 and ffmpeg.

```sh
git clone https://github.com/bas3line/amv
cd amv
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python tests/test_smoke.py       # every effect on a test card, no footage needed
```

## How it fits together

- `director.py`: the Naruto edit. The `S(...)` lines are the shot list; `frame(j)` builds output frame `j`.
- `fx/`: the effects. Most contributions go here.
- `lib/`, `lib2/`: the shared shader, motion and type helpers the effects build on.
- `voices/`: the dialogue plan and the mix that ducks the song under it.
- `song/song.json`: the beat map. `footage/*_shots.json`: the clip catalogues.
- `tools/`: Real-ESRGAN plates (`sr.py`) and the flash counter (`flash.py`).

## Write an effect

An effect is a plain function in `fx/`:

- It takes a uint8 BGR 1920x1080 frame, plus a frame index or a progress value, and returns a new frame of the same shape.
- It is deterministic: any randomness comes from a seed argument.
- It never shows an edge. If it moves or rotates the picture, overscale until the frame is covered, rather than filling with black or a mirror.
- It keeps held footage sharp. Heavy blur, smears and colour hits belong on 1-3 transition or impact frames.

Add it to `EFFECTS` in `tests/test_smoke.py`, and check it on real frames with `python director.py --stills ...` before you open a pull request.

## Checks

```sh
python -m compileall -q director.py fx lib lib2 voices tools tests
python tests/test_smoke.py
```

CI runs both on every pull request. If your change touches a render, also run `python tools/flash.py out/naruto_bday_edit.mp4` and keep it at 3 flashes a second or fewer.

## Pull requests

Branch from `main`, keep a pull request to one change, and fill in the template: what changed and how you tested it, with real output. Never commit footage, music or renders. `.gitignore` already keeps them out.
