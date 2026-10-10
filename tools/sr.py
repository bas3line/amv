"""Real-ESRGAN realesr-animevideov3 (SRVGGNetCompact, x4) on Apple MPS, for the soft SD / low-bitrate plates.

Weights: models/realesr-animevideov3.pth (official xinntao/Real-ESRGAN release v0.2.5.0,
sha256 b8a83768...b75d). The network is re-declared here (same layout as basicsr's srvgg_arch).

usage: sr.py test IN.png OUT.png [--43]     (one frame, prints timing)
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "realesr-animevideov3.pth")


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


_NET = {}


def net():
    if "n" not in _NET:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        m = SRVGGNetCompact()
        sd = torch.load(MODEL, map_location="cpu", weights_only=True)
        sd = sd.get("params", sd.get("params_ema", sd))
        m.load_state_dict(sd, strict=True)
        m.eval().to(dev)
        _NET["n"], _NET["dev"] = m, dev
    return _NET["n"], _NET["dev"]


@torch.inference_mode()
def x4(bgr):
    """uint8 BGR (h, w, 3) -> uint8 BGR (4h, 4w, 3)."""
    m, dev = net()
    t = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(dev).permute(2, 0, 1)[None].float() / 255.0
    y = m(t).clamp_(0, 1)
    y = (y[0].permute(1, 2, 0) * 255.0 + 0.5).byte().cpu().numpy()
    return np.ascontiguousarray(y[..., ::-1])


def sr43(frame1080):
    """4:3 pillarboxed 1920x1080 frame -> the 4:3 picture SR'd and resampled to 1920x1440."""
    c = frame1080[:, 240:1680]
    lo = cv2.resize(c, (720, 540), interpolation=cv2.INTER_AREA)
    return cv2.resize(x4(lo), (1920, 1440), interpolation=cv2.INTER_AREA)


def sr169(frame1080):
    """Soft 16:9 1920x1080 frame -> SR at 960x540 input, back to 1920x1080."""
    lo = cv2.resize(frame1080, (960, 540), interpolation=cv2.INTER_AREA)
    return cv2.resize(x4(lo), (1920, 1080), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        img = cv2.imread(sys.argv[2])
        f = sr43 if "--43" in sys.argv else sr169
        f(img)  # warm-up
        t0 = time.time()
        out = f(img)
        print(f"{time.time() - t0:.3f}s", out.shape)
        cv2.imwrite(sys.argv[3], out)
