"""Low-light / low-contrast frame enhancement.

Border cameras spend half their duty cycle in near-darkness, fog or rain, which
is exactly where an off-the-shelf detector falls apart. Two paths are provided:

1. `ClaheGammaEnhancer` (default) — CLAHE on the L channel plus a gamma LUT.
   Pure OpenCV, ~2 ms for 720p on one core, no weights to ship. This is what
   runs unless you supply a model.
2. `ZeroDCEEnhancer` (optional) — the Zero-DCE curve-estimation network. The
   architecture is implemented here; drop the pretrained weights at
   `backend/models/zero_dce.pth` and it is picked up automatically. Without
   that file we fall back to path 1 rather than pretending to run a model.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import MODEL_DIR, settings

ZERO_DCE_WEIGHTS = MODEL_DIR / "zero_dce.pth"


def mean_luma(frame: np.ndarray) -> float:
    """Average perceived brightness, 0..255. Cheap: samples a decimated grid."""
    small = frame[::4, ::4]
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


class ClaheGammaEnhancer:
    name = "clahe+gamma"

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=settings.clahe_clip, tileGridSize=(8, 8))
        g = max(0.1, float(settings.gamma))
        self._lut = np.array(
            [((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)],
            dtype=np.uint8)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return cv2.LUT(out, self._lut)


class ZeroDCEEnhancer:
    """Zero-Reference Deep Curve Estimation (Guo et al., CVPR 2020)."""

    name = "zero-dce"

    def __init__(self, weights: Path = ZERO_DCE_WEIGHTS) -> None:
        import torch
        import torch.nn as nn

        class DCENet(nn.Module):
            def __init__(self, ch: int = 32) -> None:
                super().__init__()
                self.relu = nn.ReLU(inplace=True)
                self.e_conv1 = nn.Conv2d(3, ch, 3, 1, 1)
                self.e_conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
                self.e_conv3 = nn.Conv2d(ch, ch, 3, 1, 1)
                self.e_conv4 = nn.Conv2d(ch, ch, 3, 1, 1)
                self.e_conv5 = nn.Conv2d(ch * 2, ch, 3, 1, 1)
                self.e_conv6 = nn.Conv2d(ch * 2, ch, 3, 1, 1)
                self.e_conv7 = nn.Conv2d(ch * 2, 24, 3, 1, 1)

            def forward(self, x):
                x1 = self.relu(self.e_conv1(x))
                x2 = self.relu(self.e_conv2(x1))
                x3 = self.relu(self.e_conv3(x2))
                x4 = self.relu(self.e_conv4(x3))
                x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
                x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
                r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
                for ri in torch.split(r, 3, dim=1):      # 8 curve iterations
                    x = x + ri * (torch.pow(x, 2) - x)
                return x

        self._torch = torch
        self.net = DCENet().eval()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.net.load_state_dict(state)
        torch.set_num_threads(max(1, (torch.get_num_threads() or 2) - 1))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = frame.shape[:2]
        # Run at <=512 wide; the curve map upsamples cleanly and it keeps the
        # CPU cost bounded.
        scale = min(1.0, 512.0 / max(1, w))
        small = cv2.resize(frame, (int(w * scale) // 4 * 4,
                                   int(h * scale) // 4 * 4)) if scale < 1 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        with torch.no_grad():
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            out = self.net(t).clamp(0, 1)
        arr = (out.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (w, h)) if bgr.shape[:2] != (h, w) else bgr


class Enhancer:
    """Chooses a backend once, then decides per frame whether to apply it."""

    def __init__(self) -> None:
        self.backend: ClaheGammaEnhancer | ZeroDCEEnhancer
        self.note = ""
        if ZERO_DCE_WEIGHTS.exists():
            try:
                self.backend = ZeroDCEEnhancer()
            except Exception as exc:            # corrupt/mismatched checkpoint
                self.backend = ClaheGammaEnhancer()
                self.note = f"zero-dce load failed ({exc.__class__.__name__})"
        else:
            self.backend = ClaheGammaEnhancer()
            self.note = "zero_dce.pth not present"

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, bool, float]:
        """-> (frame_for_inference, is_night, luma). Never mutates the input."""
        luma = mean_luma(frame)
        is_night = luma < settings.night_luma_threshold
        if not settings.enhance_auto or not is_night:
            return frame, is_night, luma
        return self.backend(frame), True, luma

    @property
    def info(self) -> dict:
        return {"backend": self.backend.name, "note": self.note,
                "auto": settings.enhance_auto,
                "threshold": settings.night_luma_threshold}
