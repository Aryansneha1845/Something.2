"""ANPR: plate localisation (classical CV) + OCR (EasyOCR), off the hot path.

OCR on CPU costs a few hundred milliseconds per crop, which would visibly stall
the video loop, so recognition runs on its own worker thread with a bounded
queue: the analytics loop submits a vehicle crop and picks the answer up later.
Dropping a submission is fine — the same vehicle is re-offered on the next
stride until a plate is confirmed.

Localisation uses blackhat morphology + Sobel gradients rather than a trained
plate detector, so there is no extra model to ship. Results are validated
against the Indian civilian plate grammar, which rejects most OCR noise
outright.
"""
from __future__ import annotations

import queue
import re
import threading

import cv2
import numpy as np

from .detect import Detection

# AA00AA0000 (state-district-series-number) and the newer BH series.
PLATE_RE = re.compile(r"^(?:[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$")
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# OCR confuses these constantly; fix by position, not blindly.
LETTER_FIX = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}
DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
             "B": "8", "S": "5", "Z": "2", "G": "6", "A": "4", "T": "7"}


def normalise(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def repair(text: str) -> str:
    """Coerce a 9-10 char read into the AA00AA0000 shape before validating."""
    t = normalise(text)
    if len(t) < 8 or len(t) > 11:
        return t
    out = list(t)
    for i in (0, 1):                                  # state code: letters
        out[i] = LETTER_FIX.get(out[i], out[i])
    for i in range(2, min(4, len(out))):              # district: digits
        if out[i].isalpha():
            out[i] = DIGIT_FIX.get(out[i], out[i])
    for i in range(len(out) - 4, len(out)):           # last four: digits
        if i >= 0 and out[i].isalpha():
            out[i] = DIGIT_FIX.get(out[i], out[i])
    return "".join(out)


def plate_candidates(crop: np.ndarray, max_boxes: int = 3) -> list[np.ndarray]:
    """Return likely plate sub-images from a vehicle crop, best first."""
    h, w = crop.shape[:2]
    if h < 40 or w < 60:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect)
    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad = np.absolute(grad)
    lo, hi = float(grad.min()), float(grad.max())
    if hi - lo < 1e-6:
        return []
    grad = ((grad - lo) / (hi - lo) * 255).astype("uint8")
    grad = cv2.GaussianBlur(grad, (5, 5), 0)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect)
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[float, np.ndarray]] = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 10 or cw < 30:
            continue
        ar = cw / float(ch)
        if not 1.8 <= ar <= 7.0:
            continue
        if cw * ch < 0.004 * w * h:
            continue
        # Plates sit low on a vehicle; prefer wide, low, high-contrast regions.
        score = (y / h) * 0.5 + min(cw * ch / (w * h), 0.25) * 2.0
        pad = 4
        sub = crop[max(0, y - pad):min(h, y + ch + pad),
                   max(0, x - pad):min(w, x + cw + pad)]
        if sub.size:
            scored.append((score, sub))
    scored.sort(key=lambda s: -s[0])
    return [s[1] for s in scored[:max_boxes]]


class ANPRWorker:
    """Background plate reader. `submit()` never blocks; `drain()` collects."""

    def __init__(self, max_pending: int = 6) -> None:
        self._in: queue.Queue = queue.Queue(maxsize=max_pending)
        self._out: queue.Queue = queue.Queue()
        self._paddle = None
        self._reader = None
        self._engine_type = "none"
        self._ready = threading.Event()
        self._failed = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="anpr",
                                        daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ state
    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def status(self) -> dict:
        return {"ready": self.ready, "engine": self._engine_type, "error": self._failed,
                "pending": self._in.qsize()}

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------- api
    def submit(self, track_id: int, frame: np.ndarray, det: Detection) -> bool:
        if not self.ready or self._stop.is_set():
            return False
        x1, y1, x2, y2 = det.box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 60:
            return False
        try:
            self._in.put_nowait((track_id, crop.copy()))
            return True
        except queue.Full:
            return False

    def drain(self) -> list[tuple[int, str, float]]:
        out: list[tuple[int, str, float]] = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        # Option A: PaddleOCR if installed
        try:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._engine_type = "paddleocr"
            self._ready.set()
        except Exception:
            self._paddle = None

        # Option B: EasyOCR
        if not self._paddle:
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._engine_type = "easyocr"
                self._ready.set()
            except Exception as exc:
                self._failed = f"{exc.__class__.__name__}: {exc}"
                return

        while not self._stop.is_set():
            try:
                track_id, crop = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                plate, conf = self._read(crop)
            except Exception as exc:            # never kill the worker
                self._failed = f"read: {exc.__class__.__name__}"
                continue
            if plate:
                self._out.put((track_id, plate, conf))

    def _read(self, crop: np.ndarray) -> tuple[str, float]:
        best: tuple[str, float] = ("", 0.0)
        regions = plate_candidates(crop) or [crop]
        for region in regions:
            h, w = region.shape[:2]
            if h < 24:                          # upscale small plates for OCR
                scale = 32.0 / max(1, h)
                region = cv2.resize(region, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) \
                if region.ndim == 3 else region
            gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

            # Route to PaddleOCR if available
            if self._paddle:
                try:
                    res = self._paddle.ocr(region, cls=True)
                    if res and res[0]:
                        for line in res[0]:
                            text, conf = line[1][0], float(line[1][1])
                            cand = repair(text)
                            if len(cand) < 8:
                                continue
                            score = conf * (1.0 if PLATE_RE.match(cand) else 0.45)
                            if score > best[1]:
                                best = (cand, score)
                except Exception:
                    pass
            elif self._reader:
                for _, text, conf in self._reader.readtext(
                        gray, allowlist=ALLOWLIST, detail=1, paragraph=False):
                    cand = repair(text)
                    if len(cand) < 8:
                        continue
                    score = float(conf) * (1.0 if PLATE_RE.match(cand) else 0.45)
                    if score > best[1]:
                        best = (cand, score)
        return best if best[1] >= 0.35 else ("", 0.0)


def is_valid(plate: str) -> bool:
    return bool(PLATE_RE.match(normalise(plate)))
