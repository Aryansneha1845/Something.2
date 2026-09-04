"""Face detection (YuNet) and optional 1:N matching against an enrolled gallery.

Scope decisions, deliberately:

* Detection always runs inside the *person* boxes the tracker already produced,
  upscaled — BOP cameras see faces at 30-60 px, and whole-frame detection at a
  CPU-affordable input size misses them.
* Recognition (SFace embeddings) only ever compares against faces an operator
  explicitly enrolled, and is disabled until `face_watchlist_enabled` is set.
  There is no open-set identification of unknown people: an unmatched face stays
  an anonymous "face detected" count.
* `blur()` supports the opposite policy — pixelate every face in the streamed
  and stored imagery — for shared or recorded views.

Both models are ~230 KB / ~37 MB ONNX files in `backend/models/`, run through
OpenCV's DNN module. No extra runtime dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import DATA_DIR, MODEL_DIR, settings

YUNET = MODEL_DIR / "face_detection_yunet.onnx"
SFACE = MODEL_DIR / "face_recognition_sface.onnx"
GALLERY_DIR = DATA_DIR / "faces"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

COSINE_MATCH = 0.363          # SFace's published same-identity threshold


@dataclass
class Face:
    box: tuple[int, int, int, int]        # absolute frame coords
    score: float
    landmarks: np.ndarray | None = None    # (5, 2), absolute
    label: str = ""
    match_score: float = 0.0
    raw: np.ndarray | None = field(default=None, repr=False)  # YuNet 15-vector

    def as_dict(self) -> dict:
        d = {"box": list(self.box), "score": round(self.score, 3)}
        if self.label:
            d |= {"label": self.label, "match": round(self.match_score, 3)}
        return d


class FaceEngine:
    def __init__(self) -> None:
        self.detector = None
        self.recognizer = None
        self.insightface = None
        self.engine_type = "yunet_sface"
        self.note = ""

        # Option A: InsightFace (Buffalo/ArcFace) if installed
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(160, 160))
            self.insightface = app
            self.engine_type = "insightface"
            self.note = "InsightFace Buffalo_S active"
        except Exception:
            self.insightface = None

        # Option B: High-performance CPU native OpenCV YuNet + SFace
        if not self.insightface:
            if YUNET.exists():
                try:
                    self.detector = cv2.FaceDetectorYN.create(
                        str(YUNET), "", (160, 160), 0.6, 0.3, 200)
                except Exception as exc:
                    self.note = f"yunet: {exc.__class__.__name__}"
            else:
                self.note = "face_detection_yunet.onnx missing"
            if SFACE.exists():
                try:
                    self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")
                except Exception as exc:
                    self.note += f" sface: {exc.__class__.__name__}"
            self.engine_type = "yunet_sface"

        self.gallery: dict[str, np.ndarray] = {}
        self.reload_gallery()

    @property
    def available(self) -> bool:
        return self.insightface is not None or self.detector is not None

    @property
    def info(self) -> dict:
        return {"detector": "insightface" if self.insightface else ("yunet" if self.detector else None),
                "recognizer": "insightface" if self.insightface else ("sface" if self.recognizer else None),
                "engine": self.engine_type,
                "watchlist_enabled": settings.face_watchlist_enabled,
                "enrolled": sorted(self.gallery), "note": self.note.strip()}

    # ------------------------------------------------------------- detection
    def detect_in(self, frame: np.ndarray,
                  person_boxes: list[tuple[int, int, int, int]]) -> list[Face]:
        if self.detector is None or not person_boxes:
            return []
        H, W = frame.shape[:2]
        found: list[Face] = []
        for (x1, y1, x2, y2) in person_boxes:
            # head region: upper 45% of the person box, padded
            bh = y2 - y1
            hy2 = min(H, y1 + int(bh * 0.45) + 8)
            hx1, hx2 = max(0, x1 - 8), min(W, x2 + 8)
            crop = frame[max(0, y1 - 8):hy2, hx1:hx2]
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                continue
            scale = max(1.0, 160.0 / max(1, min(crop.shape[:2])))
            scale = min(scale, 4.0)
            work = cv2.resize(crop, (int(crop.shape[1] * scale),
                                     int(crop.shape[0] * scale))) \
                if scale > 1.01 else crop
            try:
                self.detector.setInputSize((work.shape[1], work.shape[0]))
                _, dets = self.detector.detect(work)
            except cv2.error:
                continue
            if dets is None:
                continue
            for d in dets:
                fx, fy, fw, fh = (d[:4] / scale)
                ax1 = int(hx1 + fx); ay1 = int(max(0, y1 - 8) + fy)
                lm = (d[4:14].reshape(5, 2) / scale)
                lm[:, 0] += hx1; lm[:, 1] += max(0, y1 - 8)
                found.append(Face(
                    box=(ax1, ay1, int(ax1 + fw), int(ay1 + fh)),
                    score=float(d[14]), landmarks=lm, raw=d.copy()))
        return found

    # ----------------------------------------------------------- recognition
    # ArcFace/SFace canonical 5-point template for a 112x112 aligned face.
    _TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                          [56.0252, 71.7366], [41.5493, 92.3655],
                          [70.7299, 92.2041]], dtype=np.float32)

    @classmethod
    def align(cls, frame: np.ndarray, face: Face) -> np.ndarray | None:
        """Similarity-warp a face to the canonical 112x112 crop."""
        if face.landmarks is None or len(face.landmarks) < 5:
            return None
        src = np.asarray(face.landmarks[:5], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, cls._TEMPLATE, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray | None:
        if self.recognizer is None:
            return None
        aligned = self.align(frame, face)
        if aligned is None:
            return None
        try:
            return np.asarray(self.recognizer.feature(aligned)).ravel()
        except cv2.error:
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def match(self, emb: np.ndarray) -> tuple[str, float]:
        """1:N against the enrolled gallery only. ('', 0.0) when no match."""
        if not settings.face_watchlist_enabled or not self.gallery:
            return "", 0.0
        best, best_s = "", 0.0
        for label, ref in self.gallery.items():
            s = self._cosine(emb, ref)
            if s > best_s:
                best, best_s = label, s
        return (best, best_s) if best_s >= COSINE_MATCH else ("", best_s)

    # --------------------------------------------------------------- gallery
    def reload_gallery(self) -> None:
        self.gallery = {}
        for f in GALLERY_DIR.glob("*.npy"):
            try:
                self.gallery[f.stem] = np.load(f)
            except (ValueError, OSError):
                continue

    def enroll(self, label: str, image: np.ndarray) -> dict:
        """Enrol one reference face. Operator-initiated only."""
        if self.recognizer is None:
            return {"ok": False, "error": "SFace model not available"}
        faces = self.detect_in(image, [(0, 0, image.shape[1], image.shape[0])])
        if not faces:
            return {"ok": False, "error": "no face found in reference image"}
        face = max(faces, key=lambda f: f.score)
        emb = self.embed(image, face)
        if emb is None:
            return {"ok": False, "error": "could not compute embedding"}
        safe = "".join(c for c in label if c.isalnum() or c in "-_ ").strip()
        if not safe:
            return {"ok": False, "error": "invalid label"}
        np.save(GALLERY_DIR / f"{safe}.npy", emb)
        self.reload_gallery()
        return {"ok": True, "label": safe, "detector_score": round(face.score, 3)}

    def unenroll(self, label: str) -> bool:
        path = GALLERY_DIR / f"{label}.npy"
        if path.exists():
            path.unlink()
            self.reload_gallery()
            return True
        return False

    # --------------------------------------------------------------- privacy
    @staticmethod
    def blur(frame: np.ndarray, faces: list[Face], strength: int = 12) -> None:
        """Pixelate faces in place. Applied to both the live stream and any
        snapshot written to disk when the privacy toggle is on."""
        h, w = frame.shape[:2]
        for f in faces:
            x1, y1 = max(0, f.box[0]), max(0, f.box[1])
            x2, y2 = min(w, f.box[2]), min(h, f.box[3])
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            roi = frame[y1:y2, x1:x2]
            small = cv2.resize(roi, (max(1, (x2 - x1) // strength),
                                     max(1, (y2 - y1) // strength)),
                               interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(
                small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
