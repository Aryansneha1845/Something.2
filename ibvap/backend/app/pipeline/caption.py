"""Event narration.

Two tiers, and the UI always says which one produced a given line:

* `caption()` — deterministic template over the structured detection state. No
  model, no latency, no hallucination risk. This is what runs inside the
  analytics loop for every alert.
* `explain_with_llm()` — optional, operator-triggered from the event drawer.
  Sends the *structured event JSON* (never raw imagery) to the Claude API for a
  richer natural-language read. Off unless a key is configured.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime

from ..config import settings
from .actions import describe
from .detect import Detection
from .tracks import TrackState
from .zones import ZoneHit

ARTICLE_FOR = {"person": "a person", "car": "a car", "truck": "a truck",
               "bus": "a bus", "motorcycle": "a motorcycle",
               "bicycle": "a bicycle", "animal": "an animal"}


def _subject(det: Detection) -> str:
    return ARTICLE_FOR.get(det.name, f"a {det.name}")


def _carried(det: Detection) -> str:
    items = [a.split(":", 1)[1] for a in det.attrs if a.startswith("carrying:")]
    if not items:
        return ""
    return f", carrying {' and '.join(sorted(set(items)))}"


def caption(det: Detection, track: TrackState, hits: list[ZoneHit],
            camera_name: str, sector: str, is_night: bool,
            ts: float, enhanced: bool = False) -> str:
    """One sentence, deterministic, safe to log verbatim."""
    subject = _subject(det)
    verb = describe(det.action) if det.group == "person" else "moving"
    where = ""
    if hits:
        h = hits[0]
        if h.kind == "crossing":
            where = f" crossing {h.zone_name}"
            if h.direction:
                where += f" ({h.direction}bound)"
        elif h.kind == "loiter":
            where = f" loitering in {h.zone_name} for {h.dwell:.0f}s"
        else:
            where = f" inside {h.zone_name}"
    plate = f", plate {track.plate}" if track.plate else ""
    when = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    light = "night" if is_night else "daylight"
    if enhanced:
        light += ", low-light enhanced"
    return (f"{subject.capitalize()} {verb}{where}{_carried(det)}{plate} — "
            f"{sector} / {camera_name}, {when} ({light}).")


def scene_summary(dets: list[Detection], zone_counts: dict[str, int]) -> str:
    """Rolling one-liner for the live tile: what is in view right now."""
    if not dets:
        return "Scene clear."
    counts: dict[str, int] = {}
    for d in dets:
        if d.group in ("person", "vehicle", "animal"):
            counts[d.name] = counts.get(d.name, 0) + 1
    if not counts:
        return "Scene clear."
    parts = [f"{n} {name}{'s' if n > 1 else ''}" for name, n in
             sorted(counts.items(), key=lambda kv: -kv[1])]
    text = "In view: " + ", ".join(parts) + "."
    occupied = {z: c for z, c in zone_counts.items() if c}
    if occupied:
        text += " Zone occupancy: " + ", ".join(
            f"{z} ({c})" for z, c in occupied.items()) + "."
    actions = {d.action for d in dets
               if d.group == "person" and d.action not in ("unknown", "n/a")}
    if actions:
        text += " Activity: " + ", ".join(sorted(actions)) + "."
    return text


# --------------------------------------------------------------------------
# Optional LLM tier
# --------------------------------------------------------------------------
LLM_MODEL = os.environ.get("IBVAP_LLM_MODEL", "claude-sonnet-5")
_ENDPOINT = "https://api.anthropic.com/v1/messages"

PROMPT = (
    "You are an analyst assistant for a border-surveillance operator. Given "
    "this structured detection record from an automated video-analytics system, "
    "write 2-3 sentences: what the system observed, which stated factors drove "
    "the threat score, and what the operator should verify. Do not invent "
    "details that are not in the record. Do not speculate about identity, "
    "nationality or intent.\n\nRecord:\n"
)


def llm_available() -> bool:
    return bool(settings.llm_captions and os.environ.get("ANTHROPIC_API_KEY"))


def explain_with_llm(event: dict, timeout: float = 20.0) -> dict:
    """Best-effort. Returns {'ok', 'text'|'error', 'source'}."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not settings.llm_captions:
        return {"ok": False, "error": "LLM captions disabled "
                "(set IBVAP_LLM_CAPTIONS=1)", "source": "none"}
    if not key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY not set",
                "source": "none"}
    safe = {k: v for k, v in event.items()
            if k not in ("snapshot", "clip")}      # imagery stays on-premise
    body = json.dumps({
        "model": LLM_MODEL, "max_tokens": 300,
        "messages": [{"role": "user",
                      "content": PROMPT + json.dumps(safe, indent=2, default=str)}],
    }).encode()
    req = urllib.request.Request(_ENDPOINT, data=body, headers={
        "content-type": "application/json", "x-api-key": key,
        "anthropic-version": "2023-06-01",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in payload.get("content", []))
        return {"ok": True, "text": text.strip(), "source": LLM_MODEL}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}",
                "source": LLM_MODEL}
