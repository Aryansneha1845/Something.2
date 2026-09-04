"""Natural language query parser for surveillance event search.

Translates human queries like 'high threat crawling near fence at night' into
parameter-bound SQLite queries executed by db.query_events().
"""
from __future__ import annotations

import re
import time
from typing import Any

from . import db

ACTION_KEYWORDS = {
    "crawl": "crawl",
    "crawling": "crawl",
    "climb": "climb",
    "climbing": "climb",
    "crouch": "crouch",
    "crouching": "crouch",
    "loiter": "loiter",
    "loitering": "loiter",
    "fall": "fallen",
    "fallen": "fallen",
    "run": "run",
    "running": "run",
    "walk": "walk",
    "walking": "walk",
}

KIND_KEYWORDS = {
    "intrusion": "intrusion",
    "intrusions": "intrusion",
    "breach": "intrusion",
    "tripwire": "tripwire",
    "wire": "tripwire",
    "loiter": "loiter",
    "loitering": "loiter",
    "watchlist": "watchlist",
    "wanted": "watchlist",
    "suspect": "watchlist",
    "night motion": "night_motion",
    "activity": "activity",
}

CLASS_KEYWORDS = {
    "person": "person",
    "people": "person",
    "man": "person",
    "woman": "person",
    "individual": "person",
    "pedestrian": "person",
    "intruder": "person",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "trucks": "truck",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bus": "bus",
    "bag": "backpack",
    "backpack": "backpack",
}


def parse_query(q: str) -> dict[str, Any]:
    """Parse freeform text query into structured filters for db.query_events()."""
    cleaned = q.strip().lower()
    filters: dict[str, Any] = {
        "raw_query": q,
        "actions": set(),
        "kinds": set(),
        "object_class": None,
        "min_threat": None,
        "is_night": None,
        "acknowledged": None,
        "camera_id": None,
        "zone": None,
        "text": None,
        "since": None,
    }

    # Threat extraction
    m_score = re.search(r"(?:threat|score)\s*(?:>=|>|:)?\s*(\d+)", cleaned)
    if m_score:
        filters["min_threat"] = int(m_score.group(1))
    elif any(w in cleaned for w in ("critical", "urgent", "danger", "high threat", "severe")):
        filters["min_threat"] = 70
    elif any(w in cleaned for w in ("suspicious", "moderate threat", "warning")):
        filters["min_threat"] = 40

    # Night / lighting
    if any(w in cleaned for w in ("night", "dark", "nighttime", "midnight")):
        filters["is_night"] = True
    elif any(w in cleaned for w in ("day", "daytime", "daylight")):
        filters["is_night"] = False

    # Acknowledgment status
    if any(w in cleaned for w in ("unack", "unacknowledged", "open", "pending", "active alert")):
        filters["acknowledged"] = False
    elif any(w in cleaned for w in ("acknowledged", "resolved", "closed", "cleared")):
        filters["acknowledged"] = True

    # Camera ID
    m_cam = re.search(r"\b(?:cam|camera)[-_ ]?([a-z0-9]+)\b", cleaned)
    if m_cam:
        val = m_cam.group(1)
        filters["camera_id"] = f"CAM-{int(val):02d}" if val.isdigit() else f"CAM-{val.upper()}"

    # Time window extraction
    now = time.time()
    if "last hour" in cleaned or "past hour" in cleaned:
        filters["since"] = now - 3600
    elif "today" in cleaned or "past 24h" in cleaned or "last 24 hours" in cleaned:
        filters["since"] = now - 86400
    elif "past week" in cleaned or "last 7 days" in cleaned:
        filters["since"] = now - 7 * 86400

    # Zones
    for z in ("fence", "gate", "perimeter", "checkpoint", "border", "restricted"):
        if z in cleaned:
            filters["zone"] = z
            break

    # Actions, Kinds, Classes
    words = re.findall(r"\b[a-z0-9_-]+\b", cleaned)
    remaining_words = []
    for w in words:
        consumed = False
        if w in ACTION_KEYWORDS:
            filters["actions"].add(ACTION_KEYWORDS[w])
            consumed = True
        if w in KIND_KEYWORDS:
            filters["kinds"].add(KIND_KEYWORDS[w])
            consumed = True
        if w in CLASS_KEYWORDS:
            filters["object_class"] = CLASS_KEYWORDS[w]
            consumed = True
        if not consumed and w not in ("in", "at", "the", "a", "an", "on", "near", "by", "with", "show", "me", "all", "find"):
            remaining_words.append(w)

    if remaining_words:
        filters["text"] = " ".join(remaining_words)

    # Convert sets to lists
    filters["actions"] = list(filters["actions"]) or None
    filters["kinds"] = list(filters["kinds"]) or None

    return filters


def search_events(q: str, limit: int = 50) -> list[dict[str, Any]]:
    """Execute natural language search and return scored matching events."""
    parsed = parse_query(q)
    results = db.query_events(
        camera_id=parsed["camera_id"],
        kinds=parsed["kinds"],
        actions=parsed["actions"],
        zone=parsed["zone"],
        object_class=parsed["object_class"],
        min_threat=parsed["min_threat"],
        since=parsed["since"],
        is_night=parsed["is_night"],
        text=parsed["text"],
        acknowledged=parsed["acknowledged"],
        limit=limit,
    )

    # Rank results by relevance: exact match in caption + threat score
    q_words = set(re.findall(r"\w+", q.lower()))

    def score_event(ev: dict[str, Any]) -> float:
        s = float(ev.get("threat_score") or 0)
        cap = (ev.get("caption") or "").lower()
        matches = sum(1 for w in q_words if w in cap)
        return s + matches * 25.0

    results.sort(key=score_event, reverse=True)
    return results
