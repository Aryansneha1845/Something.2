"""SQLite persistence for events, ANPR reads, watchlists and the audit trail.

One connection per thread (the analytics workers are threads), WAL mode so
readers in the API never block the writers in the pipeline.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .config import DB_PATH

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    iso           TEXT NOT NULL,
    camera_id     TEXT NOT NULL,
    camera_name   TEXT,
    sector        TEXT,
    kind          TEXT NOT NULL,
    object_class  TEXT,
    track_id      INTEGER,
    action        TEXT,
    zone_id       TEXT,
    zone_name     TEXT,
    threat_score  INTEGER DEFAULT 0,
    threat_level  TEXT,
    caption       TEXT,
    snapshot      TEXT,
    clip          TEXT,
    is_night      INTEGER DEFAULT 0,
    meta          TEXT,
    acknowledged  INTEGER DEFAULT 0,
    ack_by        TEXT,
    ack_ts        REAL
);
CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_cam    ON events(camera_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_kind   ON events(kind, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_threat ON events(threat_score DESC);

CREATE TABLE IF NOT EXISTS plates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      INTEGER,
    ts            REAL NOT NULL,
    camera_id     TEXT,
    plate         TEXT NOT NULL,
    confidence    REAL,
    vehicle_class TEXT,
    snapshot      TEXT,
    watchlist_hit INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_plates_ts    ON plates(ts DESC);
CREATE INDEX IF NOT EXISTS idx_plates_plate ON plates(plate);

CREATE TABLE IF NOT EXISTS watchlist (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    kind     TEXT NOT NULL,          -- 'plate' | 'face'
    value    TEXT NOT NULL,
    label    TEXT,
    note     TEXT,
    added_ts REAL NOT NULL,
    added_by TEXT,
    active   INTEGER DEFAULT 1
);

-- Accountability trail: every operator action and every automated dispatch.
CREATE TABLE IF NOT EXISTS audit (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    iso    TEXT NOT NULL,
    actor  TEXT,
    action TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts DESC);
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=15000")
        _local.conn = c
    return c


def init_db() -> None:
    c = conn()
    c.executescript(SCHEMA)
    c.commit()


def iso_now(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc) \
        .astimezone().isoformat(timespec="seconds")


def insert_event(ev: dict[str, Any]) -> int:
    ev = dict(ev)
    ev.setdefault("ts", time.time())
    ev["iso"] = iso_now(ev["ts"])
    if isinstance(ev.get("meta"), (dict, list)):
        ev["meta"] = json.dumps(ev["meta"], default=str)
    cols = [k for k in ev if k != "id"]
    sql = (f"INSERT INTO events ({','.join(cols)}) "
           f"VALUES ({','.join('?' for _ in cols)})")
    c = conn()
    cur = c.execute(sql, [ev[k] for k in cols])
    c.commit()
    return int(cur.lastrowid or 0)


def audit(action: str, detail: Any = None, actor: str = "system") -> None:
    ts = time.time()
    c = conn()
    c.execute(
        "INSERT INTO audit (ts, iso, actor, action, detail) VALUES (?,?,?,?,?)",
        (ts, iso_now(ts), actor, action,
         json.dumps(detail, default=str) if detail is not None else None),
    )
    c.commit()


def row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    if d.get("meta"):
        try:
            d["meta"] = json.loads(d["meta"])
        except (ValueError, TypeError):
            pass
    return d


# --------------------------------------------------------------------------
# Queries.  Every filter is parameter-bound; no string interpolation of values.
# --------------------------------------------------------------------------
def query_events(
    camera_id: str | None = None,
    kinds: list[str] | None = None,
    actions: list[str] | None = None,
    zone: str | None = None,
    object_class: str | None = None,
    min_threat: int | None = None,
    since: float | None = None,
    until: float | None = None,
    is_night: bool | None = None,
    text: str | None = None,
    acknowledged: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    where, args = ["1=1"], []
    if camera_id:
        where.append("camera_id = ?"); args.append(camera_id)
    if kinds:
        where.append(f"kind IN ({','.join('?' * len(kinds))})"); args += kinds
    if actions:
        where.append(f"action IN ({','.join('?' * len(actions))})"); args += actions
    if zone:
        where.append("(zone_name LIKE ? OR zone_id = ?)")
        args += [f"%{zone}%", zone]
    if object_class:
        where.append("object_class = ?"); args.append(object_class)
    if min_threat is not None:
        where.append("threat_score >= ?"); args.append(min_threat)
    if since is not None:
        where.append("ts >= ?"); args.append(since)
    if until is not None:
        where.append("ts <= ?"); args.append(until)
    if is_night is not None:
        where.append("is_night = ?"); args.append(1 if is_night else 0)
    if acknowledged is not None:
        where.append("acknowledged = ?"); args.append(1 if acknowledged else 0)
    if text:
        where.append("(caption LIKE ? OR action LIKE ? OR object_class LIKE ?)")
        args += [f"%{text}%"] * 3
    sql = (f"SELECT * FROM events WHERE {' AND '.join(where)} "
           f"ORDER BY ts DESC LIMIT ? OFFSET ?")
    args += [max(1, min(limit, 500)), max(0, offset)]
    return [row_to_dict(r) for r in conn().execute(sql, args)]


def get_event(event_id: int) -> dict[str, Any] | None:
    r = conn().execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    return row_to_dict(r) if r else None


def ack_event(event_id: int, actor: str) -> bool:
    c = conn()
    cur = c.execute(
        "UPDATE events SET acknowledged=1, ack_by=?, ack_ts=? WHERE id=?",
        (actor, time.time(), event_id),
    )
    c.commit()
    if cur.rowcount:
        audit("event.acknowledge", {"event_id": event_id}, actor)
    return bool(cur.rowcount)


def insert_plate(rec: dict[str, Any]) -> int:
    rec = dict(rec)
    rec.setdefault("ts", time.time())
    cols = [k for k in rec if k != "id"]
    c = conn()
    cur = c.execute(
        f"INSERT INTO plates ({','.join(cols)}) "
        f"VALUES ({','.join('?' for _ in cols)})",
        [rec[k] for k in cols],
    )
    c.commit()
    return int(cur.lastrowid or 0)


def query_plates(plate: str | None = None, camera_id: str | None = None,
                 since: float | None = None, limit: int = 100) -> list[dict]:
    where, args = ["1=1"], []
    if plate:
        where.append("plate LIKE ?"); args.append(f"%{plate.upper()}%")
    if camera_id:
        where.append("camera_id = ?"); args.append(camera_id)
    if since is not None:
        where.append("ts >= ?"); args.append(since)
    sql = (f"SELECT * FROM plates WHERE {' AND '.join(where)} "
           f"ORDER BY ts DESC LIMIT ?")
    args.append(max(1, min(limit, 500)))
    return [dict(r) for r in conn().execute(sql, args)]


# --------------------------------------------------------------------------
# Watchlist.  Face entries are only ever created by an explicit operator
# enrolment action, and every add/remove is written to the audit trail.
# --------------------------------------------------------------------------
def watchlist_add(kind: str, value: str, label: str = "", note: str = "",
                  actor: str = "operator") -> int:
    c = conn()
    cur = c.execute(
        "INSERT INTO watchlist (kind, value, label, note, added_ts, added_by) "
        "VALUES (?,?,?,?,?,?)",
        (kind, value.strip().upper() if kind == "plate" else value.strip(),
         label, note, time.time(), actor),
    )
    c.commit()
    audit("watchlist.add", {"kind": kind, "label": label}, actor)
    return int(cur.lastrowid or 0)


def watchlist_remove(entry_id: int, actor: str = "operator") -> bool:
    c = conn()
    cur = c.execute("UPDATE watchlist SET active=0 WHERE id=?", (entry_id,))
    c.commit()
    if cur.rowcount:
        audit("watchlist.remove", {"id": entry_id}, actor)
    return bool(cur.rowcount)


def watchlist_all(kind: str | None = None) -> list[dict]:
    sql = "SELECT * FROM watchlist WHERE active=1"
    args: list[Any] = []
    if kind:
        sql += " AND kind=?"; args.append(kind)
    return [dict(r) for r in conn().execute(sql + " ORDER BY added_ts DESC", args)]


def watchlist_values(kind: str) -> set[str]:
    return {r["value"] for r in watchlist_all(kind)}


def query_audit(limit: int = 200) -> list[dict]:
    return [dict(r) for r in conn().execute(
        "SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (min(limit, 1000),))]


def stats(window_seconds: float = 24 * 3600) -> dict[str, Any]:
    since = time.time() - window_seconds
    c = conn()
    total = c.execute("SELECT COUNT(*) n FROM events WHERE ts>=?",
                      (since,)).fetchone()["n"]
    unack = c.execute(
        "SELECT COUNT(*) n FROM events WHERE ts>=? AND acknowledged=0 "
        "AND threat_score>=60", (since,)).fetchone()["n"]
    by_kind = {r["kind"]: r["n"] for r in c.execute(
        "SELECT kind, COUNT(*) n FROM events WHERE ts>=? GROUP BY kind "
        "ORDER BY n DESC", (since,))}
    by_sector = {r["sector"]: r["n"] for r in c.execute(
        "SELECT sector, COUNT(*) n FROM events WHERE ts>=? GROUP BY sector "
        "ORDER BY n DESC", (since,))}
    by_hour = [{"hour": r["h"], "count": r["n"]} for r in c.execute(
        "SELECT strftime('%Y-%m-%dT%H:00', ts, 'unixepoch', 'localtime') h, "
        "COUNT(*) n FROM events WHERE ts>=? GROUP BY h ORDER BY h", (since,))]
    peak = c.execute(
        "SELECT MAX(threat_score) m FROM events WHERE ts>=?",
        (since,)).fetchone()["m"] or 0
    plates = c.execute("SELECT COUNT(*) n FROM plates WHERE ts>=?",
                       (since,)).fetchone()["n"]
    return {
        "window_seconds": window_seconds, "total_events": total,
        "open_high_threat": unack, "peak_threat": peak, "plate_reads": plates,
        "by_kind": by_kind, "by_sector": by_sector, "by_hour": by_hour,
    }


def purge_old(days: int) -> int:
    cutoff = time.time() - days * 86400
    c = conn()
    n = c.execute("DELETE FROM events WHERE ts < ?", (cutoff,)).rowcount
    c.execute("DELETE FROM plates WHERE ts < ?", (cutoff,))
    c.commit()
    if n:
        audit("retention.purge", {"days": days, "rows": n})
    return n
