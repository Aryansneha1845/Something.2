"""IBVAP — Intelligent Border Video Analytics Platform.

Main FastAPI application: REST APIs, MJPEG streaming, WebSocket alert bus,
evidence serving, and static operator dashboard hosting.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import BASE_DIR, CLIP_DIR, SNAP_DIR, settings
from .manager import manager
from .pipeline.dispatch import bus
from .pipeline.faces import FaceEngine
from .search import parse_query, search_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ibvap.main")

STATIC_DIR = BASE_DIR.parent / "frontend"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SQLite database with WAL mode...")
    db.init_db()
    db.audit("system.startup", {"device": settings.device, "target_fps": settings.target_fps})

    logger.info("Starting IBVAP camera analytics manager...")
    manager.initialize()

    yield

    logger.info("Shutting down analytics workers...")
    manager.shutdown()
    db.audit("system.shutdown")


app = FastAPI(
    title="IBVAP — Border Video Analytics Platform",
    description="C2 Tactical Surveillance & Computer Vision Pipeline API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
app.mount("/snapshots", StaticFiles(directory=str(SNAP_DIR)), name="snapshots")
app.mount("/clips", StaticFiles(directory=str(CLIP_DIR)), name="clips")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def get_index():
    """Serve the single-page Tactical C2 Operator Dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "service": "IBVAP Backend API",
        "status": "operational",
        "docs": "/docs",
        "ui": "Dashboard building in progress",
    })


@app.get("/app.js")
async def get_app_js():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/style.css")
async def get_style_css():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


# --------------------------------------------------------------------------
# System & Health
# --------------------------------------------------------------------------
@app.get("/api/status")
async def get_system_status():
    cams = manager.list_cameras()
    active_workers = sum(1 for c in cams if c.get("status", {}).get("running"))
    total_fps = sum(c.get("status", {}).get("fps", 0) for c in cams)
    total_tracks = sum(c.get("status", {}).get("tracks", 0) for c in cams)
    max_threat = max((c.get("status", {}).get("top_threat", {}).get("score", 0) for c in cams), default=0)

    return {
        "status": "online",
        "device": settings.device,
        "target_fps": settings.target_fps,
        "active_cameras": active_workers,
        "total_cameras": len(cams),
        "aggregate_fps": round(total_fps, 1),
        "active_tracks": total_tracks,
        "max_threat_score": max_threat,
        "subscribers": bus.subscriber_count,
        "night_mode": any(c.get("status", {}).get("is_night") for c in cams),
    }


# --------------------------------------------------------------------------
# Cameras & Live Video Streaming
# --------------------------------------------------------------------------
@app.get("/api/cameras")
async def list_cameras():
    return manager.list_cameras()


@app.post("/api/cameras")
async def add_or_update_camera(request: Request):
    data = await request.json()
    if not data.get("id"):
        raise HTTPException(status_code=400, detail="Missing camera 'id'")
    result = manager.add_or_update_camera(data)
    db.audit("camera.configured", {"id": data.get("id"), "name": data.get("name")})
    return {"ok": True, "camera": result}


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    if not manager.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    db.audit("camera.deleted", {"id": camera_id})
    return {"ok": True, "deleted": camera_id}


@app.post("/api/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    if not manager.start_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"ok": True, "status": "started"}


@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    if not manager.stop_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"ok": True, "status": "stopped"}


@app.get("/api/cameras/{camera_id}/stream")
async def stream_annotated_feed(camera_id: str):
    """MJPEG stream with bounding boxes, pose skeletons, zones, and HUD."""
    cam = manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        manager.stream_mjpeg(camera_id, raw=False, max_fps=12.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras/{camera_id}/raw")
async def stream_raw_feed(camera_id: str):
    """MJPEG stream without analytics overlays."""
    cam = manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        manager.stream_mjpeg(camera_id, raw=True, max_fps=12.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: str, raw: bool = False):
    """Return the newest single JPEG frame for thumbnails or responsive mobile view."""
    worker = manager.get_worker(camera_id)
    if not worker:
        standby = manager.get_standby_jpeg(f"{camera_id}: OFFLINE")
        return Response(content=standby, media_type="image/jpeg")
    jpeg = worker.frame_jpeg(raw=raw)
    if not jpeg:
        standby = manager.get_standby_jpeg(f"{camera_id}: INITIALIZING...")
        return Response(content=standby, media_type="image/jpeg")
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/api/cameras/{camera_id}/ptz")
async def control_ptz(camera_id: str, request: Request):
    """Simulated or ONVIF PTZ slew command."""
    cam = manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    body = await request.json()
    action = body.get("action", "slew")
    pan = float(body.get("pan", 0.0))
    tilt = float(body.get("tilt", 0.0))
    zoom = float(body.get("zoom", 1.0))
    reason = body.get("reason", "manual-operator")

    payload = {
        "camera_id": camera_id,
        "ts": db.iso_now(),
        "action": action,
        "pan": pan,
        "tilt": tilt,
        "zoom": zoom,
        "reason": reason,
        "onvif_soap_payload": (
            f"<soap:Envelope><soap:Body><ContinuousMove>"
            f"<Velocity><PanTilt x='{pan:.2f}' y='{tilt:.2f}'/>"
            f"<Zoom x='{zoom:.2f}'/></Velocity>"
            f"</ContinuousMove></soap:Body></soap:Envelope>"
        ),
    }
    bus.publish({"type": "ptz", **payload})
    db.audit("ptz.command", payload, actor=body.get("actor", "operator"))
    return {"ok": True, "ptz_action": payload}


# --------------------------------------------------------------------------
# Surveillance Events & Incidents
# --------------------------------------------------------------------------
@app.get("/api/events")
async def get_events(
    camera_id: str | None = None,
    kind: str | None = None,
    action: str | None = None,
    zone: str | None = None,
    object_class: str | None = None,
    min_threat: int | None = None,
    since: float | None = None,
    until: float | None = None,
    is_night: bool | None = None,
    text: str | None = None,
    acknowledged: bool | None = None,
    limit: int = 50,
    offset: int = 0,
):
    kinds = [kind] if kind else None
    actions = [action] if action else None
    events = db.query_events(
        camera_id=camera_id,
        kinds=kinds,
        actions=actions,
        zone=zone,
        object_class=object_class,
        min_threat=min_threat,
        since=since,
        until=until,
        is_night=is_night,
        text=text,
        acknowledged=acknowledged,
        limit=limit,
        offset=offset,
    )
    return {"count": len(events), "events": events}


@app.get("/api/events/{event_id}")
async def get_event_detail(event_id: int):
    ev = db.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return ev


@app.post("/api/events/{event_id}/ack")
async def acknowledge_event(event_id: int, request: Request):
    data = await request.json() if request.headers.get("content-type") == "application/json" else {}
    actor = data.get("actor", "duty_officer")
    success = db.ack_event(event_id, actor=actor)
    if not success:
        raise HTTPException(status_code=404, detail="Event not found or already acknowledged")
    bus.publish({"type": "ack", "event_id": event_id, "actor": actor, "ts": db.iso_now()})
    return {"ok": True, "event_id": event_id, "acknowledged_by": actor}


# --------------------------------------------------------------------------
# Natural Language / Semantic Search
# --------------------------------------------------------------------------
@app.get("/api/search")
async def search_incidents(q: str = Query(..., description="Natural language search query"), limit: int = 40):
    filters = parse_query(q)
    results = search_events(q, limit=limit)
    return {
        "query": q,
        "parsed_filters": filters,
        "match_count": len(results),
        "results": results,
    }


# --------------------------------------------------------------------------
# ANPR & License Plates
# --------------------------------------------------------------------------
@app.get("/api/plates")
async def get_plate_reads(
    plate: str | None = None,
    camera_id: str | None = None,
    since: float | None = None,
    limit: int = 50,
):
    reads = db.query_plates(plate=plate, camera_id=camera_id, since=since, limit=limit)
    return {"count": len(reads), "plates": reads}


# --------------------------------------------------------------------------
# Watchlist Management (Faces & Plates)
# --------------------------------------------------------------------------
@app.get("/api/watchlist")
async def list_watchlist(kind: str | None = None):
    return {"entries": db.watchlist_all(kind=kind)}


@app.post("/api/watchlist/enroll")
async def enroll_watchlist(request: Request):
    """Enrol a license plate or face into the surveillance watchlist."""
    data = await request.json()
    kind = data.get("kind")
    value = data.get("value")
    label = data.get("label", "")
    note = data.get("note", "")
    actor = data.get("actor", "operator")

    if not kind or not value:
        raise HTTPException(status_code=400, detail="Missing 'kind' or 'value'")

    if kind == "plate":
        entry_id = db.watchlist_add(kind="plate", value=value.upper(), label=label, note=note, actor=actor)
        return {"ok": True, "id": entry_id, "kind": "plate", "plate": value.upper()}

    elif kind == "face":
        # Check if base64 image was provided for face embedding
        image_b64 = data.get("image_b64")
        if image_b64:
            try:
                raw_bytes = base64.b64decode(image_b64.split(",")[-1])
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    engine = FaceEngine()
                    res = engine.enroll(label or value, img)
                    if not res.get("ok"):
                        raise HTTPException(status_code=400, detail=f"Face enrolment failed: {res.get('error')}")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

        entry_id = db.watchlist_add(kind="face", value=value, label=label, note=note, actor=actor)
        return {"ok": True, "id": entry_id, "kind": "face", "label": label or value}

    raise HTTPException(status_code=400, detail="Invalid watchlist kind. Must be 'plate' or 'face'")


@app.delete("/api/watchlist/{entry_id}")
async def remove_watchlist_entry(entry_id: int, actor: str = "operator"):
    ok = db.watchlist_remove(entry_id, actor=actor)
    if not ok:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"ok": True, "id": entry_id}


# --------------------------------------------------------------------------
# Multi-Camera ReID & Cross-Camera Tracking
# --------------------------------------------------------------------------
@app.get("/api/reid/trajectories")
async def get_reid_trajectories():
    from .pipeline.reid import reid_engine
    return reid_engine.get_trajectory_data()


@app.get("/api/reid/crops")
async def get_reid_crops():
    from .pipeline.reid import reid_engine
    return {"crops": list(reid_engine.recent_crops)}


@app.get("/api/reid/affinity")
async def get_reid_affinity():
    from .pipeline.reid import reid_engine
    return {"associations": list(reid_engine.association_history)}


# --------------------------------------------------------------------------
# Analytics, Audit & Stats
# --------------------------------------------------------------------------
@app.get("/api/stats")
async def get_dashboard_stats(window_hours: int = 24):
    return db.stats(window_seconds=window_hours * 3600)


@app.get("/api/audit")
async def get_audit_trail(limit: int = 100):
    return {"audit": db.query_audit(limit=limit)}


# --------------------------------------------------------------------------
# Real-Time WebSocket Alerts Feed
# --------------------------------------------------------------------------
@app.websocket("/ws/alerts")
async def alert_websocket(ws: WebSocket):
    await ws.accept()
    q = bus.subscribe()
    logger.info("WebSocket client connected. Total subscribers: %d", bus.subscriber_count)

    loop = asyncio.get_running_loop()

    try:
        while True:
            # Poll bus queue in executor so it does not block the async event loop
            try:
                msg = await loop.run_in_executor(None, lambda: q.get(timeout=1.0))
                await ws.send_text(json.dumps(msg))
            except Exception:
                # Timeout on queue read — send keepalive ping to maintain connection
                await ws.send_text(json.dumps({"type": "ping", "ts": db.iso_now()}))
    except (WebSocketDisconnect, ConnectionResetError):
        pass
    finally:
        bus.unsubscribe(q)
        logger.info("WebSocket client disconnected. Total subscribers: %d", bus.subscriber_count)
