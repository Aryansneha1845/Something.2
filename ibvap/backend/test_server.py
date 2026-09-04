"""Verification test script for IBVAP FastAPI backend and endpoints."""
import sys
import time
from fastapi.testclient import TestClient

print("Importing app.main...")
from app.main import app

print("Initializing test client...")
with TestClient(app) as client:
    print("Testing GET /api/status...")
    res = client.get("/api/status")
    print("Status code:", res.status_code)
    print("Response:", res.json())
    assert res.status_code == 200

    print("\nTesting GET /api/cameras...")
    res = client.get("/api/cameras")
    print("Status code:", res.status_code)
    cams = res.json()
    print(f"Loaded {len(cams)} cameras:")
    for c in cams:
        print(f" - {c['id']}: {c['name']} (source: {c['source']})")
    assert res.status_code == 200

    print("\nTesting GET /api/events...")
    res = client.get("/api/events")
    print("Status code:", res.status_code)
    assert res.status_code == 200

    print("\nTesting Natural Language Search: GET /api/search?q=crawling...")
    res = client.get("/api/search?q=crawling near fence")
    print("Status code:", res.status_code)
    print("Search parsed filters:", res.json().get("parsed_filters"))
    assert res.status_code == 200

    print("\nTesting GET / (Dashboard HTML)...")
    res = client.get("/")
    print("Status code:", res.status_code)
    assert res.status_code == 200
    assert "IBVAP" in res.text

    print("\nTesting GET /api/watchlist...")
    res = client.get("/api/watchlist")
    print("Status code:", res.status_code)
    assert res.status_code == 200

    print("\nTesting GET /api/reid/trajectories...")
    res = client.get("/api/reid/trajectories")
    print("Status code:", res.status_code)
    assert res.status_code == 200
    traj_data = res.json()
    assert "fov_lobes" in traj_data
    assert "CAM-01" in traj_data["fov_lobes"]
    print("FOV lobes loaded successfully:", list(traj_data["fov_lobes"].keys()))

    print("\nTesting GET /api/reid/crops...")
    res = client.get("/api/reid/crops")
    print("Status code:", res.status_code)
    assert res.status_code == 200

    print("\nTesting GET /api/reid/affinity...")
    res = client.get("/api/reid/affinity")
    print("Status code:", res.status_code)
    assert res.status_code == 200

    print("\nTesting POST /api/watchlist/enroll (Plate)...")
    res = client.post("/api/watchlist/enroll", json={
        "kind": "plate",
        "value": "DL01AB1234",
        "label": "Suspect Vehicle 01",
        "note": "Flagged border vehicle"
    })
    print("Enroll plate response:", res.json())
    assert res.status_code == 200

    print("\nWaiting 3 seconds for workers to process video frames...")
    time.sleep(3)

    print("\nChecking camera worker live frames...")
    for c in cams:
        cid = c["id"]
        res = client.get(f"/api/cameras/{cid}/snapshot")
        print(f"Camera {cid} snapshot status: {res.status_code}, length: {len(res.content)} bytes")

print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
