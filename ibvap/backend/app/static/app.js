/**
 * IBVAP — Tactical C2 Command Dashboard Controller
 * Pure Vanilla JavaScript — Zero dependencies, works offline across all devices
 */

(function () {
  "use strict";

  // Smart Backend Host Resolution:
  // Works when served via FastAPI, opened directly via file://, or accessed from a remote phone/tablet
  function getApiBase() {
    if (window.location.protocol === "http:" || window.location.protocol === "https:") {
      return window.location.origin;
    }
    return localStorage.getItem("ibvap_server_url") || "http://localhost:8000";
  }

  function getWsBase() {
    const api = getApiBase();
    if (api.startsWith("https://")) {
      return api.replace("https://", "wss://");
    }
    return api.replace("http://", "ws://");
  }

  function apiFetch(path, options) {
    const base = getApiBase();
    const url = path.startsWith("http") ? path : `${base}${path.startsWith("/") ? "" : "/"}${path}`;
    return fetch(url, options);
  }

  // App State
  const state = {
    cameras: [],
    focusedCameraId: null,
    streamMode: "annotated", // "annotated" | "raw"
    soundEnabled: true,
    unreadAlerts: 0,
    ws: null,
    wsReconnectTimeout: null,
    currentFilter: "all",
    recentEvents: [],
    recentPlates: [],
  };

  // Sound generator using Web Audio API (no external audio files required!)
  let audioCtx = null;
  function playAlertBeep(frequency = 880, duration = 0.25) {
    if (!state.soundEnabled) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === "suspended") {
        audioCtx.resume();
      }
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio alert suppressed:", e);
    }
  }

  // DOM Elements
  const el = {
    camCount: document.getElementById("cam-count"),
    fpsCounter: document.getElementById("fps-counter"),
    tracksCounter: document.getElementById("tracks-counter"),
    globalThreat: document.getElementById("global-threat"),
    wsStatus: document.getElementById("ws-status"),
    clock: document.getElementById("clock"),
    cameraGrid: document.getElementById("camera-grid"),
    inspector: document.getElementById("camera-inspector"),
    inspectorStream: document.getElementById("inspector-stream"),
    inspectorCamName: document.getElementById("inspector-cam-name"),
    inspectorCamMeta: document.getElementById("inspector-cam-meta"),
    hudFps: document.getElementById("hud-fps"),
    hudNight: document.getElementById("hud-night"),
    hudTracks: document.getElementById("hud-tracks"),
    hudThreat: document.getElementById("hud-threat"),
    dockAlertsList: document.getElementById("dock-alerts-list"),
    activeAlertCount: document.getElementById("active-alert-count"),
    unreadCount: document.getElementById("unread-count"),
    incidentsContainer: document.getElementById("incidents-container"),
    nlSearchInput: document.getElementById("nl-search-input"),
    btnRunSearch: document.getElementById("btn-run-search"),
    platesTableBody: document.getElementById("plates-table-body"),
    watchlistGrid: document.getElementById("watchlist-grid"),
    ptzCamSelect: document.getElementById("ptz-cam-select"),
    ptzLogTerminal: document.getElementById("ptz-log-terminal"),
    evidenceModal: document.getElementById("evidence-modal"),
    modalImage: document.getElementById("modal-image"),
    modalDetails: document.getElementById("modal-details"),
    modalTitle: document.getElementById("modal-title"),
    enrollModal: document.getElementById("enroll-modal"),
    sectorMapCanvas: document.getElementById("sector-map-canvas"),
    serverModal: document.getElementById("server-modal"),
    inputServerUrl: document.getElementById("input-server-url"),
    serverHostLabel: document.getElementById("server-host-label"),
  };

  // Clock
  function updateClock() {
    const now = new Date();
    el.clock.textContent = now.toTimeString().split(" ")[0];
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Server Host Config Modal
  function updateServerLabel() {
    if (!el.serverHostLabel) return;
    const base = getApiBase();
    try {
      const u = new URL(base);
      el.serverHostLabel.textContent = u.host;
    } catch (e) {
      el.serverHostLabel.textContent = base;
    }
  }
  updateServerLabel();

  document.getElementById("btn-server-config")?.addEventListener("click", () => {
    if (el.inputServerUrl) el.inputServerUrl.value = getApiBase();
    if (el.serverModal) el.serverModal.style.display = "flex";
  });
  document.getElementById("btn-close-server")?.addEventListener("click", () => {
    if (el.serverModal) el.serverModal.style.display = "none";
  });
  document.getElementById("btn-save-server")?.addEventListener("click", () => {
    const val = el.inputServerUrl.value.trim().replace(/\/+$/, "");
    if (val) {
      localStorage.setItem("ibvap_server_url", val);
      if (el.serverModal) el.serverModal.style.display = "none";
      window.location.reload();
    }
  });
  document.getElementById("btn-reset-server")?.addEventListener("click", () => {
    localStorage.removeItem("ibvap_server_url");
    if (el.serverModal) el.serverModal.style.display = "none";
    window.location.reload();
  });

  // Tab Navigation
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      const targetPane = document.getElementById(`pane-${tab.dataset.tab}`);
      if (targetPane) {
        targetPane.classList.add("active");
      }
      if (tab.dataset.tab === "reid-tracking") {
        loadReidTracking();
      } else if (tab.dataset.tab === "sector-map") {
        drawSectorMap();
      } else if (tab.dataset.tab === "events-log") {
        loadEvents();
      } else if (tab.dataset.tab === "anpr-log") {
        loadPlates();
      } else if (tab.dataset.tab === "watchlist-tab") {
        loadWatchlist();
      } else if (tab.dataset.tab === "audit-tab") {
        loadAuditAndStats();
      }
      // Sync mobile bottom-bar highlight & close the drawer on selection
      syncBottomBar(tab.dataset.tab);
      closeNavDrawer();
    });
  });

  // --- Mobile nav: hamburger drawer + bottom tab bar --------------------
  function openNavDrawer() {
    document.body.classList.add("nav-open");
    document.getElementById("btn-nav-toggle")?.setAttribute("aria-expanded", "true");
  }
  function closeNavDrawer() {
    document.body.classList.remove("nav-open");
    document.getElementById("btn-nav-toggle")?.setAttribute("aria-expanded", "false");
  }
  function syncBottomBar(activeTabId) {
    document.querySelectorAll(".bottom-tab[data-tab-target]").forEach((b) => {
      b.classList.toggle("active", b.dataset.tabTarget === activeTabId);
    });
  }

  document.getElementById("btn-nav-toggle")?.addEventListener("click", () => {
    if (document.body.classList.contains("nav-open")) {
      closeNavDrawer();
    } else {
      openNavDrawer();
    }
  });
  document.getElementById("nav-backdrop")?.addEventListener("click", closeNavDrawer);

  // Bottom-bar buttons trigger the matching sidebar nav-tab (reuses all logic above)
  document.querySelectorAll(".bottom-tab[data-tab-target]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tabTarget;
      const navTab = document.querySelector(`.nav-tab[data-tab="${target}"]`);
      if (navTab) navTab.click();
    });
  });

  // "More" button opens the drawer so overflow tabs are reachable on phones
  document.getElementById("btn-bottom-more")?.addEventListener("click", openNavDrawer);

  // Stream Overlay Toggle (Annotated vs Raw)
  document.getElementById("btn-stream-annotated")?.addEventListener("click", () => {
    setStreamMode("annotated");
  });
  document.getElementById("btn-stream-raw")?.addEventListener("click", () => {
    setStreamMode("raw");
  });

  function setStreamMode(mode) {
    state.streamMode = mode;
    document.getElementById("btn-stream-annotated")?.classList.toggle("active", mode === "annotated");
    document.getElementById("btn-stream-raw")?.classList.toggle("active", mode === "raw");
    renderCameras();
    if (state.focusedCameraId) {
      focusCamera(state.focusedCameraId);
    }
  }

  // Audio Toggle
  document.getElementById("audio-toggle")?.addEventListener("click", () => {
    state.soundEnabled = !state.soundEnabled;
    const icon = document.getElementById("audio-icon");
    if (icon) icon.textContent = state.soundEnabled ? "🔔" : "🔕";
  });

  // --------------------------------------------------------------------------
  // Camera Grid & Streams
  // --------------------------------------------------------------------------
  async function loadCameras() {
    try {
      const res = await apiFetch("/api/cameras");
      if (!res.ok) throw new Error("Failed to load cameras");
      state.cameras = await res.json();
      renderCameras();
      populatePtzCamSelect();
      drawSectorMap();
    } catch (err) {
      console.error("Camera load error:", err);
      if (el.cameraGrid) {
        el.cameraGrid.innerHTML = `
          <div class="dock-empty" style="grid-column: 1 / -1; padding: 30px; text-align: center;">
            <h3 style="color:#ff3d00; margin-bottom: 8px;">Connecting to IBVAP Backend...</h3>
            <p style="color:#8899aa; margin-bottom: 14px;">Backend server at <code>${getApiBase()}</code></p>
            <button class="btn btn-sm btn-primary" onclick="localStorage.setItem('ibvap_server_url', prompt('Enter Backend IP:Port', 'http://127.0.0.1:8000')); location.reload();">
              Configure Server Address
            </button>
          </div>
        `;
      }
    }
  }

  function renderCameras() {
    if (!el.cameraGrid) return;
    el.cameraGrid.innerHTML = "";
    if (state.cameras.length === 0) {
      el.cameraGrid.innerHTML = `<div class="dock-empty">No surveillance cameras configured</div>`;
      return;
    }

    state.cameras.forEach((cam) => {
      const card = document.createElement("div");
      card.className = "camera-card";
      card.id = `card-${cam.id}`;

      const st = cam.status || {};
      const fps = st.fps || 0.0;
      const tracks = st.tracks || 0;
      const isNight = st.is_night;
      const threatScore = st.top_threat ? st.top_threat.score : 0;

      const endpoint = state.streamMode === "raw" ? "raw" : "stream";
      const streamUrl = `${getApiBase()}/api/cameras/${cam.id}/${endpoint}?t=${Date.now()}`;

      card.innerHTML = `
        <div class="cam-header">
          <span class="cam-title">${escapeHtml(cam.name)} (${cam.id})</span>
          <div class="cam-badges">
            <span class="hud-pill" id="badge-night-${cam.id}">${isNight ? "🌙 NIGHT" : "☀️ DAY"}</span>
            <span class="badge-fps" id="badge-fps-${cam.id}">${fps.toFixed(1)} FPS</span>
          </div>
        </div>
        <div class="cam-viewport" onclick="window.ibvapFocusCamera('${cam.id}')">
          <img src="${streamUrl}" alt="${cam.name}" onerror="this.onerror=null; this.src='${getApiBase()}/api/cameras/${cam.id}/snapshot?t=' + Date.now();"/>
          <div class="cam-overlay-stats">
            <span class="hud-pill" id="badge-tracks-${cam.id}">TRACKS: ${tracks}</span>
            <span class="hud-pill threat-badge ${threatScore >= 70 ? "threat-alert" : threatScore >= 40 ? "threat-amber" : "threat-normal"}" id="badge-threat-${cam.id}">
              THREAT: ${threatScore}
            </span>
          </div>
        </div>
        <div class="cam-footer">
          <span class="cam-summary" id="cam-summary-${cam.id}">${escapeHtml(st.summary || "Analyzing...")}</span>
          <button class="btn btn-sm" onclick="window.ibvapFocusCamera('${cam.id}')">Focus</button>
        </div>
      `;

      el.cameraGrid.appendChild(card);
    });
  }

  // Camera Focus / Inspector
  window.ibvapFocusCamera = function (cameraId) {
    const cam = state.cameras.find((c) => c.id === cameraId);
    if (!cam) return;
    state.focusedCameraId = cameraId;
    focusCamera(cameraId);
  };

  function focusCamera(cameraId) {
    const cam = state.cameras.find((c) => c.id === cameraId);
    if (!cam) return;
    el.inspectorCamName.textContent = `${cam.name} (${cam.id})`;
    el.inspectorCamMeta.textContent = `Sector: ${cam.sector} | Source: ${cam.source}`;

    const endpoint = state.streamMode === "raw" ? "raw" : "stream";
    el.inspectorStream.src = `${getApiBase()}/api/cameras/${cam.id}/${endpoint}?t=${Date.now()}`;
    el.inspector.style.display = "flex";
    document.getElementById("btn-reset-focus").style.display = "inline-block";

    // Scroll to inspector
    el.inspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  document.getElementById("btn-close-inspector")?.addEventListener("click", () => {
    el.inspector.style.display = "none";
    document.getElementById("btn-reset-focus").style.display = "none";
    el.inspectorStream.src = "";
    state.focusedCameraId = null;
  });

  document.getElementById("btn-reset-focus")?.addEventListener("click", () => {
    document.getElementById("btn-close-inspector")?.click();
  });

  document.getElementById("btn-refresh-cams")?.addEventListener("click", loadCameras);

  document.getElementById("btn-snapshot-now")?.addEventListener("click", () => {
    if (!state.focusedCameraId) return;
    window.open(`${getApiBase()}/api/cameras/${state.focusedCameraId}/snapshot?raw=false`, "_blank");
  });

  // --------------------------------------------------------------------------
  // Real-Time WebSocket Alerts Feed
  // --------------------------------------------------------------------------
  function initWebSocket() {
    if (state.ws) {
      try {
        state.ws.close();
      } catch (e) {}
    }

    const wsUrl = `${getWsBase()}/ws/alerts`;

    el.wsStatus.textContent = "CONNECTING";
    el.wsStatus.className = "telemetry-val ws-pill ws-offline";

    try {
      state.ws = new WebSocket(wsUrl);

      state.ws.onopen = () => {
        el.wsStatus.textContent = "ONLINE";
        el.wsStatus.className = "telemetry-val ws-pill ws-online";
      };

      state.ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          handleIncomingMessage(msg);
        } catch (e) {
          console.error("WS message parse error:", e);
        }
      };

      state.ws.onclose = () => {
        el.wsStatus.textContent = "RECONNECTING";
        el.wsStatus.className = "telemetry-val ws-pill ws-offline";
        clearTimeout(state.wsReconnectTimeout);
        state.wsReconnectTimeout = setTimeout(initWebSocket, 3000);
      };

      state.ws.onerror = () => {
        state.ws.close();
      };
    } catch (err) {
      el.wsStatus.textContent = "OFFLINE";
      clearTimeout(state.wsReconnectTimeout);
      state.wsReconnectTimeout = setTimeout(initWebSocket, 4000);
    }
  }

  function handleIncomingMessage(msg) {
    if (msg.type === "ping") return;

    if (msg.type === "event") {
      onSecurityEvent(msg);
    } else if (msg.type === "ptz") {
      logPtzCommand(msg);
    } else if (msg.type === "ack") {
      onEventAcknowledged(msg.event_id, msg.actor);
    }
  }

  function onSecurityEvent(ev) {
    state.recentEvents.unshift(ev);
    if (state.recentEvents.length > 100) state.recentEvents.pop();

    state.unreadAlerts++;
    if (el.unreadCount) el.unreadCount.textContent = state.unreadAlerts;
    if (el.activeAlertCount) el.activeAlertCount.textContent = state.unreadAlerts;

    const threat = ev.threat_score || 0;

    // Trigger alert sound & global threat styling
    if (threat >= 70) {
      playAlertBeep(920, 0.35);
      setGlobalThreatLevel("CRITICAL", "threat-alert");
    } else if (threat >= 40) {
      playAlertBeep(520, 0.2);
      if (!el.globalThreat.classList.contains("threat-alert")) {
        setGlobalThreatLevel("ELEVATED", "threat-amber");
      }
    }

    // Flash camera card
    const card = document.getElementById(`card-${ev.camera_id}`);
    if (card) {
      card.classList.add("alert-active");
      setTimeout(() => card.classList.remove("alert-active"), 8000);
    }

    // Add card to right Live Alert Stream dock
    prependAlertToDock(ev);

    // If currently on Incidents tab, reload list
    const activeTab = document.querySelector(".nav-tab.active");
    if (activeTab && activeTab.dataset.tab === "events-log") {
      loadEvents();
    }

    // Redraw sector map to pulse alert circle
    drawSectorMap();
  }

  function setGlobalThreatLevel(label, className) {
    if (!el.globalThreat) return;
    el.globalThreat.textContent = label;
    el.globalThreat.className = `telemetry-val threat-badge ${className}`;
  }

  function prependAlertToDock(ev) {
    if (!el.dockAlertsList) return;
    const emptyMsg = el.dockAlertsList.querySelector(".dock-empty");
    if (emptyMsg) emptyMsg.remove();

    const item = document.createElement("div");
    const threat = ev.threat_score || 0;
    item.className = `dock-item ${threat >= 70 ? "high-threat" : ""}`;
    item.id = `dock-item-${ev.id}`;

    item.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-weight:700; font-size:11px; color:${threat >= 70 ? "#ff3d00" : "#00e5ff"}">
          ${escapeHtml(ev.kind.toUpperCase())}
        </span>
        <span class="threat-badge ${threat >= 70 ? "threat-alert" : threat >= 40 ? "threat-amber" : "threat-normal"}">
          SCORE ${threat}
        </span>
      </div>
      <div style="font-size:12px; color:#fff;">${escapeHtml(ev.caption || "Suspicious detection")}</div>
      <div style="display:flex; justify-content:space-between; font-size:11px; color:#8899aa;">
        <span>${escapeHtml(ev.camera_name || ev.camera_id)}</span>
        <span>${new Date(ev.ts * 1000).toTimeString().split(" ")[0]}</span>
      </div>
      <div style="margin-top:4px;">
        <button class="btn btn-sm btn-block" onclick="window.ibvapAckEvent(${ev.id})">Acknowledge</button>
      </div>
    `;

    el.dockAlertsList.insertBefore(item, el.dockAlertsList.firstChild);

    // Keep dock list capped at 15 items
    while (el.dockAlertsList.children.length > 15) {
      el.dockAlertsList.lastChild.remove();
    }
  }

  function onEventAcknowledged(eventId, actor) {
    const dockItem = document.getElementById(`dock-item-${eventId}`);
    if (dockItem) {
      dockItem.style.opacity = "0.5";
      const btn = dockItem.querySelector("button");
      if (btn) {
        btn.textContent = `Ack by ${actor}`;
        btn.disabled = true;
      }
    }
    const incidentCard = document.getElementById(`incident-${eventId}`);
    if (incidentCard) {
      const ackBtn = incidentCard.querySelector(".btn-ack");
      if (ackBtn) {
        ackBtn.textContent = `✓ Acknowledged (${actor})`;
        ackBtn.classList.remove("btn-primary");
        ackBtn.disabled = true;
      }
    }
  }

  window.ibvapAckEvent = async function (eventId) {
    try {
      const res = await apiFetch(`/api/events/${eventId}/ack`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actor: "Duty Officer" }),
      });
      if (res.ok) {
        onEventAcknowledged(eventId, "Duty Officer");
        state.unreadAlerts = Math.max(0, state.unreadAlerts - 1);
        if (el.unreadCount) el.unreadCount.textContent = state.unreadAlerts;
        if (el.activeAlertCount) el.activeAlertCount.textContent = state.unreadAlerts;
      }
    } catch (e) {
      console.error("Ack error:", e);
    }
  };

  // --------------------------------------------------------------------------
  // Tactical Sector Map Canvas
  // --------------------------------------------------------------------------
  function drawSectorMap() {
    const canvas = el.sectorMapCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // Clear background
    ctx.fillStyle = "#080d14";
    ctx.fillRect(0, 0, W, H);

    // Grid gridlines
    ctx.strokeStyle = "rgba(0, 229, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = 0; y < H; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // Border line & Buffer zone
    const borderY = H * 0.65;
    ctx.fillStyle = "rgba(255, 61, 0, 0.08)";
    ctx.fillRect(0, borderY, W, H - borderY);

    // Barbed tripwire line
    ctx.strokeStyle = "#ff3d00";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.moveTo(0, borderY);
    ctx.lineTo(W, borderY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#ff3d00";
    ctx.font = "11px 'SF Mono', monospace";
    ctx.fillText("INTERNATIONAL PERIMETER LINE / VIRTUAL TRIPWIRE", 30, borderY - 8);

    // Checkpoint gate lane
    ctx.fillStyle = "rgba(40, 55, 70, 0.6)";
    ctx.fillRect(W * 0.65, 0, W * 0.18, H);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.strokeRect(W * 0.65, 0, W * 0.18, H);
    ctx.fillStyle = "#8899aa";
    ctx.fillText("POST 4 GATEWAY", W * 0.67, 30);

    // Plot camera nodes
    state.cameras.forEach((cam) => {
      const cx = (cam.map_x || 0.5) * W;
      const cy = (cam.map_y || 0.5) * H;

      const st = cam.status || {};
      const threat = st.top_threat ? st.top_threat.score : 0;
      const hasAlert = threat >= 60;

      // Draw FOV cone
      ctx.fillStyle = hasAlert ? "rgba(255, 61, 0, 0.2)" : "rgba(0, 229, 255, 0.12)";
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      const angle = cam.map_y < 0.5 ? Math.PI * 0.5 : Math.PI * 1.5;
      ctx.arc(cx, cy, 90, angle - 0.4, angle + 0.4);
      ctx.closePath();
      ctx.fill();

      // Pulsing alert ring if alert active
      if (hasAlert) {
        ctx.strokeStyle = "rgba(255, 61, 0, 0.8)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 22, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Camera node circle
      ctx.fillStyle = hasAlert ? "#ff3d00" : "#00e5ff";
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(cam.id, cx + 14, cy + 4);
      ctx.fillStyle = "#8899aa";
      ctx.font = "10px sans-serif";
      ctx.fillText(cam.name, cx + 14, cy + 18);
    });
  }

  // Click on map to focus camera
  el.sectorMapCanvas?.addEventListener("click", (evt) => {
    const rect = el.sectorMapCanvas.getBoundingClientRect();
    const scaleX = el.sectorMapCanvas.width / rect.width;
    const scaleY = el.sectorMapCanvas.height / rect.height;
    const clickX = (evt.clientX - rect.left) * scaleX;
    const clickY = (evt.clientY - rect.top) * scaleY;

    state.cameras.forEach((cam) => {
      const cx = (cam.map_x || 0.5) * el.sectorMapCanvas.width;
      const cy = (cam.map_y || 0.5) * el.sectorMapCanvas.height;
      const dist = Math.hypot(clickX - cx, clickY - cy);
      if (dist < 30) {
        document.querySelector('[data-tab="live-mosaic"]')?.click();
        window.ibvapFocusCamera(cam.id);
      }
    });
  });

  // --------------------------------------------------------------------------
  // Incidents & Natural Language Search
  // --------------------------------------------------------------------------
  async function loadEvents(query = "") {
    if (!el.incidentsContainer) return;
    el.incidentsContainer.innerHTML = `<div class="loading-state">Loading surveillance events...</div>`;

    let url = "/api/events?limit=40";
    if (query) {
      url = `/api/search?q=${encodeURIComponent(query)}&limit=40`;
    } else {
      if (state.currentFilter === "high-threat") url += "&min_threat=70";
      else if (state.currentFilter === "crawl") url += "&action=crawl";
      else if (state.currentFilter === "night") url += "&is_night=true";
      else if (state.currentFilter === "vehicle") url += "&object_class=vehicle";
      else if (state.currentFilter === "unack") url += "&acknowledged=false";
    }

    try {
      const res = await apiFetch(url);
      const data = await res.json();
      const events = query ? data.results || [] : data.events || [];
      renderEventsList(events);
    } catch (e) {
      el.incidentsContainer.innerHTML = `<div class="dock-empty">Failed to load events</div>`;
    }
  }

  function renderEventsList(events) {
    if (!el.incidentsContainer) return;
    el.incidentsContainer.innerHTML = "";
    if (events.length === 0) {
      el.incidentsContainer.innerHTML = `<div class="dock-empty">No incidents found matching criteria</div>`;
      return;
    }

    events.forEach((ev) => {
      const card = document.createElement("div");
      const threat = ev.threat_score || 0;
      card.className = `incident-card ${threat >= 70 ? "high-threat" : threat >= 40 ? "med-threat" : ""}`;
      card.id = `incident-${ev.id}`;

      const snapUrl = ev.snapshot ? `${getApiBase()}/snapshots/${ev.snapshot}` : "";
      const isAck = !!ev.acknowledged;

      card.innerHTML = `
        <div class="incident-left">
          ${
            snapUrl
              ? `<img src="${snapUrl}" class="incident-thumb" alt="Snapshot" onclick="window.ibvapShowEvidence('${snapUrl}', '${escapeHtml(ev.caption)}', ${threat})"/>`
              : `<div class="incident-thumb" style="display:flex;align-items:center;justify-content:center;color:#556;">📸</div>`
          }
          <div class="incident-meta">
            <span class="incident-caption">${escapeHtml(ev.caption || "Border Activity")}</span>
            <span class="incident-details">
              <strong>${escapeHtml(ev.camera_name || ev.camera_id)}</strong> • Sector ${escapeHtml(ev.sector || "North")} • 
              Kind: <span style="color:#00e5ff">${escapeHtml(ev.kind)}</span> • 
              Action: <span style="color:#ffb300">${escapeHtml(ev.action || "none")}</span> • 
              ${new Date(ev.ts * 1000).toLocaleString()}
            </span>
          </div>
        </div>
        <div class="incident-actions">
          <span class="threat-badge ${threat >= 70 ? "threat-alert" : threat >= 40 ? "threat-amber" : "threat-normal"}">
            SCORE ${threat}
          </span>
          <button class="btn btn-sm btn-ack ${isAck ? "" : "btn-primary"}" ${isAck ? "disabled" : ""} onclick="window.ibvapAckEvent(${ev.id})">
            ${isAck ? `✓ Acknowledged (${escapeHtml(ev.ack_by || "Operator")})` : "Acknowledge"}
          </button>
        </div>
      `;

      el.incidentsContainer.appendChild(card);
    });
  }

  // Evidence Zoom Modal
  window.ibvapShowEvidence = function (imageUrl, caption, threatScore) {
    el.modalImage.src = imageUrl;
    el.modalTitle.textContent = `Incident Evidence (Threat Score: ${threatScore})`;
    el.modalDetails.innerHTML = `<p style="font-size:14px; font-weight:600; margin-top:8px;">${caption}</p>`;
    el.evidenceModal.style.display = "flex";
  };

  document.getElementById("btn-close-modal")?.addEventListener("click", () => {
    el.evidenceModal.style.display = "none";
  });

  // Filter Chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.currentFilter = chip.dataset.filter;
      el.nlSearchInput.value = "";
      loadEvents();
    });
  });

  // Natural Language Search Button & Enter Key
  el.btnRunSearch?.addEventListener("click", () => {
    const q = el.nlSearchInput.value.trim();
    loadEvents(q);
  });
  el.nlSearchInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      el.btnRunSearch.click();
    }
  });

  // --------------------------------------------------------------------------
  // ANPR License Plates
  // --------------------------------------------------------------------------
  async function loadPlates() {
    try {
      const res = await apiFetch("/api/plates?limit=50");
      const data = await res.json();
      const plates = data.plates || [];
      state.recentPlates = plates;
      renderPlates(plates);
    } catch (e) {
      console.error("Plates load error:", e);
    }
  }

  function renderPlates(plates) {
    if (!el.platesTableBody) return;
    el.platesTableBody.innerHTML = "";
    if (plates.length === 0) {
      el.platesTableBody.innerHTML = `<tr><td colspan="7" class="text-center">No license plates captured yet</td></tr>`;
      return;
    }

    plates.forEach((p) => {
      const row = document.createElement("tr");
      const isHit = !!p.watchlist_hit;
      const snapUrl = p.snapshot ? `${getApiBase()}/snapshots/${p.snapshot}` : "";

      row.innerHTML = `
        <td>${new Date(p.ts * 1000).toLocaleTimeString()}</td>
        <td>${escapeHtml(p.camera_id || "--")}</td>
        <td><span class="plate-badge">${escapeHtml(p.plate)}</span></td>
        <td>${escapeHtml(p.vehicle_class || "vehicle")}</td>
        <td>${p.confidence ? (p.confidence * 100).toFixed(0) + "%" : "--"}</td>
        <td>
          <span class="hud-pill ${isHit ? "threat-alert" : "threat-normal"}">
            ${isHit ? "⚠️ WATCHLIST MATCH" : "CLEARED"}
          </span>
        </td>
        <td>
          ${snapUrl ? `<img src="${snapUrl}" style="height:32px;border-radius:2px;cursor:pointer;" onclick="window.open('${snapUrl}')"/>` : "--"}
        </td>
      `;
      el.platesTableBody.appendChild(row);
    });
  }

  document.getElementById("btn-refresh-plates")?.addEventListener("click", loadPlates);

  // --------------------------------------------------------------------------
  // Watchlist Management
  // --------------------------------------------------------------------------
  async function loadWatchlist() {
    try {
      const res = await apiFetch("/api/watchlist");
      const data = await res.json();
      const list = data.entries || [];
      renderWatchlist(list);
    } catch (e) {
      console.error("Watchlist error:", e);
    }
  }

  function renderWatchlist(list) {
    if (!el.watchlistGrid) return;
    el.watchlistGrid.innerHTML = "";
    if (list.length === 0) {
      el.watchlistGrid.innerHTML = `<div class="dock-empty">No target vehicles or identities enrolled in watchlist</div>`;
      return;
    }

    list.forEach((item) => {
      const card = document.createElement("div");
      card.className = "watchlist-card";
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span class="hud-pill">${item.kind === "plate" ? "🚗 LICENSE PLATE" : "👤 FACE IDENTITY"}</span>
          <button class="btn btn-sm" onclick="window.ibvapDeleteWatchlist(${item.id})">Remove</button>
        </div>
        <div style="font-size:16px; font-weight:700; font-family:'SF Mono',monospace; color:#00e5ff;">
          ${escapeHtml(item.value)}
        </div>
        <div style="font-size:13px; font-weight:600; color:#fff;">${escapeHtml(item.label || "No label")}</div>
        <div style="font-size:12px; color:#8899aa;">${escapeHtml(item.note || "Enrolled by operator")}</div>
        <div style="font-size:10px; color:#556677;">Added: ${new Date(item.added_ts * 1000).toLocaleDateString()} by ${escapeHtml(item.added_by || "admin")}</div>
      `;
      el.watchlistGrid.appendChild(card);
    });
  }

  window.ibvapDeleteWatchlist = async function (id) {
    if (!confirm("Remove this target from surveillance watchlist?")) return;
    try {
      await apiFetch(`/api/watchlist/${id}`, { method: "DELETE" });
      loadWatchlist();
    } catch (e) {
      console.error(e);
    }
  };

  document.getElementById("btn-open-enroll")?.addEventListener("click", () => {
    el.enrollModal.style.display = "flex";
  });
  document.getElementById("btn-close-enroll")?.addEventListener("click", () => {
    el.enrollModal.style.display = "none";
  });

  document.getElementById("enroll-kind")?.addEventListener("change", (e) => {
    const isPlate = e.target.value === "plate";
    document.getElementById("enroll-val-label").textContent = isPlate ? "License Plate Number" : "Target Subject Name / ID";
    document.getElementById("enroll-val").placeholder = isPlate ? "e.g. DL01AB1234" : "e.g. Target-01";
  });

  document.getElementById("enroll-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const kind = document.getElementById("enroll-kind").value;
    const value = document.getElementById("enroll-val").value.trim();
    const label = document.getElementById("enroll-label").value.trim();
    const note = document.getElementById("enroll-note").value.trim();

    try {
      const res = await apiFetch("/api/watchlist/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, value, label, note, actor: "Operator" }),
      });
      if (res.ok) {
        el.enrollModal.style.display = "none";
        document.getElementById("enroll-form").reset();
        loadWatchlist();
      }
    } catch (err) {
      alert("Failed to enroll target");
    }
  });

  // --------------------------------------------------------------------------
  // Simulated ONVIF PTZ Console & Presets
  // --------------------------------------------------------------------------
  function populatePtzCamSelect() {
    if (!el.ptzCamSelect) return;
    el.ptzCamSelect.innerHTML = "";
    state.cameras.forEach((cam) => {
      const opt = document.createElement("option");
      opt.value = cam.id;
      opt.textContent = `${cam.name} (${cam.id})`;
      el.ptzCamSelect.appendChild(opt);
    });
  }

  document.querySelectorAll(".dpad-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const camId = el.ptzCamSelect.value;
      if (!camId) return;
      const dir = btn.dataset.dir;
      let pan = 0.0,
        tilt = 0.0;
      if (dir === "up") tilt = 0.4;
      else if (dir === "down") tilt = -0.4;
      else if (dir === "left") pan = -0.5;
      else if (dir === "right") pan = 0.5;

      sendPtzCommand(camId, pan, tilt, 1.0, `manual-${dir}`);
    });
  });

  document.getElementById("ptz-zoom-in")?.addEventListener("click", () => {
    const camId = el.ptzCamSelect.value;
    if (camId) sendPtzCommand(camId, 0, 0, 1.5, "manual-zoom-in");
  });
  document.getElementById("ptz-zoom-out")?.addEventListener("click", () => {
    const camId = el.ptzCamSelect.value;
    if (camId) sendPtzCommand(camId, 0, 0, 0.8, "manual-zoom-out");
  });

  window.ibvapPtzPreset = function (name) {
    const camId = el.ptzCamSelect?.value || "CAM-01";
    if (name === "fence-gate") {
      sendPtzCommand(camId, -0.6, -0.2, 1.8, "preset:fence-gate");
    } else if (name === "road-lane") {
      sendPtzCommand(camId, 0.5, 0.1, 1.2, "preset:road-lane");
    }
  };

  async function sendPtzCommand(camId, pan, tilt, zoom, reason) {
    try {
      const res = await apiFetch(`/api/cameras/${camId}/ptz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pan, tilt, zoom, reason }),
      });
      const data = await res.json();
      if (data.ptz_action) {
        logPtzCommand(data.ptz_action);
      }
    } catch (e) {
      console.error("PTZ error:", e);
    }
  }

  function logPtzCommand(action) {
    if (!el.ptzLogTerminal) return;
    const ts = action.ts || new Date().toISOString();
    const log = `[${ts}] PTZ DISPATCH -> Cam ${action.camera_id} (${action.reason || "auto-track"})\n` +
      `PAN: ${action.pan || 0} | TILT: ${action.tilt || 0} | ZOOM: ${action.zoom || 1.0}\n` +
      `${action.onvif_soap_payload || "<SOAP-ENV:Envelope xmlns:SOAP-ENV='http://www.w3.org/2003/05/soap-envelope'>...</SOAP-ENV:Envelope>"}\n\n`;

    el.ptzLogTerminal.textContent = log + el.ptzLogTerminal.textContent;
  }

  // --------------------------------------------------------------------------
  // Tab 7: Threat Intelligence & Audit Trail
  // --------------------------------------------------------------------------
  async function loadAuditAndStats() {
    try {
      const statsRes = await apiFetch("/api/stats");
      if (statsRes.ok) {
        const stats = await statsRes.json();
        document.getElementById("metric-total-events").textContent = stats.total_events || 0;
        document.getElementById("metric-high-threat").textContent = stats.open_high_threat || 0;
        document.getElementById("metric-total-plates").textContent = stats.plate_reads || 0;
        document.getElementById("metric-peak-threat").textContent = stats.peak_threat || 0;
      }
      const auditRes = await apiFetch("/api/audit?limit=50");
      if (auditRes.ok) {
        const data = await auditRes.json();
        renderAuditTable(data.audit || []);
      }
    } catch (e) {
      console.error("Audit load error:", e);
    }
  }

  function renderAuditTable(records) {
    const tbody = document.getElementById("audit-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";
    if (records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="text-center">No audit trail entries found</td></tr>`;
      return;
    }
    records.forEach((r) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${new Date(r.ts * 1000).toLocaleString()}</td>
        <td><strong style="color:#00e5ff">${escapeHtml(r.actor || "system")}</strong></td>
        <td><span class="hud-pill">${escapeHtml(r.action)}</span></td>
        <td style="color:#8899aa; font-family:'SF Mono',monospace; font-size:11px;">${escapeHtml(r.detail || "--")}</td>
      `;
      tbody.appendChild(row);
    });
  }

  document.getElementById("btn-refresh-audit")?.addEventListener("click", loadAuditAndStats);

  // --------------------------------------------------------------------------
  // Simulation Handlers (For Immediate Demos)
  // --------------------------------------------------------------------------
  document.getElementById("btn-sim-breach")?.addEventListener("click", () => {
    const now = Math.floor(Date.now() / 1000);
    const simEvent = {
      id: Date.now(),
      ts: now,
      camera_id: "CAM-01",
      camera_name: "North Perimeter Fence & Tripwire",
      sector: "SECTOR-NORTH",
      kind: "tripwire",
      action: "crawl",
      threat_score: 88,
      threat_level: "critical",
      caption: "HIGH THREAT: Target crawling detected breaching Perimeter Tripwire line into Restricted Buffer zone!",
      snapshot: "",
      acknowledged: 0,
      meta: { simulated: true },
    };
    onSecurityEvent(simEvent);
    sendPtzCommand("CAM-01", 0.45, -0.3, 1.4, "auto-lock:crawl-intrusion");
  });

  document.getElementById("btn-sim-plate")?.addEventListener("click", () => {
    const testPlates = ["DL01AB1234", "HR26DQ5555", "UP16AB9999"];
    const chosen = testPlates[Math.floor(Math.random() * testPlates.length)];
    const mockPlate = {
      id: Date.now(),
      ts: Math.floor(Date.now() / 1000),
      camera_id: "CAM-02",
      plate: chosen,
      vehicle_class: "car",
      confidence: 0.94,
      watchlist_hit: chosen === "DL01AB1234" ? 1 : 0,
    };
    state.recentPlates = [mockPlate, ...(state.recentPlates || [])];
    renderPlates(state.recentPlates);
    if (mockPlate.watchlist_hit) {
      onSecurityEvent({
        id: Date.now(),
        ts: mockPlate.ts,
        camera_id: "CAM-02",
        camera_name: "Sector 4 Vehicle Checkpoint Gate",
        sector: "SECTOR-EAST",
        kind: "watchlist",
        action: "stop",
        threat_score: 92,
        threat_level: "critical",
        caption: `CRITICAL WATCHLIST HIT: Flagged suspect vehicle detected with license plate ${chosen}!`,
        snapshot: "",
        acknowledged: 0,
      });
    }
  });

  // --------------------------------------------------------------------------
  // Multi-Camera ReID & Trajectory Tracking Architecture (Matching Diagram)
  // --------------------------------------------------------------------------
  let reidAnimFrame = null;
  let reidAnimPhase = 0;

  async function loadReidTracking() {
    renderReidMulticam();
    drawMessagePassingCanvas();

    try {
      const res = await apiFetch("/api/reid/trajectories");
      if (res.ok) {
        const trajData = await res.json();
        drawFovTrajectoriesCanvas(trajData);
        renderBpbCrops(trajData.recent_crops || []);
        renderAffinityStatus(trajData.associations || []);
      }
    } catch (e) {
      console.warn("ReID data fetch error:", e);
      // Fallback to synthetic demonstration mode
      drawFovTrajectoriesCanvas(getFallbackTrajData());
      renderBpbCrops(getFallbackCrops());
    }
  }

  function renderReidMulticam() {
    const strip = document.getElementById("multicam-strip");
    if (!strip) return;
    strip.innerHTML = "";

    const cams = state.cameras.slice(0, 3);
    if (cams.length === 0) {
      strip.innerHTML = `<div class="dock-empty" style="grid-column:1/-1;">Initializing Multi-Camera Network...</div>`;
      return;
    }

    cams.forEach((cam, idx) => {
      const card = document.createElement("div");
      card.className = "multicam-card";
      const tagLabel = idx === 0 ? "Camera #1" : idx === 1 ? "Camera #2" : "Camera #M";
      card.innerHTML = `
        <span class="multicam-tag">${tagLabel} (${cam.id})</span>
        <img src="${getApiBase()}/api/cameras/${cam.id}/stream?t=${Date.now()}" alt="${cam.name}" onerror="this.onerror=null; this.src='${getApiBase()}/api/cameras/${cam.id}/snapshot?t=' + Date.now();"/>
      `;
      strip.appendChild(card);
    });
  }

  function renderBpbCrops(crops) {
    const grid = document.getElementById("bpb-crops-grid");
    if (!grid) return;
    grid.innerHTML = "";

    const displayCrops = (crops && crops.length > 0) ? crops : getFallbackCrops();

    displayCrops.forEach((c) => {
      const card = document.createElement("div");
      card.className = "bpb-crop-card";
      const imgSrc = c.crop_b64 ? `data:image/jpeg;base64,${c.crop_b64}` : "";

      card.innerHTML = `
        ${imgSrc ? `<img src="${imgSrc}" class="bpb-crop-thumb" alt="${c.worker_name}"/>` : `<div class="bpb-crop-thumb" style="background:#1a2332; display:flex; align-items:center; justify-content:center; font-size:24px;">👷</div>`}
        <div class="bpb-meta">
          <span class="bpb-id-pill" style="background:${c.color || "#00e5ff"}">${escapeHtml(c.worker_name || "Worker")}</span>
          <div style="font-size:10px; color:#fff; font-weight:600; margin-top:2px;">${escapeHtml(c.signature || "Attire")}</div>
          <div class="bpb-parts-legend">
            <span style="color:#00e5ff;">Head</span> • <span style="color:#ffb300;">Torso</span> • <span style="color:#90a4ae;">Legs</span>
          </div>
          <div style="font-size:9px; color:#556677; font-family:'SF Mono',monospace; margin-top:2px;">Vector: 64-D</div>
        </div>
      `;
      grid.appendChild(card);
    });
  }

  function renderAffinityStatus(assocs) {
    const box = document.getElementById("affinity-status-box");
    if (!box) return;
    if (!assocs || assocs.length === 0) {
      box.innerHTML = `<div>[Affinity] Exemplar clustering active. Responsibility r(i,k): 0.88, Availability a(i,k): 0.94</div>`;
      return;
    }
    box.innerHTML = assocs.map(a =>
      `<div>[${a.source_cam}] ${a.worker_name} &rarr; r(i,k): ${a.responsibility_r}, a(i,k): ${a.availability_a} (Exemplar: MATCH)</div>`
    ).join("");
  }

  // --------------------------------------------------------------------------
  // Message-Passing Affinity Diagram Canvas (Matching Diagram Section c Left)
  // --------------------------------------------------------------------------
  function drawMessagePassingCanvas() {
    const canvas = document.getElementById("message-passing-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.fillStyle = "#080c12";
    ctx.fillRect(0, 0, W, H);

    ctx.font = "bold 10px sans-serif";

    // Divider line between Responsibilities and Availabilities
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(W * 0.48, 20);
    ctx.lineTo(W * 0.48, H - 20);
    ctx.stroke();
    ctx.setLineDash([]);

    // --- LEFT HALF: Sending Responsibilities r(i, k) ---
    ctx.fillStyle = "#fff";
    ctx.fillText("Sending Responsibilities", 20, 24);

    const pt_k = { x: 50, y: 70, label: "Candidate\nExemplar k", isEx: true };
    const pt_kp = { x: 135, y: 70, label: "Competing\nCandidate k'", isEx: false };
    const pt_i = { x: 100, y: 200, label: "Data Point i", isEx: false };

    // Draw arrows from i to k
    drawArrow(ctx, pt_i.x - 10, pt_i.y - 10, pt_k.x + 10, pt_k.y + 15, "#ff6d00", "r(i, k)");
    // Draw competing suppression arrow from k' to i
    drawArrow(ctx, pt_kp.x - 5, pt_kp.y + 15, pt_i.x + 8, pt_i.y - 10, "#ff1744", "a(i, k')");

    drawNode(ctx, pt_k.x, pt_k.y, pt_k.isEx);
    drawNode(ctx, pt_kp.x, pt_kp.y, pt_kp.isEx);
    drawNode(ctx, pt_i.x, pt_i.y, pt_i.isEx);

    ctx.fillStyle = "#8899aa";
    ctx.fillText("Candidate", pt_k.x - 20, pt_k.y - 18);
    ctx.fillText("Exemplar k", pt_k.x - 22, pt_k.y - 8);

    ctx.fillText("Competing", pt_kp.x - 15, pt_kp.y - 18);
    ctx.fillText("Candidate k'", pt_kp.x - 18, pt_kp.y - 8);

    ctx.fillText("Data Point i", pt_i.x - 22, pt_i.y + 20);

    // --- RIGHT HALF: Sending Availabilities a(i, k) ---
    ctx.fillStyle = "#fff";
    ctx.fillText("Sending Availabilities", W * 0.52 + 10, 24);

    const pt_rk = { x: W * 0.65, y: 100, label: "Candidate\nExemplar k", isEx: true };
    const pt_ri = { x: W * 0.78, y: 205, label: "Data Point i", isEx: false };
    const pt_rip = { x: W * 0.88, y: 85, label: "Supporting\nPoint i'", isEx: false };

    drawArrow(ctx, pt_rip.x - 15, pt_rip.y + 5, pt_rk.x + 15, pt_rk.y - 5, "#ff9100", "r(i', k)");
    drawArrow(ctx, pt_rk.x + 8, pt_rk.y + 15, pt_ri.x - 8, pt_ri.y - 12, "#ffab00", "a(i, k)");

    drawNode(ctx, pt_rk.x, pt_rk.y, pt_rk.isEx);
    drawNode(ctx, pt_rip.x, pt_rip.y, pt_rip.isEx);
    drawNode(ctx, pt_ri.x, pt_ri.y, pt_ri.isEx);

    ctx.fillStyle = "#8899aa";
    ctx.fillText("Candidate", pt_rk.x - 20, pt_rk.y - 18);
    ctx.fillText("Exemplar k", pt_rk.x - 22, pt_rk.y - 8);

    ctx.fillText("Supporting", pt_rip.x - 15, pt_rip.y - 18);
    ctx.fillText("Data Point i'", pt_rip.x - 18, pt_rip.y - 8);

    ctx.fillText("Data Point i", pt_ri.x - 22, pt_ri.y + 20);

    function drawNode(c, x, y, isExemplar) {
      c.beginPath();
      c.arc(x, y, 9, 0, Math.PI * 2);
      c.fillStyle = isExemplar ? "#ff3d00" : "#00e5ff";
      c.fill();
      c.lineWidth = 2;
      c.strokeStyle = isExemplar ? "#ff9100" : "#ffffff";
      c.stroke();
    }

    function drawArrow(c, fromx, fromy, tox, toy, color, text) {
      const headlen = 8;
      const angle = Math.atan2(toy - fromy, tox - fromx);
      c.strokeStyle = color;
      c.fillStyle = color;
      c.lineWidth = 2;

      c.beginPath();
      c.moveTo(fromx, fromy);
      c.lineTo(tox, toy);
      c.stroke();

      c.beginPath();
      c.moveTo(tox, toy);
      c.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
      c.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
      c.closePath();
      c.fill();

      if (text) {
        const mx = (fromx + tox) / 2 + Math.cos(angle - Math.PI / 2) * 10;
        const my = (fromy + toy) / 2 + Math.sin(angle - Math.PI / 2) * 10;
        c.font = "italic 9px 'SF Mono', monospace";
        c.fillText(text, mx - 12, my);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Cameras' Field of Views & Workers' Tracks Canvas (Matching Diagram Section c Right)
  // --------------------------------------------------------------------------
  function drawFovTrajectoriesCanvas(data) {
    const canvas = document.getElementById("fov-trajectories-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // Clear background
    ctx.fillStyle = "#070b10";
    ctx.fillRect(0, 0, W, H);

    // Subtle coordinate grid
    ctx.strokeStyle = "rgba(0, 229, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // 1. Draw Overlapping Camera FOV Elliptical Lobes (matching paper diagram)
    // Camera #1 FOV (Top-Center Lobe)
    drawFovLobe(ctx, W * 0.50, H * 0.30, W * 0.22, H * 0.22, "rgba(255, 120, 30, 0.14)", "rgba(255, 120, 30, 0.5)", "Camera #1");

    // Camera #2 FOV (Bottom-Left Lobe)
    drawFovLobe(ctx, W * 0.32, H * 0.65, W * 0.24, H * 0.24, "rgba(200, 80, 255, 0.14)", "rgba(200, 80, 255, 0.5)", "Camera #2");

    // Camera #M FOV (Bottom-Right Lobe)
    drawFovLobe(ctx, W * 0.72, H * 0.68, W * 0.24, H * 0.24, "rgba(0, 230, 180, 0.14)", "rgba(0, 230, 180, 0.5)", "Camera #M");

    // 2. Draw Multi-Camera Handover Boundaries
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);

    // Intersection between Cam 1 and Cam 2
    ctx.beginPath();
    ctx.arc(W * 0.38, H * 0.48, 30, 0, Math.PI * 2);
    ctx.stroke();

    // Intersection between Cam 2 and Cam M
    ctx.beginPath();
    ctx.arc(W * 0.52, H * 0.67, 30, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Draw Continuous Workers' Trajectories with Arrows and Handover Dashes
    reidAnimPhase += 0.05;

    // --- Worker A Trajectory (Orange): Crosses Cam 1 -> Cam 2 -> Cam M ---
    const pathA = [
      { x: W * 0.35, y: H * 0.22, cam: "CAM-01" },
      { x: W * 0.48, y: H * 0.20, cam: "CAM-01" },
      { x: W * 0.60, y: H * 0.24, cam: "CAM-01" },
      { x: W * 0.64, y: H * 0.36, cam: "HANDOVER", dash: true },
      { x: W * 0.60, y: H * 0.48, cam: "CAM-02" },
      { x: W * 0.46, y: H * 0.56, cam: "CAM-02" },
      { x: W * 0.36, y: H * 0.65, cam: "CAM-02" },
      { x: W * 0.42, y: H * 0.75, cam: "HANDOVER", dash: true },
      { x: W * 0.55, y: H * 0.78, cam: "CAM-03" },
      { x: W * 0.68, y: H * 0.76, cam: "CAM-03" },
    ];
    drawTrajectoryPath(ctx, pathA, "#ff6d00", "Worker A");

    // --- Worker B Trajectory (Lime Green): Crosses Cam 2 -> Cam M ---
    const pathB = [
      { x: W * 0.16, y: H * 0.62, cam: "CAM-02" },
      { x: W * 0.28, y: H * 0.56, cam: "CAM-02" },
      { x: W * 0.45, y: H * 0.50, cam: "CAM-02" },
      { x: W * 0.58, y: H * 0.52, cam: "CAM-01" },
      { x: W * 0.64, y: H * 0.58, cam: "HANDOVER", dash: true },
      { x: W * 0.70, y: H * 0.68, cam: "CAM-03" },
      { x: W * 0.68, y: H * 0.82, cam: "CAM-03" },
      { x: W * 0.76, y: H * 0.88, cam: "CAM-03" },
    ];
    drawTrajectoryPath(ctx, pathB, "#00e676", "Worker B");

    // --- Worker C Trajectory (Magenta): Moving in Cam 1 ---
    const pathC = [
      { x: W * 0.12, y: H * 0.68, cam: "CAM-02" },
      { x: W * 0.18, y: H * 0.52, cam: "CAM-02" },
      { x: W * 0.26, y: H * 0.42, cam: "HANDOVER", dash: true },
      { x: W * 0.36, y: H * 0.34, cam: "CAM-01" },
      { x: W * 0.46, y: H * 0.32, cam: "CAM-01" },
    ];
    drawTrajectoryPath(ctx, pathC, "#e040fb", "Worker C");

    // 4. Position Floating Worker Avatar Badges on the Canvas
    updateWorkerBadges([
      { id: "A", name: "Worker A", color: "#ff6d00", x: pathA[pathA.length - 2].x, y: pathA[pathA.length - 2].y, avatar: "👷", tag: "Orange Vest" },
      { id: "B", name: "Worker B", color: "#00e676", x: pathB[pathB.length - 2].x, y: pathB[pathB.length - 2].y, avatar: "👷‍♂️", tag: "Lime Vest" },
      { id: "C", name: "Worker C", color: "#e040fb", x: pathC[pathC.length - 1].x, y: pathC[pathC.length - 1].y, avatar: "👨‍🔧", tag: "Navy Jacket" },
    ]);

    function drawFovLobe(c, cx, cy, rx, ry, fill, stroke, label) {
      c.save();
      c.beginPath();
      c.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      c.fillStyle = fill;
      c.fill();
      c.strokeStyle = stroke;
      c.lineWidth = 1.5;
      c.stroke();
      c.fillStyle = "#ffffff";
      c.font = "bold 12px sans-serif";
      c.fillText(label, cx - 30, cy - ry + 24);
      c.restore();
    }

    function drawTrajectoryPath(c, pts, color, workerLabel) {
      if (pts.length < 2) return;
      c.save();

      // Draw path line
      for (let i = 0; i < pts.length - 1; i++) {
        const p1 = pts[i];
        const p2 = pts[i + 1];
        c.beginPath();
        c.moveTo(p1.x, p1.y);
        c.lineTo(p2.x, p2.y);
        c.strokeStyle = color;
        c.lineWidth = p2.dash ? 3 : 4;
        if (p2.dash) {
          c.setLineDash([6, 6]);
          c.strokeStyle = "#ffffff";
        } else {
          c.setLineDash([]);
        }
        c.stroke();
      }

      // Draw arrowhead at tip
      const last = pts[pts.length - 1];
      const prev = pts[pts.length - 2];
      const angle = Math.atan2(last.y - prev.y, last.x - prev.x);
      const arrowLen = 14;
      c.fillStyle = color;
      c.beginPath();
      c.moveTo(last.x, last.y);
      c.lineTo(last.x - arrowLen * Math.cos(angle - Math.PI / 6), last.y - arrowLen * Math.sin(angle - Math.PI / 6));
      c.lineTo(last.x - arrowLen * Math.cos(angle + Math.PI / 6), last.y - arrowLen * Math.sin(angle + Math.PI / 6));
      c.closePath();
      c.fill();

      // Animated traveling pulse dot
      const pulseIdx = Math.floor((reidAnimPhase % pts.length));
      const pulsePt = pts[pulseIdx] || pts[0];
      c.beginPath();
      c.arc(pulsePt.x, pulsePt.y, 6, 0, Math.PI * 2);
      c.fillStyle = "#ffffff";
      c.fill();
      c.beginPath();
      c.arc(pulsePt.x, pulsePt.y, 11, 0, Math.PI * 2);
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.stroke();

      c.restore();
    }
  }

  function updateWorkerBadges(workers) {
    const container = document.getElementById("worker-avatar-overlay");
    if (!container) return;
    container.innerHTML = "";

    workers.forEach(w => {
      const badge = document.createElement("div");
      badge.className = "avatar-badge";
      badge.style.left = `${w.x}px`;
      badge.style.top = `${w.y - 28}px`;
      badge.style.borderColor = w.color;
      badge.innerHTML = `
        <span>${w.avatar}</span>
        <span>${w.name}</span>
        <small style="color:${w.color}; font-size:9px;">[${w.tag}]</small>
      `;
      container.appendChild(badge);
    });
  }

  function getFallbackCrops() {
    return [
      {
        worker_name: "Worker A",
        signature: "Orange Vest + White Helmet",
        color: "#ff6d00",
        crop_b64: "",
      },
      {
        worker_name: "Worker B",
        signature: "Lime Vest + Yellow Helmet",
        color: "#00e676",
        crop_b64: "",
      },
      {
        worker_name: "Worker C",
        signature: "Navy Jacket + Blue Helmet",
        color: "#e040fb",
        crop_b64: "",
      },
    ];
  }

  function getFallbackTrajData() {
    return {
      global_tracks: {},
      recent_crops: getFallbackCrops(),
      associations: [],
    };
  }

  document.getElementById("btn-reid-step")?.addEventListener("click", () => {
    drawMessagePassingCanvas();
    drawFovTrajectoriesCanvas();
  });
  document.getElementById("btn-refresh-reid")?.addEventListener("click", loadReidTracking);

  async function pollSystemStatus() {
    try {
      const res = await apiFetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();

      if (el.camCount) el.camCount.textContent = `${data.active_cameras || 0} / ${data.total_cameras || 0}`;
      if (el.fpsCounter) el.fpsCounter.textContent = (data.aggregate_fps || 0).toFixed(1);
      if (el.tracksCounter) el.tracksCounter.textContent = data.active_tracks || 0;

      // Update individual camera card badges without interrupting the MJPEG stream
      const camRes = await apiFetch("/api/cameras");
      if (camRes.ok) {
        const cams = await camRes.json();
        state.cameras = cams;
        cams.forEach((c) => {
          const st = c.status || {};
          const fpsBadge = document.getElementById(`badge-fps-${c.id}`);
          if (fpsBadge) fpsBadge.textContent = `${(st.fps || 0).toFixed(1)} FPS`;

          const tracksBadge = document.getElementById(`badge-tracks-${c.id}`);
          if (tracksBadge) tracksBadge.textContent = `TRACKS: ${st.tracks || 0}`;

          const nightBadge = document.getElementById(`badge-night-${c.id}`);
          if (nightBadge) nightBadge.textContent = st.is_night ? "🌙 NIGHT" : "☀️ DAY";

          const threatBadge = document.getElementById(`badge-threat-${c.id}`);
          if (threatBadge) {
            const sc = st.top_threat ? st.top_threat.score : 0;
            threatBadge.textContent = `THREAT: ${sc}`;
            threatBadge.className = `hud-pill threat-badge ${sc >= 70 ? "threat-alert" : sc >= 40 ? "threat-amber" : "threat-normal"}`;
          }

          const sumText = document.getElementById(`cam-summary-${c.id}`);
          if (sumText && st.summary) sumText.textContent = st.summary;
        });

        if (state.focusedCameraId) {
          const focused = cams.find((c) => c.id === state.focusedCameraId);
          if (focused && focused.status) {
            if (el.hudFps) el.hudFps.textContent = `FPS: ${(focused.status.fps || 0).toFixed(1)}`;
            if (el.hudNight) el.hudNight.textContent = focused.status.is_night ? "MODE: NIGHT (ENHANCED)" : "MODE: DAY";
            if (el.hudTracks) el.hudTracks.textContent = `TRACKS: ${focused.status.tracks || 0}`;
            const fScore = focused.status.top_threat ? focused.status.top_threat.score : 0;
            if (el.hudThreat) el.hudThreat.textContent = `THREAT: ${fScore}`;
          }
        }
      }
    } catch (e) {
      console.warn("Poll status error:", e);
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initialization
  async function init() {
    await loadCameras();
    initWebSocket();
    await loadReidTracking();
    loadEvents();
    loadPlates();
    loadWatchlist();
    setInterval(pollSystemStatus, 3000);

    // Continuous real-time animation for Multi-Camera FOV Trajectories
    setInterval(() => {
      const activeTab = document.querySelector(".nav-tab.active");
      if (activeTab && activeTab.dataset.tab === "reid-tracking") {
        drawFovTrajectoriesCanvas();
      }
    }, 120);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
