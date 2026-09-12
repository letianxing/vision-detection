const state = {
  source: "camera",
  ws: null,
};

const $ = (id) => document.getElementById(id);

function formatNumber(value, digits = 2) {
  if (Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function updateCameras(payload) {
  const select = $("cameraSelect");
  const current = select.value;
  select.innerHTML = "";

  const cameras = payload?.cameras || [];
  if (!cameras.length) {
    const option = document.createElement("option");
    option.value = "0";
    option.textContent = "No local cameras found";
    select.appendChild(option);
    return;
  }

  cameras.forEach((camera) => {
    const option = document.createElement("option");
    option.value = String(camera.index);
    option.textContent = `${camera.name} (${camera.index})`;
    select.appendChild(option);
  });
  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}

function updateState(payload) {
  if (!payload) return;

  $("connectionStatus").textContent = `topic stamp ${formatNumber(payload.stamp, 3)}`;
  $("humanPill").textContent = `near human: ${payload.near_human_present ? "true" : "false"}`;
  $("humanPill").classList.toggle("on", Boolean(payload.near_human_present));

  const emotion = Math.max(-1, Math.min(1, Number(payload.v_user_raw || 0)));
  const rawEmotion = Math.max(-1, Math.min(1, Number(payload.v_user_raw_raw || 0)));
  $("emotionValue").textContent = formatNumber(emotion, 2);
  $("emotionRawValue").textContent = formatNumber(rawEmotion, 2);
  $("emotionLabel").textContent = payload.emotion_label || "none";
  $("emotionConfidence").textContent = `${formatNumber((payload.emotion_confidence || 0) * 100, 0)}%`;
  $("emotionQuality").textContent = payload.emotion_valid ? "valid" : "held";
  $("emotionArousal").textContent = formatNumber(payload.emotion_arousal || 0, 2);
  $("emotionBackend").textContent = payload.emotion_backend || "none";
  $("emotionNeedle").style.left = `${((emotion + 1) / 2) * 100}%`;

  $("userValue").textContent = payload.nearest_user_id || "none";
  $("roleValue").textContent = payload.nearest_user_role || "none";
  $("personCount").textContent = String(payload.person_count ?? 0);
  $("areaRatio").textContent = `${formatNumber((payload.nearest_person_bbox_area_ratio || 0) * 100, 1)}%`;

  const orient = payload.face_orient || {};
  $("yawValue").textContent = orient.valid ? `${formatNumber(orient.yaw_deg, 1)} deg` : "--";
  $("pitchValue").textContent = orient.valid ? `${formatNumber(orient.pitch_deg, 1)} deg` : "--";
  $("rollValue").textContent = orient.valid ? `${formatNumber(orient.roll_deg, 1)} deg` : "--";

  const novelty = $("noveltyList");
  novelty.innerHTML = "";
  const labels = payload.novelty_labels || [];
  if (!labels.length) {
    novelty.appendChild(chip("none"));
  } else {
    labels.forEach((label, index) => {
      novelty.appendChild(chip(`${label} ${formatNumber((payload.novelty_scores?.[index] || 0) * 100, 0)}%`));
    });
  }

  const gestures = $("gestureList");
  gestures.innerHTML = "";
  const events = payload.gestures || [];
  if (!events.length) {
    gestures.appendChild(eventItem("none"));
  } else {
    events.forEach((event) => {
      gestures.appendChild(eventItem(`${event.user_role}:${event.gesture} ${formatNumber(event.score, 2)}`));
    });
  }

  const people = $("peopleList");
  people.innerHTML = "";
  const personItems = payload.people || [];
  if (!personItems.length) {
    people.appendChild(eventItem("none"));
  } else {
    personItems.forEach((person) => {
      const item = document.createElement("div");
      item.className = "person-item";
      const az = person.has_azimuth ? `${formatNumber(person.azimuth_deg, 1)} deg` : "--";
      item.innerHTML = `
        <strong>${person.person_id || "unknown"} <span>${person.role || "unknown"}</span></strong>
        <small>az ${az} · gaze ${formatNumber(person.gaze_score, 2)} · facing ${formatNumber(person.body_facing_score, 2)}</small>
        <small>${person.engagement_status || "unknown"} · ${person.proxemic_space || "unknown"} · ${person.gesture || "none"}</small>
      `;
      people.appendChild(item);
    });
  }
}

function chip(text) {
  const item = document.createElement("span");
  item.className = "chip";
  item.textContent = text;
  return item;
}

function eventItem(text) {
  const item = document.createElement("span");
  item.className = "event";
  item.textContent = text;
  return item;
}

function connectWs() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => {
    $("connectionStatus").textContent = "websocket connected";
  };
  ws.onmessage = (message) => {
    const payload = JSON.parse(message.data);
    if (payload.type === "snapshot") {
      updateState(payload.state);
      updateCameras(payload.cameras);
    }
    if (payload.type === "state") updateState(payload.state);
    if (payload.type === "cameras") updateCameras(payload.cameras);
  };
  ws.onclose = () => {
    $("connectionStatus").textContent = "websocket reconnecting";
    setTimeout(connectWs, 1000);
  };
}

async function loadInitial() {
  const cameraResponse = await fetch("/api/cameras");
  updateCameras(await cameraResponse.json());
  const stateResponse = await fetch("/api/state");
  const data = await stateResponse.json();
  updateState(data.state);
}

async function connectInput() {
  const body = {
    source_type: state.source,
    camera_index: Number($("cameraSelect").value || 0),
    image_topic: $("topicInput").value || "/camera/image_raw",
    video_path: $("videoInput").value || "",
  };

  $("connectMessage").textContent = "connecting";
  const response = await fetch("/api/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  $("connectMessage").textContent = result.message;
}

document.querySelectorAll(".segmented button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".segmented button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.source = button.dataset.source;
  });
});

$("connectButton").addEventListener("click", connectInput);

loadInitial().catch(() => {});
connectWs();
