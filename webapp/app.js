/* callpc call UI — raw WebRTC against the gateway's /api/offer.
 *
 * API base resolution (first hit wins):
 *   1. ?api=https://xyz.ngrok.app   (static-host deployment pointing at ngrok)
 *   2. window.CALLPC_API_BASE       (optional config.js on a static host)
 *   3. same origin                  (the gateway serves this app itself)
 * ICE servers come from GET /api/config (the gateway injects TURN creds there),
 * with a STUN-only fallback if that endpoint is unreachable.
 */
"use strict";

const API_BASE = (() => {
  const q = new URLSearchParams(location.search).get("api");
  const base = q || window.CALLPC_API_BASE || "";
  return base.replace(/\/+$/, "");
})();

const $ = (id) => document.getElementById(id);
const screens = {
  idle: $("screen-idle"),
  ringing: $("screen-ringing"),
  call: $("screen-call"),
  ended: $("screen-ended"),
};

let pc = null;          // RTCPeerConnection
let micStream = null;
let muted = false;
let timerStart = 0;
let timerHandle = null;
let talkingHandle = null;

function show(name) {
  for (const [k, el] of Object.entries(screens)) el.classList.toggle("active", k === name);
}

function setRingStatus(msg) { $("ring-status").textContent = msg; }

// ---------- call timer ----------
function startTimer() {
  timerStart = Date.now();
  const tick = () => {
    const s = Math.floor((Date.now() - timerStart) / 1000);
    $("timer").textContent =
      String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  };
  tick();
  timerHandle = setInterval(tick, 1000);
}
function stopTimer() { clearInterval(timerHandle); timerHandle = null; }

// ---------- "bot is talking" avatar glow (driven by remote audio level) ----------
function watchRemoteAudio(stream) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    talkingHandle = setInterval(() => {
      analyser.getByteFrequencyData(buf);
      const level = buf.reduce((a, b) => a + b, 0) / buf.length;
      $("call-avatar").classList.toggle("talking", level > 8);
    }, 150);
  } catch (_) { /* visual nicety only */ }
}

// ---------- ICE config ----------
async function iceServers() {
  try {
    const r = await fetch(API_BASE + "/api/config");
    if (r.ok) return (await r.json()).ice_servers;
  } catch (_) { /* fall through */ }
  return [{ urls: "stun:stun.l.google.com:19302" }];
}

// ---------- call lifecycle ----------
async function startCall() {
  show("ringing");
  setRingStatus("requesting microphone…");
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (e) {
    return endCall("Microphone blocked", "Allow mic access and try again.");
  }

  setRingStatus("calling…");
  try {
    pc = new RTCPeerConnection({ iceServers: await iceServers() });
    micStream.getTracks().forEach((t) => pc.addTrack(t, micStream));
    pc.ontrack = (e) => {
      $("remote").srcObject = e.streams[0];
      watchRemoteAudio(e.streams[0]);
    };
    pc.onconnectionstatechange = () => {
      if (!pc) return;
      if (pc.connectionState === "connected") {
        show("call");
        startTimer();
      } else if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
        if (screens.call.classList.contains("active") || screens.ringing.classList.contains("active")) {
          endCall("Call ended", pc.connectionState === "failed" ? "connection failed" : "");
        }
      }
    };

    await pc.setLocalDescription(await pc.createOffer());
    // non-trickle: wait (max 3 s) for ICE gathering so the offer carries candidates
    await new Promise((res) => {
      if (pc.iceGatheringState === "complete") return res();
      const t = setTimeout(res, 3000);
      pc.addEventListener("icegatheringstatechange", () => {
        if (pc.iceGatheringState === "complete") { clearTimeout(t); res(); }
      });
    });

    setRingStatus("ringing…");
    const r = await fetch(API_BASE + "/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!r.ok) return endCall("Couldn't reach your computer", `gateway answered HTTP ${r.status}`);
    await pc.setRemoteDescription(await r.json());
    // connectionstatechange flips to the in-call screen when media lands
  } catch (e) {
    console.error(e);
    endCall("Couldn't reach your computer", e.message);
  }
}

function teardown() {
  stopTimer();
  clearInterval(talkingHandle);
  talkingHandle = null;
  if (pc) { try { pc.close(); } catch (_) {} pc = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  $("remote").srcObject = null;
  muted = false;
  syncMuteUi();
}

function endCall(title, detail) {
  teardown();
  $("ended-title").textContent = title || "Call ended";
  $("ended-detail").textContent = detail || "";
  show("ended");
}

// ---------- mute ----------
function syncMuteUi() {
  $("ic-mic").style.display = muted ? "none" : "";
  $("ic-mic-off").style.display = muted ? "" : "none";
  $("mute-hint").textContent = muted ? "unmute" : "mute";
  $("btn-mute").classList.toggle("engaged", muted);
}
function toggleMute() {
  muted = !muted;
  if (micStream) micStream.getAudioTracks().forEach((t) => (t.enabled = !muted));
  syncMuteUi();
}

// ---------- wire up ----------
$("btn-call").onclick = startCall;
$("btn-again").onclick = startCall;
$("btn-cancel").onclick = () => endCall("Call cancelled", "");
$("btn-hangup").onclick = () => endCall("Call ended", "");
$("btn-mute").onclick = toggleMute;
