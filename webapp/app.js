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
let micStream = null;   // raw mic (kept for teardown)
let sendStream = null;  // mic -> gain -> destination; what we actually send
let micGain = null;     // mic gating = gain 0/1 (NOT track.enabled=false: that
                        // thins the RTP flow and the server's idle watcher kills
                        // the receiver — unmute after ~10 s went silent until redial)
let pttMode = true;     // default: push-to-talk; toggleable to hands-free
let pttLive = false;    // true while the PTT button is held
let timerStart = 0;
let timerHandle = null;
let talkingHandle = null;
let micVizHandle = null;
let ringHandle = null;
let audioCtx = null;    // one shared AudioContext (created on the call gesture)

function actx() {
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

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

// ---------- ringback tone while calling (subtle: two soft sines, 1.2s on / ~3s off) ----------
function ringOnce() {
  const c = actx();
  const t = c.currentTime;
  const g = c.createGain();
  g.connect(c.destination);
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(0.05, t + 0.08);   // quiet — barely-there ringback
  g.gain.setValueAtTime(0.05, t + 1.0);
  g.gain.linearRampToValueAtTime(0, t + 1.25);
  for (const f of [440, 480]) {                      // classic ringback dyad
    const o = c.createOscillator();
    o.type = "sine";
    o.frequency.value = f;
    o.connect(g);
    o.start(t);
    o.stop(t + 1.3);
  }
}
function startRingtone() {
  stopRingtone();
  try { ringOnce(); ringHandle = setInterval(ringOnce, 4200); } catch (_) {}
}
function stopRingtone() {
  clearInterval(ringHandle);
  ringHandle = null;
}

// ---------- "bot is talking" avatar glow (driven by remote audio level) ----------
function watchRemoteAudio(stream) {
  try {
    const c = actx();
    const src = c.createMediaStreamSource(stream);
    const analyser = c.createAnalyser();
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

// ---------- caller mic meter (you talking / stopped; flat while muted) ----------
function watchMicAudio(stream) {
  try {
    const c = actx();
    const src = c.createMediaStreamSource(stream);
    const analyser = c.createAnalyser();
    analyser.fftSize = 256;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const bars = document.querySelectorAll("#mic-viz i");
    const seg = Math.floor(buf.length / 2 / bars.length);  // voice lives in the lower half
    micVizHandle = setInterval(() => {
      analyser.getByteFrequencyData(buf);
      bars.forEach((bar, i) => {
        let sum = 0;
        for (let j = i * seg; j < (i + 1) * seg; j++) sum += buf[j];
        const v = Math.min(1, sum / seg / 150);
        bar.style.transform = `scaleY(${Math.max(0.15, v)})`;
        bar.style.opacity = v > 0.1 ? "0.95" : "0.3";
      });
    }, 90);
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
    micStream = await navigator.mediaDevices.getUserMedia({
      // explicit AEC/NS: on speakerphone the bot otherwise hears itself and
      // barges in on its own reply
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
  } catch (e) {
    return endCall("Microphone blocked", "Allow mic access and try again.");
  }

  setRingStatus("calling…");
  startRingtone();  // the call tap is the user gesture WebAudio needs
  try {
    const c = actx();
    const src = c.createMediaStreamSource(micStream);
    micGain = c.createGain();
    const dest = c.createMediaStreamDestination();
    src.connect(micGain);
    micGain.connect(dest);
    sendStream = dest.stream;

    pc = new RTCPeerConnection({ iceServers: await iceServers() });
    sendStream.getTracks().forEach((t) => pc.addTrack(t, sendStream));
    pc.ontrack = (e) => {
      $("remote").srcObject = e.streams[0];
      watchRemoteAudio(e.streams[0]);
    };
    pc.onconnectionstatechange = () => {
      if (!pc) return;
      if (pc.connectionState === "connected") {
        stopRingtone();
        show("call");
        startTimer();
        applyMicState();  // PTT default: mic gated closed until held
        if (sendStream) watchMicAudio(sendStream);  // post-gain: bars follow the gate
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
  stopRingtone();
  clearInterval(talkingHandle);
  talkingHandle = null;
  clearInterval(micVizHandle);
  micVizHandle = null;
  if (pc) { try { pc.close(); } catch (_) {} pc = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  sendStream = null;
  micGain = null;
  pttLive = false;  // mode itself persists across calls
}

function endCall(title, detail) {
  teardown();
  $("ended-title").textContent = title || "Call ended";
  $("ended-detail").textContent = detail || "";
  show("ended");
}

// ---------- mic gating: push-to-talk (default) / hands-free ----------
function applyMicState() {
  const live = pttMode ? pttLive : true;
  if (micGain) {
    // quick fade avoids a click; the stream keeps flowing (silence) so the
    // server-side receiver never idles out
    micGain.gain.setTargetAtTime(live ? 1 : 0, actx().currentTime, 0.02);
  }
  $("btn-ptt").classList.toggle("hidden", !pttMode);
  $("btn-ptt").classList.toggle("live", pttMode && pttLive);
  $("ic-mic").style.display = pttMode ? "none" : "";       // hands-free: open mic icon
  $("ic-mic-off").style.display = pttMode ? "" : "none";   // PTT: mic gated icon
  $("mode-hint").textContent = pttMode ? "hands-free" : "push to talk";
  $("btn-mode").classList.toggle("engaged", !pttMode);
}

const _ptt = $("btn-ptt");
_ptt.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  try { _ptt.setPointerCapture(e.pointerId); } catch (_) {}
  pttLive = true;
  applyMicState();
});
for (const ev of ["pointerup", "pointercancel"]) {
  _ptt.addEventListener(ev, () => { pttLive = false; applyMicState(); });
}
_ptt.addEventListener("contextmenu", (e) => e.preventDefault());  // mobile long-press

$("btn-mode").onclick = () => {
  pttMode = !pttMode;
  pttLive = false;
  applyMicState();
};

// ---------- wire up ----------
$("btn-call").onclick = startCall;
$("btn-again").onclick = startCall;
$("btn-cancel").onclick = () => endCall("Call cancelled", "");
$("btn-hangup").onclick = () => endCall("Call ended", "");
