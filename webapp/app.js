/* callpc call UI — device-aware landing + raw WebRTC against /api/offer.
 *
 * The landing is a single "setup" screen that adapts to context:
 *   - Is a computer connected & callable?  -> authoritative: GET /api/devices.
 *     This (and only this) decides whether the Call button appears.
 *   - Am I open ON that computer?          -> best-effort: a loopback probe of
 *     the executor's beacon (http://127.0.0.1:8724/whoami). Used only to refine
 *     wording ("This computer" vs "<name> — call it"). Gates nothing, so a
 *     blocked/failed probe is harmless.
 * No computer connected -> hide calling, show download + pairing instructions.
 *
 * API base resolution (first hit wins):
 *   1. ?api=https://xyz.ngrok.app   (static-host deployment pointing at ngrok)
 *   2. window.CALLPC_API_BASE       (optional config.js on a static host)
 *   3. same origin                  (the gateway serves this app itself)
 */
"use strict";

const API_BASE = (() => {
  const q = new URLSearchParams(location.search).get("api");
  const base = q || window.CALLPC_API_BASE || "";
  return base.replace(/\/+$/, "");
})();

// the executor's loopback presence beacon (override with ?localport=)
const LOCAL_PORT = new URLSearchParams(location.search).get("localport") || "8724";
const REPO = "https://github.com/pandevim/callpc";
// WSS control URL the executor connects to (http(s)->ws(s))
const WS_URL = (API_BASE || location.origin).replace(/^http/, "ws") + "/control/ws";

const $ = (id) => document.getElementById(id);

// ---------- passcode (shared access gate; used for both pairing and calling) ----------
const getStoredCode = () => (localStorage.getItem("callpc_code") || "").trim();
const setCode = (c) => localStorage.setItem("callpc_code", (c || "").trim());
const codeHeaders = () => {
  const c = ($("passcode")?.value || getStoredCode()).trim();
  return c ? { "X-Callpc-Code": c } : {};
};
let authRequired = false;

const screens = {
  setup: $("screen-setup"),
  chat: $("screen-chat"),
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
let handsFree = false;  // mic locked open (quick tap on the pill toggles it)
let pttLive = false;    // true while the pill is held (push-to-talk)
let timerStart = 0;
let timerHandle = null;
let talkingHandle = null;
let micVizHandle = null;
let ringHandle = null;
let ringGain = null;    // master gain for ringback; zeroed to stop instantly
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

// ====================================================================
//  Device-aware landing
// ====================================================================
const OSES = {
  windows: { label: "Windows", asset: "callpc-executor-windows-x64.exe",
    run: (exe) => `${exe} --url ${WS_URL} --code CODE`,
    note: "Double-click won't pass the code — run it from PowerShell. SmartScreen may warn (unsigned): More info → Run anyway." },
  macos: { label: "macOS", asset: "callpc-executor-macos-universal",
    run: (exe) => `chmod +x ${exe} && ${exe} --url ${WS_URL} --code CODE`,
    note: "First run: right-click the file → Open (unsigned). Approve Screen Recording + Accessibility when asked." },
  linux: { label: "Linux", asset: "callpc-executor-linux-x64",
    run: (exe) => `chmod +x ${exe} && ./${exe} --url ${WS_URL} --code CODE`,
    note: "Wayland (GNOME/KDE): approve the screen-share prompt. Input needs the 'input' group / uinput access." },
};

let currentOS = detectOS();   // for the install instructions
let pairCode = "······";
// user tapped "set up a different computer" (or arrived via connect.html?setup=1)
let forceInstall = new URLSearchParams(location.search).get("setup") === "1";
let ctx = { local: null, online: [] };  // last detectContext() result

function detectOS() {
  const p = (navigator.userAgentData?.platform || navigator.platform || navigator.userAgent).toLowerCase();
  if (p.includes("win")) return "windows";
  if (p.includes("mac") || p.includes("iphone") || p.includes("ipad")) return "macos";
  return "linux";
}

const isMobile = () => /android|iphone|ipad|ipod|mobile/i.test(navigator.userAgent);

// loopback probe: are we open ON the controlled computer? returns whoami | null
async function probeLocal() {
  try {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), 1000);
    const r = await fetch(`http://127.0.0.1:${LOCAL_PORT}/whoami`, { signal: ac.signal });
    clearTimeout(t);
    if (!r.ok) return null;
    const j = await r.json();
    return j && j.callpc ? j : null;
  } catch (_) { return null; }   // refused/blocked/timeout -> not this machine
}

async function fetchOnline() {
  try {
    const j = await (await fetch(API_BASE + "/api/devices")).json();
    return j.online || [];
  } catch (_) { return []; }
}

async function detectContext() {
  const [local, online] = await Promise.all([probeLocal(), fetchOnline()]);
  ctx = { local, online };
  if (screens.chat.classList.contains("active")) updateChatHead();  // live presence
  else if (screens.setup.classList.contains("active")) renderSetup();
  maybeAutoConfigure();   // if a fresh local agent is waiting, pair it automatically
}

// A local agent is running but not paired yet -> hand it the gateway URL + a
// fresh pairing code so it connects itself ("download, run, done"). Loopback +
// the agent only honours this while unpaired, so it can't be hijacked later.
let autoTried = false;
async function maybeAutoConfigure() {
  const l = ctx.local;
  // a local agent with no gateway target yet (fresh, or upgraded from a build
  // that didn't persist the URL) — hand it one. `configured` covers both the
  // unpaired and the paired-but-no-URL cases.
  if (!l || l.online || l.configured) { autoTried = false; return; }
  if (autoTried) return;                                          // already sent this round
  try {
    const r = await fetch(API_BASE + "/api/pair/new", { method: "POST", headers: codeHeaders() });
    if (r.status === 401) {                 // need a passcode first
      $("setup-sub").textContent = "Enter the passcode above to finish setup — "
        + "it's the PASSCODE line printed in the server's startup log.";
      $("passcode").style.display = "";
      return;                               // retry on the next tick once it's typed
    }
    const j = await r.json();
    await fetch(`http://127.0.0.1:${LOCAL_PORT}/configure`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: WS_URL, code: j.code }),
    });
    autoTried = true;   // sent; the agent will connect & pair within a couple seconds
  } catch (_) { /* transient (e.g. probe blocked) — retry next tick */ }
}

function osLabel(d) {
  // prefer the human distro string ("Ubuntu 24.04"), else the os family
  if (!d) return "";
  return d.distro || ({ linux: "Linux", macos: "macOS", windows: "Windows" }[d.os] || d.os || "");
}

function renderSetup() {
  const connected = ctx.online.length > 0;
  const localPresent = !!ctx.local;                   // an agent is running on this machine

  // --- choose what to show ---
  // CALL      : a computer is connected (the only thing that enables calling).
  // CONNECTING: an agent is running here but not online yet (pairing / reconnecting).
  // INSTALL   : no agent here and none connected, or the user asked for a new one.
  const connecting = !connected && localPresent;
  const installView = forceInstall || (!connected && !localPresent);

  // connected and not asked for the install view -> the chat IS the home screen
  // (unless the chat just rejected our passcode — stay parked on the prompt)
  if (!installView && connected) {
    if (!chatAuthFailed) enterChat();
    return;
  }

  $("setup-install").style.display = installView ? "" : "none";

  if (installView) {
    $("setup-emoji").textContent = "⬇️";
    $("setup-title").textContent = forceInstall ? "Set up another computer" : "Set up your computer";
    $("setup-sub").textContent = forceInstall
      ? "Download and run the agent on the computer you want to control."
      : (isMobile()
          ? "Open this page on the computer you want to control, or set it up here."
          : "Install the tiny agent so this computer can be controlled by phone.");
    renderInstall();
    // footer: a way back when something is already connected
    toggleLink(connected ? "← Back" : "", () => { forceInstall = false; renderSetup(); });
  } else if (connecting) {
    $("setup-emoji").textContent = "🔌";
    $("setup-title").textContent = ctx.local.name || "This computer";
    $("setup-sub").textContent = ctx.local.paired
      ? "Agent is running — reconnecting…"
      : "Agent detected — connecting it…";
    toggleLink("Set up a different computer →", () => { forceInstall = true; renderSetup(); });
  }
}

function toggleLink(text, onclick) {
  const el = $("setup-toggle");
  if (!text) { el.style.display = "none"; return; }
  el.style.display = "";
  el.textContent = text;
  el.onclick = (e) => { e.preventDefault(); onclick(); };
}

// ---------- install panel (OS tabs, download, pairing code, run command) ----------
function renderInstall() {
  $("os-tabs").innerHTML = "";
  for (const [key, os] of Object.entries(OSES)) {
    const b = document.createElement("button");
    b.textContent = os.label;
    b.className = key === currentOS ? "active" : "";
    b.onclick = () => { currentOS = key; renderInstall(); };
    $("os-tabs").appendChild(b);
  }
  const os = OSES[currentOS];
  $("dl").href = `${REPO}/releases/latest/download/${os.asset}`;
  $("dl").textContent = `Download for ${os.label}`;
  $("dl-note").textContent = "Single file, no installer. No releases yet? Build from source: " +
    "git clone " + REPO + " && cargo build --release --manifest-path executor/Cargo.toml";
  $("run-note").textContent = os.note;
  renderCmd();
  if (pairCode === "······") getPairCode();   // fetch a code the first time we show install
}

function renderCmd() {
  const os = OSES[currentOS];
  const exe = (currentOS === "windows" ? ".\\" : "./") + os.asset.split("/").pop();
  $("cmd-text").textContent = os.run(exe).replace("CODE", pairCode);
}

async function getPairCode() {
  try {
    const r = await fetch(API_BASE + "/api/pair/new", { method: "POST", headers: codeHeaders() });
    if (r.status === 401) { $("code").textContent = "passcode?"; $("ttl").textContent = "—"; return; }
    const j = await r.json();
    pairCode = j.code;
    $("code").textContent = pairCode;
    renderCmd();
    startTtl(j.expires_in || 600);
  } catch (_) { $("code").textContent = "error"; }
}

let ttlH = null;
function startTtl(secs) {
  clearInterval(ttlH);
  const end = Date.now() + secs * 1000;
  const tick = () => {
    const left = Math.max(0, Math.round((end - Date.now()) / 1000));
    $("ttl").textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
    if (left <= 0) { clearInterval(ttlH); getPairCode(); }  // auto-refresh expired code
  };
  tick(); ttlH = setInterval(tick, 1000);
}

document.querySelectorAll(".copy").forEach((b) => b.onclick = () => {
  navigator.clipboard.writeText($("cmd-text").textContent);
  b.textContent = "copied"; setTimeout(() => b.textContent = "copy", 1200);
});

// re-detect while we're on the setup screen, so it flips to "connected" on its own
let detectTimer = null;
function startDetectLoop() {
  detectContext();
  clearInterval(detectTimer);
  detectTimer = setInterval(() => {
    if (screens.setup.classList.contains("active") ||
        screens.chat.classList.contains("active")) detectContext();
  }, 2500);
}

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
  if (!ringGain) return;
  const c = actx();
  const t = c.currentTime;
  const g = c.createGain();
  g.connect(ringGain);
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
  try {
    const c = actx();
    ringGain = c.createGain();
    ringGain.connect(c.destination);
    ringOnce();
    ringHandle = setInterval(ringOnce, 4200);
  } catch (_) {}
}
function stopRingtone() {
  clearInterval(ringHandle);
  ringHandle = null;
  if (ringGain) {
    try {
      const now = actx().currentTime;
      ringGain.gain.cancelScheduledValues(now);
      ringGain.gain.setValueAtTime(0, now);  // kill any in-flight tone immediately
      ringGain.disconnect();
    } catch (_) {}
    ringGain = null;
  }
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

// ---------- ICE config + auth probe ----------
async function iceServers() {
  try {
    const r = await fetch(API_BASE + "/api/config");
    if (r.ok) {
      const cfg = await r.json();
      return cfg.ice_servers;
    }
  } catch (_) { /* fall through */ }
  return [{ urls: "stun:stun.l.google.com:19302" }];
}

async function probeAuth() {
  try {
    const cfg = await (await fetch(API_BASE + "/api/config")).json();
    authRequired = !!cfg.auth_required;
  } catch (_) { authRequired = false; }
  if (authRequired) {
    const f = $("passcode");
    f.style.display = "";
    f.value = getStoredCode();
    f.addEventListener("input", () => {
      setCode(f.value);
      chatAuthFailed = false;  // let the detect loop retry the chat with the new code
      // a passcode is needed to mint a pairing code — retry once typed
      clearTimeout(window._pcDeb);
      window._pcDeb = setTimeout(() => { if ($("setup-install").style.display !== "none") getPairCode(); }, 500);
    });
  }
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
    if (authRequired) setCode($("passcode").value);
    const r = await fetch(API_BASE + "/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...codeHeaders() },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (r.status === 401) return endCall("Wrong passcode", "Check the passcode and try again.");
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

// back to the device-aware landing (re-detects connection state)
function backToSetup() {
  teardown();
  show("setup");
  startDetectLoop();
}

// ---------- mic gating: ONE pill — hold to talk, quick tap locks hands-free ----------
function applyMicState() {
  const live = handsFree || pttLive;
  if (micGain) {
    // quick fade avoids a click; the stream keeps flowing (silence) so the
    // server-side receiver never idles out
    micGain.gain.setTargetAtTime(live ? 1 : 0, actx().currentTime, 0.02);
  }
  const b = $("btn-ptt");
  b.classList.toggle("live", live);
  b.textContent = handsFree ? "🎙 mic open — tap to mute"
                : pttLive   ? "talking…"
                            : "hold to talk · tap for hands-free";
}

const _ptt = $("btn-ptt");
let pttDownAt = 0;
_ptt.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  try { _ptt.setPointerCapture(e.pointerId); } catch (_) {}
  pttDownAt = Date.now();
  if (!handsFree) { pttLive = true; applyMicState(); }
});
for (const ev of ["pointerup", "pointercancel"]) {
  _ptt.addEventListener(ev, () => {
    const tap = Date.now() - pttDownAt < 250;   // quick tap = toggle hands-free
    pttLive = false;
    if (tap) handsFree = !handsFree;
    applyMicState();
  });
}
_ptt.addEventListener("contextmenu", (e) => e.preventDefault());  // mobile long-press

// ====================================================================
//  Chat — WhatsApp-style messaging with the computer (/api/chat/ws)
// ====================================================================
const CHAT_WS_URL = (API_BASE || location.origin).replace(/^http/, "ws") + "/api/chat/ws";
const chatLog = $("chat-log");
let chatWS = null, chatRetry = null;
let chatAuthFailed = false;  // last connect was rejected (4401) — stop the
                             // setup->chat redirect loop until the code changes
let statusBub = null;        // the single in-place progress bubble
let pendingVoice = null;     // my just-sent voice bubble, awaiting its transcript
let localStatus = "";        // client-only status (e.g. quick screenshot fetch)

function enterChat() {
  if (!screens.chat.classList.contains("active")) show("chat");
  updateChatHead();
  connectChat();
}

function updateChatHead() {
  const dev = ctx.online[0] || null;
  const onThis = !!(ctx.local && ctx.local.online);
  $("chat-name").textContent =
    onThis ? "This computer" : (dev?.name || ctx.local?.name || "My Computer");
  const distro = osLabel(onThis ? ctx.local : dev);
  $("chat-presence").textContent =
    dev ? "online" + (distro ? " · " + distro : "") : "offline — waiting for the computer…";
  $("chat-presence").classList.toggle("off", !dev);
}

function connectChat() {
  if (chatWS && chatWS.readyState <= 1) return;   // already open/connecting
  clearTimeout(chatRetry);
  const code = getStoredCode();
  chatWS = new WebSocket(CHAT_WS_URL + (code ? "?code=" + encodeURIComponent(code) : ""));
  chatWS.onopen = () => { chatAuthFailed = false; };
  chatWS.onmessage = (e) => { try { handleChat(JSON.parse(e.data)); } catch (_) {} };
  chatWS.onclose = (e) => {
    chatWS = null;
    if (e.code === 4401) { needPasscode(); return; }
    if (screens.chat.classList.contains("active")) chatRetry = setTimeout(connectChat, 2000);
  };
}

// wrong/missing passcode: park on the setup screen until a new code is typed
// (the passcode input listener clears chatAuthFailed; the detect loop then
// re-enters the chat, which reconnects with the new code)
function needPasscode() {
  chatAuthFailed = true;
  show("setup");
  $("setup-install").style.display = "none";
  $("setup-emoji").textContent = "🔒";
  $("setup-title").textContent = "Passcode needed";
  $("setup-sub").textContent = "It's printed when the server starts, next to "
    + "this page's URL (the PASSCODE line in the job log).";
  const f = $("passcode");
  f.style.display = "";
  f.focus();
}

function handleChat(m) {
  if (m.type === "history") {
    chatLog.querySelectorAll(".bub").forEach((el) => el.remove());
    statusBub = null;
    pendingVoice = null;
    for (const it of m.items || []) renderItem(it);
    setStatusBubble(m.status || "");
    scrollChat(true);
    return;
  }
  if (m.type === "status") { localStatus = ""; setStatusBubble(m.text || ""); return; }
  if (m.type === "error") { addInfo("⚠️ " + m.text); return; }
  if (m.type === "image" && localStatus) { localStatus = ""; setStatusBubble(""); }
  renderItem(m);
}

function renderItem(it) {
  $("chat-empty").style.display = "none";
  if (it.type === "msg") addMsg(it.role === "user" ? "user" : "bot", it.text, it.ts);
  else if (it.type === "image") addImage(it);
  else if (it.type === "transcript") addTranscript(it.text, it.ts);
}

// ---------- bubbles ----------
function bub(cls) {
  $("chat-empty").style.display = "none";
  const d = document.createElement("div");
  d.className = "bub " + cls;
  chatLog.appendChild(d);
  if (statusBub) chatLog.appendChild(statusBub);  // progress stays visually last
  return d;
}

function timeMeta(ts) {
  const d = ts ? new Date(ts * 1000) : new Date();
  const s = document.createElement("span");
  s.className = "meta";
  s.textContent = String(d.getHours()).padStart(2, "0") + ":" +
                  String(d.getMinutes()).padStart(2, "0");
  return s;
}

function addMsg(side, text, ts) {
  const d = bub(side);
  const t = document.createElement("span");
  t.className = "txt";
  t.textContent = text;
  d.append(t, timeMeta(ts));
  scrollChat();
  return d;
}

function addInfo(text) {
  const d = bub("info");
  d.textContent = text;
  scrollChat();
}

function addImage(it) {
  const d = bub("bot img");
  if (it.b64) {
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + it.b64;
    img.alt = it.caption || "screenshot";
    img.onclick = () => {                          // tap to view full-screen
      const ov = document.createElement("div");
      ov.className = "lightbox";
      const big = document.createElement("img");
      big.src = img.src;
      ov.appendChild(big);
      ov.onclick = () => ov.remove();
      document.body.appendChild(ov);
    };
    d.appendChild(img);
  } else {                                         // pixels pruned server-side
    d.classList.add("expired");
    const t = document.createElement("span");
    t.className = "txt";
    t.textContent = "📷 screenshot (no longer stored)";
    d.appendChild(t);
  }
  if (it.caption) {
    const c = document.createElement("span");
    c.className = "cap";
    c.textContent = it.caption;
    d.appendChild(c);
  }
  d.appendChild(timeMeta(it.ts));
  scrollChat();
}

// a voice note's transcription: caption my own pending bubble, or (other
// device / history replay) render it as a mic-flagged user message
function addTranscript(text, ts) {
  if (pendingVoice) {
    pendingVoice.querySelector(".cap").textContent = "“" + text + "”";
    pendingVoice = null;
    return;
  }
  const d = bub("user");
  const t = document.createElement("span");
  t.className = "txt";
  t.textContent = "🎤 “" + text + "”";
  d.append(t, timeMeta(ts));
  scrollChat();
}

function setStatusBubble(text) {
  if (!text) {
    if (statusBub) statusBub.remove();
    statusBub = null;
    return;
  }
  if (!statusBub) {
    statusBub = document.createElement("div");
    statusBub.className = "bub bot status";
    statusBub.innerHTML =
      '<span class="dots"><i></i><i></i><i></i></span><span class="stxt"></span>';
  }
  statusBub.querySelector(".stxt").textContent = text;
  chatLog.appendChild(statusBub);
  scrollChat();
}

function scrollChat(force) {
  const near = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 160;
  if (force || near) chatLog.scrollTop = chatLog.scrollHeight;
}

// ---------- composer ----------
function chatReady() {
  if (chatWS && chatWS.readyState === 1) return true;
  addInfo("⚠️ connecting to the chat — try again in a second");
  connectChat();
  return false;
}

function sendChatText() {
  const f = $("chat-input");
  const text = f.value.trim();
  if (!text || !chatReady()) return;
  addMsg("user", text);                            // local echo (server excludes me)
  chatWS.send(JSON.stringify({ type: "text", text }));
  f.value = "";
  updateComposer();
}

function updateComposer() {
  const has = $("chat-input").value.trim().length > 0;
  $("btn-send").style.display = has ? "" : "none";
  $("btn-rec").style.display = has ? "none" : "";
}

// ---------- voice notes (tap to record, tap ➤ to send, ✕ cancels) ----------
let rec = null, recChunks = [], recStartTs = 0, recTick = null,
    recStream = null, recCancelled = false, recStarting = false;

// instant feedback the moment the mic is tapped — BEFORE the permission
// prompt resolves (the old hold-gesture showed nothing until then)
function setRecUI(on) {
  $("btn-rec").classList.toggle("rec", on);
  $("ic-rec-mic").style.display = on ? "none" : "";
  $("ic-rec-send").style.display = on ? "" : "none";
  $("rec-bar").style.display = on ? "" : "none";
  if (!on) $("rec-time").textContent = "0:00";
}

async function startRec() {
  if (rec || recStarting) return;
  recStarting = true;
  recCancelled = false;
  setRecUI(true);
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (_) {
    recStarting = false;
    setRecUI(false);
    addInfo("⚠️ microphone blocked — allow mic access to send voice notes");
    return;
  }
  if (recCancelled) {  // ✕ tapped while the permission prompt was up
    recStream.getTracks().forEach((t) => t.stop());
    recStream = null;
    recStarting = false;
    return;
  }
  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
    .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
  rec = new MediaRecorder(recStream, mime ? { mimeType: mime } : undefined);
  recChunks = [];
  rec.ondataavailable = (ev) => { if (ev.data.size) recChunks.push(ev.data); };
  rec.onstop = finishRec;
  rec.start();
  recStarting = false;
  recStartTs = Date.now();
  recTick = setInterval(() => {
    const s = Math.floor((Date.now() - recStartTs) / 1000);
    $("rec-time").textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
    if (s >= 60) stopRec(false);                   // cap notes at a minute
  }, 250);
}

function stopRec(cancel) {
  if (recStarting && !rec) {   // still waiting on the permission prompt
    recCancelled = true;       // nothing recorded yet either way — discard
    setRecUI(false);
    return;
  }
  if (!rec) return;
  recCancelled = !!cancel;
  clearInterval(recTick);
  setRecUI(false);
  try { rec.stop(); } catch (_) {}                 // -> finishRec
}

function finishRec() {
  const dur = Date.now() - recStartTs;
  const blob = new Blob(recChunks, { type: (rec && rec.mimeType) || "audio/webm" });
  if (recStream) recStream.getTracks().forEach((t) => t.stop());
  rec = null;
  recStream = null;
  if (recCancelled) return;
  if (dur < 400 || blob.size < 1000) { addInfo("that recording was too short"); return; }
  if (!chatReady()) return;
  const d = bub("user voice");                     // local bubble with playback
  const au = document.createElement("audio");
  au.controls = true;
  au.src = URL.createObjectURL(blob);
  const cap = document.createElement("span");
  cap.className = "cap";
  cap.textContent = "transcribing…";
  d.append(au, cap, timeMeta());
  pendingVoice = d;
  scrollChat(true);
  const r = new FileReader();
  r.onload = () => {
    const b64 = String(r.result).split(",", 2)[1] || "";
    chatWS.send(JSON.stringify({ type: "voice", b64, mime: blob.type }));
  };
  r.readAsDataURL(blob);
}

$("btn-rec").onclick = () => ((rec || recStarting) ? stopRec(false) : startRec());
$("rec-cancel").onclick = () => stopRec(true);

// ---------- wire up ----------
$("btn-chat-call").onclick = startCall;            // the call lives top-right now
$("btn-chat-menu").onclick = () => { forceInstall = true; show("setup"); renderSetup(); };
$("btn-send").onclick = sendChatText;
$("chat-input").addEventListener("input", updateComposer);
$("chat-input").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChatText(); });
$("btn-shot").onclick = () => {                    // quick action: no LLM round-trip
  if (!chatReady()) return;
  chatWS.send(JSON.stringify({ type: "screenshot" }));
  localStatus = "fetching a screenshot…";
  setStatusBubble(localStatus);
};

$("btn-again").onclick = startCall;
$("btn-ended-chat").onclick = backToSetup;  // re-detects -> lands back in the chat
$("btn-cancel").onclick = backToSetup;
$("btn-hangup").onclick = backToSetup;

probeAuth();        // reveal the passcode field if the gateway requires one
startDetectLoop();  // device-aware landing
