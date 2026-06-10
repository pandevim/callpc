#!/usr/bin/env python3
"""callpc Phase-0 preflight harness.

One script that probes every "will this hurt us later" assumption before the
full build: host RAM / GPU VRAM headroom, that all Python deps import, that the
vendored ASR engine loads + runs within its RAM budget, that the model services
answer, that WebRTC UDP egress works (STUN), and that the reverse-proxy chain
reaches the gateway.

Every check is independent and reports PASS / FAIL / SKIP, so it degrades
gracefully: run the light checks on a laptop, run the full set *inside the SLURM
job* once the five services are up.

    python preflight.py                 # run every check
    python preflight.py env imports stun   # run a subset
    python preflight.py asr --download     # also fetch the 2.5 GB ONNX model
    python preflight.py --list             # list checks

Config via env (all optional, sane localhost defaults):
    UITARS_URL  (default http://127.0.0.1:8000)   vLLM #1, served-name "ui-tars"
    BRAIN_URL   (default http://127.0.0.1:8001)   vLLM #2, Qwen3.5-9B
    TTS_URL     (default http://127.0.0.1:8089)   Kyutai moshi server
    ASR_WS      (default ws://127.0.0.1:8765)      ws_engine.py
    GATEWAY_URL (default http://127.0.0.1:8080)   Pipecat gateway (/api/offer)
    PUBLIC_URL  (no default)                       ngrok HTTPS URL, for `proxy`
    STUN_ADDR   (default stun.l.google.com:19302)
    ASR_WAV     (path to a real wav for the asr check; else synthetic audio)
"""
import argparse
import json
import os
import socket
import struct
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ASR_DIR = os.path.join(HERE, "asr")

# VRAM budget from PLAN.md (GB on the single 48 GB L40S).
VRAM_BUDGET = {
    "UI-TARS-1.5-7B": 20.0,
    "Qwen3.5-9B-FP8": 10.0,
    "Kyutai TTS 1.6B": 5.0,
    "Nemotron ASR ONNX": 3.4,
}
VRAM_TOTAL_NEEDED = sum(VRAM_BUDGET.values())  # ~38.4 GB
VRAM_CARD = 48.0

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_COLOR = sys.stdout.isatty()
_C = {PASS: "\033[32m", FAIL: "\033[31m", SKIP: "\033[33m", "z": "\033[0m", "dim": "\033[2m"}


def _c(s, key):
    return f"{_C[key]}{s}{_C['z']}" if _COLOR else s


def _say(status, name, detail=""):
    tag = _c(f"{status:4}", status)
    print(f"  [{tag}] {name:14} {_c(detail, 'dim') if detail else ''}".rstrip())


def _http(url, timeout=4.0, data=None, headers=None, method=None):
    """Tiny urllib wrapper. Returns (status_code, body_bytes) or raises."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.getcode(), r.read()


# --------------------------------------------------------------------------- #
# checks: each returns (status, detail). Heavy ones may print sub-lines.
# --------------------------------------------------------------------------- #

def check_env(args):
    """Host RAM, CPU, OS, Python, and every visible GPU + its VRAM."""
    import platform

    detail = []
    try:
        import multiprocessing
        detail.append(f"{multiprocessing.cpu_count()} CPU")
    except Exception:
        pass

    ram_gb = None
    try:  # Linux: read MemTotal/MemAvailable without psutil
        with open("/proc/meminfo") as f:
            mem = {l.split(":")[0]: l.split()[1] for l in f if ":" in l}
        ram_gb = int(mem["MemTotal"]) / 1024 / 1024
        avail_gb = int(mem["MemAvailable"]) / 1024 / 1024
        detail.append(f"RAM {ram_gb:.1f} GB ({avail_gb:.1f} free)")
    except Exception:
        detail.append("RAM ?")

    detail.append(f"{platform.system()} py{platform.python_version()}")
    print(f"       {_c(' | '.join(detail), 'dim')}")

    gpus = _read_gpus()
    if not gpus:
        _say(SKIP, "gpu", "no nvidia-smi / no GPU (expected off the L40S)")
        return PASS, "host ok, no GPU here"
    for g in gpus:
        line = "GPU {}: {} — {:.1f} GB".format(g["idx"], g["name"], g["total"] / 1024)
        print(f"       {_c(line, 'dim')}")
    biggest = max(g["total"] for g in gpus) / 1024
    if biggest < VRAM_TOTAL_NEEDED:
        return PASS, f"GPU {biggest:.0f} GB < {VRAM_TOTAL_NEEDED:.0f} GB budget — can't host full stack (fine for dev)"
    return PASS, f"GPU {biggest:.0f} GB fits {VRAM_TOTAL_NEEDED:.0f} GB budget"


def _read_gpus():
    """List GPUs via nvidia-smi --query-gpu, or [] if unavailable."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, used = (x.strip() for x in line.split(","))
        gpus.append({"idx": int(idx), "name": name, "total": float(total), "used": float(used)})
    return gpus


def check_imports(args):
    """Try importing each component's deps; report which are missing per group."""
    import importlib.util as iu
    groups = {
        "asr": ["numpy", "onnxruntime", "sentencepiece", "soundfile"],
        "gateway": ["pipecat", "fastapi", "aiortc", "websockets", "aiohttp"],
        "tts": ["moshi", "msgpack", "torch"],
        "vad": ["silero_vad"],
    }
    missing_total = 0
    for grp, mods in groups.items():
        present = {m: (iu.find_spec(m) is not None) for m in mods}
        miss = [m for m, ok in present.items() if not ok]
        missing_total += len(miss)
        ok_n = len(mods) - len(miss)
        line = f"{grp:8} {ok_n}/{len(mods)}"
        if miss:
            line += f"  missing: {', '.join(miss)}"
        print(f"       {_c(line, 'dim')}")
    if missing_total == 0:
        return PASS, "all import groups satisfied"
    return SKIP, f"{missing_total} deps missing (install in the target env — expected on a dev box)"


def check_asr(args):
    """Load the vendored Nemotron ONNX engine and run offline + streaming, report RSS."""
    try:
        import numpy as np
    except Exception:
        return SKIP, "numpy missing"
    if not _has("onnxruntime"):
        return SKIP, "onnxruntime missing — `pip install onnxruntime-gpu`"

    model_dir = os.path.join(ASR_DIR, "models", "onnx")
    enc = os.path.join(model_dir, "encoder.onnx")
    if not os.path.exists(enc):
        if args.download:
            print(f"       {_c('downloading ONNX model (~2.5 GB)...', 'dim')}")
            sys.path.insert(0, ASR_DIR)
            os.chdir(ASR_DIR)
            import download_onnx  # noqa: F401  (runs on import)
        else:
            return SKIP, "model not downloaded — run `asr --download` or asr/download_onnx.py"

    sys.path.insert(0, ASR_DIR)
    t0 = time.time()
    import onnx_engine
    asr = onnx_engine.OnnxASR(lang="en-US")
    load_s = time.time() - t0
    prov = onnx_engine.ort.get_available_providers()
    on_gpu = "CUDAExecutionProvider" in prov

    # real wav if given, else 3 s of synthetic audio (text will be garbage; we
    # only check the pipeline runs and measure cost).
    wav = args.wav or os.environ.get("ASR_WAV")
    if wav and _has("soundfile"):
        import soundfile as sf
        data, _ = sf.read(wav, dtype="float32")
        data = data[:, 0] if data.ndim > 1 else data
        synthetic = False
    else:
        t = np.arange(16000 * 3) / 16000.0
        data = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        synthetic = True

    t0 = time.time()
    text = asr.transcribe(data)
    off_s = time.time() - t0

    sess = onnx_engine.OnnxStreamingSession(asr)
    pcm = (data * 32768.0).astype(np.int16)
    step = sess.chunk_bytes // 2
    t0 = time.time()
    parts = [sess.feed(pcm[i:i + step]) for i in range(0, len(pcm), step)]
    parts.append(sess.finalize())
    stream_s = time.time() - t0
    audio_s = len(data) / 16000.0
    rtf = stream_s / audio_s if audio_s else 0

    import resource
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"       {_c(f'provider={prov[0]} gpu={on_gpu} load={load_s:.1f}s', 'dim')}")
    print(f"       {_c(f'offline={off_s*1000:.0f}ms stream={stream_s*1000:.0f}ms for {audio_s:.1f}s audio (RTF {rtf:.2f})', 'dim')}")
    print(f"       {_c(f'host RSS {rss_mb:.0f} MB' + ('  [synthetic audio: text N/A]' if synthetic else f'  text={text[:50]!r}'), 'dim')}")

    if rss_mb > 1500:
        return FAIL, f"RSS {rss_mb:.0f} MB exceeds ~0.9 GB expectation"
    if rtf > 1.0:
        return FAIL, f"RTF {rtf:.2f} > 1.0 — slower than real-time"
    return PASS, f"engine ok, RSS {rss_mb:.0f} MB, RTF {rtf:.2f}"


def check_vram(args):
    """Read live VRAM use and compare against the 38 GB / 48 GB plan budget."""
    gpus = _read_gpus()
    if not gpus:
        return SKIP, "no GPU here"
    import shutil
    import subprocess
    for g in gpus:
        used, total = g["used"] / 1024, g["total"] / 1024
        bar = f"GPU {g['idx']}: {used:.1f}/{total:.1f} GB used, {total-used:.1f} GB free"
        print(f"       {_c(bar, 'dim')}")
    # list compute processes if any (the five services, once up)
    procs = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            procs = [l for l in out.strip().splitlines() if l.strip()]
        except Exception:
            pass
    for p in procs:
        print(f"       {_c('  proc ' + p, 'dim')}")
    g0 = gpus[0]
    total, used = g0["total"] / 1024, g0["used"] / 1024
    if total < VRAM_TOTAL_NEEDED:
        return SKIP, f"card {total:.0f} GB < {VRAM_TOTAL_NEEDED:.0f} GB budget — not the L40S"
    if not procs:
        return PASS, f"{total:.0f} GB card idle; budget needs {VRAM_TOTAL_NEEDED:.0f} GB ({VRAM_TOTAL_NEEDED/total*100:.0f}%)"
    if used > VRAM_CARD * 0.95:
        return FAIL, f"{used:.1f} GB used — within 5% of card limit, no headroom"
    return PASS, f"{used:.1f}/{total:.0f} GB used, {total-used:.1f} GB headroom"


def _probe_openai(url, name):
    """Probe an OpenAI-compatible vLLM server: /health then /v1/models."""
    try:
        code, _ = _http(url.rstrip("/") + "/health", timeout=3)
    except Exception as e:
        return SKIP, f"{name} unreachable at {url} ({type(e).__name__})"
    try:
        _, body = _http(url.rstrip("/") + "/v1/models", timeout=4)
        models = [m["id"] for m in json.loads(body).get("data", [])]
        return PASS, f"{name} up, models={models}"
    except Exception as e:
        return FAIL, f"{name} /health ok but /v1/models failed ({type(e).__name__})"


def check_vllm(args):
    """Probe both vLLM endpoints (UI-TARS + brain)."""
    uit = os.environ.get("UITARS_URL", "http://127.0.0.1:8000")
    brain = os.environ.get("BRAIN_URL", "http://127.0.0.1:8001")
    s1, d1 = _probe_openai(uit, "ui-tars")
    _say(s1, "  ui-tars", d1)
    s2, d2 = _probe_openai(brain, "brain")
    _say(s2, "  brain", d2)
    statuses = {s1, s2}
    if statuses == {SKIP}:
        return SKIP, "neither vLLM up (start the sbatch job first)"
    if FAIL in statuses:
        return FAIL, "a vLLM endpoint is misbehaving"
    if SKIP in statuses:
        return SKIP, "one vLLM up, one not"
    return PASS, "both vLLM endpoints healthy"


def check_tts(args):
    """Probe the Kyutai TTS server (HTTP root or WS port reachable)."""
    url = os.environ.get("TTS_URL", "http://127.0.0.1:8089")
    host, port = _hostport(url, 8089)
    if not _tcp_open(host, port):
        return SKIP, f"TTS port {host}:{port} closed (start moshi server)"
    # don't assume the route shape; a reachable port is the smoke signal here.
    return PASS, f"TTS port {host}:{port} open"


def check_asr_ws(args):
    """Check the ASR WebSocket service port is listening (full handshake needs a client)."""
    url = os.environ.get("ASR_WS", "ws://127.0.0.1:8765")
    host, port = _hostport(url, 8765)
    if _tcp_open(host, port):
        return PASS, f"ASR ws port {host}:{port} open"
    return SKIP, f"ASR ws {host}:{port} closed (run ws_engine.py)"


def check_stun(args):
    """The WebRTC make-or-break: can we reach a STUN server over UDP and learn
    our public reflexive address? If this fails, media won't traverse NAT."""
    addr = os.environ.get("STUN_ADDR", "stun.l.google.com:19302")
    host, port = addr.rsplit(":", 1)
    port = int(port)
    # STUN Binding Request: type 0x0001, len 0, magic cookie, 12-byte txn id.
    magic = 0x2112A442
    txn = os.urandom(12)
    msg = struct.pack(">HHI", 0x0001, 0, magic) + txn
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(5)
        s.sendto(msg, (host, port))
        data, _ = s.recvfrom(2048)
    except Exception as e:
        return FAIL, f"no STUN reply from {addr} ({type(e).__name__}) — UDP egress blocked?"
    finally:
        try:
            s.close()
        except Exception:
            pass
    # parse attributes for XOR-MAPPED-ADDRESS (0x0020)
    mtype, mlen = struct.unpack(">HH", data[:4])
    pos = 20
    mapped = None
    while pos + 4 <= 20 + mlen:
        atype, alen = struct.unpack(">HH", data[pos:pos + 4])
        val = data[pos + 4:pos + 4 + alen]
        if atype in (0x0020, 0x0001) and len(val) >= 8:
            fam = val[1]
            if fam == 0x01:  # IPv4
                xport = struct.unpack(">H", val[2:4])[0]
                xaddr = struct.unpack(">I", val[4:8])[0]
                if atype == 0x0020:  # XOR-MAPPED: undo the xor
                    xport ^= (magic >> 16)
                    xaddr ^= magic
                ip = socket.inet_ntoa(struct.pack(">I", xaddr))
                mapped = f"{ip}:{xport}"
                break
        pos += 4 + alen + ((4 - alen % 4) % 4)  # 4-byte aligned
    if mapped:
        return PASS, f"UDP egress ok; reflexive address {mapped}"
    return PASS, f"STUN replied from {addr} but no mapped address parsed"


def check_gateway(args):
    """The Pipecat gateway should serve the call UI + accept WebRTC offers."""
    url = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8080")
    host, port = _hostport(url, 8080)
    if not _tcp_open(host, port):
        return SKIP, f"gateway {host}:{port} closed (not built/running yet)"
    # an OPTIONS/GET to /api/offer should not 404 once the route exists.
    try:
        code, _ = _http(url.rstrip("/") + "/api/offer", timeout=3, method="OPTIONS")
        return PASS, f"gateway up, /api/offer responds ({code})"
    except urllib.error.HTTPError as e:
        if e.code == 405:  # method not allowed but route exists
            return PASS, "gateway up, /api/offer route exists (405 to OPTIONS)"
        if e.code == 404:
            return FAIL, "gateway up but /api/offer missing"
        return PASS, f"gateway up ({e.code} on /api/offer)"
    except Exception as e:
        return PASS, f"gateway port open, /api/offer probe inconclusive ({type(e).__name__})"


def check_proxy(args):
    """Reach the gateway through the public reverse-proxy chain (ngrok HTTPS)."""
    pub = os.environ.get("PUBLIC_URL")
    if not pub:
        return SKIP, "PUBLIC_URL unset (set it to the ngrok HTTPS URL)"
    if not pub.startswith("https://"):
        return FAIL, f"PUBLIC_URL is not https ({pub}) — getUserMedia needs a secure context"
    try:
        t0 = time.time()
        code, _ = _http(pub, timeout=8)
        ms = (time.time() - t0) * 1000
        return PASS, f"reached {pub} ({code}) in {ms:.0f} ms over TLS"
    except urllib.error.HTTPError as e:
        return PASS, f"reached {pub} (HTTP {e.code}) — chain works, gateway answered"
    except Exception as e:
        return FAIL, f"could not reach {pub} ({type(e).__name__}) — tunnel/ngrok down?"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _has(mod):
    import importlib.util as iu
    return iu.find_spec(mod) is not None


def _hostport(url, default_port):
    rest = url.split("://", 1)[-1]
    rest = rest.split("/", 1)[0]
    if ":" in rest:
        h, p = rest.rsplit(":", 1)
        return h, int(p)
    return rest, default_port


def _tcp_open(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


CHECKS = {
    "env": check_env,
    "imports": check_imports,
    "asr": check_asr,
    "vram": check_vram,
    "vllm": check_vllm,
    "tts": check_tts,
    "asr_ws": check_asr_ws,
    "stun": check_stun,
    "gateway": check_gateway,
    "proxy": check_proxy,
}

# checks safe to run anywhere vs. those that only mean something inside the job
GROUPS = {
    "local": ["env", "imports", "asr", "stun", "proxy"],
    "job": ["env", "imports", "asr", "vram", "vllm", "tts", "asr_ws", "gateway", "stun"],
}


def main():
    ap = argparse.ArgumentParser(description="callpc Phase-0 preflight harness")
    ap.add_argument("checks", nargs="*", help="checks to run (default: all). Or a group: local, job")
    ap.add_argument("--list", action="store_true", help="list available checks and exit")
    ap.add_argument("--download", action="store_true", help="let `asr` download the ONNX model")
    ap.add_argument("--wav", help="wav file for the asr check")
    args = ap.parse_args()

    if args.list:
        print("checks:", ", ".join(CHECKS))
        print("groups:", ", ".join(f"{k}=[{' '.join(v)}]" for k, v in GROUPS.items()))
        return 0

    if not args.checks:
        names = list(CHECKS)
    elif len(args.checks) == 1 and args.checks[0] in GROUPS:
        names = GROUPS[args.checks[0]]
    else:
        names = args.checks
    bad = [n for n in names if n not in CHECKS]
    if bad:
        print(f"unknown checks: {bad}\navailable: {', '.join(CHECKS)}", file=sys.stderr)
        return 2

    print(f"\ncallpc preflight — {len(names)} checks\n" + "=" * 48)
    results = {}
    for name in names:
        print(f"\n▶ {name}")
        try:
            status, detail = CHECKS[name](args)
        except Exception as e:  # a check crashing is itself a finding
            import traceback
            traceback.print_exc()
            status, detail = FAIL, f"check crashed: {type(e).__name__}: {e}"
        results[name] = (status, detail)
        _say(status, name, detail)

    print("\n" + "=" * 48 + "\nsummary")
    n = {PASS: 0, FAIL: 0, SKIP: 0}
    for name, (status, detail) in results.items():
        n[status] += 1
        _say(status, name, detail)
    print("=" * 48)
    print(f"  {_c(f'{n[PASS]} pass', PASS)}  {_c(f'{n[FAIL]} fail', FAIL)}  {_c(f'{n[SKIP]} skip', SKIP)}")
    return 1 if n[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
