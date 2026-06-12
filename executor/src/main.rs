//! callpc-executor — the ONLY thing installed on the controlled computer.
//!
//! Deliberately dumb (docs/PROTOCOL.md): connect outbound WSS to the gateway,
//! pair once (or hello with the stored token), then answer screenshot
//! requests and execute input actions. No inbound ports, no smarts.
//!
//!     callpc-executor --url wss://<host>/control/ws --code AB3K9Q   # first run
//!     callpc-executor --url wss://<host>/control/ws                 # paired
//!
//! Token persists in ~/.config/callpc/executor.json (same file the python
//! fake executor uses). Reconnects with backoff. Safety: actions are
//! rate-limited (120/min) and every action is logged to the console — the
//! visible "being controlled" indicator for v1.

mod input;
mod local;
mod screen;
#[cfg(target_os = "linux")]
mod uinput;

use anyhow::{anyhow, bail, Context, Result};
use base64::prelude::*;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[derive(Default, Serialize, Deserialize)]
struct Conf {
    token: Option<String>,
    // gateway URL, persisted so a paired agent reconnects with no arguments
    // (the website's auto-configure sets this on first pair).
    #[serde(default)]
    url: Option<String>,
}

fn conf_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("callpc")
        .join("executor.json")
}

fn load_conf() -> Conf {
    std::fs::read_to_string(conf_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_conf(conf: &Conf) -> Result<()> {
    let p = conf_path();
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&p, serde_json::to_string_pretty(conf)?)?;
    Ok(())
}

fn os_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}

struct RateLimit {
    window: Vec<Instant>,
}

impl RateLimit {
    const MAX_PER_MIN: usize = 120;

    fn allow(&mut self) -> bool {
        let now = Instant::now();
        self.window.retain(|t| now.duration_since(*t) < Duration::from_secs(60));
        if self.window.len() >= Self::MAX_PER_MIN {
            return false;
        }
        self.window.push(now);
        true
    }
}

async fn run(
    url: &str,
    code: Option<&str>,
    input: &input::InputThread,
    online: &Arc<AtomicBool>,
    distro: &str,
) -> Result<()> {
    let mut conf = load_conf();
    let code = if conf.token.is_some() {
        if code.is_some() {
            eprintln!("[exec] token already stored — ignoring --code (delete {} to re-pair)",
                      conf_path().display());
        }
        None
    } else {
        code
    };

    let (ws, _) = connect_async(url).await.context("connecting")?;
    let (mut tx, mut rx) = ws.split();

    let (w, h) = screen::size().unwrap_or((0, 0));
    let name = gethostname::gethostname().to_string_lossy().into_owned();
    let info = json!({"name": name, "os": os_name(), "distro": distro,
                      "screen_w": w, "screen_h": h});

    // pair or hello
    let first = if let Some(code) = code {
        let mut m = json!({"type": "pair", "code": code});
        merge(&mut m, &info);
        m
    } else {
        let token = conf.token.clone().ok_or_else(|| {
            anyhow!("no stored token — run once with --code <pairing code>")
        })?;
        let mut m = json!({"type": "hello", "token": token});
        merge(&mut m, &info);
        m
    };
    tx.send(Message::Text(first.to_string())).await?;

    let reply: Value = match rx.next().await {
        Some(Ok(Message::Text(t))) => serde_json::from_str(&t)?,
        other => bail!("no reply to pair/hello: {other:?}"),
    };
    match reply.get("type").and_then(Value::as_str) {
        Some("paired") => {
            conf.token = reply.get("token").and_then(Value::as_str).map(String::from);
            conf.url = Some(url.to_string()); // remember where to reconnect, no args next time
            save_conf(&conf)?;
            println!("[exec] paired! token stored in {}", conf_path().display());
        }
        Some("ready") => {
            if conf.url.as_deref() != Some(url) {
                conf.url = Some(url.to_string());
                save_conf(&conf)?;
            }
        }
        _ => bail!("rejected: {reply}"),
    }
    println!("[exec] online as {name} ({}, {w}x{h}) — waiting for requests", os_name());
    online.store(true, Ordering::Relaxed); // beacon now reports a live connection

    let mut rate = RateLimit { window: Vec::new() };

    while let Some(msg) = rx.next().await {
        let text = match msg? {
            Message::Text(t) => t,
            Message::Ping(_) | Message::Pong(_) | Message::Binary(_) => continue,
            Message::Close(_) => break,
            _ => continue,
        };
        let m: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = m.get("id").cloned().unwrap_or(Value::Null);
        match m.get("type").and_then(Value::as_str) {
            Some("screenshot_request") => {
                let reply = match capture_any().await {
                    Ok((png, w, h)) => {
                        println!("[exec] screenshot -> {} bytes", png.len());
                        json!({"type": "screenshot", "id": id,
                               "png_b64": BASE64_STANDARD.encode(&png), "w": w, "h": h})
                    }
                    Err(e) => {
                        eprintln!("[exec] screenshot failed: {e:#}");
                        json!({"type": "error", "id": id, "error": format!("screenshot: {e}")})
                    }
                };
                tx.send(Message::Text(reply.to_string())).await?;
            }
            Some("action") => {
                let kind = m.get("kind").and_then(Value::as_str).unwrap_or("").to_string();
                let args = m.get("args").cloned().unwrap_or(json!({}));
                println!("[exec] ACTION {kind} {args}");
                let result = if !rate.allow() {
                    Err("rate limited (120 actions/min)".to_string())
                } else {
                    input.execute(kind, args).await
                };
                let reply = match result {
                    Ok(()) => json!({"type": "action_result", "id": id, "ok": true, "error": null}),
                    Err(e) => {
                        eprintln!("[exec] action failed: {e}");
                        json!({"type": "action_result", "id": id, "ok": false, "error": e})
                    }
                };
                tx.send(Message::Text(reply.to_string())).await?;
            }
            Some("pong") | None => {}
            Some(other) => eprintln!("[exec] unknown message type {other:?}"),
        }
    }
    Ok(())
}

/// Screenshot, preferring the xdg-desktop-portal on Linux (the only path GNOME/
/// KDE Wayland allow) and falling back to xcap (X11/Win/Mac, and Linux-X11).
async fn capture_any() -> Result<(Vec<u8>, u32, u32)> {
    #[cfg(target_os = "linux")]
    {
        match screen::capture_portal().await {
            Ok(r) => return Ok(r),
            Err(e) => eprintln!("[exec] portal screenshot failed ({e:#}); trying xcap"),
        }
    }
    tokio::task::spawn_blocking(screen::capture).await?
}

fn merge(into: &mut Value, from: &Value) {
    if let (Some(a), Some(b)) = (into.as_object_mut(), from.as_object()) {
        for (k, v) in b {
            a.insert(k.clone(), v.clone());
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // URL resolution: --url > CALLPC_URL env > persisted conf.url > none.
    // With none (and no stored token) the agent idles and waits for the website
    // to auto-configure it via the local beacon — "download, run, done".
    let mut url: Option<String> = std::env::var("CALLPC_URL").ok();
    let mut code: Option<String> = None;
    let mut once = false;
    // Loopback presence beacon: lets the call page detect it's open ON this
    // machine. 0 disables. CALLPC_LOCAL_PORT overrides; --local-port wins.
    let mut local_port: u16 = std::env::var("CALLPC_LOCAL_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8724);

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => { i += 1; url = Some(args.get(i).cloned().context("--url needs a value")?); }
            "--code" => { i += 1; code = Some(args.get(i).cloned().context("--code needs a value")?); }
            "--local-port" => {
                i += 1;
                local_port = args.get(i).context("--local-port needs a value")?
                    .parse().context("--local-port must be a number (0 disables)")?;
            }
            "--once" => once = true,
            "-h" | "--help" => {
                println!("callpc-executor [--url wss://<host>/control/ws] [--code ABC123] \
                          [--local-port 8724] [--once]\n\
                          With no --url/--code, run it and pair from the website (auto-configure).");
                return Ok(());
            }
            other => bail!("unknown argument {other:?}"),
        }
        i += 1;
    }

    // rustls 0.23 no longer auto-selects a crypto backend (panics otherwise).
    let _ = rustls::crypto::ring::default_provider().install_default();

    let conf = load_conf();
    if url.is_none() {
        url = conf.url.clone(); // a previously-paired agent reconnects with no args
    }

    let (sw, sh) = screen::size().unwrap_or((1920, 1080));
    let input = input::InputThread::spawn(sw, sh);

    // local presence beacon (loopback only) + the auto-configure channel
    let online = Arc::new(AtomicBool::new(false));
    // "configured" == we know a gateway URL to connect to (from args/env/conf).
    // The website auto-configures us only when this is false; the agent accepts
    // /configure only while this is false (so a paired agent can't be re-targeted).
    let configured = Arc::new(AtomicBool::new(url.is_some()));
    let distro = local::distro();
    let (cfg_tx, mut cfg_rx) = tokio::sync::mpsc::unbounded_channel::<(String, String)>();
    tokio::spawn(local::serve(
        local_port,
        local::BeaconState { online: online.clone(), configured: configured.clone(), cfg_tx },
        local::BeaconInfo {
            name: gethostname::gethostname().to_string_lossy().into_owned(),
            os: os_name(),
            distro: distro.clone(),
            screen_w: sw,
            screen_h: sh,
        },
    ));

    let mut backoff = 2u64;
    loop {
        // No gateway URL known (fresh, or a token from an older build that didn't
        // persist the URL) -> wait for the website to POST /configure.
        if url.is_none() {
            println!("[exec] waiting to be configured — open callpc in a browser on this computer");
            match cfg_rx.recv().await {
                Some((u, c)) => { url = Some(u); code = Some(c); configured.store(true, Ordering::Relaxed); }
                None => return Ok(()), // beacon gone; nothing more to do
            }
        }
        let target = url.clone().expect("url set above");
        match run(&target, code.as_deref(), &input, &online, &distro).await {
            Ok(()) => {
                backoff = 2; // clean close (server restart) — retry soon
            }
            Err(e) => eprintln!("[exec] connection lost: {e:#}"),
        }
        online.store(false, Ordering::Relaxed); // connection dropped
        code = None; // a redeemed pair code must never be replayed
        // a /configure that arrived during the session re-targets us
        while let Ok((u, c)) = cfg_rx.try_recv() { url = Some(u); code = Some(c); }
        if url.is_none() { url = load_conf().url; }
        if once {
            return Ok(());
        }
        eprintln!("[exec] reconnecting in {backoff}s...");
        tokio::time::sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(60);
    }
}
