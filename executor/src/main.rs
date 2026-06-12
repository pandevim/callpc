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
mod screen;
#[cfg(target_os = "linux")]
mod uinput;

use anyhow::{anyhow, bail, Context, Result};
use base64::prelude::*;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[derive(Default, Serialize, Deserialize)]
struct Conf {
    token: Option<String>,
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

async fn run(url: &str, code: Option<&str>, input: &input::InputThread) -> Result<()> {
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
    let info = json!({"name": name, "os": os_name(), "screen_w": w, "screen_h": h});

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
            save_conf(&conf)?;
            println!("[exec] paired! token stored in {}", conf_path().display());
        }
        Some("ready") => {}
        _ => bail!("rejected: {reply}"),
    }
    println!("[exec] online as {name} ({}, {w}x{h}) — waiting for requests", os_name());

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
    let mut url = std::env::var("CALLPC_URL")
        .unwrap_or_else(|_| "ws://127.0.0.1:8723/control/ws".into());
    let mut code: Option<String> = None;
    let mut once = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => { i += 1; url = args.get(i).cloned().context("--url needs a value")?; }
            "--code" => { i += 1; code = Some(args.get(i).cloned().context("--code needs a value")?); }
            "--once" => once = true,
            "-h" | "--help" => {
                println!("callpc-executor --url wss://<host>/control/ws [--code ABC123] [--once]");
                return Ok(());
            }
            other => bail!("unknown argument {other:?}"),
        }
        i += 1;
    }

    // rustls 0.23 no longer auto-selects a crypto backend (panics otherwise).
    let _ = rustls::crypto::ring::default_provider().install_default();

    let (sw, sh) = screen::size().unwrap_or((1920, 1080));
    let input = input::InputThread::spawn(sw, sh);
    let mut backoff = 2u64;
    loop {
        match run(&url, code.as_deref(), &input).await {
            Ok(()) => {
                backoff = 2; // clean close (server restart) — retry soon
            }
            Err(e) => eprintln!("[exec] connection lost: {e:#}"),
        }
        code = None; // a redeemed pair code must never be replayed
        if once {
            return Ok(());
        }
        eprintln!("[exec] reconnecting in {backoff}s...");
        tokio::time::sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(60);
    }
}
