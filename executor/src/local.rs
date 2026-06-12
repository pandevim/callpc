//! Loopback presence beacon — lets the call web page tell whether it is open ON
//! the controlled computer (this binary running here) versus a phone / other
//! device, AND auto-configure the agent so the user never types a URL or code.
//!
//! The page fetches `http://127.0.0.1:<port>/whoami`; if it answers, the browser
//! is on this machine. If the agent is running but unconfigured, the page POSTs
//! the gateway URL + a fresh pairing code to `/configure`, and the agent connects
//! itself — "download, run, done", no terminal arguments.
//!
//! It is deliberately tiny: a hand-rolled HTTP/1.1 responder over a tokio
//! TcpListener (no hyper/axum — keeps the release binary small). It binds to
//! loopback only, so nothing is exposed beyond this host. Loopback origins are
//! "potentially trustworthy", so an HTTPS page may reach us without mixed-content
//! errors; Chrome's Private Network Access additionally needs the preflight to
//! carry `Access-Control-Allow-Private-Network: true`, which we always send.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc::UnboundedSender;

/// Static facts about this machine, gathered once at startup.
#[derive(Clone)]
pub struct BeaconInfo {
    pub name: String,
    pub os: &'static str,
    pub distro: String,
    pub screen_w: u32,
    pub screen_h: u32,
}

/// Shared flags + the channel the beacon uses to hand a `/configure` request to
/// the main connect loop.
#[derive(Clone)]
pub struct BeaconState {
    pub online: Arc<AtomicBool>,     // live gateway connection
    pub configured: Arc<AtomicBool>, // a connect target (url) is set
    pub cfg_tx: UnboundedSender<(String, String)>, // (url, pairing code) from /configure
}

/// Best-effort human-readable OS/distro. Linux reads PRETTY_NAME from
/// /etc/os-release; mac/Windows just report the family (the page only really
/// needs the distro to answer "which distro is this" on Linux).
pub fn distro() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/etc/os-release") {
            for line in s.lines() {
                if let Some(v) = line.strip_prefix("PRETTY_NAME=") {
                    return v.trim().trim_matches('"').to_string();
                }
            }
        }
        return "Linux".to_string();
    }
    #[cfg(target_os = "macos")]
    {
        return "macOS".to_string();
    }
    #[cfg(target_os = "windows")]
    {
        return "Windows".to_string();
    }
    #[allow(unreachable_code)]
    {
        "unknown".to_string()
    }
}

/// Serve the beacon until the process exits. `paired`/`online`/`configured` are
/// re-read per request so fresh pairing/connection is reflected without a restart.
pub async fn serve(port: u16, state: BeaconState, info: BeaconInfo) {
    if port == 0 {
        return; // disabled
    }
    let listener = match TcpListener::bind(("127.0.0.1", port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[exec] local beacon: can't bind 127.0.0.1:{port} ({e}) — disabled");
            return;
        }
    };
    println!("[exec] local beacon on http://127.0.0.1:{port}/whoami");
    loop {
        let (mut stream, _) = match listener.accept().await {
            Ok(p) => p,
            Err(_) => continue,
        };
        let state = state.clone();
        let info = info.clone();
        tokio::spawn(async move {
            let _ = handle(&mut stream, &state, &info).await;
        });
    }
}

// CORS + Private Network Access on every response (incl. the preflight).
const CORS: &str = "Access-Control-Allow-Origin: *\r\n\
     Access-Control-Allow-Private-Network: true\r\n\
     Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
     Access-Control-Allow-Headers: *\r\n";

async fn handle(
    stream: &mut tokio::net::TcpStream,
    state: &BeaconState,
    info: &BeaconInfo,
) -> std::io::Result<()> {
    // Read the request: headers, then (for POST) the body per Content-Length.
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    let header_end = loop {
        if let Some(pos) = find(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break buf.len(); // connection closed before full headers
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > 16 * 1024 {
            break buf.len(); // cap — a beacon request is tiny
        }
    };

    let head = String::from_utf8_lossy(&buf[..header_end.min(buf.len())]);
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");

    if method == "OPTIONS" {
        return write_resp(stream, "204 No Content", "", "").await; // preflight
    }

    if method == "POST" && path.starts_with("/configure") {
        // read the remaining body (Content-Length) if not already buffered
        let len: usize = head
            .lines()
            .find_map(|l| {
                let l = l.to_ascii_lowercase();
                l.strip_prefix("content-length:").map(|v| v.trim().parse().ok())
            })
            .flatten()
            .unwrap_or(0);
        let mut body = buf[header_end.min(buf.len())..].to_vec();
        while body.len() < len {
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..n]);
        }
        let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        let url = parsed.get("url").and_then(Value::as_str).unwrap_or("");
        let code = parsed.get("code").and_then(Value::as_str).unwrap_or("");
        let already_paired = crate::load_conf().token.is_some();
        let body = if already_paired {
            // never let a loopback page re-target an already-paired agent
            json!({"ok": false, "error": "already paired"}).to_string()
        } else if url.is_empty() || code.is_empty() {
            json!({"ok": false, "error": "url and code required"}).to_string()
        } else {
            state.configured.store(true, Ordering::Relaxed);
            let _ = state.cfg_tx.send((url.to_string(), code.to_string()));
            println!("[exec] configured by local page -> {url}");
            json!({"ok": true}).to_string()
        };
        return write_resp(stream, "200 OK", "application/json", &body).await;
    }

    // default: /whoami
    let paired = crate::load_conf().token.is_some();
    let body = json!({
        "callpc": true,
        "name": info.name,
        "os": info.os,
        "distro": info.distro,
        "screen_w": info.screen_w,
        "screen_h": info.screen_h,
        "paired": paired,
        "configured": state.configured.load(Ordering::Relaxed),
        "online": state.online.load(Ordering::Relaxed),
    })
    .to_string();
    write_resp(stream, "200 OK", "application/json", &body).await
}

async fn write_resp(
    stream: &mut tokio::net::TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let ct = if content_type.is_empty() {
        String::new()
    } else {
        format!("Content-Type: {content_type}\r\n")
    };
    let resp = format!(
        "HTTP/1.1 {status}\r\n{CORS}{ct}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.flush().await
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
