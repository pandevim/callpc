//! Loopback presence beacon — lets the call web page tell whether it is open ON
//! the controlled computer (this binary running here) versus a phone / other
//! device. The page fetches `http://127.0.0.1:<port>/whoami`; if it answers, the
//! browser is on this machine and we are connected.
//!
//! It is deliberately tiny: a hand-rolled HTTP/1.1 responder over a tokio
//! TcpListener (no hyper/axum — keeps the release binary small). It binds to
//! loopback only, so nothing is exposed beyond this host. Loopback origins are
//! "potentially trustworthy", so an HTTPS page may fetch us without mixed-content
//! errors; Chrome's Private Network Access additionally needs the preflight to
//! carry `Access-Control-Allow-Private-Network: true`, which we always send.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Static facts about this machine, gathered once at startup.
#[derive(Clone)]
pub struct BeaconInfo {
    pub name: String,
    pub os: &'static str,
    pub distro: String,
    pub screen_w: u32,
    pub screen_h: u32,
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

/// Serve the beacon until the process exits. `online` mirrors the live gateway
/// connection; `paired` is re-read from disk per request so a fresh pairing is
/// reflected without a restart.
pub async fn serve(port: u16, online: Arc<AtomicBool>, info: BeaconInfo) {
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
        let online = online.clone();
        let info = info.clone();
        tokio::spawn(async move {
            let _ = handle(&mut stream, &online, &info).await;
        });
    }
}

async fn handle(
    stream: &mut tokio::net::TcpStream,
    online: &Arc<AtomicBool>,
    info: &BeaconInfo,
) -> std::io::Result<()> {
    // Read just enough to learn the method + reach the end of headers. We never
    // need the body. Cap the read so a stray client can't make us spin.
    let mut buf = vec![0u8; 2048];
    let mut filled = 0;
    let mut is_options = false;
    loop {
        let n = stream.read(&mut buf[filled..]).await?;
        if n == 0 {
            break;
        }
        filled += n;
        if filled >= 7 {
            is_options = buf.starts_with(b"OPTIONS");
        }
        if buf[..filled].windows(4).any(|w| w == b"\r\n\r\n") || filled == buf.len() {
            break;
        }
    }

    // CORS + Private Network Access on every response (incl. the preflight).
    const CORS: &str = "Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Private-Network: true\r\n\
         Access-Control-Allow-Methods: GET, OPTIONS\r\n\
         Access-Control-Allow-Headers: *\r\n";

    if is_options {
        let resp = format!("HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\n\r\n");
        stream.write_all(resp.as_bytes()).await?;
        return stream.flush().await;
    }

    let paired = crate::load_conf().token.is_some();
    let body = json!({
        "callpc": true,
        "name": info.name,
        "os": info.os,
        "distro": info.distro,
        "screen_w": info.screen_w,
        "screen_h": info.screen_h,
        "paired": paired,
        "online": online.load(Ordering::Relaxed),
    })
    .to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\n{CORS}Content-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.flush().await
}
