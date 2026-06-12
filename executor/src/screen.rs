//! Screenshots via xcap (DXGI / CoreGraphics / X11 / Wayland-portal).

use anyhow::{anyhow, Context, Result};
use std::io::Cursor;
use xcap::Monitor;

fn primary() -> Result<Monitor> {
    let monitors = Monitor::all().context("listing monitors")?;
    monitors
        .iter()
        .find(|m| m.is_primary())
        .cloned()
        .or_else(|| monitors.into_iter().next())
        .ok_or_else(|| anyhow!("no monitors found"))
}

/// Screen size of the primary monitor (for hello{screen_w, screen_h}).
pub fn size() -> Result<(u32, u32)> {
    let m = primary()?;
    Ok((m.width(), m.height()))
}

/// Linux: capture via xdg-desktop-portal Screenshot. GNOME/KDE block xcap's
/// org.gnome.Shell.Screenshot path outright (no prompt, just "failed or
/// canceled"); the portal is the supported route — it shows a one-time
/// permission prompt, then remembers the grant. Returns PNG bytes the portal
/// already wrote. Works on X11 too (no prompt there).
#[cfg(target_os = "linux")]
pub async fn capture_portal() -> Result<(Vec<u8>, u32, u32)> {
    use ashpd::desktop::screenshot::Screenshot;

    let response = Screenshot::request()
        .interactive(false) // grab the whole screen; don't make the user pick a region
        .modal(false)
        .send()
        .await
        .context("portal screenshot request")?
        .response()
        .context("portal screenshot denied or cancelled")?;

    let path = url::Url::parse(response.uri().as_str())
        .ok()
        .and_then(|u| u.to_file_path().ok())
        .ok_or_else(|| anyhow!("portal returned a non-file uri: {}", response.uri()))?;
    let bytes = std::fs::read(&path).context("reading portal screenshot file")?;
    let img = image::load_from_memory(&bytes).context("decoding portal screenshot")?;
    let (w, h) = (img.width(), img.height());
    // portal writes PNG already, but re-encode defensively so the server always
    // gets PNG regardless of the portal backend's chosen format.
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .context("re-encoding portal screenshot")?;
    let _ = std::fs::remove_file(&path); // the portal file is ours to clean up
    Ok((out, w, h))
}

/// PNG screenshot via xcap (DXGI / CoreGraphics / X11). The Linux path prefers
/// `capture_portal` (above); this is the Windows/macOS path and the Linux-X11
/// fallback.
pub fn capture() -> Result<(Vec<u8>, u32, u32)> {
    let m = primary()?;
    let img = m.capture_image().context("capturing screen")?;
    let (w, h) = (img.width(), img.height());
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .context("encoding png")?;
    Ok((buf, w, h))
}
