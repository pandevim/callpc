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

/// PNG screenshot of the primary monitor -> (png_bytes, width, height).
///
/// GNOME/KDE Wayland gate screen capture behind an interactive xdg-desktop-
/// portal grant — the first call pops an "Allow screen sharing?" dialog the
/// user must approve (a background/headless process can't, so it returns
/// "Screenshot failed or canceled"). X11/XWayland and a plain Xorg session
/// capture with no prompt. This is the documented Wayland friction (PLAN.md).
pub fn capture() -> Result<(Vec<u8>, u32, u32)> {
    let m = primary()?;
    let img = m.capture_image().context(
        "capturing screen (on GNOME/KDE Wayland, approve the screen-share \
         portal prompt, or use an Xorg session)")?;
    let (w, h) = (img.width(), img.height());
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .context("encoding png")?;
    Ok((buf, w, h))
}
