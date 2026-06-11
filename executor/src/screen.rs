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
