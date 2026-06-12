//! Input injection via enigo (SendInput / CGEvent / x11rb), on a dedicated
//! thread: Enigo isn't Send and platform backends want a single owner, so the
//! async side talks to it over a channel.

use anyhow::{anyhow, Result};
use enigo::{
    Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings,
};
use serde_json::Value;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tokio::sync::oneshot;

pub struct InputThread {
    tx: mpsc::Sender<(String, Value, oneshot::Sender<Result<(), String>>)>,
}

impl InputThread {
    /// screen_w/h feed the Linux uinput backend's absolute pointer.
    pub fn spawn(screen_w: u32, screen_h: u32) -> Self {
        let (tx, rx) = mpsc::channel::<(String, Value, oneshot::Sender<Result<(), String>>)>();
        thread::spawn(move || {
            // Linux: prefer uinput (kernel evdev — works on Wayland, where
            // enigo/XTEST silently does nothing). Fall back to enigo only if
            // /dev/uinput isn't writable.
            #[cfg(target_os = "linux")]
            {
                match crate::uinput::Uinput::new(screen_w, screen_h) {
                    Ok(u) => {
                        println!("[exec] input backend: uinput (kernel evdev)");
                        while let Ok((kind, args, reply)) = rx.recv() {
                            let _ = reply.send(u.execute(&kind, &args).map_err(|e| e.to_string()));
                        }
                        return;
                    }
                    Err(e) => eprintln!(
                        "[exec] uinput unavailable ({e:#}); falling back to enigo \
                         (X11 only — a no-op on Wayland)"),
                }
            }
            let _ = (screen_w, screen_h); // used only by the linux branch
            // catch_unwind: some enigo backends panic on init (e.g. a denied
            // Wayland portal) instead of returning Err — don't take the
            // process down, fail every action with a clear message instead.
            let enigo = std::panic::catch_unwind(|| Enigo::new(&Settings::default()))
                .unwrap_or_else(|_| Err(enigo::NewConError::EstablishCon(
                    "input backend panicked on init")));
            let mut enigo = match enigo {
                Ok(e) => { println!("[exec] input backend: enigo"); e }
                Err(e) => {
                    eprintln!("[exec] FATAL: input backend init failed: {e}");
                    // drain and fail every request so the server sees errors
                    while let Ok((_, _, reply)) = rx.recv() {
                        let _ = reply.send(Err(format!("input backend unavailable: {e}")));
                    }
                    return;
                }
            };
            while let Ok((kind, args, reply)) = rx.recv() {
                let result = perform(&mut enigo, &kind, &args).map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
        });
        Self { tx }
    }

    pub async fn execute(&self, kind: String, args: Value) -> Result<(), String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send((kind, args, reply_tx))
            .map_err(|_| "input thread died".to_string())?;
        reply_rx.await.map_err(|_| "input thread died".to_string())?
    }
}

fn i(v: &Value, key: &str) -> Result<i32> {
    v.get(key)
        .and_then(Value::as_i64)
        .map(|n| n as i32)
        .ok_or_else(|| anyhow!("missing numeric arg {key:?}"))
}

fn perform(enigo: &mut Enigo, kind: &str, args: &Value) -> Result<()> {
    match kind {
        "click" => {
            enigo.move_mouse(i(args, "x")?, i(args, "y")?, Coordinate::Abs)?;
            enigo.button(Button::Left, Direction::Click)?;
        }
        "left_double" => {
            enigo.move_mouse(i(args, "x")?, i(args, "y")?, Coordinate::Abs)?;
            enigo.button(Button::Left, Direction::Click)?;
            thread::sleep(Duration::from_millis(80));
            enigo.button(Button::Left, Direction::Click)?;
        }
        "right_single" => {
            enigo.move_mouse(i(args, "x")?, i(args, "y")?, Coordinate::Abs)?;
            enigo.button(Button::Right, Direction::Click)?;
        }
        "drag" => {
            let (x1, y1) = (i(args, "x1")?, i(args, "y1")?);
            let (x2, y2) = (i(args, "x2")?, i(args, "y2")?);
            enigo.move_mouse(x1, y1, Coordinate::Abs)?;
            enigo.button(Button::Left, Direction::Press)?;
            // a few intermediate points so apps register the drag
            for step in 1..=8 {
                let x = x1 + (x2 - x1) * step / 8;
                let y = y1 + (y2 - y1) * step / 8;
                enigo.move_mouse(x, y, Coordinate::Abs)?;
                thread::sleep(Duration::from_millis(20));
            }
            enigo.button(Button::Left, Direction::Release)?;
        }
        "hotkey" => {
            let keys: Vec<Key> = args
                .get("keys")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("missing keys"))?
                .iter()
                .filter_map(Value::as_str)
                .map(map_key)
                .collect::<Result<_>>()?;
            for k in &keys {
                enigo.key(*k, Direction::Press)?;
            }
            for k in keys.iter().rev() {
                enigo.key(*k, Direction::Release)?;
            }
        }
        "type" => {
            let text = args.get("text").and_then(Value::as_str).unwrap_or("");
            // trailing \n means "submit" (UI-TARS convention)
            let submit = text.ends_with('\n');
            let body = text.trim_end_matches('\n');
            if !body.is_empty() {
                enigo.text(body)?;
            }
            if submit {
                enigo.key(Key::Return, Direction::Click)?;
            }
        }
        "scroll" => {
            enigo.move_mouse(i(args, "x")?, i(args, "y")?, Coordinate::Abs)?;
            let amount = args.get("amount").and_then(Value::as_i64).unwrap_or(5) as i32;
            let dir = args.get("direction").and_then(Value::as_str).unwrap_or("down");
            match dir {
                "down" => enigo.scroll(amount, Axis::Vertical)?,
                "up" => enigo.scroll(-amount, Axis::Vertical)?,
                "right" => enigo.scroll(amount, Axis::Horizontal)?,
                "left" => enigo.scroll(-amount, Axis::Horizontal)?,
                other => return Err(anyhow!("unknown scroll direction {other:?}")),
            }
        }
        "wait" => {
            let ms = args.get("ms").and_then(Value::as_u64).unwrap_or(1000).min(10_000);
            thread::sleep(Duration::from_millis(ms));
        }
        other => return Err(anyhow!("unknown action kind {other:?}")),
    }
    Ok(())
}

fn map_key(name: &str) -> Result<Key> {
    let k = match name.to_ascii_lowercase().as_str() {
        "ctrl" | "control" => Key::Control,
        "alt" | "option" => Key::Alt,
        "shift" => Key::Shift,
        "win" | "meta" | "cmd" | "command" | "super" => Key::Meta,
        "enter" | "return" => Key::Return,
        "tab" => Key::Tab,
        "esc" | "escape" => Key::Escape,
        "space" => Key::Space,
        "backspace" => Key::Backspace,
        "delete" | "del" => Key::Delete,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" | "pgup" => Key::PageUp,
        "pagedown" | "pgdn" => Key::PageDown,
        "up" => Key::UpArrow,
        "down" => Key::DownArrow,
        "left" => Key::LeftArrow,
        "right" => Key::RightArrow,
        "f1" => Key::F1, "f2" => Key::F2, "f3" => Key::F3, "f4" => Key::F4,
        "f5" => Key::F5, "f6" => Key::F6, "f7" => Key::F7, "f8" => Key::F8,
        "f9" => Key::F9, "f10" => Key::F10, "f11" => Key::F11, "f12" => Key::F12,
        s => {
            let mut chars = s.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Key::Unicode(c),
                _ => return Err(anyhow!("unknown key {name:?}")),
            }
        }
    };
    Ok(k)
}
