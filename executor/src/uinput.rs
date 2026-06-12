//! Linux input via a virtual uinput device — kernel-level evdev injection that
//! works on X11 AND Wayland (XTEST/enigo is a silent no-op on GNOME Wayland).
//!
//! Needs write access to /dev/uinput (the `input` group + a udev rule, or run
//! once with the right perms). Absolute pointer (ABS_X/Y over 0..=65535 scaled
//! to the screen) so clicks land at exact pixels — the dotool approach the PLAN
//! calls for (not ydotool, whose --absolute is broken).

use anyhow::{anyhow, Context, Result};
use input_linux::sys::input_event;
use input_linux::{
    AbsoluteAxis, AbsoluteEvent, AbsoluteInfo, AbsoluteInfoSetup, EventKind, EventTime,
    InputId, Key, KeyEvent, KeyState, RelativeAxis, RelativeEvent, SynchronizeEvent,
    SynchronizeKind, UInputHandle,
};
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::thread;
use std::time::Duration;

const ABS_MAX: i32 = 65535;

pub struct Uinput {
    handle: UInputHandle<File>,
    screen_w: i32,
    screen_h: i32,
}

impl Uinput {
    /// Open /dev/uinput and create the virtual device. Errs (so the caller can
    /// fall back) if /dev/uinput isn't writable.
    pub fn new(screen_w: u32, screen_h: u32) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .open("/dev/uinput")
            .context("opening /dev/uinput (need the 'input' group / a udev rule)")?;
        let handle = UInputHandle::new(file);

        handle.set_evbit(EventKind::Key)?;
        for key in ALL_KEYS.iter().chain(BUTTONS.iter()) {
            handle.set_keybit(*key)?;
        }
        handle.set_evbit(EventKind::Absolute)?;
        handle.set_absbit(AbsoluteAxis::X)?;
        handle.set_absbit(AbsoluteAxis::Y)?;
        handle.set_evbit(EventKind::Relative)?;
        handle.set_relbit(RelativeAxis::Wheel)?;
        handle.set_relbit(RelativeAxis::HorizontalWheel)?;

        let abs = |axis| AbsoluteInfoSetup {
            axis,
            info: AbsoluteInfo {
                value: 0,
                minimum: 0,
                maximum: ABS_MAX,
                fuzz: 0,
                flat: 0,
                resolution: 0,
            },
        };
        handle.create(
            &InputId::default(),
            b"callpc-virtual-input",
            0,
            &[abs(AbsoluteAxis::X), abs(AbsoluteAxis::Y)],
        )?;

        // Let udev create the node and the compositor notice the new device,
        // else the first events are dropped.
        thread::sleep(Duration::from_millis(400));
        Ok(Self {
            handle,
            screen_w: screen_w.max(1) as i32,
            screen_h: screen_h.max(1) as i32,
        })
    }

    fn emit(&self, events: &[input_event]) -> Result<()> {
        self.handle.write(events).context("writing uinput events")?;
        Ok(())
    }

    fn syn() -> input_event {
        *SynchronizeEvent::new(EventTime::default(), SynchronizeKind::Report, 0).as_ref()
    }

    fn move_to(&self, x: i32, y: i32) -> Result<()> {
        let ax = (x.clamp(0, self.screen_w) as i64 * ABS_MAX as i64 / self.screen_w as i64) as i32;
        let ay = (y.clamp(0, self.screen_h) as i64 * ABS_MAX as i64 / self.screen_h as i64) as i32;
        self.emit(&[
            *AbsoluteEvent::new(EventTime::default(), AbsoluteAxis::X, ax).as_ref(),
            *AbsoluteEvent::new(EventTime::default(), AbsoluteAxis::Y, ay).as_ref(),
            Self::syn(),
        ])
    }

    fn key(&self, key: Key, down: bool) -> Result<()> {
        let state = if down { KeyState::PRESSED } else { KeyState::RELEASED };
        self.emit(&[
            *KeyEvent::new(EventTime::default(), key, state).as_ref(),
            Self::syn(),
        ])
    }

    fn tap(&self, key: Key) -> Result<()> {
        self.key(key, true)?;
        thread::sleep(Duration::from_millis(8));
        self.key(key, false)
    }

    fn button(&self, btn: Key) -> Result<()> {
        self.tap(btn)
    }

    fn scroll(&self, axis: RelativeAxis, amount: i32) -> Result<()> {
        self.emit(&[
            *RelativeEvent::new(EventTime::default(), axis, amount).as_ref(),
            Self::syn(),
        ])
    }

    pub fn execute(&self, kind: &str, args: &Value) -> Result<()> {
        let i = |k: &str| args.get(k).and_then(Value::as_i64).map(|n| n as i32);
        match kind {
            "click" | "left_double" | "right_single" => {
                self.move_to(i("x").context("x")?, i("y").context("y")?)?;
                thread::sleep(Duration::from_millis(20));
                let btn = if kind == "right_single" { Key::ButtonRight } else { Key::ButtonLeft };
                self.button(btn)?;
                if kind == "left_double" {
                    thread::sleep(Duration::from_millis(60));
                    self.button(Key::ButtonLeft)?;
                }
            }
            "drag" => {
                let (x1, y1) = (i("x1").context("x1")?, i("y1").context("y1")?);
                let (x2, y2) = (i("x2").context("x2")?, i("y2").context("y2")?);
                self.move_to(x1, y1)?;
                self.key(Key::ButtonLeft, true)?;
                for s in 1..=10 {
                    self.move_to(x1 + (x2 - x1) * s / 10, y1 + (y2 - y1) * s / 10)?;
                    thread::sleep(Duration::from_millis(16));
                }
                self.key(Key::ButtonLeft, false)?;
            }
            "hotkey" => {
                let keys: Vec<Key> = args
                    .get("keys")
                    .and_then(Value::as_array)
                    .ok_or_else(|| anyhow!("missing keys"))?
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(map_key)
                    .collect();
                for k in &keys {
                    self.key(*k, true)?;
                }
                for k in keys.iter().rev() {
                    self.key(*k, false)?;
                }
            }
            "type" => {
                let text = args.get("text").and_then(Value::as_str).unwrap_or("");
                let submit = text.ends_with('\n');
                for c in text.trim_end_matches('\n').chars() {
                    if let Some((key, shift)) = char_key(c) {
                        if shift {
                            self.key(Key::LeftShift, true)?;
                        }
                        self.tap(key)?;
                        if shift {
                            self.key(Key::LeftShift, false)?;
                        }
                        thread::sleep(Duration::from_millis(6));
                    }
                }
                if submit {
                    self.tap(Key::Enter)?;
                }
            }
            "scroll" => {
                if let (Some(x), Some(y)) = (i("x"), i("y")) {
                    self.move_to(x, y)?;
                }
                let amount = args.get("amount").and_then(Value::as_i64).unwrap_or(5) as i32;
                match args.get("direction").and_then(Value::as_str).unwrap_or("down") {
                    "down" => self.scroll(RelativeAxis::Wheel, -amount)?,
                    "up" => self.scroll(RelativeAxis::Wheel, amount)?,
                    "right" => self.scroll(RelativeAxis::HorizontalWheel, amount)?,
                    "left" => self.scroll(RelativeAxis::HorizontalWheel, -amount)?,
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
}

impl Drop for Uinput {
    fn drop(&mut self) {
        let _ = self.handle.dev_destroy();
    }
}

const BUTTONS: [Key; 3] = [Key::ButtonLeft, Key::ButtonRight, Key::ButtonMiddle];

/// Keys we register on the virtual device (specials + the printable set used by
/// char_key). Registering a broad set up front is fine; unused ones cost nothing.
const ALL_KEYS: [Key; 60] = [
    Key::LeftCtrl, Key::LeftAlt, Key::LeftShift, Key::LeftMeta, Key::Enter, Key::Tab,
    Key::Esc, Key::Space, Key::Backspace, Key::Delete, Key::Home, Key::End, Key::PageUp,
    Key::PageDown, Key::Up, Key::Down, Key::Left, Key::Right,
    Key::A, Key::B, Key::C, Key::D, Key::E, Key::F, Key::G, Key::H, Key::I, Key::J, Key::K,
    Key::L, Key::M, Key::N, Key::O, Key::P, Key::Q, Key::R, Key::S, Key::T, Key::U, Key::V,
    Key::W, Key::X, Key::Y, Key::Z,
    Key::Num0, Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5, Key::Num6, Key::Num7,
    Key::Num8, Key::Num9,
    Key::Minus, Key::Equal, Key::Dot, Key::Comma, Key::Slash, Key::Semicolon,
];

/// Map a hotkey name (UI-TARS hotkey args, lowercase space-split) to a Key.
pub fn map_key(name: &str) -> Option<Key> {
    Some(match name.to_ascii_lowercase().as_str() {
        "ctrl" | "control" => Key::LeftCtrl,
        "alt" | "option" => Key::LeftAlt,
        "shift" => Key::LeftShift,
        "win" | "meta" | "cmd" | "command" | "super" => Key::LeftMeta,
        "enter" | "return" => Key::Enter,
        "tab" => Key::Tab,
        "esc" | "escape" => Key::Esc,
        "space" => Key::Space,
        "backspace" => Key::Backspace,
        "delete" | "del" => Key::Delete,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" | "pgup" => Key::PageUp,
        "pagedown" | "pgdn" => Key::PageDown,
        "up" => Key::Up,
        "down" => Key::Down,
        "left" => Key::Left,
        "right" => Key::Right,
        s if s.len() == 1 => return char_key(s.chars().next().unwrap()).map(|(k, _)| k),
        _ => return None,
    })
}

/// ASCII char -> (Key, needs_shift) on a US layout. Unmapped chars are skipped.
fn char_key(c: char) -> Option<(Key, bool)> {
    let k = match c {
        'a'..='z' => return Some((letter_key(c), false)),
        'A'..='Z' => return Some((letter_key(c.to_ascii_lowercase()), true)),
        '0' => Key::Num0, '1' => Key::Num1, '2' => Key::Num2, '3' => Key::Num3,
        '4' => Key::Num4, '5' => Key::Num5, '6' => Key::Num6, '7' => Key::Num7,
        '8' => Key::Num8, '9' => Key::Num9,
        ' ' => Key::Space,
        '-' => Key::Minus, '=' => Key::Equal, '.' => Key::Dot, ',' => Key::Comma,
        '/' => Key::Slash, ';' => Key::Semicolon,
        // shifted symbols
        '!' => return Some((Key::Num1, true)), '@' => return Some((Key::Num2, true)),
        '#' => return Some((Key::Num3, true)), '$' => return Some((Key::Num4, true)),
        '%' => return Some((Key::Num5, true)), '^' => return Some((Key::Num6, true)),
        '&' => return Some((Key::Num7, true)), '*' => return Some((Key::Num8, true)),
        '(' => return Some((Key::Num9, true)), ')' => return Some((Key::Num0, true)),
        '_' => return Some((Key::Minus, true)), '+' => return Some((Key::Equal, true)),
        '?' => return Some((Key::Slash, true)), ':' => return Some((Key::Semicolon, true)),
        _ => return None,
    };
    Some((k, false))
}

fn letter_key(c: char) -> Key {
    match c {
        'a' => Key::A, 'b' => Key::B, 'c' => Key::C, 'd' => Key::D, 'e' => Key::E, 'f' => Key::F,
        'g' => Key::G, 'h' => Key::H, 'i' => Key::I, 'j' => Key::J, 'k' => Key::K, 'l' => Key::L,
        'm' => Key::M, 'n' => Key::N, 'o' => Key::O, 'p' => Key::P, 'q' => Key::Q, 'r' => Key::R,
        's' => Key::S, 't' => Key::T, 'u' => Key::U, 'v' => Key::V, 'w' => Key::W, 'x' => Key::X,
        'y' => Key::Y, _ => Key::Z,
    }
}
