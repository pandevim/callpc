// Read-only hardware smoke test: screen size + screenshot via xcap (the
// Wayland-portal path). No input — that's validated live during the demo.
#[path = "../src/screen.rs"]
mod screen;

fn main() {
    match screen::size() {
        Ok((w, h)) => println!("screen size: {w}x{h}"),
        Err(e) => println!("size ERR: {e:#}"),
    }
    match screen::capture() {
        Ok((png, w, h)) => {
            std::fs::write("/tmp/callpc_shot.png", &png).unwrap();
            println!("screenshot OK: {w}x{h}, {} bytes -> /tmp/callpc_shot.png", png.len());
        }
        Err(e) => println!("screenshot ERR: {e:#}"),
    }
}
