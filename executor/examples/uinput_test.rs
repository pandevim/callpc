// Verify the uinput device: create it, then move the cursor to screen center
// (a zero-amount "scroll" = move only, no click, no side effects).
#[path = "../src/uinput.rs"]
mod uinput;

fn main() {
    match uinput::Uinput::new(1920, 1080) {
        Ok(u) => {
            println!("uinput device created OK");
            let mv = serde_json::json!({"x":960,"y":540,"direction":"up","amount":0});
            match u.execute("scroll", &mv) {
                Ok(()) => println!("cursor move to (960,540) sent OK — did it move to center?"),
                Err(e) => println!("move ERR: {e:#}"),
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        Err(e) => println!("uinput create ERR: {e:#}"),
    }
}
