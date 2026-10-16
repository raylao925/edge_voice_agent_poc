//! Thin native client: the dialog UI is `www/index.html` (shared with the
//! browser demo), talking to the Rust server over WebSocket.
//! Tray / global hotkey / autostart land in a later step per the roadmap.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
