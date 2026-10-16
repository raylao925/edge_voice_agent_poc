//! Thin native client: the dialog UI is `www/index.html` (shared with the
//! browser demo), talking to the Rust server over WebSocket.
//!
//! Native extras:
//! - system tray (left-click toggles window; menu: show/hide, new
//!   conversation, launch-at-login toggle, quit)
//! - global hotkey Ctrl/Cmd+Shift+Space toggles the window
//! - `--minimized` start (used by the autostart entry) hides to tray
//! - closing the window hides to tray instead of quitting
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{
    menu::{CheckMenuItemBuilder, MenuBuilder, MenuItemBuilder, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, WindowEvent,
};
use tauri_plugin_autostart::{MacosLauncher, ManagerExt};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// Global hotkey: toggle the main window.
fn toggle_shortcut() -> Shortcut {
    #[cfg(target_os = "macos")]
    let mods = Modifiers::SUPER | Modifiers::SHIFT;
    #[cfg(not(target_os = "macos"))]
    let mods = Modifiers::CONTROL | Modifiers::SHIFT;
    Shortcut::new(Some(mods), Code::Space)
}

fn toggle_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        if w.is_visible().unwrap_or(true) {
            let _ = w.hide();
        } else {
            let _ = w.show();
            let _ = w.set_focus();
        }
    }
}

/// Ask the frontend to start a fresh conversation (sends
/// `{type:"new_conversation"}` on the WebSocket).
fn new_conversation(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.set_focus();
        let _ = w.eval("window.dispatchEvent(new CustomEvent('va-new-conversation'))");
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            Some(vec!["--minimized"]),
        ))
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            // global hotkey -> toggle window (fire on press only)
            let toggle = toggle_shortcut();
            app.global_shortcut().on_shortcut(toggle, |app, _sc, event| {
                if event.state() == ShortcutState::Pressed {
                    toggle_window(app);
                }
            })?;

            // tray menu
            let auto_enabled = app.autolaunch().is_enabled().unwrap_or(false);
            let show = MenuItemBuilder::with_id("show", "Show / Hide").build(app)?;
            let new_conv =
                MenuItemBuilder::with_id("new", "New conversation").build(app)?;
            let autostart_item =
                CheckMenuItemBuilder::with_id("autostart", "Launch at login")
                    .checked(auto_enabled)
                    .build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let menu = MenuBuilder::new(app)
                .items(&[&show, &new_conv, &sep, &autostart_item, &sep, &quit])
                .build()?;

            TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("Voice Agent")
                .menu(&menu)
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "show" => toggle_window(app),
                    "new" => new_conversation(app),
                    "autostart" => {
                        let am = app.autolaunch();
                        let now_on = am.is_enabled().unwrap_or(false);
                        let ok = if now_on { am.disable() } else { am.enable() }.is_ok();
                        if ok {
                            let _ = autostart_item.set_checked(!now_on);
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        toggle_window(tray.app_handle());
                    }
                })
                .build(app)?;

            // autostart entry launches with --minimized: stay in tray
            if std::env::args().any(|a| a == "--minimized") {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.hide();
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // close -> hide to tray instead of quitting
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
