use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use base64::prelude::*;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::pipeline::{run_turn, AppContext};
use crate::protocol::{ClientMsg, Event};

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(ctx): State<Arc<AppContext>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, ctx))
}

async fn send_event(socket: &mut WebSocket, ev: &Event) {
    match serde_json::to_string(ev) {
        Ok(json) => {
            if socket.send(Message::Text(json)).await.is_err() {
                warn!("ws send failed");
            }
        }
        Err(e) => warn!("event serialize failed: {e}"),
    }
}

async fn handle_socket(mut socket: WebSocket, ctx: Arc<AppContext>) {
    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break,
        };
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };
        let parsed: ClientMsg = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(_) => continue,
        };
        match parsed {
            ClientMsg::Ping => send_event(&mut socket, &Event::Pong).await,
            ClientMsg::Utterance { data } => {
                let rec = match BASE64_STANDARD.decode(&data) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                info!("[ws] utterance: {} bytes", rec.len());

                let (tx, mut rx) = mpsc::channel::<Event>(32);
                let ctx2 = ctx.clone();
                tokio::spawn(async move {
                    if let Err(e) = run_turn(rec, ctx2, tx.clone()).await {
                        warn!("[turn] error: {e:#}");
                        tx.send(Event::Error {
                            message: format!("{e:#}")[..200.min(format!("{e:#}").len())]
                                .to_string(),
                        })
                        .await
                        .ok();
                    }
                });

                // forward events until the turn ends
                while let Some(ev) = rx.recv().await {
                    if let Event::Done(ref d) = ev {
                        info!(
                            "[done] ttfb={:?}s sentences={} total={:?}s",
                            d.ttfb, d.sentences, d.timings.get("total")
                        );
                    }
                    let is_done = matches!(ev, Event::Done(_) | Event::Error { .. });
                    send_event(&mut socket, &ev).await;
                    if is_done {
                        break;
                    }
                }
            }
        }
    }
}
