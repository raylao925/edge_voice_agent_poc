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
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::conversation::Conversation;
use crate::kws::KwsStream;
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

/// Per-connection state for an in-flight turn.
struct ActiveTurn {
    cancel: CancellationToken,
    rx: mpsc::Receiver<Event>,
}

async fn handle_socket(mut socket: WebSocket, ctx: Arc<AppContext>) {
    // One conversation per connection: in-memory multi-turn history.
    let convo = Arc::new(std::sync::Mutex::new(Conversation::new(10)));
    // Lazily-created keyword-spotter stream for `audio_chunk` listen mode.
    let mut kws_stream: Option<KwsStream> = None;
    // In-flight turn, if any. The select! loop keeps reading client messages
    // while a turn runs so `interrupt` (barge-in) is handled promptly.
    let mut active: Option<ActiveTurn> = None;

    loop {
        tokio::select! {
            msg = socket.recv() => {
                let Some(msg) = msg else { break };
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
                    ClientMsg::NewConversation => {
                        let n = {
                            let mut c = convo.lock().unwrap();
                            let n = c.len();
                            c.clear();
                            n
                        };
                        info!("[ws] new conversation (cleared {n} turns)");
                        send_event(
                            &mut socket,
                            &Event::Status {
                                text: "已开始新对话".to_string(),
                            },
                        )
                        .await;
                    }
                    ClientMsg::Utterance { data } => {
                        if active.is_some() {
                            // A turn is already running; the client should
                            // send `interrupt` first (barge-in flow).
                            warn!("[ws] utterance while turn active: ignored");
                            continue;
                        }
                        let rec = match BASE64_STANDARD.decode(&data) {
                            Ok(b) => b,
                            Err(_) => continue,
                        };
                        info!("[ws] utterance: {} bytes", rec.len());

                        let (tx, rx) = mpsc::channel::<Event>(32);
                        let cancel = CancellationToken::new();
                        let ctx2 = ctx.clone();
                        let convo2 = convo.clone();
                        let cancel2 = cancel.clone();
                        tokio::spawn(async move {
                            if let Err(e) =
                                run_turn(rec, ctx2, convo2, cancel2, tx.clone()).await
                            {
                                warn!("[turn] error: {e:#}");
                                tx.send(Event::Error {
                                    message: format!("{e:#}")
                                        [..200.min(format!("{e:#}").len())]
                                        .to_string(),
                                })
                                .await
                                .ok();
                            }
                        });
                        active = Some(ActiveTurn { cancel, rx });
                    }
                    ClientMsg::Interrupt => {
                        if let Some(turn) = active.take() {
                            info!("[ws] interrupt: cancelling in-flight turn");
                            turn.cancel.cancel();
                            // run_turn returns early without Done; the
                            // forwarder arm below will see the channel close.
                            // Drop the receiver now so no stale events leak.
                            drop(turn.rx);
                            send_event(&mut socket, &Event::Interrupted).await;
                        }
                    }
                    ClientMsg::AudioChunk { data, seq } => {
                        // Wake-word listen mode: only when idle.
                        if active.is_some() {
                            continue;
                        }
                        let Some(kws) = &ctx.kws else {
                            continue;
                        };
                        let pcm = match BASE64_STANDARD.decode(&data) {
                            Ok(b) => b,
                            Err(_) => continue,
                        };
                        // s16le mono 16kHz -> f32
                        if pcm.len() % 2 != 0 || pcm.is_empty() {
                            continue;
                        }
                        let samples: Vec<f32> = pcm
                            .chunks_exact(2)
                            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                            .collect();
                        let stream = kws_stream.get_or_insert_with(|| kws.create_stream());
                        // KWS is a few ms per chunk; run inline.
                        if let Some(keyword) = stream.accept(&samples) {
                            info!("[ws] wake word: {keyword} (seq={seq})");
                            send_event(&mut socket, &Event::Wake { keyword }).await;
                        }
                    }
                }
            }
            // Forward turn events while a turn is in flight.
            ev = async {
                match active.as_mut() {
                    Some(turn) => turn.rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match ev {
                    Some(ev) => {
                        if let Event::Done(ref d) = ev {
                            info!(
                                "[done] ttfb={:?}s sentences={} total={:?}s",
                                d.ttfb, d.sentences, d.timings.get("total")
                            );
                        }
                        let is_done = matches!(ev, Event::Done(_) | Event::Error { .. });
                        send_event(&mut socket, &ev).await;
                        if is_done {
                            active = None;
                        }
                    }
                    None => {
                        // Turn task ended (done/error/interrupted) without a
                        // terminal event.
                        active = None;
                    }
                }
            }
        }
    }
    // Connection closing: make sure a running turn stops.
    if let Some(turn) = active {
        turn.cancel.cancel();
    }
}
