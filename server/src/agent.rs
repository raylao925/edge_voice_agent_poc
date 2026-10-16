use anyhow::Result;
use tokio::sync::mpsc;

use crate::protocol::Event;

/// Output of the agent stage: extra context for the reply LLM call,
/// plus the names of tools that ran (surfaced in the `done` summary).
pub struct AgentOutput {
    pub context: String,
    pub tool_calls: Vec<String>,
}

/// Two-stage agent loop (mirrors `pipeline.py::agent_prepare`):
///  1. decide intent + which tools to run (LLM, or keyword fallback),
///  2. run tools, fold results into `context`.
///
/// Currently a pass-through; tools get wired in the next step.
pub async fn agent_prepare(
    _user_text: &str,
    status_tx: &mpsc::Sender<Event>,
) -> Result<AgentOutput> {
    let _ = status_tx; // used once tools report progress via Status events
    Ok(AgentOutput {
        context: String::new(),
        tool_calls: vec![],
    })
}
