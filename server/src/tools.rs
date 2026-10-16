use anyhow::Result;
use async_trait::async_trait;

/// A callable tool (web search, news, weather, ...). The agent decides when
/// to call them; results are folded back into the LLM context.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    async fn run(&self, query: &str) -> Result<String>;
}

/// TODO (next step): port `tools.py` here —
///   - `ddgs` web search + news search  -> reqwest against the ddgs endpoints
///   - Open-Meteo weather               -> plain reqwest GET, no key needed
///   - Jina Reader fetch_page           -> reqwest GET https://r.jina.ai/<url>
///
/// And the two-stage loop from `pipeline.py::agent_prepare`:
///   1. LLM classifies intent + picks tools (or keyword fallback, no key)
///   2. run tools, concatenate results into `context`
pub struct ToolRegistry {
    #[allow(dead_code)]
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn empty() -> Self {
        Self { tools: vec![] }
    }
}
