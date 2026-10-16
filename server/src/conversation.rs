//! Per-connection conversation memory (in-memory, MVP).
//!
//! Each WebSocket connection owns one `Conversation`. Turns are appended as
//! they complete; only the last `max_turns` are kept. Later: persist to
//! SQLite + summarize for long-term memory.

use std::collections::VecDeque;

/// One completed turn.
#[derive(Debug, Clone)]
pub struct Turn {
    pub user_text: String,
    pub reply: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMsg {
    pub role: Role,
    pub content: String,
}

pub struct Conversation {
    turns: VecDeque<Turn>,
    max_turns: usize,
}

impl Conversation {
    pub fn new(max_turns: usize) -> Self {
        Self {
            turns: VecDeque::new(),
            max_turns,
        }
    }

    pub fn push(&mut self, user_text: String, reply: String) {
        if self.max_turns == 0 || user_text.trim().is_empty() {
            return;
        }
        while self.turns.len() >= self.max_turns {
            self.turns.pop_front();
        }
        self.turns.push_back(Turn { user_text, reply });
    }

    pub fn clear(&mut self) {
        self.turns.clear();
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Most recent turn, used as reference for intent classification.
    pub fn last(&self) -> Option<Turn> {
        self.turns.back().cloned()
    }

    /// Flattened history, oldest first: user, assistant, user, assistant, ...
    pub fn history(&self) -> Vec<ChatMsg> {
        let mut out = Vec::with_capacity(self.turns.len() * 2);
        for t in &self.turns {
            out.push(ChatMsg {
                role: Role::User,
                content: t.user_text.clone(),
            });
            out.push(ChatMsg {
                role: Role::Assistant,
                content: t.reply.clone(),
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_beyond_max() {
        let mut c = Conversation::new(2);
        c.push("u1".into(), "a1".into());
        c.push("u2".into(), "a2".into());
        c.push("u3".into(), "a3".into());
        assert_eq!(c.len(), 2);
        let h = c.history();
        assert_eq!(h[0].content, "u2");
        assert_eq!(h[3].content, "a3");
    }

    #[test]
    fn skips_empty_user_text() {
        let mut c = Conversation::new(10);
        c.push("   ".into(), "a".into());
        assert_eq!(c.len(), 0);
    }
}
