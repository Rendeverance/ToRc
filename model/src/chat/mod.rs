//! chat - text in and out: the tokenizers (tok.rs, with the Unicode tables in uniflags.rs), the chat APIs the
//! servers speak (api/: Anthropic Messages and the OpenAI Responses API), what torc-server asks of a model
//! (serve.rs), and the saved chats of the families whose state is a positional cache (saved.rs). Each model's own
//! side of a chat (its prompt format, its tool calls, its server half) lives in its folder under models/; what those
//! formats share is here: format.rs (a message's text and calls, JSON as the templates write it) and json_calls.rs
//! (Qwen2.5's JSON tool format, for the models whose templates have none), and what the server is doing while a reply
//! runs (status.rs, the page at / and the counters it reads).

pub mod api;
pub mod format;
pub mod json_calls;
pub mod saved;
pub mod serve;
pub mod status;
pub mod tok;
pub(crate) mod uniflags;
