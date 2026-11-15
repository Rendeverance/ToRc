//! moonlight - Moonlight's chat format (chat.rs, Kimi's turns), with tool calls in Qwen2.5's JSON format, as its own
//! template has none. The model's network, deepseek2 with MLA, runs on the qwen engine (models/qwen), which picks this
//! format from the file's metadata.

pub mod chat;
