//! gptoss - gpt-oss (OpenAI's open-weight MoE models, 20B and 120B): the model (gptoss.rs), its chat format
//! (chat.rs) and its server half (serve.rs).

pub mod chat;
#[allow(clippy::module_inception)]
pub mod gptoss;
pub mod serve;
