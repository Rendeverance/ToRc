//! inkling - Inkling 2 Mini (Thinking Machines' looped MoE model): the model (inkling.rs), its chat format
//! (chat.rs) and its server half (serve.rs).

pub mod chat;
#[allow(clippy::module_inception)]
pub mod inkling;
pub mod serve;
