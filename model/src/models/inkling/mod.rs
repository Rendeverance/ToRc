//! inkling - Inkling 2 Mini (Thinking Machines' looped MoE model): the model (inkling.rs), its chat format
//! (chat.rs), its server half (serve.rs) and its two-machine pipeline (pipe.rs).

pub mod chat;
#[allow(clippy::module_inception)]
pub mod inkling;
pub mod pipe;
pub mod serve;
