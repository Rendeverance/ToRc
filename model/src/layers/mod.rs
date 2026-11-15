//! layers - the building blocks every model is made of: attention (attn.rs, with its kernel fattn.rs), the routed
//! experts (moe.rs), gated DeltaNet (rec.rs), and the activation functions and row operations they share (act.rs,
//! ops.rs).

pub mod act;
pub mod attn;
pub mod fattn;
pub mod mla;
pub mod moe;
pub mod ops;
pub mod rec;
