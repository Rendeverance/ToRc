//! blocks.rs - a model's structure, found from its tensor names (the file does not store it): for each layer, the
//! blocks it runs, and for an MoE block, which EXPERTS record holds its experts.

use crate::Directory;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockKind {
    /// attention over every earlier position
    AttentionFull,
    /// attention over a sliding window (the reader does not tell this from full: the model's metadata does)
    AttentionSwa,
    /// gated DeltaNet (linear attention with a state per head)
    DeltaNet,
    Mamba2,
    /// routed experts
    Moe,
    /// one feed-forward network for every token
    DenseFfn,
    SharedExpert,
    /// hyper-connection mixing between the residual streams
    HcMix,
    /// the multi-token-prediction block
    Mtp,
    /// attention without rotary position (Nemotron-H)
    AttentionNope,
    /// a layer with none of the above
    Custom,
}

/// One block of one layer. `experts` indexes Directory.experts for an MoE block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub kind: BlockKind,
    pub layer: u32,
    pub experts: Option<u32>,
}

/// The blocks of layer l, given the names of its tensors with the `blk.<l>.` prefix taken off.
fn kinds_of(arch: &str, names: &[&str]) -> Vec<BlockKind> {
    // a tensor is `<stem>.weight`, `<stem>.bias`, or a bare stem (ssm_a, ssm_d)
    let stem = |n: &str| n.strip_suffix(".weight").or_else(|| n.strip_suffix(".bias")).unwrap_or(n).to_string();
    let stems: Vec<String> = names.iter().map(|n| stem(n)).collect();
    let has = |s: &str| stems.iter().any(|x| x == s);
    let starts = |s: &str| names.iter().any(|n| n.starts_with(s));
    let mut k = Vec::new();
    if starts("hc_") {
        k.push(BlockKind::HcMix);
    }
    if has("ssm_beta_alpha") || has("ssm_beta") || has("ssm_ba") {
        k.push(BlockKind::DeltaNet);
    } else if has("ssm_in") || has("ssm_a") {
        k.push(BlockKind::Mamba2);
    } else if has("attn_q") || has("attn_qkv") || has("attn_q_a") {
        k.push(if arch == "nemotron_h_moe" { BlockKind::AttentionNope } else { BlockKind::AttentionFull });
    }
    if has("ffn_gate_inp") {
        k.push(BlockKind::Moe);
    } else if has("ffn_up") || has("ffn_down") {
        k.push(BlockKind::DenseFfn);
    }
    if starts("nextn.") {
        k.push(BlockKind::Mtp);
    }
    if k.is_empty() {
        k.push(BlockKind::Custom);
    }
    k
}

/// Every layer's blocks, in layer order. The layers are those named by a resident tensor or an EXPERTS record.
pub fn blocks(arch: &str, d: &Directory) -> Vec<Block> {
    let mut by_layer: std::collections::BTreeMap<u32, Vec<&str>> = Default::default();
    for t in &d.tensors {
        if let Some((l, rest)) = t.name.strip_prefix("blk.").and_then(|s| s.split_once('.'))
            && let Ok(l) = l.parse::<u32>()
        {
            by_layer.entry(l).or_default().push(rest);
        }
    }
    for x in &d.experts {
        by_layer.entry(x.layer).or_default();
    }
    let mut out = Vec::new();
    for (&l, names) in &by_layer {
        let moe = d.experts.iter().position(|x| x.layer == l).map(|i| i as u32);
        let kinds = kinds_of(arch, names);
        let routed = kinds.contains(&BlockKind::Moe);
        for &kind in &kinds {
            out.push(Block { kind, layer: l, experts: if kind == BlockKind::Moe { moe } else { None } });
        }
        // an EXPERTS record without a router: a dense feed-forward stored as the layer's one expert (convert.rs
        // dense_of), whose matrices are in the pack rather than resident
        if moe.is_some() && !routed {
            out.retain(|b| !(b.layer == l && b.kind == BlockKind::Custom));
            out.push(Block { kind: BlockKind::DenseFfn, layer: l, experts: moe });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        let k = |n: &[&str]| kinds_of("qwen35moe", n);
        assert_eq!(k(&["attn_q.weight", "ffn_gate_inp.weight"]), [BlockKind::AttentionFull, BlockKind::Moe]);
        assert_eq!(k(&["ssm_ba.weight", "ssm_a", "ffn_gate_inp.weight"]), [BlockKind::DeltaNet, BlockKind::Moe]);
        assert_eq!(k(&["ssm_in.weight", "ssm_a"]), [BlockKind::Mamba2]);
        assert_eq!(
            k(&["attn_qkv.weight", "ffn_up.weight", "ffn_down.weight"]),
            [BlockKind::AttentionFull, BlockKind::DenseFfn]
        );
        assert_eq!(
            k(&["hc_fn.weight", "attn_q.weight", "ffn_gate_inp.weight", "nextn.eh_proj.weight"]),
            [BlockKind::HcMix, BlockKind::AttentionFull, BlockKind::Moe, BlockKind::Mtp]
        );
        assert_eq!(kinds_of("nemotron_h_moe", &["attn_q.weight"]), [BlockKind::AttentionNope]);
        assert_eq!(k(&["something_else.weight"]), [BlockKind::Custom]);
    }

    /// A dense layer whose FFN is a one-expert record (no router): its DenseFfn block points at the record.
    #[test]
    fn dense_as_one_expert() {
        let t = |n: &str| crate::Tensor { name: n.into(), ..Default::default() };
        let x = crate::Experts { layer: 1, n_expert: 1, n_used: 1, ..Default::default() };
        let d = crate::Directory {
            metadata: Vec::new(),
            source: None,
            tensors: vec![t("blk.0.attn_qkv.weight"), t("blk.0.ffn_up.weight"), t("blk.1.ssm_ba.weight")],
            experts: vec![x],
        };
        let b = blocks("qwen35", &d);
        assert_eq!(
            b,
            [
                Block { kind: BlockKind::AttentionFull, layer: 0, experts: None },
                Block { kind: BlockKind::DenseFfn, layer: 0, experts: None },
                Block { kind: BlockKind::DeltaNet, layer: 1, experts: None },
                Block { kind: BlockKind::DenseFfn, layer: 1, experts: Some(0) }
            ]
        );
    }
}
