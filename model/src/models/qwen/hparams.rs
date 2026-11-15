//! hparams.rs - the hyperparameters from the file's GGUF metadata, with the consistency checks the forward pass relies
//! on.

use format::meta::Meta;
use format::read::Model as File;
use format::{BlockKind, Result, err};

#[derive(Clone, Debug, PartialEq)]
pub struct Ssm {
    pub d_conv: u32,
    /// the state size (the head dimension of the recurrent state)
    pub ds: u32,
    /// the key heads (ssm.group_count)
    pub n_key_heads: u32,
    /// the value heads (ssm.time_step_rank)
    pub n_value_heads: u32,
    pub d_inner: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    pub arch: String,
    /// the layers the forward pass runs (the file's, less the MTP blocks at the end)
    pub n_layer: u32,
    pub n_mtp: u32,
    pub d_model: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub n_rot: u32,
    pub eps: f32,
    pub rope_base: f32,
    pub full_attention_interval: u32,
    pub n_expert: u32,
    pub n_used: u32,
    pub d_ff: u32,
    pub n_vocab: u32,
    pub ssm: Option<Ssm>,
    /// qwen3next: the DeltaNet projections are fused per key head (ssm_in, ssm_ba), and value head h reads key head h /
    /// rf; otherwise qwen35moe (four projections, and key head h % n_key_heads)
    pub next: bool,
    /// qwen4exp (Qwen3.8 Flash Next): the residual is `hc` streams mixed by hyper-connections (hc.rs), keys and values
    /// are cached at 16 bits, and the DeltaNet output is gated by a sigmoid; 0 for other models
    pub hc: u32,
    /// the mixers' low rank (hyper_connection.low_rank)
    pub hc_rank: u32,
    /// ernie4_5-moe: RoPE on adjacent pairs (2i, 2i + 1), llama.cpp's "norm" type, rather than NeoX's (i, i + n/2)
    pub rope_pairs: bool,
    /// ernie4_5-moe: keys and values cached at sixteen bits. Its q and k have no RMS norm, and with the int8 cache its
    /// log-probabilities drift several times further from llama.cpp's and greedy tokens change. Also deepseek2 (MLA)
    /// and lfm2moe: on LFM2.5-8B the int8 cache tips a near-tie expert choice and moves its log-probabilities well away
    /// from an f32 reference; only 6 of its 24 layers keep a cache
    pub kv16: bool,
    /// deepseek2: multi-head latent attention (mla.rs). head_dim is then the cache row (kv_lora + n_rot) and n_head_kv
    /// 1
    pub mla: Option<crate::mla::MlaDims>,
    /// the attention scores' scale when it is not 1 / sqrt(head_dim) (deepseek2: 1 / sqrt(key_length_mla))
    pub attn_scale: Option<f32>,
    /// deepseek2's expert gating: sigmoid (else softmax), renormalised when norm_w, times w_scale (moe.rs pick_sigmoid)
    pub gate_sigmoid: bool,
    pub norm_w: bool,
    pub w_scale: f32,
    /// lfm2moe: which layers are attention (a layer with 0 KV heads is a short convolution, rec.rs); None: every
    /// full_attention_interval-th layer
    pub attn_layers: Option<Vec<bool>>,
    /// lfm2moe: the short convolution's taps (shortconv.l_cache, 3); 0 for other models
    pub short_conv: u32,
}

const QK: u32 = 256;

impl Hparams {
    pub fn from_file(f: &File) -> Result<Hparams> {
        let m = Meta::parse(&f.metadata)?;
        let arch = m.str("general.architecture").unwrap_or("").to_string();
        let ok = crate::models::QWEN_ENGINE;
        if !ok.contains(&arch.as_str()) {
            return err(format!("architecture {arch} not supported yet ({})", ok.join(", ")));
        }
        // qwen3moe (Qwen3-30B-A3B and its kin) and ernie4_5-moe (ERNIE-4.5-21B-A3B): attention on every layer, no
        // DeltaNet
        let ernie = arch == "ernie4_5-moe";
        // deepseek2 (GLM-4.7-Flash, Moonlight, DeepSeek-V2-Lite): attention on every layer too, as MLA (mla.rs)
        let ds = arch == "deepseek2";
        // lfm2moe (LFM2 / LFM2.5-8B-A1B): attention layers and short-convolution layers, as its per-layer KV head count
        // says (0: a convolution), no DeltaNet
        let lfm2 = arch == "lfm2moe";
        let plain = arch == "qwen3moe" || ernie || ds || lfm2;
        let hc_arch = arch == "qwen4exp";
        let need = |k: &str| m.arch_u32(k).ok_or_else(|| format::Error(format!("missing {arch}.{k}")));
        let needf = |k: &str| m.arch_f32(k).ok_or_else(|| format::Error(format!("missing {arch}.{k}")));
        let blocks = need("block_count")?;
        let n_mtp = m.arch_u32("nextn_predict_layers").filter(|&n| n < blocks).unwrap_or(0);
        let n_layer = blocks - n_mtp;
        let d_model = need("embedding_length")?;
        let n_head = need("attention.head_count")?;
        let kv_per_layer = if lfm2 { m.ints(&m.arch_key("attention.head_count_kv")) } else { None };
        let attn_layers = kv_per_layer.as_ref().map(|v| v.iter().map(|&k| k > 0).collect::<Vec<bool>>());
        let mut n_head_kv = if ds {
            1
        } else if let Some(v) = &kv_per_layer {
            v.iter().copied().max().unwrap_or(0) as u32
        } else {
            need("attention.head_count_kv")?
        };
        // ernie4_5-moe gives no key length: the heads split the model's width
        let mut head_dim = match m.arch_u32("attention.key_length") {
            Some(v) => v,
            None if ernie || lfm2 => d_model / n_head.max(1),
            None => need("attention.key_length")?,
        };
        let n_rot = m.arch_u32("rope.dimension_count").unwrap_or(head_dim);
        // deepseek2: the cache row is the latent and the rotary key (kv_lora + n_rot), one KV head; the scores are
        // scaled by the full key width of a head (key_length_mla: its no-rope part and its rope part)
        let (mut mla, mut attn_scale) = (None, None);
        let (mut gate_sigmoid, mut norm_w, mut w_scale) = (false, true, 1.0f32);
        if ds {
            if m.arch_f32("rope.scaling.factor").is_some_and(|f| f > 1.0) {
                return err("deepseek2 with YaRN rope scaling: not supported yet");
            }
            let kv_lora = need("attention.kv_lora_rank")?;
            let key_mla = m.arch_u32("attention.key_length_mla").unwrap_or(head_dim);
            let v_mla = m.arch_u32("attention.value_length_mla").unwrap_or(head_dim);
            if key_mla <= n_rot {
                return err("deepseek2: the key is no wider than its rotary part");
            }
            mla = Some(crate::mla::MlaDims {
                q_lora: m.arch_u32("attention.q_lora_rank").unwrap_or(0) as usize,
                kv_lora: kv_lora as usize,
                nope: (key_mla - n_rot) as usize,
                vd: v_mla as usize,
            });
            attn_scale = Some(1.0 / (key_mla as f32).sqrt());
            head_dim = kv_lora + n_rot;
            n_head_kv = 1;
            // llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE: 1 softmax, 2 sigmoid; unset is softmax except GLM 4.7 Lite
            let n_vocab = m.len_of("tokenizer.ggml.tokens").unwrap_or(0);
            gate_sigmoid = match m.arch_u32("expert_gating_func") {
                Some(g) => g == 2,
                None => (blocks == 47 || blocks == 48) && n_vocab == 154880,
            };
            norm_w = m.arch_u32("expert_weights_norm").is_some_and(|v| v != 0);
            w_scale = m.arch_f32("expert_weights_scale").filter(|&v| v != 0.0).unwrap_or(1.0);
        }
        if lfm2 {
            // llama.cpp's lfm2moe: the gating its metadata names (2: sigmoid), the weights always renormalised
            gate_sigmoid = m.arch_u32("expert_gating_func") == Some(2);
            norm_w = true;
            w_scale = m.arch_f32("expert_weights_scale").filter(|&v| v != 0.0).unwrap_or(1.0);
        }
        let full_attention_interval = if plain { 1 } else { m.arch_u32("full_attention_interval").unwrap_or(4) };
        let ssm = if plain {
            None
        } else {
            Some(Ssm {
                d_conv: need("ssm.conv_kernel")?,
                ds: need("ssm.state_size")?,
                n_key_heads: need("ssm.group_count")?,
                n_value_heads: need("ssm.time_step_rank")?,
                d_inner: need("ssm.inner_size")?,
            })
        };
        // the routed experts' record (ernie4_5-moe's first layer is a dense FFN stored as a one-expert pack, before
        // them)
        let g = f
            .experts
            .iter()
            .find(|x| x.n_expert > 1)
            .or(f.experts.first())
            .ok_or_else(|| format::Error("no routed experts".into()))?;
        // qwen4exp's experts keep gate's and up's rows as in the file (640), while down is widened by the converter:
        // this is the width the activations are padded to, 256 at a time
        let d_ff = if hc_arch { g.width.div_ceil(QK) * QK } else { g.width };
        let (hc, hc_rank) = if hc_arch {
            let c = need("hyper_connection.count")?;
            let r = need("hyper_connection.low_rank")?;
            if !(2..=8).contains(&c) || r == 0 {
                return err("hyper-connections: missing or inconsistent");
            }
            (c, r)
        } else {
            (0, 0)
        };
        let n_vocab = m.len_of("tokenizer.ggml.tokens").map(|n| n as u32).unwrap_or(0);
        let h = Hparams {
            arch: arch.clone(),
            n_layer,
            n_mtp,
            d_model,
            n_head,
            n_head_kv,
            head_dim,
            n_rot,
            eps: needf("attention.layer_norm_rms_epsilon")?,
            rope_base: needf("rope.freq_base")?,
            full_attention_interval,
            n_expert: g.n_expert,
            n_used: g.n_used,
            d_ff,
            n_vocab,
            ssm,
            next: arch == "qwen3next",
            hc,
            hc_rank,
            rope_pairs: ernie || ds,
            kv16: ernie || ds || lfm2,
            mla,
            attn_scale,
            gate_sigmoid,
            norm_w,
            w_scale,
            attn_layers,
            short_conv: if lfm2 { need("shortconv.l_cache")? } else { 0 },
        };
        // refuse at load a file the forward pass can't run, naming the first reason found
        let problems = [
            (
                h.n_head_kv == 0 || !h.n_head.is_multiple_of(h.n_head_kv),
                "the query heads don't divide evenly among the KV heads",
            ),
            (!h.head_dim.is_multiple_of(8), "the head size isn't a multiple of 8 (attention works in 8-wide lanes)"),
            (
                h.n_rot == 0 || !h.n_rot.is_multiple_of(2) || h.n_rot > h.head_dim,
                "the rotated width must be even, above 0 and within the head",
            ),
            (h.full_attention_interval == 0, "full_attention_interval is 0"),
            (!h.d_model.is_multiple_of(QK), "the embedding width isn't a whole number of quantisation blocks"),
            (
                h.mla.is_none() && !(h.n_head * h.head_dim).is_multiple_of(QK),
                "the attention width isn't a whole number of quantisation blocks",
            ),
            (
                h.mla.is_some_and(|d| {
                    !(h.n_head as usize * d.vd).is_multiple_of(QK as usize) || (d.q_lora % QK as usize) != 0
                }),
                "the MLA value heads or q LoRA aren't a whole number of quantisation blocks",
            ),
            (!h.d_ff.is_multiple_of(QK), "the expert width isn't a whole number of quantisation blocks"),
        ];
        if let Some((_, why)) = problems.iter().find(|p| p.0) {
            return err(format!("can't run this model: {why}"));
        }
        // the blocks must agree with the interval: attention on every `interval`-th layer, and DeltaNet otherwise
        for b in f.blocks.iter().filter(|b| b.layer < n_layer) {
            let k = Some(b.kind);
            let attn = h.is_attention(b.layer);
            if (k == Some(BlockKind::AttentionFull) && !attn) || (k == Some(BlockKind::DeltaNet) && attn) {
                return err(format!("layer {}: block kind disagrees with full_attention_interval", b.layer));
            }
        }
        Ok(h)
    }

    pub fn is_attention(&self, layer: u32) -> bool {
        match &self.attn_layers {
            Some(v) => v.get(layer as usize).copied().unwrap_or(false),
            None => (layer + 1).is_multiple_of(self.full_attention_interval),
        }
    }
}
