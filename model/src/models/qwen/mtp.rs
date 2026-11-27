//! mtp.rs - the multi-token-prediction block (qwen35moe's layer after the last one, `blk.<n_layer>.*`).
//!
//! Each row joins a token's embedding with the main model's hidden state after its final norm, as `[rms(embed(t),
//! enorm); rms(hid, hnorm)]`, projects them back to d_model (`nextn.eh_proj`), runs one gated attention layer with its
//! own KV cache and one MoE layer, and normalises (`nextn.shared_head_norm`). The result is the block's hidden state,
//! and its vocabulary product over the draft head gives a draft.
//!
//! The draft head is the first DRAFT_V rows of output.weight, followed by up to MAX_NOTED "noted" rows: the ids past
//! DRAFT_V seen in the prompt and the reply, in first-seen order, copied out of output.weight as they are noted. Index
//! i of a draft logit row is token draft_id(i).
//!
//! Drafts only decide which tokens the main model scores in a verify pass, never the output, so this block need not be
//! bit-exact against anything (although it uses the same kernels anyway).

use crate::attn::{AttnEngine, AttnLayer, KvCache, attn_block};
use crate::hparams::Hparams;
use crate::moe::{Mat, MoeLayer, apply_group_multi, moe_tokens};
use crate::ops::rms_norm;
use crate::weights::Weights;
use format::{BlockKind, Result, err};
use kernels::{Q8, Q8m, QK, ops as kops};

/// The draftable token ids: 0..DRAFT_V (a 32k draft head).
pub const DRAFT_V: usize = 32768;
/// The most noted ids past DRAFT_V the draft head takes.
pub const MAX_NOTED: usize = 8192;

pub struct MtpBlock {
    /// the block's own file (for qwen3next, a separate file of one layer), else None (the main file's layer n_layer,
    /// nextn.*); the embedding and the head are the main model's either way
    own: Option<Weights>,
    /// eh_proj's name in the block's file
    eh: String,
    attn: AttnLayer,
    kv: KvCache,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    moe: MoeLayer,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    head_norm: Vec<f32>,
    /// the noted ids (>= DRAFT_V, in first-seen order) and their output.weight rows, one after another
    noted: Vec<u32>,
    noted_rows: Vec<u8>,
    /// is_noted[t]: whether t is in `noted` (ids below DRAFT_V are never noted)
    is_noted: Vec<bool>,
}

impl MtpBlock {
    /// The block's attention cache (part of a server's saved state, see Model::state_bytes).
    pub(crate) fn kv(&mut self) -> &mut KvCache {
        &mut self.kv
    }

    /// The block, if there is one: layer h.n_layer of the model's own file when it has `nextn.eh_proj`, else the file
    /// TORC_MTP_FILE names, if set.
    pub fn load(w: &Weights, h: &Hparams, n_ctx: usize) -> Result<Option<MtpBlock>> {
        let l = h.n_layer;
        if w.tensor(&format!("blk.{l}.nextn.eh_proj.weight")).is_some() {
            return MtpBlock::from_layer(w, w, l, h, n_ctx).map(Some);
        }
        let Some(path) = crate::switches::get("TORC_MTP_FILE") else { return Ok(None) };
        // a file of the block alone: its one layer, 0, laid out and named as the layer is in a model that carries it
        let wm = Weights::load(&path)?;
        if wm.file.blocks.iter().any(|b| b.layer != 0) || wm.tensor("blk.0.nextn.eh_proj.weight").is_none() {
            return err(format!("{path}: TORC_MTP_FILE wants a file of one layer, 0, with blk.0.nextn.eh_proj"));
        }
        // its metadata is copied from the main model, so only its experts tell whether it belongs to this one
        if !wm.file.experts.first().is_some_and(|g| g.n_expert == h.n_expert && g.n_used == h.n_used) {
            return err(format!("{path}: its experts are not {} choose {} as the model's are", h.n_expert, h.n_used));
        }
        let mut b = MtpBlock::from_layer(&wm, w, 0, h, n_ctx)?;
        // the routes are counted as layer n_layer's (the NPU's hot-expert record), whatever the layer is in its file
        b.moe.layer = h.n_layer;
        b.own = Some(wm);
        Ok(Some(b))
    }

    /// The block from layer l of `src`, with the head of the main model `w`.
    fn from_layer(src: &Weights, w: &Weights, l: u32, h: &Hparams, n_ctx: usize) -> Result<MtpBlock> {
        let group = src
            .file
            .blocks
            .iter()
            .find(|b| b.layer == l && b.kind == BlockKind::Moe && b.experts.is_some())
            // a dense model (qwen35): its FFN as the layer's one expert, recorded on the mixer's block
            .or_else(|| src.file.blocks.iter().find(|b| b.layer == l && b.experts.is_some()))
            .and_then(|b| b.experts)
            .ok_or_else(|| format::Error(format!("MTP layer {l}: no routed experts")))?;
        Ok(MtpBlock {
            own: None,
            eh: format!("blk.{l}.nextn.eh_proj.weight"),
            attn: AttnLayer::load(src, l)?,
            kv: KvCache::new(h, n_ctx),
            attn_norm: src.layer(l, "attn_norm.weight")?.f32s()?,
            ffn_norm: src.layer(l, "post_attention_norm.weight")?.f32s()?,
            moe: MoeLayer::load(src, l, group)?,
            enorm: src.layer(l, "nextn.enorm.weight")?.f32s()?,
            hnorm: src.layer(l, "nextn.hnorm.weight")?.f32s()?,
            head_norm: src.layer(l, "nextn.shared_head_norm.weight")?.f32s()?,
            noted: Vec::new(),
            noted_rows: Vec::new(),
            is_noted: vec![false; w.head()?.shape().1],
        })
    }

    /// Notes the ids seen in the prompt or the reply: each id past DRAFT_V not yet noted joins the draft head while it
    /// has room. Ids outside the vocabulary are skipped.
    pub fn note(&mut self, w: &Weights, ids: &[u32]) -> Result<()> {
        let out = w.head()?;
        let rb = out.row_bytes();
        for &t in ids {
            let t = t as usize;
            if t < DRAFT_V || t >= self.is_noted.len() || self.is_noted[t] || self.noted.len() == MAX_NOTED {
                continue;
            }
            self.noted_rows.extend_from_slice(&out.data[t * rb..(t + 1) * rb]);
            self.noted.push(t as u32);
            self.is_noted[t] = true;
        }
        Ok(())
    }

    /// The width of a draft logit row: DRAFT_V plus the noted ids.
    pub fn draft_width(&self) -> usize {
        DRAFT_V + self.noted.len()
    }

    /// The token id of draft logit index i (i < draft_width()).
    pub fn draft_id(&self, i: usize) -> u32 {
        if i < DRAFT_V { i as u32 } else { self.noted[i - DRAFT_V] }
    }

    /// Rows j = 0..n at positions pos0 + j: token tokens[j] with the main hidden state hid[j] (out-normed, d_model
    /// each). Returns the block's hidden rows, and `logits` (empty, or draft_width() long) gets the last row's draft
    /// logits - the only row a draft is taken from, so the head is not run over the other rows.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        w: &Weights,
        h: &Hparams,
        eng: &AttnEngine,
        tokens: &[u32],
        hid: &[f32],
        pos0: u32,
        logits: &mut [f32],
    ) -> Result<Vec<f32>> {
        self.run(w, h, eng, tokens, hid, pos0, logits, false, false)
    }

    /// Only the rows' keys and values: the attention cache for positions pos0.., for later rows to read, without the
    /// MoE or the head (nothing after the cache is read for these rows). This is for a server's prompt rows, where only
    /// the last row's draft is wanted, and its attention reads these.
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &mut self,
        w: &Weights,
        h: &Hparams,
        eng: &AttnEngine,
        tokens: &[u32],
        hid: &[f32],
        pos0: u32,
    ) -> Result<()> {
        // the keys and values only (attn_fill_kv); TORC_MTP_FILL_FULL=1 runs the whole block
        static FULL: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_MTP_FILL_FULL"));
        // TORC_MTP_FILLHASH=1 (a diagnostic): both fills run on the same rows, with the cache hashed after each, to
        // prove the short fill writes the same bits (the full one runs last, so the cache ends up as the full one's
        // either way)
        if crate::switches::on("TORC_MTP_FILLHASH") {
            self.run(w, h, eng, tokens, hid, pos0, &mut [], true, true)?;
            let a = self.cache_hash();
            self.run(w, h, eng, tokens, hid, pos0, &mut [], true, false)?;
            let b = self.cache_hash();
            eprintln!(
                "mtp fill: {} rows from {pos0}, cache hash keys-and-values {a:016x}, full {b:016x}: {}",
                tokens.len(),
                if a == b { "SAME" } else { "DIFFERENT" }
            );
            return Ok(());
        }
        self.run(w, h, eng, tokens, hid, pos0, &mut [], true, !*FULL).map(|_| ())
    }

    /// FNV-1a over the whole attention cache (TORC_MTP_FILLHASH).
    fn cache_hash(&self) -> u64 {
        let mut hs = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |b: &[u8]| {
            for &x in b {
                hs = (hs ^ x as u64).wrapping_mul(0x100_0000_01b3);
            }
        };
        let kv = &self.kv;
        // SAFETY: plain integer and float arrays viewed as bytes
        unsafe {
            eat(std::slice::from_raw_parts(kv.k.as_ptr() as *const u8, kv.k.len()));
            eat(std::slice::from_raw_parts(kv.v.as_ptr() as *const u8, kv.v.len()));
            eat(std::slice::from_raw_parts(kv.k16.as_ptr() as *const u8, kv.k16.len() * 2));
            eat(std::slice::from_raw_parts(kv.v16.as_ptr() as *const u8, kv.v16.len() * 2));
            eat(std::slice::from_raw_parts(kv.k32.as_ptr() as *const u8, kv.k32.len() * 4));
            eat(std::slice::from_raw_parts(kv.v32.as_ptr() as *const u8, kv.v32.len() * 4));
            eat(std::slice::from_raw_parts(kv.k_scale.as_ptr() as *const u8, kv.k_scale.len() * 4));
            eat(std::slice::from_raw_parts(kv.v_scale.as_ptr() as *const u8, kv.v_scale.len() * 4));
        }
        hs
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        w: &Weights,
        h: &Hparams,
        eng: &AttnEngine,
        tokens: &[u32],
        hid: &[f32],
        pos0: u32,
        logits: &mut [f32],
        fill: bool,
        kv_only: bool,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        let dm = h.d_model as usize;
        let (nb, nbc) = (dm / QK, 2 * dm / QK);
        let wd = self.draft_width();
        if n == 0 || hid.len() != n * dm || (!logits.is_empty() && logits.len() != wd) {
            return err("mtp: bad buffer sizes");
        }
        let emb = w.need("token_embd.weight")?;
        let rb = emb.row_bytes();
        let mut e = vec![0.0f32; dm];
        let mut cat = vec![0.0f32; 2 * dm];
        let mut catq = Q8::new(n * nbc);
        for (j, &t) in tokens.iter().enumerate() {
            if t as usize >= emb.shape().1 {
                return err("mtp: token out of range");
            }
            kops::dequantize(emb.t.ty, &emb.data[t as usize * rb..(t as usize + 1) * rb], &mut e);
            rms_norm(&mut cat[..dm], &e, &self.enorm, h.eps);
            rms_norm(&mut cat[dm..], &hid[j * dm..(j + 1) * dm], &self.hnorm, h.eps);
            kops::q8k_quantize(&cat, catq.sub_mut(j * nbc..(j + 1) * nbc));
        }
        let mut x = vec![0.0f32; n * dm];
        let mw = self.own.as_ref().unwrap_or(w);
        apply_group_multi(&mut [(Mat::of(&mw.need(&self.eh)?), &mut x[..])], catq.view(), n);
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(n * nb);
        let norm_all = |xn: &mut [f32], mut xq: Q8m, x: &[f32], wn: &[f32]| {
            for j in 0..n {
                rms_norm(&mut xn[j * dm..(j + 1) * dm], &x[j * dm..(j + 1) * dm], wn, h.eps);
                kops::q8k_quantize(&xn[j * dm..(j + 1) * dm], xq.sub(j * nb..(j + 1) * nb));
            }
        };
        norm_all(&mut xn, xq.view_mut(), &x, &self.attn_norm);
        // a fill only wants the cache rows: their keys and values, with the same bits, without the queries, the
        // attention over the prompt and the output
        if fill && kv_only {
            crate::attn::attn_fill_kv(mw, h, &self.attn, &mut self.kv, pos0, &xn, xq.view(), n)?;
            return Ok(Vec::new());
        }
        attn_block(mw, h, &self.attn, eng, &mut self.kv, pos0, &xn, xq.view(), &mut x, n)?;
        if fill {
            return Ok(Vec::new());
        }
        norm_all(&mut xn, xq.view_mut(), &x, &self.ffn_norm);
        moe_tokens(mw, &self.moe, &xn, xq.view(), &mut x, n)?;
        norm_all(&mut xn, xq.view_mut(), &x, &self.head_norm);
        if !logits.is_empty() {
            let out = w.head()?;
            let orb = out.row_bytes();
            let head = Mat { ty: out.t.ty, rows: DRAFT_V, cols: dm, data: &out.data[..DRAFT_V * orb] };
            let k = self.noted.len();
            let last = xq.sub((n - 1) * nb..);
            if k == 0 {
                apply_group_multi(&mut [(head, logits)], last, 1);
            } else {
                // both products in one sweep, straight into their parts of the row
                let noted = Mat { ty: out.t.ty, rows: k, cols: dm, data: &self.noted_rows };
                let (lh, ln) = logits.split_at_mut(DRAFT_V);
                apply_group_multi(&mut [(head, lh), (noted, ln)], last, 1);
            }
        }
        Ok(xn)
    }
}
