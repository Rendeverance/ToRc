//! mtp_hc.rs - qwen4exp's multi-token-prediction block (Flash Next).
//!
//! The block is a full hc layer, `blk.<n_layer>.*` (its two mixers, attention with the QSA indexer, and a MoE with its
//! own experts), plus `nextn.{eh_proj, enorm, hnorm, hc_head_*}`, in its own file (TORC_MTP_FILE).
//!
//! Each row joins a token's embedding with one of the main model's streams before its final mixer: for token j and
//! stream s, `[norm(embed(t_j), enorm); norm(hid[j][s], hnorm[s])]` goes through eh_proj back to d_model, giving the
//! block's stream s for row j. The layer then runs as any hc layer does (mix, block, combine, twice), its streams
//! become the block's hidden state for the next draft, and the head is the block's own final mixer followed by a
//! vocabulary product.
//!
//! The head: the full vocabulary is 248320 rows, about 675 MB per draft at Q8_0, which costs more than a draft saves.
//! The default is therefore the qwen35moe draft head from mtp.rs (the first DRAFT_V rows of output.weight, plus the ids
//! past them noted in the prompt and the reply); TORC_MTP_HEAD=full uses the whole vocabulary. Drafts only decide which
//! tokens the main model scores in a verify pass, never the output.

use crate::attn::{AttnEngine, AttnLayer, KvCache, attn_block, qsa_prepare_mtp};
use crate::hc::{HcBufs, LayerHc, Mixer, combine, hc_mix, norm_gg, prep};
use crate::hparams::Hparams;
use crate::moe::{Mat, MoeLayer, apply_group_multi, moe_tokens};
use crate::mtp::{DRAFT_V, MAX_NOTED};
use crate::weights::Weights;
use format::{BlockKind, Result, err};
use kernels::{Q8, QK, ops as kops};

pub struct MtpHc {
    /// the block's file (its tensors are named for layer `l`, while its blocks and experts are the file's layer 0)
    own: Weights,
    l: u32,
    attn: AttnLayer,
    kv: KvCache,
    moe: MoeLayer,
    mix: LayerHc,
    head: Mixer,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    bufs: HcBufs,
    /// TORC_MTP_HEAD=full: the whole vocabulary; otherwise the draft head (with noted ids, as in mtp.rs)
    full: bool,
    noted: Vec<u32>,
    noted_rows: Vec<u8>,
    is_noted: Vec<bool>,
}

impl MtpHc {
    /// The block's attention cache (part of a server's saved state, see Model::state_bytes).
    pub(crate) fn kv(&mut self) -> &mut KvCache {
        &mut self.kv
    }

    /// The block from TORC_MTP_FILE, if it is set, checked against the main model (the same architecture, the model's
    /// experts and d_model, and eh_proj of 2 d_model x d_model).
    pub fn load(w: &Weights, h: &Hparams, n_ctx: usize) -> Result<Option<MtpHc>> {
        let Some(path) = crate::switches::get("TORC_MTP_FILE") else { return Ok(None) };
        // mapped whole, and never given an expert cache of its own (TORC_CACHE_MB is the main model's)
        let wm = Weights::load_with(&path, false)?;
        let l = h.n_layer;
        let dm = h.d_model as usize;
        let d = h.hc as usize * dm;
        let eh = format!("blk.{l}.nextn.eh_proj.weight");
        if !wm.tensor(&eh).is_some_and(|t| t.shape() == (2 * dm, dm)) {
            return err(format!("{path}: TORC_MTP_FILE wants {eh} of {} x {dm}", 2 * dm));
        }
        if !wm.file.experts.first().is_some_and(|g| g.n_expert == h.n_expert && g.n_used == h.n_used) {
            return err(format!("{path}: its experts are not {} choose {} as the model's are", h.n_expert, h.n_used));
        }
        let group = wm
            .file
            .blocks
            .iter()
            .find(|b| b.kind == BlockKind::Moe && b.experts.is_some())
            .and_then(|b| b.experts)
            .ok_or_else(|| format::Error(format!("{path}: no routed experts")))?;
        let mut moe = MoeLayer::load(&wm, l, group)?;
        moe.hc = true;
        let mut attn = AttnLayer::load(&wm, l)?;
        attn.load_indexer(&wm, h)?;
        let f = |s: &str, len: usize| -> Result<Vec<f32>> {
            let v = wm.layer(l, s)?.f32s()?;
            if v.len() != len { err(format!("tensor blk.{l}.{s}: unexpected size")) } else { Ok(v) }
        };
        let (enorm, hnorm) = (f("nextn.enorm.weight", dm)?, f("nextn.hnorm.weight", d)?);
        let mix = LayerHc::load(&wm, h, l)?;
        let head = Mixer::nextn_head(&wm, h, l)?;
        let full = crate::switches::is("TORC_MTP_HEAD", "full");
        eprintln!(
            "mtp: qwen4exp block from {path} (layer {l}, {} experts), {} head",
            h.n_expert,
            if full { "full vocabulary" } else { "draft" }
        );
        Ok(Some(MtpHc {
            attn,
            kv: KvCache::new(h, n_ctx),
            moe,
            mix,
            head,
            enorm,
            hnorm,
            bufs: HcBufs::default(),
            l,
            own: wm,
            full,
            noted: Vec::new(),
            noted_rows: Vec::new(),
            is_noted: vec![false; w.head()?.shape().1],
        }))
    }

    /// Notes the ids seen (in the prompt or the reply) for the draft head, as mtp::MtpBlock::note does.
    pub fn note(&mut self, w: &Weights, ids: &[u32]) -> Result<()> {
        if self.full {
            return Ok(());
        }
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

    /// The width of a draft logit row: the vocabulary (with the full head), or DRAFT_V plus the noted ids.
    pub fn draft_width(&self) -> usize {
        if self.full { self.is_noted.len() } else { DRAFT_V + self.noted.len() }
    }

    /// The token id of draft logit index i.
    pub fn draft_id(&self, i: usize) -> u32 {
        if self.full || i < DRAFT_V { i as u32 } else { self.noted[i - DRAFT_V] }
    }

    /// Rows j = 0..n at positions pos0 + j: token tokens[j] with the main model's streams hid[j] (n x hc d_model, taken
    /// before its final mixer). Returns the block's streams (n x hc d_model), and `logits` (empty, or the full
    /// vocabulary) gets the last row's draft logits.
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
        let n = tokens.len();
        let (dm, hc) = (h.d_model as usize, h.hc as usize);
        let (d, nb, nbc) = (hc * dm, dm / QK, 2 * dm / QK);
        if n == 0 || hid.len() != n * d || (!logits.is_empty() && logits.len() != self.draft_width()) {
            return err("mtp: bad buffer sizes");
        }
        let emb = w.need("token_embd.weight")?;
        let rb = emb.row_bytes();
        let mut e = vec![0.0f32; n * dm];
        for (j, &t) in tokens.iter().enumerate() {
            if t as usize >= emb.shape().1 {
                return err("mtp: token out of range");
            }
            kops::dequantize(emb.t.ty, &emb.data[t as usize * rb..(t as usize + 1) * rb], &mut e[j * dm..(j + 1) * dm]);
        }
        // the inputs to the block's streams, with row r = (token j, stream s): [enorm(e_j); hnorm_s(hid_j,s)] in Q8
        let mut catq = Q8::new(n * hc * nbc);
        {
            let qp = crate::pool::Q8Ptr::new(catq.view_mut());
            let (en, hn, eps) = (&self.enorm, &self.hnorm, h.eps);
            crate::pool::par_for(n * hc, &|r| {
                let (j, s) = (r / hc, r % hc);
                let mut cat = vec![0.0f32; 2 * dm];
                norm_gg(&mut cat[..dm], &e[j * dm..(j + 1) * dm], en, eps);
                norm_gg(&mut cat[dm..], &hid[j * d + s * dm..j * d + (s + 1) * dm], &hn[s * dm..(s + 1) * dm], eps);
                // SAFETY: item r owns row r of catq
                kops::q8k_quantize_gg(&cat, unsafe { qp.slice(r * nbc, nbc) });
            });
        }
        let ow = &self.own;
        let mut res = vec![0.0f32; n * d];
        apply_group_multi(
            &mut [(Mat::of(&ow.layer(self.l, "nextn.eh_proj.weight")?), &mut res[..])],
            catq.view(),
            n * hc,
        );
        let mut x = vec![0.0f32; n * dm];
        let mut xn = vec![0.0f32; n * dm];
        let mut xq = Q8::new(n * nb);
        for part in 0..2 {
            hc_mix(ow, h, &self.mix.mix[part], &res, 0, n, &mut xn, &mut self.bufs);
            prep(dm, &mut xn, xq.view_mut(), n);
            x.fill(0.0);
            if part == 1 {
                moe_tokens(ow, &self.moe, &xn, xq.view(), &mut x, n)?;
            } else {
                qsa_prepare_mtp(ow, h, &self.attn, &mut self.kv, &xn, n, pos0);
                attn_block(ow, h, &self.attn, eng, &mut self.kv, pos0, &xn, xq.view(), &mut x, n)?;
            }
            combine(h, &mut res, &x, &self.bufs.inj, n);
        }
        if !logits.is_empty() {
            // the head: the block's own final mixer on the last row, then the main model's vocabulary product
            hc_mix(ow, h, &self.head, &res, n - 1, n, &mut xn, &mut self.bufs);
            kops::q8k_quantize_gg(&xn[(n - 1) * dm..n * dm], xq.sub_mut((n - 1) * nb..n * nb));
            let last = xq.sub((n - 1) * nb..n * nb);
            let out = w.head()?;
            let k = self.noted.len();
            if self.full {
                Mat::of(&out).apply(last, logits);
            } else {
                let orb = out.row_bytes();
                let head = Mat { ty: out.t.ty, rows: DRAFT_V, cols: dm, data: &out.data[..DRAFT_V * orb] };
                if k == 0 {
                    apply_group_multi(&mut [(head, logits)], last, 1);
                } else {
                    // both products in one sweep, straight into their parts of the row
                    let noted = Mat { ty: out.t.ty, rows: k, cols: dm, data: &self.noted_rows };
                    let (lh, ln) = logits.split_at_mut(DRAFT_V);
                    apply_group_multi(&mut [(head, lh), (noted, ln)], last, 1);
                }
            }
        }
        Ok(res)
    }
}
