//! npu.rs - the NPU tier: the dense products of big prompt blocks on the Hexagon NPU.
//!
//! A DeltaNet layer's attn_qkv, attn_gate and ssm_out, and an attention layer's attn_q/k/v/output, are uploaded once at
//! load (TORC_NPU=1, ARM64 Windows). [`apply`] then runs a set of products over the same f32 rows there, for blocks of
//! at least TORC_NPU_MIN_N tokens (default 32). The shared expert is started there by moe_tokens ([`begin_ffns`]) and
//! runs while the CPU computes the routed experts. With TORC_NPU_PROFILE (an expert-use profile), the TORC_NPU_HOT most
//! used routed experts of each layer are uploaded too, and moe_tokens starts their tokens there in the same graph as
//! the shared expert. The router stays on the CPU, so the routing is exactly the CPU path's. Everything else, and every
//! block below the minimum (decode, MTP verification), stays on the CPU path, which is unchanged: with the tier off,
//! the output is byte-identical to the CPU engine. With it on, the prompt's products are the NPU's (on f32 activations,
//! which are closer to an f64 product than the CPU's Q8 ones: see npucheck), so the result is checked by tolerance
//! against the CPU path rather than by bytes.

use crate::moe::Mat;

/// One SwiGLU expert for [`begin_ffns`]: its NPU ids (gate, up, down), n f32 input rows of the expert's cols at x, and
/// the outputs [n][rows of down] at y.
#[derive(Clone, Copy)]
pub struct Ffn {
    pub ids: [i32; 3],
    pub x: *const f32,
    pub n: usize,
    pub y: *mut f32,
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
mod imp {
    use super::{Ffn, Mat};
    use crate::weights::Weights;
    use kernels::npu::{Npu, NpuGroup, NpuMm, configure};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    struct Tier {
        npu: Npu,
        /// a weight's data address -> its NPU id
        ids: HashMap<usize, i32>,
        min_tokens: usize,
        /// for each layer, its hot experts in increasing id, with their NPU ids (gate, up, down)
        hot: Vec<Vec<(usize, [i32; 3])>>,
    }

    static TIER: OnceLock<Mutex<Tier>> = OnceLock::new();

    /// The products put on the NPU, per layer: the dense set minus the router, which stays on the CPU so that the
    /// routing is exactly the CPU path's. (gemma4: its dense branch, ffn_gate / ffn_up / ffn_down, applied one product
    /// at a time, with GELU on the CPU)
    const DENSE: [&str; 13] = [
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_out.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];

    /// A matrix's rows as Q8_0: dequantised, then per 32 weights d = amax / 127 (as fp16) and q = round-to-even(v / d).
    /// This is lossy, so it is only for a type the NPU does not take (the 35B's Q4_1 downs); IQ4_NL (Flash Next's
    /// experts) converts exactly instead.
    fn to_q8_0(m: &Mat) -> Vec<u8> {
        let rb = m.data.len() / m.rows;
        let ob = m.cols / 32 * 34;
        let mut out = vec![0u8; m.rows * ob];
        if m.ty == kernels::mix::IQ4_NL {
            // exact: the block's fp16 scale as it is, and each code as its int8 value (the low nibble of qs[j] is
            // weight j and its high nibble weight j + 16, as ggml dequantises IQ4_NL)
            let kv = kernels::mix::KVALUES_IQ4NL;
            for r in 0..m.rows {
                let src = &m.data[r * rb..(r + 1) * rb];
                for (b, o) in out[r * ob..(r + 1) * ob].chunks_exact_mut(34).enumerate() {
                    let s = &src[b * 18..b * 18 + 18];
                    o[..2].copy_from_slice(&s[..2]);
                    for j in 0..16 {
                        o[2 + j] = kv[(s[2 + j] & 15) as usize] as u8;
                        o[2 + 16 + j] = kv[(s[2 + j] >> 4) as usize] as u8;
                    }
                }
            }
            return out;
        }
        if m.ty == 6 {
            // Q5_0, exact: the block's fp16 scale as it is, and each weight q - 16 (its five bits less 16) as an int8
            for r in 0..m.rows {
                let src = &m.data[r * rb..(r + 1) * rb];
                for (b, o) in out[r * ob..(r + 1) * ob].chunks_exact_mut(34).enumerate() {
                    let s = &src[b * 22..b * 22 + 22];
                    o[..2].copy_from_slice(&s[..2]);
                    let qh = u32::from_le_bytes([s[2], s[3], s[4], s[5]]);
                    for j in 0..16 {
                        let lo = (s[6 + j] & 15) | ((qh >> j) as u8 & 1) << 4;
                        let hi = (s[6 + j] >> 4) | ((qh >> (j + 16)) as u8 & 1) << 4;
                        o[2 + j] = (lo as i8 - 16) as u8;
                        o[2 + 16 + j] = (hi as i8 - 16) as u8;
                    }
                }
            }
            return out;
        }
        let mut f = vec![0f32; m.cols];
        for r in 0..m.rows {
            kernels::ops::dequantize(m.ty, &m.data[r * rb..(r + 1) * rb], &mut f);
            for (b, o) in out[r * ob..(r + 1) * ob].chunks_exact_mut(34).enumerate() {
                let v = &f[b * 32..b * 32 + 32];
                let amax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
                let d = amax / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                o[..2].copy_from_slice(&kernels::ops::fp32_to_fp16(d).to_le_bytes());
                for i in 0..32 {
                    o[2 + i] = (v[i] * id).round_ties_even() as i32 as i8 as u8;
                }
            }
        }
        out
    }

    /// Reads a usage file (the expert cache's, or one from TORC_NPU_PROFILE_OUT; EXPERT-CACHE.md) as counts
    /// [n_layer][n_expert], whichever model file wrote it.
    fn read_profile(path: &str, n_layer: usize, ne: usize) -> Result<Vec<u32>, String> {
        let c = crate::ecache::usage_read(std::path::Path::new(path), None, n_layer as u32, ne as u32)
            .map_err(|e| format!("{e}; hot experts OFF"))?;
        Ok(c.into_iter().map(|v| v.min(u32::MAX as u64) as u32).collect())
    }

    /// This machine's layers on a pipeline (pipe::own_layers), else all of them.
    fn own_layers(n_layer: u32) -> std::ops::Range<u32> {
        crate::pipe::own_layers(n_layer as usize).map_or(0..n_layer, |(a, b)| a as u32..b as u32)
    }

    /// Builds the tier: the dense products and, with k > 0, the k most used experts of each layer (by the profile).
    /// On a pipeline only this machine's layers go up: the others' are never run here.
    fn build(w: &Weights, n_layer: u32, groups: &[u32], k: usize, profile: &[u32]) -> Result<Tier, String> {
        let mut npu = match Npu::create() {
            Ok(Some(p)) => p,
            Ok(None) => return Err(String::new()),
            Err(e) => return Err(e),
        };
        let own = own_layers(n_layer);
        let mut pending = Vec::new();
        // Q8_0 copies of matrices the NPU cannot take as they are, kept until the upload
        let mut conv: Vec<Vec<u8>> = Vec::new();
        for l in own.clone() {
            for name in DENSE {
                if let Some(t) = w.tensor(&format!("blk.{l}.{name}")) {
                    let (cols, rows) = t.shape();
                    // a Q5_0 matrix (a GGUF's shared experts) goes up as its exact Q8_0 copy, still found by the mapped
                    // matrix's address
                    let (ty, ptr) = if t.t.ty == 6 {
                        conv.push(to_q8_0(&Mat { ty: 6, rows, cols, data: t.data }));
                        (8, conv.last().unwrap().as_ptr())
                    } else {
                        (t.t.ty, t.data.as_ptr())
                    };
                    // SAFETY: the mmap and conv's buffers (growing conv does not move them) outlive the upload below
                    if let Some(id) = unsafe { npu.weight(ty, ptr, rows, cols) } {
                        pending.push((t.data.as_ptr() as usize, id, rows, cols));
                    }
                }
            }
        }
        // Roadmap: one copy of the hot experts instead of two. The NPU is handed its own, and the mmap keeps the CPU's
        // as well: about 1 GB on a 32 GB machine. the hot experts: gate and up go straight from the mmap, and down too
        // unless the NPU cannot take its type (then a Q8_0 copy, kept in conv until the upload)
        let mut hot: Vec<Vec<(usize, [i32; 3])>> = vec![Vec::new(); groups.len()];
        // an expert's input width and output rows (the same for every routed expert)
        let mut dims = (0, 0);
        // a profile row is as wide as the widest layer: a model whose first layers are dense (one expert) has the
        // same row width in those layers as in the rest
        let width = profile.len() / groups.len().max(1);
        for (l, &g) in groups.iter().enumerate().filter(|&(l, _)| k > 0 && own.contains(&(l as u32))) {
            let ne = w.layer_experts(g).n_expert as usize;
            let c = &profile[l * width..l * width + ne];
            let mut ord: Vec<usize> = (0..ne).collect();
            ord.sort_by_key(|&e| std::cmp::Reverse(c[e]));
            for &e in ord.iter().take(k).filter(|&&e| c[e] > 0) {
                let (slot, lay) = w.expert(g, e as u32).map_err(|e| e.to_string())?;
                let [gm, um, mut dm] = crate::moe::mats_of(slot, lay);
                dims = (gm.cols, dm.rows);
                // a padded expert (Flash Next's are 640 wide, stored as 768 so that a row is whole blocks, and the CPU
                // path pads the hidden with zeros): the NPU's down takes only the real columns, or its product with the
                // 640-wide hidden does not fit (ggml_can_mul_mat). This is exact, since the dropped columns only met
                // zeros. It only works through to_q8_0's exact IQ4_NL copy, which reads a row's first blocks; other
                // types get no hot experts
                let narrowed = dm.cols != gm.rows;
                if narrowed {
                    if dm.ty != kernels::mix::IQ4_NL || dm.cols < gm.rows || gm.rows % 32 != 0 {
                        return Err(format!("expert down {} x {} against a {}-wide hidden", dm.rows, dm.cols, gm.rows));
                    }
                    dm.cols = gm.rows;
                }
                // any of the three in a type the NPU cannot take goes up as a Q8_0 copy (the NPU refuses raw IQ4_NL;
                // IQ4_NL to Q8_0 is exact, with the same scale and each code an int8)
                let mut put = |mx: &Mat, copy: bool| {
                    if matches!(mx.ty, 2 | 8 | 14) && !copy {
                        (mx.ty, mx.data.as_ptr())
                    } else {
                        conv.push(to_q8_0(mx));
                        (8, conv.last().unwrap().as_ptr())
                    }
                };
                let ((gty, gptr), (uty, uptr), (dty, dptr)) = (put(&gm, false), put(&um, false), put(&dm, narrowed));
                // SAFETY: the mmap and conv's buffers (growing conv does not move them) outlive the upload
                let ids = unsafe {
                    [
                        npu.weight(gty, gptr, gm.rows, gm.cols),
                        npu.weight(uty, uptr, um.rows, um.cols),
                        npu.weight(dty, dptr, dm.rows, dm.cols),
                    ]
                };
                if let [Some(a), Some(b), Some(c)] = ids {
                    hot[l].push((e, [a, b, c]));
                }
            }
            hot[l].sort_by_key(|h| h.0);
        }
        let (bytes, secs) = npu.upload().map_err(|()| "upload failed".to_string())?;
        drop(conv);
        pending.retain(|&(_, id, ..)| npu.ok(id));
        let ids: HashMap<usize, i32> = pending.iter().map(|&(a, id, ..)| (a, id)).collect();
        for h in &mut hot {
            h.retain(|(_, i)| i.iter().all(|&id| npu.ok(id)));
        }
        let min_tokens = npu.min_tokens();
        let n_hot: usize = hot.iter().map(|h| h.len()).sum();
        eprintln!(
            "npu: {} dense products and {n_hot} hot experts on the NPU ({:.0} MiB in {:.2} s), blocks of {min_tokens}+ tokens",
            ids.len(),
            bytes as f64 / 1048576.0,
            secs
        );
        // every weight is used once here (64 zero rows; the dense products one per graph, and a layer's hot experts one
        // graph), so a prompt's first block does not pay for their first use. TORC_NPU_WARM=0 skips this
        if crate::switches::get("TORC_NPU_WARM").is_none_or(|v| v != "0") {
            let t = std::time::Instant::now();
            for &(_, id, rows, cols) in &pending {
                let x = vec![0f32; 64 * cols];
                let mut y = vec![0f32; 64 * rows];
                let mm = [NpuMm { w: id, y: y.as_mut_ptr(), y_stride: rows }];
                let g = [NpuGroup {
                    x: x.as_ptr(),
                    cols,
                    n: 64,
                    mm: mm.as_ptr(),
                    n_mm: 1,
                    ffn: [-1; 3],
                    ffn_y: std::ptr::null_mut(),
                }];
                // SAFETY: x and y hold the rows across the call
                if !unsafe { npu.run(&g) } {
                    eprintln!("npu: a warm-up graph failed");
                    break;
                }
            }
            let (cols, rows) = dims;
            let x = vec![0f32; 64 * cols];
            let mut y = vec![0f32; 64 * 64 * rows];
            for h in hot.iter().filter(|h| !h.is_empty()) {
                let g: Vec<NpuGroup> = h
                    .iter()
                    .take(64)
                    .enumerate()
                    .map(|(i, &(_, ids))| NpuGroup {
                        x: x.as_ptr(),
                        cols,
                        n: 64,
                        mm: std::ptr::null(),
                        n_mm: 0,
                        ffn: ids,
                        ffn_y: y[i * 64 * rows..].as_mut_ptr(),
                    })
                    .collect();
                // SAFETY: x and y hold every group's rows across the call
                if !unsafe { npu.run(&g) } {
                    eprintln!("npu: a warm-up graph failed");
                    break;
                }
            }
            eprintln!("npu: weights warmed in {:.2} s", t.elapsed().as_secs_f64());
        }
        Ok(Tier { npu, ids, min_tokens, hot })
    }

    pub fn load(w: &Weights, n_layer: u32, groups: &[u32]) {
        if TIER.get().is_some() || !configure() {
            return;
        }
        // on one machine (own_usage): a model with no expert cache has its routes counted for the usage file, and the
        // hot experts come from that file when no profile is named
        let own = super::OWN.load(std::sync::atomic::Ordering::Relaxed);
        super::OWN_COUNT.store(own && w.cache.is_none(), std::sync::atomic::Ordering::Relaxed);
        // hot experts: TORC_NPU_PROFILE=<counts file>, and TORC_NPU_HOT=k per layer, else the tuned or imported
        // "npu_hot", else the model's default (tune::npu_hot_default: 16 for most), at most 64. Anything above 16
        // that TORC_NPU_HOT did not ask for is only taken on one machine with the whole model in memory and 8 GB to
        // spare: each hot expert is a second copy, and a tuned or read-in count may come from a bigger machine
        let hot_set = crate::switches::given("TORC_NPU_HOT");
        let roomy = w.cache.is_none()
            && !crate::switches::given("TORC_PIPE")
            && crate::weights::total_phys().is_some_and(|m| m.saturating_sub(w.file.header.file_bytes) >= 8 << 30);
        let mut k = match crate::switches::num::<usize>("TORC_NPU_HOT") {
            Some(k) => k,
            None => crate::tune::get("npu_hot")
                .map_or_else(crate::tune::npu_hot_default, |v| v as usize)
                .min(if roomy { 64 } else { 16 }),
        }
        .min(64);
        let mut profile = Vec::new();
        // a dense model (each FFN a one-expert pack, convert.rs dense_of): every token uses its layer's one expert,
        // so the FFNs go up as each layer's hot expert without a profile (TORC_NPU_HOT=0 keeps them on the CPU)
        let dense = !groups.is_empty() && groups.iter().all(|&g| w.layer_experts(g).n_expert == 1);
        // TORC_NPU_HOT without a profile: the expert cache's own usage file for this model on this machine, so a
        // config need not name a path that differs per user and machine (only if that file exists yet)
        let own_usage = || {
            crate::ecache::usage_path(&w.file.header.model_id)
                .filter(|p| p.exists())
                .map(|p| p.to_string_lossy().into_owned())
        };
        let chosen =
            crate::switches::get("TORC_NPU_PROFILE").or_else(|| if hot_set || own { own_usage() } else { None });
        let asked = k;
        match chosen.ok_or(()) {
            _ if dense && k > 0 => {
                k = 1;
                profile = vec![1; groups.len()];
            }
            Ok(p) if k > 0 => {
                let ne = groups.iter().map(|&g| w.layer_experts(g).n_expert as usize).max().unwrap_or(0);
                match read_profile(&p, groups.len(), ne) {
                    Ok(c) => profile = c,
                    Err(e) => {
                        eprintln!("npu: {e}; no hot experts");
                        k = 0;
                    }
                }
            }
            _ => k = 0,
        }
        // the count asked for, which a profile passes on: not the 0 a first session (no usage file yet) or a refused
        // file leaves. A dense model asks for its one expert a layer
        HOT_ASKED.store(if dense { k } else { asked }, std::sync::atomic::Ordering::Relaxed);
        let t = match build(w, n_layer, groups, k, &profile) {
            Err(e) if k > 0 && !e.is_empty() => {
                eprintln!("npu: with {k} hot experts a layer: {e}; trying the dense products alone");
                build(w, n_layer, groups, 0, &profile)
            }
            r => r,
        };
        match t {
            Ok(t) => {
                let _ = TIER.set(Mutex::new(t));
            }
            Err(e) if e.is_empty() => {}
            Err(e) => eprintln!("npu: not used: {e}"),
        }
    }

    pub fn apply(jobs: &mut [(Mat, &mut [f32])], x: &[f32], n: usize) -> bool {
        let Some(t) = TIER.get() else { return false };
        let mut t = t.lock().unwrap();
        if n < t.min_tokens || jobs.is_empty() {
            return false;
        }
        let cols = jobs[0].0.cols;
        let mut mm = Vec::with_capacity(jobs.len());
        for (m, y) in jobs.iter_mut() {
            let Some(&id) = t.ids.get(&(m.data.as_ptr() as usize)) else { return false };
            if m.cols != cols || y.len() < n * m.rows || x.len() < n * cols {
                return false;
            }
            mm.push(NpuMm { w: id, y: y.as_mut_ptr(), y_stride: m.rows });
        }
        let g = [NpuGroup {
            x: x.as_ptr(),
            cols,
            n,
            mm: mm.as_ptr(),
            n_mm: mm.len(),
            ffn: [-1; 3],
            ffn_y: std::ptr::null_mut(),
        }];
        // SAFETY: x and every y live across the call (run waits), and each y holds n * rows
        unsafe { t.npu.run(&g) }
    }

    pub fn stats() -> Option<(f64, u64)> {
        TIER.get().map(|t| t.lock().unwrap().npu.stats())
    }

    pub fn up() -> bool {
        TIER.get().is_some()
    }

    /// The hot experts a layer the tier was asked for (load), whatever it ended up with.
    pub static HOT_ASKED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    pub fn hot_asked() -> usize {
        HOT_ASKED.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn ffn_ids(m: &[Mat; 3], n: usize) -> Option<[i32; 3]> {
        let t = TIER.get()?.lock().unwrap();
        if n < t.min_tokens {
            return None;
        }
        let id = |m: &Mat| t.ids.get(&(m.data.as_ptr() as usize)).copied();
        Some([id(&m[0])?, id(&m[1])?, id(&m[2])?])
    }

    pub fn hot(layer: u32, n: usize) -> Vec<(usize, [i32; 3])> {
        let Some(t) = TIER.get() else { return Vec::new() };
        let t = t.lock().unwrap();
        if n < t.min_tokens {
            return Vec::new();
        }
        t.hot.get(layer as usize).cloned().unwrap_or_default()
    }

    pub unsafe fn begin_ffns(f: &[Ffn], cols: usize) -> bool {
        let Some(t) = TIER.get() else { return false };
        let g: Vec<NpuGroup> = f
            .iter()
            .map(|f| NpuGroup { x: f.x, cols, n: f.n, mm: std::ptr::null(), n_mm: 0, ffn: f.ids, ffn_y: f.y })
            .collect();
        // SAFETY: the caller's contract (super::begin_ffns)
        !g.is_empty() && unsafe { t.lock().unwrap().npu.begin(&g) }
    }

    pub fn finish() -> bool {
        TIER.get().is_some_and(|t| t.lock().unwrap().npu.finish())
    }

    pub fn set_geglu() -> bool {
        TIER.get().is_some_and(|t| t.lock().unwrap().npu.set_geglu(true))
    }
}

#[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux"))))]
mod imp {
    use super::{Ffn, Mat};
    pub fn load(_: &crate::weights::Weights, _: u32, _: &[u32]) {}
    pub fn apply(_: &mut [(Mat, &mut [f32])], _: &[f32], _: usize) -> bool {
        false
    }
    pub fn stats() -> Option<(f64, u64)> {
        None
    }
    pub fn up() -> bool {
        false
    }
    pub fn hot_asked() -> usize {
        0
    }
    pub fn ffn_ids(_: &[Mat; 3], _: usize) -> Option<[i32; 3]> {
        None
    }
    pub fn hot(_: u32, _: usize) -> Vec<(usize, [i32; 3])> {
        Vec::new()
    }
    pub unsafe fn begin_ffns(_: &[Ffn], _: usize) -> bool {
        false
    }
    pub fn finish() -> bool {
        false
    }
    pub fn set_geglu() -> bool {
        false
    }
}

/// Uploads the dense products when TORC_NPU is set (ARM64 Windows), and, with TORC_NPU_PROFILE, the hot experts of each
/// layer (groups[l] is layer l's EXPERTS record); otherwise does nothing.
pub fn load(w: &crate::weights::Weights, n_layer: u32, groups: &[u32]) {
    imp::load(w, n_layer, groups)
}

/// The products y_j = W_j x over n f32 rows x (n x cols, with y_j[t * rows_j + r], as in apply_group_multi), on the
/// NPU, when the tier is up, n is at least its min_tokens and every W_j was uploaded. Returns false, having written
/// nothing, otherwise, and the caller then runs its CPU path.
pub fn apply(jobs: &mut [(Mat, &mut [f32])], x: &[f32], n: usize) -> bool {
    imp::apply(jobs, x, n)
}

/// The NPU ids of an expert held as resident tensors (the shared expert), for a block of n tokens; None unless the tier
/// is up, n >= min_tokens and all three were uploaded.
pub fn ffn_ids(m: &[Mat; 3], n: usize) -> Option<[i32; 3]> {
    imp::ffn_ids(m, n)
}

/// The hot experts a layer the NPU tier was asked for at load (TORC_NPU_HOT, the tune file or the model's default;
/// 0 when there was no usage file to take them from, 1 on a dense model), whatever the upload then managed.
pub fn hot_asked() -> usize {
    imp::hot_asked()
}

/// Layer `layer`'s hot experts (in increasing id, with their NPU ids) for a block of n tokens: empty unless the tier is
/// up with hot experts and n >= min_tokens.
pub fn hot(layer: u32, n: usize) -> Vec<(usize, [i32; 3])> {
    imp::hot(layer, n)
}

/// Starts the experts f in one graph (SwiGLU in f32 on the NPU), each over its own rows; their outputs are written when
/// [`finish`] returns. Returns false if nothing was started.
///
/// # Safety
/// Each x holds n x cols floats until this returns (they are copied in). Each y holds n x rows-of-down floats and must
/// stay valid, and untouched by anything else, until finish returns. After a true return the caller must call finish.
pub unsafe fn begin_ffns(f: &[Ffn], cols: usize) -> bool {
    unsafe { imp::begin_ffns(f, cols) }
}

/// Waits for the work [`begin_ffns`] started and writes its outputs; returns false if it failed.
pub fn finish() -> bool {
    imp::finish()
}

static GEGLU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// gemma4: the feed-forwards the NPU runs (in begin_ffns: hot experts, and a model's own) use GEGLU, with its tanh
/// GELU, instead of SwiGLU for the whole run. Returns false, keeping SwiGLU, when the tier is down or the NPU cannot do
/// it.
pub fn set_geglu() -> bool {
    let ok = imp::set_geglu();
    GEGLU.store(ok, std::sync::atomic::Ordering::Relaxed);
    ok
}

/// Whether [`set_geglu`] took effect.
pub fn geglu() -> bool {
    GEGLU.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the tier is up (loaded, with its dense products uploaded).
pub fn up() -> bool {
    imp::up()
}

/// (seconds waited for the NPU, graphs run), when the tier is up.
pub fn stats() -> Option<(f64, u64)> {
    imp::stats()
}

/// Expert use, counted by moe_tokens for the hot set. TORC_NPU_PROFILE_OUT=file writes the counts at exit
/// ([`write_profile`]) as a usage file with an all-zero model id (EXPERT-CACHE.md). TORC_ROUTES=file writes every
/// call's routes (u32 layer, u32 n, u32 k, then n x k u16 expert ids), for coverage studies. Neither changes any
/// product.
struct Use {
    /// [layer][expert], grown as layers appear
    counts: Vec<Vec<u32>>,
    tokens: u64,
    routes: Vec<u8>,
}

static USE: std::sync::Mutex<Use> = std::sync::Mutex::new(Use { counts: Vec::new(), tokens: 0, routes: Vec::new() });

fn counting() -> (bool, bool) {
    static ON: std::sync::LazyLock<(bool, bool)> = std::sync::LazyLock::new(|| {
        (crate::switches::given("TORC_NPU_PROFILE_OUT"), crate::switches::given("TORC_ROUTES"))
    });
    *ON
}

/// Counts one block's routes (each token's experts, from moe_tokens) when counting is on.
pub fn record(layer: u32, routes: &[Vec<(usize, f32)>]) {
    let (cnt, rts) = counting();
    let cnt = cnt || OWN_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    if !cnt && !rts {
        return;
    }
    let mut u = USE.lock().unwrap();
    if cnt {
        let l = layer as usize;
        if u.counts.len() <= l {
            u.counts.resize(l + 1, Vec::new());
        }
        for r in routes {
            for &(e, _) in r {
                if u.counts[l].len() <= e {
                    u.counts[l].resize(e + 1, 0);
                }
                u.counts[l][e] += 1;
            }
        }
        if layer == 0 {
            u.tokens += routes.len() as u64;
        }
    }
    if rts {
        let k = routes.first().map_or(0, |r| r.len());
        for v in [layer, routes.len() as u32, k as u32] {
            u.routes.extend(v.to_le_bytes());
        }
        for r in routes {
            for &(e, _) in r {
                u.routes.extend((e as u16).to_le_bytes());
            }
        }
    }
}

/// own_usage(true), set by torc-server when it serves on one machine: this machine's usage file is then the hot
/// experts' profile when none is named, and load() switches OWN_COUNT on for a model with no expert cache, whose
/// routes nothing else would count (with a cache, the cache keeps the usage file itself).
static OWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static OWN_COUNT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// See OWN. Call it before the model is loaded.
pub fn own_usage(on: bool) {
    OWN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Adds the routes counted since the last call to this machine's usage file for the model (ecache.rs), so the next
/// load can choose its hot experts from it. A server calls it after each reply. Does nothing unless load() switched
/// the count on, and leaves the counts alone when TORC_NPU_PROFILE_OUT wants them at exit.
pub fn usage_flush(w: &crate::weights::Weights, n_layer: u32, n_expert: u32) {
    if !OWN_COUNT.load(std::sync::atomic::Ordering::Relaxed) || counting().0 {
        return;
    }
    let model = &w.file.header.model_id;
    let Some(path) = crate::ecache::usage_path(model) else { return };
    let mut u = USE.lock().unwrap();
    if u.tokens == 0 {
        return;
    }
    let (nl, ne) = (n_layer as usize, n_expert as usize);
    let mut total =
        crate::ecache::usage_read(&path, Some(model), n_layer, n_expert).unwrap_or_else(|_| vec![0; nl * ne]);
    if total.len() != nl * ne {
        total = vec![0; nl * ne];
    }
    for l in 0..nl {
        for e in 0..ne {
            total[l * ne + e] += u.counts.get(l).and_then(|c| c.get(e)).copied().unwrap_or(0) as u64;
        }
    }
    if crate::ecache::usage_write(&path, model, n_layer, n_expert, &total).is_ok() {
        u.counts.clear();
        u.tokens = 0;
    }
}

/// Writes the counts for layers 0..n_layer (TORC_NPU_PROFILE_OUT) and the routes (TORC_ROUTES), if they were asked for.
pub fn write_profile(n_layer: u32, n_expert: u32) {
    let u = USE.lock().unwrap();
    if let Some(p) = crate::switches::get("TORC_NPU_PROFILE_OUT") {
        let mut c = Vec::with_capacity(n_layer as usize * n_expert as usize);
        for l in 0..n_layer as usize {
            for e in 0..n_expert as usize {
                c.push(u.counts.get(l).and_then(|c| c.get(e)).copied().unwrap_or(0) as u64);
            }
        }
        match crate::ecache::usage_write(std::path::Path::new(&p), &[0; 32], n_layer, n_expert, &c) {
            Ok(()) => eprintln!("profile: {} tokens of expert use -> {}", u.tokens, p),
            Err(e) => eprintln!("profile: {p}: {e}"),
        }
    }
    if let Some(p) = crate::switches::get("TORC_ROUTES")
        && let Err(e) = std::fs::write(&p, &u.routes)
    {
        eprintln!("routes: {p}: {e}");
    }
}
