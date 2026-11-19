//! gpu.rs - attention over a prompt block on a GPU through Vulkan (wgpu), an opt-in tier: built with the `gpu` feature
//! and switched on with TORC_GPU_ATTN=1.
//!
//! The GPU keeps its own copy of each attention layer's keys and values; KvCache::gpu_lo marks the first position where
//! that copy may be stale, and every block brings it up to date before it runs. A block's first tokens go to the GPU
//! and the rest to fattn.rs on the CPU, at the same time. The GPU's share of the block's work (which grows with each
//! token's position) is fixed by TORC_GPU_SHARE (0..1); without it the share follows the two sides' measured speeds, so
//! that both finish together (see Gpu::adapt), starting from the share this machine settled on last time (kept in the
//! tune file, tune.rs, per adapter, kernel and shape) or 0.3. TORC_GPU picks the adapter by a piece of its name
//! (default: the first integrated GPU, else the first GPU). TORC_GPU_MIN (default 32) is the smallest block it takes
//! and TORC_GPU_FROM (default 4096) the fewest keys its last token must see: below that, attention is a small part of
//! the work and waking the GPU costs more than it saves. TORC_GPU_CHECK=1 runs the GPU's tokens again on the CPU and
//! prints the largest difference.
//!
//! A small block gives the GPU few workgroups (8 tokens and one KV head each), too few to keep it busy, so a block's
//! keys are also split into parts that run side by side and are merged after (the shader's `combine`): enough parts
//! to reach TORC_GPU_WG workgroups (default 64), each part at least 16 tiles of 64 keys. TORC_GPU_SPLIT fixes the
//! number of parts (1: no split).
//!
//! The generic kernel (gpu_attn.wgsl) is compiled once per shape a model needs: head dim any multiple of 64 up to 576,
//! 1 to 64 query heads per KV head, keys and values at 8 or 16 bits, a sliding window, a sink per query head
//! (gpt-oss), and the indexer's picks (qwen4exp), applied as a mask. Its sums run in another order than fattn.rs's, so
//! its outputs agree with the CPU's to rounding (about 1e-5 relative at 32k keys), not to the bit, and which tokens
//! go to the GPU follows the two sides' measured speeds, so with the tier on a reply can differ from run to run where
//! the model's choice is a near tie. With the tier off the CPU path is the same as without it and repeats exactly.
//!
//! A shape the generic kernel is slower on has a kernel file of its own, and Kern::of is the table that says which
//! shape runs on which: head dim 256 with 8 query heads per KV head at int8 (the Qwen 35B's full-attention layers) on
//! gpu_attn_gqa8.wgsl, MLA's absorbed head (one KV head of 576: GLM-4.7-Flash, Moonlight) on gpu_attn_mla.wgsl. Work
//! for one model goes in that model's kernel file, not in the generic one, which every other shape runs through. The
//! host code here (run_block) is the same for every kernel, and each share is kept per adapter, kernel and shape, so
//! nothing one shape learns reaches another.
//!
//! Windows resets a GPU whose one job runs past ~2 s, so the GPU's tokens go in slices, each its own submission, when
//! one submission would hold more than about half a second of this GPU's measured work (Gpu::cap; TORC_GPU_SLICE fixes
//! the most head-keys of work in a slice instead). A GPU quick enough for its share gets it in one submission.
//! TORC_GPU_STATS=1 prints where the time went every 256 blocks.
//!
//! Roadmap: experts on the GPU as well. Only a share of a long prompt's attention comes here today, and the expert
//! products, most of the work, stay on the CPU and the NPU.

use crate::attn::KvCache;
use crate::fattn::Fattn;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    pos0: u32,
    cap: u32,
    scale: f32,
    nh: u32,
    /// the parts the keys are split into, and the key tiles in each
    ns: u32,
    tps: u32,
    /// the sliding window (0: none)
    window: u32,
    /// the indexer's picks (qwen4exp): bytes per token's row of pool flags, and positions per pool
    stride: u32,
    kp: u32,
    pad: [u32; 2],
}

/// A compiled kernel's shape: head dim, query heads per KV head, 16-bit cache, sinks, the indexer's picks.
type Shape = (usize, usize, bool, bool, bool);

/// The kernel and its merge for one shape, and the shape's tokens per workgroup.
struct Pipes {
    main: wgpu::ComputePipeline,
    comb: wgpu::ComputePipeline,
    tq: usize,
}

/// One layer's keys, values and their scales on the GPU, laid out as in KvCache: `rows` positions for each KV head,
/// the first of them the cache's position `base`. A full-attention layer's copy is the whole cache (base 0). A
/// sliding-window layer's holds only the positions its window can reach, and moves on as the prompt is read: Gemma 4
/// has 25 such layers, and a whole-context copy of each is 7 GB at 32k tokens for keys no token reads.
struct Layer {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    ks: wgpu::Buffer,
    vs: wgpu::Buffer,
    base: usize,
    rows: usize,
}

/// Which kernel file a shape runs on: Kern::of is the table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kern {
    /// gpu_attn.wgsl: any shape
    Generic,
    /// gpu_attn_gqa8.wgsl: head dim 256, 8 query heads per KV head, int8 (the Qwen 35B's full-attention layers)
    Gqa8,
    /// gpu_attn_mla.wgsl: one KV head of 576 (MLA's absorbed head: GLM-4.7-Flash, Moonlight)
    Mla,
}

impl Kern {
    /// The kernel for a shape with nkv KV heads and this window. A shape that needs a kernel of its own gets a line
    /// here and a kernel file; every other shape stays on the generic one, which is then left alone. TORC_GPU_KERNEL=
    /// generic puts every shape on the generic kernel (to measure a kernel of its own against it).
    fn of(s: Shape, nkv: usize, window: usize) -> Kern {
        static GENERIC: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| crate::switches::is("TORC_GPU_KERNEL", "generic"));
        let (hd, g, k16, sink, qsa) = s;
        let plain = window == 0 && !sink && !qsa;
        if *GENERIC {
            Kern::Generic
        } else if plain && hd == 256 && g == 8 && !k16 {
            Kern::Gqa8
        } else if plain && hd == 576 && nkv == 1 {
            Kern::Mla
        } else {
            Kern::Generic
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kern::Generic => "generic",
            Kern::Gqa8 => "gqa8",
            Kern::Mla => "mla",
        }
    }

    /// An older tune key for this kernel's share (one for all adapters and shapes): read once, so
    /// a machine starts from what it had learned.
    fn old_key(self) -> &'static str {
        match self {
            Kern::Generic => "gpu_share_permille_v2",
            Kern::Gqa8 | Kern::Mla => "gpu_share_permille",
        }
    }
}

/// What a share belongs to: the kernel, the shape, and whether the layer has a sliding window (a model can mix
/// windowed and full layers of one shape, and their work differs).
type Use = (Kern, Shape, bool);

/// The GPU's share of a block's work for one Use, and adapt's state for it.
struct Share {
    share: f64,
    /// adapt's calls, and the share last written to the tune file
    adapted: u64,
    saved: f64,
    /// adapt's blocks running in which the GPU's fair share was under 2%
    slow: u32,
    /// over the blocks since the last check: the seconds they took with the GPU, and the seconds the CPU would have
    /// taken alone at the speed it showed on its own tokens
    pair: f64,
    alone: f64,
    /// checks running (one every 32 blocks) in which the CPU alone would have been at least as fast
    lost: u32,
    /// the tune file's key
    key: String,
}

struct Gpu {
    dev: wgpu::Device,
    queue: wgpu::Queue,
    /// the adapter's name
    name: String,
    /// the kernels compiled so far
    kerns: HashMap<(Kern, Shape), Pipes>,
    /// TORC_GPU_SHARE (None: each share follows the measured speeds)
    fixed: Option<f64>,
    /// the shares in use so far
    shares: HashMap<Use, Share>,
    /// the last block's share (share())
    last: f64,
    /// the GPU's speed on the last block in head-keys a second, never over its true speed (pace()); 0 before any
    rate: f64,
    max_binding: u64,
    /// the layers' copies, by their cache's address and size
    layers: HashMap<(usize, usize), Layer>,
    /// the buffers a block uses, grown as blocks need
    io: Io,
}

/// A block's queries, its outputs (one part per key split), each split row's max and sum, the merged outputs and the
/// read-back copy, each with its size in bytes (0 until first made).
#[derive(Default)]
struct Io {
    q: Option<(wgpu::Buffer, u64)>,
    o: Option<(wgpu::Buffer, u64)>,
    ml: Option<(wgpu::Buffer, u64)>,
    fo: Option<(wgpu::Buffer, u64)>,
    rb: Option<(wgpu::Buffer, u64)>,
    /// the layer's sinks (gpt-oss)
    sk: Option<(wgpu::Buffer, u64)>,
    /// the GPU tokens' pool flags and pool counts (qwen4exp)
    sel: Option<(wgpu::Buffer, u64)>,
    nv: Option<(wgpu::Buffer, u64)>,
}

/// `slot`'s buffer, made again (a power of two in size) when smaller than `bytes`.
fn grow(dev: &wgpu::Device, slot: &mut Option<(wgpu::Buffer, u64)>, bytes: u64, usage: wgpu::BufferUsages) {
    if slot.as_ref().is_none_or(|b| b.1 < bytes) {
        let size = bytes.next_power_of_two().max(256);
        *slot = Some((
            dev.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false }),
            size,
        ));
    }
}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

/// TORC_GPU_SPLIT (a fixed number of key parts, None: chosen per block) and TORC_GPU_WG (the workgroups to aim for).
static SPLIT: std::sync::LazyLock<(Option<usize>, usize)> = std::sync::LazyLock::new(|| {
    let env = |k: &str| crate::switches::num::<usize>(k);
    (env("TORC_GPU_SPLIT").map(|v| v.max(1)), env("TORC_GPU_WG").unwrap_or(64).max(1))
});

/// The key parts for a block of m GPU tokens ending at position last, with nkv KV heads and tq tokens a workgroup,
/// and the tiles of 64 keys in each: enough parts to reach the workgroups aimed for, none under 16 tiles, and no part
/// left empty.
fn parts(m: usize, last: usize, nkv: usize, tq: usize) -> (usize, usize) {
    let tiles = last.div_ceil(64);
    let (fixed, aim) = *SPLIT;
    let want = fixed.unwrap_or_else(|| aim.div_ceil(m.div_ceil(tq) * nkv).min(tiles / 16));
    let want = want.clamp(1, tiles.max(1));
    if want == 1 {
        return (1, tiles.max(1));
    }
    let tps = tiles.div_ceil(want);
    (tiles.div_ceil(tps), tps)
}

/// TORC_GPU_MIN and TORC_GPU_FROM, read before the GPU is opened, so short prompts never wake it.
static LIMITS: std::sync::LazyLock<(usize, usize)> = std::sync::LazyLock::new(|| {
    let env = |k: &str| crate::switches::num::<usize>(k);
    (env("TORC_GPU_MIN").unwrap_or(32).max(1), env("TORC_GPU_FROM").unwrap_or(4096))
});

/// Where run_block's time goes, in ns summed over calls: writing the cache rows and queries, the CPU's tokens, waiting
/// for the GPU after them, reading the outputs back; and the number of calls (stats()).
static NS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Set when the GPU fails (it runs out of memory, or the device is lost): the tier is off for the rest of the run and
/// the CPU takes every block. A card too small for a model's keys at a long context did this with a panic before.
static OFF: AtomicBool = AtomicBool::new(false);

/// An error the device raised outside an error scope during a block (wgpu's own handler would panic on it).
static FAULT: Mutex<Option<String>> = Mutex::new(None);

/// Turns the tier off for the rest of the run, says why once, and gives back the GPU's memory.
fn turn_off(g: &mut Gpu, why: &str) {
    if !OFF.swap(true, Ordering::Relaxed) {
        eprintln!("gpu attention: off for the rest of this run, the CPU takes every block ({why})");
    }
    g.layers.clear();
    g.io = Io::default();
}

/// The out-of-memory error, if any, from the device work since `scope` was pushed.
fn oom(scope: wgpu::ErrorScopeGuard) -> Option<String> {
    pollster::block_on(scope.pop()).map(|e| e.to_string())
}

/// run_block's calls and its seconds writing to the GPU, on the CPU's tokens, waiting for the GPU and reading back,
/// or None when the tier is off.
pub fn stats() -> Option<(u64, [f64; 4])> {
    gpu()?;
    let s = |i: usize| NS[i].load(Ordering::Relaxed) as f64 * 1e-9;
    Some((NS[4].load(Ordering::Relaxed), [s(0), s(1), s(2), s(3)]))
}

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| {
        if !crate::switches::on("TORC_GPU_ATTN") {
            return None;
        }
        match Gpu::open() {
            Ok(g) => Some(Mutex::new(g)),
            Err(e) => {
                eprintln!("gpu attention: off ({e})");
                None
            }
        }
    })
    .as_ref()
}

impl Gpu {
    fn open() -> Result<Gpu, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let inst = wgpu::Instance::new(desc);
        let ads = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::VULKAN));
        let want = crate::switches::get("TORC_GPU");
        let pick = match &want {
            Some(w) => ads.iter().position(|a| a.get_info().name.contains(w.as_str())),
            None => ads
                .iter()
                .position(|a| a.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
                .or(if ads.is_empty() { None } else { Some(0) }),
        };
        let ad = pick.map(|i| &ads[i]).ok_or("no Vulkan GPU matches")?;
        let lim = ad.limits();
        let name = ad.get_info().name;
        let max_binding = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size);
        let (dev, queue) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
            label: Some("torc"),
            required_features: wgpu::Features::empty(),
            required_limits: lim,
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        }))
        .map_err(|e| e.to_string())?;
        // An error no scope caught is kept for run_block to act on: wgpu's default handler panics, which takes the
        // whole server down.
        dev.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            let mut f = FAULT.lock().unwrap_or_else(|p| p.into_inner());
            f.get_or_insert_with(|| e.to_string());
        }));
        let env = |k: &str| crate::switches::num::<f64>(k);
        let fixed = env("TORC_GPU_SHARE").map(|s| s.clamp(0.0, 1.0));
        let (min, from) = *LIMITS;
        let how = match fixed {
            Some(s) => format!("share {s}"),
            None => "share by measured speed".to_string(),
        };
        eprintln!("gpu attention: {name}, {how}, blocks of {min} or more seeing {from} keys or more");
        Ok(Gpu {
            dev,
            queue,
            name,
            kerns: HashMap::new(),
            fixed,
            shares: HashMap::new(),
            last: fixed.unwrap_or(0.3),
            rate: 0.0,
            max_binding,
            layers: HashMap::new(),
            io: Io::default(),
        })
    }

    /// Kernel `k` for shape `s`, compiled on first use. The generic and MLA kernels get the shape's constants in front
    /// (the generic one lays its rows out by the power of two at or above g, the MLA one by g itself) and keep their
    /// sink and pick lines only when the shape has them (an unused binding must not appear in the layout); the gqa8
    /// kernel is its file as it is.
    fn kernel(&mut self, k: Kern, s: Shape) -> &Pipes {
        if !self.kerns.contains_key(&(k, s)) {
            let (hd, g, k16, sink, qsa) = s;
            let kw = if k16 { hd / 2 } else { hd / 4 };
            // keys per value chunk: the largest power of two with vc x hd / 4 vec4s within sa's 512
            let vc = 1usize << (512 / (hd / 4)).ilog2();
            let (label, src, tq) = match k {
                Kern::Gqa8 => ("gpu_attn_gqa8", include_str!("gpu_attn_gqa8.wgsl").to_string(), 8),
                Kern::Mla => {
                    let body = include_str!("gpu_attn_mla.wgsl").replace("/*SINK*/", "//").replace("/*QSA*/", "//");
                    let src = format!(
                        "const HD: u32 = {hd}u;\nconst G: u32 = {g}u;\nconst K16: bool = {k16};\nconst KW: u32 = {kw}u;\n\
                        const VC: u32 = {vc}u;\n{body}"
                    );
                    ("gpu_attn_mla", src, 64 / g)
                }
                Kern::Generic => {
                    let body = include_str!("gpu_attn.wgsl")
                        .replace("/*SINK*/", if sink { "" } else { "//" })
                        .replace("/*QSA*/", if qsa { "" } else { "//" });
                    // rows laid out by the power of two at or above g (gpu_attn.wgsl says why)
                    let gp = g.next_power_of_two();
                    let src = format!(
                        "const HD: u32 = {hd}u;\nconst G: u32 = {g}u;\nconst GP: u32 = {gp}u;\nconst K16: bool = {k16};\nconst KW: u32 = {kw}u;\n\
                        const VC: u32 = {vc}u;\n{body}"
                    );
                    ("gpu_attn", src, 64 / gp)
                }
            };
            let dev = &self.dev;
            let sm = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let mk = |entry: &str| {
                dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &sm,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let pipes = Pipes { main: mk("main"), comb: mk("combine"), tq };
            self.kerns.insert((k, s), pipes);
        }
        &self.kerns[&(k, s)]
    }

    /// The share for `at`, made on first use: TORC_GPU_SHARE, else the share this machine settled on with this
    /// adapter, kernel and shape (else, the first time, the one kept under the kernel's old key), else 0.3. A learned
    /// 0 means adapt switched the GPU off for it (too slow): it stays off.
    fn state(&mut self, at: Use) -> &mut Share {
        if !self.shares.contains_key(&at) {
            let (k, (hd, g, k16, sink, qsa), windowed) = at;
            let what = format!(
                "{} {hd}x{g}{}{}{}{}",
                k.name(),
                if k16 { " 16-bit" } else { "" },
                if sink { " sinks" } else { "" },
                if qsa { " picks" } else { "" },
                if windowed { " window" } else { "" }
            );
            let key = format!("gpu_share_permille {} {what}", self.name);
            let learned = crate::tune::get(&key)
                .or_else(|| crate::tune::get(k.old_key()))
                .map(|v| if v == 0 { 0.0 } else { (v as f64 / 1000.0).clamp(SHARE_MIN, 0.95) });
            let share = self.fixed.or(learned).unwrap_or(0.3);
            let how = match (self.fixed, learned) {
                (Some(_), _) => "",
                (None, Some(0.0)) => " (off: measured too slow for this model and layout earlier)",
                (None, Some(_)) => " (this machine's last) to start, then by measured speed",
                (None, None) => " to start, then by measured speed",
            };
            eprintln!("gpu attention: {what} kernel, share {share}{how}");
            self.shares
                .insert(at, Share { share, adapted: 0, saved: share, slow: 0, pair: 0.0, alone: 0.0, lost: 0, key });
        }
        self.shares.get_mut(&at).unwrap()
    }

    /// After a block of `at` split m / n at pos0: the next share from the GPU's and the CPU's speeds on it, cpu seconds
    /// on the CPU's tokens and wait seconds waiting for the GPU after them. When the GPU finished first its time is
    /// unknown (at most cpu), so the share only steps up. Kept between SHARE_MIN and 0.95, so both sides keep being
    /// measured.
    ///
    /// A GPU that does not pay is switched off (share 0, kept in the tune file), on either of two measurements:
    /// - its fair share stays under 2% for 8 blocks running: at the SHARE_MIN floor it would hold the CPU waiting
    ///   on every block;
    /// - two checks running, each over 32 blocks, find that those blocks took longer with the GPU (cpu + wait) than
    ///   the CPU would have taken alone at the speed it showed on its own tokens (cpu x the block's work / the CPU's
    ///   part of it). This catches a GPU that is quick when run alone but slow to start each job (a desktop's
    ///   integrated GPU beside a fast CPU): its share looks fair on the blocks it finishes first, yet over many blocks
    ///   the CPU spends longer waiting for it than working. The CPU's speed is taken while the GPU runs beside it, so
    ///   the comparison leans towards keeping the GPU.
    fn adapt(&mut self, at: Use, n: usize, m: usize, pos0: usize, cpu: f64, wait: f64) {
        if self.fixed.is_some() || m >= n || cpu <= 0.0 {
            return;
        }
        let Some(st) = self.shares.get_mut(&at) else { return };
        let (wg, wc) = (work(m, pos0), work(n, pos0) - work(m, pos0));
        st.pair += cpu + wait;
        st.alone += cpu * (wg + wc) / wc;
        let next = if wait > 0.02 * cpu {
            let (rg, rc) = (wg / (cpu + wait), wc / cpu);
            let fair = rg / (rg + rc);
            st.slow = if fair < 0.02 { st.slow + 1 } else { 0 };
            0.5 * st.share + 0.5 * fair
        } else {
            st.slow = 0;
            st.share + 0.03
        };
        st.adapted += 1;
        if st.adapted % 32 == 0 {
            st.lost = if st.pair >= st.alone { st.lost + 1 } else { 0 };
            (st.pair, st.alone) = (0.0, 0.0);
        }
        if st.slow >= 8 || st.lost >= 2 {
            let why = if st.lost >= 2 {
                "the CPU alone measured faster than the CPU and this GPU together"
            } else {
                "the GPU measured under 2% of the work's fair share"
            };
            eprintln!("gpu attention: off for this model and layout, {why}");
            st.share = 0.0;
            self.last = 0.0;
            let entry =
                serde_json::json!({ "value": 0, "how": format!("measured while running (gpu.rs adapt): {why}") });
            if crate::tune::put(&st.key, entry).is_ok() {
                st.saved = 0.0;
            }
            return;
        }
        st.share = next.clamp(SHARE_MIN, 0.95);
        self.last = st.share;
        // every 32 blocks, a share that has moved goes into the tune file, so the next start begins there
        if st.adapted % 32 == 0 && (st.share - st.saved).abs() >= 0.01 {
            let v = (st.share * 1000.0).round() as u64;
            let entry = serde_json::json!({ "value": v, "how": "measured while running (gpu.rs adapt)" });
            if crate::tune::put(&st.key, entry).is_ok() {
                st.saved = st.share;
            }
        }
    }

    /// The most head-keys of work one submission may hold (slices()): TORC_GPU_SLICE, else what this GPU was last
    /// measured to finish in SLICE_S seconds, and never under 2e6, the cap before any block has been measured (a
    /// fraction of a second on a slow integrated GPU). So a GPU that finishes a block's tokens well inside Windows'
    /// ~2 s gets them in one submission, and a slow one still gets slices.
    fn cap(&self) -> f64 {
        static FIXED: std::sync::LazyLock<Option<f64>> =
            std::sync::LazyLock::new(|| crate::switches::num::<f64>("TORC_GPU_SLICE").filter(|&c| c > 0.0));
        FIXED.unwrap_or((self.rate * SLICE_S).max(2e6))
    }

    /// Notes the GPU's speed on a block: its m tokens at pos0 (nh query heads) were written, run and read back within
    /// `secs`, which also holds the CPU's tokens, so the speed kept is never over the GPU's own.
    fn pace(&mut self, nh: usize, m: usize, pos0: usize, secs: f64) {
        if secs > 0.0 {
            self.rate = nh as f64 * work(m, pos0) / secs;
        }
    }
}

/// The least share a GPU in use is given (Gpu::adapt): under it, the GPU's own speed could no longer be measured.
const SHARE_MIN: f64 = 0.05;

/// The seconds of measured GPU work one submission may hold (Gpu::cap): a quarter of Windows' ~2 s limit, as the speed
/// it is worked out from is the last block's.
const SLICE_S: f64 = 0.5;

/// The GPU's tokens of a block of n at pos0: the first m, whose work (each token's key count) is the share's part of
/// the block's, a multiple of the kernel's tq tokens unless it is the whole block.
fn split(share: f64, n: usize, pos0: usize, tq: usize) -> usize {
    if share >= 1.0 {
        return n;
    }
    let goal = share * work(n, pos0);
    let mut m = 0;
    while m < n && work(m, pos0) < goal {
        m += tq;
    }
    m.min(n)
}

/// The work of the first m tokens of a block at pos0: the keys they see.
fn work(m: usize, pos0: usize) -> f64 {
    (m * pos0 + m * (m + 1) / 2) as f64
}

/// The GPU's m tokens of a block at pos0 as slices [a, b) of whole kernel rows (tq tokens), each at most `cap`
/// head-keys of work (nh query heads x the keys its tokens see; one row group may be more): Gpu::cap.
fn slices(m: usize, pos0: usize, nh: usize, tq: usize, cap: f64) -> Vec<(usize, usize)> {
    let cost = |a: usize, b: usize| nh as f64 * (work(b, pos0) - work(a, pos0));
    let mut out = Vec::new();
    let mut a = 0;
    while a < m {
        let mut b = (a + tq).min(m);
        while b < m && cost(a, (b + tq).min(m)) <= cap {
            b = (b + tq).min(m);
        }
        out.push((a, b));
        a = b;
    }
    out
}

/// The GPU's share of the last block's work (moving unless TORC_GPU_SHARE fixed it), or None when the tier is off.
pub fn share() -> Option<f64> {
    if OFF.load(Ordering::Relaxed) {
        return None;
    }
    Some(gpu()?.lock().unwrap().last)
}

/// Attention over n tokens at positions pos0.. (q and out are n x qd, as Fattn::run_block takes them), the first
/// tokens on the GPU and the rest on the CPU through `f`, at the same time. Returns the position up to which the GPU's
/// copy of the cache is now current (the caller stores it in kv.gpu_lo), or None, with nothing done, when the tier is
/// off or does not cover this shape or block: then the caller runs the block on the CPU.
pub fn run_block(
    kv: &KvCache,
    f: &Fattn,
    q: &[f32],
    qd: usize,
    n: usize,
    pos0: usize,
    out: &mut [f32],
) -> Option<usize> {
    let (min, from) = *LIMITS;
    if n < min || pos0 + n < from || OFF.load(Ordering::Relaxed) {
        return None;
    }
    let g = gpu()?;
    let (nh, nkv, hd, scale, window) = f.shape();
    if hd % 64 != 0 || hd > 576 || nkv == 0 || nh % nkv != 0 || nh / nkv > 64 || qd != nh * hd || kv.cap == 0 {
        return None;
    }
    let k16 = !kv.k16.is_empty();
    let sinks = f.sinks();
    // the indexer's picks for this block's tokens (qwen4exp), as attn.rs's select_of hands them to the CPU kernel
    let qsa = kv.qsa && kv.ix_npool > 0 && kv.nv.len() >= n && kv.sel.len() >= n * kv.ix_npool;
    if kv.qsa && !qsa {
        return None;
    }
    let shape: Shape = (hd, nh / nkv, k16, sinks.is_some(), qsa);
    let kern = Kern::of(shape, nkv, window);
    let at: Use = (kern, shape, window > 0);
    let mut g = g.lock().unwrap();
    let tq = g.kernel(kern, shape).tq;
    let share = g.state(at).share;
    g.last = share;
    let m = split(share, n, pos0, tq);
    if m == 0 {
        return None;
    }
    let cap = kv.cap;
    let hi = pos0 + n;
    // bytes per cache number
    let eb = if k16 { 2 } else { 1 };
    // the layer's copy: made on first use (and then written from its first position), else brought up to date from
    // gpu_lo
    let key = (if k16 { kv.k16.as_ptr() as usize } else { kv.k.as_ptr() as usize }, cap);
    let have = g.layers.get(&key).map(|ly| (ly.base, ly.rows));
    // a windowed layer's copy starts at the first position this block can see, and has room for a few blocks more
    // before it must move on (then the window's positions are written again)
    let first = if window > 0 { pos0.saturating_sub(window) } else { 0 };
    let (base, rows, make) = match have {
        Some((b, r)) if window == 0 || (first >= b && hi - b <= r) => (b, r, false),
        Some((_, r)) if hi - first <= r => (first, r, false),
        _ if window == 0 => (0, cap, true),
        _ => (first, cap.min(window + n + window.max(n)), true),
    };
    let moved = have.is_some_and(|(b, _)| b != base);
    let kbytes = (nkv * rows * hd * eb) as u64;
    if kbytes > g.max_binding {
        return None;
    }
    let lo = if make || moved { base } else { kv.gpu_lo.min(pos0).max(base) };
    if make {
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let scope = g.dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let mk = |size: u64| {
            g.dev.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: st, mapped_at_creation: false })
        };
        let sb = (nkv * rows * 4) as u64;
        let ly = Layer { k: mk(kbytes), v: mk(kbytes), ks: mk(sb), vs: mk(sb), base, rows };
        g.layers.insert(key, ly);
        // a card without room for this layer's keys: nothing is done here and the CPU runs the block
        if let Some(e) = oom(scope) {
            turn_off(&mut g, &format!("the GPU has no room for another layer's keys: {e}"));
            return None;
        }
    } else if moved {
        g.layers.get_mut(&key).unwrap().base = base;
    }
    let t0 = std::time::Instant::now();
    {
        let ly = &g.layers[&key];
        for h in 0..nkv {
            let (r0, r1) = (h * cap + lo, h * cap + hi);
            // the copy's row for the cache's row r0
            let g0 = h * rows + lo - base;
            let at = (g0 * hd * eb) as u64;
            if k16 {
                g.queue.write_buffer(&ly.k, at, bytemuck::cast_slice(&kv.k16[r0 * hd..r1 * hd]));
                g.queue.write_buffer(&ly.v, at, bytemuck::cast_slice(&kv.v16[r0 * hd..r1 * hd]));
            } else {
                g.queue.write_buffer(&ly.k, at, bytemuck::cast_slice(&kv.k[r0 * hd..r1 * hd]));
                g.queue.write_buffer(&ly.v, at, bytemuck::cast_slice(&kv.v[r0 * hd..r1 * hd]));
            }
            g.queue.write_buffer(&ly.ks, (g0 * 4) as u64, bytemuck::cast_slice(&kv.k_scale[r0..r1]));
            g.queue.write_buffer(&ly.vs, (g0 * 4) as u64, bytemuck::cast_slice(&kv.v_scale[r0..r1]));
        }
    }
    let bytes = (m * qd * 4) as u64;
    // the GPU's tokens go in slices, each its own submission, so that no single GPU job runs long: Windows resets a GPU
    // whose job passes ~2 s (Windows then reports a display driver fault: first a wrong block, then the
    // device lost). Each slice's buffers are written just before its submission (the queue runs them in order).
    let cuts = slices(m, pos0, nh, tq, g.cap());
    let mmax = cuts.iter().map(|&(a, b)| b - a).max().unwrap_or(m);
    // the kernel sees the copy's positions: the cache's less `base`
    let nsmax = cuts.iter().map(|&(a, b)| parts(b - a, pos0 + b - base, nkv, tq).0).max().unwrap_or(1);
    {
        use wgpu::BufferUsages as U;
        let scope = g.dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let g = &mut *g;
        let st = U::STORAGE;
        let sb = (mmax * qd * 4) as u64;
        grow(&g.dev, &mut g.io.q, sb, st | U::COPY_DST);
        grow(&g.dev, &mut g.io.o, sb * nsmax as u64, st | U::COPY_SRC);
        grow(&g.dev, &mut g.io.ml, (nsmax * mmax * nh * 8) as u64, st);
        grow(&g.dev, &mut g.io.fo, sb, st | U::COPY_SRC);
        grow(&g.dev, &mut g.io.rb, bytes, U::MAP_READ | U::COPY_DST);
        grow(&g.dev, &mut g.io.sk, (nh * 4) as u64, st | U::COPY_DST);
        grow(&g.dev, &mut g.io.sel, (mmax * kv.ix_npool.max(1)).next_multiple_of(4) as u64, st | U::COPY_DST);
        grow(&g.dev, &mut g.io.nv, (mmax * 4) as u64, st | U::COPY_DST);
        if let Some(e) = oom(scope) {
            turn_off(g, &format!("the GPU has no room for a block's work buffers: {e}"));
            return None;
        }
    }
    fn buf(b: &Option<(wgpu::Buffer, u64)>) -> &wgpu::Buffer {
        &b.as_ref().unwrap().0
    }
    let (qb, ob, mlb, fob, rb, skb) =
        (buf(&g.io.q), buf(&g.io.o), buf(&g.io.ml), buf(&g.io.fo), buf(&g.io.rb), buf(&g.io.sk));
    let (selb, nvb) = (buf(&g.io.sel), buf(&g.io.nv));
    if let Some(s) = sinks {
        g.queue.write_buffer(skb, 0, bytemuck::cast_slice(s));
    }
    let ly = &g.layers[&key];
    let pp = &g.kerns[&(kern, shape)];
    fn e(i: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding: i, resource: b.as_entire_binding() }
    }
    for &(a, b) in &cuts {
        let mc = b - a;
        let cb = (mc * qd * 4) as u64;
        let (ns, tps) = parts(mc, pos0 + b - base, nkv, tq);
        g.queue.write_buffer(qb, 0, bytemuck::cast_slice(&q[a * qd..b * qd]));
        if qsa {
            // the slice's rows of pool flags, padded to whole words
            let mut rows = kv.sel[a * kv.ix_npool..b * kv.ix_npool].to_vec();
            rows.resize(rows.len().next_multiple_of(4), 0);
            g.queue.write_buffer(selb, 0, &rows);
            g.queue.write_buffer(nvb, 0, bytemuck::cast_slice(&kv.nv[a..b]));
        }
        let pr = Params {
            n: mc as u32,
            pos0: (pos0 + a - base) as u32,
            cap: rows as u32,
            scale,
            nh: nh as u32,
            ns: ns as u32,
            tps: tps as u32,
            window: window as u32,
            stride: kv.ix_npool as u32,
            kp: kv.ix_kp as u32,
            pad: [0; 2],
        };
        // the gqa8 kernel's parameters are the first 32 bytes (it has no window or picks, and its 8th word is padding)
        let pb = bytemuck::bytes_of(&pr);
        let ub = g.dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: if kern == Kern::Gqa8 { &pb[..32] } else { pb },
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut me =
            vec![e(0, &ub), e(1, qb), e(2, &ly.k), e(3, &ly.v), e(4, &ly.ks), e(5, &ly.vs), e(6, ob), e(7, mlb)];
        if sinks.is_some() {
            me.push(e(9, skb));
        }
        if qsa {
            me.push(e(10, selb));
            me.push(e(11, nvb));
        }
        let bg = g.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pp.main.get_bind_group_layout(0),
            entries: &me,
        });
        let mut enc = g.dev.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pp.main);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(mc.div_ceil(tq) as u32, nkv as u32, ns as u32);
        }
        let at = (a * qd * 4) as u64;
        if ns == 1 {
            enc.copy_buffer_to_buffer(ob, 0, rb, at, cb);
        } else {
            // the parts merged into fo, one thread per 4 outputs
            let mut ce = vec![e(0, &ub), e(6, ob), e(7, mlb), e(8, fob)];
            if sinks.is_some() {
                ce.push(e(9, skb));
            }
            let cg = g.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pp.comb.get_bind_group_layout(0),
                entries: &ce,
            });
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pp.comb);
                cp.set_bind_group(0, &cg, &[]);
                cp.dispatch_workgroups((mc * nh * hd / 4).div_ceil(256) as u32, 1, 1);
            }
            enc.copy_buffer_to_buffer(fob, 0, rb, at, cb);
        }
        g.queue.submit([enc.finish()]);
    }
    let t1 = std::time::Instant::now();
    // the CPU's tokens while the GPU runs
    if m < n {
        // the CPU's tokens keep their picks too (rows from token m on)
        let sel = qsa.then(|| crate::fattn::Sel {
            rows: &kv.sel[m * kv.ix_npool..],
            nv: &kv.nv[m..],
            stride: kv.ix_npool,
            kp: kv.ix_kp,
        });
        f.run_block(kv, &q[m * qd..], qd, n - m, (pos0 + m) as u32, &mut out[m * qd..], sel.as_ref());
    }
    let t2 = std::time::Instant::now();
    let sl = rb.slice(..bytes);
    let mapped = std::sync::Arc::new(AtomicBool::new(false));
    let flag = mapped.clone();
    sl.map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), Ordering::Relaxed));
    let waited = g.dev.poll(wgpu::PollType::wait_indefinitely()).is_ok();
    let t3 = std::time::Instant::now();
    let fault = FAULT.lock().unwrap_or_else(|p| p.into_inner()).take();
    let read = (waited && mapped.load(Ordering::Relaxed) && fault.is_none())
        .then(|| sl.get_mapped_range().ok().map(|r| out[..m * qd].copy_from_slice(bytemuck::cast_slice(&r))))
        .flatten();
    if read.is_none() {
        // the GPU failed partway (out of memory in a submission, a lost device): its tokens again on the CPU, so the
        // block is still right, and the tier off from here
        let sel =
            qsa.then(|| crate::fattn::Sel { rows: &kv.sel[..], nv: &kv.nv[..], stride: kv.ix_npool, kp: kv.ix_kp });
        f.run_block(kv, q, qd, m, pos0 as u32, &mut out[..m * qd], sel.as_ref());
        turn_off(&mut g, &fault.unwrap_or_else(|| "its result could not be read back".to_string()));
        return Some(hi);
    }
    rb.unmap();
    for (i, d) in [t1 - t0, t2 - t1, t3 - t2, t3.elapsed()].into_iter().enumerate() {
        NS[i].fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }
    let calls = NS[4].fetch_add(1, Ordering::Relaxed) + 1;
    g.pace(nh, m, pos0, (t3 - t0).as_secs_f64());
    g.adapt(at, n, m, pos0, (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64());
    // TORC_GPU_STATS=1: where the time went so far, every 256 blocks (a server never reaches torc-run's summary)
    static STATS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switches::on("TORC_GPU_STATS"));
    if *STATS && calls.is_multiple_of(256) {
        let s = |i: usize| NS[i].load(Ordering::Relaxed) as f64 * 1e-9;
        eprintln!(
            "gpu attention: {calls} blocks; s writing {:.2}, CPU's tokens {:.2}, waiting {:.2}, reading back {:.2}; share now {:.2}",
            s(0),
            s(1),
            s(2),
            s(3),
            g.last
        );
    }
    // TORC_GPU_CHECK=1: the GPU's tokens again on the CPU, and the largest difference (debugging the tier)
    if crate::switches::on("TORC_GPU_CHECK") {
        let mut c = vec![0f32; m * qd];
        let sel =
            qsa.then(|| crate::fattn::Sel { rows: &kv.sel[..], nv: &kv.nv[..], stride: kv.ix_npool, kp: kv.ix_kp });
        f.run_block(kv, q, qd, m, pos0 as u32, &mut c, sel.as_ref());
        let (mut d, mut a) = (0f32, 0f32);
        for (x, y) in out[..m * qd].iter().zip(&c) {
            d = d.max((x - y).abs());
            a = a.max(y.abs());
        }
        eprintln!("gpu check: block of {n} at {pos0} (GPU {m}): largest difference {d:.3e}, largest output {a:.3e}");
        if qsa {
            let pv = kv.nv[..m].iter().filter(|&&x| x != u32::MAX);
            let (picked, nvmax) = (pv.clone().count(), pv.max().copied().unwrap_or(0));
            eprintln!(
                "gpu check: picks: pools {} of {} positions, GPU tokens with picks {picked} of {m}, largest nv {nvmax}, cap {}",
                kv.ix_npool, kv.ix_kp, kv.cap
            );
        }
        eprintln!(
            "gpu check: shape: {nh} query heads, {nkv} KV heads, head dim {hd}, 16-bit cache {k16}, kernel {}, slices {}",
            kern.name(),
            cuts.len()
        );
    }
    Some(hi)
}
