//! weights.rs - the model file, memory-mapped.
//!
//! Resident tensors and expert slots are slices of the page cache, read without a copy. A token of the 35B touches
//! about 360 expert slots, so copying them would mean hundreds of MB of memcpy a token.
//!
//! Mapless (25 Nov 2026): while any mapping of the file is alive, Windows throttles every unbuffered read of that file,
//! wherever it lands (4 KiB reads 280k/s -> 12k/s; 1.75 MB expert reads 6.8 -> 3.0 GB/s on a FireCuda 530), and
//! dropping the mapping clears it at once. So when the expert cache streams the experts from the file, the load copies
//! every resident tensor into its own memory and then drops the mapping (TORC_MAPLESS=0 keeps it). Design:
//! WORK/torc-gpu-plan/mapping-fix.md.

use format::read::Model as File;
use format::{Experts, Pack, Result, Tensor, err};
use memmap2::Mmap;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Weights {
    pub file: File,
    /// the model file's mapping; None once a mapless load has copied the resident tensors out and dropped it
    map: Option<Mmap>,
    /// the routed experts go through the expert cache when the file is bigger than RAM (ecache.rs)
    pub cache: Option<crate::ecache::ExpertCache>,
    /// tensor name -> index in file.tensors (the directory would otherwise be searched for every token)
    index: std::collections::HashMap<String, usize>,
    warm: Option<Warm>,
    /// resident tensors copied into the process's own memory (tensor index -> bytes), see `own_resident`
    own: std::collections::HashMap<usize, Owned>,
    /// TORC_KLAY=1: for each MoE layer's experts, whether their K-quant matrices have been laid out yet (klay_group;
    /// empty when off)
    klay: Vec<std::sync::OnceLock<()>>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    /// TORC_KLAY=1: every layer's pages of the laid-out types have been handed back once (on klay_group's first call)
    klay_all: std::sync::OnceLock<()>,
    /// TORC_EXPERTS_LARGE=1: for each MoE layer, its packs' copy on large pages and the file offset it starts at
    /// (None: read from the mapping; empty when off)
    exl: Vec<Option<(&'static [u8], usize)>>,
    /// a mapless load: the tables read a row at a time that are not copied (Flash Next's n-gram table), by tensor
    /// index, read from the file (rows.rs)
    tables: std::collections::HashMap<usize, crate::runtime::rows::RowReader>,
}

/// Bytes in 64-byte aligned storage, as the file's tensors are (the kernels read them as u16, f32 or blocks).
struct Owned {
    buf: Vec<A64>,
    len: usize,
    /// the copy is in a large-page block instead (own_resident with TORC_OWN_LARGE=1); buf is then empty
    large: Option<&'static [u8]>,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct A64([u8; 64]);

impl Owned {
    fn copy(src: &[u8]) -> Owned {
        let mut buf = vec![A64([0; 64]); src.len().div_ceil(64)];
        // SAFETY: buf holds at least src.len() bytes
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), buf.as_mut_ptr() as *mut u8, src.len()) };
        Owned { buf, len: src.len(), large: None }
    }

    fn bytes(&self) -> &[u8] {
        if let Some(l) = self.large {
            return l;
        }
        // SAFETY: the first len bytes of buf were written by copy
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.len) }
    }
}

impl Drop for Weights {
    // the warm-up reads the mapping, so stop it and wait for it before the fields (the map) are dropped
    fn drop(&mut self) {
        if let Some(w) = self.warm.as_mut() {
            w.stop.store(true, Ordering::Relaxed);
            if let Some(t) = w.thread.take() {
                let _ = t.join();
            }
        }
    }
}

/// One resident tensor: its entry and its bytes.
pub struct TensorRef<'a> {
    pub t: &'a Tensor,
    pub data: &'a [u8],
}

impl TensorRef<'_> {
    /// The row length in weights (ggml's dims[0]) and the row count (the product of the rest).
    pub fn shape(&self) -> (usize, usize) {
        let cols = self.t.dims[0] as usize;
        let rows = self.t.dims[1..self.t.n_dims as usize].iter().product::<u64>() as usize;
        (cols, rows.max(1))
    }

    pub fn row_bytes(&self) -> usize {
        self.data.len() / self.shape().1
    }

    /// The tensor as f32: F32 as it is, and BF16 (type 30: the MTP block's router and shared gate) widened exactly.
    pub fn f32s(&self) -> Result<Vec<f32>> {
        match self.t.ty {
            0 => Ok(self.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
            30 => Ok(self
                .data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()),
            // F16 widened exactly (the qwen3next shared-expert gate)
            1 => Ok(self.data.chunks_exact(2).map(|c| kernels::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            ty => err(format!("{}: wanted F32, F16 or BF16, file has type {ty}", self.t.name)),
        }
    }
}

/// Reads one byte of every 4 KiB page of p[..n], until stop is set.
fn touch(p: *const u8, n: usize, stop: &AtomicBool) {
    let mut s = 0u8;
    for i in (0..n).step_by(4096) {
        if i % (64 << 20) == 0 && stop.load(Ordering::Relaxed) {
            break;
        }
        // SAFETY: i < n, inside the mapping, which outlives the warm-up (Weights joins it before unmapping)
        s = s.wrapping_add(unsafe { std::ptr::read_volatile(p.add(i)) });
    }
    std::hint::black_box(s);
}

/// Takes the whole pages inside d out of the process's working set (on Windows, VirtualUnlock on pages that were never
/// locked does that). Mapped file pages go to the standby list, where they are free for anything else but can be read
/// back without the disk if touched.
#[cfg(target_arch = "aarch64")]
fn drop_pages(d: &[u8]) {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn VirtualUnlock(addr: *const u8, size: usize) -> i32;
        }
        const PAGE: usize = 4096;
        let a = (d.as_ptr() as usize).next_multiple_of(PAGE);
        let b = (d.as_ptr() as usize + d.len()) / PAGE * PAGE;
        if b > a {
            // SAFETY: [a, b) lies inside d, a live mapping, and the call only changes residency, never the contents
            unsafe { VirtualUnlock(a as *const u8, b - a) };
        }
    }
    #[cfg(not(windows))]
    let _ = d;
}

/// The physical memory installed, in bytes (None where it is not known). Where it is not known a model is taken not to
/// fit, and runs through a small expert cache: several times slower than mapped whole.
pub(crate) fn total_phys() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Status {
            len: u32,
            load: u32,
            total_phys: u64,
            avail_phys: u64,
            rest: [u64; 5],
        }
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(s: *mut Status) -> i32;
        }
        let mut s =
            Status { len: std::mem::size_of::<Status>() as u32, load: 0, total_phys: 0, avail_phys: 0, rest: [0; 5] };
        if unsafe { GlobalMemoryStatusEx(&mut s) } != 0 {
            return Some(s.total_phys);
        }
    }
    #[cfg(target_os = "linux")]
    if let Some(b) = meminfo("MemTotal:") {
        return Some(b);
    }
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                old: *mut std::ffi::c_void,
                old_len: *mut usize,
                new: *mut std::ffi::c_void,
                new_len: usize,
            ) -> i32;
        }
        let (mut b, mut len) = (0u64, std::mem::size_of::<u64>());
        // SAFETY: hw.memsize is a u64, and b and len are that size
        let r = unsafe { sysctlbyname(c"hw.memsize".as_ptr(), (&raw mut b).cast(), &mut len, std::ptr::null_mut(), 0) };
        if r == 0 && b > 0 {
            return Some(b);
        }
    }
    avail_phys()
}

/// ggml's name for a weight type, for the messages (the types a .torc file can hold that have no kernel here, and
/// the common ones).
/// The first weight of `file` (a tensor or an expert piece) whose type ToRc has no kernel for, with that type.
pub fn unsupported_weight(file: &File) -> Option<(String, u32)> {
    let pieces = file.experts.iter().flat_map(|x| x.pack.pieces().map(|p| (&p.name, p.ty)));
    file.tensors
        .iter()
        .map(|t| (&t.name, t.ty))
        .chain(pieces)
        .find(|&(_, ty)| kernels::dequant::block_geometry(ty).is_none())
        .map(|(name, ty)| (name.clone(), ty))
}

/// What the load and the preflight say about such a weight.
pub fn unsupported_message(name: &str, ty: u32) -> String {
    format!(
        "{name} is {}, a type ToRc has no kernel for yet. Use a file of this model without it (a Q4_K_M or Q4_0, say)",
        type_name(ty)
    )
}

fn type_name(ty: u32) -> String {
    let n = match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        30 => "BF16",
        39 => "MXFP4",
        _ => return format!("ggml type {ty}"),
    };
    format!("{n} (ggml type {ty})")
}

/// A line of /proc/meminfo ("MemTotal:", "MemAvailable:"), in bytes.
#[cfg(target_os = "linux")]
fn meminfo(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = s.lines().find_map(|l| l.strip_prefix(key))?.trim().strip_suffix("kB")?.trim().parse().ok()?;
    Some(kb << 10)
}

/// The physical memory free now, in bytes (None where it is not known).
fn avail_phys() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Status {
            len: u32,
            load: u32,
            total_phys: u64,
            avail_phys: u64,
            rest: [u64; 5],
        }
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(s: *mut Status) -> i32;
        }
        let mut s =
            Status { len: std::mem::size_of::<Status>() as u32, load: 0, total_phys: 0, avail_phys: 0, rest: [0; 5] };
        if unsafe { GlobalMemoryStatusEx(&mut s) } != 0 {
            return Some(s.avail_phys);
        }
    }
    // Linux: what the kernel says can be had without swapping (free memory and the file cache it would give back)
    #[cfg(target_os = "linux")]
    if let Some(b) = meminfo("MemAvailable:") {
        return Some(b);
    }
    None
}

/// TORC_OWN=1: the resident tensors that every token reads are copied into the process's own memory at load. Read
/// through the mapping, they are file pages, which are the first pages Windows takes back when memory is tight (with a
/// big expert cache), and decoding then re-reads them from the disk every token. Tables used one row per token (the
/// token embedding, Flash Next's n-gram table) are not copied. Only with TORC_OWN=1 (see below), or `all`: every
/// resident tensor, the token embedding included, for a load that drops the mapping afterwards (`mapless`).
fn own_resident(file: &File, map: &Mmap, all: bool) -> Result<std::collections::HashMap<usize, Owned>> {
    let mut own = std::collections::HashMap::new();
    // opt-in only: an owned copy is counted against the cache budget as well as the mapping, so the mapped file is
    // the default
    if !all && crate::switches::get("TORC_OWN").as_deref() != Some("1") && !crate::switches::on("TORC_OWN_LARGE") {
        return Ok(own);
    }
    let t0 = std::time::Instant::now();
    let r0 = file.resident.offset as usize;
    let mut bytes = 0usize;
    // with all, only the n-gram table stays in the file (tens of GB; rows.rs reads its rows)
    let skip = |t: &format::Tensor| {
        t.name == "per_layer_token_embd.weight" || (!all && t.name == "token_embd.weight")
    };
    // TORC_OWN_LARGE=1: the copies go in one block of large pages, each 64-aligned within it, and the
    // mapped pages are handed back (the copy is what the products read, so only one copy is held, as klay_group does).
    // Otherwise, or if the system refuses, each copy gets a buffer from the allocator. The bytes are the same either
    // way.
    let mut arena = if crate::switches::on("TORC_OWN_LARGE") {
        let need: usize = file.tensors.iter().filter(|t| !skip(t)).map(|t| (t.bytes as usize).div_ceil(64) * 64).sum();
        let a = crate::ecache_own::large_leak(need);
        eprintln!(
            "resident tensors: {} MB on large pages: {}",
            need >> 20,
            if a.is_some() { "yes" } else { "no (allocator)" }
        );
        a
    } else {
        None
    };
    for (i, t) in file.tensors.iter().enumerate() {
        if skip(t) {
            continue;
        }
        let (a, b) = (r0 + t.offset as usize, r0 + (t.offset + t.bytes) as usize);
        let o = match arena.take() {
            Some(ar) => {
                let (mine, rest) = ar.split_at_mut((b - a).div_ceil(64) * 64);
                arena = Some(rest);
                mine[..b - a].copy_from_slice(&map[a..b]);
                #[cfg(target_arch = "aarch64")]
                drop_pages(&map[a..b]);
                Owned { buf: Vec::new(), len: b - a, large: Some(&mine[..b - a]) }
            }
            None => Owned::copy(&map[a..b]),
        };
        own.insert(i, o);
        bytes += b - a;
    }
    eprintln!("resident tensors in own memory: {:.0} MB in {:.1} s", bytes as f64 / 1e6, t0.elapsed().as_secs_f64());
    Ok(own)
}

/// The background warm-up: a thread that takes the mapping's first-touch page faults before decoding needs the pages
/// (each process pays them once per page). It only runs when the file fits in 80% of free memory: a bigger model
/// streams its experts, and touching all of it would evict them.
struct Warm {
    stop: Arc<AtomicBool>,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    /// the warm-up has touched everything (klay_group waits for it, because pages it hands back must stay handed back)
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Weights {
    pub fn load(path: &str) -> Result<Weights> {
        Weights::load_with(path, true)
    }

    /// `cache`: false means never use an expert cache, whatever TORC_CACHE_MB says (for the MTP draft file, which is
    /// small and mapped whole, and must not get a cache of its own from the main model's TORC_CACHE_MB).
    pub fn load_with(path: &str, cache: bool) -> Result<Weights> {
        let file = File::open(path)?;
        // a weight of a type ToRc has no kernel for stops the load here, named, rather than panicking in the first
        // reply that reaches it (a file holding Q8_K weights, say: the format can carry it, the engine cannot read it)
        if let Some((name, ty)) = unsupported_weight(&file) {
            return err(format!("{path}: {}", unsupported_message(&name, ty)));
        }
        let handle = fs::File::open(path)?;
        // SAFETY: the model file is not modified while it is mapped (it is only read, and model files do not change)
        let map = unsafe { Mmap::map(&handle) }.map_err(|e| format::Error(format!("cannot map {path}: {e}")))?;
        if map.len() as u64 != file.header.file_bytes {
            return err("file size changed since it was opened");
        }
        // TORC_WARM=0 turns the warm-up off, 1 forces it on, and wait does it before returning (a benchmark's steady
        // state: a server stays loaded, so its warm-up is long done) "fits" means within 80% of the installed memory,
        // not of the free memory: the standby list counts as used but is given back as the mapping is touched, so free
        // memory would turn a model that fits into a streamed one
        // A TORC_CACHE_MB above 0 asks for the expert cache, so the experts are streamed through it even when the file
        // would fit: mapping the whole file as well would hold the model twice.
        let streamed = cache && crate::switches::num::<u64>("TORC_CACHE_MB").is_some_and(|mb| mb > 0);
        let fits = !streamed && total_phys().is_some_and(|a| map.len() as u64 <= a / 10 * 8);
        // TORC_KLAY_LARGE=1: the laid-out copies' large pages are reserved now, before the warm-up breaks up free
        // memory (repack::klay_reserve)
        #[cfg(target_arch = "aarch64")]
        if crate::repack::klay_large()
            && crate::repack::klay_on()
            && fits
            && cache
            && !crate::switches::given("TORC_CACHE_MB")
        {
            // each layer's bytes, as klay_group will ask for them
            let needs: Vec<usize> = file
                .experts
                .iter()
                .map(|x| {
                    let one: usize = x
                        .pack
                        .pieces()
                        .into_iter()
                        .filter(|m| crate::repack::klay_type_on(m.ty) && m.rows % 4 == 0)
                        .map(|m| crate::repack::klay_bytes(m.rows as usize, m.cols as usize, m.ty))
                        .sum();
                    one * x.n_expert as usize
                })
                .collect();
            let total: usize = needs.iter().sum();
            if total > 0 {
                let t0 = std::time::Instant::now();
                let got = crate::repack::klay_reserve(&needs);
                eprintln!(
                    "klay: large pages reserved for {got} of {} layers ({} MB asked, {:.1} s)",
                    needs.iter().filter(|&&b| b > 0).count(),
                    total >> 20,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // mapless: the expert cache will stream the experts from this file (the same test as the budget below), so the
        // mapping is dropped once the resident tensors are copied out (Flash Next's n-gram table, tens of GB read a row
        // a token, is not copied: rows.rs reads its rows from the file). TORC_MAPLESS=0 keeps the mapping. A build with
        // the GPU tiers keeps it unless TORC_MAPLESS=1: their expert paths still read the mapping.
        let will_cache = cache
            && file.expert_region.bytes > 0
            && match crate::switches::num::<u64>("TORC_CACHE_MB") {
                Some(mb) => mb > 0,
                None => !fits,
            };
        let mapless = will_cache
            && match crate::switches::get("TORC_MAPLESS").as_deref() {
                Some("0") => false,
                Some("1") => true,
                _ => !cfg!(feature = "gpu"),
            };
        let mode = crate::switches::get("TORC_WARM");
        let warm = match mode.as_deref() {
            // the warm-up touches the mapping, which a mapless load is about to drop
            _ if mapless => false,
            Some("0") => false,
            Some("1") | Some("wait") => true,
            _ => fits,
        };
        // a file bigger than RAM (with the warm-up forced on): only the resident section, less the n-gram table (Flash
        // Next's 28.8 GB, looked up one row per token); its experts stream (ecache.rs)
        let ranges: Vec<(usize, usize)> = if fits {
            vec![(0, map.len())]
        } else {
            let r = &file.resident;
            let (r0, r1) = (r.offset as usize, (r.offset + r.bytes) as usize);
            match file.tensors.iter().find(|t| t.name == "per_layer_token_embd.weight") {
                Some(t) => vec![(r0, r0 + t.offset as usize), (r0 + (t.offset + t.bytes) as usize, r1)],
                None => vec![(r0, r1)],
            }
        };
        if warm && mode.as_deref() == Some("wait") {
            let t = std::time::Instant::now();
            for &(a, b) in &ranges {
                touch(map[a..].as_ptr(), b - a, &AtomicBool::new(false));
            }
            eprintln!("warm-up: {:.1} s", t.elapsed().as_secs_f64());
        }
        let warm = (warm && mode.as_deref() != Some("wait")).then(|| {
            let stop = Arc::new(AtomicBool::new(false));
            let done = Arc::new(AtomicBool::new(false));
            let (p, s, d) = (map.as_ptr() as usize, stop.clone(), done.clone());
            let rs = ranges.clone();
            let thread = std::thread::spawn(move || {
                for (a, b) in rs {
                    touch((p + a) as *const u8, b - a, &s);
                }
                d.store(true, Ordering::Release);
            });
            Warm { stop, done, thread: Some(thread) }
        });
        let index = file.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        let own = own_resident(&file, &map, mapless)?;
        let mut w = Weights {
            file,
            map: Some(map),
            index,
            warm,
            cache: None,
            own,
            klay: Vec::new(),
            klay_all: std::sync::OnceLock::new(),
            exl: Vec::new(),
            tables: std::collections::HashMap::new(),
        };
        // a region hash of zero is not checked. A shard (`<model>.layers`, written by torc check --layers once every
        // tensor and pack it holds has passed its CRC) has holes in the region, so its hash cannot match, and it is not
        // checked here
        let shard = std::path::Path::new(&format!("{path}.layers")).exists();
        if shard {
            eprintln!("shard: resident section hash not checked (torc check --layers verified this box's tensors)");
        }
        let check = |w: &Weights| -> Result<()> {
            if !shard && w.file.resident.hash != [0; 32] && *blake3::hash(w.resident()).as_bytes() != w.file.resident.hash
            {
                return err("resident section hash mismatch");
            }
            Ok(())
        };
        // mapless: every resident tensor is in own memory now, so after the hash check (the mapping's last reader) the
        // mapping goes, before the expert cache opens and warms itself from the file at full speed
        if mapless {
            check(&w)?;
            // the tables own_resident left in the file: their rows are read from it directly
            for (i, t) in w.file.tensors.iter().enumerate() {
                if w.own.contains_key(&i) {
                    continue;
                }
                let rows = (t.dims[1..t.n_dims as usize].iter().product::<u64>()).max(1);
                let rb = (t.bytes / rows) as usize;
                let base = w.file.resident.offset + t.offset;
                let r = crate::runtime::rows::RowReader::new(path, base, rb)
                    .map_err(|e| format::Error(format!("{path}: {}: {e}", t.name)))?;
                w.tables.insert(i, r);
            }
            w.map = None;
            eprintln!("mapless: resident tensors in own memory, the model file no longer mapped (TORC_MAPLESS=0 keeps it)");
        }
        // the expert cache: TORC_CACHE_MB, or by default, when the file does not fit, the free memory less the
        // resident weights and a margin
        let budget = match crate::switches::num::<u64>("TORC_CACHE_MB") {
            _ if !cache => 0,
            Some(mb) => mb << 20,
            None if !fits && w.file.expert_region.bytes > 0 => {
                let ple =
                    w.file.tensors.iter().find(|t| t.name == "per_layer_token_embd.weight").map_or(0, |t| t.bytes);
                let res = w.file.resident.bytes - ple;
                // a pipeline machine (pipe::own_layers) only touches its own layers' resident weights: their share
                // (+ 1 GB for the embeddings and the head) plus a 3 GB margin; a single machine touches all of them,
                // plus 6 GB
                let nl = w.file.blocks.iter().map(|b| b.layer + 1).max().unwrap_or(1).max(1) as u64;
                let span = crate::pipe::own_layers(nl as usize)
                    .map(|(a, b)| (b as u64).min(nl).saturating_sub(a as u64))
                    .filter(|&k| k > 0 && k < nl);
                let (own, margin) = match span {
                    Some(k) => (res * k / nl + (1 << 30), 3u64 << 30),
                    None => (res, 6u64 << 30),
                };
                // at least 2 GB, and never more than the experts take (a small model can need less than that)
                avail_phys().unwrap_or(0).saturating_sub(own + margin).max(2 << 30).min(w.file.expert_region.bytes)
            }
            None => 0,
        };
        if budget > 0 && w.file.expert_region.bytes > 0 {
            let n_layer = w.file.blocks.iter().map(|b| b.layer + 1).max().unwrap_or(0);
            // the widest layer's: a model whose first layers are dense (one expert) keeps its usage rows that wide
            let n_expert = w.file.experts.iter().map(|x| x.n_expert).max().unwrap_or(0);
            let tc = std::time::Instant::now();
            // I/O threads: TORC_IO_THREADS, else up to 8, one per CPU, for more reads in flight at once
            let nio = crate::switches::num::<i32>("TORC_IO_THREADS")
                .filter(|&n| n > 0)
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get().min(8) as i32));
            let c = crate::ecache::ExpertCache::open(path, budget, nio, w.file.header.model_id, n_layer, n_expert)?;
            if crate::switches::get("TORC_PROF").is_some_and(|v| v != "0") {
                eprintln!("load: expert cache opened in {:.1} s", tc.elapsed().as_secs_f64());
            }
            eprintln!("expert cache: {} MB", budget >> 20);
            w.cache = Some(c);
        }
        // TORC_Q4X4=1: the resident Q4_0 matrices are also repacked for SMMLA (repack.rs)
        #[cfg(target_arch = "aarch64")]
        if crate::repack::on() {
            let t0 = std::time::Instant::now();
            let mut bytes = 0usize;
            let names: Vec<(String, usize, usize)> = w
                .file
                .tensors
                .iter()
                .filter(|t| {
                    t.ty == kernels::GGML_Q4_0 && t.n_dims == 2 && t.dims[1] % 4 == 0 && !t.name.contains("token_embd")
                })
                .map(|t| (t.name.clone(), t.dims[0] as usize, t.dims[1] as usize))
                .collect();
            for (name, cols, rows) in &names {
                if let Some(r) = w.tensor(name) {
                    bytes += crate::repack::register(r.data, *rows, *cols);
                }
            }
            eprintln!(
                "q4x4: {} matrices repacked, {:.0} MB in {:.1} s",
                names.len(),
                bytes as f64 / 1e6,
                t0.elapsed().as_secs_f64()
            );
            // TORC_Q4X4_EXPERTS=1 (no expert cache): the experts' Q4_0 matrices are repacked too (their mapped pages
            // are read once here, and then only the copy). The bits are the same. Off by default: the experts see 1-2
            // vectors and are bound by bandwidth
            if w.cache.is_none() && crate::switches::on("TORC_Q4X4_EXPERTS") {
                let t0 = std::time::Instant::now();
                let mut mats = Vec::new();
                for g in 0..w.file.experts.len() as u32 {
                    for e in 0..w.layer_experts(g).n_expert {
                        let (pack, shape) = w.expert(g, e)?;
                        for m in shape.pieces() {
                            if m.ty == kernels::GGML_Q4_0 && m.rows % 4 == 0 {
                                mats.push((m.of(pack), m.rows as usize, m.cols as usize));
                            }
                        }
                    }
                }
                let b = crate::repack::register_many(&mats);
                eprintln!(
                    "q4x4: {} expert matrices repacked, {:.0} MB in {:.1} s",
                    mats.len(),
                    b as f64 / 1e6,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // TORC_KLAY=1 (no expert cache): the experts' Q4_K / Q5_K matrices are also laid out for SMMLA (repack.rs,
        // klay_on), each layer's experts at their first use (klay_group), so on two machines each one lays out only the
        // layers it runs, wherever the split moves to
        if crate::repack::klay_on() && w.cache.is_none() {
            w.klay = (0..w.file.experts.len()).map(|_| std::sync::OnceLock::new()).collect();
            eprintln!("klay: on, each layer's experts laid out at its first use");
        }
        // TORC_KLAY_HEAD=1: a Q6_K output head is laid out like the experts' Q6_K (repack.rs, q6_lay4; the same size
        // again), so its products go through q6_lay_matmul for 1-3 vectors and gemm_q6_lay from 4 (moe.rs use_q6m),
        // with the same bits
        #[cfg(target_arch = "aarch64")]
        if crate::repack::klay_on()
            && crate::repack::klay_type_on(14)
            && crate::switches::on("TORC_KLAY_HEAD")
            && let Ok(t) = w.head()
        {
            let (cols, rows) = t.shape();
            if t.t.ty == 14 && rows % 4 == 0 {
                let t0 = std::time::Instant::now();
                let b = crate::repack::register_many_klay(&[(t.data, rows, cols, 14)]);
                eprintln!("klay: the head laid out, {} MB in {:.1} s", b >> 20, t0.elapsed().as_secs_f64());
            }
        }
        // TORC_EXPERTS_LARGE=1 (no expert cache, with the file in RAM): each layer's expert packs are copied into a
        // block of large pages and their mapped pages handed back (only one copy is held), so a token's experts cost a
        // few TLB entries instead of one per 4 KiB page. Not for the layers laid out by klay (their copies are what the
        // products read, and they are already on large pages with TORC_KLAY_LARGE). The bytes are the same, so the bits
        // are the same. (not with TORC_Q4X4's repacked experts, which are found by their address in the mapping)
        if crate::switches::on("TORC_EXPERTS_LARGE") && w.cache.is_none() && fits && !crate::repack::on() {
            let t0 = std::time::Instant::now();
            if let Some(wm) = w.warm.as_ref() {
                // a background warm-up would touch the handed-back pages again
                while !wm.done.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
            let (mut n_large, mut bytes) = (0usize, 0usize);
            let mut exl = Vec::with_capacity(w.file.experts.len());
            for g in 0..w.file.experts.len() {
                let x = &w.file.experts[g];
                #[cfg(target_arch = "aarch64")]
                let laid = !w.klay.is_empty() && x.pack.pieces().iter().any(|m| crate::repack::klay_type_on(m.ty));
                #[cfg(not(target_arch = "aarch64"))]
                let laid = false;
                if laid {
                    exl.push(None);
                    continue;
                }
                // a layer's packs are one run of whole pages, so each pack keeps its alignment in the copy
                let (at, n) = x.span();
                let base = (w.file.expert_region.offset + at) as usize;
                let end = base + n as usize;
                match crate::ecache_own::large_leak(end - base) {
                    Some(a) => {
                        a[..end - base].copy_from_slice(&w.mapped()[base..end]);
                        #[cfg(target_arch = "aarch64")]
                        drop_pages(&w.mapped()[base..end]);
                        n_large += 1;
                        bytes += end - base;
                        let a: &'static [u8] = a;
                        exl.push(Some((&a[..end - base], base)));
                    }
                    None => exl.push(None),
                }
            }
            eprintln!(
                "experts on large pages: {n_large} of {} layers, {} MB in {:.1} s",
                w.file.experts.len(),
                bytes >> 20,
                t0.elapsed().as_secs_f64()
            );
            w.exl = exl;
        }
        if !mapless {
            check(&w)?;
        }
        Ok(w)
    }

    /// The mapping, for the paths that only run with it alive (no expert cache, so never mapless).
    fn mapped(&self) -> &Mmap {
        self.map.as_ref().expect("weights: the mapping is needed here but a mapless load dropped it")
    }

    /// Whether the model file is still mapped (false after a mapless load).
    pub fn is_mapped(&self) -> bool {
        self.map.is_some()
    }

    fn resident(&self) -> &[u8] {
        let r = &self.file.resident;
        &self.mapped()[r.offset as usize..(r.offset + r.bytes) as usize]
    }

    pub fn tensor(&self, name: &str) -> Option<TensorRef<'_>> {
        let i = *self.index.get(name)?;
        let t = &self.file.tensors[i];
        if let Some(v) = self.own.get(&i) {
            return Some(TensorRef { t, data: v.bytes() });
        }
        // a mapless load: the tables it left in the file have no bytes in memory (table_rows reads them)
        self.map.as_ref()?;
        Some(TensorRef { t, data: &self.resident()[t.offset as usize..(t.offset + t.bytes) as usize] })
    }

    /// A tensor's directory entry (type, shape), whether or not its bytes are in memory.
    pub fn entry(&self, name: &str) -> Option<&Tensor> {
        self.index.get(name).map(|&i| &self.file.tensors[i])
    }

    /// Rows of a table looked up a row per token (the n-gram table), row_bytes each, in the order asked: from memory
    /// when the tensor is there, else read from the file (a mapless load, rows.rs).
    pub fn table_rows(&self, name: &str, rows: &[u32]) -> Result<Vec<u8>> {
        let &i = self.index.get(name).ok_or_else(|| format::Error(format!("missing tensor {name}")))?;
        if let Some(r) = self.tables.get(&i) {
            return r.read(rows).map_err(|e| format::Error(format!("{name}: rows from the file: {e}")));
        }
        let t = self.need(name)?;
        let rb = t.row_bytes();
        let mut out = Vec::with_capacity(rows.len() * rb);
        for &r in rows {
            let r = r as usize;
            if (r + 1) * rb > t.data.len() {
                return err(format!("{name}: row {r} out of range"));
            }
            out.extend_from_slice(&t.data[r * rb..(r + 1) * rb]);
        }
        Ok(out)
    }

    pub fn need(&self, name: &str) -> Result<TensorRef<'_>> {
        self.tensor(name).ok_or_else(|| format::Error(format!("missing tensor {name}")))
    }

    /// The output head: output.weight, or the token embeddings for a model that ties the two (the small Qwen3.5s), as
    /// llama.cpp does.
    pub fn head(&self) -> Result<TensorRef<'_>> {
        self.tensor("output.weight")
            .or_else(|| self.tensor("token_embd.weight"))
            .ok_or_else(|| format::Error("missing tensor output.weight (and no token_embd.weight to tie it to)".into()))
    }

    /// The layer's tensor `blk.<l>.<suffix>`.
    pub fn layer(&self, l: u32, suffix: &str) -> Result<TensorRef<'_>> {
        self.need(&format!("blk.{l}.{suffix}"))
    }

    /// The routed experts of MoE record g (file.experts[g]).
    pub fn layer_experts(&self, g: u32) -> &Experts {
        &self.file.experts[g as usize]
    }

    /// Experts ids of layer l (MoE record g): their packs and the packs' shape, through the expert cache when there is
    /// one (they are then only valid until the next call, so the caller computes on them first), else slices of the
    /// mapping.
    pub fn experts_of(&self, l: u32, g: u32, ids: &[u32]) -> Result<Vec<(&[u8], &Pack)>> {
        let Some(c) = &self.cache else {
            self.klay_group(g)?;
            return ids.iter().map(|&e| self.expert(g, e)).collect();
        };
        let x = self.layer_experts(g);
        let ptrs = c.take(l, ids)?;
        ids.iter()
            .zip(ptrs)
            .map(|(&e, p)| {
                if e >= x.n_expert {
                    return Err(format::Error(format!("expert {e} out of range")));
                }
                // SAFETY: the cache's slot holds the expert's whole pack and stays live until the cache is asked again
                Ok((unsafe { std::slice::from_raw_parts(p, x.pack.bytes as usize) }, &x.pack))
            })
            .collect()
    }

    /// experts_of in two steps (the cache's ask / settle): the packs, and whether each is in RAM now; the others may
    /// only be read after experts_settle. Without a cache, every pack is the mapping's and all count as in RAM.
    pub fn experts_ask(&self, l: u32, g: u32, ids: &[u32]) -> Result<(Vec<(&[u8], &Pack)>, Vec<bool>)> {
        let Some(c) = &self.cache else {
            self.klay_group(g)?;
            return Ok((ids.iter().map(|&e| self.expert(g, e)).collect::<Result<_>>()?, vec![true; ids.len()]));
        };
        let x = self.layer_experts(g);
        let (ptrs, ready) = c.ask(l, ids)?;
        let packs = ids
            .iter()
            .zip(ptrs)
            .map(|(&e, p)| {
                if e >= x.n_expert {
                    return Err(format::Error(format!("expert {e} out of range")));
                }
                // SAFETY: the cache's slot holds the expert's whole pack and stays live until the next request; the
                // caller only reads a pack that was not in RAM after experts_settle
                Ok((unsafe { std::slice::from_raw_parts(p, x.pack.bytes as usize) }, &x.pack))
            })
            .collect::<Result<_>>()?;
        Ok((packs, ready))
    }

    /// Completes the reads of the last experts_ask (nothing to do without a cache).
    pub fn experts_settle(&self) -> Result<()> {
        match &self.cache {
            Some(c) => c.settle(),
            None => Ok(()),
        }
    }

    /// TORC_KLAY=1: MoE record g's matrices of the laid-out types (repack::klay_type_on), as (bytes, rows, cols, type).
    #[cfg(target_arch = "aarch64")]
    fn klay_mats(&self, g: u32) -> Result<Vec<(&[u8], usize, usize, u32)>> {
        let mut mats = Vec::new();
        for e in 0..self.layer_experts(g).n_expert {
            let (pack, shape) = self.expert(g, e)?;
            for m in shape.pieces() {
                if crate::repack::klay_type_on(m.ty) && m.rows % 4 == 0 {
                    mats.push((m.of(pack), m.rows as usize, m.cols as usize, m.ty));
                }
            }
        }
        Ok(mats)
    }

    /// TORC_KLAY=1: the layers' own Q3_K matrices (and a Q3_K output head), as (bytes, rows, cols, type), for the
    /// same layout as the experts' Q3_K. Not the token embeddings, which are looked up a row at a time.
    #[cfg(target_arch = "aarch64")]
    fn klay_dense_mats(&self) -> Vec<(&[u8], usize, usize, u32)> {
        if !crate::repack::klay_type_on(11) {
            return Vec::new();
        }
        let mut mats = Vec::new();
        for t in &self.file.tensors {
            if t.ty != 11 || !(t.name.starts_with("blk.") || t.name == "output.weight") {
                continue;
            }
            if let Some(r) = self.tensor(&t.name) {
                let (cols, rows) = r.shape();
                if rows % 4 == 0 {
                    mats.push((r.data, rows, cols, 11));
                }
            }
        }
        mats
    }

    /// TORC_KLAY=1: lays out every layer now, for a process that runs every layer anyway (torc-server on one machine,
    /// or as the main machine of two, which writes alone), so the layout costs load time rather than the first prompt.
    /// Returns the seconds it took (0 when off).
    pub fn klay_all_now(&self) -> Result<f64> {
        if self.klay.is_empty() {
            return Ok(0.0);
        }
        let t0 = std::time::Instant::now();
        for g in 0..self.klay.len() as u32 {
            self.klay_group(g)?;
        }
        Ok(t0.elapsed().as_secs_f64())
    }

    /// TORC_KLAY=1: lays out MoE record g's matrices of the laid-out types (repack::register_many_klay) the first time
    /// it is asked for, then takes their mapped pages out of the working set (to the standby list: the copy is what the
    /// products read now, so the machine holds one copy, not two). The first call hands back every layer's such pages,
    /// since the warm-up touched them all and, on two machines, this one never runs the other machine's layers (a
    /// layer's pages come back from the standby list, or the disk, when it is laid out). Does nothing when off or
    /// already done.
    fn klay_group(&self, g: u32) -> Result<()> {
        let Some(once) = self.klay.get(g as usize) else { return Ok(()) };
        if once.get().is_some() {
            return Ok(());
        }
        #[allow(unused_mut)]
        let mut failed = None;
        once.get_or_init(|| {
            #[cfg(target_arch = "aarch64")]
            {
                let t0 = std::time::Instant::now();
                // a background warm-up still running would touch the pages again after they are handed back
                while self.warm.as_ref().is_some_and(|w| !w.done.load(Ordering::Acquire)) {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                self.klay_all.get_or_init(|| {
                    for h in 0..self.klay.len() as u32 {
                        if let Ok(ms) = self.klay_mats(h) {
                            for (d, ..) in &ms {
                                drop_pages(d);
                            }
                        }
                    }
                    // the layers' own Q3_K matrices too (the DeltaNet gates and the like, in Q3_K_S / Q3_K_M files),
                    // whose prompts otherwise go a row at a time: laid out once, their mapped pages handed back
                    let dense = self.klay_dense_mats();
                    if !dense.is_empty() {
                        let t1 = std::time::Instant::now();
                        let b = crate::repack::register_many_klay(&dense);
                        for (d, ..) in &dense {
                            drop_pages(d);
                        }
                        eprintln!(
                            "klay: {} of the layers' own Q3_K matrices, {:.0} MB in {:.0} ms",
                            dense.len(),
                            b as f64 / 1e6,
                            t1.elapsed().as_secs_f64() * 1e3
                        );
                    }
                });
                let mats = match self.klay_mats(g) {
                    Ok(m) => m,
                    Err(x) => {
                        failed = Some(x);
                        return;
                    }
                };
                let b = crate::repack::register_many_klay(&mats);
                for (d, ..) in &mats {
                    drop_pages(d);
                }
                eprintln!(
                    "klay: layer record {g}: {} matrices, {:.0} MB in {:.0} ms",
                    mats.len(),
                    b as f64 / 1e6,
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
        });
        failed.map_or(Ok(()), Err)
    }

    /// Expert e of MoE record g: its pack (a slice of the mapping, or of its large-page copy) and the pack's shape.
    pub fn expert(&self, g: u32, e: u32) -> Result<(&[u8], &Pack)> {
        let x = self.layer_experts(g);
        if e >= x.n_expert {
            return err(format!("expert {e} out of range"));
        }
        let at = (self.file.expert_region.offset + x.pack_at(e)) as usize;
        let n = x.pack.bytes as usize;
        if let Some(Some((a, base))) = self.exl.get(g as usize) {
            return Ok((&a[at - base..at - base + n], &x.pack));
        }
        match &self.map {
            Some(m) => Ok((&m[at..at + n], &x.pack)),
            // a mapless load: every expert read goes through the expert cache (experts_of / experts_ask)
            None => err(format!("expert {e} of record {g} asked for directly, but a mapless load has no mapping")),
        }
    }
}

#[cfg(all(test, any(windows, target_os = "linux", target_os = "macos")))]
mod mem_tests {
    #[test]
    fn the_machines_memory_is_known() {
        let total = super::total_phys().expect("installed memory");
        assert!(total >= 1 << 28, "{total}");
        // free memory is known on Windows and Linux, and is never more than what is installed
        #[cfg(any(windows, target_os = "linux"))]
        {
            let avail = super::avail_phys().expect("free memory");
            assert!(avail > 0 && avail <= total, "{avail} of {total}");
        }
    }
}

#[cfg(test)]
mod type_tests {
    use format::write::{plan, write};
    use format::*;

    /// A file whose experts' gate is Q8_K (no kernel here) is refused at load, naming the weight and its type, and
    /// the same file with Q8_0 there loads. IQ2_S, which this used to be about, is a weight ToRc can run now, so the
    /// same file with IQ2_S there is now asserted to load.
    #[test]
    fn a_type_with_no_kernel_is_refused_at_load() {
        let dir = std::env::temp_dir().join(format!("torc-types-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tiny = |gate_ty: u32, name: &str| {
            let path = dir.join(name).to_str().unwrap().to_string();
            let piece = |name: &str, ty: u32, rows: u32, cols: u32, at| Piece {
                name: name.into(),
                ty,
                rows,
                cols,
                at,
                bytes: matrix_bytes(ty, rows as u64, cols as u64).unwrap(),
                origin: Origin::default(),
                stride: 0,
            };
            let gate = piece("blk.0.ffn_gate_exps.weight", gate_ty, 64, 256, 0);
            let up = piece("blk.0.ffn_up_exps.weight", 8, 64, 256, page_up(gate.bytes));
            let down = piece("blk.0.ffn_down_exps.weight", 8, 256, 64, page_up(up.at + up.bytes));
            let pack_bytes = page_up(down.at + down.bytes);
            let t = |name: &str, dims: [u64; 4]| Tensor {
                name: name.into(),
                ty: 0,
                n_dims: 2,
                dims,
                bytes: 4 * dims[0] * dims[1],
                ..Default::default()
            };
            let d = Directory {
                metadata: 0u64.to_le_bytes().to_vec(),
                source: None,
                tensors: vec![
                    t("token_embd.weight", [256, 10, 1, 1]),
                    t("blk.0.ffn_gate_inp.weight", [256, 3, 1, 1]),
                    t("blk.0.attn_q.weight", [256, 256, 1, 1]),
                ],
                experts: vec![Experts {
                    layer: 0,
                    n_expert: 3,
                    n_used: 2,
                    d_in: 256,
                    width: 64,
                    first: 0,
                    pack: Pack { bytes: pack_bytes, gate, up, down },
                    crcs: vec![],
                }],
            };
            write(&path, plan(d, "qwen35moe").unwrap(), |_, _, _| Ok(()), |_, _, _| Ok(())).unwrap();
            path
        };
        let bad = tiny(15, "q8k.torc");
        match super::Weights::load_with(&bad, false) {
            Ok(_) => panic!("a Q8_K weight was loaded"),
            Err(e) => assert!(
                e.0.contains("blk.0.ffn_gate_exps.weight is Q8_K (ggml type 15), a type ToRc has no kernel for"),
                "{}",
                e.0
            ),
        }
        assert!(super::Weights::load_with(&tiny(8, "q8.torc"), false).is_ok());
        // the IQ2_S file this test used to be written about: a kernel for it is what makes the model runnable at all,
        // so the load gate has to let it through.
        assert!(super::Weights::load_with(&tiny(22, "iq2s.torc"), false).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }
}
