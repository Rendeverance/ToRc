//! repack.rs - TORC_Q4X4=1 (ARM with FEAT_I8MM): the resident Q4_0 matrices are also held repacked for SMMLA, 4 rows at
//! a time (kernels::neon::q4_0_repack4), so that products over a few vectors read each set of 4 rows as one stream,
//! with nothing left to unpack but a mask and a shift (kernels::neon::gemm_q4x4, with the same bits). It is an
//! experiment ahead of the ToRc v2 layout, and the copy costs RAM the size of those matrices. A copy is looked up by
//! the address of the rows a product is given. Entries live as long as the process (the mapping they mirror is never
//! unmapped while the model runs).

use std::sync::LazyLock;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

#[derive(Clone, Copy)]
struct Entry {
    start: usize,
    end: usize,
    row_bytes: usize,
    /// the packed bytes per 4 source rows (for Q4_0, the same as the source; for the K-quant layout, more)
    pq: usize,
    packed: &'static [u8],
}

// The table as products read it: a published snapshot, never freed (there are a few per load), so a lookup takes no
// lock (a lock read per product would put every thread on one shared counter). LO..HI spans every
// registered matrix, so the experts' products, which fall outside it, are turned away at once.
static TABLE: AtomicPtr<Vec<Entry>> = AtomicPtr::new(std::ptr::null_mut());
static LO: AtomicUsize = AtomicUsize::new(usize::MAX);
static HI: AtomicUsize = AtomicUsize::new(0);
#[cfg(target_arch = "aarch64")]
static WRITE: std::sync::Mutex<()> = std::sync::Mutex::new(());

static DEFAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The model's default when TORC_Q4X4 is unset (on, for Gemma on the CPU). Call it before the model's weights load.
pub fn set_default(on: bool) {
    DEFAULT.store(on, Ordering::Relaxed);
}

/// Whether the repacked copies are made and used: TORC_Q4X4=1 / 0, else the model's default, and only on ARM with
/// FEAT_I8MM. Fixed at the first call.
pub fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            let want = match crate::switches::get("TORC_Q4X4").as_deref() {
                Some("1") => true,
                Some("0") => false,
                _ => DEFAULT.load(Ordering::Relaxed),
            };
            return want && std::arch::is_aarch64_feature_detected!("i8mm");
        }
        #[allow(unreachable_code)]
        false
    });
    *ON
}

/// Repacks many matrices at once (the experts: thousands of them), each by one thread, and publishes the table once.
/// Returns the bytes.
#[cfg(target_arch = "aarch64")]
pub fn register_many(mats: &[(&[u8], usize, usize)]) -> usize {
    use kernels::neon::{Q4X4_GROUP, q4_0_repack4};
    let mut fresh = Vec::with_capacity(mats.len());
    let mut total = 0usize;
    for &(data, rows, cols) in mats {
        let nb = cols / kernels::QK;
        let row_bytes = data.len() / rows;
        assert!(rows % 4 == 0 && row_bytes == nb * 8 * kernels::Q4_0_BLOCK);
        let len = rows / 4 * nb * 8 * Q4X4_GROUP;
        #[repr(C, align(64))]
        #[derive(Clone, Copy)]
        struct A64([u8; 64]);
        let buf: &'static mut [A64] = Vec::leak(vec![A64([0; 64]); len.div_ceil(64)]);
        // SAFETY: buf holds at least len bytes
        let out: &'static mut [u8] = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
        let start = data.as_ptr() as usize;
        fresh.push((
            data,
            nb,
            row_bytes,
            Entry { start, end: start + rows * row_bytes, row_bytes, pq: 4 * row_bytes, packed: out },
        ));
        total += len;
    }
    let nt = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..nt {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(data, nb, row_bytes, ref e)) = fresh.get(i) else { break };
                    let quad = 4 * row_bytes;
                    // SAFETY: entry i's buffer is only written by this thread
                    let out = unsafe { std::slice::from_raw_parts_mut(e.packed.as_ptr() as *mut u8, e.packed.len()) };
                    for (q, o) in out.chunks_mut(quad).enumerate() {
                        q4_0_repack4(&data[4 * q * row_bytes..], row_bytes, nb, o);
                    }
                }
            });
        }
    });
    let _g = WRITE.lock().unwrap();
    let old = TABLE.load(Ordering::Acquire);
    // SAFETY: published snapshots are never freed
    let mut tab: Vec<Entry> = if old.is_null() { Vec::new() } else { unsafe { &*old }.clone() };
    for (_, _, _, e) in &fresh {
        LO.fetch_min(e.start, Ordering::Relaxed);
        HI.fetch_max(e.end, Ordering::Relaxed);
        tab.push(*e);
    }
    tab.sort_by_key(|e| e.start);
    tab.dedup_by_key(|e| e.start);
    TABLE.store(Box::into_raw(Box::new(tab)), Ordering::Release);
    total
}

/// Repacks a Q4_0 matrix (rows x cols, row-major, with rows a multiple of 4) and records it. Returns its bytes.
#[cfg(target_arch = "aarch64")]
pub fn register(data: &[u8], rows: usize, cols: usize) -> usize {
    use kernels::neon::{Q4X4_GROUP, q4_0_repack4};
    let nb = cols / kernels::QK;
    let row_bytes = data.len() / rows;
    assert!(rows.is_multiple_of(4) && row_bytes == nb * 8 * kernels::Q4_0_BLOCK);
    let quad = nb * 8 * Q4X4_GROUP;
    // 64-byte aligned, as the file's tensors are
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    struct A64([u8; 64]);
    let len = rows / 4 * quad;
    let buf: &'static mut [A64] = Vec::leak(vec![A64([0; 64]); len.div_ceil(64)]);
    // SAFETY: buf holds at least len bytes
    let out: &'static mut [u8] = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
    let nt = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = (rows / 4).div_ceil(nt);
    std::thread::scope(|s| {
        for (t, o) in out.chunks_mut(per * quad).enumerate() {
            s.spawn(move || {
                for (i, q) in o.chunks_mut(quad).enumerate() {
                    let r = (t * per + i) * 4;
                    q4_0_repack4(&data[r * row_bytes..], row_bytes, nb, q);
                }
            });
        }
    });
    let start = data.as_ptr() as usize;
    let end = start + rows * row_bytes;
    let _g = WRITE.lock().unwrap();
    let old = TABLE.load(Ordering::Acquire);
    let mut tab: Vec<Entry> = if old.is_null() {
        Vec::new()
    } else {
        // SAFETY: published snapshots are never freed
        unsafe { &*old }.clone()
    };
    tab.push(Entry { start, end, row_bytes, pq: 4 * row_bytes, packed: out });
    tab.sort_by_key(|e| e.start);
    LO.fetch_min(start, Ordering::Relaxed);
    HI.fetch_max(end, Ordering::Relaxed);
    TABLE.store(Box::into_raw(Box::new(tab)), Ordering::Release);
    len
}

/// The repacked bytes of `rows` rows starting at w (a row range of a registered matrix, starting on a multiple of 4
/// rows, with rows a multiple of 4), or None.
pub fn find(w: &[u8], rows: usize) -> Option<&'static [u8]> {
    if !rows.is_multiple_of(4) {
        return None;
    }
    let p = w.as_ptr() as usize;
    if p < LO.load(Ordering::Relaxed) || p >= HI.load(Ordering::Relaxed) {
        return None;
    }
    let t = TABLE.load(Ordering::Acquire);
    if t.is_null() {
        return None;
    }
    // SAFETY: published snapshots are never freed
    let tab = unsafe { &*t };
    let i = tab.partition_point(|e| e.start <= p).checked_sub(1)?;
    let e = &tab[i];
    if p >= e.end || !(p - e.start).is_multiple_of(4 * e.row_bytes) || p + rows * e.row_bytes > e.end {
        return None;
    }
    let q0 = (p - e.start) / (4 * e.row_bytes);
    Some(&e.packed[q0 * e.pq..(q0 + rows / 4) * e.pq])
}

/// Whether the experts' K-quant matrices (klay_kind: Q4_K, Q5_K, Q6_K) are also held laid out for SMMLA (lay4), with
/// their products going through gemm_q45k_lay / gemm_q6_lay, and 1-3 vectors through q45k_lay_matmul / q6_lay_matmul,
/// all with the same bits. On by default; TORC_KLAY=0 turns it off. ARM with FEAT_I8MM, and no expert cache. It is a
/// copy of the mapped originals (1.028x the Q4_K bytes, 1.023x the Q5_K, 1x the Q6_K), made per layer (at load in
/// torc-server, at first use elsewhere: weights::klay_group), after which the originals' pages leave the working set.
pub fn klay_on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            return crate::switches::get("TORC_KLAY").is_none_or(|v| v != "0")
                && std::arch::is_aarch64_feature_detected!("i8mm");
        }
        #[allow(unreachable_code)]
        false
    });
    *ON
}

/// The ggml types with a 4-row layout (the experts' K-quants), as (source block bytes, laid-out bytes per block of 4
/// rows). There is one entry per type here, with its lay function in lay4 and its kernels in moe.rs's klay_* dispatch.
#[cfg(target_arch = "aarch64")]
pub fn klay_kind(ty: u32) -> Option<(usize, usize)> {
    use kernels::neon::{Q4_K_BLOCK, Q5_K_BLOCK, Q6_K_BLOCK, Q6_LAY_BYTES, q45k_lay_bytes};
    match ty {
        12 => Some((Q4_K_BLOCK, q45k_lay_bytes(false))),
        13 => Some((Q5_K_BLOCK, q45k_lay_bytes(true))),
        14 => Some((Q6_K_BLOCK, Q6_LAY_BYTES)),
        _ => None,
    }
}

/// Whether matrices of type ty are laid out: it must be a klay_kind and, for Q6_K, TORC_KLAY_Q6 must not be 0.
#[cfg(target_arch = "aarch64")]
pub fn klay_type_on(ty: u32) -> bool {
    static Q6: LazyLock<bool> = LazyLock::new(|| crate::switches::get("TORC_KLAY_Q6").as_deref() != Some("0"));
    klay_kind(ty).is_some() && (ty != 14 || *Q6)
}

/// 4 rows of type ty (a klay_kind), laid out.
#[cfg(target_arch = "aarch64")]
fn lay4(ty: u32, src: &[u8], row_bytes: usize, nb: usize, out: &mut [u8]) {
    match ty {
        12 | 13 => kernels::neon::q45k_lay4(ty == 13, src, row_bytes, nb, out),
        14 => kernels::neon::q6_lay4(src, row_bytes, nb, out),
        _ => unreachable!("klay type {ty}"),
    }
}

/// TORC_KLAY_LARGE=1: laid-out copies on large pages.
pub fn klay_large() -> bool {
    static LARGE: LazyLock<bool> = LazyLock::new(|| crate::switches::on("TORC_KLAY_LARGE"));
    *LARGE
}

/// The bytes a matrix's laid-out copy takes in a large-page block (its length rounded up to 64), for klay_reserve.
#[cfg(target_arch = "aarch64")]
pub fn klay_bytes(rows: usize, cols: usize, ty: u32) -> usize {
    klay_kind(ty).map_or(0, |(_, lb)| (rows / 4 * (cols / kernels::QK * lb)).div_ceil(64) * 64)
}

/// Large pages for the groups' laid-out copies, reserved before the file is touched (TORC_KLAY_LARGE=1): one block per
/// group, each its group's bytes. Reserving them first matters on 32 GB machines: asked for one group at a time after
/// the warm-up, large pages run out partway through the model. register_many_klay takes a reserved block of exactly its
/// batch's size, or else asks for one itself. Returns the blocks the system gave.
static RESERVE: std::sync::Mutex<Vec<&'static mut [u8]>> = std::sync::Mutex::new(Vec::new());
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub fn klay_reserve(groups: &[usize]) -> usize {
    let mut r = RESERVE.lock().unwrap();
    for &bytes in groups.iter().filter(|&&b| b > 0) {
        match crate::ecache_own::large_leak(bytes) {
            Some(a) => r.push(<[u8]>::split_at_mut(a, bytes).0),
            None => break,
        }
    }
    r.len()
}

/// Lays out many matrices (data, rows, cols, ggml type: a klay_kind) by their type's lay4, each by one thread, and
/// publishes the table once. Returns the bytes.
#[cfg(target_arch = "aarch64")]
pub fn register_many_klay(mats: &[(&[u8], usize, usize, u32)]) -> usize {
    let mut fresh = Vec::with_capacity(mats.len());
    let mut total = 0usize;
    let len_of = |_data: &[u8], rows: usize, cols: usize, ty: u32| {
        let (_, lb) = klay_kind(ty).expect("a klay type");
        rows / 4 * (cols / kernels::QK * lb)
    };
    // TORC_KLAY_LARGE=1: the batch's copies go in one block of large pages (otherwise every 4 KiB page is a TLB entry,
    // and a token's experts touch a great many of them), each copy 64-aligned within it. Otherwise, or if the system
    // refuses, each copy gets a buffer from the allocator. The bytes are the same either way.
    let mut arena: Option<&'static mut [u8]> = if klay_large() {
        let need: usize = mats.iter().map(|&(d, r, c, t)| len_of(d, r, c, t).div_ceil(64) * 64).sum();
        // from the block reserved at load (klay_reserve) while it lasts, else from a block of its own
        let from_reserve = {
            let mut r = RESERVE.lock().unwrap();
            r.iter().position(|b| b.len() == need).map(|i| r.swap_remove(i))
        };
        let reserved = from_reserve.is_some();
        let a = from_reserve.or_else(|| crate::ecache_own::large_leak(need));
        eprintln!(
            "klay: {} MB on large pages: {}",
            need >> 20,
            if reserved {
                "yes (reserved)"
            } else if a.is_some() {
                "yes"
            } else {
                "no (allocator)"
            }
        );
        a
    } else {
        None
    };
    for &(data, rows, cols, ty) in mats {
        let nb = cols / kernels::QK;
        let row_bytes = data.len() / rows;
        let (bs, lb) = klay_kind(ty).expect("a klay type");
        assert!(rows % 4 == 0 && row_bytes == nb * bs);
        let pq = nb * lb;
        let len = rows / 4 * pq;
        assert_eq!(len, len_of(data, rows, cols, ty));
        #[repr(C, align(64))]
        #[derive(Clone, Copy)]
        struct A64([u8; 64]);
        let out: &'static mut [u8] = match arena.take() {
            Some(a) => {
                // a's start is page-aligned and every cut is a multiple of 64, so each copy is 64-aligned
                let (mine, rest) = a.split_at_mut(len.div_ceil(64) * 64);
                arena = Some(rest);
                &mut mine[..len]
            }
            None => {
                let buf: &'static mut [A64] = Vec::leak(vec![A64([0; 64]); len.div_ceil(64)]);
                // SAFETY: buf holds at least len bytes
                unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) }
            }
        };
        let start = data.as_ptr() as usize;
        fresh.push((data, nb, ty, Entry { start, end: start + rows * row_bytes, row_bytes, pq, packed: out }));
        total += len;
    }
    let nt = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..nt {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(data, nb, ty, ref e)) = fresh.get(i) else { break };
                    // SAFETY: entry i's buffer is only written by this thread
                    let out = unsafe { std::slice::from_raw_parts_mut(e.packed.as_ptr() as *mut u8, e.packed.len()) };
                    for (q, o) in out.chunks_mut(e.pq).enumerate() {
                        lay4(ty, &data[4 * q * e.row_bytes..], e.row_bytes, nb, o);
                    }
                }
            });
        }
    });
    let _g = WRITE.lock().unwrap();
    let old = TABLE.load(Ordering::Acquire);
    // SAFETY: published snapshots are never freed
    let mut tab: Vec<Entry> = if old.is_null() { Vec::new() } else { unsafe { &*old }.clone() };
    for (_, _, _, e) in &fresh {
        LO.fetch_min(e.start, Ordering::Relaxed);
        HI.fetch_max(e.end, Ordering::Relaxed);
        tab.push(*e);
    }
    tab.sort_by_key(|e| e.start);
    tab.dedup_by_key(|e| e.start);
    TABLE.store(Box::into_raw(Box::new(tab)), Ordering::Release);
    total
}
