//! ecache_own.rs - ToRc's expert cache (EXPERT-CACHE.md): the routed experts of a model larger than RAM, kept in
//! page-aligned slots and read from the file with unbuffered reads.
//!
//! Each distinct slot size gets one arena. Replacement is CLOCK with a frequency hint: every slot carries a little
//! credit, a use adds to it, the sweeping hand spends it, and experts the machine's usage file ranks hot start with
//! more. When a pass over the layers asks for more experts than an arena holds, the arena switches to evicting by layer
//! distance instead (see Arena::victim). A request's own slots cannot be evicted while it runs, nor can a slot whose
//! read is still in flight. `ask` returns as soon as the misses are queued, so the caller can start on the experts that
//! are already resident; `settle` waits for the rest. `read_ahead` queues reads of the next layer's likely experts into
//! slots no request holds.

use crate::ecache::Stats;
use format::{Result, err};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::File;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// Unbuffered reads need the buffer, offset and length on sector boundaries; 4 KiB covers the sector size of any drive
/// in use today.
pub(crate) const ALIGN: usize = 4096;

/// The experts the cache serves, from the file's EXPERTS records: (layers, experts per layer, each layer's pack bytes
/// (0 for none), each (layer, expert)'s absolute offset in the file).
fn expert_table(path: &str) -> Result<(usize, usize, Vec<u64>, Vec<u64>)> {
    let m = format::read::Model::open(path)?;
    let meta = format::meta::Meta::parse(&m.metadata)?;
    let n_layer =
        meta.arch_u32("block_count").ok_or_else(|| format::Error("expert cache: no block_count".into()))? as usize;
    let n_expert = m.experts.iter().map(|x| x.n_expert as usize).max().unwrap_or(0);
    let mut slots = vec![0u64; n_layer];
    let mut offset = vec![0u64; n_layer * n_expert];
    for x in &m.experts {
        let l = x.layer as usize;
        if l >= n_layer {
            return err(format!("expert cache: experts of layer {l}, past block_count"));
        }
        slots[l] = x.pack.bytes;
        for e in 0..x.n_expert {
            offset[l * n_expert + e as usize] = m.expert_region.offset + x.pack_at(e);
        }
    }
    Ok((n_layer, n_expert, slots, offset))
}

/// A slot is read in pieces of this many bytes (TORC_IO_PIECE_KB, a multiple of 4). The default is whole slots, one
/// read each.
static PIECE_BYTES: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    crate::switches::num::<usize>("TORC_IO_PIECE_KB")
        .filter(|&k| k >= 4 && k % 4 == 0)
        .map_or(usize::MAX / 2, |k| k << 10)
});
#[allow(non_snake_case)]
fn PIECE_() -> usize {
    *PIECE_BYTES
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    #[repr(C)]
    pub struct Luid {
        pub low: u32,
        pub high: i32,
    }
    #[repr(C)]
    pub struct TokenPrivileges {
        pub count: u32,
        pub luid: Luid,
        pub attributes: u32,
    }
    // the privilege calls are advapi32's, named here because without the C library nothing else links it
    #[link(name = "advapi32")]
    unsafe extern "system" {
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn OpenProcessToken(p: *mut c_void, access: u32, token: *mut *mut c_void) -> i32;
        pub fn LookupPrivilegeValueW(system: *const u16, name: *const u16, luid: *mut Luid) -> i32;
        pub fn AdjustTokenPrivileges(
            t: *mut c_void,
            disable_all: i32,
            new: *const TokenPrivileges,
            len: u32,
            prev: *mut c_void,
            ret: *mut u32,
        ) -> i32;
        pub fn GetLastError() -> u32;
        pub fn CloseHandle(h: *mut c_void) -> i32;
        pub fn GetLargePageMinimum() -> usize;
        pub fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        pub fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    }
    /// Enables SeLockMemoryPrivilege for this process (needed for large pages); returns false when the account lacks
    /// it.
    pub fn lock_memory_privilege() -> bool {
        const TOKEN_ADJUST_PRIVILEGES: u32 = 0x20;
        const TOKEN_QUERY: u32 = 0x8;
        const SE_PRIVILEGE_ENABLED: u32 = 2;
        let name: Vec<u16> = "SeLockMemoryPrivilege".encode_utf16().chain([0]).collect();
        unsafe {
            let mut tok = std::ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut tok) == 0 {
                return false;
            }
            let mut tp = TokenPrivileges { count: 1, luid: Luid { low: 0, high: 0 }, attributes: SE_PRIVILEGE_ENABLED };
            let ok = LookupPrivilegeValueW(std::ptr::null(), name.as_ptr(), &mut tp.luid) != 0
                && AdjustTokenPrivileges(tok, 0, &tp, 0, std::ptr::null_mut(), std::ptr::null_mut()) != 0
                && GetLastError() == 0;
            CloseHandle(tok);
            ok
        }
    }
}

/// `bytes` (rounded up to the large page size) of zeroed memory on large pages, never freed, or None where the system
/// does not allow it (no SeLockMemoryPrivilege, no large pages, or no contiguous memory left). It is for memory that
/// the products stream through and that lives as long as the process (repack.rs's laid-out experts).
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn large_leak(bytes: usize) -> Option<&'static mut [u8]> {
    #[cfg(windows)]
    {
        static PRIV: std::sync::LazyLock<bool> = std::sync::LazyLock::new(win::lock_memory_privilege);
        // SAFETY: plain Win32 calls, where a null result means there are no large pages here. VirtualAlloc's memory is
        // zeroed and stays committed for the life of the process
        unsafe {
            let lp = win::GetLargePageMinimum();
            if !*PRIV || lp == 0 || bytes == 0 {
                return None;
            }
            let size = bytes.div_ceil(lp) * lp;
            let p = win::VirtualAlloc(std::ptr::null_mut(), size, 0x1000 | 0x2000 | 0x2000_0000, 4);
            if p.is_null() {
                return None;
            }
            Some(std::slice::from_raw_parts_mut(p as *mut u8, size))
        }
    }
    #[cfg(not(windows))]
    {
        let _ = bytes;
        None
    }
}

/// One block of memory behind an arena's slots: (base, bytes, large).
type Piece = (*mut u8, usize, bool);

/// The memory for n slots of slot_size, in pieces of whole slots: one piece on large pages where the system allows it,
/// else pieces of about 512 MiB, each on large pages if it can still get them (a large-page block needs free physical
/// memory in large-page units, which a few server restarts can fragment so that one ~20 GB block fails while 512 MiB
/// ones still succeed), else from the allocator. Returns the pieces and the slots each holds (the last may hold fewer).
fn new_pieces(n: usize, slot_size: usize) -> Result<(Vec<Piece>, usize)> {
    if !large_wanted() {
        return Ok((vec![new_arena(n * slot_size, false)?], n));
    }
    let whole = new_arena(n * slot_size, true)?;
    if whole.1 != 0 {
        return Ok((vec![whole], n));
    }
    let per = ((512usize << 20) / slot_size).clamp(1, n);
    let mut pieces = Vec::new();
    let mut left = n;
    while left > 0 {
        let k = per.min(left);
        match new_arena(k * slot_size, false) {
            Ok(p) => pieces.push(p),
            Err(e) => {
                for &(b, s, l) in &pieces {
                    // SAFETY: from new_arena with these values, and nothing uses them yet
                    unsafe { arena_free(b, s, l) };
                }
                return Err(e);
            }
        }
        left -= k;
    }
    Ok((pieces, per))
}

/// Whether large pages are wanted (TORC_LARGE_PAGES=0 turns them off).
fn large_wanted() -> bool {
    cfg!(windows) && !crate::switches::is("TORC_LARGE_PAGES", "0")
}

/// An arena of at least `bytes`: on large pages where the system allows it (fewer TLB misses while the experts'
/// products stream through it; TORC_LARGE_PAGES=0 turns this off), else page-aligned from the allocator. With
/// probe, a large-page failure returns a zero-size piece instead of allocating the bytes from the allocator (new_pieces
/// then tries smaller pieces). Returns (base, bytes, large).
fn new_arena(bytes: usize, probe: bool) -> Result<(*mut u8, usize, bool)> {
    #[cfg(windows)]
    if large_wanted() {
        static PRIV: std::sync::LazyLock<bool> = std::sync::LazyLock::new(win::lock_memory_privilege);
        // SAFETY: plain Win32 calls, where a null result means there are no large pages here
        unsafe {
            let lp = win::GetLargePageMinimum();
            if *PRIV && lp > 0 {
                let size = bytes.div_ceil(lp) * lp;
                const MEM_COMMIT: u32 = 0x1000;
                const MEM_RESERVE: u32 = 0x2000;
                const MEM_LARGE_PAGES: u32 = 0x2000_0000;
                const PAGE_READWRITE: u32 = 4;
                let p = win::VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                    PAGE_READWRITE,
                );
                if !p.is_null() {
                    return Ok((p as *mut u8, size, true));
                }
                if probe {
                    return Ok((std::ptr::null_mut(), 0, false));
                }
            }
        }
    }
    let _ = probe;
    let layout = std::alloc::Layout::from_size_align(bytes, ALIGN)
        .map_err(|_| format::Error("expert cache: arena too large".into()))?;
    // SAFETY: a non-zero size, freed by arena_free with the same layout
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        return err(format!("expert cache: cannot allocate {} MiB", bytes >> 20));
    }
    Ok((base, bytes, false))
}

/// Frees an arena from new_arena.
///
/// # Safety
/// base, bytes and large must be as new_arena returned them, and the memory must not be used afterwards.
unsafe fn arena_free(base: *mut u8, bytes: usize, large: bool) {
    #[cfg(windows)]
    if large {
        const MEM_RELEASE: u32 = 0x8000;
        unsafe { win::VirtualFree(base as *mut std::ffi::c_void, 0, MEM_RELEASE) };
        return;
    }
    let _ = large;
    if let Ok(layout) = std::alloc::Layout::from_size_align(bytes, ALIGN) {
        unsafe { std::alloc::dealloc(base, layout) };
    }
}

/// The number of pieces a read of len bytes takes.
fn pieces(len: usize) -> u16 {
    u16::try_from(len.div_ceil(PIECE_())).expect("TORC_IO_PIECE_KB is too small for this model's experts")
}

/// The most credit a slot can hold. A use adds one, and each time the clock hand passes the slot it takes one away, so
/// an expert used three times in a row survives three sweeps without a use.
const MAX_CREDIT: u8 = 3;

/// The key of a slot that holds no expert.
const EMPTY: u32 = u32::MAX;

/// One slot size's arena: the memory, and the replacement state of its slots (CLOCK with a frequency hint,
/// EXPERT-CACHE.md).
struct Arena {
    slot_size: usize,
    n: usize,
    /// the memory behind the slots (new_pieces), `per` slots to a piece: each piece's base, the bytes reserved, and
    /// whether they are on large pages (freed with VirtualFree) or from the allocator
    pieces: Vec<Piece>,
    per: usize,
    n_expert: usize,
    /// the expert in each slot, as layer * n_expert + expert, or EMPTY
    key: Vec<u32>,
    /// each slot's CLOCK credit, 0..=MAX_CREDIT
    credit: Vec<u8>,
    /// the request each slot was last handed to: the request in progress may not lose its own slots
    holder: Vec<u64>,
    /// where the next sweep starts
    hand: usize,
    /// slots holding an expert
    used: usize,
    /// the layers whose experts live in this arena
    layers: usize,
    /// what a pass over the layers will want from this arena if every layer asks what the latest request did
    demand: usize,
}

impl Arena {
    fn new(slot_size: usize, n: usize, pieces: Vec<Piece>, per: usize, n_expert: usize) -> Arena {
        Arena {
            slot_size,
            n,
            pieces,
            per: per.max(1),
            n_expert,
            key: vec![EMPTY; n],
            credit: vec![0; n],
            holder: vec![0; n],
            hand: 0,
            used: 0,
            layers: 1,
            demand: 0,
        }
    }

    fn slot(&self, i: usize) -> *mut u8 {
        // SAFETY: i < n, and piece i / per holds slot i % per (new_pieces gives each piece per slots, the last the
        // rest)
        unsafe { self.pieces[i / self.per].0.add(i % self.per * self.slot_size) }
    }

    fn bytes(&self) -> usize {
        self.pieces.iter().map(|p| p.1).sum()
    }

    fn large_bytes(&self) -> usize {
        self.pieces.iter().filter(|p| p.2).map(|p| p.1).sum()
    }

    /// A use of slot s, by request `req` (0 for a read-ahead, which holds nothing).
    fn hit(&mut self, s: usize, req: u64) {
        self.credit[s] = (self.credit[s] + 1).min(MAX_CREDIT);
        if req > 0 {
            self.holder[s] = req;
        }
    }

    /// Slot s now holds expert k, starting with `credit`.
    fn fill(&mut self, s: usize, k: u32, credit: u8, req: u64) {
        if self.key[s] == EMPTY {
            self.used += 1;
        }
        self.key[s] = k;
        self.credit[s] = credit.min(MAX_CREDIT);
        if req > 0 {
            self.holder[s] = req;
        }
    }

    /// Whether the cache is being swept: no empty slot is left, and the request in hand, repeated on every layer of
    /// the arena, wants more experts than the arena holds (a prompt block, where each layer asks for most of its
    /// experts). Every expert then comes back once a pass, in layer order, so how recently it was used says nothing
    /// about when it is next wanted.
    fn cycling(&self) -> bool {
        self.used == self.n && self.demand > self.n
    }

    /// The slot to give a new expert while the model is at layer `cur` of `n_layer`, or None when every slot is held
    /// by request `holder` or has a read in flight (`loading`, per slot, non-zero).
    ///
    /// Normally CLOCK: the hand walks the slots, taking one credit from each it passes, and stops at the first empty
    /// slot or slot with no credit left. While the cache is swept (cycling), the next use of every slot is known from
    /// the layer order alone: a slot of layer l is wanted again after `wait(l)` layers (l - cur ahead of the current
    /// layer, a whole pass for the current layer itself), and the slot with the longest wait goes, the one with the
    /// least credit among equal waits. That is the optimal choice when the future is known, which here it is.
    fn victim(&mut self, cur: usize, n_layer: usize, holder: u64, loading: &[AtomicU16]) -> Option<usize> {
        if self.cycling() {
            let free = |s: usize| self.holder[s] != holder && loading[s].load(Ordering::Acquire) == 0;
            let wait = |k: u32| {
                let l = k as usize / self.n_expert;
                if l > cur { l - cur } else { l + n_layer - cur }
            };
            return (0..self.n).filter(|&s| free(s)).max_by_key(|&s| (wait(self.key[s]), MAX_CREDIT - self.credit[s]));
        }
        for _ in 0..self.n * (MAX_CREDIT as usize + 1) {
            let s = self.hand;
            self.hand = (s + 1) % self.n;
            if self.holder[s] == holder || loading[s].load(Ordering::Acquire) > 0 {
                continue;
            }
            if self.key[s] == EMPTY || self.credit[s] == 0 {
                return Some(s);
            }
            self.credit[s] -= 1;
        }
        None
    }
}

struct State {
    arenas: Vec<Arena>,
    /// the slots handed to the request in progress, as (arena, slot), which settle waits on
    held: Vec<(usize, usize)>,
    /// where each cached expert is: key -> (arena, slot)
    index: HashMap<u32, (usize, usize)>,
    stats: Stats,
    /// requests so far (the newest is the one in progress), and the layer it asked for
    requests: u64,
    layer_now: usize,
}

impl State {
    /// Takes a slot of arena a for expert k (a request's miss, or a read-ahead when req is 0) and points the index at
    /// it, dropping whatever the slot held. None when every slot is held by request holder or still being read.
    fn claim(
        &mut self,
        a: usize,
        k: u32,
        credit: u8,
        req: u64,
        holder: u64,
        at: (usize, usize),
        loading: &[AtomicU16],
    ) -> Option<usize> {
        let ar = &mut self.arenas[a];
        let s = ar.victim(at.0, at.1, holder, loading)?;
        let old = std::mem::replace(&mut ar.key[s], EMPTY);
        if old != EMPTY {
            ar.used -= 1;
            self.index.remove(&old);
        }
        let ar = &mut self.arenas[a];
        ar.fill(s, k, credit, req);
        self.index.insert(k, (a, s));
        Some(s)
    }
}

// SAFETY: the arenas' memory is owned by the cache and only touched under the Mutex (reads fill slots no request is
// using)
unsafe impl Send for State {}

/// A read for the reader threads: into slot (arena, slot) at ptr, len bytes from the file at off.
#[derive(Clone, Copy)]
struct Job {
    ptr: usize,
    len: usize,
    off: u64,
    arena: usize,
    slot: usize,
    /// a request's read (settle waits for these), as opposed to a read-ahead
    req: bool,
}

/// The reader threads' queue: the jobs, how many are queued or running, failures, shutdown, and how many of those
/// queued or running belong to a request.
struct Queue {
    jobs: Mutex<(VecDeque<Job>, usize, bool, bool, usize)>,
    work: Condvar,
    idle: Condvar,
}

#[cfg(windows)]
#[path = "ecache_overlapped.rs"]
mod overlapped;

pub struct OwnCache {
    _file: Arc<File>,
    q: Arc<Queue>,
    readers: Vec<std::thread::JoinHandle<()>>,
    /// on Windows, the reads as overlapped I/O started by the asking thread (the default; TORC_IO=threads uses reader
    /// threads instead)
    #[cfg(windows)]
    ov: Option<overlapped::Overlap>,
    st_shared: Arc<Mutex<State>>,
    /// for each arena and slot, the pieces of a read still queued or running (0 means it is in RAM); the readers count
    /// them down without taking the state's lock
    loading: Arc<Vec<Vec<AtomicU16>>>,
    n_expert: usize,
    /// the credit a newly read expert starts with: 2 for the experts the machine's usage file ranks hot enough to warm,
    /// 1 for the rest (the frequency hint), per layer * n_expert + expert
    start_credit: Vec<u8>,
    /// for each layer: its arena (usize::MAX for a layer without routed experts)
    layer_arena: Vec<usize>,
    /// for each (layer, expert): the slot's offset in the file
    offset: Vec<u64>,
}

/// The file, opened for unbuffered reads: FILE_FLAG_NO_BUFFERING on Windows, O_DIRECT on Linux (x86-64 and ARM64).
/// TORC_IO_BUFFERED=1 opens it plainly, so the reads go through the page cache, and so does a Linux file system that
/// refuses O_DIRECT (tmpfs, and some network and FUSE file systems), which is said once.
pub(crate) fn open_unbuffered(path: &str) -> std::io::Result<File> {
    if crate::switches::on("TORC_IO_BUFFERED") {
        return File::open(path);
    }
    open_direct(path)
}

#[cfg(windows)]
fn open_direct(path: &str) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path)
}

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn open_direct(path: &str) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    // the flag's value is not the same on every architecture
    #[cfg(target_arch = "x86_64")]
    const O_DIRECT: i32 = 0o40000;
    #[cfg(target_arch = "aarch64")]
    const O_DIRECT: i32 = 0o200000;
    static SAID: std::sync::Once = std::sync::Once::new();
    let why = match std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path) {
        // some file systems take the flag at open and refuse the reads, so one read decides
        Ok(f) => match direct_read_works(&f) {
            Ok(()) => return Ok(f),
            Err(e) => e,
        },
        Err(e) => e,
    };
    SAID.call_once(|| eprintln!("expert cache (own): no unbuffered reads here ({why}), so through the page cache"));
    File::open(path)
}

#[cfg(not(any(windows, all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))))]
fn open_direct(path: &str) -> std::io::Result<File> {
    File::open(path)
}

/// One aligned read at the start of a file opened with O_DIRECT, which a file system without it refuses.
#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
fn direct_read_works(f: &File) -> std::io::Result<()> {
    let layout = std::alloc::Layout::from_size_align(ALIGN, ALIGN).expect("one aligned page");
    // SAFETY: a non-zero size, freed below with the same layout; the read writes at most ALIGN bytes into it
    unsafe {
        let p = std::alloc::alloc(layout);
        if p.is_null() {
            return Err(std::io::Error::other("no memory for a test read"));
        }
        let r = std::os::unix::fs::FileExt::read_at(f, std::slice::from_raw_parts_mut(p, ALIGN), 0);
        std::alloc::dealloc(p, layout);
        r.map(|_| ())
    }
}

/// Reads len bytes at off into p (all multiples of ALIGN): the whole length, or an error.
pub(crate) fn read_at(f: &File, p: *mut u8, len: usize, off: u64) -> std::io::Result<()> {
    // SAFETY: p holds len writable bytes that nothing else touches during the read
    let buf = unsafe { std::slice::from_raw_parts_mut(p, len) };
    let mut done = 0;
    while done < len {
        #[cfg(windows)]
        let r = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..], off + done as u64)?;
        #[cfg(unix)]
        let r = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..], off + done as u64)?;
        if r == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        }
        done += r;
    }
    Ok(())
}

/// A reader thread: takes jobs until shutdown, reads each into its slot, and clears the slot's loading mark.
fn reader(q: &Queue, f: &File, loading: &[Vec<AtomicU16>]) {
    loop {
        let job = {
            let mut g = q.jobs.lock().unwrap();
            loop {
                if let Some(j) = g.0.pop_front() {
                    break j;
                }
                if g.3 {
                    return;
                }
                g = q.work.wait(g).unwrap();
            }
        };
        let ok = read_at(f, job.ptr as *mut u8, job.len, job.off).is_ok();
        let landed = loading[job.arena][job.slot].fetch_sub(1, Ordering::AcqRel) == 1;
        let mut g = q.jobs.lock().unwrap();
        g.1 -= 1;
        if job.req {
            g.4 -= 1;
        }
        if !ok {
            g.2 = true;
        }
        if g.1 == 0 || job.req && g.4 == 0 || landed {
            q.idle.notify_all();
        }
    }
}

impl OwnCache {
    /// Queues reads (the caller has already marked their slots as loading).
    fn queue(&self, jobs: Vec<Job>, front: bool) {
        if jobs.is_empty() {
            return;
        }
        // each slot goes as pieces of PIECE bytes, so one slot's read runs on several readers at once (more reads in
        // flight than the requests have misses)
        let mut parts = Vec::new();
        let mut jobs = jobs;
        if !crate::switches::is("TORC_IO_SORT", "0") {
            jobs.sort_by_key(|j| j.off);
        }
        for j in jobs {
            let n = pieces(j.len) as usize;
            for i in 0..n {
                let o = i * PIECE_();
                parts.push(Job { ptr: j.ptr + o, len: PIECE_().min(j.len - o), off: j.off + o as u64, ..j });
            }
        }
        #[cfg(windows)]
        if let Some(ov) = &self.ov {
            ov.issue(parts, &self.loading);
            return;
        }
        let mut g = self.q.jobs.lock().unwrap();
        g.1 += parts.len();
        g.4 += parts.iter().filter(|j| j.req).count();
        if front {
            for j in parts.into_iter().rev() {
                g.0.push_front(j);
            }
        } else {
            g.0.extend(parts);
        }
        drop(g);
        self.q.work.notify_all();
    }

    /// Waits until no read is queued or running (with `all`), or until no request's read is; returns Err if any read
    /// has ever failed.
    fn wait_idle(&self, all: bool) -> Result<()> {
        #[cfg(windows)]
        if let Some(ov) = &self.ov {
            return if ov.wait(&self.loading, |j| all || j.req) {
                Ok(())
            } else {
                err("expert cache (own): a read failed")
            };
        }
        let mut g = self.q.jobs.lock().unwrap();
        while if all { g.1 > 0 } else { g.4 > 0 } {
            g = self.q.idle.wait(g).unwrap();
        }
        // a failure is kept: the slot the read was for is still indexed, and must never be computed on
        if g.2 {
            return err("expert cache (own): a read failed");
        }
        Ok(())
    }

    /// A cache of about `budget` bytes over the model file at `path`, split over its slot sizes in proportion to the
    /// layers that use each. `warm` gives per-(layer, expert) counts, and the hottest experts are read first until the
    /// cache is full.
    pub fn open(path: &str, budget: u64, io_threads: usize, warm: Option<&[u32]>) -> Result<OwnCache> {
        let (n_layer, n_expert, slots, pack_offset) = expert_table(path)?;
        let mut sizes: Vec<usize> = Vec::new();
        let mut layer_arena = vec![usize::MAX; n_layer];
        for (l, &slot) in slots.iter().enumerate() {
            let b = slot as usize;
            if b == 0 {
                continue;
            }
            if !b.is_multiple_of(ALIGN) {
                return err(format!("expert cache: layer {l}'s slot of {b} bytes is not 4 KiB aligned"));
            }
            let a = sizes.iter().position(|&s| s == b).unwrap_or_else(|| {
                sizes.push(b);
                sizes.len() - 1
            });
            layer_arena[l] = a;
        }
        let total_layers = layer_arena.iter().filter(|&&a| a != usize::MAX).count().max(1);
        let mut arenas = Vec::new();
        for (a, &b) in sizes.iter().enumerate() {
            let users = layer_arena.iter().filter(|&&x| x == a).count();
            let share = budget as u128 * users as u128 / total_layers as u128;
            let n = ((share / b as u128) as usize).max(16);
            let (pieces, per) = new_pieces(n, b)?;
            let mut ar = Arena::new(b, n, pieces, per, n_expert);
            ar.layers = users.max(1);
            arenas.push(ar);
        }
        let all = arenas.iter().map(|a: &Arena| a.bytes()).sum::<usize>();
        let large = arenas.iter().map(|a: &Arena| a.large_bytes()).sum::<usize>();
        let np = arenas.iter().map(|a: &Arena| a.pieces.len()).sum::<usize>();
        eprintln!(
            "expert cache (own): {} MiB in {} arena(s){}{}",
            all >> 20,
            arenas.len(),
            if np > arenas.len() { format!(", {np} pieces") } else { String::new() },
            if large == all { ", large pages" } else { "" }
        );
        if large < all && large_wanted() {
            eprintln!(
                "expert cache (own): WARNING only {} of {} MiB on large pages: Windows had too little unfragmented \
                memory (often after several restarts; a reboot restores it). Without them the first prompt pays to fault \
                the cache in (tens of seconds on a ~20 GB cache) and later prompts read a few % slower",
                large >> 20,
                all >> 20
            );
        }
        let mut offset = vec![0u64; n_layer * n_expert];
        for (i, o) in offset.iter_mut().enumerate() {
            *o = pack_offset.get(i).copied().unwrap_or(0);
        }
        let file = Arc::new(open_unbuffered(path).map_err(|e| format::Error(format!("expert cache: {path}: {e}")))?);
        let st_shared = Arc::new(Mutex::new(State {
            arenas,
            held: Vec::new(),
            index: HashMap::new(),
            stats: Stats::default(),
            requests: 0,
            layer_now: 0,
        }));
        let loading: Arc<Vec<Vec<AtomicU16>>> = Arc::new(
            st_shared.lock().unwrap().arenas.iter().map(|ar| (0..ar.n).map(|_| AtomicU16::new(0)).collect()).collect(),
        );
        let q = Arc::new(Queue {
            jobs: Mutex::new((VecDeque::new(), 0, false, false, 0)),
            work: Condvar::new(),
            idle: Condvar::new(),
        });
        // on Windows the reads are overlapped unless TORC_IO=threads, in which case no reader threads are needed
        #[cfg(windows)]
        let ov = if crate::switches::is("TORC_IO", "threads") {
            None
        } else {
            Some(overlapped::Overlap::open(path).map_err(|e| format::Error(format!("expert cache: {path}: {e}")))?)
        };
        #[cfg(windows)]
        let io_threads = if ov.is_some() { 0 } else { io_threads.max(1) };
        let readers = (0..io_threads)
            .map(|i| {
                // each reader gets its own handle: Windows serialises synchronous reads on one handle, so readers
                // sharing one would take turns
                let (q, ld) = (q.clone(), loading.clone());
                let f = match open_unbuffered(path) {
                    Ok(f) => Arc::new(f),
                    Err(_) => file.clone(),
                };
                std::thread::Builder::new()
                    .name(format!("torc-io-{i}"))
                    .spawn(move || reader(&q, &f, &ld))
                    .expect("expert cache reader")
            })
            .collect();
        let mut c = OwnCache {
            _file: file,
            q,
            readers,
            #[cfg(windows)]
            ov,
            st_shared,
            loading,
            n_expert,
            start_credit: vec![1; n_layer * n_expert],
            layer_arena,
            offset,
        };
        if let Some(counts) = warm {
            c.warm(counts)?;
        }
        Ok(c)
    }

    /// Reads the hottest experts by count until every arena is full, and marks them hot for the frequency hint.
    fn warm(&mut self, counts: &[u32]) -> Result<()> {
        let mut keys: Vec<u32> = (0..counts.len() as u32).filter(|&k| counts[k as usize] > 0).collect();
        keys.sort_by_key(|&k| std::cmp::Reverse(counts[k as usize]));
        let t = Instant::now();
        let mut jobs = Vec::new();
        {
            let mut st = self.st_shared.lock().unwrap();
            let mut used: Vec<usize> = vec![0; st.arenas.len()];
            for k in keys {
                let a = match self.layer_arena.get(k as usize / self.n_expert) {
                    Some(&a) if a != usize::MAX => a,
                    _ => continue,
                };
                if used[a] >= st.arenas[a].n {
                    continue;
                }
                let s = used[a];
                used[a] += 1;
                if let Some(c) = self.start_credit.get_mut(k as usize) {
                    *c = 2;
                }
                let ar = &mut st.arenas[a];
                ar.fill(s, k, 2, 0);
                self.loading[a][s].store(pieces(ar.slot_size), Ordering::Release);
                jobs.push(Job {
                    ptr: ar.slot(s) as usize,
                    len: ar.slot_size,
                    off: self.offset[k as usize],
                    arena: a,
                    slot: s,
                    req: false,
                });
                st.index.insert(k, (a, s));
            }
        }
        let (n, bytes) = (jobs.len(), jobs.iter().map(|j| j.len as u64).sum::<u64>());
        self.queue(jobs, false);
        self.wait_idle(true)?;
        self.st_shared.lock().unwrap().stats.warmup_bytes += bytes;
        eprintln!("expert cache (own): warmed {n} experts, {} MiB in {:.1} s", bytes >> 20, t.elapsed().as_secs_f64());
        Ok(())
    }

    /// A request: the slots for ids (misses given slots and put at the front of the read queue), and which of them are
    /// in RAM now. The previous request's slots are released, and this one's are held until the next.
    fn request(&self, layer: u32, ids: &[u32]) -> Result<(Vec<*const u8>, Vec<bool>)> {
        let l = layer as usize;
        let a = match self.layer_arena.get(l) {
            Some(&a) if a != usize::MAX => a,
            _ => return err(format!("expert cache (own): layer {layer} has no experts")),
        };
        // the reads done since then: their slots are ready, and free to become victims
        #[cfg(windows)]
        if let Some(ov) = &self.ov {
            ov.poll(&self.loading);
        }
        let mut g = self.st_shared.lock().unwrap();
        let st = &mut *g;
        st.requests += 1;
        let req = st.requests;
        st.layer_now = l;
        let ar = &mut st.arenas[a];
        ar.demand = ids.len() * ar.layers;
        st.stats.asked += ids.len() as u64;
        st.held.clear();
        let mut out = vec![std::ptr::null::<u8>(); ids.len()];
        let mut ready = vec![false; ids.len()];
        let mut jobs = Vec::new();
        // hits are claimed for this request before any slot is chosen for a miss, so a miss cannot take one of them
        let mut missed = Vec::new();
        for (i, &e) in ids.iter().enumerate() {
            let k = (l * self.n_expert + e as usize) as u32;
            let Some(&(_, s)) = st.index.get(&k) else {
                missed.push(i);
                continue;
            };
            let ar = &mut st.arenas[a];
            ar.hit(s, req);
            out[i] = ar.slot(s);
            ready[i] = self.loading[a][s].load(Ordering::Acquire) == 0;
            st.held.push((a, s));
            st.stats.resident += 1;
        }
        for i in missed {
            let k = (l * self.n_expert + ids[i] as usize) as u32;
            // an expert named twice in one request: the first mention already queued its read
            if let Some(&(_, s)) = st.index.get(&k) {
                out[i] = st.arenas[a].slot(s);
                continue;
            }
            let at = (l, self.layer_arena.len());
            let Some(s) = st.claim(a, k, self.start_credit[k as usize], req, req, at, &self.loading[a]) else {
                return err(format!("expert cache (own): no free slot for a request of {} experts", ids.len()));
            };
            let ar = &st.arenas[a];
            self.loading[a][s].store(pieces(ar.slot_size), Ordering::Release);
            out[i] = ar.slot(s);
            let len = ar.slot_size;
            jobs.push(Job {
                ptr: ar.slot(s) as usize,
                len,
                off: self.offset[k as usize],
                arena: a,
                slot: s,
                req: true,
            });
            st.held.push((a, s));
            st.stats.loaded += 1;
            st.stats.loaded_bytes += len as u64;
        }
        drop(g);
        self.queue(jobs, true);
        Ok((out, ready))
    }

    /// The slots for experts ids of the layer, reading any that are missing (all are ready on return).
    pub fn take(&self, layer: u32, ids: &[u32]) -> Result<Vec<*const u8>> {
        let (s, _) = self.request(layer, ids)?;
        self.settle()?;
        Ok(s)
    }

    /// take without the wait: the slots and which are in RAM now; the rest are being read until settle.
    pub fn ask(&self, layer: u32, ids: &[u32]) -> Result<(Vec<*const u8>, Vec<bool>)> {
        self.request(layer, ids)
    }

    /// Waits for every queued read (the request's and any read-ahead).
    pub fn settle(&self) -> Result<()> {
        let t = Instant::now();
        #[cfg(windows)]
        if let Some(ov) = &self.ov {
            // the request's reads, plus any read-ahead into a slot the request uses
            let held = self.st_shared.lock().unwrap().held.clone();
            let ok = ov.wait(&self.loading, |j| j.req || held.contains(&(j.arena, j.slot)));
            self.st_shared.lock().unwrap().stats.blocked += t.elapsed().as_secs_f64();
            return if ok { Ok(()) } else { err("expert cache (own): a read failed") };
        }
        let r = self.wait_idle(false);
        // a request's hit on a slot that is still being read ahead: wait for that read too (rare); the readers signal
        // `idle` whenever a slot lands
        let held = self.st_shared.lock().unwrap().held.clone();
        let busy = || held.iter().any(|&(a, s)| self.loading[a][s].load(Ordering::Acquire) > 0);
        if busy() {
            let mut g = self.q.jobs.lock().unwrap();
            while busy() {
                g = self.q.idle.wait_timeout(g, std::time::Duration::from_millis(1)).unwrap().0;
            }
        }
        self.st_shared.lock().unwrap().stats.blocked += t.elapsed().as_secs_f64();
        r
    }

    /// Read-ahead: up to max of ids (likeliest first) that are not in RAM, into victims no request holds, queued after
    /// the requests' reads. The result is how many reads it queued.
    pub fn read_ahead(&self, layer: u32, ids: &[u32], max: usize) -> usize {
        // TORC_OWN_PREFETCH=0: no read-ahead
        static OFF: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| crate::switches::is("TORC_OWN_PREFETCH", "0"));
        if *OFF {
            return 0;
        }
        let l = layer as usize;
        let a = match self.layer_arena.get(l) {
            Some(&a) if a != usize::MAX => a,
            _ => return 0,
        };
        let mut st = self.st_shared.lock().unwrap();
        let mut jobs = Vec::new();
        for &e in ids {
            if jobs.len() >= max {
                break;
            }
            let k = (l * self.n_expert + e as usize) as u32;
            if st.index.contains_key(&k) {
                continue;
            }
            // never into a slot of the request in progress; the new expert gets its credit like a miss, so it is
            // still there when its layer asks for it
            let at = (st.layer_now, self.layer_arena.len());
            let holder = st.requests;
            let Some(s) = st.claim(a, k, self.start_credit[k as usize], 0, holder, at, &self.loading[a]) else { break };
            let ar = &st.arenas[a];
            self.loading[a][s].store(pieces(ar.slot_size), Ordering::Release);
            let len = ar.slot_size;
            jobs.push(Job {
                ptr: ar.slot(s) as usize,
                len,
                off: self.offset[k as usize],
                arena: a,
                slot: s,
                req: false,
            });
            st.stats.ahead += 1;
            st.stats.ahead_bytes += len as u64;
        }
        drop(st);
        let n = jobs.len();
        self.queue(jobs, false);
        n
    }

    /// Which of ids are in RAM or being read into it (for a layer the cache does not hold: all of them).
    pub fn held(&self, layer: u32, ids: &[u32]) -> Vec<bool> {
        let l = layer as usize;
        if !matches!(self.layer_arena.get(l), Some(&a) if a != usize::MAX) {
            return vec![true; ids.len()];
        }
        let st = self.st_shared.lock().unwrap();
        ids.iter().map(|&e| st.index.contains_key(&((l * self.n_expert + e as usize) as u32))).collect()
    }

    /// The most experts a request for the layer may take: its arena's slots, less a margin.
    pub fn layer_slots(&self, layer: u32) -> usize {
        let st = self.st_shared.lock().unwrap();
        match self.layer_arena.get(layer as usize) {
            Some(&a) if a != usize::MAX => st.arenas[a].n.saturating_sub(1).max(1),
            _ => 128,
        }
    }

    pub fn stats(&self) -> Stats {
        self.st_shared.lock().unwrap().stats
    }
}

impl Drop for OwnCache {
    fn drop(&mut self) {
        // no read may still be writing into the arenas when they are freed
        #[cfg(windows)]
        if let Some(ov) = &self.ov {
            ov.wait(&self.loading, |_| true);
        }
        {
            let mut g = self.q.jobs.lock().unwrap();
            g.0.clear();
            g.3 = true;
        }
        self.q.work.notify_all();
        for h in self.readers.drain(..) {
            let _ = h.join();
        }
        let st = self.st_shared.lock().unwrap();
        for ar in &st.arenas {
            for &(base, bytes, large) in &ar.pieces {
                // SAFETY: allocated by new_arena with these values, and no reader runs any more
                unsafe { arena_free(base, bytes, large) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No read in flight on any of n slots.
    fn idle(n: usize) -> Vec<AtomicU16> {
        (0..n).map(|_| AtomicU16::new(0)).collect()
    }

    /// n slots for layers of 8 experts, with no memory behind them (only the replacement state is used).
    fn arena(n: usize) -> Arena {
        Arena::new(4096, n, vec![(std::ptr::null_mut(), 0, false)], n, 8)
    }

    #[test]
    fn empty_slots_first_then_the_hand_spends_credit() {
        let (mut ar, busy) = (arena(4), idle(4));
        for k in [1, 2, 9, 10] {
            let s = ar.victim(0, 2, 1, &busy).unwrap();
            ar.fill(s, k, 1, 1);
        }
        assert_eq!((ar.key.clone(), ar.used), (vec![1, 2, 9, 10], 4));
        // request 1 holds every slot, so it cannot evict any of them
        assert_eq!(ar.victim(0, 2, 1, &busy), None);
        // expert 9 is used twice more; one sweep takes a credit from each slot and slot 0 runs out first
        ar.hit(2, 2);
        ar.hit(2, 2);
        assert_eq!(ar.victim(0, 2, 3, &busy), Some(0));
        assert_eq!(ar.credit, [0, 0, 2, 0]);
    }

    #[test]
    fn hot_experts_outlast_cold_ones() {
        let (mut ar, busy) = (arena(4), idle(4));
        for (s, k, credit) in [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1)] {
            ar.fill(s, k, credit, 0);
        }
        assert_eq!(ar.victim(0, 2, 1, &busy), Some(1));
        // a slot being read is passed over, credit or not
        busy[3].store(1, Ordering::Release);
        assert_eq!(ar.victim(0, 2, 1, &busy), Some(0));
    }

    #[test]
    fn cycling_takes_the_layer_needed_last() {
        // layers 0, 1, 2, 2 in four slots, and a request of two experts on each of three layers
        let (mut ar, busy) = (arena(4), idle(4));
        for (s, k) in [(0, 0), (1, 8), (2, 16), (3, 17)] {
            ar.fill(s, k, 1, 0);
        }
        (ar.layers, ar.demand) = (3, 6);
        assert!(ar.cycling());
        // at layer 1 the order to come is 2, 0, 1: layer 1's slot is wanted last
        assert_eq!(ar.victim(1, 3, 99, &busy), Some(1));
        // at layer 2, with request 7 holding slot 2, the other slot of layer 2
        ar.holder[2] = 7;
        assert_eq!(ar.victim(2, 3, 7, &busy), Some(3));
    }
}
