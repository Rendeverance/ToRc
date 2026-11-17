//! pool.rs - one compute pool for everything.
//!
//! Pinned workers spin between runs, the caller acts as thread 0, and items are split statically (tid, tid + n, ...).
//! TORC_POOL=rayon swaps in rayon, for comparison. ToRc's attention runs on the same pool, so no second set of threads
//! competes for the cores.

#[cfg(windows)]
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentThread() -> *mut c_void;
    fn SetThreadAffinityMask(h: *mut c_void, mask: usize) -> usize;
    fn GetCurrentProcess() -> *mut c_void;
    fn SetProcessAffinityMask(h: *mut c_void, mask: usize) -> i32;
    fn GetProcessAffinityMask(h: *mut c_void, process: *mut usize, system: *mut usize) -> i32;
}

pub struct Pool {
    pub threads: usize,
    rayon: bool,
    own: Option<OwnPool>,
    /// each thread's CPU when pinned (thread 0 is the caller)
    cpus: Vec<i32>,
    /// whether the threads are pinned now (set_pin changes it)
    pinned: AtomicBool,
    /// the CPUs the process was started on, which an unpinned pool on Linux and a pinned one on Windows go back to
    home: Home,
}

/// The CPUs a process may run on: Windows' affinity mask (processor group 0), Linux's CPU set.
#[cfg(windows)]
type Home = usize;
#[cfg(not(windows))]
type Home = [u64; 16];

/// The CPUs this process (on Linux, this thread) may run on now.
fn home() -> Home {
    #[cfg(windows)]
    {
        let (mut p, mut s) = (0usize, 0usize);
        unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut p, &mut s) };
        p
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
        }
        let mut set = [0u64; 16];
        // SAFETY: pid 0 is this thread, and the set is a whole cpu_set_t
        unsafe { sched_getaffinity(0, std::mem::size_of_val(&set), set.as_mut_ptr()) };
        set
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        [0; 16]
    }
}

// SAFETY: the pool is driven from one thread at a time, which is how the forward pass uses it
unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

static POOL: OnceLock<Pool> = OnceLock::new();

/// The logical CPUs of the fastest core class (processor group 0), in order: the P-cores of a hybrid part (both threads
/// of each), every CPU of a uniform one (and every CPU on ARM, see fast_cores). Every thread of a pool split takes an
/// equal share, so one slow core sets the pace. None where this is not known.
pub fn fast_cpus() -> Option<Vec<i32>> {
    let mut cpus: Vec<i32> = fast_cores()?.into_iter().flatten().collect();
    cpus.sort_unstable();
    Some(cpus)
}

/// The cores of the fastest class, each as its logical CPUs (two on a core with SMT), in order.
fn fast_cores() -> Option<Vec<Vec<i32>>> {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetLogicalProcessorInformationEx(rel: u32, buf: *mut u8, len: *mut u32) -> i32;
        }
        const RELATION_PROCESSOR_CORE: u32 = 0;
        let mut len = 0u32;
        unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len) };
        if len == 0 {
            return None;
        }
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        let b = buf.as_mut_ptr().cast::<u8>();
        if unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, b, &mut len) } == 0 {
            return None;
        }
        // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship u32 @0, Size u32 @4, then PROCESSOR_RELATIONSHIP: Flags
        // u8 @8, EfficiencyClass u8 @9, GroupCount u16 @30, GroupMask[] @32 (GROUP_AFFINITY: Mask usize, Group u16, 16
        // bytes on 64-bit)
        let bytes = unsafe { std::slice::from_raw_parts(b, len as usize) };
        let mut cores: Vec<(u8, Vec<i32>)> = Vec::new();
        let mut off = 0usize;
        while off + 32 <= bytes.len() {
            let rd32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            let size = rd32(off + 4) as usize;
            if size == 0 {
                break;
            }
            let class = bytes[off + 9];
            let groups = u16::from_le_bytes([bytes[off + 30], bytes[off + 31]]) as usize;
            let mut ids = Vec::new();
            for g in 0..groups {
                let ga = off + 32 + 16 * g;
                let mask = u64::from_le_bytes(bytes[ga..ga + 8].try_into().unwrap());
                let group = u16::from_le_bytes([bytes[ga + 8], bytes[ga + 9]]);
                if group == 0 {
                    ids.extend((0..64).filter(|i| mask >> i & 1 == 1));
                }
            }
            if !ids.is_empty() {
                cores.push((class, ids));
            }
            off += size;
        }
        let top = cores.iter().map(|c| c.0).max()?;
        // on ARM, every core: some parts report their clusters as different efficiency classes even when the cores are
        // the same design, and the top class alone can be a small fraction of them
        let all = cfg!(target_arch = "aarch64");
        let mut cores: Vec<Vec<i32>> = cores.into_iter().filter(|c| all || c.0 == top).map(|c| c.1).collect();
        cores.sort_unstable();
        return Some(cores);
    }
    // Roadmap: Linux, from /sys/devices/cpu_core/cpus on a hybrid Intel part (WSL2 hides it, so it waits for a real
    // machine to test on). Until then auto takes every CPU there.
    #[allow(unreachable_code)]
    None
}

/// The first n CPUs of the fastest class, one per core before any core's second (SMT) thread, so n threads land on as
/// many cores as there are. CPUs 0..n would put them on n / 2 cores of an SMT part, and a decode step is bound by how
/// much each core can have in flight from memory. With n = all of them the order still puts each core's first thread
/// first, so a narrower width (set_width, the decode threads) lands on whole cores. None where the cores are not known
/// or n is above their CPU count. TORC_SPREAD=0 uses CPUs 0..n.
fn spread_cpus(n: usize) -> Option<Vec<i32>> {
    if crate::switches::is("TORC_SPREAD", "0") {
        return None;
    }
    let cores = fast_cores()?;
    let deepest = cores.iter().map(|c| c.len()).max()?;
    let cpus: Vec<i32> = (0..deepest).flat_map(|t| cores.iter().filter_map(move |c| c.get(t).copied())).collect();
    (n <= cpus.len()).then(|| cpus[..n].to_vec())
}

/// The cores of the fastest class (their count, not their threads), or None where not known.
pub fn fast_core_count() -> Option<usize> {
    fast_cores().map(|c| c.len())
}

/// How many of the pool's threads a run uses (0: all). Set around the decode steps (set_width / Width) so a reply can
/// run on fewer threads than a prompt read: on an SMT part the prompt gains from both threads of each core, while one
/// token at a time is bound by memory and the second thread of a core only competes for it. The threads past the width
/// sit the run out; the items and bits are the same.
static WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Sets the width (0: all threads) and returns the previous one.
pub fn set_width(w: usize) -> usize {
    WIDTH.swap(w, Ordering::Relaxed)
}

/// The decode width: TORC_DECODE_THREADS, else the tuned "decode_threads" (torc-tune), else 0 (all threads).
pub fn decode_width() -> usize {
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| {
        crate::switches::num("TORC_DECODE_THREADS")
            .or_else(|| crate::tune::get("decode_threads").map(|v| v as usize))
            .unwrap_or(0)
    })
}

/// A decode width that overrides decode_width() (usize::MAX: none), for torc-tune, which measures each width in turn.
static FORCED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Forces the decode width (usize::MAX lifts it).
pub fn force_decode_width(w: usize) {
    FORCED.store(w, Ordering::Relaxed);
}

/// Holds the decode width for its lifetime (a decode step), then puts the previous width back.
pub struct Width(usize);

impl Width {
    pub fn decode() -> Width {
        let w = match FORCED.load(Ordering::Relaxed) {
            usize::MAX => decode_width(),
            w => w,
        };
        Width(set_width(w))
    }
}

impl Drop for Width {
    fn drop(&mut self) {
        set_width(self.0);
    }
}

/// Creates the pool (once): `n` threads on [`spread_cpus`] (else CPUs 0..n), or, with n = 0, one thread per CPU of
/// [`fast_cpus`]. The caller is thread 0, pinned to the first CPU. There are never more threads than CPUs: a thread
/// pinned to a CPU that doesn't exist slows every step of the pool down badly.
pub fn init(n: usize) -> &'static Pool {
    POOL.get_or_init(|| {
        let ncpu = std::thread::available_parallelism().map_or(n, |p| p.get());
        let n = if n > ncpu {
            eprintln!("threads: {n} asked, this machine has {ncpu} CPUs: using {ncpu}");
            ncpu
        } else {
            n
        };
        let cpus: Vec<i32> = match (n, fast_cpus()) {
            (0, Some(c)) => spread_cpus(c.len()).unwrap_or(c),
            (0, None) => (0..std::thread::available_parallelism().map_or(8, |p| p.get()) as i32).collect(),
            _ => spread_cpus(n).unwrap_or_else(|| (0..n as i32).collect()),
        };
        let n = cpus.len();
        let home = home();
        let rayon = crate::switches::is("TORC_POOL", "rayon");
        if rayon {
            crate::init_threads(n);
            return Pool { threads: n, rayon, own: None, cpus, pinned: AtomicBool::new(false), home };
        }
        // TORC_PIN=0: no thread is pinned, and the process is held to the fastest class's CPUs instead,
        // so the OS can move a worker off a core that something else (a scanner, a sensor poller) takes for a moment.
        // Pinned, every sweep waits for that one worker.
        // Pinned by default on Windows only: there the engine reads the core classes and pins to the right CPUs. On
        // Linux and macOS it can't see them yet, so pinning would place workers blind (slow cores on a big.LITTLE
        // part), and the OS's scheduler does better. TORC_PIN=1 pins on Linux too (CPUs 0..n); macOS has no way to pin
        // a thread to a core, so there it changes nothing.
        let pin = match crate::switches::get("TORC_PIN") {
            Some(v) => v != "0",
            None => cfg!(windows),
        };
        let own = Some(OwnPool::new(pin.then_some(&cpus[..]), n));
        #[cfg(windows)]
        unsafe {
            if pin {
                if let Some(m) = mask(cpus[0]) {
                    SetThreadAffinityMask(GetCurrentThread(), m);
                }
            } else if let Some(f) = fast_cpus() {
                SetProcessAffinityMask(GetCurrentProcess(), f.iter().filter_map(|&c| mask(c)).fold(0, |m, b| m | b));
            }
        }
        #[cfg(target_os = "linux")]
        if pin {
            pin_linux(cpus[0]);
        }
        Pool { threads: n, rayon, own, cpus, pinned: AtomicBool::new(pin), home }
    })
}

/// Pins the pool's threads to their CPUs, or frees them, while it runs: what TORC_PIN chooses at the start, with the
/// same result. For torc-tune, which measures both, and for apply_tuned_pin. Returns false where there is nothing to
/// change: rayon's pool, or macOS, which has no way to pin a thread.
pub fn set_pin(pin: bool) -> bool {
    let p = get();
    let Some(o) = &p.own else { return false };
    if !cfg!(any(windows, target_os = "linux")) {
        return false;
    }
    // every thread sets its own CPUs, so all of them must take part
    let was = set_width(0);
    #[cfg(windows)]
    unsafe {
        // unpinned, the process is held to the fastest class's CPUs (as init does); pinned, it has the CPUs it started
        // with, since a thread can only be pinned inside its process's set
        let fast = fast_cpus().map(|f| f.iter().filter_map(|&c| mask(c)).fold(0, |m, b| m | b)).filter(|&m| m != 0);
        let all = if pin { p.home } else { fast.unwrap_or(p.home) };
        if all != 0 {
            SetProcessAffinityMask(GetCurrentProcess(), all);
        }
        o.run(&|t, _| {
            let m = if pin { mask(p.cpus[t]) } else { Some(all) };
            if let Some(m) = m.filter(|&m| m != 0) {
                SetThreadAffinityMask(GetCurrentThread(), m);
            }
        });
    }
    #[cfg(target_os = "linux")]
    o.run(&|t, _| {
        if pin {
            pin_linux(p.cpus[t]);
        } else {
            unsafe extern "C" {
                fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
            }
            // SAFETY: pid 0 is this thread, and the set is a whole cpu_set_t
            unsafe { sched_setaffinity(0, std::mem::size_of_val(&p.home), p.home.as_ptr()) };
        }
    });
    #[cfg(not(any(windows, target_os = "linux")))]
    let _ = (o, &p.cpus, &p.home);
    set_width(was);
    p.pinned.store(pin, Ordering::Relaxed);
    true
}

/// Whether the pool's threads are pinned now.
pub fn pinned() -> bool {
    get().pinned.load(Ordering::Relaxed)
}

/// Applies the pinning torc-tune measured best for this machine and model ("pin"), unless TORC_PIN chose. Called once
/// the model is loaded: the tune file is the model's, and the pool is made before the model. Linux keeps its default
/// (not pinned) whatever the file says: a thread started from a pinned one inherits its CPU there, so the threads the
/// engine starts later (the expert cache's readers, the GPU's) would all sit on the caller's core.
pub fn apply_tuned_pin() {
    if cfg!(target_os = "linux") || crate::switches::given("TORC_PIN") {
        return;
    }
    if let Some(v) = crate::tune::get("pin") {
        let want = v == 1;
        if want != pinned() && set_pin(want) {
            eprintln!("threads: {} (auto-tuned)", if want { "pinned" } else { "not pinned" });
        }
    }
}

/// The pool, made with the automatic thread count (init(0)) if nothing made it first.
pub fn get() -> &'static Pool {
    POOL.get().unwrap_or_else(|| init(0))
}

/// Runs f(i) for every i in 0..n, spread over the pool.
pub fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = get();
    if n == 0 {
        return;
    }
    if p.rayon {
        use rayon::prelude::*;
        (0..n).into_par_iter().for_each(f);
        return;
    }
    if n == 1 {
        f(0);
        return;
    }
    if let Some(o) = &p.own {
        o.run(&|t, nt| {
            let mut i = t;
            while i < n {
                f(i);
                i += nt;
            }
        });
        return;
    }
    unreachable!("no pool");
}

/// par_for with each thread taking one contiguous run of the items (thread t takes n * t / T .. n * (t + 1) / T)
/// instead of every T-th item, so consecutive row chunks of a matrix stream through one core, which its prefetcher can
/// follow. On by default on ARM64 only; TORC_BLOCKED=0/1 overrides. The items are the same calls either way.
pub fn par_for_runs(n: usize, f: &(dyn Fn(usize) + Sync)) {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| match crate::switches::get("TORC_BLOCKED") {
        Some(v) => v != "0",
        None => cfg!(target_arch = "aarch64"),
    });
    let nt = get().threads;
    if !*ON || n <= nt {
        return par_for(n, f);
    }
    par_for(nt, &|t| {
        for i in n * t / nt..n * (t + 1) / nt {
            f(i);
        }
    });
}

/// par_for with guided self-scheduling: there is a shared cursor, and each thread claims the next contiguous run of
/// max(1, remaining / (2 T)) items. Runs start long (prefetcher-friendly, as in par_for_runs) and shrink at the tail,
/// so no thread is left with a long last run while the others wait. The items are the same calls in any order or on any
/// thread, so the bits are the same. On by default on both targets; TORC_GUIDED=0 uses par_for_runs instead.
pub fn par_for_guided(n: usize, f: &(dyn Fn(usize) + Sync)) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_GUIDED", "0"));
    let nt = get().threads;
    if !*ON {
        return par_for_runs(n, f);
    }
    if n <= nt {
        return par_for(n, f);
    }
    let next = AtomicUsize::new(0);
    par_for(nt, &|_| loop {
        let cur = next.load(Ordering::Relaxed);
        if cur >= n {
            break;
        }
        let k = ((n - cur) / (2 * nt)).max(1);
        if next.compare_exchange_weak(cur, cur + k, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            for i in cur..cur + k {
                f(i);
            }
        }
    });
}

/// par_for with dynamic scheduling: there is a shared cursor, and each thread claims the next item until none are left.
/// This is for machines whose cores run at different speeds (P- and E-cores): a fast core takes more items instead of
/// waiting for a slow one's fixed share. The items are the same calls in any order or on any thread, so the bits are
/// the same.
pub fn par_for_dynamic(n: usize, f: &(dyn Fn(usize) + Sync)) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let nt = get().threads;
    if n <= 1 || nt <= 1 {
        return par_for(n, f);
    }
    let next = AtomicUsize::new(0);
    par_for(nt, &|_| loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        f(i);
    });
}

/// Whether an item of a par_for may wait (spin) for items with lower indices. On ToRc's own pool, every thread of a run
/// takes its items in increasing order and all of them run at once, so an item being waited on is held by a thread
/// still working through lower items, never by one that is waiting. Not under rayon, whose work stealing can queue the
/// item behind the waiter. TORC_FLOW=0 says never (a fork-join per stage instead).
pub fn waits_ok() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| !crate::switches::is("TORC_FLOW", "0"));
    *ON && !get().rayon
}

/// Spins until c reaches n (Acquire, so the writes of the items that counted are visible).
pub fn wait_for(c: &std::sync::atomic::AtomicUsize, n: usize) {
    while c.load(std::sync::atomic::Ordering::Acquire) < n {
        std::hint::spin_loop();
    }
}

/// Stages in one sweep of the pool: stage s + 1's items start once all of stage s's are done (a counter is spun on
/// instead of a fork-join, with no step on one thread in between). Every item is the same call as with a par_for per
/// stage, which is what runs where waiting is not safe ([`waits_ok`]).
pub fn par_stages(stages: &[(usize, &(dyn Fn(usize) + Sync))]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    if !waits_ok() || stages.len() == 1 {
        for &(n, f) in stages {
            par_for(n, f);
        }
        return;
    }
    let done: Vec<AtomicUsize> = stages.iter().map(|_| AtomicUsize::new(0)).collect();
    let total = stages.iter().map(|s| s.0).sum();
    par_for(total, &|mut i| {
        let mut s = 0;
        while i >= stages[s].0 {
            i -= stages[s].0;
            s += 1;
        }
        for p in 0..s {
            wait_for(&done[p], stages[p].0);
        }
        (stages[s].1)(i);
        done[s].fetch_add(1, Ordering::Release);
    });
}

/// ToRc's own pool (SPEC.md section 6): n - 1 pinned workers that spin on a generation counter between runs and then
/// park, with the caller as thread 0.
struct OwnPool {
    sh: &'static Shared,
    workers: Vec<std::thread::Thread>,
    n: usize,
}

#[repr(align(128))]
struct Padded<T>(T);

struct Shared {
    /// bumped once per run; a worker that sees it change picks up the new job
    epoch: Padded<AtomicU64>,
    /// per worker, the newest epoch whose job it has completed. Each sits on its own 128-byte line, so workers
    /// publishing completion never contend for the same cache line
    finished: Vec<Padded<AtomicU64>>,
    job: AtomicPtr<JobRef<'static>>,
    /// per worker, true while it is (about to be) parked, so the caller knows to unpark it
    parked: Vec<Padded<AtomicBool>>,
    spin: Duration,
    /// set while a run is in progress: the pool takes one caller at a time, and a second thread entering it would
    /// overwrite the job the workers are reading
    busy: AtomicBool,
}

struct JobRef<'a> {
    f: &'a (dyn Fn(usize, usize) + Sync),
}

thread_local! {
    static IN_JOB: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl OwnPool {
    fn new(cpus: Option<&[i32]>, n: usize) -> OwnPool {
        // a worker spins for TORC_SPIN_US (default 200) before it parks
        let spin = crate::switches::num("TORC_SPIN_US").unwrap_or(200u64);
        let sh: &'static Shared = Box::leak(Box::new(Shared {
            epoch: Padded(AtomicU64::new(0)),
            finished: (0..n).map(|_| Padded(AtomicU64::new(0))).collect(),
            job: AtomicPtr::new(std::ptr::null_mut()),
            parked: (0..n).map(|_| Padded(AtomicBool::new(false))).collect(),
            spin: Duration::from_micros(spin),
            busy: AtomicBool::new(false),
        }));
        let workers = (1..n)
            .map(|t| {
                let cpu = cpus.map(|c| c[t]);
                std::thread::Builder::new()
                    .name(format!("torc-{t}"))
                    .spawn(move || worker(sh, t, n, cpu))
                    .expect("pool worker")
                    .thread()
                    .clone()
            })
            .collect();
        OwnPool { sh, workers, n }
    }

    /// Runs f(t, n) on every thread t of the pool (this one is 0) and returns when all of them have returned.
    fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        if IN_JOB.with(|c| c.get()) {
            eprintln!("torc pool: a run started from inside a run (the pool is one level)");
            std::process::abort();
        }
        let sh = self.sh;
        if sh.busy.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            eprintln!(
                "torc pool: a run started while another thread's run was in progress (the pool takes one caller)"
            );
            std::process::abort();
        }
        // a width below the pool's (set_width): only threads 0..w take part, and the job sees w threads
        let w = match WIDTH.load(Ordering::Relaxed) {
            0 => self.n,
            w => w.min(self.n),
        };
        let narrow = move |t: usize, _n: usize| {
            if t < w {
                f(t, w)
            }
        };
        let f: &(dyn Fn(usize, usize) + Sync) = if w < self.n { &narrow } else { f };
        let job = JobRef { f };
        // SAFETY: `job` lives on this stack frame until the function returns. A worker dereferences it only after it
        // observes the new epoch and before it publishes that epoch in `finished`, and the loop at the end waits for
        // every worker to publish it
        let p = &job as *const JobRef as *mut JobRef<'static>;
        sh.job.store(p, Ordering::Relaxed);
        let g = sh.epoch.0.fetch_add(1, Ordering::SeqCst) + 1;
        for (t, w) in self.workers.iter().enumerate() {
            if sh.parked[t + 1].0.load(Ordering::SeqCst) {
                w.unpark();
            }
        }
        IN_JOB.with(|c| c.set(true));
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0, self.n))).is_err() {
            eprintln!("torc pool: a job panicked on thread 0");
            std::process::abort();
        }
        IN_JOB.with(|c| c.set(false));
        for d in &sh.finished[1..] {
            while d.0.load(Ordering::Acquire) != g {
                std::hint::spin_loop();
            }
        }
        sh.busy.store(false, Ordering::Release);
    }
}

/// CPU c's bit in an affinity mask, which holds processor group 0's first 64 CPUs. None past those: a thread given an
/// explicit count above 64 is left to the OS rather than pinned to the wrong CPU.
#[cfg(windows)]
fn mask(c: i32) -> Option<usize> {
    (0..usize::BITS as i32).contains(&c).then(|| 1usize << c)
}

/// Pins the calling thread to CPU c (Linux; sched_setaffinity, from the C library, on the thread itself). CPUs past
/// the set's 1024 are left to the OS.
#[cfg(target_os = "linux")]
pub(crate) fn pin_linux(c: i32) {
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }
    let mut set = [0u64; 16];
    if !(0..(set.len() * 64) as i32).contains(&c) {
        return;
    }
    set[c as usize / 64] |= 1u64 << (c % 64);
    // SAFETY: pid 0 is this thread, and the mask is a whole cpu_set_t (1024 bits) that outlives the call
    unsafe { sched_setaffinity(0, std::mem::size_of_val(&set), set.as_ptr()) };
}

fn worker(sh: &'static Shared, t: usize, n: usize, cpu: Option<i32>) {
    #[cfg(windows)]
    if let Some(m) = cpu.and_then(mask) {
        unsafe { SetThreadAffinityMask(GetCurrentThread(), m) };
    }
    #[cfg(target_os = "linux")]
    if let Some(c) = cpu {
        pin_linux(c);
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    let _ = cpu;
    IN_JOB.with(|c| c.set(true));
    let mut last = 0u64;
    loop {
        let mut since = Instant::now();
        let mut i = 0u32;
        let g = loop {
            let g = sh.epoch.0.load(Ordering::Acquire);
            if g != last {
                break g;
            }
            std::hint::spin_loop();
            i = i.wrapping_add(1);
            if i.is_multiple_of(1024) && since.elapsed() >= sh.spin {
                sh.parked[t].0.store(true, Ordering::SeqCst);
                if sh.epoch.0.load(Ordering::SeqCst) == last {
                    std::thread::park();
                }
                sh.parked[t].0.store(false, Ordering::SeqCst);
                since = Instant::now();
            }
        };
        last = g;
        // SAFETY: OwnPool::run keeps epoch g's job alive until this worker stores g in `finished`
        let job = unsafe { &*sh.job.load(Ordering::Acquire) };
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (job.f)(t, n))).is_err() {
            eprintln!("torc pool: a job panicked on thread {t}");
            std::process::abort();
        }
        sh.finished[t].0.store(g, Ordering::Release);
    }
}

/// For tests: the pool takes one caller at a time (run aborts otherwise), and the test harness runs tests on parallel
/// threads, so a test that uses the pool takes this first.
#[cfg(test)]
pub(crate) fn one_caller() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A pointer the pool's tasks write through, each task to its own disjoint range.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T: Send> Send for SyncPtr<T> {}
unsafe impl<T: Send> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(s: &mut [T]) -> SyncPtr<T> {
        SyncPtr(s.as_mut_ptr())
    }
    /// The element pointer at off (a method, so that a closure captures the Sync wrapper rather than the raw field).
    ///
    /// # Safety
    /// off must be inside the slice this was made from.
    pub unsafe fn at(&self, off: usize) -> *mut T {
        unsafe { self.0.add(off) }
    }
    /// # Safety
    /// No other task may touch [off, off + len) while the returned slice lives.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, off: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// SyncPtr for both arrays of a quantised vector (kernels::Q8m), offsets counted in blocks.
pub struct Q8Ptr {
    pub k: SyncPtr<kernels::BlockQ8K>,
    pub g: SyncPtr<kernels::Groups32>,
}

impl Q8Ptr {
    pub fn new(x: kernels::Q8m) -> Q8Ptr {
        Q8Ptr { k: SyncPtr::new(x.k), g: SyncPtr::new(x.g) }
    }
    /// # Safety
    /// As SyncPtr::slice, for blocks [off, off + len) of both arrays.
    pub unsafe fn slice(&self, off: usize, len: usize) -> kernels::Q8m<'_> {
        unsafe { kernels::Q8m { k: self.k.slice(off, len), g: self.g.slice(off, len) } }
    }
    /// Blocks [off, off + len) of both arrays, read-only.
    ///
    /// # Safety
    /// No task may write those blocks while the returned view lives.
    pub unsafe fn part(&self, off: usize, len: usize) -> kernels::Q8s<'_> {
        unsafe {
            kernels::Q8s {
                k: std::slice::from_raw_parts(self.k.at(off), len),
                g: std::slice::from_raw_parts(self.g.at(off), len),
            }
        }
    }
    /// A pointer to nothing, for a path that never uses it.
    pub fn null() -> Q8Ptr {
        Q8Ptr { k: SyncPtr(std::ptr::null_mut()), g: SyncPtr(std::ptr::null_mut()) }
    }
}

/// TORC_TIMER_1MS=1: sets the Windows system timer to 1 ms (timeBeginPeriod) for this process, so a
/// short sleep or timed wait is not rounded up to the default tick of about 15.6 ms.
pub fn timer_1ms_if_asked() {
    #[cfg(windows)]
    if crate::switches::on("TORC_TIMER_1MS") {
        #[link(name = "winmm")]
        unsafe extern "system" {
            fn timeBeginPeriod(period: u32) -> u32;
        }
        // SAFETY: a plain Win32 call; the period ends with the process
        let r = unsafe { timeBeginPeriod(1) };
        eprintln!("timer: 1 ms ({})", if r == 0 { "set" } else { "refused" });
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    /// The CPUs this thread may run on, as the kernel reports them.
    fn allowed() -> String {
        let s = std::fs::read_to_string("/proc/thread-self/status").unwrap();
        s.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:")).unwrap().trim().to_string()
    }

    #[test]
    fn pinning_holds_a_thread_to_its_cpu() {
        // on a thread of its own, so the test harness's thread is left as it was
        let (before, after) = std::thread::spawn(|| {
            let before = allowed();
            super::pin_linux(0);
            (before, allowed())
        })
        .join()
        .unwrap();
        assert_eq!(after, "0", "was {before}");
        // a CPU past the set is left to the OS: nothing changes
        let same = std::thread::spawn(|| {
            let before = allowed();
            super::pin_linux(5000);
            before == allowed()
        })
        .join()
        .unwrap();
        assert!(same);
    }

    #[test]
    fn a_pinned_pool_puts_each_worker_on_its_cpu() {
        if std::thread::available_parallelism().map_or(1, |p| p.get()) < 3 {
            return;
        }
        let _g = super::one_caller();
        // CPUs 2 and 1 for threads 0 and 1: the worker (thread 1) must land on CPU 1; the caller is pinned by init,
        // not by the pool, so it is not checked here
        let seen = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            s.spawn(|| {
                let p = super::OwnPool::new(Some(&[2, 1]), 2);
                p.run(&|t, _| seen.lock().unwrap().push((t, allowed())));
            });
        });
        let v = seen.into_inner().unwrap();
        assert!(v.contains(&(1, "1".to_string())), "{v:?}");
    }

    #[test]
    fn set_pin_pins_and_frees_the_running_pool() {
        if std::thread::available_parallelism().map_or(1, |p| p.get()) < 2 {
            return;
        }
        let _g = super::one_caller();
        // on a thread of its own: the caller is the pool's thread 0, and set_pin changes its CPUs too
        let (start, pinned, freed) = std::thread::scope(|s| {
            s.spawn(|| {
                let p = super::init(2);
                let o = p.own.as_ref().unwrap();
                let each = || {
                    let seen = std::sync::Mutex::new(vec![String::new(); p.threads]);
                    o.run(&|t, _| seen.lock().unwrap()[t] = allowed());
                    seen.into_inner().unwrap()
                };
                super::set_pin(false);
                let start = each();
                assert!(super::set_pin(true) && super::pinned());
                let pinned = each();
                assert!(super::set_pin(false) && !super::pinned());
                (start, pinned, each())
            })
            .join()
            .unwrap()
        });
        // the pool is the process's one (another test may have made it first, with its own count): CPUs 0..n here
        assert!(pinned.len() >= 2);
        for (t, c) in pinned.iter().enumerate() {
            assert_eq!(*c, t.to_string());
        }
        // freed, a worker has the CPUs the process started with again
        assert_eq!(freed[1], start[1]);
        assert_ne!(freed[1], "1");
    }
}
