//! ecache_overlapped.rs - the expert cache's reads on Windows, as overlapped I/O.
//!
//! The thread that needs some bytes issues the ReadFile itself (FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, with an
//! OVERLAPPED per read) and the kernel fills the buffer while that thread carries on with its own work: there is no
//! I/O thread to hand requests to. `poll` gathers whatever has finished without blocking; `wait` blocks until the reads
//! a predicate selects are in. Compared with reader threads, this keeps the compute threads, one pinned to each core,
//! from sharing their cores with the I/O.

use super::Job;
use std::ffi::c_void;
use std::fs::File;
use std::os::windows::io::AsRawHandle;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU16, Ordering};

#[repr(C)]
struct Overlapped {
    internal: usize,
    internal_high: usize,
    offset: u32,
    offset_high: u32,
    event: *mut c_void,
}

unsafe extern "system" {
    fn ReadFile(h: *mut c_void, buf: *mut u8, n: u32, got: *mut u32, o: *mut Overlapped) -> i32;
    fn GetOverlappedResult(h: *mut c_void, o: *mut Overlapped, got: *mut u32, wait: i32) -> i32;
    fn CreateEventW(attr: *mut c_void, manual: i32, initial: i32, name: *const u16) -> *mut c_void;
    fn ResetEvent(e: *mut c_void) -> i32;
    fn CloseHandle(h: *mut c_void) -> i32;
    fn GetLastError() -> u32;
}

const ERROR_IO_PENDING: u32 = 997;
const ERROR_IO_INCOMPLETE: u32 = 996;

/// The most reads in flight at once (TORC_IO_DEPTH). Each holds its buffer's pages locked while it runs, so the
/// warm-up's thousands of reads go in waves of this many.
static DEPTH: std::sync::LazyLock<usize> =
    std::sync::LazyLock::new(|| crate::switches::num("TORC_IO_DEPTH").filter(|&d: &usize| d > 0).unwrap_or(64));

/// A read in flight: its OVERLAPPED (boxed, because the system writes to it while the read runs), the job, and the
/// bytes done.
struct Read {
    o: Box<Overlapped>,
    job: Job,
    done: usize,
}

struct Inner {
    flight: Vec<Read>,
    /// the events of finished reads, to reuse for the next ones
    free: Vec<*mut c_void>,
    failed: bool,
}

pub struct Overlap {
    file: File,
    inner: Mutex<Inner>,
}

// SAFETY: the raw handles (the file's and the events') are either used under the Mutex or are the file's own, which
// Windows lets any thread use
unsafe impl Send for Overlap {}
unsafe impl Sync for Overlap {}

/// What a read had done when it was looked at.
enum Step {
    Running,
    Done,
    Failed,
}

impl Overlap {
    /// `path`, opened for overlapped, unbuffered reads.
    pub fn open(path: &str) -> std::io::Result<Overlap> {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
            .open(path)?;
        Ok(Overlap { file, inner: Mutex::new(Inner { flight: Vec::new(), free: Vec::new(), failed: false }) })
    }

    fn handle(&self) -> *mut c_void {
        self.file.as_raw_handle()
    }

    /// Issues the ReadFile for r's remaining bytes.
    fn start(&self, r: &mut Read) -> Step {
        let off = r.job.off + r.done as u64;
        r.o.internal = 0;
        r.o.internal_high = 0;
        r.o.offset = off as u32;
        r.o.offset_high = (off >> 32) as u32;
        // SAFETY: the event belongs to this cache; the buffer holds job.len bytes that nothing else touches while the
        // read runs, and the OVERLAPPED lives (boxed, in the flight list) until the read is collected
        unsafe {
            ResetEvent(r.o.event);
            // ReadFile takes a u32 count; the largest page multiple keeps an unbuffered read's length aligned
            let n = (r.job.len - r.done).min(u32::MAX as usize & !4095) as u32;
            if ReadFile(self.handle(), (r.job.ptr as *mut u8).add(r.done), n, std::ptr::null_mut(), &mut *r.o) == 0
                && GetLastError() != ERROR_IO_PENDING
            {
                return Step::Failed;
            }
        }
        Step::Running
    }

    /// Checks on r's read (blocking with `wait`); when fewer bytes came back than asked, the remainder is issued as a
    /// fresh ReadFile.
    fn step(&self, r: &mut Read, wait: bool) -> Step {
        loop {
            let mut got = 0u32;
            // SAFETY: r.o is the OVERLAPPED of the read started on this handle
            let ok = unsafe { GetOverlappedResult(self.handle(), &mut *r.o, &mut got, wait as i32) } != 0;
            if !ok {
                return if !wait && unsafe { GetLastError() } == ERROR_IO_INCOMPLETE {
                    Step::Running
                } else {
                    Step::Failed
                };
            }
            r.done += got as usize;
            if r.done >= r.job.len {
                return Step::Done;
            }
            if got == 0 {
                return Step::Failed;
            }
            if let Step::Failed = self.start(r) {
                return Step::Failed;
            }
            if !wait {
                return Step::Running;
            }
        }
    }

    /// Collects a read: its slot's piece is counted as landed, and its event is free again.
    fn land(inner: &mut Inner, r: Read, ok: bool, loading: &[Vec<AtomicU16>]) {
        if !ok {
            inner.failed = true;
        }
        loading[r.job.arena][r.job.slot].fetch_sub(1, Ordering::AcqRel);
        inner.free.push(r.o.event);
    }

    /// Starts the jobs' reads (the caller has already marked their slots as loading). Beyond the depth limit, the
    /// oldest reads in flight are waited for first.
    pub fn issue(&self, jobs: Vec<Job>, loading: &[Vec<AtomicU16>]) {
        let mut g = self.inner.lock().unwrap();
        for job in jobs {
            while g.flight.len() >= *DEPTH {
                let mut r = g.flight.remove(0);
                let ok = matches!(self.step(&mut r, true), Step::Done);
                Self::land(&mut g, r, ok, loading);
            }
            // SAFETY: a manual-reset event, unnamed
            let event =
                g.free.pop().unwrap_or_else(|| unsafe { CreateEventW(std::ptr::null_mut(), 1, 0, std::ptr::null()) });
            let o = Box::new(Overlapped { internal: 0, internal_high: 0, offset: 0, offset_high: 0, event });
            let mut r = Read { o, job, done: 0 };
            match self.start(&mut r) {
                Step::Failed => Self::land(&mut g, r, false, loading),
                _ => g.flight.push(r),
            }
        }
    }

    /// Collects the reads that are done, without waiting.
    pub fn poll(&self, loading: &[Vec<AtomicU16>]) {
        let mut g = self.inner.lock().unwrap();
        let mut i = 0;
        while i < g.flight.len() {
            let st = self.step(&mut g.flight[i], false);
            match st {
                Step::Running => i += 1,
                Step::Done | Step::Failed => {
                    let r = g.flight.swap_remove(i);
                    Self::land(&mut g, r, matches!(st, Step::Done), loading);
                }
            }
        }
    }

    /// Waits for the reads `pick` chooses (and collects any others that are done); returns true if no read has
    /// ever failed (a failure is kept).
    pub fn wait(&self, loading: &[Vec<AtomicU16>], pick: impl Fn(&Job) -> bool) -> bool {
        let mut g = self.inner.lock().unwrap();
        let mut i = 0;
        while i < g.flight.len() {
            let w = pick(&g.flight[i].job);
            let st = self.step(&mut g.flight[i], w);
            match st {
                Step::Running => i += 1,
                Step::Done | Step::Failed => {
                    let r = g.flight.swap_remove(i);
                    Self::land(&mut g, r, matches!(st, Step::Done), loading);
                }
            }
        }
        // a failure is kept: the slot the read was for is still indexed, and must never be computed on
        !g.failed
    }
}

impl Drop for Overlap {
    fn drop(&mut self) {
        let mut g = self.inner.lock().unwrap();
        let flight = std::mem::take(&mut g.flight);
        for mut r in flight {
            // the buffer must not be freed under a running read
            let _ = self.step(&mut r, true);
            g.free.push(r.o.event);
        }
        for &e in &g.free {
            // SAFETY: events this cache created
            unsafe { CloseHandle(e) };
        }
    }
}
