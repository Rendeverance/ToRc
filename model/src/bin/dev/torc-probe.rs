//! torc-probe: what the drive under a model file does with the kind of reads an expert cache makes. It reads a given
//! size, unbuffered (no file cache: Windows FILE_FLAG_NO_BUFFERING), at random 4 KiB-aligned places in the file, from 1
//! to 16 threads at once, for a few seconds each. Each thread has one read in flight, so the thread count is the queue
//! depth. It reports MB/s, reads a second, and the median and 99th-percentile time of one read.
//! Its figures are what the drive settings are chosen from (slot size, I/O threads, how experts are grouped on
//! disk). It only reads; the file is not changed.
//!
//!   torc-probe <model file> [seconds per point, default 2] [sizes in KiB, default 64,256,1024,2048,4096]

use std::fs::File;
use std::time::{Duration, Instant};

const ALIGN: usize = 4096;

fn open_unbuffered(path: &str) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path)
    }
    #[cfg(not(windows))]
    {
        File::open(path)
    }
}

fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..], off + done as u64)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::other("short read"));
        }
        done += n;
    }
    Ok(())
}

/// A 4 KiB-aligned buffer (unbuffered reads need the address on a sector boundary).
struct Aligned {
    ptr: *mut u8,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Aligned {
        let layout = std::alloc::Layout::from_size_align(len, ALIGN).unwrap();
        // SAFETY: len > 0 (a probe size); freed in drop with the same layout
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Aligned { ptr, len }
    }
    fn slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr holds len bytes, owned by self
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: allocated in new with this layout
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, ALIGN).unwrap()) }
    }
}

/// One point: `threads` threads, each reading `size` bytes at random places until `secs` have passed, each through its
/// own handle (Windows serialises synchronous reads on one handle, as ecache_own's readers note). Returns (MB/s,
/// reads/s, median ms, p99 ms).
fn point(path: &str, file_len: u64, size: usize, threads: usize, secs: f64) -> (f64, f64, f64, f64) {
    let files: Vec<File> = (0..threads).map(|_| open_unbuffered(path).expect("open")).collect();
    let t0 = Instant::now();
    let until = t0 + Duration::from_secs_f64(secs);
    let handles: Vec<_> = files
        .into_iter()
        .enumerate()
        .map(|(t, f)| {
            std::thread::spawn(move || {
                let mut buf = Aligned::new(size);
                let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ ((t as u64 + 1) * 0xbf58_476d_1ce4_e5b9) ^ size as u64;
                let span = (file_len - size as u64) / ALIGN as u64;
                let mut lat = Vec::new();
                while Instant::now() < until {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let off = (s % span) * ALIGN as u64;
                    let a = Instant::now();
                    read_at(&f, buf.slice(), off).expect("read");
                    lat.push(a.elapsed().as_secs_f64());
                }
                lat
            })
        })
        .collect();
    let mut lat: Vec<f64> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    let el = t0.elapsed().as_secs_f64();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lat.len().max(1);
    let q = |p: f64| lat.get(((n - 1) as f64 * p) as usize).copied().unwrap_or(0.0) * 1e3;
    (lat.len() as f64 * size as f64 / el / 1e6, lat.len() as f64 / el, q(0.5), q(0.99))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: torc-probe <model file> [seconds per point] [sizes in KiB, comma separated]");
        std::process::exit(2);
    };
    let secs: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let sizes: Vec<usize> = args
        .get(2)
        .map_or("64,256,1024,2048,4096", |s| s.as_str())
        .split(',')
        .filter_map(|k| k.trim().parse::<usize>().ok())
        .map(|k| k << 10)
        .collect();
    let f = open_unbuffered(path).unwrap_or_else(|e| {
        eprintln!("torc-probe: {path}: {e}");
        std::process::exit(1)
    });
    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    println!("{path}: {:.1} GiB, unbuffered random reads, {secs} s a point", file_len as f64 / (1u64 << 30) as f64);
    println!("{:>8} {:>7} {:>9} {:>9} {:>8} {:>8}", "KiB", "threads", "MB/s", "reads/s", "p50 ms", "p99 ms");
    for &size in &sizes {
        if size as u64 * 2 > file_len || size % ALIGN != 0 {
            continue;
        }
        for threads in [1usize, 2, 4, 8, 16] {
            let (mbs, rps, p50, p99) = point(path, file_len, size, threads, secs);
            println!("{:>8} {threads:>7} {mbs:>9.0} {rps:>9.0} {p50:>8.2} {p99:>8.2}", size >> 10);
        }
    }
}
