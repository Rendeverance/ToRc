//! disk.rs - drive random reads: the expert-miss pattern. 2 MiB blocks at random 4 KiB-aligned offsets, cache-bypassing
//! (FILE_FLAG_NO_BUFFERING), at queue depth 1 and 8 (one thread + one handle per outstanding read).
//!
//! Reads an existing large model file when the drive has one; otherwise a 2 GiB temp file in %TEMP% that is deleted
//! afterwards. Nothing else is written.

use super::{Mode, Rng, median, percentile};
use crate::inventory::drives::Drive;
use crate::sys::{self, AlignedBuf};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Barrier;
use std::time::{Duration, Instant};

const BLOCK: usize = 2 << 20;
const ALIGN: usize = 4096;
const READS: usize = 256;
const TEMP_BYTES: u64 = 2 << 30;
const TEMP_MARGIN: u64 = 4 << 30;

#[derive(Debug, Clone, Serialize)]
pub struct QdResult {
    pub queue_depth: usize,
    pub reads: usize,
    /// MB = 1e6 bytes.
    pub mbs: f64,
    pub lat_ms_median: f64,
    pub lat_ms_p90: f64,
    pub lat_ms_p99: f64,
    pub lat_ms_max: f64,
    pub seconds: f64,
    pub capped: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DriveResult {
    pub root: String,
    pub bus: String,
    pub target_kind: &'static str,
    pub target: String,
    pub file_bytes: u64,
    pub block_bytes: usize,
    pub temp_write_mbs: Option<f64>,
    pub qd1: Option<QdResult>,
    pub qd8: Option<QdResult>,
    pub error: Option<String>,
}

/// Deletes the temp file however the test ends.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn read_full(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        let n = sys::read_at(f, &mut buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        }
        done += n;
    }
    Ok(())
}

fn random_offset(rng: &mut Rng, file_bytes: u64) -> u64 {
    rng.below((file_bytes - BLOCK as u64) / ALIGN as u64) * ALIGN as u64
}

fn seed() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0x9E37, |d| d.as_nanos() as u64)
}

/// One worker's outcome: first start (ns), last end (ns), per-read latencies (ms), hit the cap.
type WorkerOut = Result<(u128, u128, Vec<f64>, bool), String>;

/// `depth` threads, each with its own handle and buffer, READS reads in total.
fn run_qd(path: &Path, file_bytes: u64, depth: usize, cap: Duration, seed: u64) -> Result<QdResult, String> {
    let per = READS / depth;
    let barrier = Barrier::new(depth);
    let base = Instant::now();
    let results: Vec<WorkerOut> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..depth)
            .map(|t| {
                let b = &barrier;
                s.spawn(move || {
                    let f = sys::open_unbuffered(path, false).map_err(|e| format!("open: {e}"));
                    let mut buf = AlignedBuf::new(BLOCK, ALIGN).ok_or("alloc failed".to_string());
                    b.wait();
                    let (f, buf) = (f?, buf.as_mut().map_err(|e| e.clone())?);
                    let mut rng = Rng::new(seed ^ (t as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let t0 = base.elapsed().as_nanos();
                    let mut lat = Vec::with_capacity(per);
                    let mut capped = false;
                    for _ in 0..per {
                        if base.elapsed() > cap {
                            capped = true;
                            break;
                        }
                        let off = random_offset(&mut rng, file_bytes);
                        let r = Instant::now();
                        read_full(&f, buf.as_mut_slice(), off).map_err(|e| format!("read @{off}: {e}"))?;
                        lat.push(r.elapsed().as_secs_f64() * 1e3);
                    }
                    Ok((t0, base.elapsed().as_nanos(), lat, capped))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("thread panicked".into()))).collect()
    });
    let mut lat = Vec::new();
    let (mut t0, mut t1, mut capped) = (u128::MAX, 0u128, false);
    for r in results {
        let (a, b, l, c) = r?;
        t0 = t0.min(a);
        t1 = t1.max(b);
        capped |= c;
        lat.extend(l);
    }
    let secs = (t1.saturating_sub(t0)) as f64 / 1e9;
    Ok(QdResult {
        queue_depth: depth,
        reads: lat.len(),
        mbs: if secs > 0.0 { (lat.len() * BLOCK) as f64 / secs / 1e6 } else { 0.0 },
        lat_ms_median: median(&lat),
        lat_ms_p90: percentile(&lat, 90.0),
        lat_ms_p99: percentile(&lat, 99.0),
        lat_ms_max: lat.iter().copied().fold(0.0, f64::max),
        seconds: secs,
        capped,
    })
}

/// Two uncounted reads: wakes a sleeping drive (USB bridges idle down) and checks access.
fn warm_up(path: &Path, file_bytes: u64) -> Result<(), String> {
    let f = sys::open_unbuffered(path, false).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf = AlignedBuf::new(BLOCK, ALIGN).ok_or("alloc failed")?;
    let mut rng = Rng::new(seed());
    for _ in 0..2 {
        let off = random_offset(&mut rng, file_bytes);
        read_full(&f, buf.as_mut_slice(), off).map_err(|e| format!("read: {e}"))?;
    }
    Ok(())
}

/// Write a 2 GiB file of non-repeating data with unbuffered, write-through I/O.
fn make_temp(path: &Path) -> Result<f64, String> {
    let chunk = 8 << 20;
    let mut buf = AlignedBuf::new(chunk, ALIGN).ok_or("alloc failed")?;
    let mut rng = Rng::new(seed());
    for w in buf.as_mut_slice().chunks_mut(8) {
        let v = rng.next_u64().to_ne_bytes();
        w.copy_from_slice(&v[..w.len()]);
    }
    let mut f = sys::open_unbuffered(path, true).map_err(|e| format!("create {}: {e}", path.display()))?;
    let t = Instant::now();
    for i in 0..(TEMP_BYTES / chunk as u64) {
        // Make every 4 KiB page unique so nothing can dedupe or compress it.
        for page in buf.as_mut_slice().chunks_mut(ALIGN) {
            page[..8].copy_from_slice(&i.to_ne_bytes());
        }
        f.write_all(buf.as_slice()).map_err(|e| format!("write: {e}"))?;
    }
    f.sync_all().map_err(|e| format!("flush: {e}"))?;
    Ok(TEMP_BYTES as f64 / t.elapsed().as_secs_f64() / 1e6)
}

fn test_file(d: &Drive, kind: &'static str, path: &Path, bytes: u64, mode: Mode, log: &dyn Fn(&str)) -> DriveResult {
    let mut r = DriveResult {
        root: d.root.clone(),
        bus: d.bus.clone(),
        target_kind: kind,
        target: path.display().to_string(),
        file_bytes: bytes,
        block_bytes: BLOCK,
        temp_write_mbs: None,
        qd1: None,
        qd8: None,
        error: None,
    };
    let cap = mode.pick(Duration::from_secs(10), Duration::from_secs(4));
    if let Err(e) = warm_up(path, bytes) {
        r.error = Some(e);
        return r;
    }
    log(&format!("drive {}: QD1", d.root));
    match run_qd(path, bytes, 1, cap, seed()) {
        Ok(q) => r.qd1 = Some(q),
        Err(e) => r.error = Some(e),
    }
    log(&format!("drive {}: QD8", d.root));
    match run_qd(path, bytes, 8, cap, seed().rotate_left(17)) {
        Ok(q) => r.qd8 = Some(q),
        Err(e) => r.error = Some(e),
    }
    r
}

pub fn run(drives: &[Drive], mode: Mode, log: &dyn Fn(&str)) -> Vec<DriveResult> {
    let mut out = Vec::new();
    for d in drives {
        if let (Some(p), Some(b)) = (&d.largest_model_file, d.largest_model_bytes) {
            out.push(test_file(d, "model file", Path::new(p), b, mode, log));
        }
    }

    // The %TEMP% drive gets a temp-file test if it had no model file (full mode always; quick mode only when nothing
    // else was tested).
    let tmp_dir = std::env::temp_dir();
    let tmp_root = tmp_dir.to_string_lossy().chars().take(3).collect::<String>().to_ascii_uppercase();
    let tmp_drive = drives.iter().find(|d| d.root.eq_ignore_ascii_case(&tmp_root));
    let already = out.iter().any(|r| r.root.eq_ignore_ascii_case(&tmp_root));
    if already || (mode == Mode::Quick && !out.is_empty()) {
        return out;
    }
    let Some(d) = tmp_drive else { return out };
    let path = tmp_dir.join(format!("hwscan-{}.tmp", std::process::id()));
    if d.free_bytes.is_some_and(|f| f < TEMP_BYTES + TEMP_MARGIN) {
        out.push(DriveResult {
            root: d.root.clone(),
            bus: d.bus.clone(),
            target_kind: "temp file",
            target: path.display().to_string(),
            file_bytes: 0,
            block_bytes: BLOCK,
            temp_write_mbs: None,
            qd1: None,
            qd8: None,
            error: Some("skipped: not enough free space for a 2 GiB temp file".into()),
        });
        return out;
    }
    log(&format!("drive {}: writing 2 GiB temp file {}", d.root, path.display()));
    let guard = TempFile(path.clone());
    let write = make_temp(&path);
    let mut r = match write {
        Ok(_) => test_file(d, "temp file", &path, TEMP_BYTES, mode, log),
        Err(ref e) => DriveResult {
            root: d.root.clone(),
            bus: d.bus.clone(),
            target_kind: "temp file",
            target: path.display().to_string(),
            file_bytes: 0,
            block_bytes: BLOCK,
            temp_write_mbs: None,
            qd1: None,
            qd8: None,
            error: Some(e.clone()),
        },
    };
    r.temp_write_mbs = write.ok();
    drop(guard);
    out.push(r);
    out
}
