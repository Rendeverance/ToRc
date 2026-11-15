//! ecache.rs - the routed experts of a model bigger than RAM.
//!
//! The cache itself is in ecache_own.rs; this file picks the budget, keeps the usage file and counts the requests.
//! Without a cache, a token's experts come from the mapping, page fault by page fault. The bytes are the file's, so the
//! products are the same bits either way.
//!
//! TORC_CACHE_MB=m sets the budget (0 turns it off). By default the cache is on when the file does not fit in 80% of
//! the installed memory, and it takes the free memory less the resident weights and a margin (6 GB, or 3 GB on one
//! machine of two), at least 2 GB (weights.rs).

use format::Result;
use std::sync::Mutex;

/// The cache's counters since it opened. An expert here is one (layer, expert) a request names.
#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    /// experts named by requests
    pub asked: u64,
    /// of those, already in a slot
    pub resident: u64,
    /// of those, read from the file for the request, and the bytes read
    pub loaded: u64,
    pub loaded_bytes: u64,
    /// read-ahead: reads queued, and their bytes
    pub ahead: u64,
    pub ahead_bytes: u64,
    /// bytes read by the warm-up at open
    pub warmup_bytes: u64,
    /// seconds callers spent blocked on reads
    pub blocked: f64,
}

pub struct ExpertCache {
    /// the cache (ecache_own.rs, EXPERT-CACHE.md)
    own: crate::ecache_own::OwnCache,
    pub budget: u64,
    /// this session's requests per (layer, expert), added to the machine's usage file at the end
    used: Mutex<Vec<u32>>,
    n_expert: usize,
    usage: Option<std::path::PathBuf>,
    /// the model file's id (its header's BLAKE3 model id), written into a usage file this engine starts
    model: [u8; 32],
}

/// A usage file: how often each (layer, expert) was asked for on this machine, kept to warm the cache at the next
/// start and to pick the NPU's hot experts. Little-endian: "TORCUSE1", the model id (32 bytes; all zero for a file
/// not tied to one model file), u32 layers, u32 experts per layer, then a u64 count per (layer, expert), layer by
/// layer.
pub(crate) const USAGE_MAGIC: &[u8; 8] = b"TORCUSE1";
const USAGE_HEAD: usize = 48;

/// Reads a usage file of n_layer x n_expert counts; `model`, when given, must match the file's model id.
pub(crate) fn usage_read(
    path: &std::path::Path,
    model: Option<&[u8; 32]>,
    n_layer: u32,
    n_expert: u32,
) -> std::result::Result<Vec<u64>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let le32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    if b.len() < USAGE_HEAD || &b[..8] != USAGE_MAGIC {
        return Err(format!("{}: not a ToRc usage file", path.display()));
    }
    if model.is_some_and(|m| b[8..40] != m[..]) {
        return Err(format!("{}: written for another model file", path.display()));
    }
    let n = n_layer as usize * n_expert as usize;
    if (le32(40), le32(44)) != (n_layer, n_expert) || b.len() != USAGE_HEAD + 8 * n {
        return Err(format!(
            "{}: {} layers x {} experts, this model has {n_layer} x {n_expert}",
            path.display(),
            le32(40),
            le32(44)
        ));
    }
    Ok(b[USAGE_HEAD..].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Writes a usage file (`counts` holds n_layer x n_expert), through a temporary name in the same folder so a reader
/// never sees half of one.
pub(crate) fn usage_write(
    path: &std::path::Path,
    model: &[u8; 32],
    n_layer: u32,
    n_expert: u32,
    counts: &[u64],
) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(USAGE_HEAD + 8 * counts.len());
    b.extend_from_slice(USAGE_MAGIC);
    b.extend_from_slice(model);
    b.extend(n_layer.to_le_bytes());
    b.extend(n_expert.to_le_bytes());
    for c in counts {
        b.extend(c.to_le_bytes());
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    // the process id keeps two engines on one model (a server and a benchmark) off each other's temporary file
    let tmp = path.with_file_name(format!(
        ".{}.{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    std::fs::write(&tmp, &b)?;
    std::fs::rename(&tmp, path)
}

/// Where this machine keeps the usage of the model with this id: usage/<id, 32 hex digits> in ToRc's cache folder
/// (tune::cache_dir).
pub(crate) fn usage_path(model: &[u8; 32]) -> Option<std::path::PathBuf> {
    let id: String = model[..16].iter().map(|x| format!("{x:02x}")).collect();
    Some(crate::tune::cache_dir()?.join("usage").join(id))
}

impl ExpertCache {
    /// A cache of `budget` bytes over the file at `path` (model id `model`), read by `io_threads` I/O threads.
    pub fn open(
        path: &str,
        budget: u64,
        io_threads: i32,
        model: [u8; 32],
        n_layer: u32,
        n_expert: u32,
    ) -> Result<ExpertCache> {
        let usage = usage_path(&model);
        let warm = if crate::switches::is("TORC_WARM_PROFILE", "0") {
            None
        } else {
            let seen = usage.as_deref().and_then(|p| usage_read(p, Some(&model), n_layer, n_expert).ok());
            seen.map(|c| c.into_iter().map(|v| v.min(u32::MAX as u64) as u32).collect::<Vec<u32>>()).map(|mut c| {
                if let Some((a, b)) = crate::switches::get("TORC_CACHE_LAYERS").and_then(|v| {
                    let (a, b) = v.split_once('-')?;
                    Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))
                }) {
                    for (l, row) in c.chunks_mut(n_expert as usize).enumerate() {
                        if l < a || l >= b {
                            row.fill(0);
                        }
                    }
                }
                c
            })
        };
        let own = crate::ecache_own::OwnCache::open(path, budget, io_threads.max(1) as usize, warm.as_deref())?;
        Ok(ExpertCache {
            own,
            budget,
            used: Mutex::new(vec![0; (n_layer * n_expert) as usize]),
            n_expert: n_expert as usize,
            usage,
            model,
        })
    }

    /// Pointers to the packs of experts `ids` of `layer`, all loaded by the time this returns. The next take() or ask()
    /// may reuse their slots, so finish computing on them before calling again. At most layer_slots(layer) ids.
    pub fn take(&self, layer: u32, ids: &[u32]) -> Result<Vec<*const u8>> {
        self.count(layer, ids);
        self.own.take(layer, ids)
    }

    /// take, split in two so the computing can start early: the pointers, and for each whether its pack is already
    /// loaded. A pack marked false may be read only after settle(). Call settle() before asking again.
    pub fn ask(&self, layer: u32, ids: &[u32]) -> Result<(Vec<*const u8>, Vec<bool>)> {
        self.count(layer, ids);
        self.own.ask(layer, ids)
    }

    /// Loads ahead of use: up to `max` of `ids` (ranked best first) for `layer`, skipping those already held. The
    /// router's guess for the next layer and the idle time between requests both come here. Gives the loads set going.
    pub fn read_ahead(&self, layer: u32, ids: &[u32], max: usize) -> usize {
        self.own.read_ahead(layer, ids, max)
    }

    /// Returns once every pack the last ask() left loading is in.
    pub fn settle(&self) -> Result<()> {
        self.own.settle()
    }

    fn count(&self, layer: u32, ids: &[u32]) {
        let mut u = self.used.lock().unwrap();
        for &e in ids {
            if let Some(c) = u.get_mut(layer as usize * self.n_expert + e as usize) {
                *c = c.saturating_add(1);
            }
        }
    }

    /// Adds this session's counts to the machine's usage file for the model (starting one if there is none, or if the
    /// one there does not fit). The counts are u64, so they never need scaling down.
    fn usage_save(&self) {
        let (Some(path), Ok(u)) = (self.usage.as_deref(), self.used.lock()) else { return };
        if u.iter().all(|&v| v == 0) {
            return;
        }
        let (nl, ne) = ((u.len() / self.n_expert) as u32, self.n_expert as u32);
        let mut total = usage_read(path, Some(&self.model), nl, ne).unwrap_or_else(|_| vec![0; u.len()]);
        for (t, &v) in total.iter_mut().zip(u.iter()) {
            *t = t.saturating_add(v as u64);
        }
        let _ = usage_write(path, &self.model, nl, ne, &total);
    }

    /// Saves this session's counts now and starts counting again (for a server after each reply, and a pipeline
    /// worker when its main machine hangs up, since neither may exit cleanly).
    pub fn usage_flush(&self) {
        self.usage_save();
        if let Ok(mut u) = self.used.lock() {
            u.fill(0);
        }
    }

    /// For each of `ids` of `layer`: whether its pack is in memory or already being read. A pure lookup (no loads, no
    /// change to what gets evicted); use it between a settle() and the next take() or ask().
    pub fn held(&self, layer: u32, ids: &[u32]) -> Vec<bool> {
        self.own.held(layer, ids)
    }

    /// The slots a request for the layer may take (from its size class).
    pub fn layer_slots(&self, layer: u32) -> usize {
        self.own.layer_slots(layer)
    }

    pub fn stats(&self) -> Stats {
        self.own.stats()
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        self.usage_save();
    }
}
