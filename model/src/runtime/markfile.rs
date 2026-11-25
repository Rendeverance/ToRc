//! markfile.rs - a server's saved states on disk: the system prompt's (torc-server's sysmark) and whole chats' (saved
//! when the server is idle), each written and read by Model::mark_save / mark_load. Each machine keeps its own layers'
//! half in TORC_MARK_DIR (by default `marks` next to the executable), one file per key (the model's identity, the
//! layers, the positions and the tokens: see Model::mark_key), with at most TORC_MARKS files (default 16, the oldest
//! removed; 0 keeps none and reads nothing).
//!
//! The file holds "TORCMRK1", the model's identity (u64), layers l0 / l1 and the positions (u32 each), the payload's
//! length (u64) and its BLAKE3 hash (32 bytes), then the payload (Model::state_bytes). A file that does not match all
//! of these is not loaded.

use std::path::PathBuf;

const MAGIC: &[u8; 8] = b"TORCMRK1";

/// This machine's build and the settings that change its numbers (the executable's size and time, TORC_NPU,
/// TORC_NPU_HOT, TORC_GPU_ATTN, TORC_TAIL_SKIP and the rounding tests TORC_ATTN_EXACT, TORC_ROUTER_F32 and
/// TORC_REC_F32, and TORC_NGRAM_IDMAP and TORC_MTP_FILE with their files' size and time), folded into the identity a
/// file is checked against. So a state saved by another build, with the NPU or a GPU tier switched, with another id map
/// or with another MTP file is never read: a sequence resumed from it would not give the tokens of a fresh read (a
/// GPU's products, for example, are not the CPU's bits). TORC_MTP is in it too: a server without drafts saves a state
/// without the MTP block's rows, and one with drafts that read it would draft over nothing.
fn fingerprint() -> u64 {
    static F: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let mut h = blake3::Hasher::new();
        if let Some(m) = std::env::current_exe().ok().and_then(|p| std::fs::metadata(p).ok()) {
            h.update(&m.len().to_le_bytes());
            let t =
                m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
            h.update(&t.to_le_bytes());
        }
        for v in [
            "TORC_NPU",
            "TORC_NPU_HOT",
            "TORC_GPU_ATTN",
            "TORC_TAIL_SKIP",
            "TORC_ATTN_EXACT",
            "TORC_ROUTER_F32",
            "TORC_REC_F32",
            "TORC_KV16",
            "TORC_MTP",
        ] {
            h.update(v.as_bytes());
            h.update(crate::switches::get(v).unwrap_or_default().as_bytes());
            h.update(&[0]);
        }
        // a file-valued setting: the file may change under the same name
        for v in ["TORC_NGRAM_IDMAP", "TORC_MTP_FILE"] {
            let p = crate::switches::get(v).unwrap_or_default();
            h.update(v.as_bytes());
            h.update(p.as_bytes());
            h.update(&[0]);
            if let Ok(m) = std::fs::metadata(&p) {
                h.update(&m.len().to_le_bytes());
                let t = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                h.update(&t.to_le_bytes());
            }
        }
        u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap())
    })
}

/// How many files a machine keeps (TORC_MARKS, default 16; 0 turns it off).
pub fn keep() -> usize {
    crate::switches::num("TORC_MARKS").unwrap_or(16)
}

fn dir() -> PathBuf {
    match crate::switches::get("TORC_MARK_DIR") {
        Some(d) => PathBuf::from(d),
        None => std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("marks")))
            .unwrap_or_else(|| PathBuf::from("marks")),
    }
}

fn path(key: u64, who: &str) -> PathBuf {
    dir().join(format!("{key:016x}-{who}.mark"))
}

pub fn exists(key: u64, who: &str) -> bool {
    keep() > 0 && path(key, who).is_file()
}

/// Writes the file (through a temporary name, so a crash never leaves half a file), then keeps the newest keep() files.
pub fn save(key: u64, who: &str, ident: u64, l0: u32, l1: u32, npos: u32, payload: &[u8]) -> std::io::Result<()> {
    if keep() == 0 {
        return Ok(());
    }
    let d = dir();
    std::fs::create_dir_all(&d)?;
    let mut b = Vec::with_capacity(72 + payload.len());
    b.extend_from_slice(MAGIC);
    b.extend((ident ^ fingerprint()).to_le_bytes());
    for v in [l0, l1, npos] {
        b.extend(v.to_le_bytes());
    }
    b.extend((payload.len() as u64).to_le_bytes());
    b.extend_from_slice(blake3::hash(payload).as_bytes());
    b.extend_from_slice(payload);
    let p = path(key, who);
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, &b)?;
    std::fs::rename(&tmp, &p)?;
    prune(&d, ident ^ fingerprint());
    Ok(())
}

/// The payload of the file for `key`, and its end layer, if the file is there and matches the identity, start layer and
/// positions asked for, and the end layer when one is asked for (None accepts whichever it holds, as for a main
/// machine's split).
pub fn load(key: u64, who: &str, ident: u64, l0: u32, l1: Option<u32>, npos: u32) -> Option<(Vec<u8>, u32)> {
    if keep() == 0 {
        return None;
    }
    let p = path(key, who);
    let b = std::fs::read(&p).ok()?;
    let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
    let ok = b.len() >= 68
        && &b[..8] == MAGIC
        && u64_at(8) == ident ^ fingerprint()
        && u32_at(16) == l0
        && l1.is_none_or(|l| u32_at(20) == l)
        && u32_at(24) == npos
        && u64_at(28) as usize == b.len() - 68
        && blake3::hash(&b[68..]).as_bytes()[..] == b[36..68];
    if !ok {
        eprintln!("marks: {} does not match this model or is damaged: not used", p.display());
        return None;
    }
    // touched, so the newest stay when the oldest are removed
    let _ = std::fs::File::options().append(true).open(&p).and_then(|f| f.set_modified(std::time::SystemTime::now()));
    Some((b[68..].to_vec(), u32_at(20)))
}

/// Saves the tokens a main machine's state was saved after, next to its file (<key>-m.tok: the identity, then the
/// tokens), for index(), so that a new prompt can start from the longest saved state it begins with.
pub fn save_tokens(key: u64, ident: u64, tokens: &[u32]) -> std::io::Result<()> {
    if keep() == 0 {
        return Ok(());
    }
    let mut b = Vec::with_capacity(8 + tokens.len() * 4);
    b.extend((ident ^ fingerprint()).to_le_bytes());
    for t in tokens {
        b.extend(t.to_le_bytes());
    }
    std::fs::write(dir().join(format!("{key:016x}-m.tok")), b)
}

/// The main machine's saved states for this model (ident): their tokens, from the .tok files that have their state
/// file.
pub fn index(ident: u64) -> Vec<Vec<u32>> {
    let Ok(rd) = std::fs::read_dir(dir()) else { return Vec::new() };
    rd.filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "tok") && p.with_extension("mark").is_file())
        .filter_map(|p| std::fs::read(&p).ok())
        .filter(|b| {
            b.len() >= 8 && b.len() % 4 == 0 && u64::from_le_bytes(b[..8].try_into().unwrap()) == ident ^ fingerprint()
        })
        .map(|b| b[8..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
        .collect()
}

/// A file's identity (bytes 8..16), or 0 for a file that is too short or not a mark file.
fn file_ident(p: &PathBuf) -> u64 {
    use std::io::Read;
    let mut b = [0u8; 16];
    match std::fs::File::open(p).and_then(|mut f| f.read_exact(&mut b)) {
        Ok(()) if &b[..8] == MAGIC => u64::from_le_bytes(b[8..16].try_into().unwrap()),
        _ => 0,
    }
}

/// Keeps the newest keep() files of this model and build (`mine`: identity ^ fingerprint), and keep() more of all the
/// others together, so another model's states survive a switch away and back (otherwise one model's saves pushed out
/// another's). A key's files (main machine, worker) count as one entry each, and age decides, newest first.
fn prune(d: &PathBuf, mine: u64) {
    let Ok(rd) = std::fs::read_dir(d) else { return };
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "mark"))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
        .collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.0));
    let (mut n_mine, mut n_other) = (0usize, 0usize);
    for (_, p) in v {
        let n = if file_ident(&p) == mine { &mut n_mine } else { &mut n_other };
        *n += 1;
        if *n > keep() {
            let _ = std::fs::remove_file(&p);
            let _ = std::fs::remove_file(p.with_extension("tok"));
        }
    }
}

/// The folder every test that saves states uses: TORC_MARK_DIR is one setting for the whole process, and the tests
/// run at the same time, so they share it and keep apart by their models' identities. One folder per test process, so
/// two `cargo test` runs never clear each other's files.
#[cfg(test)]
pub(crate) fn test_dir() {
    static D: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let d = std::env::temp_dir().join(format!("torc-marks-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        crate::switches::set("TORC_MARK_DIR", &d.to_string_lossy());
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_round_trip_and_refusals() {
        test_dir();
        let payload: Vec<u8> = (0..1000u32).flat_map(|v| v.to_le_bytes()).collect();
        save(7, "m", 42, 0, 22, 518, &payload).unwrap();
        save_tokens(7, 42, &[1, 2, 3]).unwrap();
        let (b, l1) = load(7, "m", 42, 0, None, 518).unwrap();
        assert_eq!((b, l1), (payload.clone(), 22));
        assert!(load(7, "m", 42, 0, Some(22), 518).is_some());
        // another model, another split asked for, or other positions: not loaded
        assert!(load(7, "m", 43, 0, None, 518).is_none());
        assert!(load(7, "m", 42, 0, Some(21), 518).is_none());
        assert!(load(7, "m", 42, 0, None, 517).is_none());
        assert_eq!(index(42), vec![vec![1, 2, 3]]);
        assert!(index(43).is_empty());
        // a damaged payload fails its hash
        let p = path(7, "m");
        let mut raw = std::fs::read(&p).unwrap();
        let n = raw.len();
        raw[n - 1] ^= 1;
        std::fs::write(&p, raw).unwrap();
        assert!(load(7, "m", 42, 0, None, 518).is_none());
    }
}
