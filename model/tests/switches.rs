//! switches.rs - SWITCHES.md and the code agree: every TORC_ name the code reads from the environment is listed
//! there, and every name listed there is read somewhere.

use std::collections::BTreeSet;
use std::path::Path;

/// The names in the code that are not switches: the switch module's own test name and this file's example.
const NOT_SWITCHES: [&str; 2] = ["TORC_TEST_SWITCH", "TORC_NAME"];

/// Every "TORC_NAME" string in the Rust files under `dir`.
fn in_code(dir: &Path, out: &mut BTreeSet<String>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n != "target") {
                in_code(&p, out);
            }
        } else if p.extension().is_some_and(|x| x == "rs") {
            names(&std::fs::read_to_string(&p).unwrap(), '"', out);
        }
    }
    out.retain(|n| !NOT_SWITCHES.contains(&n.as_str()));
}

/// The TORC_ names in `text` that have a `quote` character on each side.
fn names(text: &str, quote: char, out: &mut BTreeSet<String>) {
    for part in text.split(quote) {
        if part.len() > 5
            && part.starts_with("TORC_")
            && part.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
        {
            out.insert(part.to_string());
        }
    }
}

#[test]
fn the_doc_lists_what_the_code_reads() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut code = BTreeSet::new();
    for krate in ["model", "kernels", "format", "hwscan"] {
        for dir in ["src", "tests"] {
            let d = root.join(krate).join(dir);
            if d.is_dir() {
                in_code(&d, &mut code);
            }
        }
    }
    let mut doc = BTreeSet::new();
    names(&std::fs::read_to_string(root.join("SWITCHES.md")).unwrap(), '`', &mut doc);
    let missing: Vec<&String> = code.difference(&doc).collect();
    let stale: Vec<&String> = doc.difference(&code).collect();
    assert!(missing.is_empty(), "read by the code but not in SWITCHES.md: {missing:?}");
    assert!(stale.is_empty(), "in SWITCHES.md but read nowhere: {stale:?}");
}
