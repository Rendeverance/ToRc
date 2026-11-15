//! switches.rs - the TORC_ environment switches, read once. The first use takes every TORC_ variable as the process
//! was started with, and nothing looks at the environment for one after that, so a run cannot change its settings
//! half way through. A program that fixes a switch for the code under it (the tuner trying a split, a benchmark
//! pinning the GPU's share) says so with `set`, before that code runs. SWITCHES.md lists every switch, and a test
//! (model/tests/switches.rs) keeps that list and the code in step.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{OnceLock, RwLock};

fn all() -> &'static RwLock<HashMap<String, String>> {
    static ALL: OnceLock<RwLock<HashMap<String, String>>> = OnceLock::new();
    ALL.get_or_init(|| {
        // a value that is not Unicode counts as not set, as it always has
        RwLock::new(
            std::env::vars_os()
                .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
                .filter(|(k, _)| k.starts_with("TORC_"))
                .collect(),
        )
    })
}

/// A switch's value, when it is set.
pub fn get(name: &str) -> Option<String> {
    all().read().unwrap().get(name).cloned()
}

/// Whether the switch is set, to anything.
pub fn given(name: &str) -> bool {
    all().read().unwrap().contains_key(name)
}

/// Whether the switch is set to exactly `value`.
pub fn is(name: &str, value: &str) -> bool {
    all().read().unwrap().get(name).is_some_and(|v| v == value)
}

/// Whether the switch is set to 1.
pub fn on(name: &str) -> bool {
    is(name, "1")
}

/// The switch as a number, when it is set and reads as one.
pub fn num<T: FromStr>(name: &str) -> Option<T> {
    all().read().unwrap().get(name).and_then(|v| v.parse().ok())
}

/// Sets a switch for the rest of the run. For a program that fixes one for the code under it; call it before that
/// code first runs, since much of it reads its switches once and keeps the answer.
pub fn set(name: &str, value: &str) {
    all().write().unwrap().insert(name.to_string(), value.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_switch_reads_as_what_it_was_set_to() {
        assert!(!given("TORC_TEST_SWITCH") && get("TORC_TEST_SWITCH").is_none() && !on("TORC_TEST_SWITCH"));
        assert_eq!(num::<usize>("TORC_TEST_SWITCH"), None);
        set("TORC_TEST_SWITCH", "1");
        assert!(
            given("TORC_TEST_SWITCH")
                && on("TORC_TEST_SWITCH")
                && is("TORC_TEST_SWITCH", "1")
                && !is("TORC_TEST_SWITCH", "0")
        );
        assert_eq!(get("TORC_TEST_SWITCH").as_deref(), Some("1"));
        assert_eq!(num::<usize>("TORC_TEST_SWITCH"), Some(1));
        set("TORC_TEST_SWITCH", "fast");
        assert_eq!(num::<usize>("TORC_TEST_SWITCH"), None);
        assert!(!on("TORC_TEST_SWITCH"));
    }
}
