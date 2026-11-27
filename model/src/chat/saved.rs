//! saved.rs - saved chats on disk for Gemma 4, gpt-oss and Inkling 2 (their serve.rs). Gemma's and gpt-oss's whole
//! state is a positional cache, so the cache's first n rows are the state after the prompt's first n tokens, whatever
//! was read after them; Inkling's also has a convolution state, which its marks keep at each prompt's end (its
//! serve.rs gives the state from the mark, or none once its ring has gone past it). A chat is saved (markfile) by the rules torc-server uses for the Qwen family: when a request for another
//! chat comes in and the one being left has had two turns or more, or when no request has come for TORC_IDLE_SAVE
//! seconds and it has had two turns or a prompt of TORC_IDLE_SAVE_MIN tokens; never under 512 tokens. What is saved is
//! the chat's last prompt, which the next turn's prompt begins with. A prompt that begins with a saved chat loads it
//! and reads only the rest, when that is at least 512 tokens less to read than the cache in memory leaves.

/// What saving and loading needs from a model.
pub trait Positional {
    /// The cache at positions 0..npos, every layer.
    fn state_bytes(&self, npos: usize) -> Vec<u8>;
    /// Puts state_bytes back. Fails if they do not fit, leaving the cache partly written.
    fn set_state_bytes(&mut self, npos: usize, b: &[u8]) -> format::Result<()>;
    fn n_layer(&self) -> usize;
}

/// The shortest chat worth a file, and the least reading a load must save.
const MIN: usize = 512;

pub struct Saved {
    /// off when the model's state is not in its CPU caches (Gemma's whole-NPU tier), or with TORC_MARKS=0
    on: bool,
    /// the model's identity in the files: its file's id and the settings that change its numbers
    ident: u64,
    /// the tokens of this model's saved chats
    index: Vec<Vec<u32>>,
    /// the current chat's turns, and its last prompt once that has been read
    turns: usize,
    prompt: Vec<u32>,
}

impl Saved {
    /// `model_id` is the file's id (its header), and `settings` anything else a state depends on.
    pub fn new(on: bool, model_id: &[u8; 32], settings: &str) -> Saved {
        let mut h = blake3::Hasher::new();
        h.update(model_id);
        h.update(settings.as_bytes());
        let ident = u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap());
        let on = on && crate::markfile::keep() > 0;
        Saved {
            on,
            ident,
            index: if on { crate::markfile::index(ident) } else { Vec::new() },
            turns: 0,
            prompt: Vec::new(),
        }
    }

    fn key(&self, tokens: &[u32]) -> u64 {
        let mut h = blake3::Hasher::new();
        h.update(&self.ident.to_le_bytes());
        for t in tokens {
            h.update(&t.to_le_bytes());
        }
        u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap())
    }

    /// Called with a request's prompt before it is read, `keep` being the tokens of it the cache holds already.
    /// Saves the chat being left, if any, and loads the longest saved chat the prompt begins with when that saves
    /// reading. Returns the tokens of the prompt the cache now holds.
    pub fn begin(&mut self, m: &mut dyn Positional, ids: &[u32], keep: usize) -> usize {
        if !self.on {
            return keep;
        }
        let same = !self.prompt.is_empty() && ids.starts_with(&self.prompt);
        if !same {
            if self.turns >= 2 {
                self.save(m);
            }
            self.turns = 0;
        }
        self.turns += 1;
        // the rows past `keep` are about to be written over
        self.prompt.clear();
        if same {
            return keep;
        }
        let mut c: Vec<Vec<u32>> = self
            .index
            .iter()
            .filter(|t| t.len() < ids.len() && t.len() >= keep + MIN && ids.starts_with(t))
            .cloned()
            .collect();
        c.sort_by_key(|t| std::cmp::Reverse(t.len()));
        for t in c {
            let n = t.len();
            let Some((b, _)) =
                crate::markfile::load(self.key(&t), "m", self.ident, 0, Some(m.n_layer() as u32), n as u32)
            else {
                // its file is gone (only the newest few are kept)
                self.index.retain(|x| *x != t);
                continue;
            };
            let t0 = std::time::Instant::now();
            return match m.set_state_bytes(n, &b) {
                Ok(()) => {
                    eprintln!("torc-server: {n} tokens loaded from disk in {:.2} s", t0.elapsed().as_secs_f64());
                    n
                }
                Err(e) => {
                    // part of it may be in the cache, over rows the prompt shared: it is read from zero
                    eprintln!("torc-server: loading a saved state: {}; reading the prompt from zero", e.0);
                    self.index.retain(|x| *x != t);
                    0
                }
            };
        }
        keep
    }

    /// Called once the prompt has been read: its rows are in the cache, and they are what a save writes.
    pub fn read(&mut self, ids: &[u32]) {
        if self.on {
            self.prompt = ids.to_vec();
        }
    }

    /// Called when no request has come for a while (TORC_IDLE_SAVE).
    pub fn idle(&mut self, m: &mut dyn Positional) {
        let min = crate::switches::num::<usize>("TORC_IDLE_SAVE_MIN").unwrap_or(8192);
        if self.on && (self.turns >= 2 || self.prompt.len() >= min) {
            self.save(m);
        }
    }

    /// Saves the current chat's last prompt, if it is long enough and not saved already.
    fn save(&mut self, m: &mut dyn Positional) {
        let n = self.prompt.len();
        if n < MIN || self.index.contains(&self.prompt) {
            return;
        }
        let t0 = std::time::Instant::now();
        let b = m.state_bytes(n);
        // a model that no longer holds this prompt's state (Inkling, once its sliding-window ring has gone past it)
        if b.is_empty() {
            return;
        }
        let key = self.key(&self.prompt);
        let r = crate::markfile::save(key, "m", self.ident, 0, m.n_layer() as u32, n as u32, &b)
            .and_then(|_| crate::markfile::save_tokens(key, self.ident, &self.prompt));
        match r {
            Ok(()) => {
                eprintln!("torc-server: chat ({n} tokens) saved in {:.2} s", t0.elapsed().as_secs_f64());
                self.index.push(self.prompt.clone());
                // the disk keeps at most markfile::keep() files; the list is trimmed to match now and then
                if self.index.len() > 2 * crate::markfile::keep() {
                    self.index = crate::markfile::index(self.ident);
                }
            }
            Err(e) => eprintln!("torc-server: saving the chat: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model whose cache row p holds the token read at position p.
    struct Fake {
        rows: Vec<u32>,
    }

    impl Fake {
        fn read(&mut self, ids: &[u32], from: usize) {
            self.rows.truncate(from);
            self.rows.extend_from_slice(&ids[from..]);
        }
    }

    impl Positional for Fake {
        fn state_bytes(&self, npos: usize) -> Vec<u8> {
            self.rows[..npos].iter().flat_map(|t| t.to_le_bytes()).collect()
        }
        fn set_state_bytes(&mut self, npos: usize, b: &[u8]) -> format::Result<()> {
            if b.len() != npos * 4 {
                return format::err("wrong length");
            }
            self.rows = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            Ok(())
        }
        fn n_layer(&self) -> usize {
            1
        }
    }

    /// One request: what the cache keeps of the last one, begin, the read, and the tokens read.
    fn turn(s: &mut Saved, m: &mut Fake, ids: &[u32]) -> usize {
        let keep = m.rows.iter().zip(ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1);
        let from = s.begin(m, ids, keep);
        m.read(ids, from);
        s.read(ids);
        // the reply goes into the cache after the prompt
        m.rows.extend([7, 7, 7]);
        ids.len() - from
    }

    #[test]
    fn a_chat_left_is_saved_and_comes_back_from_disk() {
        crate::markfile::test_dir();
        let mut s = Saved::new(true, &[1; 32], "test");
        let mut m = Fake { rows: Vec::new() };
        let a1: Vec<u32> = (0..1000).collect();
        let a2: Vec<u32> = (0..1200).collect();
        let b1: Vec<u32> = (5000..5600).collect();
        assert_eq!(turn(&mut s, &mut m, &a1), 1000);
        // the same chat: only the new tokens
        assert_eq!(turn(&mut s, &mut m, &a2), 200);
        // another chat: the first is saved on the way out, and nothing of it is kept for this one
        assert_eq!(turn(&mut s, &mut m, &b1), 600);
        assert_eq!(s.index, vec![a2.clone()]);
        // back to the first, after a restart: loaded from disk, and only the rest is read
        let mut s = Saved::new(true, &[1; 32], "test");
        let mut m = Fake { rows: Vec::new() };
        let a3: Vec<u32> = (0..1300).collect();
        assert_eq!(turn(&mut s, &mut m, &a3), 100);
        assert_eq!(m.rows[..1300], a3[..]);
        // another model, or other settings, never reads it
        let mut s = Saved::new(true, &[1; 32], "other");
        let mut m = Fake { rows: Vec::new() };
        assert_eq!(turn(&mut s, &mut m, &a3), 1300);
    }

    #[test]
    fn short_and_one_turn_chats_are_not_saved_on_a_switch() {
        crate::markfile::test_dir();
        let mut s = Saved::new(true, &[2; 32], "test");
        let mut m = Fake { rows: Vec::new() };
        // one turn, then another chat: not saved
        turn(&mut s, &mut m, &(0..2000).collect::<Vec<u32>>());
        turn(&mut s, &mut m, &(9000..9100).collect::<Vec<u32>>());
        assert!(s.index.is_empty());
        // two turns of under 512 tokens, then another chat: not saved
        turn(&mut s, &mut m, &(9000..9200).collect::<Vec<u32>>());
        turn(&mut s, &mut m, &(100..300).collect::<Vec<u32>>());
        assert!(s.index.is_empty());
    }
}
