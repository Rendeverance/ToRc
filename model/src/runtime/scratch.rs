//! scratch.rs - reused working buffers for the prompt-block path.
//!
//! A block's buffers run to tens of MB. Allocated fresh for every layer, they would come back from the system as new
//! zero pages, and the first write to each 4 KiB page would fault. A buffer here keeps its memory between uses on its
//! thread: it grows once to the largest size asked for (zero-filled, so every element is always initialised) and is
//! then handed out again as it is. Callers write every element they read.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use kernels::{BlockQ8K, Blocks, Groups32, Q8m, Q8s};

thread_local! {
    static FREE: RefCell<HashMap<(&'static str, TypeId), Box<dyn Any>>> = RefCell::new(HashMap::new());
}

/// A buffer of `len` elements taken from the thread's store under `name` and its element type (only one in use per name
/// and type at a time), returned to the store when dropped.
pub struct Buf<T: Copy + Default + 'static> {
    v: Vec<T>,
    len: usize,
    name: &'static str,
}

/// The buffer named `name`, `len` long. Its contents are whatever the last user left (or zeros), so write before
/// reading.
pub fn buf<T: Copy + Default + 'static>(name: &'static str, len: usize) -> Buf<T> {
    let mut v: Vec<T> = FREE
        .with(|f| f.borrow_mut().remove(&(name, TypeId::of::<T>())))
        .and_then(|b| b.downcast::<Vec<T>>().ok())
        .map(|b| *b)
        .unwrap_or_default();
    if v.len() < len {
        v.resize(len, T::default());
    }
    Buf { v, len, name }
}

impl<T: Copy + Default + 'static> Deref for Buf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.v[..self.len]
    }
}

impl<T: Copy + Default + 'static> DerefMut for Buf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.v[..self.len]
    }
}

impl<T: Copy + Default + 'static> Drop for Buf<T> {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.v);
        FREE.with(|f| f.borrow_mut().insert((self.name, TypeId::of::<T>()), Box::new(v)));
    }
}

/// A quantised vector's two arrays (kernels::Q8), each a buffer as above under the same name.
pub struct Q8Buf {
    pub k: Buf<BlockQ8K>,
    pub g: Buf<Groups32>,
}

/// `n` blocks of each array, named `name`; write before reading, as with buf.
pub fn q8(name: &'static str, n: usize) -> Q8Buf {
    Q8Buf { k: buf(name, n), g: buf(name, n) }
}

impl Q8Buf {
    pub fn len(&self) -> usize {
        self.k.len()
    }
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }
    pub fn view(&self) -> Q8s<'_> {
        Q8s { k: &self.k, g: &self.g }
    }
    pub fn view_mut(&mut self) -> Q8m<'_> {
        Q8m { k: &mut self.k, g: &mut self.g }
    }
    pub fn sub(&self, r: impl Blocks) -> Q8s<'_> {
        self.view().sub(r)
    }
    pub fn sub_mut(&mut self, r: impl Blocks) -> Q8m<'_> {
        Q8m { k: &mut self.k[r.clone()], g: &mut self.g[r] }
    }
}
