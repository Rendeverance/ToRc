//! split.rs - the Snapdragon NPU for chosen products of a model that the CPU runs. The weights are uploaded once; then
//! each block's products, and optionally a gated feed-forward, run as one ggml graph on GenieX's HTP backend
//! ([`super::ggml`]). The graph is started while the CPU carries on, and its outputs are written when the caller asks
//! for them. This is the device layer only: it holds no model state.
//!
//! Settings: TORC_NPU (> 0 turns it on), TORC_NPU_DIR (where GenieX's ggml is), TORC_NPU_MIN_N (the fewest tokens a
//! block needs, 32), TORC_NPU_Q8 (Q4_0 weights given to the NPU as Q8_0) and TORC_NPU_VERBOSE.
//!
//! The NPU's products are not bit-identical to the CPU's (f32 activations, and the HTP's own sums), so a tier built on
//! it is checked against the CPU path by tolerance, never by bytes.

use super::ggml::{self, Alloc, Buffer, Ctx, Ggml, T};
use std::sync::mpsc;
use std::time::Instant;

/// One product of a group: weight id w, with the output rows written at y[t * y_stride ..] for token t (a y_stride of 0
/// means rows).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NpuMm {
    pub w: i32,
    pub y: *mut f32,
    pub y_stride: usize,
}

/// A group: n f32 input rows of cols at x, its products mm, and optionally a gated feed-forward (gate, up and down ids)
/// writing into ffn_y ([n][rows of down]; null for none).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NpuGroup {
    pub x: *const f32,
    pub cols: usize,
    pub n: usize,
    pub mm: *const NpuMm,
    pub n_mm: usize,
    pub ffn: [i32; 3],
    pub ffn_y: *mut f32,
}

#[allow(non_snake_case)]
unsafe extern "system" {
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn SetThreadPriority(h: *mut std::ffi::c_void, priority: i32) -> i32;
}
const THREAD_PRIORITY_HIGHEST: i32 = 2;

/// The nodes a graph may hold (per group: an input, its products, and four for a feed-forward).
const GRAPH: usize = 400;

/// Whether TORC_NPU asks for the NPU (> 0).
pub fn configure() -> bool {
    crate::switches::num::<i32>("TORC_NPU").unwrap_or(0) > 0
}

fn env_on(name: &str) -> bool {
    crate::switches::num::<i64>(name).unwrap_or(0) > 0
}

/// A weight matrix as recorded, and then as uploaded.
struct Weight {
    ty: i32,
    /// a Q4_0 matrix given to the NPU as Q8_0 (TORC_NPU_Q8)
    from_q4: bool,
    src: *const u8,
    rows: usize,
    cols: usize,
    t: T,
    ok: bool,
}

/// An output of the graph in flight, and where it goes.
struct Out {
    t: T,
    y: *mut f32,
    rows: usize,
    n: usize,
    stride: usize,
}

/// The backend pointer and its synchronize function, for the helper thread.
struct Syncer(ggml::Backend, unsafe extern "C" fn(ggml::Backend));
// SAFETY: ggml_backend_synchronize may be called from a thread other than the one that queued the graph; only this
// helper calls it while a graph is in flight, and the caller waits for it before doing anything else
unsafe impl Send for Syncer {}

pub struct Npu {
    g: Ggml,
    min_tokens: usize,
    q8: bool,
    verbose: bool,
    w: Vec<Weight>,
    wctx: Ctx,
    wbuf: Buffer,
    galloc: Alloc,
    /// the graph in flight: its context and outputs
    gctx: Ctx,
    out: Vec<Out>,
    busy: bool,
    stage: Vec<f32>,
    /// the feed-forward's activation: GEGLU (gemma4) once set_geglu has found that the backend takes it, else SwiGLU
    geglu: bool,
    /// the helper that flushes a started graph to the DSP and waits for it, so the caller's thread is free in the
    /// meantime
    go: Option<mpsc::Sender<()>>,
    done: mpsc::Receiver<()>,
    helper: Option<std::thread::JoinHandle<()>>,
    wait_s: f64,
    graphs: u64,
}

// SAFETY: one engine owns it and drives it from one thread (load, begin, finish: model/src/tiers/npu.rs), and &mut self
// gives exclusion. The raw pointers are the caller's buffers, valid by the contracts of begin and weight.
unsafe impl Send for Npu {}

impl Npu {
    /// Ok(None) when TORC_NPU is not set above 0; Err (with the reason) when it was asked for but cannot be used.
    pub fn create() -> Result<Option<Npu>, String> {
        if !configure() {
            return Ok(None);
        }
        let g = Ggml::open()?;
        let min_tokens = crate::switches::num("TORC_NPU_MIN_N").filter(|&n: &usize| n > 0).unwrap_or(32);
        // SAFETY: plain calls on the open session
        let galloc = unsafe { (g.f.ggml_gallocr_new)(g.buft) };
        if galloc.is_null() {
            return Err("NPU setup failed".into());
        }
        let (go, go_rx) = mpsc::channel::<()>();
        let (done_tx, done) = mpsc::channel::<()>();
        let s = Syncer(g.be, g.f.ggml_backend_synchronize);
        let helper = std::thread::Builder::new()
            .name("torc-npu".into())
            .spawn(move || {
                let s = s;
                // above the compute pool's threads: those are pinned one per core and spin for a while after each job,
                // so a flush waiting for one of their cores at equal priority waits for a time slice to end
                // (TORC_NPU_HELPER_PRIO=1; opt-in)
                if crate::switches::on("TORC_NPU_HELPER_PRIO") {
                    // SAFETY: plain Win32 calls on this thread
                    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) };
                }
                while go_rx.recv().is_ok() {
                    // SAFETY: a graph was started on this backend, and nothing else touches it until it is done
                    unsafe { (s.1)(s.0) };
                    if done_tx.send(()).is_err() {
                        break;
                    }
                }
            })
            .map_err(|_| "no NPU helper thread".to_string())?;
        Ok(Some(Npu {
            g,
            min_tokens,
            q8: env_on("TORC_NPU_Q8"),
            verbose: env_on("TORC_NPU_VERBOSE"),
            w: Vec::new(),
            wctx: std::ptr::null_mut(),
            wbuf: std::ptr::null_mut(),
            galloc,
            gctx: std::ptr::null_mut(),
            out: Vec::new(),
            busy: false,
            stage: Vec::new(),
            geglu: false,
            go: Some(go),
            done,
            helper: Some(helper),
            wait_s: 0.0,
            graphs: 0,
        }))
    }

    /// The fewest tokens a block needs to use the NPU (TORC_NPU_MIN_N, 32).
    pub fn min_tokens(&self) -> usize {
        self.min_tokens
    }

    /// Records a weight (ggml type 0 F32, 1 F16, 2 Q4_0, 8 Q8_0 or 14 Q6_K; rows = outputs) and returns its id, or
    /// None.
    ///
    /// # Safety
    /// w must hold the matrix and stay valid and unchanged until [`Npu::upload`] returns.
    pub unsafe fn weight(&mut self, ty: u32, w: *const u8, rows: usize, cols: usize) -> Option<i32> {
        let ty = ty as i32;
        if !self.wctx.is_null()
            || w.is_null()
            || ![ggml::F32, ggml::F16, ggml::Q4_0, ggml::Q8_0, ggml::Q6_K].contains(&ty)
        {
            return None;
        }
        // TORC_NPU_Q8: Q4_0 goes in as Q8_0 with the same numbers (a Q4_0 weight is d (q - 8) with q in 0..15, and Q8_0
        // holds d and q - 8 exactly)
        let from_q4 = ty == ggml::Q4_0 && self.q8 && cols % 32 == 0;
        self.w.push(Weight {
            ty: if from_q4 { ggml::Q8_0 } else { ty },
            from_q4,
            src: w,
            rows,
            cols,
            t: std::ptr::null_mut(),
            ok: false,
        });
        Some(self.w.len() as i32 - 1)
    }

    /// Whether the backend takes a product of weight i by n vectors.
    fn supports(&self, i: usize, n: usize) -> bool {
        let f = &self.g.f;
        let c = self.g.meta_ctx(4, false);
        if c.is_null() {
            return false;
        }
        // SAFETY: a metadata context made here, and the uploaded weight's tensor
        unsafe {
            let x = (f.ggml_new_tensor_2d)(c, ggml::F32, self.w[i].cols as i64, n as i64);
            let y = (f.ggml_mul_mat)(c, self.w[i].t, x);
            let ok = (f.ggml_backend_supports_op)(self.g.be, y);
            (f.ggml_free)(c);
            ok
        }
    }

    /// Uploads every recorded weight, once, and returns (bytes, seconds).
    pub fn upload(&mut self) -> Result<(usize, f64), ()> {
        let t0 = Instant::now();
        if self.w.is_empty() || !self.wctx.is_null() {
            return Err(());
        }
        let f = &self.g.f;
        self.wctx = self.g.meta_ctx(self.w.len() + 1, false);
        if self.wctx.is_null() {
            return Err(());
        }
        // SAFETY: tensors of this context; the sources hold their matrices (by weight's contract)
        unsafe {
            for w in self.w.iter_mut() {
                w.t = (f.ggml_new_tensor_2d)(self.wctx, w.ty, w.cols as i64, w.rows as i64);
            }
            self.wbuf = (f.ggml_backend_alloc_ctx_tensors_from_buft)(self.wctx, self.g.buft);
            if self.wbuf.is_null() {
                return Err(());
            }
            // before the copies, because the HTP repacks weights as they go in
            (f.ggml_backend_buffer_set_usage)(self.wbuf, ggml::USAGE_WEIGHTS);
        }
        let mut bytes = 0;
        for i in 0..self.w.len() {
            let ok = [64, 256, 512].iter().all(|&n| self.supports(i, n));
            self.w[i].ok = ok;
            let w = &self.w[i];
            if !ok {
                if self.verbose {
                    eprintln!("npu: weight {i} (type {}, {}x{}) not supported, stays on the CPU", w.ty, w.rows, w.cols);
                }
                continue;
            }
            // SAFETY: the tensor is in the weights' buffer, and src holds nbytes of its type (or the Q4_0 it comes
            // from)
            unsafe {
                let nb = (f.ggml_nbytes)(w.t);
                if w.from_q4 {
                    let blocks = w.rows * (w.cols / 32);
                    let q = std::slice::from_raw_parts(w.src, blocks * 18);
                    let mut tmp = vec![0u8; blocks * 34];
                    for (o, i) in tmp.chunks_exact_mut(34).zip(q.chunks_exact(18)) {
                        o[..2].copy_from_slice(&i[..2]);
                        for k in 0..16 {
                            o[2 + k] = ((i[2 + k] & 15) as i8 - 8) as u8;
                            o[18 + k] = ((i[2 + k] >> 4) as i8 - 8) as u8;
                        }
                    }
                    (f.ggml_backend_tensor_set)(w.t, tmp.as_ptr().cast(), 0, nb);
                } else {
                    (f.ggml_backend_tensor_set)(w.t, w.src.cast(), 0, nb);
                }
                bytes += nb;
            }
        }
        Ok((bytes, t0.elapsed().as_secs_f64()))
    }

    /// Whether weight id made it onto the NPU (the backend supports its type and shape).
    pub fn ok(&self, id: i32) -> bool {
        id >= 0 && (id as usize) < self.w.len() && self.w[id as usize].ok
    }

    /// The feed-forward groups' activation: GEGLU (gemma4) when on, SwiGLU when off. Returns false when the backend
    /// does not take GEGLU, and SwiGLU stays.
    pub fn set_geglu(&mut self, on: bool) -> bool {
        if !on {
            self.geglu = false;
            return true;
        }
        let f = &self.g.f;
        let c = self.g.meta_ctx(4, false);
        if c.is_null() {
            return false;
        }
        // SAFETY: a metadata context made here
        let ok = unsafe {
            let a = (f.ggml_new_tensor_2d)(c, ggml::F32, 768, 64);
            let b = (f.ggml_new_tensor_2d)(c, ggml::F32, 768, 64);
            let ok = (f.ggml_backend_supports_op)(self.g.be, (f.ggml_geglu_split)(c, a, b));
            (f.ggml_free)(c);
            ok
        };
        self.geglu = ok;
        ok
    }

    /// Starts the groups as one graph, so the NPU works while the caller does; [`Npu::finish`] waits and writes the
    /// outputs. Returns false, with nothing started, when the graph cannot be made (the caller then computes on the
    /// CPU).
    ///
    /// # Safety
    /// The inputs may be freed after this returns, but every output pointer must stay valid, and untouched by anything
    /// else, until finish returns.
    pub unsafe fn begin(&mut self, g: &[NpuGroup]) -> bool {
        if self.busy || g.is_empty() {
            return false;
        }
        let nodes: usize = g.iter().map(|x| 1 + x.n_mm + if x.ffn_y.is_null() { 0 } else { 4 }).sum();
        if nodes > GRAPH {
            return false;
        }
        // SAFETY: the caller's groups (by this function's contract)
        let mms = |x: &NpuGroup| unsafe { std::slice::from_raw_parts(x.mm, x.n_mm) };
        for x in g {
            if mms(x).iter().any(|m| !self.ok(m.w)) || !x.ffn_y.is_null() && x.ffn.iter().any(|&id| !self.ok(id)) {
                return false;
            }
        }
        let f = &self.g.f;
        let c = self.g.meta_ctx(GRAPH, true);
        if c.is_null() {
            return false;
        }
        self.out.clear();
        let mut xs = Vec::with_capacity(g.len());
        // SAFETY: tensors of this context over the uploaded weights; the inputs hold n x cols floats
        unsafe {
            let gf = (f.ggml_new_graph_custom)(c, GRAPH, false);
            for x in g {
                let xt = (f.ggml_new_tensor_2d)(c, ggml::F32, x.cols as i64, x.n as i64);
                (f.ggml_set_input)(xt);
                xs.push(xt);
                for m in mms(x) {
                    let w = &self.w[m.w as usize];
                    let y = (f.ggml_mul_mat)(c, w.t, xt);
                    (f.ggml_set_output)(y);
                    (f.ggml_build_forward_expand)(gf, y);
                    self.out.push(Out {
                        t: y,
                        y: m.y,
                        rows: w.rows,
                        n: x.n,
                        stride: if m.y_stride == 0 { w.rows } else { m.y_stride },
                    });
                }
                if !x.ffn_y.is_null() {
                    let gate = (f.ggml_mul_mat)(c, self.w[x.ffn[0] as usize].t, xt);
                    let up = (f.ggml_mul_mat)(c, self.w[x.ffn[1] as usize].t, xt);
                    let h =
                        if self.geglu { (f.ggml_geglu_split)(c, gate, up) } else { (f.ggml_swiglu_split)(c, gate, up) };
                    let d = &self.w[x.ffn[2] as usize];
                    let y = (f.ggml_mul_mat)(c, d.t, h);
                    (f.ggml_set_output)(y);
                    (f.ggml_build_forward_expand)(gf, y);
                    self.out.push(Out { t: y, y: x.ffn_y, rows: d.rows, n: x.n, stride: d.rows });
                }
            }
            if !(f.ggml_gallocr_alloc_graph)(self.galloc, gf) {
                (f.ggml_free)(c);
                return false;
            }
            for (x, &xt) in g.iter().zip(&xs) {
                (f.ggml_backend_tensor_set)(xt, x.x.cast(), 0, (f.ggml_nbytes)(xt));
            }
            if (f.ggml_backend_graph_compute_async)(self.g.be, gf) != 0 {
                (f.ggml_backend_synchronize)(self.g.be);
                (f.ggml_free)(c);
                return false;
            }
        }
        self.gctx = c;
        self.busy = true;
        self.graphs += 1;
        // the backend only queues the graph; the helper flushes it to the DSP and waits
        self.go.as_ref().is_some_and(|go| go.send(()).is_ok())
    }

    /// Waits for the graph begin started and writes its outputs. Returns false when nothing was started (or a copy
    /// failed).
    pub fn finish(&mut self) -> bool {
        if !self.busy {
            return false;
        }
        let t0 = Instant::now();
        let synced = self.done.recv().is_ok();
        self.wait_s += t0.elapsed().as_secs_f64();
        let f = &self.g.f;
        for o in &self.out {
            // SAFETY: the outputs are the graph's, and computed (the helper synchronised); y holds n rows at stride (by
            // begin's contract)
            unsafe {
                if o.stride == o.rows {
                    (f.ggml_backend_tensor_get)(o.t, o.y.cast(), 0, o.rows * o.n * 4);
                    continue;
                }
                let need = o.rows * o.n;
                if self.stage.len() < need {
                    self.stage.resize(need, 0.0);
                }
                (f.ggml_backend_tensor_get)(o.t, self.stage.as_mut_ptr().cast(), 0, need * 4);
                for v in 0..o.n {
                    std::ptr::copy_nonoverlapping(self.stage.as_ptr().add(v * o.rows), o.y.add(v * o.stride), o.rows);
                }
            }
        }
        // SAFETY: the graph's context, no longer in flight
        unsafe { (f.ggml_free)(self.gctx) };
        self.gctx = std::ptr::null_mut();
        self.busy = false;
        synced
    }

    /// Runs the groups and waits for them.
    ///
    /// # Safety
    /// Every pointer in the groups must be valid for its use (inputs read, outputs written) until this returns.
    pub unsafe fn run(&mut self, g: &[NpuGroup]) -> bool {
        let started = unsafe { self.begin(g) };
        started && self.finish()
    }

    /// (seconds waited for the NPU, graphs run).
    pub fn stats(&self) -> (f64, u64) {
        (self.wait_s, self.graphs)
    }
}

impl Drop for Npu {
    fn drop(&mut self) {
        if self.busy {
            self.finish();
        }
        if self.verbose {
            eprintln!("npu: {} graphs, {:.3} s waited for", self.graphs, self.wait_s);
        }
        // the helper ends when its channel closes
        drop(self.go.take());
        if let Some(h) = self.helper.take() {
            let _ = h.join();
        }
        let f = &self.g.f;
        // SAFETY: what this object made, freed once; the session itself goes with self.g
        unsafe {
            (f.ggml_gallocr_free)(self.galloc);
            if !self.wbuf.is_null() {
                (f.ggml_backend_buffer_free)(self.wbuf);
            }
            if !self.wctx.is_null() {
                (f.ggml_free)(self.wctx);
            }
        }
    }
}
