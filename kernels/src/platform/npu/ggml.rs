//! ggml.rs - ggml (MIT, ggml-org) as the GenieX CLI ships it, loaded at run time: the graph API and its Hexagon (HTP)
//! backend, for running whole models on the NPU (TORC_NPU=pure). Only the functions a model graph needs are used,
//! called through pointers taken from GenieX's ggml.dll / ggml-base.dll, so nothing of ggml is linked, and a machine
//! without GenieX simply reports the NPU as missing. The signatures are as ggml.h (llama.cpp b11700) declares them.

use std::ffi::{CStr, CString, c_char, c_void};

/// Opaque ggml objects.
pub type Ctx = *mut c_void;
pub type T = *mut c_void;
pub type Graph = *mut c_void;
pub type Backend = *mut c_void;
pub type Buffer = *mut c_void;
pub type BufType = *mut c_void;
pub type Alloc = *mut c_void;

/// The ggml_type values used here.
pub const F32: i32 = 0;
pub const F16: i32 = 1;
pub const Q4_0: i32 = 2;
pub const Q8_0: i32 = 8;
pub const Q6_K: i32 = 14;
pub const I32: i32 = 26;
pub const I64: i32 = 27;
/// rope mode: NeoX (rotate the two halves)
pub const ROPE_NEOX: i32 = 2;
pub const PREC_F32: i32 = 10;
/// ggml_backend_buffer_usage: weights (the HTP repacks them as they are set)
pub const USAGE_WEIGHTS: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct InitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

type LogCb = unsafe extern "C" fn(level: i32, text: *const c_char, user: *mut c_void);

macro_rules! fns {
    ($($name:ident : ($($a:ty),*) $(-> $r:ty)?;)*) => {
        #[allow(non_snake_case)]
        pub struct Fns { $(pub $name: unsafe extern "C" fn($($a),*) $(-> $r)?,)* }
        impl Fns {
            /// Every pointer from `get`, or the name of the first one that is missing.
            unsafe fn load(get: &dyn Fn(&CStr) -> *const c_void) -> Result<Fns, String> {
                Ok(Fns { $($name: {
                    let n = CString::new(stringify!($name)).unwrap();
                    let p = get(&n);
                    if p.is_null() {
                        return Err(format!("{} missing from GenieX's ggml", stringify!($name)));
                    }
                    // SAFETY: the export has the signature ggml.h declares for it
                    unsafe { std::mem::transmute::<*const c_void, unsafe extern "C" fn($($a),*) $(-> $r)?>(p) }
                },)* })
            }
        }
    };
}

fns! {
    ggml_init: (InitParams) -> Ctx;
    ggml_free: (Ctx);
    ggml_tensor_overhead: () -> usize;
    ggml_graph_overhead_custom: (usize, bool) -> usize;
    ggml_new_graph_custom: (Ctx, usize, bool) -> Graph;
    ggml_build_forward_expand: (Graph, T);
    ggml_graph_n_nodes: (Graph) -> i32;
    ggml_new_tensor_1d: (Ctx, i32, i64) -> T;
    ggml_new_tensor_2d: (Ctx, i32, i64, i64) -> T;
    ggml_new_tensor_3d: (Ctx, i32, i64, i64, i64) -> T;
    ggml_set_input: (T);
    ggml_set_output: (T);
    ggml_set_name: (T, *const c_char) -> T;
    ggml_get_rows: (Ctx, T, T) -> T;
    ggml_rms_norm: (Ctx, T, f32) -> T;
    ggml_mul: (Ctx, T, T) -> T;
    ggml_add: (Ctx, T, T) -> T;
    ggml_div: (Ctx, T, T) -> T;
    ggml_scale: (Ctx, T, f32) -> T;
    ggml_mul_mat: (Ctx, T, T) -> T;
    ggml_mul_mat_id: (Ctx, T, T, T) -> T;
    ggml_rope_ext: (Ctx, T, T, T, i32, i32, i32, f32, f32, f32, f32, f32, f32) -> T;
    ggml_flash_attn_ext: (Ctx, T, T, T, T, f32, f32, f32) -> T;
    ggml_prec_set_acc: (T, i32) -> bool;
    ggml_set_rows: (Ctx, T, T, T) -> T;
    ggml_soft_max: (Ctx, T) -> T;
    ggml_argsort_top_k: (Ctx, T, i32) -> T;
    ggml_reshape_2d: (Ctx, T, i64, i64) -> T;
    ggml_reshape_3d: (Ctx, T, i64, i64, i64) -> T;
    ggml_view_2d: (Ctx, T, i64, i64, usize, usize) -> T;
    ggml_view_3d: (Ctx, T, i64, i64, i64, usize, usize, usize) -> T;
    ggml_permute: (Ctx, T, i32, i32, i32, i32) -> T;
    ggml_cont: (Ctx, T) -> T;
    ggml_tanh: (Ctx, T) -> T;
    ggml_geglu_split: (Ctx, T, T) -> T;
    ggml_swiglu_split: (Ctx, T, T) -> T;
    ggml_sum_rows: (Ctx, T) -> T;
    ggml_repeat_4d: (Ctx, T, i64, i64, i64, i64) -> T;
    ggml_row_size: (i32, i64) -> usize;
    ggml_nbytes: (T) -> usize;
    ggml_log_set: (Option<LogCb>, *mut c_void);
    ggml_backend_dev_count: () -> usize;
    ggml_backend_dev_get: (usize) -> *mut c_void;
    ggml_backend_dev_name: (*mut c_void) -> *const c_char;
    ggml_backend_dev_init: (*mut c_void, *const c_char) -> Backend;
    ggml_backend_dev_buffer_type: (*mut c_void) -> BufType;
    ggml_backend_free: (Backend);
    ggml_backend_alloc_ctx_tensors_from_buft: (Ctx, BufType) -> Buffer;
    ggml_backend_buffer_set_usage: (Buffer, i32);
    ggml_backend_buffer_clear: (Buffer, u8);
    ggml_backend_buffer_free: (Buffer);
    ggml_backend_buffer_get_size: (Buffer) -> usize;
    ggml_backend_tensor_set: (T, *const c_void, usize, usize);
    ggml_backend_tensor_get: (T, *mut c_void, usize, usize);
    ggml_backend_graph_compute: (Backend, Graph) -> i32;
    ggml_backend_graph_compute_async: (Backend, Graph) -> i32;
    ggml_backend_synchronize: (Backend);
    ggml_backend_supports_op: (Backend, T) -> bool;
    ggml_gallocr_new: (BufType) -> Alloc;
    ggml_gallocr_alloc_graph: (Alloc, Graph) -> bool;
    ggml_gallocr_free: (Alloc);
}

#[allow(non_snake_case)]
unsafe extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *const c_void;
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
    fn SetDllDirectoryW(path: *const u16) -> i32;
    fn GetLastError() -> u32;
}

unsafe extern "C" {
    fn _putenv_s(name: *const c_char, value: *const c_char) -> i32;
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

static VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

unsafe extern "C" fn log_cb(level: i32, text: *const c_char, _: *mut c_void) {
    // ggml's log levels: 4 is an error, and the rest only print with TORC_NPU_VERBOSE
    if (level >= 4 || VERBOSE.load(std::sync::atomic::Ordering::Relaxed)) && !text.is_null() {
        // SAFETY: ggml passes a NUL-terminated message
        eprint!("{}", unsafe { CStr::from_ptr(text) }.to_string_lossy());
    }
}

/// GenieX's ggml, with an HTP0 session open.
pub struct Ggml {
    pub f: Fns,
    pub be: Backend,
    pub buft: BufType,
}

// SAFETY: one model owns it and calls it from one thread at a time
unsafe impl Send for Ggml {}
unsafe impl Sync for Ggml {}

impl Ggml {
    /// Loads ggml from TORC_NPU_DIR (by default %LOCALAPPDATA%\GenieX CLI\llama_cpp) and opens an HTP0 session.
    pub fn open() -> Result<Ggml, String> {
        Self::open_dev("HTP0")
    }

    /// As open, on device `name` (HTP1.. exist when GGML_HEXAGON_NDEV asks for more sessions on the one NPU).
    pub fn open_dev(name: &str) -> Result<Ggml, String> {
        let dir = match crate::switches::get("TORC_NPU_DIR") {
            Some(d) => d,
            None => {
                format!("{}\\GenieX CLI\\llama_cpp", std::env::var("LOCALAPPDATA").map_err(|_| "LOCALAPPDATA not set")?)
            }
        };
        VERBOSE.store(
            crate::switches::get("TORC_NPU_VERBOSE").is_some_and(|v| v != "0"),
            std::sync::atomic::Ordering::Relaxed,
        );
        // the DSP side (libggml-htp-v73.so) is found through ADSP_LIBRARY_PATH, which is read with the C runtime's
        // getenv
        if std::env::var_os("ADSP_LIBRARY_PATH").is_none() {
            // SAFETY: before any of this process's threads read the environment
            unsafe { std::env::set_var("ADSP_LIBRARY_PATH", &dir) };
            let (n, v) =
                (CString::new("ADSP_LIBRARY_PATH").unwrap(), CString::new(dir.as_str()).map_err(|e| e.to_string())?);
            // SAFETY: two valid C strings, which the C runtime copies
            unsafe { _putenv_s(n.as_ptr(), v.as_ptr()) };
        }
        let wdir = wide(&dir);
        let dll = wide(&format!("{dir}\\ggml.dll"));
        // SAFETY: valid wide strings; ggml.dll pulls in ggml-base, ggml-cpu and ggml-hexagon from its own folder
        let (m, base) = unsafe {
            SetDllDirectoryW(wdir.as_ptr());
            let m = LoadLibraryExW(dll.as_ptr(), std::ptr::null_mut(), 0x100 | 0x1000);
            let base = if m.is_null() { m } else { GetModuleHandleW(wide("ggml-base.dll").as_ptr()) };
            SetDllDirectoryW(std::ptr::null());
            (m, base)
        };
        if m.is_null() || base.is_null() {
            // SAFETY: a plain call
            return Err(format!("cannot load {dir}\\ggml.dll (error {})", unsafe { GetLastError() }));
        }
        let get = |n: &CStr| {
            // SAFETY: valid module handles and a NUL-terminated name
            let p = unsafe { GetProcAddress(m, n.as_ptr()) };
            if p.is_null() { unsafe { GetProcAddress(base, n.as_ptr()) } } else { p }
        };
        // SAFETY: the pointers are ggml's exports
        let f = unsafe { Fns::load(&get)? };
        unsafe {
            (f.ggml_log_set)(Some(log_cb), std::ptr::null_mut());
            let mut dev = std::ptr::null_mut();
            for i in 0..(f.ggml_backend_dev_count)() {
                let d = (f.ggml_backend_dev_get)(i);
                if CStr::from_ptr((f.ggml_backend_dev_name)(d)).to_bytes() == name.as_bytes() {
                    dev = d;
                    break;
                }
            }
            if dev.is_null() {
                return Err(format!("no {name} device in GenieX's ggml"));
            }
            let be = (f.ggml_backend_dev_init)(dev, std::ptr::null());
            if be.is_null() {
                return Err(format!("{name} session failed to open"));
            }
            let buft = (f.ggml_backend_dev_buffer_type)(dev);
            Ok(Ggml { f, be, buft })
        }
    }

    /// A context for tensor and graph metadata only (no data), with room for `tensors` tensors and a graph of that
    /// size.
    pub fn meta_ctx(&self, tensors: usize, graph: bool) -> Ctx {
        // SAFETY: plain calls
        unsafe {
            let mut size = (self.f.ggml_tensor_overhead)() * tensors;
            if graph {
                size += (self.f.ggml_graph_overhead_custom)(tensors, false);
            }
            (self.f.ggml_init)(InitParams { mem_size: size, mem_buffer: std::ptr::null_mut(), no_alloc: true })
        }
    }
}

impl Drop for Ggml {
    fn drop(&mut self) {
        // SAFETY: the session this struct opened; the DLLs stay loaded, since their threads may outlive it
        unsafe { (self.f.ggml_backend_free)(self.be) };
    }
}
