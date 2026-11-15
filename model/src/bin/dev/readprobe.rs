//! readprobe <file> [reads] [bytes]: how long an overlapped, unbuffered ReadFile takes to *return* on Windows. The
//! expert cache issues its misses this way and then computes, so if the call itself takes as long as the read, there is
//! nothing to overlap. It reads at random 4 KiB-aligned offsets, one read in flight at a time, and reports the call's
//! time and the wait's, averaged. Windows only.

#[cfg(windows)]
fn main() {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    #[repr(C)]
    struct Overlapped {
        internal: usize,
        internal_high: usize,
        offset: u32,
        offset_high: u32,
        event: *mut c_void,
    }
    unsafe extern "system" {
        fn CreateFileW(n: *const u16, a: u32, s: u32, sa: *mut c_void, d: u32, f: u32, t: *mut c_void) -> *mut c_void;
        fn ReadFile(h: *mut c_void, b: *mut c_void, n: u32, r: *mut u32, o: *mut Overlapped) -> i32;
        fn GetOverlappedResult(h: *mut c_void, o: *mut Overlapped, n: *mut u32, w: i32) -> i32;
        fn CreateEventW(sa: *mut c_void, m: i32, i: i32, n: *const u16) -> *mut c_void;
        fn GetLastError() -> u32;
        fn VirtualAlloc(a: *mut c_void, s: usize, t: u32, p: u32) -> *mut c_void;
    }
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let reads: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(200);
    let bytes: u32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1_572_864);
    let len = std::fs::metadata(path).unwrap().len();
    let w: Vec<u16> = std::ffi::OsStr::new(path).encode_wide().chain([0]).collect();
    // GENERIC_READ, share read|write|delete, OPEN_EXISTING, NO_BUFFERING | OVERLAPPED
    let h = unsafe {
        CreateFileW(
            w.as_ptr(),
            0x8000_0000,
            7,
            std::ptr::null_mut(),
            3,
            0x2000_0000 | 0x4000_0000,
            std::ptr::null_mut(),
        )
    };
    assert!(h as isize != -1, "open failed");
    let buf = unsafe { VirtualAlloc(std::ptr::null_mut(), bytes as usize, 0x3000, 4) };
    let ev = unsafe { CreateEventW(std::ptr::null_mut(), 1, 0, std::ptr::null()) };
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let (mut call, mut wait, mut inline) = (0f64, 0f64, 0usize);
    for _ in 0..reads {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let at = (rng % (len - bytes as u64)) & !4095;
        let mut o =
            Overlapped { internal: 0, internal_high: 0, offset: at as u32, offset_high: (at >> 32) as u32, event: ev };
        let t = std::time::Instant::now();
        let ok = unsafe { ReadFile(h, buf, bytes, std::ptr::null_mut(), &mut o) };
        let c = t.elapsed().as_secs_f64();
        if ok != 0 {
            inline += 1;
        } else {
            assert_eq!(unsafe { GetLastError() }, 997, "ReadFile failed"); // ERROR_IO_PENDING
        }
        let t2 = std::time::Instant::now();
        let mut got = 0u32;
        assert!(unsafe { GetOverlappedResult(h, &mut o, &mut got, 1) } != 0);
        wait += t2.elapsed().as_secs_f64();
        call += c;
    }
    println!(
        "{reads} reads of {bytes} bytes: ReadFile call {:.1} us, then wait {:.1} us (mean); completed inline {inline}",
        call * 1e6 / reads as f64,
        wait * 1e6 / reads as f64
    );
}

#[cfg(not(windows))]
fn main() {
    eprintln!("readprobe: Windows only");
}
