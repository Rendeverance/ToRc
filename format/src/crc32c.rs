//! crc32c.rs - CRC-32C (Castagnoli, reflected polynomial 0x82F63B78), the checksum of every tensor and expert pack.
//! Both CPU families have an instruction for it (x86 SSE4.2 `crc32`, ARMv8 `crc32c`), used when the CPU has it; the
//! table is the fallback and the reference the tests compare against. `Crc32c` takes the data in pieces, for a tensor
//! too big to hold at once.

const POLY: u32 = 0x82F6_3B78;

const fn table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

static TABLE: [u32; 256] = table();

/// The running state (before the final inversion) after data, one byte at a time through the table.
fn update_table(mut c: u32, data: &[u8]) -> u32 {
    for &b in data {
        c = TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn update_x86(c: u32, data: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};
    let mut c = c as u64;
    let mut words = data.chunks_exact(8);
    for w in &mut words {
        c = _mm_crc32_u64(c, u64::from_le_bytes(w.try_into().unwrap()));
    }
    let mut c = c as u32;
    for &b in words.remainder() {
        c = _mm_crc32_u8(c, b);
    }
    c
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn update_arm(mut c: u32, data: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    let mut words = data.chunks_exact(8);
    for w in &mut words {
        c = __crc32cd(c, u64::from_le_bytes(w.try_into().unwrap()));
    }
    for &b in words.remainder() {
        c = __crc32cb(c, b);
    }
    c
}

fn update(c: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.2") {
        // SAFETY: the CPU has SSE4.2, checked above
        return unsafe { update_x86(c, data) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        // SAFETY: the CPU has the CRC instructions, checked above
        return unsafe { update_arm(c, data) };
    }
    update_table(c, data)
}

/// A CRC-32C over data given in pieces.
#[derive(Clone, Copy, Debug)]
pub struct Crc32c(u32);

impl Default for Crc32c {
    fn default() -> Self {
        Crc32c(!0)
    }
}

impl Crc32c {
    pub fn update(&mut self, data: &[u8]) {
        self.0 = update(self.0, data);
    }
    pub fn finish(self) -> u32 {
        !self.0
    }
}

/// The CRC-32C of data.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut c = Crc32c::default();
    c.update(data);
    c.finish()
}

/// The CRC-32C of data through the table only: the reference the tests check crc32c against.
#[cfg(test)]
pub fn crc32c_table(data: &[u8]) -> u32 {
    !update_table(!0, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // the check value of CRC-32C is that of "123456789"; the 32-byte values are from RFC 3720 (iSCSI), B.4
        assert_eq!(crc32c_table(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c_table(&[0u8; 32]), 0x8A91_36AA);
        assert_eq!(crc32c_table(&[0xFFu8; 32]), 0x62A8_AB43);
        assert_eq!(crc32c_table(b""), 0);
    }

    #[test]
    fn instruction_and_pieces_match_table() {
        let data: Vec<u8> = (0..10_007u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for n in [0, 1, 7, 8, 9, 63, 64, 65, 4096, 10_007] {
            assert_eq!(crc32c(&data[..n]), crc32c_table(&data[..n]), "{n} bytes");
        }
        let mut c = Crc32c::default();
        for piece in data.chunks(333) {
            c.update(piece);
        }
        assert_eq!(c.finish(), crc32c_table(&data));
    }
}
