//! Named ±1 patterns: a stable seed from a name, the random pattern belonging to a name, text as bits, and
//! text masked by its name so it looks like random bits (what an SDM stores best).
//!
//! Patterns here are `Vec<f64>` holding -1.0 and +1.0 (the form the SETTLE interpreter keeps in its things).
//! The packed and `i8` forms live in `bits`.

use crate::rng::Rng;

/// A stable 64-bit seed from a name (FNV-1a).
pub fn seed_of(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// The random ±1 pattern (or mask) belonging to a name: `size` fair coin flips from `Rng::new(seed_of(name))`.
pub fn code(name: &str, size: usize) -> Vec<f64> {
    let mut r = Rng::new(seed_of(name));
    (0..size).map(|_| if r.unit() < 0.5 { -1.0 } else { 1.0 }).collect()
}

/// Text as ±1 bits, most significant bit of each byte first.
pub fn text_bits(t: &str) -> Vec<f64> {
    t.bytes().flat_map(|b| (0..8).rev().map(move |k| if (b >> k) & 1 == 1 { 1.0 } else { -1.0 })).collect()
}

/// Bytes as ±1 bits, most significant bit first.
pub fn bytes_bits(bytes: &[u8]) -> Vec<f64> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |k| if (b >> k) & 1 == 1 { 1.0 } else { -1.0 })).collect()
}

/// The first `len` bytes of a ±1 bit vector as text (a bit is 1 when its value is above 0). Invalid UTF-8
/// becomes the replacement character and control characters become `?`.
pub fn bits_text(bits: &[f64], len: usize) -> String {
    let bytes: Vec<u8> = bits[..len * 8]
        .chunks(8)
        .map(|c| c.iter().fold(0u8, |acc, &v| (acc << 1) | (v > 0.0) as u8))
        .collect();
    String::from_utf8_lossy(&bytes).chars().map(|c| if c.is_control() { '?' } else { c }).collect()
}

/// The pattern stored for a name: its pure code, or text bits multiplied by the name's mask, padded by the mask.
pub fn pattern(name: &str, saved: Option<&str>, size: usize) -> Vec<f64> {
    let mask = code(name, size);
    match saved {
        None => mask,
        Some(t) => {
            let mut p = mask.clone();
            for (i, b) in text_bits(t).into_iter().enumerate() {
                p[i] = b * mask[i];
            }
            p
        }
    }
}

/// Overlap of two ±1 vectors: the mean of the products, divided by the length of `b`. 1 is equal, -1 is
/// every bit flipped, about 0 is unrelated.
pub fn overlap(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / b.len() as f64
}

/// A copy of `p` with each bit flipped with probability `damage` (one uniform draw per bit, in order).
pub fn with_address_noise(p: &[f64], damage: f64, rng: &mut Rng) -> Vec<f64> {
    p.iter().map(|&v| if rng.unit() < damage { -v } else { v }).collect()
}

/// `n` fair ±1 coin flips (one uniform draw per bit, in order).
pub fn noise(n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_of_the_empty_string_and_of_a_is_the_published_value() {
        // FNV-1a 64-bit reference values
        assert_eq!(seed_of(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(seed_of("a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn text_goes_through_bits_and_back() {
        let b = text_bits("hi!");
        assert_eq!(b.len(), 24);
        // 'h' = 0x68 = 0110 1000
        assert_eq!(&b[..8], &[-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]);
        assert_eq!(bits_text(&b, 3), "hi!");
        assert_eq!(bytes_bits(b"hi!"), b);
    }

    #[test]
    fn a_masked_text_pattern_looks_random_and_unmasks() {
        let p = pattern("note", Some("aaaaaaaa"), 256);
        let mask = code("note", 256);
        let back: Vec<f64> = p.iter().zip(&mask).map(|(a, b)| a * b).collect();
        assert_eq!(bits_text(&back, 8), "aaaaaaaa");
        // negative control: the raw text bits of a repeated letter are far from random, the masked bits are not
        let raw = text_bits("aaaaaaaa");
        let raw_ones = raw.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(raw_ones, 24); // 'a' = 0110 0001: 3 ones per byte
        assert!(overlap(&p[..64], &raw).abs() < 0.5);
    }

    #[test]
    fn damage_flips_about_the_asked_fraction_and_zero_damage_flips_nothing() {
        let p = code("x", 10_000);
        let mut r = Rng::new(1);
        let d = with_address_noise(&p, 0.3, &mut r);
        let flipped = p.iter().zip(&d).filter(|(a, b)| a != b).count();
        assert!((2_850..3_150).contains(&flipped), "{}", flipped);
        assert_eq!(with_address_noise(&p, 0.0, &mut r), p);
        assert!((overlap(&d, &p) - 0.4).abs() < 0.03);
    }
}
