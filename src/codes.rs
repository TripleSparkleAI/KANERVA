//! Named ±1 patterns: a stable seed from a name, the random pattern belonging to a name, text as bits, and
//! text masked by its name so it looks like random bits (what an SDM stores best).
//!
//! Patterns here are `Vec<f64>` holding -1.0 and +1.0 (the form the SETTLE interpreter keeps in its things).
//! The packed and `i8` forms live in `bits`.

use crate::rng::Rng;

/// A stable 64-bit seed from a name (FNV-1a).
///
/// ```
/// use kanerva::codes::seed_of;
/// assert_eq!(seed_of(""), 0xcbf2_9ce4_8422_2325); // the FNV-1a offset basis
/// assert_eq!(seed_of("a"), 0xaf63_dc4c_8601_ec8c); // the FNV-1a reference value for "a"
/// ```
pub fn seed_of(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// The random ±1 pattern (or mask) belonging to a name: `size` fair coin flips from `Rng::new(seed_of(name))`.
///
/// ```
/// use kanerva::codes::{code, overlap};
/// let cat = code("cat", 1024);
/// assert_eq!(cat, code("cat", 1024));
/// assert!(cat.iter().all(|&v| v == 1.0 || v == -1.0));
/// assert!(overlap(&cat, &code("owl", 1024)).abs() < 0.1); // two names give unrelated patterns
/// ```
pub fn code(name: &str, size: usize) -> Vec<f64> {
    let mut r = Rng::new(seed_of(name));
    (0..size).map(|_| if r.unit() < 0.5 { -1.0 } else { 1.0 }).collect()
}

/// Text as ±1 bits, most significant bit of each byte first.
///
/// ```
/// use kanerva::codes::text_bits;
/// // 'A' is 0x41 = 0100_0001, most significant bit first
/// assert_eq!(text_bits("A"), vec![-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]);
/// ```
pub fn text_bits(t: &str) -> Vec<f64> {
    t.bytes().flat_map(|b| (0..8).rev().map(move |k| if (b >> k) & 1 == 1 { 1.0 } else { -1.0 })).collect()
}

/// Bytes as ±1 bits, most significant bit first.
///
/// ```
/// use kanerva::codes::bytes_bits;
/// assert_eq!(bytes_bits(&[0x80]), vec![1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
/// ```
pub fn bytes_bits(bytes: &[u8]) -> Vec<f64> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |k| if (b >> k) & 1 == 1 { 1.0 } else { -1.0 })).collect()
}

/// The first `len` bytes of a ±1 bit vector as text (a bit is 1 when its value is above 0). Invalid UTF-8
/// becomes the replacement character and control characters become `?`.
///
/// ```
/// use kanerva::codes::{bits_text, bytes_bits, text_bits};
/// assert_eq!(bits_text(&text_bits("hello"), 5), "hello");
/// assert_eq!(bits_text(&bytes_bits(&[7]), 1), "?"); // a control character reads as ?
/// ```
pub fn bits_text(bits: &[f64], len: usize) -> String {
    let bytes: Vec<u8> = bits[..len * 8]
        .chunks(8)
        .map(|c| c.iter().fold(0u8, |acc, &v| (acc << 1) | (v > 0.0) as u8))
        .collect();
    String::from_utf8_lossy(&bytes).chars().map(|c| if c.is_control() { '?' } else { c }).collect()
}

/// The pattern stored for a name: its pure code, or text bits multiplied by the name's mask, padded by the mask.
///
/// ```
/// use kanerva::codes::{code, pattern, text_bits};
/// assert_eq!(pattern("cat", None, 256), code("cat", 256));
/// let p = pattern("note", Some("hi"), 256);
/// let mask = code("note", 256);
/// let bits = text_bits("hi");
/// assert!((0..16).all(|i| p[i] * mask[i] == bits[i])); // the text, unmasked by the name
/// assert_eq!(&p[16..], &mask[16..]); // the rest is the mask
/// ```
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
///
/// ```
/// use kanerva::codes::overlap;
/// assert_eq!(overlap(&[1.0, 1.0], &[1.0, 1.0]), 1.0);
/// assert_eq!(overlap(&[1.0, 1.0], &[1.0, -1.0]), 0.0);
/// ```
pub fn overlap(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / b.len() as f64
}

/// A copy of `p` with address-noise: each bit flipped with probability `noise` (one uniform draw per bit, in order).
///
/// ```
/// use kanerva::{codes::{code, with_address_noise}, Rng};
/// let p = code("cat", 64);
/// let mut rng = Rng::new(1);
/// assert_eq!(with_address_noise(&p, 0.0, &mut rng), p);
/// let flipped: Vec<f64> = p.iter().map(|v| -v).collect();
/// assert_eq!(with_address_noise(&p, 1.0, &mut rng), flipped);
/// ```
pub fn with_address_noise(p: &[f64], noise: f64, rng: &mut Rng) -> Vec<f64> {
    p.iter().map(|&v| if rng.unit() < noise { -v } else { v }).collect()
}

/// `n` fair ±1 coin flips (one uniform draw per bit, in order).
///
/// ```
/// use kanerva::{codes::noise, Rng};
/// let z = noise(128, &mut Rng::new(5));
/// assert_eq!(z.len(), 128);
/// assert!(z.iter().all(|&v| v == 1.0 || v == -1.0));
/// ```
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
