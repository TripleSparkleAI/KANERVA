//! Keyed notes: a key names a turn of the cube {-1,+1}^n, a sign flip on every bit (a mask) plus a shuffle of
//! which position holds which bit (a permutation). Sign flips and shuffles are exactly the rotations of the
//! cube, so a turn keeps every Hamming distance.
//!
//! The stored pattern is the turned payload `[length byte][text][padding of +1s]`. With the key you know the
//! padding, so the key both FINDS the memory (the turned all-+1 payload is a read-address that agrees with the pattern
//! on every padding bit) and READS it (turn back, check the padding, decode). A store never needs the text.
//!
//! This is NOT cryptography: the key is hashed to 64 bits by FNV-1a, the mask and the shuffle come from a
//! xorshift generator, and a guessed key can be checked offline against the padding.

use crate::codes::{bits_text, bytes_bits, code, seed_of};
use crate::rng::Rng;

/// A turn of the cube: first flip signs by `mask`, then move bit k to position `perm[k]`.
pub struct KeyTurn {
    pub mask: Vec<f64>,
    pub perm: Vec<usize>,
}

impl KeyTurn {
    /// Payload order to stored order.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        for k in 0..x.len() {
            out[self.perm[k]] = x[k] * self.mask[k];
        }
        out
    }
    /// Stored order back to payload order.
    pub fn undo(&self, s: &[f64]) -> Vec<f64> {
        (0..s.len()).map(|k| s[self.perm[k]] * self.mask[k]).collect()
    }
}

/// The turn a key names. The mask is `code("mask:<key>")`; the permutation is a Fisher-Yates shuffle driven
/// by `seed_of("turn:<key>")`.
pub fn key_turn(key: &str, size: usize) -> KeyTurn {
    let mask = code(&format!("mask:{}", key), size);
    let mut r = Rng::new(seed_of(&format!("turn:{}", key)));
    let mut perm: Vec<usize> = (0..size).collect();
    for k in (1..size).rev() {
        let j = r.below(k + 1);
        perm.swap(k, j);
    }
    KeyTurn { mask, perm }
}

/// The largest text a keyed pattern of `size` bits holds (one length byte, then the text, at most 255 bytes).
pub fn keyed_capacity(size: usize) -> usize {
    (size / 8).saturating_sub(1).min(255)
}

/// `[length byte][text bits][+1 padding]` in payload order, before the turn.
pub fn keyed_payload(t: &str, size: usize) -> Vec<f64> {
    let mut p = vec![1.0; size];
    let mut bytes = vec![t.len() as u8];
    bytes.extend(t.bytes());
    for (i, b) in bytes_bits(&bytes).into_iter().enumerate() {
        p[i] = b;
    }
    p
}

/// The pattern a keyed save stores: the payload, turned by the key.
pub fn keyed_pattern(key: &str, t: &str, size: usize) -> Vec<f64> {
    key_turn(key, size).apply(&keyed_payload(t, size))
}

/// What the key-holder reads from: the turned all-+1 payload (right on every padding bit).
pub fn keyed_read_address(key: &str, size: usize) -> Vec<f64> {
    key_turn(key, size).apply(&vec![1.0; size])
}

/// Turn a recalled state back with a key and read the text, or None. For each sign (a mirror image has every
/// bit flipped), decode the length byte, then check the padding that must follow the text: at least 32 padding
/// bits must exist and at least 85% of them must read +1. A wrong key turns a state into noise, so its padding
/// reads +1 about half the time and fails. A note that fills the memory has no padding to check, so it is
/// refused: the key cannot tell it from noise.
pub fn keyed_read(key: &str, s: &[f64]) -> Option<String> {
    let size = s.len();
    let b = key_turn(key, size).undo(s);
    for sg in [1.0, -1.0] {
        let len = b[..8].iter().fold(0usize, |acc, &v| (acc << 1) | (sg * v > 0.0) as usize);
        if len > keyed_capacity(size) {
            continue;
        }
        let pad = &b[8 * (len + 1)..];
        if pad.len() < 32 || pad.iter().filter(|&&v| sg * v > 0.0).count() * 20 < pad.len() * 17 {
            continue;
        }
        let bits: Vec<f64> = b[8..].iter().map(|v| sg * v).collect();
        return Some(bits_text(&bits, len));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::overlap;

    #[test]
    fn capacity_of_small_sizes_by_hand() {
        assert_eq!(keyed_capacity(8), 0);
        assert_eq!(keyed_capacity(256), 31);
        assert_eq!(keyed_capacity(4096), 255);
    }

    #[test]
    fn a_turn_keeps_every_overlap_and_undoes() {
        let t = key_turn("k", 128);
        let (x, y) = (code("x", 128), code("y", 128));
        assert_eq!(t.undo(&t.apply(&x)), x);
        assert_eq!(overlap(&t.apply(&x), &t.apply(&y)), overlap(&x, &y));
    }

    #[test]
    fn the_right_key_reads_the_note_and_the_mirror_too() {
        let p = keyed_pattern("blue heron", "under the mat", 256);
        assert_eq!(keyed_read("blue heron", &p).as_deref(), Some("under the mat"));
        let mirror: Vec<f64> = p.iter().map(|v| -v).collect();
        assert_eq!(keyed_read("blue heron", &mirror).as_deref(), Some("under the mat"));
        // the read-address agrees with the pattern on every padding bit: (256 - 8 * 14) / 256 of the bits at least
        assert!(overlap(&keyed_read_address("blue heron", 256), &p) > 0.1);
    }

    #[test]
    fn a_wrong_key_reads_nothing() {
        let p = keyed_pattern("blue heron", "under the mat", 256);
        assert_eq!(keyed_read("red heron", &p), None);
    }

    #[test]
    fn a_note_that_fills_the_memory_is_refused() {
        // 31 bytes in 256 bits leave no padding, so even the right key refuses it
        let t = "x".repeat(31);
        assert_eq!(keyed_read("k", &keyed_pattern("k", &t, 256)), None);
        let t = "x".repeat(27); // 256 - 8 * 28 = 32 padding bits: readable
        assert_eq!(keyed_read("k", &keyed_pattern("k", &t, 256)).as_deref(), Some(t.as_str()));
    }
}
