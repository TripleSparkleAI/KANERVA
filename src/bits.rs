//! ±1 patterns as `i8` and as packed 64-bit words, and the Hamming arithmetic on them.
//!
//! Packing convention: bit j of the pattern is bit (j mod 64) of word (j / 64), set when the value is +1.
//! Unused high bits of the last word are zero, so the Hamming distance between packed patterns is a plain
//! popcount of the XOR.

use crate::rng::Rng;

/// Pack a ±1 pattern into bits (bit set for +1).
///
/// ```
/// use kanerva::bits::pack;
/// assert_eq!(pack(&[1, -1, 1, 1]), vec![0b1101]);
/// assert_eq!(pack(&vec![1i8; 65]).len(), 2); // 65 bits take two 64-bit words
/// ```
pub fn pack(z: &[i8]) -> Vec<u64> {
    let mut w = vec![0u64; z.len().div_ceil(64)];
    for (j, &v) in z.iter().enumerate() {
        if v > 0 {
            w[j / 64] |= 1 << (j % 64);
        }
    }
    w
}

/// Pack a ±1 pattern held as `f64` (bit set when the value is above 0).
///
/// ```
/// use kanerva::bits::pack_f64;
/// assert_eq!(pack_f64(&[1.0, -1.0, 1.0, 1.0]), vec![0b1101]);
/// assert_eq!(pack_f64(&[0.0]), vec![0]); // a zero is not above 0, so its bit is clear
/// ```
pub fn pack_f64(z: &[f64]) -> Vec<u64> {
    let mut w = vec![0u64; z.len().div_ceil(64)];
    for (j, &v) in z.iter().enumerate() {
        if v > 0.0 {
            w[j / 64] |= 1 << (j % 64);
        }
    }
    w
}

/// Pack a list of ±1 patterns.
///
/// ```
/// use kanerva::bits::pack_all;
/// let ps = vec![vec![1i8, -1], vec![-1, 1]];
/// assert_eq!(pack_all(&ps), vec![vec![0b01], vec![0b10]]);
/// ```
pub fn pack_all(ps: &[Vec<i8>]) -> Vec<Vec<u64>> {
    ps.iter().map(|p| pack(p)).collect()
}

/// A random ±1 pattern (one uniform draw per bit).
///
/// ```
/// use kanerva::{bits::random_pattern, Rng};
/// let p = random_pattern(256, &mut Rng::new(1));
/// assert_eq!(p.len(), 256);
/// assert!(p.iter().all(|&b| b == 1 || b == -1));
/// assert_eq!(p, random_pattern(256, &mut Rng::new(1)));
/// ```
pub fn random_pattern(n: usize, r: &mut Rng) -> Vec<i8> {
    (0..n).map(|_| if r.unit() < 0.5 { -1 } else { 1 }).collect()
}

/// Flip each bit with probability `d` (one uniform draw per bit).
///
/// ```
/// use kanerva::{bits::{add_address_noise, random_pattern}, Rng};
/// let mut rng = Rng::new(2);
/// let p = random_pattern(256, &mut rng);
/// assert_eq!(add_address_noise(&p, 0.0, &mut rng), p);
/// let all: Vec<i8> = p.iter().map(|&b| -b).collect();
/// assert_eq!(add_address_noise(&p, 1.0, &mut rng), all);
/// ```
pub fn add_address_noise(p: &[i8], d: f64, r: &mut Rng) -> Vec<i8> {
    p.iter().map(|&b| if r.unit() < d { -b } else { b }).collect()
}

/// Flip exactly `k` distinct bits (a partial Fisher-Yates choice of positions).
///
/// ```
/// use kanerva::{bits::{flip_exactly, hd, pack, random_pattern}, Rng};
/// let mut rng = Rng::new(4);
/// let p = random_pattern(256, &mut rng);
/// let q = flip_exactly(&p, 17, &mut rng);
/// assert_eq!(hd(&pack(&p), &pack(&q)), 17);
/// ```
pub fn flip_exactly(p: &[i8], k: usize, r: &mut Rng) -> Vec<i8> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    for i in 0..k.min(p.len()) {
        let j = i + r.below(p.len() - i);
        idx.swap(i, j);
    }
    let mut z = p.to_vec();
    for &i in &idx[..k.min(p.len())] {
        z[i] = -z[i];
    }
    z
}

/// Overlap of two ±1 patterns: (agreements - disagreements) / length of `a`.
///
/// ```
/// use kanerva::bits::overlap;
/// assert_eq!(overlap(&[1, 1, -1, -1], &[1, 1, -1, -1]), 1.0);
/// assert_eq!(overlap(&[1, 1, -1, -1], &[1, -1, -1, 1]), 0.0);
/// assert_eq!(overlap(&[1, 1], &[-1, -1]), -1.0);
/// ```
pub fn overlap(a: &[i8], b: &[i8]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as i32) * (y as i32)).sum::<i32>() as f64 / a.len() as f64
}

/// Hamming distance between packed patterns.
///
/// ```
/// use kanerva::bits::hd;
/// assert_eq!(hd(&[0b1011], &[0b0110]), 3);
/// ```
pub fn hd(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones() as usize).sum()
}

/// (index, distance) of the nearest of `pats` (packed) to `q`, ties to the lowest index.
///
/// ```
/// use kanerva::bits::nearest;
/// let pats = vec![vec![0b0000u64], vec![0b0111], vec![0b0001]];
/// assert_eq!(nearest(&pats, &[0b0011]), (1, 1)); // two at distance 1: the lower index wins
/// ```
pub fn nearest(pats: &[Vec<u64>], q: &[u64]) -> (usize, usize) {
    let mut best = (0usize, usize::MAX);
    for (i, p) in pats.iter().enumerate() {
        let d = hd(p, q);
        if d < best.1 {
            best = (i, d);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_by_hand() {
        let z: Vec<i8> = vec![1, -1, 1, 1];
        assert_eq!(pack(&z), vec![0b1101]);
        assert_eq!(pack_f64(&[1.0, -1.0, 1.0, 1.0]), vec![0b1101]);
        let long: Vec<i8> = (0..70).map(|j| if j == 65 { 1 } else { -1 }).collect();
        assert_eq!(pack(&long), vec![0, 0b10]);
    }

    #[test]
    fn hamming_distance_equals_the_overlap_identity() {
        // overlap = 1 - 2 d / n for every pair
        let mut r = Rng::new(3);
        for _ in 0..20 {
            let a = random_pattern(200, &mut r);
            let b = add_address_noise(&a, 0.3, &mut r);
            let d = hd(&pack(&a), &pack(&b));
            assert!((overlap(&a, &b) - (1.0 - 2.0 * d as f64 / 200.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn flip_exactly_flips_exactly_and_zero_flips_nothing() {
        let mut r = Rng::new(5);
        let p = random_pattern(100, &mut r);
        for k in [0usize, 1, 37, 100] {
            let z = flip_exactly(&p, k, &mut r);
            assert_eq!(hd(&pack(&p), &pack(&z)), k);
        }
    }

    #[test]
    fn nearest_breaks_ties_to_the_lowest_index() {
        let a = pack(&[1, 1, 1, 1]);
        let b = pack(&[1, 1, 1, -1]);
        let c = pack(&[1, 1, -1, 1]);
        assert_eq!(nearest(&[b.clone(), c.clone()], &a), (0, 1));
        assert_eq!(nearest(&[c, b, a.clone()], &a), (2, 0));
    }
}
