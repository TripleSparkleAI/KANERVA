//! Hard-location addresses and Kanerva's iterated address read for bit-counters held anywhere.
//!
//! `Addresses` is the fixed random address matrix of an SDM, bit-packed, with the Hamming-ball activation
//! (which hard-locations wake for a state). `iterated_read` is Kanerva's read, written against a callback that
//! adds one hard-location's counter row to a running sum, so a caller can keep the bit-counters wherever it likes:
//! the SETTLE interpreter keeps them as pulls between things, `store::Store` keeps them as bytes.
//!
//! The read: awake(z) = { i : hamming(a_i, z) <= r }; s_j = sum over awake i of C\[i\]\[j\];
//! z_j <- sign(s_j), a zero sum keeping the old bit; repeat until z stops changing or `iters` reads are done.

use crate::bits::pack_f64;
use crate::codes::seed_of;
use crate::rng::Rng;

/// The address matrix: `m` random n-bit addresses, `wpl` 64-bit words each, row after row.
///
/// ```
/// use kanerva::address::Addresses;
/// let a = Addresses::named("s", 1, 256, 100);
/// assert_eq!((a.n, a.m, a.wpl, a.words.len()), (256, 100, 4, 400));
/// ```
#[derive(Clone)]
pub struct Addresses {
    /// The word-size: bits in each address.
    pub n: usize,
    /// The number of hard-locations M.
    pub m: usize,
    /// Words per hard-location: the 64-bit words one packed address takes, `n.div_ceil(64)`.
    pub wpl: usize,
    /// The packed addresses, `wpl` words per hard-location, hard-location 0 first.
    pub words: Vec<u64>,
}

impl Addresses {
    /// The addresses an SDM named `name` with seed `seed` owns: one generator seeded by
    /// `seed_of("sdm-addresses:<name>:<seed>")`, n fair coin flips per hard-location in order. Nothing needs to
    /// be stored: the same name and seed always give the same matrix.
    ///
    /// ```
    /// use kanerva::address::Addresses;
    /// let a = Addresses::named("s", 1, 256, 50);
    /// let b = Addresses::named("s", 1, 256, 50);
    /// let c = Addresses::named("s", 2, 256, 50);
    /// assert_eq!(a.words, b.words); // the same name and seed rebuild the same matrix
    /// assert_ne!(a.words, c.words);
    /// ```
    pub fn named(name: &str, seed: u64, n: usize, m: usize) -> Addresses {
        let wpl = n.div_ceil(64);
        let mut words = Vec::with_capacity(m * wpl);
        let mut r = Rng::new(seed_of(&format!("sdm-addresses:{}:{}", name, seed)));
        for _ in 0..m {
            let a: Vec<f64> = (0..n).map(|_| if r.unit() < 0.5 { -1.0 } else { 1.0 }).collect();
            words.extend(pack_f64(&a));
        }
        Addresses { n, m, wpl, words }
    }

    /// Location i's packed address.
    ///
    /// ```
    /// use kanerva::address::Addresses;
    /// let a = Addresses::named("s", 1, 256, 10);
    /// assert_eq!(a.row(3), &a.words[3 * a.wpl..4 * a.wpl]);
    /// ```
    pub fn row(&self, i: usize) -> &[u64] {
        &self.words[i * self.wpl..(i + 1) * self.wpl]
    }

    /// Hamming distance from hard-location i's address to a packed state.
    ///
    /// ```
    /// use kanerva::{address::Addresses, bits::hd};
    /// let a = Addresses::named("s", 1, 256, 10);
    /// assert_eq!(a.dist(4, a.row(4)), 0);
    /// assert_eq!(a.dist(4, a.row(5)), hd(a.row(4), a.row(5)));
    /// ```
    pub fn dist(&self, i: usize, q: &[u64]) -> usize {
        self.row(i).iter().zip(q).map(|(x, y)| (x ^ y).count_ones() as usize).sum()
    }

    /// Hard-locations whose address is within `radius` of the ±1 state z, in index order.
    ///
    /// ```
    /// use kanerva::{address::Addresses, codes::code};
    /// let a = Addresses::named("s", 1, 256, 2000);
    /// let z = code("cat", 256);
    /// assert_eq!(a.awake(&z, 256).len(), 2000); // every address is within 256 bits
    /// assert!(a.awake(&z, 100).len() <= a.awake(&z, 112).len()); // a wider activation-radius activates more
    /// ```
    pub fn awake(&self, z: &[f64], radius: usize) -> Vec<usize> {
        let q = pack_f64(z);
        (0..self.m)
            .filter(|&i| {
                let a = &self.words[i * self.wpl..(i + 1) * self.wpl];
                a.iter().zip(&q).map(|(x, y)| (x ^ y).count_ones() as usize).sum::<usize>() <= radius
            })
            .collect()
    }
}

/// The sign of a vote with a hold: +1 above zero, -1 below, and the old bit on a zero sum.
///
/// ```
/// use kanerva::address::sign_hold;
/// assert_eq!(sign_hold(&[2.0, -1.0, 0.0], &[-1.0, 1.0, -1.0]), vec![1.0, -1.0, -1.0]);
/// ```
pub fn sign_hold(sum: &[f64], old: &[f64]) -> Vec<f64> {
    sum.iter().zip(old).map(|(&s, &o)| if s > 0.0 { 1.0 } else if s < 0.0 { -1.0 } else { o }).collect()
}

/// Kanerva's read iterated to a fixed point (at most `iters` reads, at least one). `add_row(i, sum)` adds
/// hard-location i's counter row to `sum`. Returns (final state, reads done, hard-locations awake on the last read).
///
/// ```
/// use kanerva::address::{Addresses, iterated_read};
/// use kanerva::codes::{code, with_address_noise};
/// use kanerva::{theory::radius_for, Rng};
/// let (n, m) = (256, 2000);
/// let a = Addresses::named("s", 1, n, m);
/// let r = radius_for(n, 0.02);
/// let p = code("cat", n);
/// // the caller keeps the bit-counters: here a plain table, written once with p
/// let mut counters = vec![vec![0.0; n]; m];
/// for i in a.awake(&p, r) {
///     for j in 0..n { counters[i][j] += p[j]; }
/// }
/// let read_address = with_address_noise(&p, 0.1, &mut Rng::new(3));
/// let (z, reads, activated) = iterated_read(&a, r, &read_address, 10, |i, sum| {
///     for j in 0..n { sum[j] += counters[i][j]; }
/// });
/// assert_eq!(z, p);
/// assert!(reads >= 1 && activated > 0);
/// ```
pub fn iterated_read<F: FnMut(usize, &mut [f64])>(addr: &Addresses, radius: usize, read_address: &[f64], iters: usize, mut add_row: F) -> (Vec<f64>, usize, usize) {
    let mut z = read_address.to_vec();
    let mut awake = 0;
    for t in 0..iters.max(1) {
        let act = addr.awake(&z, radius);
        awake = act.len();
        let mut sum = vec![0.0; addr.n];
        for &i in &act {
            add_row(i, &mut sum);
        }
        let next = sign_hold(&sum, &z);
        if next == z {
            return (z, t + 1, awake);
        }
        z = next;
    }
    (z, iters.max(1), awake)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::{code, overlap};
    use crate::theory::radius_for;

    #[test]
    fn a_three_location_memory_by_hand() {
        // n 4, addresses built by hand; activation-radius 1 wakes the hard-locations at distance 0 or 1
        let a = Addresses { n: 4, m: 3, wpl: 1, words: vec![0b0000, 0b0001, 0b1111] };
        let z = [-1.0, -1.0, -1.0, -1.0]; // packs to 0b0000
        assert_eq!(a.awake(&z, 1), vec![0, 1]);
        assert_eq!(a.awake(&z, 0), vec![0]);
        assert_eq!(a.awake(&[1.0, 1.0, 1.0, 1.0], 0), vec![2]);
        // bit-counters: hard-location 0 holds +2 on bit 0, hard-location 1 holds -1 on bit 0 and +1 on bit 3
        let rows = [[2.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 1.0], [0.0; 4]];
        let (out, reads, awake) = iterated_read(&a, 1, &z, 5, |i, s| {
            for j in 0..4 {
                s[j] += rows[i][j];
            }
        });
        // read 1: sum = [1, 0, 0, 1], so bits 0 and 3 turn +1 and bits 1 and 2 hold: A = 0b1001.
        // read 2: A is 1 bit from hard-location 1 only, sum = [-1, 0, 0, 1]: B = 0b1000.
        // read 3: B is 1 bit from hard-location 0 only, sum = [2, 0, 0, 0]: A again. The read cycles with period 2
        // and never reaches a fixed point, so it stops after all 5 reads on A, with one hard-location awake.
        assert_eq!(out, vec![1.0, -1.0, -1.0, 1.0]);
        assert_eq!((reads, awake), (5, 1));
        // with 4 reads it stops on B
        let (out4, _, _) = iterated_read(&a, 1, &z, 4, |i, s| s.iter_mut().zip(&rows[i]).for_each(|(x, y)| *x += y));
        assert_eq!(out4, vec![-1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn a_written_pattern_comes_back_and_shuffled_rows_do_not() {
        let (n, m) = (128usize, 800usize);
        let a = Addresses::named("t", 1, n, m);
        let r = radius_for(n, 0.05);
        let pats: Vec<Vec<f64>> = (0..8).map(|k| code(&format!("p{}", k), n)).collect();
        let mut c = vec![vec![0.0f64; n]; m];
        for p in &pats {
            for i in a.awake(p, r) {
                for j in 0..n {
                    c[i][j] += p[j];
                }
            }
        }
        let mut rng = Rng::new(2);
        let read_address = crate::codes::with_address_noise(&pats[3], 0.1, &mut rng);
        let (z, _, _) = iterated_read(&a, r, &read_address, 10, |i, s| s.iter_mut().zip(&c[i]).for_each(|(x, y)| *x += y));
        assert!(overlap(&z, &pats[3]) > 0.95);
        // negative control: the same rows dealt to other hard-locations
        let perm: Vec<usize> = (0..m).map(|i| (i * 7 + 3) % m).collect();
        let (z2, _, _) = iterated_read(&a, r, &read_address, 10, |i, s| s.iter_mut().zip(&c[perm[i]]).for_each(|(x, y)| *x += y));
        assert!(overlap(&z2, &pats[3]) < 0.9);
    }
}
