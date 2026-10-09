//! The baseline an SDM is judged against: a zero-temperature Hopfield memory with Hebbian weights, sized to
//! a pull budget, optionally with a random sign expansion so it can spend a larger budget on n-bit patterns.

use crate::bits::{pack, random_pattern};
use crate::rng::Rng;


/// A zero-temperature Hopfield memory on nh units, patterns held column-wise (unit-major), so the local
/// field of unit k is sum_mu x_mu\[k\] m_mu - T s_k (Hebbian weights, no self-pull). With `expand`, an
/// n-bit pattern p is stored as x = [p ; sign(R p)] for a fixed random ±1 matrix R ((nh - n) x n): the
/// pattern plus random sign features, the same random-expansion trick SDM's addresses play, so a
/// Hopfield net can spend a larger pull budget on n-bit patterns. The read-address is encoded the same way and the
/// answer is the first n units.
///
/// ```
/// use kanerva::hopfield::Hop;
/// let h = Hop::new(64, 0, false, 1);
/// assert_eq!((h.n, h.nh, h.t), (64, 64, 0));
/// ```
pub struct Hop {
    /// The pattern length in bits: the first `n` units hold the pattern.
    pub n: usize,
    /// The number of units: `n` plus the random sign features when `expand` is on, else `n`.
    pub nh: usize,
    /// True when the net stores each pattern with random sign features appended.
    pub expand: bool,
    wpl: usize,
    r: Vec<u64>,
    /// cols\[k\] = the k-th unit's value in every stored pattern.
    cols: Vec<Vec<i8>>,
    /// The number of patterns stored so far.
    pub t: usize,
}

/// Units whose pull count nh(nh-1)/2 matches a budget (the smallest nh reaching it).
///
/// ```
/// use kanerva::hopfield::hop_units_for;
/// assert_eq!(hop_units_for(32_640), 256); // 256 x 255 / 2 pulls
/// assert_eq!(hop_units_for(256_000_000), 22_628); // the budget of a million 256-bit hard-locations
/// ```
pub fn hop_units_for(budget: usize) -> usize {
    let mut nh = ((2.0 * budget as f64).sqrt()) as usize;
    while nh * (nh.saturating_sub(1)) / 2 < budget {
        nh += 1;
    }
    nh
}

impl Hop {
    /// A net for n-bit patterns on `nh` units (`nh` is raised to `n` if smaller, and ignored without `expand`); `seed` draws the random matrix R.
    ///
    /// ```
    /// use kanerva::hopfield::Hop;
    /// assert_eq!(Hop::new(64, 200, true, 1).nh, 200);
    /// assert_eq!(Hop::new(64, 200, false, 1).nh, 64); // without expand nh is ignored
    /// assert_eq!(Hop::new(64, 10, true, 1).nh, 64); // nh is raised to n
    /// ```
    pub fn new(n: usize, nh: usize, expand: bool, seed: u64) -> Hop {
        let wpl = n.div_ceil(64);
        let nh = if expand { nh.max(n) } else { n };
        let mut rr = Rng::new(seed ^ 0x40F1_E1D0);
        let r = (0..nh - n).flat_map(|_| pack(&random_pattern(n, &mut rr))).collect();
        Hop { n, nh, expand, wpl, r, cols: vec![Vec::new(); nh], t: 0 }
    }
    /// x = [p ; sign(R p)] (a systematic code: the pattern itself, then nh - n random sign features; ties +1).
    ///
    /// ```
    /// use kanerva::{hopfield::Hop, bits::random_pattern, Rng};
    /// let h = Hop::new(64, 200, true, 1);
    /// let p = random_pattern(64, &mut Rng::new(2));
    /// let x = h.encode(&p);
    /// assert_eq!(x.len(), 200);
    /// assert_eq!(&x[..64], &p[..]); // a systematic code: the pattern comes first
    /// ```
    pub fn encode(&self, p: &[i8]) -> Vec<i8> {
        let q = pack(p);
        let mut x = p.to_vec();
        x.extend((0..self.nh - self.n).map(|k| {
            let d: u32 = self.r[k * self.wpl..(k + 1) * self.wpl].iter().zip(&q).map(|(a, b)| (a ^ b).count_ones()).sum();
            if 2 * (d as usize) <= self.n {
                1
            } else {
                -1
            }
        }));
        x
    }
    /// The pattern part: the first n units.
    ///
    /// ```
    /// use kanerva::{hopfield::Hop, bits::random_pattern, Rng};
    /// let h = Hop::new(64, 200, true, 1);
    /// let p = random_pattern(64, &mut Rng::new(2));
    /// assert_eq!(h.decode(&h.encode(&p)), p);
    /// ```
    pub fn decode(&self, x: &[i8]) -> Vec<i8> {
        x[..self.n].to_vec()
    }
    /// Encode the pattern and add it to every unit's column: one more Hebbian pattern.
    ///
    /// ```
    /// use kanerva::{hopfield::Hop, bits::random_pattern, Rng};
    /// let mut h = Hop::new(64, 0, false, 1);
    /// h.store(&random_pattern(64, &mut Rng::new(2)));
    /// assert_eq!(h.t, 1);
    /// ```
    pub fn store(&mut self, p: &[i8]) {
        let x = self.encode(p);
        for (k, &v) in x.iter().enumerate() {
            self.cols[k].push(v);
        }
        self.t += 1;
    }
    /// Asynchronous zero-temperature sweeps (random order) from the encoded read-address until nothing flips.
    /// Returns (decoded pattern, sweeps).
    ///
    /// ```
    /// use kanerva::{hopfield::Hop, bits::{flip_exactly, random_pattern}, Rng};
    /// let mut rng = Rng::new(3);
    /// let mut h = Hop::new(128, 0, false, 1);
    /// let ps: Vec<Vec<i8>> = (0..3).map(|_| random_pattern(128, &mut rng)).collect();
    /// for p in &ps { h.store(p); }
    /// let (back, sweeps) = h.recall(&flip_exactly(&ps[0], 12, &mut rng), 20, &mut rng);
    /// assert_eq!(back, ps[0]);
    /// assert!(sweeps >= 1);
    /// ```
    pub fn recall(&self, read_address: &[i8], sweeps: usize, rng: &mut Rng) -> (Vec<i8>, usize) {
        let mut s = self.encode(read_address);
        let t = self.t;
        let mut mo = vec![0i32; t];
        for k in 0..self.nh {
            let sk = s[k] as i32;
            for (m, &x) in mo.iter_mut().zip(&self.cols[k]) {
                *m += x as i32 * sk;
            }
        }
        let mut order: Vec<usize> = (0..self.nh).collect();
        let mut done = 0;
        for sw in 0..sweeps {
            for i in (1..order.len()).rev() {
                let k = rng.below(i + 1);
                order.swap(i, k);
            }
            let mut flips = 0;
            for &k in &order {
                let col = &self.cols[k];
                let h: i64 = col.iter().zip(&mo).map(|(&x, &m)| x as i64 * m as i64).sum::<i64>() - t as i64 * s[k] as i64;
                let new = if h > 0 { 1 } else if h < 0 { -1 } else { s[k] };
                if new != s[k] {
                    let d = 2 * new as i32;
                    for (m, &x) in mo.iter_mut().zip(col) {
                        *m += d * x as i32;
                    }
                    s[k] = new;
                    flips += 1;
                }
            }
            done = sw + 1;
            if flips == 0 {
                break;
            }
        }
        (self.decode(&s), done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{add_address_noise, overlap};

    #[test]
    fn units_for_a_budget_by_hand() {
        // 256 units have 256 * 255 / 2 = 32,640 pulls; 255 units have 32,385
        assert_eq!(hop_units_for(32_640), 256);
        assert_eq!(hop_units_for(32_386), 256);
        assert_eq!(hop_units_for(32_385), 255);
    }

    #[test]
    fn a_few_patterns_come_back_and_an_overloaded_net_fails() {
        let mut r = Rng::new(3);
        let mut h = Hop::new(256, 256, false, 1);
        let ps: Vec<Vec<i8>> = (0..10).map(|_| random_pattern(256, &mut r)).collect();
        for p in &ps {
            h.store(p);
        }
        let ok = ps.iter().filter(|p| overlap(&h.recall(&add_address_noise(p, 0.1, &mut r), 30, &mut r).0, p) >= 0.95).count();
        assert_eq!(ok, 10);
        // negative control: 100 patterns in 256 units is far above Hopfield's 0.14 n, so recall collapses
        let mut h2 = Hop::new(256, 256, false, 1);
        let many: Vec<Vec<i8>> = (0..100).map(|_| random_pattern(256, &mut r)).collect();
        for p in &many {
            h2.store(p);
        }
        let ok2 = many.iter().take(20).filter(|p| overlap(&h2.recall(&add_address_noise(p, 0.1, &mut r), 30, &mut r).0, p) >= 0.95).count();
        assert!(ok2 <= 2, "{}", ok2);
    }
}
