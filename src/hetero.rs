//! Heteroassociative writes: the data-word written at an address is a different word from the address, with its own
//! word-size. Kanerva 1988 treats the general case, "the address and the datum need not be the same"; the `Store`
//! keeps the autoassociative case, where a word is written at itself. A classifier is the plain example: the address
//! is a 784-bit handwritten digit and the data-word is its label as ten bits.
//!
//! ```
//! use kanerva::hetero::{Hetero, Wake};
//! use kanerva::address::Addresses;
//! use kanerva::bits::pack;
//!
//! // two hard-locations at 8 bits, all +1 and all -1; data-words of 3 bits
//! let a: Vec<u64> = [[1i8; 8], [-1i8; 8]].iter().flat_map(|r| pack(r)).collect();
//! let mut h = Hetero::from_addresses(Addresses { n: 8, m: 2, wpl: 1, words: a }, 3);
//! let near_plus = pack(&[1, 1, 1, 1, 1, 1, 1, -1]);
//! h.write(&near_plus, &[1, -1, -1], Wake::Nearest(1)); // wakes hard-location 0 only
//! let r = h.read(&near_plus, Wake::Nearest(1));
//! assert_eq!((r.woken, r.radius, r.sums), (vec![0], 1, vec![1, -1, -1]));
//! assert_eq!(h.read(&pack(&[-1i8; 8]), Wake::Radius(0)).sums, vec![0, 0, 0]); // hard-location 1 holds nothing
//! ```
//!
//! <claudes_code_comments>
//! ** Function List **
//! Wake                         - which hard-locations a write or read activates: a fixed radius, or the nearest k
//! HeteroRead                   - one read: the column sums, the woken hard-locations, the radius used
//! Hetero::from_addresses(a, w) - hard-locations at the caller's addresses, w int32 bit-counters each
//! Hetero::dists(q)             - the Hamming distance from every hard-location to a packed address
//! Hetero::activate(q, wake)    - the woken hard-locations in index order, and the activation-radius used
//! Hetero::write(q, d, wake)    - add the +-1 data-word d to the bit-counters of every woken hard-location
//! Hetero::read(q, wake)        - sum the bit-counter rows of the woken hard-locations
//! Hetero::clear()              - empty every bit-counter (the addresses stay)
//!
//! ** Technical Review **
//! - Addresses are the crate's packed `Addresses` (bit j in bit j mod 64 of word j / 64). Bits at or above n in the
//!   last word are cleared on construction, so a caller's stray high bits cannot add to a distance.
//! - `Wake::Radius(r)` is Kanerva's activation: every hard-location within r. `Wake::Nearest(k)` is a departure from
//!   Kanerva's design: the activation-radius is chosen per address as the smallest r whose ball holds at least k
//!   hard-locations, and every hard-location at that distance wakes, so a tie can wake more than k. A memory whose
//!   hard-locations sit on the data (not at random) needs it: a fixed radius wakes thousands for one address and
//!   none for another.
//! - Bit-counters are `i32`, one row of `width` per hard-location, unclamped. A read returns the column sums as
//!   `i64`; turning them into an answer (a sign word, or the best-match among a few data-words) is the caller's.
//! - `writes_at[i]` counts the writes hard-location i took. Nothing here draws a random number: the caller places
//!   the hard-locations, so the same addresses always give the same memory.
//! - Cost: an activation is M distances of `wpl` popcounts plus one pass over M; a write or read adds `width` per
//!   woken hard-location.
//!
//! </claudes_code_comments>

use crate::address::Addresses;

/// Which hard-locations a write or a read activates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wake {
    /// Kanerva's activation: every hard-location within this activation-radius.
    Radius(usize),
    /// The departure: the smallest activation-radius whose ball holds at least this many hard-locations.
    Nearest(usize),
}

/// What one read found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeteroRead {
    /// The sum of each bit-counter column over the woken hard-locations.
    pub sums: Vec<i64>,
    /// The woken hard-locations, in index order.
    pub woken: Vec<u32>,
    /// The activation-radius the read used.
    pub radius: usize,
}

/// Hard-locations at given addresses, each with a row of `width` int32 bit-counters for its data-words.
#[derive(Clone)]
pub struct Hetero {
    /// The hard-locations' addresses.
    pub addr: Addresses,
    /// The data-word size: bit-counters per hard-location.
    pub width: usize,
    /// The bit-counters, `width` per hard-location, hard-location 0 first.
    pub ctr: Vec<i32>,
    /// How many writes each hard-location took.
    pub writes_at: Vec<u32>,
    /// How many data-words were written.
    pub writes: usize,
}

impl Hetero {
    /// A memory over the caller's addresses with `width` bit-counters per hard-location, all zero.
    ///
    /// ```
    /// use kanerva::{hetero::Hetero, address::Addresses};
    /// let h = Hetero::from_addresses(Addresses::named("h", 1, 100, 50), 10);
    /// assert_eq!((h.ctr.len(), h.writes), (500, 0));
    /// ```
    pub fn from_addresses(mut addr: Addresses, width: usize) -> Hetero {
        let tail = addr.n % 64;
        if tail != 0 {
            let mask = (1u64 << tail) - 1;
            for i in 0..addr.m {
                addr.words[i * addr.wpl + addr.wpl - 1] &= mask;
            }
        }
        let m = addr.m;
        Hetero { addr, width, ctr: vec![0; m * width], writes_at: vec![0; m], writes: 0 }
    }

    /// The Hamming distance from every hard-location's address to the packed address q.
    ///
    /// ```
    /// use kanerva::{hetero::Hetero, address::Addresses};
    /// let a = Addresses::named("h", 1, 64, 3);
    /// let q = a.row(2).to_vec();
    /// let h = Hetero::from_addresses(a, 1);
    /// assert_eq!(h.dists(&q)[2], 0);
    /// ```
    pub fn dists(&self, q: &[u64]) -> Vec<u32> {
        (0..self.addr.m).map(|i| self.addr.dist(i, q) as u32).collect()
    }

    /// The woken hard-locations for the packed address q, in index order, and the activation-radius used.
    /// `Nearest(k)` with k above M wakes every hard-location at radius n.
    ///
    /// ```
    /// use kanerva::{hetero::{Hetero, Wake}, address::Addresses};
    /// let a = Addresses::named("h", 1, 64, 40);
    /// let q = a.row(7).to_vec();
    /// let h = Hetero::from_addresses(a, 1);
    /// let (woken, r) = h.activate(&q, Wake::Nearest(1));
    /// assert_eq!((woken, r), (vec![7], 0)); // its own hard-location, at distance 0
    /// assert!(h.activate(&q, Wake::Nearest(5)).0.len() >= 5);
    /// assert_eq!(h.activate(&q, Wake::Radius(64)).0.len(), 40);
    /// ```
    pub fn activate(&self, q: &[u64], wake: Wake) -> (Vec<u32>, usize) {
        let n = self.addr.n;
        let d = self.dists(q);
        let r = match wake {
            Wake::Radius(r) => r,
            Wake::Nearest(k) => {
                let mut hist = vec![0usize; n + 1];
                for &x in &d {
                    hist[(x as usize).min(n)] += 1;
                }
                let (mut r, mut c) = (0, hist[0]);
                while c < k && r < n {
                    r += 1;
                    c += hist[r];
                }
                r
            }
        };
        let woken = d.iter().enumerate().filter(|&(_, &x)| x as usize <= r).map(|(i, _)| i as u32).collect();
        (woken, r)
    }

    /// Write the ±1 data-word `data` (length `width`) at the packed address q: add it to the bit-counters of every
    /// woken hard-location. Returns how many took it.
    ///
    /// ```
    /// use kanerva::{hetero::{Hetero, Wake}, address::Addresses};
    /// let a = Addresses::named("h", 1, 64, 40);
    /// let q = a.row(3).to_vec();
    /// let mut h = Hetero::from_addresses(a, 2);
    /// assert_eq!(h.write(&q, &[1, -1], Wake::Nearest(1)), 1);
    /// assert_eq!(&h.ctr[6..8], &[1, -1]);
    /// ```
    pub fn write(&mut self, q: &[u64], data: &[i8], wake: Wake) -> usize {
        assert_eq!(data.len(), self.width, "a data-word has width values");
        let (woken, _) = self.activate(q, wake);
        let w = self.width;
        for &i in &woken {
            let row = &mut self.ctr[i as usize * w..(i as usize + 1) * w];
            for (c, &v) in row.iter_mut().zip(data) {
                *c += v as i32;
            }
            self.writes_at[i as usize] += 1;
        }
        self.writes += 1;
        woken.len()
    }

    /// Read at the packed address q: the column sums of the woken hard-locations' bit-counters.
    ///
    /// ```
    /// use kanerva::{hetero::{Hetero, Wake}, address::Addresses};
    /// let a = Addresses::named("h", 1, 64, 40);
    /// let q = a.row(3).to_vec();
    /// let mut h = Hetero::from_addresses(a, 2);
    /// h.write(&q, &[1, -1], Wake::Nearest(1));
    /// h.write(&q, &[1, 1], Wake::Nearest(1));
    /// assert_eq!(h.read(&q, Wake::Nearest(1)).sums, vec![2, 0]);
    /// ```
    pub fn read(&self, q: &[u64], wake: Wake) -> HeteroRead {
        let (woken, radius) = self.activate(q, wake);
        let w = self.width;
        let mut sums = vec![0i64; w];
        for &i in &woken {
            for (s, &c) in sums.iter_mut().zip(&self.ctr[i as usize * w..(i as usize + 1) * w]) {
                *s += c as i64;
            }
        }
        HeteroRead { sums, woken, radius }
    }

    /// Empty every bit-counter and the write counts; the addresses stay.
    ///
    /// ```
    /// use kanerva::{hetero::{Hetero, Wake}, address::Addresses};
    /// let a = Addresses::named("h", 1, 64, 4);
    /// let q = a.row(0).to_vec();
    /// let mut h = Hetero::from_addresses(a, 1);
    /// h.write(&q, &[1], Wake::Radius(64));
    /// h.clear();
    /// assert!(h.ctr.iter().all(|&c| c == 0) && h.writes == 0);
    /// ```
    pub fn clear(&mut self) {
        self.ctr.iter_mut().for_each(|c| *c = 0);
        self.writes_at.iter_mut().for_each(|c| *c = 0);
        self.writes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{hd, pack};
    use crate::rng::Rng;

    fn random_addresses(n: usize, m: usize, seed: u64) -> Addresses {
        let mut r = Rng::new(seed);
        let rows: Vec<Vec<i8>> = (0..m).map(|_| crate::bits::random_pattern(n, &mut r)).collect();
        Addresses { n, m, wpl: n.div_ceil(64), words: rows.iter().flat_map(|p| pack(p)).collect() }
    }

    // the nearest-k rule by brute force: sort the distances, take the k-th smallest as the radius
    fn nearest_by_sorting(h: &Hetero, q: &[u64], k: usize) -> (Vec<u32>, usize) {
        let mut d: Vec<usize> = (0..h.addr.m).map(|i| hd(h.addr.row(i), q)).collect();
        let woke = |r: usize, d: &[usize]| d.iter().enumerate().filter(|&(_, &x)| x <= r).map(|(i, _)| i as u32).collect::<Vec<_>>();
        let all = d.clone();
        d.sort_unstable();
        let r = if k == 0 { 0 } else if k > d.len() { h.addr.n } else { d[k - 1] };
        (woke(r, &all), r)
    }

    #[test]
    fn nearest_k_agrees_with_sorting_the_distances() {
        let h = Hetero::from_addresses(random_addresses(784, 300, 3), 10);
        let mut r = Rng::new(9);
        for k in [1, 2, 5, 17, 300, 301] {
            for _ in 0..20 {
                let q = pack(&crate::bits::random_pattern(784, &mut r));
                assert_eq!(h.activate(&q, Wake::Nearest(k)), nearest_by_sorting(&h, &q, k), "k {k}");
            }
        }
    }

    #[test]
    fn a_write_and_a_read_by_hand() {
        // three hard-locations at 4 bits: 0000, 0011, 1111; a read at 0001 is 1 from the first two, 3 from the last
        let rows: [[i8; 4]; 3] = [[-1, -1, -1, -1], [1, 1, -1, -1], [1, 1, 1, 1]];
        let a = Addresses { n: 4, m: 3, wpl: 1, words: rows.iter().flat_map(|p| pack(p)).collect() };
        let mut h = Hetero::from_addresses(a, 2);
        let q = pack(&[1, -1, -1, -1]);
        assert_eq!(h.activate(&q, Wake::Nearest(2)), (vec![0, 1], 1));
        assert_eq!(h.activate(&q, Wake::Nearest(1)), (vec![0, 1], 1), "a tie wakes both");
        assert_eq!(h.write(&q, &[1, -1], Wake::Nearest(1)), 2);
        assert_eq!(h.write(&q, &[1, 1], Wake::Radius(3)), 3);
        assert_eq!(h.ctr, vec![2, 0, 2, 0, 1, 1]);
        assert_eq!(h.writes_at, vec![2, 2, 1]);
        assert_eq!(h.read(&q, Wake::Radius(3)).sums, vec![5, 1]);
    }

    #[test]
    fn stray_high_bits_are_cleared() {
        let mut a = Addresses { n: 4, m: 1, wpl: 1, words: vec![0b1111_0000] };
        a.words[0] |= 1 << 40;
        let h = Hetero::from_addresses(a, 1);
        assert_eq!(h.addr.words[0], 0);
        assert_eq!(h.dists(&[0]), vec![0]);
    }

    #[test]
    fn negative_control_a_shuffled_label_reads_differently() {
        // write label 0 at one address, then read: the sum names label 0; the same writes with the labels swapped
        // name label 1, so the read sees the data and not only the address
        let a = random_addresses(128, 60, 5);
        let q = a.row(10).to_vec();
        let mut x = Hetero::from_addresses(a.clone(), 2);
        let mut y = Hetero::from_addresses(a, 2);
        x.write(&q, &[1, -1], Wake::Nearest(5));
        y.write(&q, &[-1, 1], Wake::Nearest(5));
        let (sx, sy) = (x.read(&q, Wake::Nearest(5)).sums, y.read(&q, Wake::Nearest(5)).sums);
        assert!(sx[0] > sx[1] && sy[1] > sy[0]);
    }
}
