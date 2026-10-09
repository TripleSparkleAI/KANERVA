//! Diagnostics: is the data what an SDM assumes, and why did one read fail?
//!
//! - [`balance`]: how far stored words are from Kanerva's assumption of fair random bits. Biased bits pull words
//!   together, and words close together crowd each other's access circles. Measured in this crate
//!   (`examples/bit_balance.rs`): at n 256, M 10,000, the SNR activation-radius and 10% address-noise, recall
//!   (overlap >= 0.95, 100 reads per cell) stayed at least 0.91 up to 300 words with fair bits, up to 100 words
//!   when each bit is +1 with probability 0.6, up to 10 at 0.7, and was 0.77 already at 10 words at 0.8.
//! - [`race`]: how many hard-locations a read-address shares with the word it should reach, and with its strongest
//!   rival. SETTLE lane SDMRADIUS (post-hoc, `runs/sdmradius/REPORT_SDMRADIUS.md` §4, M 10^5, r 107, T 5, 40%
//!   address-noise, 600 read-addresses) found every read whose best rival shared more hard-locations failed
//!   (55 of 55), 25 of 28 ties failed, and none of 435 reads three or more ahead failed.
//!
//! <claudes_code_comments>
//! ** Function List **
//! Balance                         - per-bit share of +1 and the summaries a reader needs
//! balance(words)                  - measure the bit balance of a set of words
//! Store::shared_locations(a, b)   - hard-locations activated by both of two addresses
//! Race / race(store, ra, target, others) - shared hard-locations with the target and the best rival
//!
//! ** Technical Review **
//! - balance: share_j = fraction of words with bit j = +1; the expected Hamming distance between two words drawn
//!   independently with those shares is sum_j 2 q_j (1 - q_j), which is n / 2 for fair bits.
//! - shared_locations merges the two sorted wake lists from Store::awake; race calls it once per candidate.
//! - Neither changes the store.
//!
//! </claudes_code_comments>

use crate::store::Store;

/// The bit balance of a set of words.
#[derive(Clone, Debug, PartialEq)]
pub struct Balance {
    /// For each bit, the share of words in which it is +1.
    pub share: Vec<f64>,
    /// The mean over bits of |2 share - 1|: 0 for fair bits, 1 when every bit is constant.
    pub mean_bias: f64,
    /// Bits that are the same in every word.
    pub constant_bits: usize,
    /// The expected Hamming distance between two words drawn with these bit shares, sum of 2 q (1 - q).
    /// n / 2 for fair bits; SDM theory assumes it.
    pub expected_distance: f64,
}

/// Measure the bit balance of `words` (all of one length).
///
/// ```
/// use kanerva::{diagnose::balance, Rng, random_word};
/// let mut rng = Rng::new(1);
/// let fair: Vec<Vec<i8>> = (0..2_000).map(|_| random_word(256, &mut rng)).collect();
/// let b = balance(&fair);
/// assert!(b.mean_bias < 0.05 && b.constant_bits == 0);
/// assert!((b.expected_distance - 128.0).abs() < 1.0);
/// // text as bits is far from fair: the top bit of every ASCII byte is 0
/// let text: Vec<Vec<i8>> = ["a note", "another", "notes!!"].iter().map(|t| {
///     t.bytes().flat_map(|c| (0..8).map(move |k| if (c >> k) & 1 == 1 { 1i8 } else { -1 })).take(48).collect()
/// }).collect();
/// assert!(balance(&text).constant_bits >= 6);
/// ```
pub fn balance(words: &[Vec<i8>]) -> Balance {
    let n = words.first().map(|w| w.len()).unwrap_or(0);
    let t = words.len().max(1) as f64;
    let mut plus = vec![0usize; n];
    for w in words {
        for (c, &b) in plus.iter_mut().zip(w) {
            *c += (b > 0) as usize;
        }
    }
    let share: Vec<f64> = plus.iter().map(|&c| c as f64 / t).collect();
    let mean_bias = if n == 0 { 0.0 } else { share.iter().map(|q| (2.0 * q - 1.0).abs()).sum::<f64>() / n as f64 };
    let constant_bits = plus.iter().filter(|&&c| c == 0 || c == words.len()).count();
    let expected_distance = share.iter().map(|q| 2.0 * q * (1.0 - q)).sum();
    Balance { share, mean_bias, constant_bits, expected_distance }
}

impl Store {
    /// The number of hard-locations activated by both of two addresses.
    ///
    /// ```
    /// use kanerva::{store::Store, Rng, random_word};
    /// let st = Store::new(256, 2_000, 112, 1);
    /// let a = random_word(256, &mut Rng::new(2));
    /// assert_eq!(st.shared_locations(&a, &a), st.awake(&a).len());
    /// let far: Vec<i8> = a.iter().map(|&b| -b).collect();
    /// assert_eq!(st.shared_locations(&a, &far), 0); // 256 bits apart, more than twice the radius
    /// ```
    pub fn shared_locations(&self, a: &[i8], b: &[i8]) -> usize {
        let (x, y) = (self.awake(a), self.awake(b));
        let (mut i, mut j, mut k) = (0, 0, 0);
        while i < x.len() && j < y.len() {
            if x[i] == y[j] {
                k += 1;
                i += 1;
                j += 1;
            } else if x[i] < y[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        k
    }
}

/// The race one read runs: the hard-locations a read-address shares with its target and with its best rival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Race {
    /// Hard-locations activated by both the read-address and the target word.
    pub target: usize,
    /// The most hard-locations it shares with any other word.
    pub best_rival: usize,
    /// The index (into `others`) of that rival, or None when there are no others.
    pub rival: Option<usize>,
}

impl Race {
    /// The target's lead in shared hard-locations; at or below 0 the read was measured to fail almost always.
    ///
    /// ```
    /// let r = kanerva::diagnose::Race { target: 9, best_rival: 6, rival: Some(2) };
    /// assert_eq!(r.margin(), 3);
    /// ```
    pub fn margin(&self) -> i64 {
        self.target as i64 - self.best_rival as i64
    }
}

/// The race a read from `read_address` toward `target` runs against the other stored words.
///
/// ```
/// use kanerva::{store::Store, diagnose::race, Rng, random_word, add_address_noise};
/// let mut rng = Rng::new(3);
/// let words: Vec<Vec<i8>> = (0..5).map(|_| random_word(256, &mut rng)).collect();
/// let st = Store::new(256, 20_000, 107, 1);
/// let close = add_address_noise(&words[0], 0.05, &mut rng);
/// let r = race(&st, &close, &words[0], &words[1..]);
/// assert!(r.margin() > 0); // 5% noise: the target is far ahead
/// ```
pub fn race(store: &Store, read_address: &[i8], target: &[i8], others: &[Vec<i8>]) -> Race {
    let own = store.shared_locations(read_address, target);
    let mut best = (0usize, None);
    for (i, w) in others.iter().enumerate() {
        let s = store.shared_locations(read_address, w);
        if best.1.is_none() || s > best.0 {
            best = (s, Some(i));
        }
    }
    Race { target: own, best_rival: best.0, rival: best.1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{add_address_noise, overlap};
    use crate::rng::Rng;
    use crate::sdm::random_word;

    #[test]
    fn shared_locations_by_brute_force() {
        let st = Store::new(64, 3_000, 26, 4);
        let mut r = Rng::new(9);
        for _ in 0..20 {
            let (a, b) = (random_word(64, &mut r), random_word(64, &mut r));
            let (x, y) = (st.awake(&a), st.awake(&b));
            let want = x.iter().filter(|i| y.contains(i)).count();
            assert_eq!(st.shared_locations(&a, &b), want);
        }
    }

    #[test]
    fn a_read_behind_in_the_race_fails_as_measured() {
        // SDMRADIUS's post-hoc table, re-checked here at a smaller size: reads with margin <= 0 rarely land,
        // reads three or more ahead almost always do
        let mut r = Rng::new(17);
        let words: Vec<Vec<i8>> = (0..5).map(|_| random_word(256, &mut r)).collect();
        let mut st = Store::new(256, 100_000, 107, 5);
        st.write_many(&words);
        let (mut behind, mut behind_ok, mut ahead, mut ahead_ok) = (0, 0, 0, 0);
        for k in 0..400 {
            let w = &words[k % 5];
            let others: Vec<Vec<i8>> = words.iter().filter(|x| *x != w).cloned().collect();
            let a = add_address_noise(w, 0.4, &mut r);
            let m = race(&st, &a, w, &others).margin();
            let ok = overlap(&st.read_addresses(&a, 20).z, w) >= 0.95;
            if m <= 0 {
                behind += 1;
                behind_ok += ok as usize;
            } else if m >= 3 {
                ahead += 1;
                ahead_ok += ok as usize;
            }
        }
        assert!(behind >= 20 && ahead >= 20, "both groups populated: {} {}", behind, ahead);
        assert!((behind_ok as f64) < 0.2 * behind as f64, "behind: {} of {} landed", behind_ok, behind);
        assert!((ahead_ok as f64) > 0.9 * ahead as f64, "ahead: {} of {} landed", ahead_ok, ahead);
    }
}
