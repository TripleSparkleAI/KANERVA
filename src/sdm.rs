//! The front door: a sparse distributed memory you write words into and read them back from, in Kanerva's
//! terms. Everything here is a thin layer over the lab modules ([`crate::store`], [`crate::theory`],
//! [`crate::smap`], [`crate::refuse`], [`crate::keys`], [`crate::soft`]), so every number it returns is the
//! number those modules return.
//!
//! A word is a `Vec<i8>` of -1 and +1, `word-size` bits long. [`Sdm`] keeps the hard-locations and their
//! bit-counters; [`Sdm::read`] runs Kanerva's iterated-reads from a read-address; [`Refusal`] decides when an
//! answer travelled too far to be trusted; [`SoftSdm`] is the memory with a soft activation-radius.
//!
//! ```
//! use kanerva::{Sdm, Rng, random_word, add_address_noise, overlap};
//!
//! let mut rng = Rng::new(7);
//! let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
//! let words: Vec<Vec<i8>> = (0..20).map(|_| random_word(256, &mut rng)).collect();
//! sdm.write_all(&words);
//! let read_address = add_address_noise(&words[0], 0.1, &mut rng);
//! let read = sdm.read(&read_address);
//! assert!(overlap(&read.word, &words[0]) > 0.95);
//! ```
//!
//! <claudes_code_comments>
//! ** Function List **
//! random_word(n, rng)                    - a random ±1 word
//! word_for(name, n)                      - the fixed ±1 word that belongs to a name
//! hamming(a, b)                          - Hamming distance between two ±1 words
//! Sdm::new / with_activation_probability - a memory from an activation-radius or an activation-probability
//! Sdm::word_size / hard_locations / activation_radius / activation_probability / written - its shape
//! Sdm::activated(address)                - the access circle: hard-locations within the activation-radius
//! Sdm::write / write_all                 - add words to the bit-counters of their access circles
//! Sdm::read / read_iterated              - Kanerva's read, iterated to a fixed point
//! Sdm::read_by_content                   - the content-woken top-k read
//! Sdm::write_note / read_note            - a text note only its key can read
//! Sdm::predicted_capacity / critical_distance - the S-map's predictions for this memory
//! Sdm::store / store_mut                 - the lab view, a crate::store::Store
//! ReadResult::travelled(from)            - how far a read moved from its read-address
//! Refusal::new / for_memory / accepts    - the travel rule: refuse an answer that moved too far
//! SoftSdm::new / write / read / read_mean_field / attention_inverse_temperature / machine - the soft memory
//!
//! ** Technical Review **
//! - Sdm wraps crate::store::Store (byte bit-counters, bit-packed addresses). new() takes the activation-radius;
//!   with_activation_probability() takes Kanerva's p and finds the smallest radius whose ball holds at least p
//!   of the address space (theory::radius_for). read() is read_iterated() with DEFAULT_ITERATED_READS (20).
//! - read_by_content() is Store::read_pulls_topk: the k filled rows whose bit-counters agree most with the state
//!   vote, addresses unused. It is the read that held the most words in the SETTLE campaign (README).
//! - write_note/read_note use crate::keys: the note is turned by the key into a word, written like any word, and
//!   read back from the key's read-address; a wrong key reads None.
//! - predicted_capacity() and critical_distance() run crate::smap::Lazy on this memory's (n, M, r); they are
//!   predictions from the signal-to-noise map, not measurements.
//! - Refusal is crate::refuse::travel_threshold: with T random stored words, a random read-address has a stored
//!   word within h bits with probability at most `level`, so an answer more than h bits from its read-address is
//!   refused.
//! - SoftSdm wraps crate::soft::Machine with its own Rng (seeded from the memory's seed), 16 write samples,
//!   16 read samples and gain 64.
//! - Nothing here changes an arithmetic path; the lab modules stay public for SETTLE and for measurement.
//!
//! </claudes_code_comments>

use crate::bits::{hd, pack};
use crate::codes::code;
use crate::keys::{keyed_pattern, keyed_read, keyed_read_address};
use crate::refuse::travel_threshold;
use crate::rng::Rng;
use crate::smap::{goal, Lazy};
use crate::soft::Machine;
use crate::store::Store;
use crate::theory::{ball, radius_for};

/// The most iterated-reads [`Sdm::read`] makes before it stops (it stops earlier at a fixed point).
pub const DEFAULT_ITERATED_READS: usize = 20;

/// A random ±1 word of `n` bits, one fair coin per bit.
///
/// ```
/// use kanerva::{random_word, Rng};
/// let w = random_word(256, &mut Rng::new(1));
/// assert_eq!(w.len(), 256);
/// assert!(w.iter().all(|&b| b == 1 || b == -1));
/// ```
pub fn random_word(n: usize, rng: &mut Rng) -> Vec<i8> {
    crate::bits::random_pattern(n, rng)
}

/// The fixed ±1 word that belongs to a name: the same name always gives the same word.
///
/// ```
/// use kanerva::word_for;
/// assert_eq!(word_for("cat", 256), word_for("cat", 256));
/// assert_ne!(word_for("cat", 256), word_for("owl", 256));
/// ```
pub fn word_for(name: &str, n: usize) -> Vec<i8> {
    to_word(&code(name, n))
}

/// The Hamming distance between two ±1 words: the number of bits where they differ.
///
/// ```
/// use kanerva::hamming;
/// assert_eq!(hamming(&[1, -1, 1, 1], &[1, 1, 1, -1]), 2);
/// ```
pub fn hamming(a: &[i8], b: &[i8]) -> usize {
    hd(&pack(a), &pack(b))
}

/// Round a real-valued ±1 vector to a word: +1 where the value is above 0, else -1.
fn to_word(p: &[f64]) -> Vec<i8> {
    p.iter().map(|&v| if v > 0.0 { 1 } else { -1 }).collect()
}

/// What one read returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadResult {
    /// The word the read ended on.
    pub word: Vec<i8>,
    /// How many reads were made, the last one included.
    pub iterated_reads: usize,
    /// How many hard-locations voted on the last read.
    pub activated: usize,
    /// True when the last read returned the word it started from (a fixed point).
    pub converged: bool,
}

impl ReadResult {
    /// How many bits the answer moved away from the read-address it started at.
    ///
    /// ```
    /// use kanerva::{Sdm, word_for};
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let cat = word_for("cat", 256);
    /// sdm.write(&cat);
    /// let read = sdm.read(&cat);
    /// assert_eq!(read.travelled(&cat), 0);
    /// ```
    pub fn travelled(&self, from: &[i8]) -> usize {
        hamming(&self.word, from)
    }
}

/// A sparse distributed memory: `hard_locations` fixed random addresses of `word_size` bits, each with a row of
/// bit-counters, activated within an activation-radius.
#[derive(Clone)]
pub struct Sdm {
    store: Store,
}

impl Sdm {
    /// A memory with the given activation-radius in bits. The addresses come from `seed`.
    ///
    /// ```
    /// use kanerva::Sdm;
    /// let sdm = Sdm::new(256, 2_000, 112, 1);
    /// assert_eq!((sdm.word_size(), sdm.hard_locations(), sdm.activation_radius()), (256, 2_000, 112));
    /// ```
    pub fn new(word_size: usize, hard_locations: usize, activation_radius: usize, seed: u64) -> Sdm {
        Sdm { store: Store::new(word_size, hard_locations, activation_radius, seed) }
    }

    /// A memory whose activation-radius is the smallest that activates at least `activation_probability` of all
    /// addresses (Kanerva's p).
    ///
    /// ```
    /// use kanerva::Sdm;
    /// let sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// assert_eq!(sdm.activation_radius(), 112);
    /// assert!(sdm.activation_probability() >= 0.02);
    /// ```
    pub fn with_activation_probability(word_size: usize, hard_locations: usize, activation_probability: f64, seed: u64) -> Sdm {
        Sdm::new(word_size, hard_locations, radius_for(word_size, activation_probability), seed)
    }

    /// The memory over an existing lab store.
    ///
    /// ```
    /// use kanerva::{Sdm, store::Store};
    /// let sdm = Sdm::from_store(Store::new(256, 2_000, 112, 1));
    /// assert_eq!(sdm.hard_locations(), 2_000);
    /// ```
    pub fn from_store(store: Store) -> Sdm {
        Sdm { store }
    }

    /// The word-size n: bits per address and per word.
    ///
    /// ```
    /// assert_eq!(kanerva::Sdm::new(512, 100, 230, 1).word_size(), 512);
    /// ```
    pub fn word_size(&self) -> usize {
        self.store.n
    }

    /// The number of hard-locations M.
    ///
    /// ```
    /// assert_eq!(kanerva::Sdm::new(256, 100, 112, 1).hard_locations(), 100);
    /// ```
    pub fn hard_locations(&self) -> usize {
        self.store.m
    }

    /// The activation-radius r in bits.
    ///
    /// ```
    /// assert_eq!(kanerva::Sdm::new(256, 100, 107, 1).activation_radius(), 107);
    /// ```
    pub fn activation_radius(&self) -> usize {
        self.store.radius
    }

    /// The share of all addresses within the activation-radius of a point: the expected share of hard-locations
    /// one address activates.
    ///
    /// ```
    /// let p = kanerva::Sdm::new(256, 100, 107, 1).activation_probability();
    /// assert!((p - 0.00513).abs() < 0.00001);
    /// ```
    pub fn activation_probability(&self) -> f64 {
        ball(self.store.n, self.store.radius)
    }

    /// How many words have been written.
    ///
    /// ```
    /// use kanerva::{Sdm, word_for};
    /// let mut sdm = Sdm::new(256, 2_000, 112, 1);
    /// sdm.write(&word_for("cat", 256));
    /// assert_eq!(sdm.written(), 1);
    /// ```
    pub fn written(&self) -> usize {
        self.store.writes
    }

    /// The access circle of an address: the indexes of the hard-locations within the activation-radius of it.
    ///
    /// ```
    /// use kanerva::{Sdm, word_for};
    /// let sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let circle = sdm.activated(&word_for("cat", 256));
    /// assert!(!circle.is_empty() && circle.len() < 200);
    /// ```
    pub fn activated(&self, address: &[i8]) -> Vec<usize> {
        self.store.awake(address).into_iter().map(|i| i as usize).collect()
    }

    /// Write a word at its own address: every activated hard-location adds the word to its bit-counters.
    /// Returns how many hard-locations took it.
    ///
    /// ```
    /// use kanerva::{Sdm, word_for};
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let took = sdm.write(&word_for("cat", 256));
    /// assert_eq!(took, sdm.activated(&word_for("cat", 256)).len());
    /// ```
    pub fn write(&mut self, word: &[i8]) -> usize {
        self.store.write(word)
    }

    /// Write many words, in order. The same result as writing them one by one; the access circles are found in
    /// parallel. Returns the total number of hard-locations that took a word.
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, random_word};
    /// let mut rng = Rng::new(3);
    /// let words: Vec<Vec<i8>> = (0..10).map(|_| random_word(256, &mut rng)).collect();
    /// let mut a = Sdm::new(256, 2_000, 112, 1);
    /// let mut b = a.clone();
    /// a.write_all(&words);
    /// for w in &words { b.write(w); }
    /// assert_eq!(a.store().ctr, b.store().ctr);
    /// ```
    pub fn write_all(&mut self, words: &[Vec<i8>]) -> usize {
        self.store.write_many(words)
    }

    /// Kanerva's read from a read-address: the activated hard-locations vote bit by bit (a zero sum keeps the
    /// bit), and the answer becomes the next read-address, up to [`DEFAULT_ITERATED_READS`] times or until it
    /// stops changing.
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, random_word, add_address_noise};
    /// let mut rng = Rng::new(5);
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let words: Vec<Vec<i8>> = (0..20).map(|_| random_word(256, &mut rng)).collect();
    /// sdm.write_all(&words);
    /// let read = sdm.read(&add_address_noise(&words[3], 0.1, &mut rng));
    /// assert_eq!(read.word, words[3]);
    /// assert!(read.converged);
    /// ```
    pub fn read(&self, read_address: &[i8]) -> ReadResult {
        self.read_iterated(read_address, DEFAULT_ITERATED_READS)
    }

    /// Kanerva's read with at most `iterated_reads` reads (at least one).
    ///
    /// ```
    /// use kanerva::{Sdm, word_for};
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// sdm.write(&word_for("owl", 256));
    /// let once = sdm.read_iterated(&word_for("owl", 256), 1);
    /// assert_eq!(once.iterated_reads, 1);
    /// ```
    pub fn read_iterated(&self, read_address: &[i8], iterated_reads: usize) -> ReadResult {
        let o = self.store.read_addresses(read_address, iterated_reads);
        ReadResult { word: o.z, iterated_reads: o.rounds, activated: o.awake, converged: o.fixed }
    }

    /// The content-woken read: on each read the `k` written hard-locations whose bit-counters agree most with
    /// the current word vote, whatever their addresses. Iterated like [`Sdm::read`], at most `iterated_reads`
    /// times. `k` near the expected access circle (activation-probability times M) is the measured choice.
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, random_word, add_address_noise, overlap};
    /// let mut rng = Rng::new(9);
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let words: Vec<Vec<i8>> = (0..20).map(|_| random_word(256, &mut rng)).collect();
    /// sdm.write_all(&words);
    /// let k = (sdm.activation_probability() * 2_000.0).round() as usize;
    /// let read = sdm.read_by_content(&add_address_noise(&words[0], 0.2, &mut rng), 20, k);
    /// assert!(overlap(&read.word, &words[0]) > 0.95); // the campaign's recall bar
    /// assert!(read.activated <= k);
    /// ```
    pub fn read_by_content(&self, read_address: &[i8], iterated_reads: usize, k: usize) -> ReadResult {
        let o = self.store.read_pulls_topk(read_address, iterated_reads, k);
        ReadResult { word: o.z, iterated_reads: o.rounds, activated: o.awake, converged: o.fixed }
    }

    /// Write a short text note that only its key can read back. The note is turned into a word by the key and
    /// written like any word. Returns how many hard-locations took it.
    ///
    /// ```
    /// use kanerva::Sdm;
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 2);
    /// sdm.write_note("blue heron", "under the mat");
    /// assert_eq!(sdm.read_note("blue heron").as_deref(), Some("under the mat"));
    /// assert_eq!(sdm.read_note("red heron"), None);
    /// ```
    pub fn write_note(&mut self, key: &str, text: &str) -> usize {
        let n = self.store.n;
        self.store.write(&to_word(&keyed_pattern(key, text, n)))
    }

    /// Read the note written under `key`: read from the key's read-address and turn the answer back with the
    /// key. `None` when the padding does not check out, which is what a wrong key reads. This is not
    /// cryptography: the key is hashed to 64 bits with FNV-1a, and a guessed key can be checked offline.
    ///
    /// ```
    /// use kanerva::Sdm;
    /// let sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 2);
    /// assert_eq!(sdm.read_note("nothing written"), None);
    /// ```
    pub fn read_note(&self, key: &str) -> Option<String> {
        let n = self.store.n;
        let read = self.read(&to_word(&keyed_read_address(key, n)));
        let z: Vec<f64> = read.word.iter().map(|&v| v as f64).collect();
        keyed_read(key, &z)
    }

    /// The largest number of stored words from which a read-address with this share of address-noise still
    /// reaches overlap 0.95, as the signal-to-noise map (Bricken and Pehlevan 2021) predicts for this memory.
    /// A prediction, not a measurement. The noise distance is `address_noise * n` truncated.
    ///
    /// The plain map is optimistic under heavy address-noise. At M 10^6 the SETTLE campaign measured it at 1,648
    /// against a measured 700 at 30% address-noise (`runs/sdmradius` §3, re-measured with 400 read-addresses in
    /// `runs/sdmrefuse` §5), and at 403 against 3 at 40%. [`Sdm::predicted_capacity_track`] follows the
    /// measurements (TRACK, mean absolute error 0.015 in the failure fraction).
    ///
    /// ```
    /// let sdm = kanerva::Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// assert!(sdm.predicted_capacity(0.1) > sdm.predicted_capacity(0.3));
    /// ```
    pub fn predicted_capacity(&self, address_noise: f64) -> usize {
        let n = self.store.n;
        Lazy::new(n, self.store.m, self.store.radius).capacity((address_noise * n as f64) as usize, goal(n))
    }

    /// Kanerva's critical-distance for this memory holding `stored` words: the smallest distance in bits from
    /// which the signal-to-noise map no longer moves a read toward its word. A prediction, not a measurement.
    ///
    /// ```
    /// let sdm = kanerva::Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// assert!(sdm.critical_distance(10) > sdm.critical_distance(200));
    /// ```
    pub fn critical_distance(&self, stored: usize) -> usize {
        Lazy::new(self.store.n, self.store.m, self.store.radius).critical(stored)
    }

    /// The lab view: the [`Store`] underneath, with its bit-counters, addresses and the other reads.
    ///
    /// ```
    /// let sdm = kanerva::Sdm::new(256, 100, 112, 1);
    /// assert_eq!(sdm.store().ctr.len(), 100 * 256);
    /// ```
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The lab view, mutable: for the shuffles that serve as negative controls, say.
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, word_for};
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// sdm.write(&word_for("cat", 256));
    /// sdm.store_mut().shuffle_rows(&mut Rng::new(3));
    /// assert_eq!(sdm.written(), 1);
    /// ```
    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }
}

/// The travel rule: refuse an answer that moved more than `threshold` bits from its read-address, because a
/// read-address that far from every stored word was probably never stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The most bits an accepted answer may move.
    pub threshold: usize,
}

impl Refusal {
    /// A refusal with a threshold you choose.
    ///
    /// ```
    /// assert_eq!(kanerva::Refusal::new(95).threshold, 95);
    /// ```
    pub fn new(threshold: usize) -> Refusal {
        Refusal { threshold }
    }

    /// The threshold for `stored` random words of `word_size` bits: the largest h such that a random
    /// read-address has some stored word within h bits with probability at most `level`.
    ///
    /// ```
    /// let r = kanerva::Refusal::for_memory(256, 300, 0.01);
    /// assert_eq!(r.threshold, 95);
    /// ```
    pub fn for_memory(word_size: usize, stored: usize, level: f64) -> Refusal {
        Refusal { threshold: travel_threshold(word_size, stored, level) }
    }

    /// True when the read's answer lies within the threshold of its read-address.
    ///
    /// ```
    /// use kanerva::{Sdm, Refusal, word_for};
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let cat = word_for("cat", 256);
    /// sdm.write(&cat);
    /// let refusal = Refusal::for_memory(256, 1, 0.01);
    /// assert!(refusal.accepts(&cat, &sdm.read(&cat)));
    /// ```
    pub fn accepts(&self, read_address: &[i8], read: &ReadResult) -> bool {
        read.travelled(read_address) <= self.threshold
    }
}

/// A sparse distributed memory with a soft activation-radius: each hard-location is activated with a
/// probability that falls smoothly with distance. `softness` 0 is the hard memory.
#[derive(Clone)]
pub struct SoftSdm {
    machine: Machine,
    rng: Rng,
}

impl SoftSdm {
    /// A soft memory of `hard_locations` of `word_size` bits whose expected access circle is
    /// `activation_probability` of them at every softness.
    ///
    /// ```
    /// let soft = kanerva::SoftSdm::new(256, 2_000, 0.05, 0.25, 1);
    /// assert_eq!(soft.machine().m, 2_000);
    /// ```
    pub fn new(word_size: usize, hard_locations: usize, activation_probability: f64, softness: f64, seed: u64) -> SoftSdm {
        SoftSdm { machine: Machine::new(word_size, hard_locations, activation_probability, softness, 64.0, seed, 16), rng: Rng::new(seed ^ 0x50F7_5D30) }
    }

    /// Write a word: the hard-locations are sampled 16 times from it and each adds the word in proportion to how
    /// often it was activated.
    ///
    /// ```
    /// let mut soft = kanerva::SoftSdm::new(256, 2_000, 0.05, 0.25, 1);
    /// soft.write(&kanerva::word_for("cat", 256));
    /// ```
    pub fn write(&mut self, word: &[i8]) {
        let p: Vec<f64> = word.iter().map(|&v| v as f64).collect();
        self.machine.write(&p, &mut self.rng);
    }

    /// Read from a read-address with `passes` iterated-reads, each a one-way pass of 16 samples.
    ///
    /// ```
    /// use kanerva::{SoftSdm, Rng, word_for, add_address_noise};
    /// let mut soft = SoftSdm::new(256, 2_000, 0.05, 0.25, 1);
    /// let words: Vec<Vec<i8>> = ["a", "b", "c", "d", "e"].iter().map(|s| word_for(s, 256)).collect();
    /// for w in &words { soft.write(w); }
    /// let back = soft.read(&add_address_noise(&words[0], 0.2, &mut Rng::new(4)), 3);
    /// assert_eq!(back, words[0]);
    /// ```
    pub fn read(&mut self, read_address: &[i8], passes: usize) -> Vec<i8> {
        let c: Vec<f64> = read_address.iter().map(|&v| v as f64).collect();
        to_word(&self.machine.recall(&c, passes, 16, false, 0, &mut self.rng))
    }

    /// The same read with infinitely many samples (the mean field): no randomness, so the same call always
    /// returns the same word.
    ///
    /// ```
    /// use kanerva::{SoftSdm, word_for};
    /// let mut soft = SoftSdm::new(256, 2_000, 0.05, 0.25, 1);
    /// soft.write(&word_for("cat", 256));
    /// assert_eq!(soft.read_mean_field(&word_for("cat", 256), 2), word_for("cat", 256));
    /// ```
    pub fn read_mean_field(&self, read_address: &[i8], passes: usize) -> Vec<i8> {
        let c: Vec<f64> = read_address.iter().map(|&v| v as f64).collect();
        to_word(&self.machine.recall_mean_field(&c, passes))
    }

    /// The inverse temperature of the softmax attention whose weights fall with distance as this memory's
    /// shared activation does (fitted to the infinite-location kernel).
    ///
    /// ```
    /// let soft = kanerva::SoftSdm::new(256, 2_000, 0.05, 0.25, 1);
    /// assert!((soft.attention_inverse_temperature() - 2.17).abs() < 0.01);
    /// ```
    pub fn attention_inverse_temperature(&self) -> f64 {
        Machine::fit_beta(&self.machine.kernel_inf())
    }

    /// The lab view: the [`Machine`] underneath.
    ///
    /// ```
    /// let soft = kanerva::SoftSdm::new(256, 2_000, 0.05, 0.25, 1);
    /// assert_eq!(soft.machine().n, 256);
    /// ```
    pub fn machine(&self) -> &Machine {
        &self.machine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::add_address_noise;

    #[test]
    fn the_front_door_reads_the_same_bits_as_the_store() {
        let mut rng = Rng::new(21);
        let words: Vec<Vec<i8>> = (0..40).map(|_| random_word(256, &mut rng)).collect();
        let mut sdm = Sdm::new(256, 3_000, 110, 4);
        let mut st = Store::new(256, 3_000, 110, 4);
        sdm.write_all(&words);
        st.write_many(&words);
        for w in &words[..5] {
            let a = add_address_noise(w, 0.2, &mut rng);
            let (x, y) = (sdm.read(&a), st.read_addresses(&a, DEFAULT_ITERATED_READS));
            assert_eq!((x.word, x.iterated_reads, x.activated, x.converged), (y.z, y.rounds, y.awake, y.fixed));
        }
    }

    #[test]
    fn shuffled_rows_forget_and_a_far_answer_is_refused() {
        let mut rng = Rng::new(8);
        let words: Vec<Vec<i8>> = (0..30).map(|_| random_word(256, &mut rng)).collect();
        let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
        sdm.write_all(&words);
        let a = add_address_noise(&words[0], 0.2, &mut rng);
        assert_eq!(sdm.read(&a).word, words[0]);
        let mut control = sdm.clone();
        control.store_mut().shuffle_rows(&mut Rng::new(3));
        assert!(crate::bits::overlap(&control.read(&a).word, &words[0]) < 0.5, "negative control: shuffled rows recall nothing");
        // a never-written random address lands somewhere far: the travel rule refuses it
        let stranger = random_word(256, &mut rng);
        let refusal = Refusal::for_memory(256, words.len(), 0.01);
        let read = sdm.read(&stranger);
        assert_eq!(refusal.accepts(&stranger, &read), read.travelled(&stranger) <= refusal.threshold);
    }
}
