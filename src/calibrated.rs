//! The calibrated refusal: a memory that says "I never stored that" by comparing a read with reads from random
//! read-addresses it never stored.
//!
//! The rule (SETTLE campaign lane SDMTRACK, `runs/sdmtrack/REPORT_SDMTRACK.md`, its "calibrated minimum" rule):
//!
//! 1. Read from two sets of random read-addresses (probes) the memory never stored, with the same read the memory
//!    will answer with (Kanerva's address read, or the top-k content read).
//! 2. Each read gives four signals, each smaller when the read is more like a recall: the travel (Hamming
//!    distance from read-address to answer), minus the first-round vote's cosine with the answer, minus the
//!    first-round mean match of the woken rows, and the number of reads made.
//! 3. A read's score is the smallest tail fraction of its four signals among the first probe set
//!    ([`TailCal::min`]); the cut is the score at rank floor(alpha P) of the second probe set ([`calibrate`]).
//! 4. Accept a read iff its score is strictly below the cut: at most alpha of never-stored probes pass.
//!
//! Use at least 1,000 probes per set at alpha 0.01 (the campaign used 1,500): with too few, the cut can sit at the
//! smallest possible score, and then every read is refused ([`CalibratedRefusal::is_degenerate`] says so).
//!
//! Measured (fresh seeds, n 256, M 10^5 and 10^6, three reads, three loads each; `runs/sdmtrack/selfcal2_*.txt`):
//! it refused at least 0.98 of never-stored read-addresses in 18 of 18 cells, where the travel rule with the
//! fixed threshold h(T) ([`crate::sdm::Refusal`]) did so in 12. At the heaviest cell (top-k, M 10^6, T 10,000) it
//! refused 0.990 and kept recall 1.00 / 1.00 / 0.56 at 10 / 20 / 30% address-noise, where the fixed threshold
//! refused 0.056. Its price is recall at 40% address-noise. `examples/calibrated_refusal.rs` re-measures it at a
//! smaller size inside this crate.
//!
//! <claudes_code_comments>
//! ** Function List **
//! Watch                                  - which read the rule watches: address or top-k content
//! signals(read_address, diag)            - the four refusal signals of one read, smaller = more like a recall
//! CalibratedRefusal::from_probes         - calibrate on two probe sets the caller supplies
//! CalibratedRefusal::calibrate           - calibrate on two probe sets drawn from a seed
//! CalibratedRefusal::read                - read and decide in one call
//! CalibratedRefusal::score / accepts     - the score of a read and the decision
//! CalibratedRefusal::floor / is_degenerate - the smallest score, and a cut that refuses everything
//! impl Sdm: calibrated_refusal / read_checked - the same, on the front door
//! CheckedRead                            - a read with its score and decision
//!
//! ** Technical Review **
//! - The rule is assembled from `refuse::Fast` (the diagnostic reads), `track::TailCal` and `track::calibrate`,
//!   exactly as SDMTRACK's instrument `settle-rs/examples/sdmtrack_measure.rs` (part `selfcal2`) assembled it, so
//!   the same probes give the same cut; the crate test `calibrated_matches_the_instrument_by_hand` rebuilds the
//!   instrument's arithmetic inline and checks score and cut are equal.
//! - Probe reads are independent, so `from_probes` reads them on `store::threads()` threads; the order of the
//!   results does not depend on the thread count.
//! - The cut is taken against the store as it was at calibration. A store that is written to afterwards needs a
//!   new calibration (the rule's own measurement recalibrated at every load).
//!
//! </claudes_code_comments>

use crate::bits::{hd, pack, random_pattern};
use crate::refuse::{Diag, Fast};
use crate::rng::Rng;
use crate::sdm::{ReadResult, Sdm};
use crate::store::{threads, Store};
use crate::theory::ball;
use crate::track::{calibrate, TailCal};

/// Which read the calibrated refusal watches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Watch {
    /// Kanerva's address read ([`Store::read_addresses`], [`Sdm::read`]).
    Address,
    /// The top-k content read waking `k` rows ([`Store::read_pulls_topk`], [`Sdm::read_by_content`]).
    TopK(usize),
}

/// The four refusal signals of one read, each smaller when the read looks more like a recall: travel in bits,
/// minus the first-round vote's cosine with the answer, minus the first-round mean match, and the reads made.
///
/// ```
/// use kanerva::{calibrated::signals, refuse::Diag};
/// let d = Diag { z: vec![1, 1, -1, -1], rounds: 2, fixed: true, cos1: 0.5, cosf: 1.0, dot1: 3.0, dotf: 3.0 };
/// assert_eq!(signals(&[1, -1, -1, -1], &d), [1.0, -0.5, -3.0, 2.0]);
/// ```
pub fn signals(read_address: &[i8], d: &Diag) -> [f64; 4] {
    [hd(&pack(read_address), &pack(&d.z)) as f64, -d.cos1, -d.dot1, d.rounds as f64]
}

/// The calibrated refusal of one store at one load: the probe tails and the cut.
#[derive(Clone, Debug)]
pub struct CalibratedRefusal {
    /// The read the rule watches.
    pub watch: Watch,
    /// The most reads each read makes (the campaign used 20).
    pub iterated_reads: usize,
    /// The sorted signals of the first probe set.
    pub tails: TailCal,
    /// Accept a read iff its score is strictly below this.
    pub cut: f64,
    /// The share of never-stored probes the cut lets through, at most.
    pub alpha: f64,
}

/// One read with the calibrated decision.
#[derive(Clone, Debug)]
pub struct CheckedRead {
    /// The read itself.
    pub read: ReadResult,
    /// The read's score: its smallest signal tail fraction among the probes. Smaller is more like a recall.
    pub score: f64,
    /// True when the score is below the cut.
    pub accepted: bool,
}

fn diag(fast: &Fast, watch: Watch, iterated_reads: usize, read_address: &[i8]) -> Diag {
    match watch {
        Watch::Address => fast.address(read_address, iterated_reads),
        Watch::TopK(k) => fast.topk(read_address, iterated_reads, k),
    }
}

fn signals_of_all(fast: &Fast, watch: Watch, iterated_reads: usize, probes: &[Vec<i8>]) -> Vec<Vec<f64>> {
    let th = threads().min(probes.len().max(1));
    if th == 1 {
        return probes.iter().map(|p| signals(p, &diag(fast, watch, iterated_reads, p)).to_vec()).collect();
    }
    let chunk = probes.len().div_ceil(th.max(1)).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = probes
            .chunks(chunk)
            .map(|c| s.spawn(move || c.iter().map(|p| signals(p, &diag(fast, watch, iterated_reads, p)).to_vec()).collect::<Vec<_>>()))
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

impl CalibratedRefusal {
    /// Calibrate on two probe sets you supply: the first sets the signal tails, the second sets the cut at
    /// `alpha`. Both should be read-addresses the store never stored.
    ///
    /// ```
    /// use kanerva::{Rng, bits::random_pattern, store::Store, calibrated::{CalibratedRefusal, Watch}};
    /// let mut rng = Rng::new(4);
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&(0..20).map(|_| random_pattern(256, &mut rng)).collect::<Vec<_>>());
    /// let first: Vec<Vec<i8>> = (0..200).map(|_| random_pattern(256, &mut rng)).collect();
    /// let second: Vec<Vec<i8>> = (0..200).map(|_| random_pattern(256, &mut rng)).collect();
    /// let rule = CalibratedRefusal::from_probes(&st, Watch::Address, 20, &first, &second, 0.01);
    /// assert_eq!(rule.tails.sorted.len(), 4);
    /// assert!(rule.cut > 0.0 && rule.cut <= 1.0);
    /// ```
    pub fn from_probes(store: &Store, watch: Watch, iterated_reads: usize, first: &[Vec<i8>], second: &[Vec<i8>], alpha: f64) -> CalibratedRefusal {
        let fast = Fast::new(store);
        let tails = TailCal::new(&signals_of_all(&fast, watch, iterated_reads, first));
        let scores: Vec<f64> = signals_of_all(&fast, watch, iterated_reads, second).iter().map(|x| tails.min(x)).collect();
        let cut = calibrate(&scores, alpha);
        CalibratedRefusal { watch, iterated_reads, tails, cut, alpha }
    }

    /// Calibrate on two sets of `probes` random read-addresses drawn from `seed`.
    ///
    /// ```
    /// use kanerva::{Rng, bits::random_pattern, store::Store, calibrated::{CalibratedRefusal, Watch}};
    /// let mut rng = Rng::new(4);
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&(0..20).map(|_| random_pattern(256, &mut rng)).collect::<Vec<_>>());
    /// let a = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 200, 0.01, 9);
    /// let b = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 200, 0.01, 9);
    /// assert_eq!(a.cut, b.cut); // the same seed gives the same rule
    /// ```
    pub fn calibrate(store: &Store, watch: Watch, iterated_reads: usize, probes: usize, alpha: f64, seed: u64) -> CalibratedRefusal {
        let mut r = Rng::new(seed ^ 0xCA1B_0000_0000_0001);
        let first: Vec<Vec<i8>> = (0..probes).map(|_| random_pattern(store.n, &mut r)).collect();
        let second: Vec<Vec<i8>> = (0..probes).map(|_| random_pattern(store.n, &mut r)).collect();
        CalibratedRefusal::from_probes(store, watch, iterated_reads, &first, &second, alpha)
    }

    /// The score of a read already made: the smallest tail fraction of its signals.
    ///
    /// ```
    /// use kanerva::{Rng, bits::random_pattern, store::Store, refuse::Fast, calibrated::{CalibratedRefusal, Watch}};
    /// let mut rng = Rng::new(4);
    /// let words: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&words);
    /// let rule = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 1_500, 0.01, 9);
    /// let d = Fast::new(&st).address(&words[0], 20);
    /// assert!(rule.score(&words[0], &d) < rule.cut); // a stored word, read from itself
    /// ```
    pub fn score(&self, read_address: &[i8], d: &Diag) -> f64 {
        self.tails.min(&signals(read_address, d))
    }

    /// True when the read's score is strictly below the cut.
    ///
    /// ```
    /// use kanerva::{Rng, bits::random_pattern, store::Store, refuse::Fast, calibrated::{CalibratedRefusal, Watch}};
    /// let mut rng = Rng::new(4);
    /// let words: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&words);
    /// let rule = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 1_500, 0.01, 9);
    /// let d = Fast::new(&st).address(&words[3], 20);
    /// assert!(rule.accepts(&words[3], &d));
    /// ```
    pub fn accepts(&self, read_address: &[i8], d: &Diag) -> bool {
        self.score(read_address, d) < self.cut
    }

    /// The smallest score any read can have: 1 / (P + 1) for P probes in the first set.
    ///
    /// ```
    /// use kanerva::{Rng, bits::random_pattern, store::Store, calibrated::{CalibratedRefusal, Watch}};
    /// let st = Store::new(256, 2_000, 112, 1);
    /// let rule = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 99, 0.01, 9);
    /// assert_eq!(rule.floor(), 0.01);
    /// ```
    pub fn floor(&self) -> f64 {
        let p = self.tails.sorted.first().map(|v| v.len()).unwrap_or(0);
        1.0 / (p + 1) as f64
    }

    /// True when the cut sits at the floor, so the rule refuses every read. It happens when more than
    /// alpha P of the second set's probes already score the floor (one of their four signals below every probe of
    /// the first set). Measured on 10 seeds at alpha 0.01, n 256, the SNR radius: with 100 probes per set 4 of 10
    /// rules were degenerate at M 2,000 (T 20) and 7 of 10 at M 20,000 (T 200); with 600, 0 and 1; with 1,000 or
    /// 1,500, none. The campaign used 1,500.
    ///
    /// ```
    /// use kanerva::{Rng, bits::random_pattern, store::Store, calibrated::{CalibratedRefusal, Watch}};
    /// let mut rng = Rng::new(6);
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&(0..20).map(|_| random_pattern(256, &mut rng)).collect::<Vec<_>>());
    /// let few = (0..10).filter(|&s| CalibratedRefusal::calibrate(&st, Watch::Address, 20, 100, 0.01, s).is_degenerate()).count();
    /// assert!(few > 0); // too few probes: some cuts refuse everything
    /// assert!((0..3).all(|s| !CalibratedRefusal::calibrate(&st, Watch::Address, 20, 1_500, 0.01, s).is_degenerate()));
    /// ```
    pub fn is_degenerate(&self) -> bool {
        self.cut <= self.floor()
    }

    /// Read from a read-address with the watched read and decide. The answer is the store's own read; the
    /// score comes from the diagnostic read, which gives the same answer. Each call lists the store's filled rows
    /// and reads twice; for many reads, build one [`Fast`] and use [`CalibratedRefusal::score`].
    ///
    /// ```
    /// use kanerva::{Rng, bits::{random_pattern, add_address_noise}, store::Store, calibrated::{CalibratedRefusal, Watch}};
    /// let mut rng = Rng::new(4);
    /// let words: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&words);
    /// let rule = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 1_500, 0.01, 9);
    /// let checked = rule.read(&st, &add_address_noise(&words[1], 0.1, &mut rng));
    /// assert!(checked.accepted && checked.read.word == words[1]);
    /// ```
    pub fn read(&self, store: &Store, read_address: &[i8]) -> CheckedRead {
        let fast = Fast::new(store);
        let d = diag(&fast, self.watch, self.iterated_reads, read_address);
        let score = self.score(read_address, &d);
        // the answer and its counts come from the store's own read (the same answer as the diagnostic read)
        let o = match self.watch {
            Watch::Address => store.read_addresses(read_address, self.iterated_reads),
            Watch::TopK(k) => store.read_pulls_topk(read_address, self.iterated_reads, k),
        };
        debug_assert_eq!(o.z, d.z);
        let read = ReadResult { word: o.z, iterated_reads: o.rounds, activated: o.awake, converged: o.fixed };
        CheckedRead { read, score, accepted: score < self.cut }
    }
}

impl Sdm {
    /// The calibrated refusal for this memory as it holds now, watching the address read ([`Sdm::read`]):
    /// `probes` never-stored random read-addresses per set, at most `alpha` of them accepted. The campaign used
    /// 1,500 probes per set and alpha 0.01. Recalibrate after more words are written.
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, random_word, add_address_noise};
    /// let mut rng = Rng::new(6);
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let words: Vec<Vec<i8>> = (0..20).map(|_| random_word(256, &mut rng)).collect();
    /// sdm.write_all(&words);
    /// let rule = sdm.calibrated_refusal(1_500, 0.01, 2);
    /// assert!(sdm.read_checked(&rule, &add_address_noise(&words[0], 0.1, &mut rng)).accepted);
    /// ```
    pub fn calibrated_refusal(&self, probes: usize, alpha: f64, seed: u64) -> CalibratedRefusal {
        CalibratedRefusal::calibrate(self.store(), Watch::Address, crate::sdm::DEFAULT_ITERATED_READS, probes, alpha, seed)
    }

    /// The calibrated refusal watching the content read with the measured k (activation-probability times M).
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, random_word, add_address_noise, calibrated::Watch};
    /// let mut rng = Rng::new(6);
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// sdm.write_all(&(0..20).map(|_| random_word(256, &mut rng)).collect::<Vec<_>>());
    /// let rule = sdm.calibrated_content_refusal(1_500, 0.01, 2);
    /// assert_eq!(rule.watch, Watch::TopK(sdm.content_k()));
    /// ```
    pub fn calibrated_content_refusal(&self, probes: usize, alpha: f64, seed: u64) -> CalibratedRefusal {
        CalibratedRefusal::calibrate(self.store(), Watch::TopK(self.content_k()), crate::sdm::DEFAULT_ITERATED_READS, probes, alpha, seed)
    }

    /// Read with the rule's read and decide: the answer, its score and whether it is accepted.
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, random_word};
    /// let mut rng = Rng::new(6);
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// sdm.write_all(&(0..20).map(|_| random_word(256, &mut rng)).collect::<Vec<_>>());
    /// let rule = sdm.calibrated_refusal(1_500, 0.01, 2);
    /// let stranger = random_word(256, &mut rng);
    /// let checked = sdm.read_checked(&rule, &stranger);
    /// assert_eq!(checked.accepted, checked.score < rule.cut);
    /// ```
    pub fn read_checked(&self, rule: &CalibratedRefusal, read_address: &[i8]) -> CheckedRead {
        rule.read(self.store(), read_address)
    }

    /// The k the content read should wake: the expected access circle, activation-probability times M, rounded
    /// and at least 1. SDMRADIUS measured this k as the content read's best (`runs/sdmradius`, §5).
    ///
    /// ```
    /// let sdm = kanerva::Sdm::new(256, 2_000, 112, 1);
    /// assert_eq!(sdm.content_k(), (kanerva::theory::ball(256, 112) * 2_000.0).round() as usize);
    /// ```
    pub fn content_k(&self) -> usize {
        (ball(self.word_size(), self.activation_radius()) * self.hard_locations() as f64).round().max(1.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::add_address_noise;

    /// The instrument's selfcal2 arithmetic, inline: tails from P1, the cut at rank floor(alpha P) of P2's min scores.
    #[test]
    fn calibrated_matches_the_instrument_by_hand() {
        let mut rng = Rng::new(31);
        let words: Vec<Vec<i8>> = (0..40).map(|_| random_pattern(256, &mut rng)).collect();
        let mut st = Store::new(256, 3_000, 110, 2);
        st.write_many(&words);
        let p1: Vec<Vec<i8>> = (0..300).map(|_| random_pattern(256, &mut rng)).collect();
        let p2: Vec<Vec<i8>> = (0..300).map(|_| random_pattern(256, &mut rng)).collect();
        let rule = CalibratedRefusal::from_probes(&st, Watch::TopK(60), 20, &p1, &p2, 0.01);
        let f = Fast::new(&st);
        let sig = |c: &[i8]| {
            let d = f.topk(c, 20, 60);
            vec![hd(&pack(c), &pack(&d.z)) as f64, -d.cos1, -d.dot1, d.rounds as f64]
        };
        let s1: Vec<Vec<f64>> = p1.iter().map(|c| sig(c)).collect();
        let s2: Vec<Vec<f64>> = p2.iter().map(|c| sig(c)).collect();
        let tc = TailCal::new(&s1);
        let mut m: Vec<f64> = s2.iter().map(|x| tc.min(x)).collect();
        m.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(rule.cut, m[3]); // floor(0.01 * 300) = 3
        assert_eq!(rule.tails.sorted, tc.sorted);
    }

    #[test]
    fn strangers_are_mostly_refused_and_recalls_mostly_kept() {
        let mut rng = Rng::new(12);
        let words: Vec<Vec<i8>> = (0..30).map(|_| random_pattern(256, &mut rng)).collect();
        let mut st = Store::new(256, 4_000, 109, 3);
        st.write_many(&words);
        let rule = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 1_500, 0.01, 5);
        let refused = (0..200).filter(|_| !rule.read(&st, &random_pattern(256, &mut rng)).accepted).count();
        let kept = (0..60).filter(|i| {
            let w = &words[i % 30];
            let c = rule.read(&st, &add_address_noise(w, 0.1, &mut rng));
            c.accepted && c.read.word == *w
        }).count();
        assert!(refused >= 190, "refused {} of 200 strangers", refused);
        assert!(kept >= 54, "kept {} of 60 recalls", kept);
        // negative control: a rule calibrated with alpha 1 accepts every probe-like read, so strangers pass
        let open = CalibratedRefusal::calibrate(&st, Watch::Address, 20, 1_500, 1.0, 5);
        let passed = (0..200).filter(|_| open.read(&st, &random_pattern(256, &mut rng)).accepted).count();
        assert!(passed > 150, "an open rule lets strangers through: {}", passed);
    }
}
