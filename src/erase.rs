//! Erasing a word, and reading as if one word had never been written.
//!
//! A write adds a word to the bit-counters of its access circle, so subtracting it from the same circle takes it
//! out again, exactly, as long as no counter it touched was clamped at +-127 in between. [`Store::erase`] does
//! that and refuses when the store has ever clamped a counter, because then the subtraction would not undo the
//! write.
//!
//! [`Sdm::read_without`] reads as if one stored word had never been written: the leave-one-out read. Kanerva's
//! read from a noisy copy of a stored word is self-inclusive: the word votes for itself. That is right for
//! recall, and it says nothing about whether the other words would lead a read to it. The campaign's address
//! studies found a self-inclusive score can be highest for the least useful memory (`WIKI/theory/173` §1, on
//! n-gram stores, not on an SDM); this read is the protocol that does not have that hole. What it shows on an
//! SDM has not been measured.
//!
//! <claudes_code_comments>
//! ** Function List **
//! EraseRefused                 - why an erase was refused
//! Store::erase(word)           - subtract a written word from its access circle, exactly
//! Sdm::erase(word)             - the same on the front door
//! Sdm::read_without(word, ra)  - read from ra with `word` erased, the store untouched
//!
//! ** Technical Review **
//! - erase checks `overflow == 0` before touching anything; then for each activated row it subtracts the word in
//!   i16 and clamps (a subtraction that clamps means the word was never written there, and it is counted in
//!   `overflow` like any clamp), recomputes `filled`, and decrements `writes` (saturating).
//! - read_without clones the store, so it costs one copy of the counters (M n bytes).
//!
//! </claudes_code_comments>

use crate::sdm::{ReadResult, Sdm};
use crate::store::Store;

/// Why [`Store::erase`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EraseRefused {
    /// The store has clamped a counter at least once, so a subtraction might not undo a write.
    Clamped,
    /// The word's length is not the store's word-size.
    WrongLength,
}

impl std::fmt::Display for EraseRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EraseRefused::Clamped => write!(f, "the store has clamped a bit-counter, so erasing would not undo a write"),
            EraseRefused::WrongLength => write!(f, "the word's length is not the store's word-size"),
        }
    }
}

impl std::error::Error for EraseRefused {}

impl Store {
    /// Subtract `word` from the bit-counters of its access circle: the exact undo of [`Store::write`] while no
    /// counter has been clamped. Returns how many hard-locations it touched.
    ///
    /// ```
    /// use kanerva::{store::Store, Rng, random_word};
    /// let mut rng = Rng::new(1);
    /// let words: Vec<Vec<i8>> = (0..10).map(|_| random_word(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&words[..9]);
    /// let before = st.ctr.clone();
    /// st.write(&words[9]);
    /// st.erase(&words[9]).unwrap();
    /// assert_eq!(st.ctr, before); // exactly as if it had never been written
    /// assert_eq!(st.writes, 9);
    /// ```
    pub fn erase(&mut self, word: &[i8]) -> Result<usize, EraseRefused> {
        if word.len() != self.n {
            return Err(EraseRefused::WrongLength);
        }
        if self.overflow > 0 {
            return Err(EraseRefused::Clamped);
        }
        let n = self.n;
        let act = self.awake(word);
        for &i in &act {
            let i = i as usize;
            let row = &mut self.ctr[i * n..(i + 1) * n];
            let mut clamped = 0u64;
            for (c, &x) in row.iter_mut().zip(word) {
                let v = *c as i16 - x as i16;
                clamped += u64::from(!(-127..=127).contains(&v));
                *c = v.clamp(-127, 127) as i8;
            }
            self.overflow += clamped;
            self.filled[i] = row.iter().any(|&c| c != 0);
        }
        self.writes = self.writes.saturating_sub(1);
        Ok(act.len())
    }
}

impl Sdm {
    /// Erase a written word ([`Store::erase`]).
    ///
    /// ```
    /// use kanerva::{Sdm, word_for};
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let cat = word_for("cat", 256);
    /// sdm.write(&cat);
    /// sdm.erase(&cat).unwrap();
    /// assert_eq!(sdm.written(), 0);
    /// assert!(sdm.store().ctr.iter().all(|&c| c == 0));
    /// ```
    pub fn erase(&mut self, word: &[i8]) -> Result<usize, EraseRefused> {
        self.store_mut().erase(word)
    }

    /// Read from `read_address` as if `word` had never been written (the leave-one-out read). The memory itself
    /// is not changed.
    ///
    /// ```
    /// use kanerva::{Sdm, Rng, random_word, add_address_noise, overlap};
    /// let mut rng = Rng::new(2);
    /// let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    /// let words: Vec<Vec<i8>> = (0..10).map(|_| random_word(256, &mut rng)).collect();
    /// sdm.write_all(&words);
    /// let ra = add_address_noise(&words[4], 0.1, &mut rng);
    /// assert_eq!(sdm.read(&ra).word, words[4]);                       // with itself in the store: recalled
    /// let without = sdm.read_without(&words[4], &ra).unwrap();
    /// assert!(overlap(&without.word, &words[4]) < 0.95);             // without: random words cannot lead back to it
    /// assert_eq!(sdm.written(), 10);
    /// ```
    pub fn read_without(&self, word: &[i8], read_address: &[i8]) -> Result<ReadResult, EraseRefused> {
        let mut other = self.clone();
        other.erase(word)?;
        Ok(other.read(read_address))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::sdm::random_word;

    #[test]
    fn erase_undoes_any_write_in_any_order() {
        let mut r = Rng::new(5);
        let words: Vec<Vec<i8>> = (0..30).map(|_| random_word(128, &mut r)).collect();
        let mut st = Store::new(128, 1_500, 52, 3);
        st.write_many(&words);
        let mut want = Store::new(128, 1_500, 52, 3);
        want.write_many(&words[10..]);
        for w in words[..10].iter().rev() {
            st.erase(w).unwrap();
        }
        assert_eq!((st.ctr, st.filled, st.writes, st.overflow), (want.ctr, want.filled, want.writes, 0));
    }

    #[test]
    fn a_clamped_store_refuses() {
        let mut st = Store::new(8, 1, 8, 0);
        for _ in 0..130 {
            st.write(&[1; 8]);
        }
        assert_eq!(st.erase(&[1; 8]), Err(EraseRefused::Clamped));
        assert_eq!(st.counter(0, 0), 127); // nothing was touched
        assert_eq!(Store::new(8, 1, 8, 0).erase(&[1; 4]), Err(EraseRefused::WrongLength));
    }
}
