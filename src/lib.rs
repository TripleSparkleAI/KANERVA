//! KANERVA: a toolbox for Kanerva's sparse distributed memory (SDM, Kanerva 1988). Zero dependencies.
//!
//! An SDM keeps M hard-locations, each with a fixed random n-bit address and a row of n bit-counters. Writing a
//! pattern adds it to the bit-counters of every hard-location whose address lies within the activation-radius r of
//! it. Reading at a read-address sums the counter rows of the hard-locations within r of the read-address and takes
//! the sign of each column; iterated-reads move a noisy read-address toward the stored pattern.
//!
//! Modules:
//! - `rng`: the xorshift64* generator every module draws from (same stream as the SETTLE interpreter).
//! - `codes`: named ±1 patterns, text as bits, masking text by its name, overlap, address-noise.
//! - `keys`: keyed notes (a key names a rotation of the cube; the key finds and reads its note).
//! - `bits`: `i8` and bit-packed patterns, Hamming distance, nearest neighbour.
//! - `theory`: the Hamming ball, the intersection of two balls, the activation-radius for an activation-probability,
//!   the normal CDF, binomial tables, and Kanerva 1993's optimal activation-probability and memory capacity.
//! - `address`: a named address matrix, the Hamming-ball activation, and the iterated address read over
//!   bit-counters held anywhere (a callback adds one row).
//! - `store`: `Store`, the byte-counter SDM for 10^5 to 10^6 hard-locations: address reads, content-woken reads
//!   (threshold and top-k), shuffles for negative controls, density-scaled wake thresholds.
//! - `smap`: the Bricken-Pehlevan signal-to-noise map: critical-distance, capacity, the activation-radius for an
//!   address-noise level, and the refined predictors (averaged noise, mirror field, Poisson first read, RACE).
//! - `soft`: the soft SDM, hard-locations as p-bits with a logistic cut-off and a softness dial, the mean-field
//!   read, the infinite-location kernel and the softmax-attention read fitted to it.
//! - `refuse`: refusal rules (travel threshold), the exact best-match oracle, diagnostic reads,
//!   and predictor TRACK for the address read.
//! - `track`: predictors for the content-woken reads (TRACK-C, TRACK-G, TRACK-R), read traces and
//!   censuses, probe-calibrated combined scores.
//! - `hopfield`: the zero-temperature Hopfield baseline sized to a pull budget.
//! - `sizing`: the activation-radius to build with (the SNR policy, or tuned for an address-noise) and TRACK's
//!   failure rate and capacity, on `Sdm`.
//! - `calibrated`: the calibrated refusal, which compares a read with reads from never-stored random probes.
//! - `diagnose`: the bit balance of a set of words, and the race for shared hard-locations behind a failed read.
//! - `erase`: the exact undo of a write, and the leave-one-out read.
//! - `hetero`: heteroassociative writes (a data-word of its own width written at an address), int32
//!   bit-counters, and the nearest-k activation for hard-locations placed on the data.
//!
//! See README.md for the equations, a worked example per module and the measured results, KANERVA_TERMS.md
//! for Kanerva's words with his sentences and pages, and DISCOVERIES.md for what each finding became.

#![warn(missing_docs)]
// The loops index several arrays by one position, written as the sums are written in the README; `track::Content::sample`
// keeps its eight arguments as a public call. Both stay as lifted so every output stays bit-identical.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod address;
pub mod bits;
pub mod calibrated;
pub mod codes;
pub mod diagnose;
pub mod erase;
pub mod hetero;
pub mod hopfield;
pub mod keys;
pub mod rails;
pub mod lang;
pub mod refuse;
pub mod rng;
pub mod sdm;
pub mod sizing;
pub mod smap;
pub mod soft;
pub mod store;
pub mod theory;
pub mod track;
pub mod words;

pub use bits::{add_address_noise, overlap};
pub use rng::Rng;
pub use sdm::{hamming, random_word, word_for, ReadResult, Refusal, Sdm, SoftSdm, DEFAULT_ITERATED_READS};
pub use store::{ReadOut, Store};
