//! KANERVA: a toolbox for Kanerva's sparse distributed memory (SDM, Kanerva 1988). Zero dependencies.
//!
//! An SDM keeps M hard locations, each with a fixed random n-bit address and a row of n bit-counters. Writing a
//! pattern adds it to the bit-counters of every hard location whose address lies within a activation radius r of it.
//! Reading a read-address sums the counter rows of the hard locations within r of the read-address and takes the sign of each
//! column; repeating the read moves a noisy read-address toward the stored pattern.
//!
//! Modules:
//! - `rng`: the xorshift64* generator every module draws from (same stream as the SETTLE interpreter).
//! - `codes`: named ±1 patterns, text as bits, masking text by its name, overlap, address-noise.
//! - `keys`: keyed notes (a key names a rotation of the cube; the key finds and reads its note).
//! - `bits`: `i8` and bit-packed patterns, Hamming distance, nearest neighbour.
//! - `theory`: the Hamming ball, the intersection of two balls, the activation radius for a activated fraction, the
//!   normal CDF, binomial tables.
//! - `address`: a named address matrix, the Hamming-ball activation, and the iterated address read over
//!   bit-counters held anywhere (a callback adds one row).
//! - `store`: `Store`, the byte-counter SDM for 10^5 to 10^6 hard locations: address reads, content-woken reads
//!   (threshold and top-k), shuffles for negative controls, density-scaled wake thresholds.
//! - `smap`: the Bricken-Pehlevan signal-to-noise map: critical distance, capacity, the activation radius for a address-noise
//!   level, and the refined predictors (averaged noise, mirror field, Poisson first read, RACE).
//! - `soft`: the soft SDM, hard locations as p-bits with a logistic cut-off and a softness dial, the mean-field
//!   read, the infinite-location kernel and the softmax-attention read fitted to it.
//! - `refuse`: refusal rules (travel threshold), the exact nearest-neighbour oracle, diagnostic reads,
//!   and predictor TRACK for the address read.
//! - `track`: predictors for the content-woken reads (TRACK-C, TRACK-G, TRACK-R), read traces and
//!   censuses, probe-calibrated combined scores.
//! - `hopfield`: the zero-temperature Hopfield baseline sized to a pull budget.
//!
//! See README.md for the equations, a worked example per module and the measured results.

pub mod address;
pub mod bits;
pub mod codes;
pub mod hopfield;
pub mod keys;
pub mod refuse;
pub mod rng;
pub mod smap;
pub mod soft;
pub mod store;
pub mod theory;
pub mod track;

pub use rng::Rng;
pub use store::{ReadOut, Store};
