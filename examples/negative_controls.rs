//! The negative controls every measurement here carries: shuffle the bit-counter rows between hard-locations, or
//! shuffle every bit-counter, and the memory forgets. A result that survives its control is not a memory.
//!
//! Run: `cargo run --release --example negative_controls`

use kanerva::{add_address_noise, overlap, random_word, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `negative_controls.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(17);
    let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    let words: Vec<Vec<i8>> = (0..30).map(|_| random_word(256, &mut rng)).collect();
    sdm.write_all(&words);
    let probes: Vec<Vec<i8>> = words.iter().take(20).map(|w| add_address_noise(w, 0.15, &mut rng)).collect();
    let mean = |m: &Sdm| probes.iter().zip(&words).map(|(p, w)| overlap(&m.read(p).word, w)).sum::<f64>() / probes.len() as f64;
    out += &format!("the memory:                mean overlap {:.3}\n", mean(&sdm));
    let mut rows = sdm.clone();
    rows.store_mut().shuffle_rows(&mut Rng::new(3));
    out += &format!("rows dealt to other hard-locations: {:.3}\n", mean(&rows));
    let mut entries = sdm.clone();
    entries.store_mut().shuffle_entries(&mut Rng::new(4));
    out += &format!("every bit-counter shuffled:  {:.3}\n", mean(&entries));
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
