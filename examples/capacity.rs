//! How many words a memory holds: write more and more words, test recall from 10% address-noise after each
//! batch, and set the result beside the capacity the signal-to-noise map predicts.
//!
//! Run: `cargo run --release --example capacity`

use kanerva::{add_address_noise, overlap, random_word, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `capacity.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(3);
    let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    out += &format!("{} hard-locations of 256 bits, activation-radius {}\n", sdm.hard_locations(), sdm.activation_radius());
    out += &format!("predicted capacity at 10% address-noise: {} words\n", sdm.predicted_capacity(0.1));
    out += "stored   recalled from 10% address-noise (overlap >= 0.95, 40 probes)\n";
    let mut words: Vec<Vec<i8>> = Vec::new();
    for target in [25, 50, 100, 150, 200, 300] {
        let more: Vec<Vec<i8>> = (words.len()..target).map(|_| random_word(256, &mut rng)).collect();
        sdm.write_all(&more);
        words.extend(more);
        let probes = 40;
        let hit = (0..probes)
            .filter(|&i| {
                let w = &words[i * words.len() / probes];
                overlap(&sdm.read(&add_address_noise(w, 0.1, &mut rng)).word, w) >= 0.95
            })
            .count();
        out += &format!("{:>6}   {:>2} of {}\n", words.len(), hit, probes);
    }
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
