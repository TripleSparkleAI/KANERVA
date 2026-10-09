//! How much address-noise a read survives: recall of one stored word as its read-address gets noisier, beside
//! the critical-distance the signal-to-noise map predicts for the same memory.
//!
//! Run: `cargo run --release --example address_noise`

use kanerva::{add_address_noise, overlap, random_word, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `address_noise.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(11);
    let mut sdm = Sdm::with_activation_probability(256, 10_000, 0.01, 1);
    let words: Vec<Vec<i8>> = (0..100).map(|_| random_word(256, &mut rng)).collect();
    sdm.write_all(&words);
    out += &format!("{} words in {} hard-locations of 256 bits\n", words.len(), sdm.hard_locations());
    out += "address-noise   recalled (overlap >= 0.95)   mean overlap\n";
    for pct in [10, 20, 30, 40] {
        let trials = 20;
        let (mut hit, mut sum) = (0, 0.0);
        for w in words.iter().take(trials) {
            let read = sdm.read(&add_address_noise(w, pct as f64 / 100.0, &mut rng));
            let o = overlap(&read.word, w);
            sum += o;
            if o >= 0.95 {
                hit += 1;
            }
        }
        out += &format!("{:>9}%      {:>3} of {}                      {:.3}\n", pct, hit, trials, sum / trials as f64);
    }
    out += &format!("predicted critical-distance at {} words: {} bits ({:.0}% address-noise)\n", words.len(), sdm.critical_distance(words.len()), 100.0 * sdm.critical_distance(words.len()) as f64 / 256.0);
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
