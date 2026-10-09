//! Choosing the activation-radius for the noise you expect, and predicting how many words a memory holds.
//! The SNR radius suits light address-noise; a radius tuned for 30% address-noise holds more at 30% and fewer at
//! 10%. TRACK predicts the share of reads that miss; the plain signal-to-noise map is optimistic at heavy noise.
//!
//! Run: `cargo run --release --example sizing`

use kanerva::{add_address_noise, overlap, random_word, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `sizing.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(8);
    let words: Vec<Vec<i8>> = (0..600).map(|_| random_word(256, &mut rng)).collect();
    out += "n 256, M 10,000; recall = the address read ends at overlap >= 0.95; 100 reads per cell\n";
    out += "TRACK = predicted recall (1 - predicted_failure_rate, 400 sampled reads)\n";
    for (name, sdm0) in [("SNR radius", Sdm::with_snr_radius(256, 10_000, 4)), ("radius for 30% noise", Sdm::for_address_noise(256, 10_000, 0.3, 4))] {
        out += &format!("\n{} (activation-radius {}):\n", name, sdm0.activation_radius());
        for noise in [0.1, 0.3] {
            out += &format!("  {:.0}% address-noise; S-map capacity {}, TRACK capacity at recall 0.9 {}\n", noise * 100.0, sdm0.predicted_capacity(noise), sdm0.predicted_capacity_track(noise, 0.9, 400, 1));
            for load in [10usize, 50, 300] {
                let mut sdm = sdm0.clone();
                sdm.write_all(&words[..load]);
                let ok = (0..100).filter(|i| {
                    let w = &words[i % load];
                    overlap(&sdm.read(&add_address_noise(w, noise, &mut rng)).word, w) >= 0.95
                }).count();
                let track = 1.0 - sdm.predicted_failure_rate(noise, load, 400, 1);
                out += &format!("    {:>3} words: measured recall {:.2}, TRACK {:.2}\n", load, ok as f64 / 100.0, track);
            }
        }
    }
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
