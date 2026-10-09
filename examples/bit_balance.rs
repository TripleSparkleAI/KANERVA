//! Kanerva's theory assumes every bit of every word is a fair coin. Words whose bits lean one way sit closer
//! together, crowd each other's access circles, and the memory holds far fewer of them. `diagnose::balance`
//! measures the lean before you write; the q 0.5 row is the control (fair bits).
//!
//! Run: `cargo run --release --example bit_balance`

use kanerva::{add_address_noise, diagnose::balance, overlap, Rng, Sdm};

fn leaning(n: usize, q: f64, r: &mut Rng) -> Vec<i8> {
    (0..n).map(|_| if r.unit() < q { 1 } else { -1 }).collect()
}

/// The example's whole output, so `tests/examples.rs` can check it against `bit_balance.out`.
pub fn run() -> String {
    let mut out = String::new();
    out += "n 256, M 10,000, the SNR activation-radius; each bit is +1 with probability q\n";
    out += "recall = the address read from 10% address-noise ends at overlap >= 0.95; 100 reads per cell\n\n";
    out += "   q  mean bias  expected distance |  10 words   30 words  100 words  300 words\n";
    for q in [0.5, 0.6, 0.7, 0.8] {
        let mut r = Rng::new(40);
        let words: Vec<Vec<i8>> = (0..300).map(|_| leaning(256, q, &mut r)).collect();
        let b = balance(&words);
        let mut row = format!("{:>4.1}  {:>9.2}  {:>17.1} |", q, b.mean_bias, b.expected_distance);
        for load in [10usize, 30, 100, 300] {
            let mut sdm = Sdm::with_snr_radius(256, 10_000, 7);
            sdm.write_all(&words[..load]);
            let ok = (0..100).filter(|i| {
                let w = &words[i % load];
                overlap(&sdm.read(&add_address_noise(w, 0.1, &mut r)).word, w) >= 0.95
            }).count();
            row += &format!("  {:>9.2}", ok as f64 / 100.0);
        }
        out += &row;
        out += "\n";
    }
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
