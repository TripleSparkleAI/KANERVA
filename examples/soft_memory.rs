//! The soft activation-radius: as softness rises, each hard-location is activated with a probability that falls
//! smoothly with distance, and the memory's shared activation is fitted by a softmax attention whose inverse
//! temperature falls with it.
//!
//! Run: `cargo run --release --example soft_memory`

use kanerva::{add_address_noise, overlap, word_for, Rng, SoftSdm};

/// The example's whole output, so `tests/examples.rs` can check it against `soft_memory.out`.
pub fn run() -> String {
    let mut out = String::new();
    let words: Vec<Vec<i8>> = (0..5).map(|i| word_for(&format!("s{}", i), 256)).collect();
    out += "softness   recall overlap (20% address-noise)   attention inverse temperature\n";
    for softness in [0.0, 0.25, 0.5, 1.0] {
        let mut soft = SoftSdm::new(256, 2_000, 0.05, softness, 1);
        for w in &words {
            soft.write(w);
        }
        let mut rng = Rng::new(9);
        let back = soft.read(&add_address_noise(&words[0], 0.2, &mut rng), 3);
        out += &format!("{:>8}   {:>8.3}                              {:.2}\n", softness, overlap(&back, &words[0]), soft.attention_inverse_temperature());
    }
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
