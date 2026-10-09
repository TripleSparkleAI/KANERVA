//! Saying "I never stored that": a read whose answer travels far from its read-address is refused. Stored words
//! read from noisy read-addresses travel a few bits; random read-addresses that were never written travel far.
//!
//! Run: `cargo run --release --example refusal`

use kanerva::{add_address_noise, random_word, Refusal, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `refusal.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(13);
    let mut sdm = Sdm::with_activation_probability(256, 10_000, 0.01, 1);
    let words: Vec<Vec<i8>> = (0..100).map(|_| random_word(256, &mut rng)).collect();
    sdm.write_all(&words);
    let refusal = Refusal::for_memory(256, words.len(), 0.01);
    out += &format!("{} words stored; refuse an answer that travels more than {} bits\n", words.len(), refusal.threshold);
    let (mut accepted, mut refused) = (0, 0);
    for w in words.iter().take(50) {
        let ra = add_address_noise(w, 0.2, &mut rng);
        if refusal.accepts(&ra, &sdm.read(&ra)) { accepted += 1 } else { refused += 1 }
    }
    out += &format!("stored words, 20% address-noise: {} accepted, {} refused\n", accepted, refused);
    let (mut accepted, mut refused) = (0, 0);
    for _ in 0..50 {
        let ra = random_word(256, &mut rng);
        if refusal.accepts(&ra, &sdm.read(&ra)) { accepted += 1 } else { refused += 1 }
    }
    out += &format!("never-stored read-addresses:     {} accepted, {} refused\n", accepted, refused);
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
