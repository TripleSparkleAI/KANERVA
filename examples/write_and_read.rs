//! Write words into a sparse distributed memory and read one back from a noisy read-address.
//!
//! Run: `cargo run --release --example write_and_read`

use kanerva::{add_address_noise, hamming, random_word, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `write_and_read.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(7);
    // 2,000 hard-locations of 256 bits; each address activates about 2% of them
    let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1);
    let words: Vec<Vec<i8>> = (0..20).map(|_| random_word(256, &mut rng)).collect();
    sdm.write_all(&words);
    out += &format!("{} words written into {} hard-locations, activation-radius {} bits\n", sdm.written(), sdm.hard_locations(), sdm.activation_radius());

    let read_address = add_address_noise(&words[0], 0.15, &mut rng);
    let read = sdm.read(&read_address);
    out += &format!("read-address: {} of 256 bits wrong\n", hamming(&read_address, &words[0]));
    out += &format!("answer: {} of 256 bits wrong, after {} iterated-reads\n", hamming(&read.word, &words[0]), read.iterated_reads);
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
