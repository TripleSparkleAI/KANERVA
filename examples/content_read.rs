//! The two reads side by side: Kanerva's address read activates the hard-locations near the read-address; the
//! content-woken read lets the k hard-locations whose bit-counters agree most with it vote. At heavy load and
//! heavy address-noise the content read recalls words the address read loses.
//!
//! Run: `cargo run --release --example content_read`

use kanerva::{add_address_noise, overlap, random_word, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `content_read.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(5);
    let mut sdm = Sdm::with_activation_probability(256, 10_000, 0.01, 1);
    let words: Vec<Vec<i8>> = (0..300).map(|_| random_word(256, &mut rng)).collect();
    sdm.write_all(&words);
    let k = (sdm.activation_probability() * sdm.hard_locations() as f64).round() as usize;
    out += &format!("{} words in {} hard-locations; content read uses the top k = {} rows\n", words.len(), sdm.hard_locations(), k);
    out += "address-noise   address read   content read   (recalled of 30, overlap >= 0.95)\n";
    for pct in [10, 20, 30] {
        let (mut a, mut c) = (0, 0);
        for w in words.iter().take(30) {
            let ra = add_address_noise(w, pct as f64 / 100.0, &mut rng);
            a += (overlap(&sdm.read(&ra).word, w) >= 0.95) as usize;
            c += (overlap(&sdm.read_by_content(&ra, 20, k).word, w) >= 0.95) as usize;
        }
        out += &format!("{:>9}%      {:>6}         {:>6}\n", pct, a, c);
    }
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
