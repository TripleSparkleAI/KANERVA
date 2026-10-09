//! Saying "I never stored that" when the store is crowded. The travel rule ([`kanerva::Refusal`]) refuses an
//! answer that moved far from its read-address; in a crowded store a never-stored read-address read by content
//! barely moves, so the travel rule lets it through. The calibrated refusal compares each read with reads from
//! random read-addresses the memory never stored, and keeps refusing (SETTLE lane SDMTRACK's calibrated-minimum
//! rule). The read is the content read with the measured k.
//!
//! Run: `cargo run --release --example calibrated_refusal`

use kanerva::{add_address_noise, overlap, random_word, Refusal, Rng, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `calibrated_refusal.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut rng = Rng::new(21);
    let words: Vec<Vec<i8>> = (0..2_000).map(|_| random_word(256, &mut rng)).collect();
    let strangers: Vec<Vec<i8>> = (0..200).map(|_| random_word(256, &mut rng)).collect();
    out += "n 256, M 10,000, the SNR activation-radius, the content read; 200 never-stored read-addresses; 60 stored words per noise level\n";
    out += "kept = accepted and read back at overlap >= 0.95\n";
    for load in [50usize, 2_000] {
        let mut sdm = Sdm::with_snr_radius(256, 10_000, 3);
        sdm.write_all(&words[..load]);
        let travel = Refusal::for_memory(256, load, 0.01);
        let rule = sdm.calibrated_content_refusal(1_000, 0.01, 5);
        let (mut t_refused, mut c_refused) = (0, 0);
        for s in &strangers {
            let checked = sdm.read_checked(&rule, s);
            t_refused += !travel.accepts(s, &checked.read) as usize;
            c_refused += !checked.accepted as usize;
        }
        out += &format!("\n{} words stored (activation-radius {}, k {}):\n", load, sdm.activation_radius(), sdm.content_k());
        out += &format!("  never-stored refused: travel rule {} of 200, calibrated {} of 200\n", t_refused, c_refused);
        for noise in [0.1, 0.2, 0.3] {
            let (mut t_kept, mut c_kept) = (0, 0);
            for w in words[..load].iter().cycle().take(60) {
                let ra = add_address_noise(w, noise, &mut rng);
                let checked = sdm.read_checked(&rule, &ra);
                let good = overlap(&checked.read.word, w) >= 0.95;
                t_kept += (good && travel.accepts(&ra, &checked.read)) as usize;
                c_kept += (good && checked.accepted) as usize;
            }
            out += &format!("  {:.0}% address-noise kept: travel rule {} of 60, calibrated {} of 60\n", noise * 100.0, t_kept, c_kept);
        }
    }
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
