//! Notes only their key can read: write a few text notes beside plain words, read each back with its key, and
//! read with a wrong key, which returns nothing.
//!
//! Run: `cargo run --release --example keyed_notes`

use kanerva::{word_for, Sdm};

/// The example's whole output, so `tests/examples.rs` can check it against `keyed_notes.out`.
pub fn run() -> String {
    let mut out = String::new();
    let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 2);
    for name in ["cat", "owl", "zebra"] {
        sdm.write(&word_for(name, 256));
    }
    let notes = [("blue heron", "under the mat"), ("north gate", "meet at nine"), ("tin box", "behind the clock")];
    for (key, text) in notes {
        sdm.write_note(key, text);
    }
    for (key, _) in notes {
        out += &format!("{:<11} -> {:?}\n", key, sdm.read_note(key));
    }
    out += &format!("{:<11} -> {:?}\n", "red heron", sdm.read_note("red heron"));
    out += "a note is not encrypted: the key is a 64-bit FNV-1a hash and can be guessed offline\n";
    out
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
