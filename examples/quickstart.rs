//! KANERVA quickstart: build a memory, write words, read a noisy read-address back, predict the capacity,
//! decide whether to refuse, store a keyed note, and try the soft memory. Every call is the crate's front door
//! (`kanerva::Sdm` and friends).
//!
//! Run: `cargo run --release --example quickstart`

use kanerva::{add_address_noise, overlap, random_word, word_for, Refusal, Rng, Sdm, SoftSdm};

fn main() {
    // 1. A memory of 50,000 hard-locations of 256 bits; the activation-radius activates about 0.5% of them.
    let (n, m) = (256usize, 50_000usize);
    let mut sdm = Sdm::with_activation_probability(n, m, 0.005, 1);
    println!(
        "memory: {} hard-locations, word-size {} bits, activation-radius {} (activates {:.3}% of hard-locations)",
        sdm.hard_locations(),
        sdm.word_size(),
        sdm.activation_radius(),
        100.0 * sdm.activation_probability()
    );

    // 2. Write 300 random words.
    let mut rng = Rng::new(7);
    let words: Vec<Vec<i8>> = (0..300).map(|_| random_word(n, &mut rng)).collect();
    let activated = sdm.write_all(&words);
    println!("wrote {} words; each write activated {:.1} hard-locations on average", words.len(), activated as f64 / words.len() as f64);

    // 3. Read word 0 back from a read-address with 20% of its bits flipped.
    let read_address = add_address_noise(&words[0], 0.2, &mut rng);
    let read = sdm.read(&read_address);
    println!(
        "address read from a read-address with 20% address-noise: overlap {:.3} after {} iterated-reads ({} hard-locations activated, fixed point {})",
        overlap(&read.word, &words[0]),
        read.iterated_reads,
        read.activated,
        read.converged
    );

    // 4. The same read-address through the content-woken top-k read (the k rows whose bit-counters agree best).
    let k = (sdm.activation_probability() * m as f64).round() as usize;
    let topk = sdm.read_by_content(&read_address, 20, k);
    println!("top-k read (k = {}): overlap {:.3}", k, overlap(&topk.word, &words[0]));

    // 5. Negative control: deal the counter rows to other hard-locations; the memory is gone.
    let mut shuffled = sdm.clone();
    shuffled.store_mut().shuffle_rows(&mut Rng::new(3));
    println!("shuffled rows, same read-address: overlap {:.3}", overlap(&shuffled.read(&read_address).word, &words[0]));

    // 6. Theory: how many words can this memory hold and still recall 20%-noisy read-addresses?
    println!("S-map capacity at 20% address-noise: {} words", sdm.predicted_capacity(0.2));

    // 7. Refusal: accept an answer only if it lies within h bits of the read-address.
    let refusal = Refusal::for_memory(n, words.len(), 0.01);
    let travel = read.travelled(&read_address);
    println!(
        "refusal threshold h = {} bits; this answer travelled {} bits: {}",
        refusal.threshold,
        travel,
        if refusal.accepts(&read_address, &read) { "accept" } else { "refuse" }
    );

    // 8. A keyed note: the key finds the note and reads it; a wrong key reads nothing.
    let mut notes = Sdm::with_activation_probability(n, m, 0.005, 2);
    notes.write_note("blue heron", "under the mat");
    notes.write(&word_for("cat", n));
    println!("keyed note, right key: {:?}; wrong key: {:?}", notes.read_note("blue heron"), notes.read_note("red heron"));

    // 9. The soft memory: hard-locations as p-bits with a softness dial; softness 0 is the hard SDM.
    let mut soft = SoftSdm::new(256, 2000, 0.05, 0.25, 1);
    let sp: Vec<Vec<i8>> = (0..5).map(|i| word_for(&format!("s{}", i), 256)).collect();
    for w in &sp {
        soft.write(w);
    }
    let back = soft.read(&add_address_noise(&sp[0], 0.2, &mut rng), 3);
    println!(
        "soft memory (softness 0.25): overlap {:.3}; fitted softmax inverse temperature {:.2}",
        overlap(&back, &sp[0]),
        soft.attention_inverse_temperature()
    );
}
