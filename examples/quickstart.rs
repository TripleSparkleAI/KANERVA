//! KANERVA quickstart: build a memory, write patterns, read a noisy read-address back, predict the capacity,
//! decide whether to refuse, and store a keyed note.
//!
//! Run: `cargo run --release --example quickstart`

use kanerva::bits::{add_address_noise, hd, overlap, pack, random_pattern};
use kanerva::codes::code;
use kanerva::keys::{keyed_read_address, keyed_pattern, keyed_read};
use kanerva::refuse::travel_threshold;
use kanerva::smap::{goal, Lazy};
use kanerva::soft::Machine;
use kanerva::store::Store;
use kanerva::theory::{ball, radius_for};
use kanerva::Rng;

fn main() {
    // 1. A memory of 50,000 hard locations of 256 bits; the activation radius wakes about 0.5% of them.
    let (n, m) = (256usize, 50_000usize);
    let r = radius_for(n, 0.005);
    let mut store = Store::new(n, m, r, 1);
    println!("memory: {} hard locations, word-size {} bits, activation radius {} (activates {:.3}% of hard locations)", m, n, r, 100.0 * ball(n, r));

    // 2. Write 300 random patterns.
    let mut rng = Rng::new(7);
    let pats: Vec<Vec<i8>> = (0..300).map(|_| random_pattern(n, &mut rng)).collect();
    let woken = store.write_many(&pats);
    println!("wrote {} patterns; each write activated {:.1} hard locations on average", pats.len(), woken as f64 / pats.len() as f64);

    // 3. Read pattern 0 back from a read-address with 20% of its bits flipped.
    let cue = add_address_noise(&pats[0], 0.2, &mut rng);
    let out = store.read_addresses(&cue, 20);
    println!(
        "address read from a read-address with 20% address-noise: overlap {:.3} after {} iterated reads ({} hard locations activated, fixed point {})",
        overlap(&out.z, &pats[0]),
        out.rounds,
        out.awake,
        out.fixed
    );

    // 4. The same read-address through the content-woken top-k read (wake the k rows whose bit-counters match best).
    let k = (ball(n, r) * m as f64).round() as usize;
    let topk = store.read_pulls_topk(&cue, 20, k);
    println!("top-k read (k = {}): overlap {:.3}", k, overlap(&topk.z, &pats[0]));

    // 5. Negative control: shuffle the counter rows across hard locations; the memory is gone.
    let mut shuffled = store.clone();
    shuffled.shuffle_rows(&mut Rng::new(3));
    println!("shuffled rows, same read-address: overlap {:.3}", overlap(&shuffled.read_addresses(&cue, 20).z, &pats[0]));

    // 6. Theory: how many patterns can this memory hold and still recall 20%-noisy read-addresses?
    let smap = Lazy::new(n, m, r);
    println!("S-map capacity at 20% address-noise: {} patterns", smap.capacity((0.2 * n as f64) as usize, goal(n)));

    // 7. Refusal: accept an answer only if it lies within h bits of the read-address.
    let h = travel_threshold(n, pats.len(), 0.01);
    let travel = hd(&pack(&cue), &pack(&out.z));
    println!("refusal threshold h = {} bits; this answer travelled {} bits: {}", h, travel, if travel <= h { "accept" } else { "refuse" });

    // 8. A keyed note: the key finds the note and reads it; a wrong key reads nothing.
    let mut notes = Store::new(n, m, r, 2);
    notes.write(&to_i8(&keyed_pattern("blue heron", "under the mat", n)));
    notes.write(&to_i8(&code("cat", n)));
    let got = notes.read_addresses(&to_i8(&keyed_read_address("blue heron", n)), 20);
    let z: Vec<f64> = got.z.iter().map(|&v| v as f64).collect();
    println!("keyed note, right key: {:?}; wrong key: {:?}", keyed_read("blue heron", &z), keyed_read("red heron", &z));

    // 9. The soft machine: hard locations as p-bits with a softness dial; softness 0 is the hard SDM.
    let mut soft = Machine::new(256, 2000, 0.05, 0.25, 64.0, 1, 16);
    let sp: Vec<Vec<f64>> = (0..5).map(|i| code(&format!("s{}", i), 256)).collect();
    for p in &sp {
        soft.write(p, &mut rng);
    }
    let cue = kanerva::codes::with_address_noise(&sp[0], 0.2, &mut rng);
    let back = soft.recall(&cue, 3, 16, false, 0, &mut rng);
    println!("soft machine (softness 0.25): overlap {:.3}; fitted softmax inverse temperature {:.2}", kanerva::codes::overlap(&back, &sp[0]), Machine::fit_beta(&soft.kernel_inf()));
}

fn to_i8(p: &[f64]) -> Vec<i8> {
    p.iter().map(|&v| if v > 0.0 { 1 } else { -1 }).collect()
}
