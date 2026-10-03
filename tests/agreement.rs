//! Cross-module agreement through the public API: the two implementations of Kanerva's read in this crate
//! (counters held by the caller via `address::iterated_read`, and the byte store `store::Store`) give the
//! same answer bit for bit; the theory predicts what the store does; and the negative controls fail.

use kanerva::address::{iterated_read, Addresses};
use kanerva::bits::{add_address_noise, overlap, random_pattern};
use kanerva::refuse::{oracle_point, travel_threshold};
use kanerva::smap::{goal, Lazy};
use kanerva::store::Store;
use kanerva::theory::{ball, radius_for};
use kanerva::Rng;

#[test]
fn caller_held_counters_and_the_byte_store_agree_bit_for_bit() {
    let (n, m) = (64usize, 400usize);
    let r = radius_for(n, 0.05);
    let addr = Addresses::named("s", 3, n, m);
    let mut store = Store::like_view("s", n, m, r, 3);
    let mut counters = vec![vec![0.0f64; n]; m];
    let mut rng = Rng::new(11);
    let pats: Vec<Vec<i8>> = (0..12).map(|_| random_pattern(n, &mut rng)).collect();
    for p in &pats {
        let pf: Vec<f64> = p.iter().map(|&v| v as f64).collect();
        let woken = addr.awake(&pf, r);
        for &i in &woken {
            for j in 0..n {
                counters[i][j] += pf[j];
            }
        }
        assert_eq!(store.write(p), woken.len());
    }
    for i in 0..m {
        for j in 0..n {
            assert_eq!(counters[i][j] as i8, store.counter(i, j));
        }
    }
    for (k, p) in pats.iter().enumerate() {
        let cue = add_address_noise(p, 0.2, &mut Rng::new(100 + k as u64));
        let cf: Vec<f64> = cue.iter().map(|&v| v as f64).collect();
        let (z, reads, awake) = iterated_read(&addr, r, &cf, 10, |i, s| s.iter_mut().zip(&counters[i]).for_each(|(a, b)| *a += b));
        let o = store.read_addresses(&cue, 10);
        assert_eq!(z, o.z.iter().map(|&v| v as f64).collect::<Vec<_>>());
        assert_eq!((reads, awake), (o.rounds, o.awake));
    }
}

#[test]
fn the_smap_capacity_is_reached_and_far_overload_fails() {
    // the S-map's capacity from a 10%-damaged cue, written into a real store at a fifth of that load,
    // recalls at least 90% of probes; at 30 times the capacity it recalls almost none
    let (n, m) = (256usize, 20_000usize);
    let r = radius_for(n, 0.005);
    let cap = Lazy::new(n, m, r).capacity(26, goal(n));
    let recall = |t: usize, seed: u64| {
        let mut st = Store::new(n, m, r, seed);
        let mut rng = Rng::new(seed);
        let pats: Vec<Vec<i8>> = (0..t).map(|_| random_pattern(n, &mut rng)).collect();
        st.write_many(&pats);
        let q = 40.min(t);
        pats.iter().take(q).filter(|p| overlap(&st.read_addresses(&add_address_noise(p, 0.1, &mut rng), 20).z, p) >= 0.95).count() as f64 / q as f64
    };
    assert!(recall(cap / 5, 1) >= 0.9, "cap {}", cap);
    assert!(recall(cap * 30, 2) <= 0.1, "cap {}", cap);
}

#[test]
fn the_refusal_threshold_holds_its_level_on_random_cues() {
    // with T random stored patterns, a random cue has a stored pattern within h bits at most 1% of the time
    let (n, t) = (256usize, 200usize);
    let h = travel_threshold(n, t, 0.01);
    let mut rng = Rng::new(5);
    let pats: Vec<Vec<i8>> = (0..t).map(|_| random_pattern(n, &mut rng)).collect();
    let near = (0..2000)
        .filter(|_| {
            let c = random_pattern(n, &mut rng);
            pats.iter().any(|p| (n as f64 * (1.0 - overlap(&c, p)) / 2.0) as usize <= h)
        })
        .count();
    assert!(near <= 40, "{} of 2000", near);
    // and the oracle with that threshold still names a lone stored pattern from a 10%-damaged cue
    let (rec, _, _) = oracle_point(n, 0.1, 1, h);
    assert!(rec > 0.99);
    assert!(ball(n, h) < 0.01);
}
