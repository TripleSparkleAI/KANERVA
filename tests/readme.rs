//! Every Rust snippet in README.md compiles and runs, and the README's printed quickstart output is the
//! example's own.
//!
//! <claudes_code_comments>
//! ** Function List **
//! the_readme_snippets_are_this_file          - each ```rust block of README.md appears here verbatim
//! the_readme_output_is_the_quickstarts       - the "Example output" block equals examples/quickstart.out
//! the_quickstart_snippet_prints_what_it_says - the Quickstart snippet, run, with the numbers its prose quotes
//! every_lab_snippet_compiles_and_runs        - the "Using it" snippets, run in order with stand-in data
//! the_rails_snippet_prints_what_it_says      - the top snippet (the builder) prints the line the README and hello.out hold
//! the_rails_snippet_comments_are_the_hello_program - each comment of the top snippet is a line of programs/hello.kanerva
//! the_quoted_files_are_the_files              - the README's hello.kanerva, its output and two example printouts equal their files
//!
//! ** Technical Review **
//! - The Quickstart snippet stands alone and runs in its own test. The lab snippets share names: the first two
//!   define `p`, `read_address`, `r`, `rng`, `st` and `k`, which later ones use, so those two run at the top of
//!   their test and every later snippet runs in its own block.
//! - Stand-in data for the names the README says are yours: `patterns` (20 random patterns),
//!   `read_address_f64` and `p_f64` (the same patterns as f64), `my_counters` (a zero row per hard-location),
//!   `stored` (one pattern).
//! - The README cannot drift from code that compiles: a changed snippet fails the first test.
//!
//! </claudes_code_comments>
#![allow(unused_variables, unused_mut, clippy::needless_range_loop)]

#[test]
fn the_readme_snippets_are_this_file() {
    let readme = include_str!("../README.md");
    let here = include_str!("readme.rs");
    let mut n = 0;
    for block in readme.split("```rust\n").skip(1) {
        let code = block.split("```").next().unwrap();
        // this file holds each snippet indented by 4 or 8 spaces
        let found = [4, 8].iter().any(|&w| {
            let pad = " ".repeat(w);
            let indented: String = code.lines().map(|l| if l.is_empty() { "\n".to_string() } else { format!("{}{}\n", pad, l) }).collect();
            here.contains(&indented)
        });
        assert!(found, "README snippet not in tests/readme.rs:\n{}", code);
        n += 1;
    }
    assert_eq!(n, 9, "README.md has 9 rust snippets");
}

#[test]
fn the_readme_output_is_the_quickstarts() {
    let readme = include_str!("../README.md");
    let section = readme.split("## Example output").nth(1).expect("an Example output section");
    let block = section.split("```text\n").nth(1).unwrap().split("```").next().unwrap();
    assert_eq!(block, include_str!("../examples/quickstart.out"));
}

#[test]
fn the_quickstart_snippet_prints_what_it_says() {
    use kanerva::{add_address_noise, hamming, random_word, Rng, Sdm};

    let mut rng = Rng::new(7);
    let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1); // word-size, hard-locations, p, seed
    let words: Vec<Vec<i8>> = (0..20).map(|_| random_word(256, &mut rng)).collect();
    sdm.write_all(&words);
    let read_address = add_address_noise(&words[0], 0.15, &mut rng); // flip about 15% of the bits
    let read = sdm.read(&read_address);
    assert_eq!(read.word, words[0]);
    println!("{} bits wrong in, {} out", hamming(&read_address, &words[0]), hamming(&read.word, &words[0]));

    // the prose under the snippet quotes these two numbers
    assert_eq!(hamming(&read_address, &words[0]), 38);
    assert_eq!(read.iterated_reads, 2);
}

#[test]
fn every_lab_snippet_compiles_and_runs() {
    // stand-in data for the names the README says are the reader's own
    let mut seed_rng = kanerva::Rng::new(11);
    let patterns: Vec<Vec<i8>> = (0..20).map(|_| kanerva::bits::random_pattern(256, &mut seed_rng)).collect();
    let my_counters = vec![vec![0.0f64; 256]; 2000];

    // the README's `rng`, `bits`, `theory` snippet
    use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap, pack, hd}, theory::{radius_for, ball}};

    let mut rng = Rng::new(7);
    let p = random_pattern(256, &mut rng);                    // Vec<i8> of -1 and +1
    let read_address = add_address_noise(&p, 0.2, &mut rng);  // each bit flipped with probability 0.2
    let dist = hd(&pack(&p), &pack(&read_address));           // Hamming distance on packed words
    assert!((overlap(&p, &read_address) - (1.0 - 2.0 * dist as f64 / 256.0)).abs() < 1e-12);
    let r = radius_for(256, 0.005);                           // smallest r activating at least 0.5% of hard-locations
    assert!(ball(256, r) >= 0.005);

    // the README's `store` snippet
    use kanerva::store::Store;

    let mut st = Store::new(256, 50_000, r, 1);         // n, M, activation-radius, address seed
    st.write(&p);                                       // returns how many hard-locations took the write
    st.write_many(&patterns);                           // same result as one by one; activated sets in parallel
    let out = st.read_addresses(&read_address, 20);     // Kanerva's read, at most 20 iterated-reads
    println!("{} {} {}", overlap(&out.z, &p), out.rounds, out.fixed);
    let k = (ball(256, r) * 50_000.0).round() as usize;
    let top = st.read_pulls_topk(&read_address, 20, k); // content-woken read, the k best-matching rows
    let thr = st.read_pulls(&read_address, 20);         // content-woken read, threshold 0.4 n
    let mut control = st.clone();
    control.shuffle_rows(&mut Rng::new(3));             // negative control: the memory is gone

    let read_address_f64: Vec<f64> = read_address.iter().map(|&v| v as f64).collect();
    let p_f64: Vec<f64> = p.iter().map(|&v| v as f64).collect();
    let stored = vec![p_f64.clone()];

    // the README's `address` snippet
    {
        use kanerva::address::{Addresses, iterated_read};

        let a = Addresses::named("s", 1, 256, 2000);        // the same name and seed always give the same matrix
        let active = a.awake(&read_address_f64, r);         // the hard-locations within r of the state
        let (z, reads, activated) = iterated_read(&a, r, &read_address_f64, 10, |i, sum| {
            for j in 0..256 { sum[j] += my_counters[i][j]; }
        });
    }
    // the README's `smap` snippet
    {
        use kanerva::smap::{Lazy, SMap, goal, radius_for_address_noise, search_window};

        let l = Lazy::new(256, 100_000, 103);
        let cap = l.capacity(77, goal(256));  // largest T whose map from 77 of 256 bits of address-noise reaches overlap 0.95
        let dcrit = SMap::new(1000, 1_000_000, 451).critical(10_000);
        let (lo, hi) = search_window(256, 100_000);
        let (r30, _) = radius_for_address_noise(256, 100_000, 0.3, lo, hi);
        let p30 = Lazy::averaged(256, 100_000, r30).p_converge_flips(0.3, 1000, goal(256));
        let race = l.p_converge_race(0.4, 5, goal(256), 3000, 1);
    }
    // the README's `soft` snippet
    {
        use kanerva::soft::{Machine, attention_read, Attn};

        // n, M, activation-probability, softness, gain, seed, write samples
        let mut mc = Machine::new(256, 2000, 0.05, 0.25, 64.0, 1, 16);
        mc.write(&p_f64, &mut rng);
        let back = mc.recall(&read_address_f64, 3, 16, false, 0, &mut rng); // 3 one-way passes, 16 samples each
        let mf = mc.recall_mean_field(&read_address_f64, 3);                // infinitely many samples
        let kern = mc.kernel_inf();
        let beta = Machine::fit_beta(&kern);
        let att = attention_read(&read_address_f64, &stored, &Attn::Softmax(beta));
    }
    // the README's `keys` snippet
    {
        use kanerva::keys::{keyed_pattern, keyed_read_address, keyed_read};

        let note = keyed_pattern("blue heron", "under the mat", 256); // write this into any memory
        let key_address = keyed_read_address("blue heron", 256);      // read from this
        assert_eq!(keyed_read("blue heron", &note).as_deref(), Some("under the mat"));
        assert_eq!(keyed_read("red heron", &note), None);
    }
    // the README's `refuse` and `track` snippet
    {
        use kanerva::refuse::{travel_threshold, oracle_point, Fast};
        use kanerva::track::{Content, Wake};

        let h = travel_threshold(256, 3000, 0.01);       // accept an answer only within h bits of the read-address
        let (recall, no_refusal, refusal) = oracle_point(256, 0.4, 3000, h);
        let diag = Fast::new(&st).topk(&read_address, 20, k); // the answer plus first-read and last-read signals
        let c = Content::new(256, 100_000, 103);
        let pc = c.p_converge(Wake::Topk(c.k()), 0.3, 3000, 400, true, 1);
    }
}

#[test]
fn the_rails_snippet_prints_what_it_says() {
    let run = || -> Result<String, kanerva::rails::Error> {
        use kanerva::rails::{Sdm, Via};

        let mut s = Sdm::build().word_size(256).hard_locations(2000).seed(1); // sdm :s, word-size: 256, hard-locations: 2000
        s.write("cat")?;                                                       // s.write :cat
        s.write("dog")?;                                                       // s.write :dog
        s.write("owl")?;                                                       // s.write :owl
        s.write_text("note", "meet at nine")?;                                 // s.write :note, "meet at nine"
        let read = s.read("cat").address_noise(0.2).seed(1).answer()?;         // s.read read-address: :cat, address-noise: 0.2, seed: 1
        assert_eq!(read.recalled.as_deref(), Some("cat"));
        let pulls = s.read("cat").address_noise(0.2).via(Via::Pulls).seed(1).answer()?;
        println!("{read}");
        assert_eq!(pulls.recalled.as_deref(), Some("cat"));
        Ok(read.to_string())
    };
    let line = run().unwrap();
    let readme = include_str!("../README.md");
    assert!(readme.contains(&format!("It prints:\n\n```text\n{}\n```", line)), "the README's printed Rails line differs: {}", line);
    // the prose under it quotes the read's count of woken hard-locations
    let woke = line.split(" of 2000 hard locations").next().unwrap().rsplit(' ').next().unwrap();
    assert!(readme.contains(&format!("The read woke {} of the 2,000 hard-locations", woke)), "the README's count of woken hard-locations");
    // the file face prints the same line: programs/hello.kanerva holds the snippet's lines as a file
    assert_eq!(include_str!("../programs/hello.out").lines().next(), Some(line.as_str()));
}

// the top snippet's comments are the lines of programs/hello.kanerva, in order, each one a line the file holds
#[test]
fn the_rails_snippet_comments_are_the_hello_program() {
    let readme = include_str!("../README.md");
    let snippet = readme.split("```rust\n").nth(1).unwrap().split("```").next().unwrap();
    let comments: Vec<&str> = snippet.lines().filter_map(|l| l.split_once("// ").map(|(_, c)| c.trim())).collect();
    let hello: Vec<&str> = include_str!("../programs/hello.kanerva").lines().map(|l| l.trim()).collect();
    assert_eq!(comments.len(), 6, "the snippet's six commented lines");
    let mut at = 0;
    for c in &comments {
        let found = hello[at..].iter().position(|l| l == c).unwrap_or_else(|| panic!("not a line of hello.kanerva, in order: {}", c));
        at += found + 1;
    }
}

// the README quotes programs/hello.kanerva and its output, and two example printouts; each must equal its file
#[test]
fn the_quoted_files_are_the_files() {
    let readme = include_str!("../README.md");
    let face = readme.split("## The file face").nth(1).expect("a file face section");
    let program = face.split("```kanerva\n").nth(1).unwrap().split("```").next().unwrap();
    assert_eq!(program, include_str!("../programs/hello.kanerva"));
    let printed = face.split("```text\n").nth(1).unwrap().split("```").next().unwrap();
    assert_eq!(printed, include_str!("../programs/hello.out"));
    for (name, out) in [("content_read", include_str!("../examples/content_read.out")), ("capacity", include_str!("../examples/capacity.out"))] {
        let after = readme.split(&format!("`{}` prints:", name)).nth(1).unwrap_or_else(|| panic!("the README quotes {}", name));
        let block = after.split("```text\n").nth(1).unwrap().split("```").next().unwrap();
        assert_eq!(block, out, "README's copy of examples/{}.out", name);
    }
}
