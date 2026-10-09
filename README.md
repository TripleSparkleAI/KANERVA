<!-- settle-banner -->
```text
░   ░   ░░░   ░   ░  ░░░░░  ░░░░   ░   ░   ░░░
▒  ▒   ▒   ▒  ▒▒  ▒  ▒      ▒   ▒  ▒   ▒  ▒   ▒
▓▓▓    ▓▓▓▓▓  ▓ ▓ ▓  ▓▓▓▓   ▓▓▓▓   ▓   ▓  ▓▓▓▓▓
█  █   █   █  █  ██  █      █  █    █ █   █   █
█   █  █   █  █   █  █████  █   █    █    █   █

✦ sparse distributed memory in Rust, like Redis but for SDM
```

# KANERVA

It's like Redis! But for Sparse Distributed Memory!

KANERVA stores long binary words and reads them back from an address with many of its bits wrong. Four words go
in, and a copy of `cat` with a fifth of its bits flipped brings `cat` back:

```rust
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
```

It prints:

```text
read :s from read-address :cat with 20% address-noise via addresses (2 iterated reads, 48 of 2000 hard locations activated): :cat +1.00  :owl -0.09  :note +0.09  -> :cat
```

The read woke 48 of the 2,000 hard-locations, they voted bit by bit, and the answer is `cat` exactly. The comment on
each line is the same line in a `.kanerva` file: `programs/hello.kanerva` holds those lines, and
`cargo run --release --bin kanerva -- programs/hello.kanerva` prints the same line. SETTLE runs the same file and
prints it too. One word list (`src/words.rs`), three faces: the Rust builder, the `kanerva` command, and SETTLE.

KANERVA is a Rust library for Pentti Kanerva's sparse distributed memory (1988). It needs nothing else to run: no
SETTLE, no other crate, no dependencies. Besides the reads, it predicts how much a memory will hold before you build
it, and says when an answer should be refused. The **engine** is the lab modules: data in, data out, no parsing and no
printing. The **words** are the one word list, spoken by the builder in `src/rails.rs` and by `.kanerva` files, and a
test holds each face equal to the list in both directions. SETTLE is one user of KANERVA, and KANERVA never depends
on it.

The crate uses Kanerva's own words, joined by a hyphen where his word is two: read-address, address-noise,
hard-locations, activation-radius, activation-probability, word-size, bit-counters, iterated-reads,
critical-distance. Each one, with Kanerva's sentence and page, is in [KANERVA_TERMS.md](KANERVA_TERMS.md).

> **The repository is `github.com/TripleSparkleAI/KANERVA`, and it is public.** Anyone can clone it or name it as a
> git dependency. The SETTLE interpreter (`github.com/TripleSparkleAI/SETTLE`) depends on it by its git URL.
>
> **Licence: MIT.** The text is in `LICENSE`.

Version 0.1.0. Rust 1.87 or newer. Every public item carries an example that runs as a test, and every example's
printout is recorded and checked (`ls src examples` for the current set).

## Install

As a git dependency:

```bash
cargo add kanerva --git https://github.com/TripleSparkleAI/KANERVA
```

From a clone:

```bash
git clone https://github.com/TripleSparkleAI/KANERVA && cd KANERVA && cargo test --release
```

Or by hand, under `[dependencies]` in your `Cargo.toml`:

```toml
kanerva = { git = "https://github.com/TripleSparkleAI/KANERVA" }
```

A crate in a folder beside a clone can name it by path instead: `kanerva = { path = "../KANERVA" }`.

## Quickstart

```rust
use kanerva::{add_address_noise, hamming, random_word, Rng, Sdm};

let mut rng = Rng::new(7);
let mut sdm = Sdm::with_activation_probability(256, 2_000, 0.02, 1); // word-size, hard-locations, p, seed
let words: Vec<Vec<i8>> = (0..20).map(|_| random_word(256, &mut rng)).collect();
sdm.write_all(&words);
let read_address = add_address_noise(&words[0], 0.15, &mut rng); // flip about 15% of the bits
let read = sdm.read(&read_address);
assert_eq!(read.word, words[0]);
println!("{} bits wrong in, {} out", hamming(&read_address, &words[0]), hamming(&read.word, &words[0]));
```

It prints `38 bits wrong in, 0 out`: the read-address had 38 of its 256 bits wrong, and two iterated-reads
brought back the stored word exactly. `examples/write_and_read.rs` is the same program.

## What a sparse distributed memory is, in plain words

Picture a million mailboxes. Each mailbox has a fixed random address of 256 bits and a row of 256
bit-counters. To store a 256-bit pattern, you find every mailbox whose address is close to the pattern (a
few hundred out of the million) and add the pattern to their bit-counters: +1 where the pattern has a 1,
-1 where it has a 0. To read, you take a read-address, find the mailboxes close to the read-address, add up their counter
rows, and keep the sign of each column. If the read-address is a noisy copy of a stored pattern, many of the
mailboxes it opens are mailboxes the pattern was written to, their votes agree, and the read comes back
closer to the pattern than the read-address was. Reading again from the result moves closer still.

No single mailbox holds a pattern. Every pattern is spread over hundreds of mailboxes, and every
mailbox holds pieces of many patterns. That is the "distributed" in the name. Only a small fraction of
mailboxes take part in any one write or read. That is the "sparse".

Kanerva described this memory in *Sparse Distributed Memory* (MIT Press, 1988).

## Kanerva's words, and where they live in the crate

| Kanerva's word | what it is | in the crate |
|---|---|---|
| hard-locations | the M memory locations that are built, each at a fixed random address | `Sdm::hard_locations()` |
| word-size | the bits in an address and in a stored word, n | `Sdm::word_size()` |
| activation-radius | the Hamming distance within which an address activates a hard-location, r | `Sdm::new(n, M, r, seed)`, `Sdm::activation_radius()` |
| activation-probability | the share of hard-locations one address activates, p | `Sdm::with_activation_probability(n, M, p, seed)`, `Sdm::activation_probability()` |
| access circle | the hard-locations an address activates | `Sdm::activated(address)` |
| bit-counters | the integer counters each hard-location keeps, one per bit | `Sdm::store().row(i)` |
| read-address | the address a read starts from | the argument of `Sdm::read` |
| address-noise | the share of a read-address's bits flipped at random | `add_address_noise(word, share, rng)` |
| iterated-reads | reading again from the last answer until it stops changing | `Sdm::read_iterated`, `ReadResult::iterated_reads` |
| read-threshold | a summed bit-counter above 0 reads +1, below 0 reads -1 (0 keeps the bit) | inside every read |
| critical-distance | how far a read-address can start and still come back to its word | `Sdm::critical_distance(stored)` (a prediction) |
| best-match | the stored word nearest a given word | `refuse::oracle_point`, `bits::nearest` |

Two words are the crate's own. The **content-woken read** (`Sdm::read_by_content`) activates hard-locations by
what their bit-counters hold rather than by where their addresses sit. The **travel rule** (`Refusal`) refuses
an answer that moved too far from its read-address.

## The front door

Everything at the crate root is one small API, in `src/sdm.rs`, with a few more methods on `Sdm` from
`src/sizing.rs`, `src/calibrated.rs` and `src/erase.rs`:

| item | what it does |
|---|---|
| `Sdm::new(word_size, hard_locations, activation_radius, seed)` | a memory with fixed random addresses from `seed` |
| `Sdm::with_activation_probability(word_size, hard_locations, p, seed)` | the same, choosing the smallest activation-radius that activates at least p of all addresses |
| `sdm.write(&word)` / `sdm.write_all(&words)` | add each word to the bit-counters of its access circle; returns the hard-locations that took it |
| `sdm.read(&read_address)` | Kanerva's read, iterated up to `DEFAULT_ITERATED_READS` (20) times or until the answer stops changing |
| `sdm.read_iterated(&read_address, max)` | the same with your own limit |
| `sdm.read_by_content(&read_address, max, k)` | the content-woken read: the k hard-locations whose bit-counters agree most with the word vote |
| `sdm.write_note(key, text)` / `sdm.read_note(key)` | a short text that only its key reads back (not cryptography) |
| `sdm.predicted_capacity(address_noise)` | how many words the signal-to-noise map predicts this memory holds at that address-noise |
| `sdm.critical_distance(stored)` | the critical-distance the same map predicts with `stored` words written |
| `Sdm::with_snr_radius(word_size, hard_locations, seed)` | the activation-radius for light address-noise at Kanerva's design load (measured: capacity grows about as M^0.96) |
| `Sdm::for_address_noise(word_size, hard_locations, address_noise, seed)` | the activation-radius tuned for the address-noise you expect |
| `sdm.predicted_failure_rate(address_noise, stored, samples, seed)` | the share of reads predictor TRACK expects to miss (measured error 0.015) |
| `sdm.predicted_capacity_track(address_noise, recall, samples, seed)` | the most words TRACK expects at that recall; use it rather than `predicted_capacity` at heavy noise |
| `sdm.content_k()` | the k for `read_by_content`: the expected access circle, activation-probability times M |
| `sdm.calibrated_refusal(probes, alpha, seed)` / `sdm.read_checked(&rule, &read_address)` | refuse never-stored read-addresses by comparing each read with reads from random probes; keeps working in a crowded store where the travel rule stops |
| `sdm.erase(&word)` / `sdm.read_without(&word, &read_address)` | take a written word out again, exactly; read as if it had never been written |
| `sdm.activated(&address)` | the access circle of an address |
| `sdm.store()` / `sdm.store_mut()` | the byte store underneath, for the lab and for negative controls |
| `ReadResult { word, iterated_reads, activated, converged }` | what one read returned; `.travelled(&from)` counts the bits it moved |
| `Refusal::for_memory(word_size, stored, level)` | the travel rule's threshold; `.accepts(&read_address, &read)` |
| `SoftSdm::new(word_size, hard_locations, p, softness, seed)` | the soft memory: hard-locations activated with a probability that falls with distance |
| `random_word`, `word_for(name, n)`, `hamming`, `add_address_noise`, `overlap`, `Rng` | words, their distances, and the one random generator |

Words are `Vec<i8>` of -1 and +1. The front door changes no arithmetic: each call is one call into the lab
modules below, and `src/sdm.rs`'s tests check it reads the same bits as the store it wraps.

## The Rails face

The same memory can be written three ways, with the same words: in a `.kanerva` file, in a SETTLE program's sdm
statements, and in Rust with the builder in `src/rails.rs`. The snippet at the top of this page is the builder. It is
the Rust spelling of the word list: each statement is a type, each keyword is a method of the same name with its
hyphens turned into underscores, and each verb is a method that writes or reads. Each of its lines carries, as a
comment, the line of `programs/hello.kanerva` it equals, and `println!` prints the line the file prints for the same
read.

| statement | type | settings (the keywords, plus `name`) | verbs |
|---|---|---|---|
| `sdm :s` | `rails::Sdm` | `word_size` `hard_locations` `activation_radius` `seed` `fade` | `write(name)`, `write_text(name, text)`, `write_keyed(name, text, key)`; `read(read_address)`, `read_key(key)`, `read_from_noise()`, each taking `.read_address` `.key` `.address_noise` `.iterated_reads` `.via` `.seed` |
| `softsdm :s` | `rails::SoftSdm` | `word_size` `hard_locations` `activation_probability` `softness` `gain` `seed` `write_samples` | `write`, `write_text`; `read(..)` with `.address_noise` `.rounds` `.samples` `.mode` `.burn` `.seed`; `attend(..)` with `.address_noise` `.rounds` `.seed` |
| `sdmscale :k` | `rails::SdmScale` | `word_size` `hard_locations` `activation_probability` `activation_radius` `tolerate_noise` `seed` | `put(name)`, `fill(count)`; `read(read_address)` with `.address_noise` `.iterated_reads` `.via` `.wake` `.seed` |
| `refusal` | `rails::Refusal` | `word_size` `load` `level` | `answer()` |
| `contenttrack` | `rails::ContentTrack` | `word_size` `hard_locations` `load` `address_noise` `block` `samples` | `answer()` |

How it behaves:

- **The answers are the file's answers.** A read returns an `Answer` (the word it ended on, the scoreboard, the
  name it recalled, any text read back, the reads made); printing it prints the file's line. `examples/rails.rs`
  re-reads SETTLE's `sdm.settle` and `softsdm.settle` and an sdmscale, refusal and contenttrack program through the
  builder, and its 23 lines equal SETTLE's own output byte for byte (recorded in `examples/rails.out`).
- **Defaults come from the word list.** `Sdm::build()` reads every default from `src/words.rs`, so a default is
  written in one place. A memory left unnamed is named `s`, the name the address seed is mixed with.
- **Settings are fixed at the first write or read.** That is when the memory is made and checked. A setting out
  of range is an `Error` in the file's words (`sdm word-size must be between 16 and 4096`); a setter called after
  the memory is made panics, because a memory cannot change shape under its contents.
- **A read draws noise from the memory's own stream,** which starts where a SETTLE run block starts it (seed
  0x5eed). `.seed(n)` restarts it, and later reads run on from there; `rewind()` starts it again. Memories that
  should share one stream, as the memories of one run block do, pass the same `Rng` with `.stream(&mut rng)`,
  the one method the files have no word for.
- **A `via: :pulls` read settles this memory alone.** In SETTLE the same read settles every thing in the model,
  so the two agree when the memory is the only thing declared.
- `memory` (the Hopfield memory) is SETTLE's own statement and has no type here.

The parity test, both ways: `tests/rails_words.rs` reads `src/rails.rs` and holds its marked methods and their
parameters equal to `src/words.rs` (every keyword has a method of its name, and every method is a keyword), with
four planted defects that each must be named. SETTLE's `tests/kanerva_rails.rs` holds the keywords its sdm
statements accept equal to the same list, and its printed lines equal to the builder's.

## Examples

One idea per example. Each keeps its whole output in a `run()` function, its printout is recorded in
`examples/<name>.out`, and `tests/examples.rs` fails if the two ever differ.

| example | the idea | run |
|---|---|---|
| `write_and_read` | write 20 words and read one back from a noisy read-address | `cargo run --release --example write_and_read` |
| `address_noise` | recall as the read-address gets noisier, beside the predicted critical-distance | `cargo run --release --example address_noise` |
| `capacity` | write more and more words and watch recall fall, beside the predicted capacity | `cargo run --release --example capacity` |
| `content_read` | the address read and the content-woken read side by side at heavy load | `cargo run --release --example content_read` |
| `keyed_notes` | notes that only their key reads back | `cargo run --release --example keyed_notes` |
| `refusal` | stored words are accepted, never-stored read-addresses are refused | `cargo run --release --example refusal` |
| `soft_memory` | recall and the attention inverse temperature as softness rises | `cargo run --release --example soft_memory` |
| `negative_controls` | shuffle the bit-counters and the memory forgets | `cargo run --release --example negative_controls` |
| `calibrated_refusal` | the travel rule and the calibrated refusal on a light and a crowded store | `cargo run --release --example calibrated_refusal` |
| `sizing` | the SNR radius against a radius tuned for 30% noise; TRACK and the S-map against measured recall | `cargo run --release --example sizing` |
| `bit_balance` | words whose bits lean one way, and how few of them the memory holds | `cargo run --release --example bit_balance` |
| `rails` | SETTLE's sdm-family programs written with the builder, printing SETTLE's own lines | `cargo run --release --example rails` |
| `quickstart` | the whole tour in one program; its output is under "Example output" below | `cargo run --release --example quickstart` |

`content_read` prints:

```text
300 words in 10000 hard-locations; content read uses the top k = 103 rows
address-noise   address read   content read   (recalled of 30, overlap >= 0.95)
       10%          18             30
       20%           4             30
       30%           0             30
```

`capacity` prints:

```text
2000 hard-locations of 256 bits, activation-radius 112
predicted capacity at 10% address-noise: 93 words
stored   recalled from 10% address-noise (overlap >= 0.95, 40 probes)
    25   40 of 40
    50   34 of 40
   100    9 of 40
   150    0 of 40
   200    0 of 40
   300    0 of 40
```

The signal-to-noise map predicts 93 words for this memory. Measured recall falls from 34 of 40 probes at 50 words
to 9 of 40 at 100. The prediction is a prediction, not a measurement.

## The same thing, precisely

Patterns and addresses are vectors in {-1, +1}^n. The memory has M hard-locations with fixed random
addresses a_1 .. a_M and integer counter rows C_1 .. C_M.

Hamming distance and the activated set (Kanerva's access circle):

    d(x, y) = (n - x . y) / 2
    A(z) = { i : d(a_i, z) <= r }

`d` counts the bits where two vectors differ. A(z) is the set of hard-locations a state z activates.

The fraction of hard-locations a random state activates, Kanerva's activation-probability:

    p = P[ Bin(n, 1/2) <= r ]

The distance from a fixed point to a random address is Binomial(n, 1/2), so p is its lower tail.

Write:

    C_i <- C_i + x    for every i in A(x)

Every activated hard-location adds the pattern to its bit-counters.

Kanerva's read, iterated:

    s_j = sum over i in A(z) of C_ij
    z_j <- +1 if s_j > 0,  -1 if s_j < 0,  z_j if s_j = 0

The activated hard-locations vote bit by bit; a tied bit keeps its old value. The read repeats until z stops
changing or the read budget runs out.

The content-woken read (the "pulls" read):

    A_theta(z) = { i : C_i . z > theta },  theta = 0.4 n by default

A hard-location wakes because of what it holds, not where its address sits. The top-k variant wakes the k
filled rows with the largest C_i . z.

Shared hard-locations of two points d bits apart (Bricken and Pehlevan 2021, Eq. 2):

    I(d) = sum over b, c of C(d, b) C(n - d, c) 2^-n   [b + c <= r and d - b + c <= r]

A read-address d bits from a stored pattern activates on average M I(d) of the hard-locations the pattern was written to.

The signal-to-noise map for an iterated read with T stored patterns (Bricken and Pehlevan 2021, Eq. 25):

    SNR(d) = M I(d) / sqrt( M I(d) + (T - 1)(M I_o + (M I_o)^2) ),   I_o = I(n/2)
    d_next = n Phi(-SNR(d))

The read-address's own shared hard-locations are the signal; the other T - 1 patterns are the noise. Iterating the map
says whether a read from distance d converges, and its largest converging T is the predicted capacity.

The soft cut-off:

    phi(d) = 1 / (1 + exp(-(t - d) / w)),   w = s sqrt(n) / 2

A hard-location is activated with a probability that falls smoothly with distance. Softness s = 0 gives the hard
ball. The threshold t is refitted for each s so the expected number of activated hard-locations never changes.

The infinite-location kernel and its attention fit:

    K(d) = sum over a, b of Bin(a; n - d, 1/2) Bin(b; d, 1/2) phi(a + b) phi(a + d - b)
    ln K(d) ~ a - c d   =>   beta = c n / 2

K(d) is the expected share of hard-locations two points d bits apart both activate. Fitting its log slope gives
the inverse temperature of the softmax attention whose weights fall like the memory's.

The refusal threshold:

    h(T) = largest h with 1 - (1 - P[Bin(n, 1/2) <= h])^T <= level

With T random stored patterns, a random read-address has a stored pattern within h bits with probability at most
`level`. A read whose answer travelled more than h bits from the read-address is refused.

## The modules

The front door is `sdm`, with `sizing`, `calibrated` and `erase` adding to it. `diagnose` checks data and failed
reads. `words`, `rails` and `lang` are the word list and its two faces. The rest are the lab: the instruments the
SETTLE campaign measured with, public so SETTLE and any measurement can call them directly.

| module | what it does | the SETTLE statements that use it |
|---|---|---|
| `sdm` | the front door: `Sdm`, `ReadResult`, `Refusal`, `SoftSdm` and the word helpers | none yet; settle-rs calls the lab modules directly (see "The SETTLE seam") |
| `rng` | xorshift64* generator: uniform, signed, bounded, normal | every statement (`State::rng` is this type) |
| `codes` | FNV-1a seeds, named ±1 patterns, text as bits, text masked by its name, overlap, address-noise | `memory`, `sdm`, `softsdm`, `sdmscale` writes and read-addresses |
| `keys` | keyed notes: a key names a rotation of the cube, finds its note and reads it | `m.save ..., key:` · `m.recall key:` · `s.write ..., key:` · `s.read key:` |
| `bits` | `i8` and bit-packed patterns, Hamming distance, address-noise, best-match (nearest neighbour) | `sdmscale`, `refusal`, `contenttrack` |
| `theory` | binomial tables, the Hamming ball, the intersection of two balls, the activation-radius for an activation probability, the normal CDF | default activation-radius of `sdm` and `sdmscale` (`activation-probability:`), `softsdm` |
| `address` | a named address matrix, the activated set, Kanerva's iterated read over bit-counters held by the caller | `sdm :s` (bit-counters held as SETTLE pulls), `s.write`, `s.read via: :addresses` |
| `store` | `Store`: one-byte bit-counters for 10^5 to 10^6 hard-locations; address, threshold and top-k reads; row and entry shuffles; density-scaled thresholds | `sdmscale :k`, `k.put`, `k.fill`, `k.read via: :addresses / :pulls, wake: :fixed / :density / :top` |
| `smap` | the S-map: critical-distance, capacity, activation-radius for an address-noise level; averaged noise, mirror field, Poisson first read, RACE | `sdmscale ..., tolerate-noise:` |
| `soft` | the soft machine: p-bit hard-locations with a softness dial, pass and settle reads, mean field, kernel, softmax-attention read | `softsdm :s`, `s.write`, `s.read`, `s.attend` |
| `refuse` | travel threshold, the exact best-match oracle, diagnostic reads, predictor TRACK for the address read | `refusal size:, load:, level:` |
| `track` | predictors for the content-woken reads (TRACK-C, TRACK-G, TRACK-R), read traces, censuses, calibrated refusal scores | `contenttrack ...` |
| `hopfield` | the zero-temperature Hopfield baseline sized to a pull budget | no statement; the measurement instruments use it |
| `sizing` | the SNR activation-radius, the radius for an address-noise, TRACK's failure rate and capacity on `Sdm` | none yet |
| `calibrated` | the calibrated refusal (SDMTRACK's calibrated minimum) over the address or the content read | none yet; `contenttrack` measures the same rule |
| `diagnose` | the bit balance of a set of words; the race between a target and its rivals for shared hard-locations | none |
| `erase` | the exact undo of a write, and the leave-one-out read | none |
| `hetero` | a data-word of its own width written at an address, with int32 bit-counters and a radius or nearest-k wake | none; #/sdm's MNIST run uses it through the WebAssembly build |
| `words` | the one word list: every statement, its keywords, their kinds and defaults | all five sdm-family statements (SETTLE mounts it) |
| `rails` | the Rust builder, the word list spoken in Rust | none; `examples/rails.rs` prints SETTLE's lines with it |
| `lang` | the one parser and runner for `.kanerva` files and SETTLE's sdm-family lines | the five sdm-family statements, parsed for SETTLE (`src/plug.rs`) |

## Using it

The snippets below call the lab modules directly, as version 0.1.0 has them. They run in order, each after the one before; names such as
`patterns`, `read_address_f64` and `my_counters` stand for your own data. `tests/readme.rs` compiles and runs
every snippet on this page with stand-in data, so a snippet that stops compiling fails `cargo test`. The complete
program is `examples/quickstart.rs` (`cargo run --release --example quickstart`).

### `rng`, `bits`, `theory`: patterns and radii

```rust
use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap, pack, hd}, theory::{radius_for, ball}};

let mut rng = Rng::new(7);
let p = random_pattern(256, &mut rng);                    // Vec<i8> of -1 and +1
let read_address = add_address_noise(&p, 0.2, &mut rng);  // each bit flipped with probability 0.2
let dist = hd(&pack(&p), &pack(&read_address));           // Hamming distance on packed words
assert!((overlap(&p, &read_address) - (1.0 - 2.0 * dist as f64 / 256.0)).abs() < 1e-12);
let r = radius_for(256, 0.005);                           // smallest r activating at least 0.5% of hard-locations
assert!(ball(256, r) >= 0.005);
```

### `store`: the byte store, built for 10^5 to 10^6 hard-locations

```rust
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
```

Thresholds that scale with the load (`density_threshold`, `density_threshold_blocks`) feed
`read_pulls_at`.

### `address`: bit-counters kept anywhere

```rust
use kanerva::address::{Addresses, iterated_read};

let a = Addresses::named("s", 1, 256, 2000);        // the same name and seed always give the same matrix
let active = a.awake(&read_address_f64, r);         // the hard-locations within r of the state
let (z, reads, activated) = iterated_read(&a, r, &read_address_f64, 10, |i, sum| {
    for j in 0..256 { sum[j] += my_counters[i][j]; }
});
```

The SETTLE `sdm` family keeps its bit-counters as pulls between things and reads them through this callback.

### `smap`: predict before you build

```rust
use kanerva::smap::{Lazy, SMap, goal, radius_for_address_noise, search_window};

let l = Lazy::new(256, 100_000, 103);
let cap = l.capacity(77, goal(256));  // largest T whose map from 77 of 256 bits of address-noise reaches overlap 0.95
let dcrit = SMap::new(1000, 1_000_000, 451).critical(10_000);
let (lo, hi) = search_window(256, 100_000);
let (r30, _) = radius_for_address_noise(256, 100_000, 0.3, lo, hi);
let p30 = Lazy::averaged(256, 100_000, r30).p_converge_flips(0.3, 1000, goal(256));
let race = l.p_converge_race(0.4, 5, goal(256), 3000, 1);
```

`cd_optimal_radius` searches the same window for the activation-radius with the largest critical-distance.

### `soft`: the soft cut-off and the attention limit

```rust
use kanerva::soft::{Machine, attention_read, Attn};

// n, M, activation-probability, softness, gain, seed, write samples
let mut mc = Machine::new(256, 2000, 0.05, 0.25, 64.0, 1, 16);
mc.write(&p_f64, &mut rng);
let back = mc.recall(&read_address_f64, 3, 16, false, 0, &mut rng); // 3 one-way passes, 16 samples each
let mf = mc.recall_mean_field(&read_address_f64, 3);                // infinitely many samples
let kern = mc.kernel_inf();
let beta = Machine::fit_beta(&kern);
let att = attention_read(&read_address_f64, &stored, &Attn::Softmax(beta));
```

### `keys`: a note only its key can read

```rust
use kanerva::keys::{keyed_pattern, keyed_read_address, keyed_read};

let note = keyed_pattern("blue heron", "under the mat", 256); // write this into any memory
let key_address = keyed_read_address("blue heron", 256);      // read from this
assert_eq!(keyed_read("blue heron", &note).as_deref(), Some("under the mat"));
assert_eq!(keyed_read("red heron", &note), None);
```

This is not cryptography. The key is hashed to 64 bits with FNV-1a and a guessed key can be checked
offline against the padding.

### `refuse` and `track`: say "I never stored that"

```rust
use kanerva::refuse::{travel_threshold, oracle_point, Fast};
use kanerva::track::{Content, Wake};

let h = travel_threshold(256, 3000, 0.01);       // accept an answer only within h bits of the read-address
let (recall, no_refusal, refusal) = oracle_point(256, 0.4, 3000, h);
let diag = Fast::new(&st).topk(&read_address, 20, k); // the answer plus first-read and last-read signals
let c = Content::new(256, 100_000, 103);
let pc = c.p_converge(Wake::Topk(c.k()), 0.3, 3000, 400, true, 1);
```

`refuse::Track` predicts the address read; `track::TailCal` and `track::calibrate` turn the memory's own random
probes into a refusal score.

## Example output

`cargo run --release --example quickstart` prints this. Every number comes from a fixed seed, so a re-run prints the
same lines. This copy was taken on an M5 Max laptop on 2026-10-05, from the source at commit `a1e22e933`, and
`tests/readme.rs` fails if it ever differs from `examples/quickstart.out`:

```text
memory: 50000 hard-locations, word-size 256 bits, activation-radius 107 (activates 0.513% of hard-locations)
wrote 300 words; each write activated 258.4 hard-locations on average
address read from a read-address with 20% address-noise: overlap 1.000 after 3 iterated-reads (267 hard-locations activated, fixed point true)
top-k read (k = 256): overlap 1.000
shuffled rows, same read-address: overlap -0.047
S-map capacity at 20% address-noise: 738 words
refusal threshold h = 95 bits; this answer travelled 40 bits: accept
keyed note, right key: Some("under the mat"); wrong key: None
soft memory (softness 0.25): overlap 1.000; fitted softmax inverse temperature 2.17
```

## The file face: `.kanerva` programs and the `kanerva` command

KANERVA reads its own statements from text. A `.kanerva` program is written in SETTLE's syntax and holds only the
sdm family: `sdm`, `softsdm`, `sdmscale`, `refusal` and `contenttrack`, inside `model :name do ... end` and
`run :name do ... end` blocks. This is `programs/hello.kanerva`, the builder at the top of this page as a file:

```kanerva
# The first KANERVA program. Write four words into a sparse distributed memory,
# then read cat back from a copy with a fifth of its bits flipped.
# README.md opens with the same lines in Rust.
model :mind do
  sdm :s, word-size: 256, hard-locations: 2000
  s.write :cat
  s.write :dog
  s.write :owl
  s.write :note, "meet at nine"
end

run :mind do
  s.read read-address: :cat, address-noise: 0.2, seed: 1
end
```

The `kanerva` command runs it on KANERVA's engine, with no SETTLE anywhere:

```bash
cargo run --release --bin kanerva -- programs/hello.kanerva
```

```text
read :s from read-address :cat with 20% address-noise via addresses (2 iterated reads, 48 of 2000 hard locations activated): :cat +1.00  :owl -0.09  :note +0.09  -> :cat
```

`settle programs/hello.kanerva` prints the same line, because SETTLE reads the file with the same parser.

The pieces, all in the crate:

| where | what |
|---|---|
| `src/words.rs` | the one word list: every statement, the keywords it takes, their kinds and defaults, the retired words and the aliases |
| `src/lang/lex.rs` | the tokens of a line and the argument helpers (`kwargs`, `only`, `suggest`, `locate`) |
| `src/lang/parse.rs` | the one parser: `Family::parse` turns a line into a typed `Stmt`, checking in SETTLE's own order |
| `src/lang/say.rs` | every line the family prints |
| `src/lang/run.rs` | the runner: `kanerva::lang::run(src)` executes a program on `address`, `soft`, `store`, `refuse` and `track` |
| `src/bin/kanerva.rs` | the command: `--help`, `--version`, one program; an error prints the line with a caret |
| `programs/*.kanerva` | the programs, each with the `.out` it prints (`tests/lang_programs.rs` holds them) |

**One parser for two commands.** SETTLE mounts this parser for the same lines in a `.settle` program, so the two can
never read a line two ways. A `.kanerva` file is also a valid `.settle` file, and both commands print the same lines
for it; settle-rs's `tests/oneparser_parity.rs` holds that on every example program and on a corpus of 118 small
programs (every option and every error of every statement). One thing only SETTLE runs: `s.read ..., via: :pulls`
settles the pulls of a SETTLE model, so `kanerva` refuses it by name. `write-samples:` is the spelling of record for
softsdm's write samples; `write_samples:` still works.

## The SETTLE seam

The SETTLE interpreter (settle-rs) is KANERVA's one consumer today. It depends on the crate by path inside the
research repository and by git URL in SETTLE's own repository, with its `sdm` feature (on by default). Since
2026-10-06 its sdm-family lines are parsed here: settle-rs mounts `kanerva::lang` as a plug-in (`src/plug.rs`), and
each statement then runs on the lab modules below. Each of its files re-exports what it uses under the name it had
before the lift, so its own paths stay stable:

| settle-rs file | the SETTLE statements | the KANERVA items it uses |
|---|---|---|
| `src/engine/rng.rs` | every statement | `rng::Rng` |
| `src/plug.rs` | the five sdm-family statements, parsed | `lang::{Family, Stmt}`, `words` |
| `src/memory.rs` | `memory`, `m.save`, `m.recall`, keyed notes | `codes::{code, pattern, seed_of, bits_text, overlap}`, `keys::*` |
| `src/sdm.rs` | `sdm :s`, `s.write`, `s.read` | `address::{Addresses, iterated_read}`, `theory::radius_for`, `store::WAKE`, `codes`, `keys` |
| `src/sdmscale.rs` | `sdmscale :k`, `k.put`, `k.fill`, `k.read` | `store::{Store, ReadOut, threads}`, `bits`, `theory`, `smap::SMap`, `hopfield` |
| `src/sdmradius.rs` | `sdmscale ..., tolerate-noise:` | `smap::{Lazy, goal, search_window, cd_optimal_radius, radius_for_address_noise*}`, density thresholds |
| `src/softsdm.rs` | `softsdm :s`, `s.attend` | `soft::{Machine, Attn, attention_read, calibrate, phi, sign_of}`, `theory::{binom_log_pmf, hard_radius}` |
| `src/sdmrefuse.rs` | `refusal` | `refuse::{travel_threshold, oracle_point, Fast, Track, ...}`, `bits::{hd, nearest, pack_all}` |
| `src/sdmtrack.rs` | `contenttrack` | `track::*` |

settle-rs builds without KANERVA too. With its `sdm` feature off it uses its own byte-identical copies of `rng::Rng`
and `codes`' `seed_of` and `code`, held equal to KANERVA's by a test; with the feature on, `Rng` is KANERVA's own
type, because the sdm statements hand their generator straight to KANERVA's reads. The statements `valleys` and
`coded` use settle-rs's own copies whichever way it is built.

The rules of the seam:

- **No arithmetic lives on both sides.** settle-rs keeps its things, pulls and statements; every number about the
  memory comes from KANERVA. When the code was lifted, 45 recorded program outputs matched byte for byte.
- **A rename in a lab module is a change to settle-rs.** The lab names are public API. Renaming one means
  changing the re-export in settle-rs in the same change, which is why the soft machine's field is still `fire`.
- **The front door does not change the seam.** `Sdm` wraps the same `Store`, so moving a settle-rs statement onto
  it later is a call-for-call swap with the same output, checked by `src/sdm.rs`'s own tests.
- **The statements are parsed here.** Since 2026-10-06 settle-rs parses no sdm-family line itself: its registry
  mounts `kanerva::lang` (settle-rs `src/plug.rs`), and the typed statement comes back to the settle-rs file in the
  table above, which runs it on SETTLE's pulls. See "The file face" above.
- **The retired keywords stay retired.** settle-rs's `tests/kanerva_terms.rs` scans every SETTLE program and source in
  settle-rs, KANERVA, settle-mcp, the site and the brand folder for a keyword KANERVA_TERMS.md retired.

## What it has measured

`DISCOVERIES.md` maps every finding from the campaign, the wiki pages and the papers to what it became in this
crate, or why it did not. These results come from the SETTLE campaign. Every experiment ran its own instrument on this code (before it
moved into KANERVA), sealed its predictions first, and carried a negative control. The numbers are copied
from the reports; read the report for the conditions. The reports live in the SETTLE campaign's research
repository, under `runs/`, which is not public. P90 is the largest stored count before the first
checkpoint under 90% recall; recall means final overlap at least 0.95.

**Capacity at Kanerva's own scale** (`REPORT_SDMSCALE.md` §3). With
the signal-to-noise activation-radius at n = 256, address-read P90 at 10% address-noise grows from 50 patterns at
M = 2,000 to 20,000 at M = 1,000,000, about M^0.96. At 30% and 40% address-noise it does not grow with M,
because a 40%-noisy read-address shares about one hard-location with its pattern at every M under that activation-radius.

**More patterns than Hopfield at equal memory, once the read wakes by content**
(`REPORT_SDMRADIUS.md` §7). At a budget of M = 1,000,000 x 256
bit-counters, the top-k read holds P90 70,000 / 30,000 / 10,000 / 10 patterns at 10/20/30/40% address-noise. A
Hopfield net with the same number of pulls (22,628 units) holds 100 / 70 / 30 / 3. At a small budget
the order reverses: 128 hard-locations (32,768 bit-counters) hold 5 / 0 / 0 / 0 with the address read and
5 / 5 / 0 / 0 with the content read, against 25 / 25 / 20 / 0 for a 256-unit Hopfield net with 32,640 pulls (`REPORT_SDMKEYS.md` §2).

**The soft cut-off is attention at a low temperature** (`REPORT_SOFTSDM.md`
§6, §9). At softness 0 the machine equals an independent hard SDM on every non-tie bit. At softness 1
its one-round read agrees with softmax attention on 99.2% of bits, and both recall 0%: the fitted inverse
temperature (0.71 at activation-probability 0.05) is too low to separate patterns.

**Refusing a never-stored read-address** (`REPORT_SDMREFUSE.md` §2-3,
`REPORT_SDMTRACK.md` §3-4). At light load every never-stored read-address
lands on a stored pattern (500/500), so no signal read off the final state can refuse it. The travel rule
follows the exact best-match (nearest-neighbour) oracle within 0.01 to 0.05 there (top-k, M = 10^5, T = 10: refusal
0.992, 40% recall 0.515 against the oracle's 0.506). In crowded stores never-stored read-addresses stop moving
(median travel 8 bits at M = 10^6, T = 10,000) and the travel rule's refusal falls to 0.044. Calibrating on
the memory's own random probes recovers it: the calibrated-minimum rule and travel alone each refuse at
least 0.98 of never-stored read-addresses in 18 cells, against 12 cells for the fixed threshold h(T).

**A key reads its note and a wrong key reads nothing** (`REPORT_SDMKEYS.md`
§6, a Hopfield memory of 512 things holding a 27-byte keyed note among public memories). The right key
read the note 20 of 20 times at 2 to 20 public memories, the wrong key 0 of 20. A stranger starting
from noise lands in the note's valley 14.2% of the time but reads 50.2% of its text bits, which is
chance.

## How it relates to `wikis/WIKI_SDR/canonical/sdm.py`

`wikis/WIKI_SDR/canonical/sdm.py` (with `sdm_trainable.py`) is a separate SDM written in Python with
numpy. KANERVA does not import it, and it does not import KANERVA. They cite the same theory (Kanerva 1988,
Bricken and Pehlevan 2021) and the same test value (activation-radius 451 for n = 1000 at p = 0.001; the exact
share at radius 451 is 0.00107, 1,072 of a million hard-locations, and Kanerva 1993's own sample memory uses radius
447, which activates 445). Reading both
shows these differences:

- **Values.** `sdm.py` stores {0, 1} bits and adds `2d - 1` to the bit-counters. KANERVA stores ±1 directly.
- **Ties and empty reads.** `sdm.py` sets a bit to 1 when its vote sum is 0 (`sums >= 0`) and returns all
  zeros when no hard-location is activated. KANERVA keeps the old bit on a zero sum, so a read that activates nothing
  leaves the read-address unchanged and stops.
- **Counters.** `sdm.py` uses unbounded `int32` bit-counters unless `counter_range` clamps them. KANERVA's
  `Store` uses one byte per counter, clamps at ±127, and counts every clamp in `overflow`.
- **Addressing.** `sdm.py` offers the Hamming ball and Jaeckel's selected-coordinate design. KANERVA
  offers the Hamming ball only, and adds the content-woken reads (threshold, density-scaled threshold,
  top-k) that `sdm.py` does not have.
- **Radius rules.** `sdm.py` defaults to p = 0.001 and has Kanerva's write-budget formula
  r* = n/2 + Phi^-1((2MT)^(-1/3)) sqrt(n/4). KANERVA inverts the exact binomial tail (`radius_for`), and
  `smap::search_window` applies the same (2MT)^(-1/3) rule with T = M/20 through that exact inverse.
- **What else each has.** `sdm.py` has heteroassociative writes, pointer-chain sequences, a
  surprise-gated write and per-bit vote margins. KANERVA has the S-map and its refinements, refusal and
  the best-match oracle, the content-read predictors, keyed notes, the soft p-bit machine and a
  Hopfield baseline.
- **Soft reads.** `sdm_trainable.py` is a differentiable softmax read in PyTorch for training by
  gradient. KANERVA's `soft` module samples p-bit hard-locations with a logistic cut-off and fits a softmax to
  its kernel; it has no gradients.
- **Randomness.** `sdm.py` draws from numpy's generator; KANERVA from xorshift64*. The same seed builds
  different memories in the two.

## Known limits, recorded rather than changed

KANERVA was lifted out of settle-rs with every output kept identical, so these stay as they were:

- `refuse::poisson` (normal approximation above mean 30, returns `u32`) and the private Poisson draw in
  `smap` (normal approximation above mean 40, returns `f64`) are two Poisson samplers.
- `store::threads()` reads the environment variable `SDMSCALE_THREADS` (default 8), a name from the SETTLE
  experiment that wrote it. The thread count changes speed only; `write_many` gives the same result as writing
  one by one.
- `smap::Lazy` memoises through `RefCell`, so one `Lazy` cannot be shared between threads.
- `codes::overlap` divides by the length of its second argument and `bits::overlap` by the first. They agree
  whenever the two words have the same length, which is every call in this crate and in settle-rs.
- `soft::Machine`'s field is still named `fire` (the activation-probability), because settle-rs reads it by that
  name. The constructor's parameter is named `activation_probability`.
- `smap::SMap::critical` and `Lazy::critical` return the first distance the map does not move toward 0. Under
  heavy load that is 1, because a read next to its word drifts out to a residual error; reads from farther out may
  still settle there. At Kanerva's point (n 1000, M 10^6, r 451, T 10^4) the map gives 165, Bricken and Pehlevan
  report 188, and a real store measured 204.3 (SDMSCALE); the gap is not explained.
- `Sdm::predicted_capacity` truncates `address_noise * n`, where `smap::radius_for_address_noise` rounds it.
- `soft::Machine::read_settle` (the Gibbs read with the data held) landed on the mixture of three stored words in a
  small probe at softness 0.25 (n 128, M 400), where `read_pass` recalled the word exactly; at softness 0 both
  recall. The cause is not established. `SoftSdm` uses `read_pass`.

## Tests

`cargo test --release` runs the unit tests in `src/`, the integration tests in `tests/` and a doc-test for every
public item. On the M5 on 2026-10-07 that was 82 unit tests, 35 integration tests (agreement 3, examples 13,
programs 6, word list 6, README 7) and 269 doc-tests, all passing; count them again with
`cargo test --release 2>&1 | grep 'test result'`. The examples run seconds in release, so run the suite with
`--release`.
`src/theory.rs` and `src/smap.rs` check the crate against the papers on disk: Kanerva 1993's sample memory and
capacity, Kanerva 1988's Table 7.1 as Jaeckel 1989 quotes it, Kanerva 2009 and Bricken and Pehlevan's Table 1. Each module has at least one case
computed by hand (a three-location memory whose read cycles with period 2, a two-location store, the
FNV-1a reference values, Poisson as a one-value compound law, refusal with one pattern of 4 bits) and at
least one negative control (shuffled rows or bit-counters, a wrong key, an activation-radius too small to share a
hard-location, an overloaded Hopfield net, a far-overloaded store). `tests/agreement.rs` proves the
caller-held read and the byte store agree bit for bit. `tests/readme.rs` runs every snippet on this page and fails
when a snippet here differs from the one it runs, or when a program or printout quoted here differs from its file.
`tests/examples.rs` holds each example to its recorded output.

## History

The algorithms were written in the SETTLE campaign (2026-09-30) by the experiments SDMKEYS, SOFTSDM, SDMSCALE,
SDMRADIUS, SDMREFUSE and SDMTRACK inside the SETTLE interpreter, and lifted into this crate by SDMKIT
on 2026-10-01. On 2026-10-06 KANERVAPERFECT moved the measured rules the lab held onto the front door and
checked the theory against the papers. See `CHANGELOG.md` and `DISCOVERIES.md`.
