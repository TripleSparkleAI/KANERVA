# KANERVA

It's like Redis! But for Sparse Distributed Memory!

KANERVA is a Rust library that stores binary patterns in Pentti Kanerva's sparse distributed memory and
reads them back from noisy read-addresses.

The crate uses Kanerva's own words, joined by a hyphen where his word is two: read-address, address-noise, hard
locations, activation radius, bit-counters, iterated reads. Each one, with Kanerva's sentence and page, is in
[KANERVA_TERMS.md](KANERVA_TERMS.md).

> **The repository is `github.com/triplesparkle/KANERVA`, private for now.** A clone or a git dependency
> needs access until it is made public. The SETTLE interpreter (`github.com/triplesparkle/SETTLE`) depends
> on it by its git URL.
>
> **Licence: to be chosen.** No licence has been granted yet, so no rights are given by this folder.
> This is flagged for the owner to decide before the repository is opened to anyone.

Zero dependencies. Version 0.1.0. 12 modules, 49 tests (46 unit tests and 3 cross-module tests), one
example.

## Install

As a git dependency:

```bash
cargo add kanerva --git https://github.com/triplesparkle/KANERVA
```

From a clone:

```bash
git clone https://github.com/triplesparkle/KANERVA && cd KANERVA && cargo test --release
```

Or by hand, under `[dependencies]` in your `Cargo.toml`:

```toml
kanerva = { git = "https://github.com/triplesparkle/KANERVA" }
```

All three need access while the repository is private. A crate in a folder beside a clone can name it by
path instead: `kanerva = { path = "../KANERVA" }`.

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

## The same thing, precisely

Patterns and addresses are vectors in {-1, +1}^n. The memory has M hard locations with fixed random
addresses a_1 .. a_M and integer counter rows C_1 .. C_M.

Hamming distance and the wake set:

    d(x, y) = (n - x . y) / 2
    A(z) = { i : d(a_i, z) <= r }

`d` counts the bits where two vectors differ. A(z) is the set of hard locations a state z wakes.

The fraction of hard locations a random state wakes:

    p = P[ Bin(n, 1/2) <= r ]

The distance from a fixed point to a random address is Binomial(n, 1/2), so p is its lower tail.

Write:

    C_i <- C_i + x    for every i in A(x)

Every woken hard location adds the pattern to its bit-counters.

Kanerva's read, iterated:

    s_j = sum over i in A(z) of C_ij
    z_j <- +1 if s_j > 0,  -1 if s_j < 0,  z_j if s_j = 0

The woken hard locations vote bit by bit; a tied bit keeps its old value. The read repeats until z stops
changing or the read budget runs out.

The content-woken read (the "pulls" read):

    A_theta(z) = { i : C_i . z > theta },  theta = 0.4 n by default

A hard location wakes because of what it holds, not where its address sits. The top-k variant wakes the k
filled rows with the largest C_i . z.

Shared hard locations of two points d bits apart (Bricken and Pehlevan 2021, Eq. 2):

    I(d) = sum over b, c of C(d, b) C(n - d, c) 2^-n   [b + c <= r and d - b + c <= r]

A read-address d bits from a stored pattern wakes on average M I(d) of the hard locations the pattern was written to.

The signal-to-noise map for an iterated read with T stored patterns (Bricken and Pehlevan 2021, Eq. 25):

    SNR(d) = M I(d) / sqrt( M I(d) + (T - 1)(M I_o + (M I_o)^2) ),   I_o = I(n/2)
    d_next = n Phi(-SNR(d))

The read-address's own shared hard locations are the signal; the other T - 1 patterns are the noise. Iterating the map
says whether a read from distance d converges, and its largest converging T is the predicted capacity.

The soft cut-off:

    phi(d) = 1 / (1 + exp(-(t - d) / w)),   w = s sqrt(n) / 2

A hard location is activated with a probability that falls smoothly with distance. Softness s = 0 gives the hard
ball. The threshold t is refitted for each s so the expected number of activated hard locations never changes.

The infinite-location kernel and its attention fit:

    K(d) = sum over a, b of Bin(a; n - d, 1/2) Bin(b; d, 1/2) phi(a + b) phi(a + d - b)
    ln K(d) ~ a - c d   =>   beta = c n / 2

K(d) is the expected share of hard locations two points d bits apart both activate. Fitting its log slope gives
the inverse temperature of the softmax attention whose weights fall like the memory's.

The refusal threshold:

    h(T) = largest h with 1 - (1 - P[Bin(n, 1/2) <= h])^T <= level

With T random stored patterns, a random read-address has a stored pattern within h bits with probability at most
`level`. A read whose answer travelled more than h bits from the read-address is refused.

## The modules

| module | what it does | the SETTLE statements that use it |
|---|---|---|
| `rng` | xorshift64* generator: uniform, signed, bounded, normal | every statement (`State::rng` is this type) |
| `codes` | FNV-1a seeds, named ±1 patterns, text as bits, text masked by its name, overlap, address-noise | `memory`, `sdm`, `softsdm`, `sdmscale` writes and read-addresses |
| `keys` | keyed notes: a key names a rotation of the cube, finds its note and reads it | `m.save ..., key:` · `m.recall key:` · `s.write ..., key:` · `s.read key:` |
| `bits` | `i8` and bit-packed patterns, Hamming distance, address-noise, nearest neighbour | `sdmscale`, `refusal`, `contenttrack` |
| `theory` | binomial tables, the Hamming ball, the intersection of two balls, the activation radius for an activation probability, the normal CDF | default activation radius of `sdm` and `sdmscale` (`activation-probability:`), `softsdm` |
| `address` | a named address matrix, the wake set, Kanerva's iterated read over bit-counters held by the caller | `sdm :s` (bit-counters held as SETTLE pulls), `s.write`, `s.read via: :addresses` |
| `store` | `Store`: one-byte bit-counters for 10^5 to 10^6 hard locations; address, threshold and top-k reads; row and entry shuffles; density-scaled thresholds | `sdmscale :k`, `k.put`, `k.fill`, `k.read via: :addresses / :pulls, wake: :fixed / :density / :top` |
| `smap` | the S-map: critical distance, capacity, activation radius for an address-noise level; averaged noise, mirror field, Poisson first read, RACE | `sdmscale ..., tolerate-noise:` |
| `soft` | the soft machine: p-bit hard locations with a softness dial, pass and settle reads, mean field, kernel, softmax-attention read | `softsdm :s`, `s.write`, `s.read`, `s.attend` |
| `refuse` | travel threshold, the exact nearest-neighbour oracle, diagnostic reads, predictor TRACK for the address read | `refusal size:, load:, level:` |
| `track` | predictors for the content-woken reads (TRACK-C, TRACK-G, TRACK-R), read traces, censuses, calibrated refusal scores | `contenttrack ...` |
| `hopfield` | the zero-temperature Hopfield baseline sized to a pull budget | no statement; the measurement instruments use it |

## Using it

The snippets below show the calls of version 0.1.0; names such as `patterns` or `cue_f64` stand for
your own data. The complete program that compiles and runs is `examples/quickstart.rs`
(`cargo run --release --example quickstart`).

### `rng`, `bits`, `theory`: patterns and radii

```rust
use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap, pack, hd}, theory::{radius_for, ball}};

let mut rng = Rng::new(7);
let p = random_pattern(256, &mut rng);         // Vec<i8> of -1 and +1
let cue = add_address_noise(&p, 0.2, &mut rng);           // each bit flipped with probability 0.2
let dist = hd(&pack(&p), &pack(&cue));         // Hamming distance on packed words
assert!((overlap(&p, &cue) - (1.0 - 2.0 * dist as f64 / 256.0)).abs() < 1e-12);
let r = radius_for(256, 0.005);                // smallest r waking at least 0.5% of locations
assert!(ball(256, r) >= 0.005);
```

### `store`: a memory of a million hard locations

```rust
use kanerva::store::Store;

let mut st = Store::new(256, 50_000, r, 1);    // n, M, radius, address seed
st.write(&p);                                  // returns how many locations took the write
st.write_many(&patterns);                      // same result as writing one by one, wake sets in parallel
let out = st.read_addresses(&cue, 20);         // Kanerva's read, at most 20 rounds
println!("{} {} {}", overlap(&out.z, &p), out.rounds, out.fixed);
let k = (ball(256, r) * 50_000.0).round() as usize;
let top = st.read_pulls_topk(&cue, 20, k);     // content-woken read, k best-matching rows
let thr = st.read_pulls(&cue, 20);             // content-woken read, threshold 0.4 n
let mut control = st.clone();
control.shuffle_rows(&mut Rng::new(3));        // negative control: the memory is gone
```

Thresholds that scale with the load (`density_threshold`, `density_threshold_blocks`) feed
`read_pulls_at`.

### `address`: bit-counters kept anywhere

```rust
use kanerva::address::{Addresses, iterated_read};

let a = Addresses::named("s", 1, 256, 2000);   // the same name and seed always give the same matrix
let woken = a.awake(&cue_f64, r);              // locations within r of the state
let (z, reads, awake) = iterated_read(&a, r, &cue_f64, 10, |i, sum| {
    for j in 0..256 { sum[j] += my_counters[i][j]; }
});
```

The SETTLE `sdm` family keeps its bit-counters as pulls between things and reads them through this callback.

### `smap`: predict before you build

```rust
use kanerva::smap::{Lazy, SMap, goal, radius_for_address_noise, search_window, cd_optimal_radius};

let l = Lazy::new(256, 100_000, 103);
let cap = l.capacity(77, goal(256));            // largest T whose map from a 30%-damaged cue reaches overlap 0.95
let dcrit = SMap::new(1000, 1_000_000, 451).critical(10_000);
let (lo, hi) = search_window(256, 100_000);
let (r30, _) = radius_for_address_noise(256, 100_000, 0.3, lo, hi);
let p = Lazy::averaged(256, 100_000, r30).p_converge_flips(0.3, 1000, goal(256));
let race = l.p_converge_race(0.4, 5, goal(256), 3000, 1);
```

### `soft`: the soft cut-off and the attention limit

```rust
use kanerva::soft::{Machine, attention_read, Attn};

let mut mc = Machine::new(256, 2000, 0.05, 0.25, 64.0, 1, 16); // n, M, fire, softness, gain, seed, write samples
mc.write(&p_f64, &mut rng);
let back = mc.recall(&cue_f64, 3, 16, false, 0, &mut rng);      // 3 one-way passes, 16 samples each
let mf = mc.recall_mean_field(&cue_f64, 3);                      // infinitely many samples
let kern = mc.kernel_inf();
let beta = Machine::fit_beta(&kern);
let att = attention_read(&cue_f64, &stored, &Attn::Softmax(beta));
```

### `keys`: a note only its key can read

```rust
use kanerva::keys::{keyed_pattern, keyed_read_address, keyed_read};

let stored = keyed_pattern("blue heron", "under the mat", 256);  // write this into any memory
let cue = keyed_read_address("blue heron", 256);                          // read from this
assert_eq!(keyed_read("blue heron", &stored).as_deref(), Some("under the mat"));
assert_eq!(keyed_read("red heron", &stored), None);
```

This is not cryptography. The key is hashed to 64 bits with FNV-1a and a guessed key can be checked
offline against the padding.

### `refuse` and `track`: say "I never stored that"

```rust
use kanerva::refuse::{travel_threshold, oracle_point, Fast, Track};
use kanerva::track::{Content, Wake, TailCal, calibrate};

let h = travel_threshold(256, 3000, 0.01);        // accept an answer only within h bits of the cue
let (recall, no_refusal, refusal) = oracle_point(256, 0.4, 3000, h);
let diag = Fast::new(&st).topk(&cue, 20, k);       // answer plus first-read and last-read signals
let c = Content::new(256, 100_000, 103);
let p = c.p_converge(Wake::Topk(c.k()), 0.3, 3000, 400, true, 1);
```

## Example output

`cargo run --release --example quickstart`, on the M5 at commit `415242d0e` (2026-10-01, load 30,
power mode 2):

```text
memory: 50000 hard locations, word-size 256 bits, activation radius 107 (activates 0.513% of hard locations)
wrote 300 patterns; each write activated 258.4 hard locations on average
address read from a read-address with 20% address-noise: overlap 1.000 after 3 iterated reads (267 hard locations activated, fixed point true)
top-k read (k = 256): overlap 1.000
shuffled rows, same read-address: overlap -0.047
S-map capacity at 20% address-noise: 738 patterns
refusal threshold h = 95 bits; this answer travelled 40 bits: accept
keyed note, right key: Some("under the mat"); wrong key: None
soft machine (softness 0.25): overlap 1.000; fitted softmax inverse temperature 2.17
```

## What it has measured

These results come from the SETTLE campaign. Every lane ran its own instrument on this code (before it
moved into KANERVA), sealed its predictions first, and carried a negative control. The numbers are copied
from the reports; read the report for the conditions. The reports live in the SETTLE campaign's research
repository, under `runs/`, which is not public. P90 is the largest stored count before the first
checkpoint under 90% recall; recall means final overlap at least 0.95.

**Capacity at Kanerva's own scale** (`REPORT_SDMSCALE.md` §3). With
the signal-to-noise activation radius at n = 256, address-read P90 at 10% address-noise grows from 50 patterns at
M = 2,000 to 20,000 at M = 1,000,000, about M^0.96. At 30% and 40% address-noise it does not grow with M,
because a 40%-noisy read-address shares about one hard location with its pattern at every M under that activation radius.

**More patterns than Hopfield at equal memory, once the read wakes by content**
(`REPORT_SDMRADIUS.md` §7). At a budget of M = 1,000,000 x 256
bit-counters, the top-k read holds P90 70,000 / 30,000 / 10,000 / 10 patterns at 10/20/30/40% address-noise. A
Hopfield net with the same number of pulls (22,628 units) holds 100 / 70 / 30 / 3. At a small budget
the order reverses: 128 hard locations (32,768 bit-counters) hold 5 / 0 / 0 / 0 against 25 / 25 / 20 / 0 for a
256-unit Hopfield net with 32,640 pulls (`REPORT_SDMKEYS.md` §2).

**The soft cut-off is attention at a low temperature** (`REPORT_SOFTSDM.md`
§6, §9). At softness 0 the machine equals an independent hard SDM on every non-tie bit. At softness 1
its one-round read agrees with softmax attention on 99.2% of bits, and both recall 0%: the fitted inverse
temperature (0.71 at fire 0.05) is too low to separate patterns.

**Refusing a never-stored read-address** (`REPORT_SDMREFUSE.md` §2-3,
`REPORT_SDMTRACK.md` §3-4). At light load every never-stored read-address
lands on a stored pattern (500/500), so no signal read off the final state can refuse it. The travel rule
follows the exact nearest-neighbour oracle within 0.01 to 0.05 there (top-k, M = 10^5, T = 10: refusal
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
Bricken and Pehlevan 2021) and the same test value (activation radius 451 for n = 1000 at p = 0.001). Reading both
shows these differences:

- **Values.** `sdm.py` stores {0, 1} bits and adds `2d - 1` to the bit-counters. KANERVA stores ±1 directly.
- **Ties and empty reads.** `sdm.py` sets a bit to 1 when its vote sum is 0 (`sums >= 0`) and returns all
  zeros when no hard location wakes. KANERVA keeps the old bit on a zero sum, so a read that wakes nothing
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
  the nearest-neighbour oracle, the content-read predictors, keyed notes, the soft p-bit machine and a
  Hopfield baseline.
- **Soft reads.** `sdm_trainable.py` is a differentiable softmax read in PyTorch for training by
  gradient. KANERVA's `soft` module samples p-bit hard locations with a logistic cut-off and fits a softmax to
  its kernel; it has no gradients.
- **Randomness.** `sdm.py` draws from numpy's generator; KANERVA from xorshift64*. The same seed builds
  different memories in the two.

## Known limits, recorded rather than changed

KANERVA was lifted out of settle-rs with every output kept identical, so these stay as they were:

- `theory::radius_for` and `theory::hard_radius` compute the same activation radius by two code paths. A test checks
  that they agree on 20 cases.
- `refuse::poisson` (normal approximation above mean 30, returns `u32`) and the private Poisson draw in
  `smap` (normal approximation above mean 40, returns `f64`) are two Poisson samplers.
- `store::threads()` reads the environment variable `SDMSCALE_THREADS` (default 8), a name from the SETTLE
  lane that wrote it. The thread count changes speed only; `write_many` gives the same result as writing
  one by one.
- `smap::Lazy` memoises through `RefCell`, so one `Lazy` cannot be shared between threads.
- `track::Geo::new` carries two dead statements (`let _ = half;` and a `break` that cannot fire).

## Tests

`cargo test --release` runs 46 unit tests and 3 cross-module tests. Each module has at least one case
computed by hand (a three-location memory whose read cycles with period 2, a two-location store, the
FNV-1a reference values, Poisson as a one-value compound law, refusal with one pattern of 4 bits) and at
least one negative control (shuffled rows or bit-counters, a wrong key, an activation radius too small to share a
hard location, an overloaded Hopfield net, a far-overloaded store). `tests/agreement.rs` proves the
caller-held read and the byte store agree bit for bit.

## History

The algorithms were written in the SETTLE campaign (2026-09-30) by the lanes SDMKEYS, SOFTSDM, SDMSCALE,
SDMRADIUS, SDMREFUSE and SDMTRACK inside the SETTLE interpreter, and lifted into this crate by lane SDMKIT
on 2026-10-01. See `CHANGELOG.md`.
