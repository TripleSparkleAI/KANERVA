# DISCOVERIES

This file maps what the SETTLE campaign, the theory pages, the SDR wiki, the papers on disk and the SDM language
model research found about sparse distributed memory to what it became in this crate. Each finding gets one of
four outcomes:

- **BUILT**: a measured rule or predictor that is now a function, with its source cited in the code.
- **ALREADY HERE**: the crate held it before this pass; only its docs or tests changed.
- **CANDIDATE**: a capability nobody has measured in this setting. It is not built, and the line says what
  measurement would earn it.
- **POINTER**: true and measured, but about another system (a trained model, an n-gram store, the site's
  JavaScript), so it stays a citation.

Paths under `runs/` are `SETTLE/runs/`. Paths under `WIKI/` are the repository's `WIKI/`. P90 is the largest stored
count before the first checkpoint under 90% recall, and recall means a final overlap of at least 0.95.

Updated 2026-10-06.

## 1. The SETTLE campaign (`runs/`)

### Refusal

| finding | measured | outcome |
|---|---|---|
| The calibrated-minimum refusal keeps refusing never-stored read-addresses in a crowded store | refused at least 0.98 in 18 of 18 fresh cells; the fixed travel threshold h(T) did in 12 of 18. At top-k, M 10^6, T 10,000 it refused 0.990 with recall 1.00 / 1.00 / 0.56 at 10 / 20 / 30% noise, where h(T) refused 0.056 (`runs/sdmtrack/selfcal2_*.txt`) | **BUILT**: `calibrated::CalibratedRefusal`, `Sdm::calibrated_refusal`, `Sdm::calibrated_content_refusal`, `Sdm::read_checked`. Re-measured in `examples/calibrated_refusal.rs`: at M 10,000 and 2,000 words the travel rule refused 125 of 200 never-stored read-addresses and the calibrated rule 197 |
| The rule needs enough probes | measured here: at alpha 0.01, 100 probes per set gave a cut at the floor (refusing everything) on 4 of 10 seeds at M 2,000 and 7 of 10 at M 20,000; 600 probes on 0 and 1; 1,000 or 1,500 on none | **BUILT**: `CalibratedRefusal::is_degenerate`, `floor`; docs say use at least 1,000 |
| The travel rule h(T) follows the exact best-match oracle where reads land | refusal 0.992 against the oracle's 0.993 (`runs/sdmrefuse`, §2-3); it collapses where never-stored reads stop moving (0.044 at top-k, M 10^6, T 10,000) | **ALREADY HERE**: `Refusal`, `refuse::travel_threshold`, `refuse::oracle_point` |
| Rule S (four signals, each at alpha/4) | refuses 0.982 to 1.000 in 18 of 18 cells, at more cost to 40% recall than the calibrated minimum (`runs/sdmrefuse`) | **CANDIDATE** as a front-door rule. The calibrated minimum was measured as the better trade (`runs/sdmtrack` §5), so only that one was built |
| No final-state signal refuses at light load | every never-stored read-address lands on a stored word (500 of 500, top-k, T 10) | **ALREADY HERE** as a documented limit of signals read off the answer; the calibrated rule uses first-round signals for this reason |
| The threshold content read refuses by staying silent | 150 of 150 never-stored read-addresses wake nothing at 20 stored, 59 of 150 at 100 (`runs/sdmkeys`) | **CANDIDATE**: the data is exposed (`ReadResult::activated`); a rule on it fades with load and was not measured as a rule |

### Sizing and prediction

| finding | measured | outcome |
|---|---|---|
| The SNR activation-radius, p = (2MT)^(-1/3) at T = M/20, makes capacity grow with M | P90 at 10% noise from 50 at M 2,000 to 20,000 at M 10^6, about M^0.96; a fixed radius saturates near 50 (`runs/sdmscale` §3) | **BUILT**: `sizing::snr_activation_radius`, `Sdm::with_snr_radius` (it was already the start of `smap::search_window`) |
| A radius tuned for the expected noise | at 30%, P90 10 to 300 from M 10^4 to 10^6 (about M^0.74); the SNR radius stays at 1 to 70 and holds 1.4x to 3.0x more at 10% (`runs/sdmradius` §3) | **BUILT**: `sizing::noise_activation_radius`, `Sdm::for_address_noise` |
| The content read's k is the expected access circle, pM | the best k in SDMRADIUS's sweep (§5) | **BUILT**: `Sdm::content_k` |
| TRACK-P predicts the address read's failure fraction | mean absolute error 0.015 over 12 sealed cells, 0.013 over 8 post-hoc cells; the plain S-map erred by 0.120 (`runs/sdmrefuse` §5) | **BUILT**: `Sdm::predicted_failure_rate`, `Sdm::predicted_capacity_track`. Re-measured in `examples/sizing.rs` at M 10,000: TRACK within 0.06 of measured recall in all 12 cells |
| The plain S-map is optimistic at heavy noise | 1,648 predicted against 700 measured at 30%, M 10^6; 403 against 3 at 40% (`runs/sdmradius` §3, `runs/sdmrefuse` §5) | **ALREADY HERE**; `Sdm::predicted_capacity`'s docs now say so and point at TRACK |
| TRACK-C for the content reads | accurate only at light load: error about 0.021 at pT < 2, 0.26 to 0.32 above (`runs/sdmtrack`) | **ALREADY HERE** (`track::Content`); not put on the front door because no content predictor was measured accurate at heavy load |
| Reads fail by races between whole words | M 10^5, r 107, T 5, 40% noise: best rival ahead, 55 of 55 failed; tie, 25 of 28; three or more ahead, 0 of 435 (`runs/sdmradius` §4, post-hoc) | **BUILT** as a diagnostic: `diagnose::race`, `Race::margin`, `Store::shared_locations`. A test re-checks the direction at M 10^5 |

### Reads and memories

| finding | measured | outcome |
|---|---|---|
| The content read holds far more words than the address read, and more than Hopfield at equal memory | top-k P90 70,000 / 30,000 / 10,000 / 10 at M 10^6; Hopfield with the same pulls 100 / 70 / 30 / 3 (`runs/sdmradius` §7) | **ALREADY HERE**: `Store::read_pulls_topk`, `Sdm::read_by_content` |
| Softness costs recall; the soft end is low-temperature attention | at softness 1 the machine agrees with softmax attention on 99.2% of bits and both recall 0% (`runs/softsdm`) | **ALREADY HERE**: `soft`, `SoftSdm` |
| A one-more-read stability check removes chance recalls under shuffled rows | recalled words still in place after a fourth read: 5/20, 2/20, 1/6, 0/1 (`runs/sdmscale`, `anomaly_softsdm.txt`) | **CANDIDATE**: `SoftSdm::read_stable`. Measured as an explanation of an anomaly, not as a read rule against a control |
| Fade (forgetting old writes) | at fade 0.98 the SDM recalls the 7 newest of 300 at 90% or better, against 0 at fade 1 (`runs/sdmkeys`, unsealed) | **CANDIDATE**: measured on settle-rs's f64 bit-counters; `Store` keeps one byte per counter, where a fade by 0.98 rounds away, so the result does not carry |
| The temperature (p-bit) read of a hard SDM melts recall | share of reads on the word 0.893, 0.917, 0.913, 0.581, 0.125 at T 0.25 to 4 (`runs/sdmexplore`, the site's tests) | **CANDIDATE**: lives in the site's `sdmstate.js`; a port needs a parity test against that code first |
| Compress, frame with a CRC, and optionally an LDPC code, for text notes | 0 silent wrong texts in 96,000 recalls; about twice the bytes back per note (`runs/sdmcoded`) | **POINTER**: text coding, not memory; it lives in settle-rs (`src/coded.rs`) |
| Keyed notes | right key 20 of 20, wrong key 0 of 20 (`runs/sdmkeys`) | **ALREADY HERE**: `keys`, `Sdm::write_note`, `Sdm::read_note` |

## 2. The theory pages (`WIKI/theory/167` to `179`)

These were measured on hashed or exact n-gram stores holding a language model's states, never on a hard-location
SDM. The numbers do not carry; some protocols do.

| finding | outcome |
|---|---|
| A self-inclusive score is highest for the least useful memory; leave-one-out is the honest protocol (`WIKI/theory/173` §1) | **BUILT** as a protocol: `Store::erase` (the exact undo of a write while no counter has clamped) and `Sdm::read_without` (the leave-one-out read). What the leave-one-out read shows on an SDM is not measured, and the docs say so |
| A shuffled store is a weak control; a capability control (same shape, informative contents) shows the instrument can see contents (`MEASURED_FROZENTABLE`) | **ALREADY HERE** in part: the shuffles are negative controls. **CANDIDATE**: a capability-control helper |
| Count memory in cells, not entries (`WIKI/theory/173` §5) | **POINTER**: an SDM's memory is fixed at M x n counters, so the inflation cannot arise; report bit-counters, not hard-locations, when comparing memories of different n |
| Min-support backoff address and its context-length law (`WIKI/theory/173`, `MEASURED_ADDRESSSCALE`) | **POINTER**: n-gram specific. An SDM's specificity is its activation-radius, which `sizing` sizes from the load. An adaptive-radius read is a **CANDIDATE**, and `MEASURED_MINSUPTRANSFER` measured the same cure losing 7.5% to 24.3% where its defect was absent |
| Centring a sign-LSH address: biased bits waste the address space (`WIKI/theory/176` §18) | **BUILT** as a diagnostic, measured here on an SDM: `diagnose::balance`. In `examples/bit_balance.rs` (n 256, M 10,000, 10% noise) recall stayed at least 0.91 to 300 words with fair bits, to 100 words when each bit is +1 with probability 0.6, to 10 at 0.7, and was 0.77 at 10 words at 0.8 |
| Invertible value maps are no-ops under mean-pool writes (`WIKI/theory/170`, `174`) | **POINTER**: an SDM's read is sign(), so only signed permutations commute with it; `keys` already uses exactly those. A property test is a **CANDIDATE** |
| Effective-rank pitfalls (noise maximises it; two definitions disagree by about 2x) | **POINTER**: the crate reports no spectra |
| The ceiling-and-search method (`WIKI/theory/178`) and the fail-loud harness (`179`) | **POINTER**: the crate's `refuse::oracle_point` is already a ceiling; `CalibratedRefusal::is_degenerate` refuses to pass a cut that cannot work rather than returning one quietly |

## 3. The SDR wiki and the papers on disk (`wikis/WIKI_SDR/`)

| finding | outcome |
|---|---|
| Kanerva 1993's sample memory: optimal p 0.000368, H 447 captures 445 locations, H 446 captures 354, D = N - 2H (P. 10) | **BUILT** as tests (`theory::tests::kanerva_1993_sample_memory`) and `theory::optimal_activation_probability`. K1993's p = 0.000445 is the rounded share at H 447, so `radius_for(1000, 0.000445)` is 448 |
| Kanerva 1993's capacity: tau = 1 / [Phi^-1(phi)]^2, 0.105 at phi 0.999, 0.15 at 0.995, and 0.096 for the million-location memory (P. 16-17) | **BUILT**: `theory::asymptotic_capacity`, `theory::capacity_fraction`, each tested against those numbers |
| Kanerva 1988 Table 7.1 (shared locations against distance), as Jaeckel 1989 p. 22 quotes it | **BUILT** as a test: `theory::intersection` reproduces every row within 0.6 once normalised to 1,000 at d 0. The exact share at r 451 is 0.00107 (1,072 locations), not 1/1,000 |
| Bricken and Pehlevan 2021 Table 1: d*_SNR 447, d*_CD 448 | **BUILT** as a test (`smap::tests`) |
| The critical-distance at Kanerva's point (n 1000, M 10^6, r 451, T 10^4) | **ALREADY HERE**, now pinned: the S-map gives 165, Bricken and Pehlevan report 188 (their reproduction of Kanerva 1988 Fig. 7.3), and a real store measured 204.3 (`runs/sdmscale/anchor_kanerva.txt`). The gap is recorded, not explained |
| `critical()` returns 1 under heavy load | **ALREADY HERE**, docs corrected: it returns the first distance not moved toward 0, which under heavy load is 1 because a read next to its word drifts out to a residual error. Behaviour unchanged, because settle-rs prints it |
| Kanerva 2009 p. 143: at n 10,000, less than a millionth of the space within 0.476 n, a thousand-millionth within 0.47 n | **BUILT** as a test |
| Jaeckel's selected-coordinate design (Jaeckel 1989 claims 86% more words at d 100) | **CANDIDATE**: the paper's claim is analytic, and the wiki's own experiment (`wikis/WIKI_SDR/code/experiments/critical_distance_results.txt`) compared unmatched memories (M 4,000 against 2,000). A matched measurement in this crate would earn it |
| Heteroassociative writes and pointer-chain sequences (K1993 P. 9; tested in `canonical/sdm.py`) | **CANDIDATE**: the standard SDM operation, tested only in the Python canonical code. A Rust port needs a sequence-recall test against a control |
| A configurable counter range (K1993's c = -15..15) | **CANDIDATE**: `Store` is fixed at one byte (+-127), which every settle-rs output depends on |
| Cleanup memory, bind and bundle, resonator networks (`canonical/hdc.py`, `resonator.py`) | **POINTER**: vector-symbolic algebra, outside the SDM itself |
| B&P's Eq. 26-27 capacity form and p*_Mem | **CANDIDATE**: a first check did not reproduce B&P's beta_Mem of 35.5 (a log-linear fit gave 28.4 to 28.8), so it is not a doc-test |

## 4. The SDM language model research (`experiments/track4/sdmllm/`)

Almost everything there is a fact about a gradient-trained product-key memory: the query gradient destroying
addressing, ties between k and between table sizes, usage collapse, Muon and AdamW, frozen values being ignored,
fading heads. None of it is a fact about a write-once Kanerva memory, so all of it is a **POINTER**. Two items are
general:

| finding | outcome |
|---|---|
| A streaming write-then-read with a normalised read is exact and costs nothing per M, by sorting (location, time, weight) triples (`MIXERFAST_RESULTS_2026-10-03.md` §1-2) | **CANDIDATE**: a method for a weighted read the crate does not have |
| The SETTLE capacity and race findings, quoted there | the same findings as section 1; built there |

## 5. Performance (measured on this machine)

Apple M5 Max, power mode high, `SDMSCALE_THREADS=1`, release build, minimum of interleaved runs; the box carried
load 9 to 66 from other work, so the numbers are a floor rather than a quiet-box figure.

| path | before | after | change |
|---|---|---|---|
| `Store::write_many`, 1,000 words, M 100,000, n 256 | 0.670 s | 0.385 s | the counter update is branch-free, so it vectorizes; identical counters and overflow count, held by a test against the old loop |
| `Store::awake`, 1,000 scans of M 100,000 | 0.16 s | unchanged | popcount scan; already fast |
| `Store::read_addresses`, 50 reads, M 100,000, 1,000 words | about 0.09 to 0.14 s | unchanged | |
| `Store::read_pulls_topk`, 50 reads, k 1,000 | about 0.14 to 0.21 s | unchanged | the dot products already vectorize |
| `theory::radius_for`, 200 calls at n 1,000 | 0.8 ms | 0.8 ms | now one code path with `hard_radius` |

## 6. Pruned

- `theory::radius_for` and `theory::hard_radius` were two code paths for one radius. They computed the same terms
  in the same order, so `radius_for` now calls `hard_radius` (66,000 cases checked; the returned share is held to
  the last bit by a test).
- `track::Geo::new` held `let _ = half;` and a `break` inside `for k in lo..=hi` that could not fire. Both are gone.

## 7. Number checks: reports whose prose disagrees with their own data

Quoted so nobody copies the prose figure into code. The data files are right.

- `runs/sdmscale`: "4 to 5 times below 0.1 M"; its own table gives 3.3 to 5.
- `runs/sdmradius`: the 30% capacity growth "about M^0.54"; its frontier files give 0.74 by least squares and
  0.77 from the endpoints. Its 40% claim "P90 1 to 3 at every M" has 0 in its own table for the SNR radius.
- `runs/sdmrefuse`: rule R "follows the oracle within 0.01 to 0.05" holds for the top-k cells only; on the address
  read 40% recall is 0.450 against the oracle's 0.506 at M 10^5, T 10.
