# Changelog

## Unreleased (2026-10-06): heteroassociative writes (lane MNISTCRATE)

- Added: `kanerva::hetero`, a memory whose data-word is not its address: `Hetero::from_addresses(addresses, width)`
  over the caller's hard-locations, `write(address, data, wake)` and `read(address, wake)`, int32 bit-counters,
  `dists`, `activate`, `clear`, `writes_at`. Two wakes: `Wake::Radius(r)`, Kanerva's activation, and
  `Wake::Nearest(k)`, the smallest activation-radius whose ball holds at least k hard-locations (ties wake), which is a
  departure from Kanerva's design for hard-locations placed on the data. It is the memory the SETTLE site's LEARN SDM
  page uses to learn MNIST (a 784-bit digit as the address, its label as a 10-bit data-word), now run on the crate in
  the browser through kanerva-wasm's `kw_hetero_*` door. No existing public item changed.

## Unreleased (2026-10-06): the Rails face (lane KANERVARAILS)

- Added: `kanerva::rails`, a builder per sdm statement (`Sdm::build().word_size(256)...`, `s.write("cat")`,
  `s.read("cat").address_noise(0.2).answer()`) over the one word list `kanerva::words`, printing the file face's
  lines byte for byte; parity tests both ways.

## Unreleased (2026-10-06): what the campaign measured, on the front door (lane KANERVAPERFECT)

Every addition is a rule or a predictor the SETTLE campaign measured, or a check against a paper on disk.
`DISCOVERIES.md` maps each finding to what it became, or why it did not. No public item was renamed or removed.

- `calibrated` (new): the calibrated refusal, SDMTRACK's calibrated-minimum rule (refused at least 0.98 of
  never-stored read-addresses in 18 of 18 cells, where the travel rule did in 12). `CalibratedRefusal`
  (`from_probes`, `calibrate`, `score`, `accepts`, `floor`, `is_degenerate`, `read`), `Watch`, `signals`,
  `CheckedRead`; on `Sdm`: `calibrated_refusal`, `calibrated_content_refusal`, `read_checked`, `content_k`.
- `sizing` (new): `snr_activation_radius`, `noise_activation_radius`; on `Sdm`: `with_snr_radius`,
  `for_address_noise`, `predicted_failure_rate` and `predicted_capacity_track` (TRACK-P, measured error 0.015).
- `diagnose` (new): `balance` / `Balance`, `Store::shared_locations`, `race` / `Race` (with `margin`).
- `erase` (new): `Store::erase`, `Sdm::erase`, `Sdm::read_without`, `EraseRefused`.
- `theory`: `optimal_activation_probability`, `asymptotic_capacity`, `capacity_fraction` (Kanerva 1993 P. 15-17).
  `radius_for` now calls `hard_radius`; the two computed the same terms in the same order (66,000 cases checked),
  and `hard_radius` no longer builds the whole pmf table, so neither got slower.
- `store`: `Store::from_addresses` (a store over a matrix the caller holds). The write's counter update is
  branch-free and vectorizes: `write_many` of 1,000 words into M 100,000 went from 0.670 s to 0.385 s (one
  thread, interleaved A/B, minimum of 4). The counters and the overflow count are the same; a test holds the
  update to the old branching loop.
- `Cargo.toml`: `[profile.test] opt-level = 3`, so a plain `cargo test` runs the examples in 14 s rather than
  339 s (measured); overflow checks stay on.
- `store::threads()` is 1 on wasm32, so the parallel parts (`write_many`, calibration) run inline there instead
  of failing to spawn. Native builds are unchanged.
- `track`: `TailCal` derives `Clone`, `Debug`, `PartialEq`; two dead statements in `Geo::new` removed.
- Docs: `smap::critical` says what it returns under heavy load; `Sdm::predicted_capacity` records the S-map's
  measured optimism at heavy noise; `theory` cites Kanerva 1993's sample memory and where "451 at p 0.001"
  comes from.
- Tests: the papers on disk (Kanerva 1993's sample memory and capacity, Kanerva 1988 Table 7.1 as Jaeckel 1989
  quotes it, Kanerva 2009 p. 143, FKB 1989's 2r bound, Bricken and Pehlevan's Table 1, the critical-distance
  at Kanerva's point pinned at 165). Three examples with recorded output: `calibrated_refusal`, `sizing`,
  `bit_balance`.

## Unreleased (2026-10-05): the front door, for release

- New module `sdm`, re-exported at the crate root: `Sdm` (new, with_activation_probability, write, write_all,
  read, read_iterated, read_by_content, write_note, read_note, predicted_capacity, critical_distance, activated,
  store, store_mut), `ReadResult` (with `travelled`), `Refusal`, `SoftSdm`, `random_word`, `word_for`,
  `hamming` and `DEFAULT_ITERATED_READS`. Each call is one call into the lab modules; the arithmetic is unchanged
  and `src/sdm.rs`'s tests check the front door reads the same bits as the store.
- Every public item has a doc comment and a doc-test (222 doc-tests); `#![warn(missing_docs)]` keeps it so.
- Eight new examples, one per idea (`write_and_read`, `address_noise`, `capacity`, `content_read`,
  `keyed_notes`, `refusal`, `soft_memory`, `negative_controls`), each with its output recorded in
  `examples/<name>.out` and checked by `tests/examples.rs`. The quickstart now uses the front door and prints
  "words", "hard-locations", "activation-radius" and "iterated-reads".
- `tests/readme.rs` runs every Rust snippet in the README and checks the README's printed output.
- Parameters renamed to Kanerva's terms (no caller breaks, a parameter name is not API): `cue` is
  `read_address`, `dmg` is `address_noise`, and `fire` is `activation_probability` in `theory::hard_radius` and
  `soft::Machine::new`. The field `soft::Machine::fire` keeps its name because settle-rs reads it.
- `Rng` derives `Clone` and `Debug`.
- Edition 2024 (merged doc-tests) and `rust-version = "1.87"`, checked with `cargo +1.87 test --release`.
- Clippy clean: `div_ceil` and `is_multiple_of` where the code did the same by hand (exact equivalents);
  `needless_range_loop` and `too_many_arguments` allowed with the reason in `lib.rs`.
- Cargo.toml: `repository`, `keywords`, `categories`, `license-file` (the placeholder), `publish = false` kept.
- Docs: hyphenated terms throughout (hard-locations, activation-radius, critical-distance, iterated-reads,
  word-size), the README rebuilt as a library README (quickstart, the words, the front door, examples, the
  SETTLE seam), KANERVA_TERMS.md checked word for word against the four papers on disk (K2010's page corrected
  from p. 2 to p. 3, FKB1989's on-disk copy named, the retired-keyword error quoted as the interpreter prints it).
- `RELEASE_CHECKLIST.md`: what the owner must decide before the repository is shared.

## Unreleased (2026-10-01): Kanerva's terms

- Renamed for Kanerva's own words: `keys::keyed_cue` is `keys::keyed_read_address`, `bits::damage` is
  `bits::add_address_noise`, `codes::damaged` is `codes::with_address_noise`, `smap::radius_for_damage` is
  `smap::radius_for_address_noise` (and its `_poisson` twin). The arithmetic is unchanged.
- Docs, comments and the quickstart's printout say read-address, address-noise, hard locations, activation
  radius and bit-counters. Sources and quotes: `KANERVA_TERMS.md`.

## 0.1.0 (2026-10-01)

First release, inside the dwarfstar repository at `SETTLE/kanerva/`.

- Lifted out of the SETTLE interpreter's seven SDM files (`sdm.rs`, `memory.rs`, `softsdm.rs`,
  `sdmradius.rs`, `sdmrefuse.rs`, `sdmscale.rs`, `sdmtrack.rs`) as twelve modules: `rng`, `codes`,
  `keys`, `bits`, `theory`, `address`, `store`, `smap`, `soft`, `refuse`, `track`, `hopfield`.
- The arithmetic is unchanged. settle-rs now calls this crate; its 215 tests pass with the same names,
  and 45 recorded program outputs match the outputs taken before the move (record and comparer in
  `SETTLE/runs/sdmkit/`).
- New: `address::Addresses` and `address::iterated_read` (Kanerva's read over counters the caller keeps),
  `bits::pack_f64`, `codes::bytes_bits`, `codes::noise`, `theory::log_choose` made public,
  `smap::Lazy::step` and `smap::largest_t` made public, `Store::ctr` made public.
- Tests: 46 unit tests and 3 cross-module tests; an example, `examples/quickstart.rs`.
- Licence: not yet chosen.
