# Changelog

## Unreleased (2026-10-01): Kanerva's terms

- Renamed for Kanerva's own words: `keys::keyed_cue` is `keys::keyed_read_address`, `bits::damage` is
  `bits::add_address_noise`, `codes::damaged` is `codes::with_address_noise`, `smap::radius_for_damage` is
  `smap::radius_for_address_noise` (and its `_poisson` twin). The arithmetic is unchanged.
- Docs, comments and the quickstart's printout say read-address, address-noise, hard locations, activation
  radius and bit-counters. Sources and quotes: `KANERVA_TERMS.md`.

## 0.1.0 (2026-10-01)

First release, inside the dwarfstar repository at `experiments/thermosim/kanerva/`.

- Lifted out of the SETTLE interpreter's seven SDM files (`sdm.rs`, `memory.rs`, `softsdm.rs`,
  `sdmradius.rs`, `sdmrefuse.rs`, `sdmscale.rs`, `sdmtrack.rs`) as twelve modules: `rng`, `codes`,
  `keys`, `bits`, `theory`, `address`, `store`, `smap`, `soft`, `refuse`, `track`, `hopfield`.
- The arithmetic is unchanged. settle-rs now calls this crate; its 215 tests pass with the same names,
  and 45 recorded program outputs match the outputs taken before the move (record and comparer in
  `experiments/thermosim/runs/sdmkit/`).
- New: `address::Addresses` and `address::iterated_read` (Kanerva's read over counters the caller keeps),
  `bits::pack_f64`, `codes::bytes_bits`, `codes::noise`, `theory::log_choose` made public,
  `smap::Lazy::step` and `smap::largest_t` made public, `Store::ctr` made public.
- Tests: 46 unit tests and 3 cross-module tests; an example, `examples/quickstart.rs`.
- Licence: not yet chosen.
