# RELEASE_CHECKLIST - what the owner decides before KANERVA is shared

Everything below is a decision only the owner can make. Nothing here is decided by a lane, and nothing was
invented to fill a gap.

## Decisions owed

1. **The licence.** `LICENSE.md` is a placeholder that grants nothing. Choose a licence, replace `LICENSE.md`
   with its text, and in `Cargo.toml` replace `license-file = "LICENSE.md"` with `license = "<SPDX id>"`. The
   README's licence box and the CHANGELOG line "Licence: not yet chosen" change with it.
2. **Public or private.** The repository `github.com/triplesparkle/KANERVA` exists and reads PRIVATE (checked
   with `gh repo view` on 2026-10-05). Making it public is the owner's act; the site's one switch for it is
   `kanerva: { repo: 'KANERVA', visibility: ... }` in `SETTLE/settle-site/src/repo.js`.
3. **crates.io or git only.** `publish = false` keeps `cargo publish` from running. Publishing needs a licence,
   an owner account, and a check that the name `kanerva` is free on crates.io (not checked here).
4. **Authors.** `Cargo.toml` names no authors. Add them if the owner wants names on the crate.
5. **The version.** The crate is 0.1.0 with an "Unreleased" changelog entry on top. Cut 0.2.0 (the front door
   is new public API) or keep 0.1.0, and date the entry.
6. **The research-repository paths in the docs.** KANERVA_TERMS.md and KANERVA_LECTURES_CONDENSED.md name files
   in the research repository (the papers under `wikis/WIKI_SDR/papers/` and the vault of lecture transcripts), and the README
   cites `REPORT_*.md` files there. They are pointers, not secrets, but a public reader cannot open them.
   Decide whether to keep them as provenance, or to cite the papers by their public references alone.

## Checks to run before sharing

```bash
cd SETTLE/kanerva
cargo test --release                        # unit, cross-module, examples, README, doc-tests
cargo clippy --all-targets                  # no warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo +1.87 test --release                  # the declared rust-version
bash ../tools/export_settle_repos.sh --dry-run --only KANERVA
DEST_ROOT=<a scratch folder> bash ../tools/export_settle_repos.sh --only KANERVA   # then read the scan lines
```

The export copies only tracked files, refuses never-export paths and scans for secret patterns; read its
`scan:` line and the reviewed list (`SETTLE/tools/export_settle_repos.reviewed`) before any push.
