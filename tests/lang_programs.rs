//! The file face: every program in programs/ prints exactly its `.out`, and the `kanerva` command behaves.
//!
//! <claudes_code_comments>
//! ** Function List **
//! programs()                        - every programs/*.kanerva, sorted
//! every_program_prints_its_out      - each program's run() equals its .out, line for line
//! a_changed_line_is_caught          - the negative control: one changed line fails the comparison
//! the_command_runs_a_program        - `kanerva <file>` prints the run's lines, exit 0
//! the_command_points_at_an_error    - an error prints the line, a caret and the column, exit 2
//! the_command_refuses_what_only_settle_runs - via: :pulls and a SETTLE statement are refused by name
//! the_command_help_lists_every_family - --help names every family; --version; an unknown flag exits 2
//!
//! ** Technical Review **
//! - The `.out` files were written by `kanerva programs/<name>.kanerva > programs/<name>.out` and checked equal to
//!   `settle programs/<name>.kanerva` (a .kanerva file is a valid .settle program); SETTLE's own test
//!   `tests/oneparser_parity.rs` holds that equality on every run.
//! - The command is run through `CARGO_BIN_EXE_kanerva`, the binary cargo builds for this test.
//!
//! </claudes_code_comments>

use std::path::PathBuf;
use std::process::Command;

fn programs() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("programs");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "kanerva")).collect();
    v.sort();
    v
}

fn compare(want: &str, got: &str) -> Result<(), String> {
    for (i, (w, g)) in want.lines().zip(got.lines()).enumerate() {
        if w != g {
            return Err(format!("line {} differs:\n  want {}\n  got  {}", i + 1, w, g));
        }
    }
    if want.lines().count() != got.lines().count() {
        return Err(format!("{} lines wanted, {} got", want.lines().count(), got.lines().count()));
    }
    Ok(())
}

#[test]
fn every_program_prints_its_out() {
    let ps = programs();
    assert!(ps.len() >= 4, "programs/ holds {} programs", ps.len());
    for p in ps {
        let src = std::fs::read_to_string(&p).unwrap();
        let got = kanerva::lang::run(&src).unwrap_or_else(|e| panic!("{}: {}", p.display(), e)).join("\n") + "\n";
        let want = std::fs::read_to_string(p.with_extension("out")).unwrap();
        if let Err(e) = compare(&want, &got) {
            panic!("{}: {}", p.display(), e);
        }
    }
}

#[test]
fn a_changed_line_is_caught() {
    let p = &programs()[0];
    let src = std::fs::read_to_string(p).unwrap();
    let got = kanerva::lang::run(&src).unwrap().join("\n") + "\n";
    let mut lines: Vec<String> = got.lines().map(String::from).collect();
    lines[0].push('!');
    assert!(compare(&got, &(lines.join("\n") + "\n")).is_err());
    assert!(compare(&got, &got).is_ok());
}

fn kanerva(args: &[&str], dir: &std::path::Path) -> (String, String, i32) {
    let o = Command::new(env!("CARGO_BIN_EXE_kanerva")).args(args).current_dir(dir).output().unwrap();
    (String::from_utf8(o.stdout).unwrap(), String::from_utf8(o.stderr).unwrap(), o.status.code().unwrap_or(-1))
}

fn scratch(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kanerva-lang-test-{}-{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("p.kanerva"), src).unwrap();
    dir
}

#[test]
fn the_command_runs_a_program() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let (out, err, code) = kanerva(&["programs/sdm.kanerva"], &dir);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(out, std::fs::read_to_string(dir.join("programs/sdm.out")).unwrap());
}

#[test]
fn the_command_points_at_an_error() {
    let dir = scratch("error", "model :m do\n  sdm :s, word-size: 64, sed: 1\nend\n");
    let (out, err, code) = kanerva(&["p.kanerva"], &dir);
    assert_eq!((out.as_str(), code), ("", 2));
    assert_eq!(
        err,
        "kanerva: line 2: sdm does not take `sed:`; did you mean `seed:`?\n   2 |   sdm :s, word-size: 64, sed: 1\n     |                          ^^^^ column 26\n"
    );
}

#[test]
fn the_command_refuses_what_only_settle_runs() {
    let src = "model :m do\n  sdm :s, word-size: 64, hard-locations: 100\n  s.write :cat\nend\nrun :m do\n  s.read read-address: :cat, via: :pulls\nend\n";
    let (_, err, code) = kanerva(&["p.kanerva"], &scratch("pulls", src));
    assert_eq!(code, 2);
    assert!(err.starts_with("kanerva: line 6: via: :pulls settles the pulls of a SETTLE model"), "{}", err);
    let (_, err, code) = kanerva(&["p.kanerva"], &scratch("thing", "model :m do\n  thing :rain\nend\n"));
    assert_eq!(code, 2);
    assert!(err.contains("kanerva knows only the sdm family; a SETTLE statement runs with `settle`"), "{}", err);
    // the control: the same memory read via :addresses runs
    let (out, _, code) = kanerva(&["p.kanerva"], &scratch("addresses", &src.replace(", via: :pulls", "")));
    assert_eq!(code, 0);
    assert!(out.starts_with("read :s from read-address :cat"), "{}", out);
}

#[test]
fn the_command_help_lists_every_family() {
    let dir = std::env::temp_dir();
    let (out, _, code) = kanerva(&["--help"], &dir);
    assert_eq!(code, 0);
    for f in kanerva::lang::Family::ALL {
        assert!(out.contains(&format!("[{}]", f.ext_name())), "{} missing from --help", f.ext_name());
    }
    let (out, _, code) = kanerva(&["--version"], &dir);
    assert_eq!((out.trim(), code), (format!("kanerva {}", env!("CARGO_PKG_VERSION")).as_str(), 0));
    let (_, err, code) = kanerva(&["--frobnicate"], &dir);
    assert_eq!(code, 2);
    assert!(err.contains("unknown option --frobnicate"), "{}", err);
}
