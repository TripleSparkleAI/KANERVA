//! Every example prints exactly what its `.out` file beside it records.
//!
//! <claudes_code_comments>
//! ** Function List **
//! check(name, got)              - compare an example's output with examples/<name>.out, naming the first differing line
//! each example test             - one test per example, so a failure names the example
//! a_changed_line_is_caught      - the negative control: an output with one changed line fails the comparison
//!
//! ** Technical Review **
//! - Each example keeps its whole output in `pub fn run() -> String`, and `main` only prints it, so this file
//!   includes the example as a module and calls `run()` without spawning cargo.
//! - The `.out` files were written by `cargo run --release --example <name> > examples/<name>.out`. Every number
//!   comes from fixed seeds. To refresh one after a deliberate change, run that command again and read the diff.
//! - The quickstart keeps its own print-as-you-go `main`, so it is checked by the README test instead
//!   (tests/readme.rs compares the README's printed block with examples/quickstart.out).
//!
//! </claudes_code_comments>

#[path = "../examples/write_and_read.rs"]
mod write_and_read;
#[path = "../examples/address_noise.rs"]
mod address_noise;
#[path = "../examples/capacity.rs"]
mod capacity;
#[path = "../examples/content_read.rs"]
mod content_read;
#[path = "../examples/keyed_notes.rs"]
mod keyed_notes;
#[path = "../examples/refusal.rs"]
mod refusal;
#[path = "../examples/soft_memory.rs"]
mod soft_memory;
#[path = "../examples/negative_controls.rs"]
mod negative_controls;
#[path = "../examples/rails.rs"]
mod rails;
#[path = "../examples/calibrated_refusal.rs"]
mod calibrated_refusal;
#[path = "../examples/sizing.rs"]
mod sizing;
#[path = "../examples/bit_balance.rs"]
mod bit_balance;

fn check(name: &str, got: &str) -> Result<(), String> {
    let path = format!("{}/examples/{}.out", env!("CARGO_MANIFEST_DIR"), name);
    let want = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path, e))?;
    if want == got {
        return Ok(());
    }
    let first = want.lines().zip(got.lines()).position(|(a, b)| a != b).unwrap_or(want.lines().count().min(got.lines().count()));
    Err(format!("{} differs at line {}:\n  recorded: {:?}\n  printed:  {:?}", name, first + 1, want.lines().nth(first), got.lines().nth(first)))
}

#[test]
fn write_and_read_prints_its_record() {
    check("write_and_read", &write_and_read::run()).unwrap();
}
#[test]
fn address_noise_prints_its_record() {
    check("address_noise", &address_noise::run()).unwrap();
}
#[test]
fn capacity_prints_its_record() {
    check("capacity", &capacity::run()).unwrap();
}
#[test]
fn content_read_prints_its_record() {
    check("content_read", &content_read::run()).unwrap();
}
#[test]
fn keyed_notes_prints_its_record() {
    check("keyed_notes", &keyed_notes::run()).unwrap();
}
#[test]
fn refusal_prints_its_record() {
    check("refusal", &refusal::run()).unwrap();
}
#[test]
fn soft_memory_prints_its_record() {
    check("soft_memory", &soft_memory::run()).unwrap();
}
#[test]
fn negative_controls_prints_its_record() {
    check("negative_controls", &negative_controls::run()).unwrap();
}

#[test]
fn rails_prints_its_record() {
    check("rails", &rails::run()).unwrap();
}
#[test]
fn calibrated_refusal_prints_its_record() {
    check("calibrated_refusal", &calibrated_refusal::run()).unwrap();
}
#[test]
fn sizing_prints_its_record() {
    check("sizing", &sizing::run()).unwrap();
}
#[test]
fn bit_balance_prints_its_record() {
    check("bit_balance", &bit_balance::run()).unwrap();
}

#[test]
fn a_changed_line_is_caught() {
    let got = keyed_notes::run().replace("under the mat", "under the rug");
    let err = check("keyed_notes", &got).unwrap_err();
    assert!(err.contains("line 1"), "{}", err);
}
