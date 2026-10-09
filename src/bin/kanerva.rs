//! The `kanerva` command: run a `.kanerva` program of sdm-family statements and print what it says.
//!
//! <claudes_code_comments>
//! ** Function List **
//! usage()  - the help text: usage, every statement by family, the example programs
//! main()   - reads the arguments (--help, --version, one program path), runs it, prints its output
//! report() - an error as printed: the message, the program line with a caret at the place, the rest
//!
//! ** Technical Review **
//! - `kanerva <program.kanerva>` runs one program on KANERVA's engine (`kanerva::lang::run`). Output is printed
//!   only when the whole program succeeds; an error goes to standard error with the line it points at and a
//!   caret (`lang::locate`), and the exit status is 2. The same program run by `settle` prints the same lines.
//! - `--help`, `-h` or no argument prints the usage; `--version` or `-V` prints `kanerva <version>`. Any other
//!   argument starting with `-` is refused with exit status 2, so a mistyped flag is never read as a file.
//!
//! </claudes_code_comments>

use kanerva::lang::{locate, run, Family};

fn usage() {
    println!("kanerva <program.kanerva>\nkanerva --version\n\nKANERVA: write words into a sparse distributed memory and read them back, in Kanerva's terms.");
    println!("A .kanerva program is a SETTLE program holding only these statements; `settle` runs the same file.");
    println!("\nStatements, by family:");
    for f in Family::ALL {
        for s in f.help() {
            println!("  [{}] {}", f.ext_name(), s);
        }
    }
    println!("\nExamples: programs/sdm.kanerva · programs/softsdm.kanerva · programs/sdmscale.kanerva · programs/theory.kanerva");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        None | Some("--help") | Some("-h") => {
            usage();
            return;
        }
        Some("--version") | Some("-V") => {
            println!("kanerva {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some(f) if f.starts_with('-') => {
            eprintln!("kanerva: unknown option {} (see kanerva --help)", f);
            std::process::exit(2);
        }
        _ => {}
    }
    if args.len() > 2 {
        eprintln!("kanerva: one program at a time; got {} arguments (see kanerva --help)", args.len() - 1);
        std::process::exit(2);
    }
    let path = &args[1];
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("kanerva: cannot read {}: {}", path, e);
            std::process::exit(2);
        }
    };
    match run(&src) {
        Ok(lines) => lines.iter().for_each(|l| println!("{}", l)),
        Err(e) => {
            eprint!("{}", report(&src, &e.0));
            std::process::exit(2);
        }
    }
}

/// An error as the command prints it: the message's first line, the program line it points at with a caret
/// under the place, then any further lines of the message.
fn report(src: &str, msg: &str) -> String {
    let mut lines = msg.lines();
    let mut out = format!("kanerva: {}\n", lines.next().unwrap_or(""));
    if let Some((ln, col, width)) = locate(src, msg) {
        let code = src.lines().nth(ln - 1).unwrap_or("");
        let gutter = ln.to_string().len().max(4);
        out += &format!("{:>g$} | {}\n", ln, code.trim_end(), g = gutter);
        out += &format!("{:>g$} | {}{} column {}\n", "", " ".repeat(col - 1), "^".repeat(width), col, g = gutter);
    }
    for l in lines {
        out += l;
        out += "\n";
    }
    out
}
