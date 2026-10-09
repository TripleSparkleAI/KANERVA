//! The one word list of KANERVA's statements: every statement of the sdm family, the keywords each one takes,
//! their kinds of value and their defaults, exactly as SETTLE accepts them. Data only: no parsing, no engine calls.
//!
//! Both faces of KANERVA read this list. The file face ([`crate::lang`]) refuses any keyword a statement does not
//! list here, so the parser cannot take a word the list lacks. The builder face reads it too, and its tests hold
//! the builder's method names equal to these words in both directions. SETTLE mounts the file face, so a `.settle`
//! program and a `.kanerva` program take the same words.
//!
//! ```
//! use kanerva::words::{statement, Place};
//!
//! let read = statement("sdm", "read").unwrap();
//! assert!(read.places.contains(&Place::Run));
//! assert!(read.words.iter().any(|w| w.name == "address-noise"));
//! ```
//!
//! <claudes_code_comments>
//! ** Function List **
//! statement(family, verb)  - the statement entry for a family and verb, if the list has it
//! Statement::word(name)    - one keyword of a statement
//! Statement::names()       - the keyword names, in the order error messages list them
//! canonical(family, word)  - an alias resolved to its spelling of record
//!
//! ** Technical Review **
//! - STATEMENTS holds thirteen entries in five families: `sdm` (declare, write, read), `softsdm` (declare, write,
//!   read, attend), `sdmscale` (declare, put, fill, read), `refusal` and `contenttrack` (run commands that build
//!   no memory). The keyword order of each entry is the order SETTLE's errors have always listed them in.
//! - A family's `ext` name is the name SETTLE's help prints in brackets (`[sdmrefuse] run: refusal ...`).
//! - RETIRED maps the words SETTLE used before Kanerva's terms to Kanerva's words; a retired word is refused with
//!   its new word. ALIASES are other spellings that still work: `write_samples:` is `write-samples:`.
//! - HELP holds the help lines of each family, verbatim as SETTLE prints them.
//! - `memory` is not here: it is a Hopfield memory recalled by SETTLE's own sampler, so it stays a SETTLE word.
//!
//! </claudes_code_comments>

/// Where a statement may stand: inside `model :name do ... end` or inside `run :name do ... end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// Inside a model block.
    Model,
    /// Inside a run block.
    Run,
}

/// How a statement begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    /// `sdm :s, ...`: the verb, then the new memory's name as a symbol.
    Declare,
    /// `s.read ...`: a declared memory's name, a dot, the verb.
    Method,
    /// `refusal ...`: the verb alone, then keywords.
    Command,
}

/// The positional part after the head, before any keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Args {
    /// Nothing positional.
    None,
    /// A symbol: `k.put :cat`.
    Name,
    /// A symbol, then optionally a quoted text: `s.write :note, "meet at nine"`.
    NameText,
    /// A number: `k.fill 500`.
    Count,
}

/// The kind of value a keyword takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    /// A number: `word-size: 256`.
    Number,
    /// A quoted text: `key: "blue heron"`.
    Text,
    /// A symbol. An empty list means any name (`read-address: :cat`); otherwise one of the listed names.
    Symbol(&'static [&'static str]),
}

/// What a keyword is when it is left out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultValue {
    /// This number.
    Num(f64),
    /// This symbol.
    Sym(&'static str),
    /// A rule, said in words (it depends on other keywords or on the run).
    Rule(&'static str),
    /// Absent unless written.
    Absent,
}

/// One keyword of one statement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Word {
    /// The keyword as written before the colon, in Kanerva's terms.
    pub name: &'static str,
    /// The kind of value it takes.
    pub value: Value,
    /// Its value when left out.
    pub default: DefaultValue,
}

/// One statement of the sdm family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Statement {
    /// The family: `sdm`, `softsdm`, `sdmscale`, `refusal` or `contenttrack`.
    pub family: &'static str,
    /// The verb: the family word for a declaration or a command, the method for a method (`read`).
    pub verb: &'static str,
    /// How the statement begins.
    pub head: Head,
    /// Where it may stand.
    pub places: &'static [Place],
    /// Its positional part.
    pub args: Args,
    /// Its keywords, in the order error messages list them.
    pub words: &'static [Word],
}

impl Statement {
    /// One keyword of this statement, by name.
    pub fn word(&self, name: &str) -> Option<&'static Word> {
        self.words.iter().find(|w| w.name == name)
    }

    /// The keyword names, in the order error messages list them.
    pub fn names(&self) -> Vec<&'static str> {
        self.words.iter().map(|w| w.name).collect()
    }
}

const fn w(name: &'static str, value: Value, default: DefaultValue) -> Word {
    Word { name, value, default }
}

use DefaultValue::{Absent, Num, Rule, Sym};
use Value::{Number, Symbol, Text};

const ANY: Value = Symbol(&[]);
const RUN_SEED: DefaultValue = Rule("the run's random stream, seeded 0x5eed at the start of each run block");
const MODEL: &[Place] = &[Place::Model];
const RUN: &[Place] = &[Place::Run];
const BOTH: &[Place] = &[Place::Model, Place::Run];

/// Every statement of the sdm family.
pub const STATEMENTS: &[Statement] = &[
    Statement {
        family: "sdm",
        verb: "sdm",
        head: Head::Declare,
        places: MODEL,
        args: Args::None,
        words: &[
            w("word-size", Number, Num(256.0)),
            w("hard-locations", Number, Num(2000.0)),
            w("activation-radius", Number, Rule("the radius that activates 2% of the address space (theory::radius_for(word-size, 0.02))")),
            w("seed", Number, Num(1.0)),
            w("fade", Number, Num(1.0)),
        ],
    },
    Statement { family: "sdm", verb: "write", head: Head::Method, places: BOTH, args: Args::NameText, words: &[w("key", Text, Absent)] },
    Statement {
        family: "sdm",
        verb: "read",
        head: Head::Method,
        places: RUN,
        args: Args::None,
        words: &[
            w("read-address", ANY, Rule("pure noise")),
            w("key", Text, Absent),
            w("address-noise", Number, Rule("0.3 from a read-address, 0 from a key")),
            w("iterated-reads", Number, Num(10.0)),
            w("via", Symbol(&["addresses", "pulls"]), Sym("addresses")),
            w("seed", Number, RUN_SEED),
        ],
    },
    Statement {
        family: "softsdm",
        verb: "softsdm",
        head: Head::Declare,
        places: MODEL,
        args: Args::None,
        words: &[
            w("word-size", Number, Num(256.0)),
            w("hard-locations", Number, Num(2000.0)),
            w("activation-probability", Number, Num(0.05)),
            w("softness", Number, Num(0.3)),
            w("gain", Number, Num(64.0)),
            w("seed", Number, Num(1.0)),
            w("write-samples", Number, Num(16.0)),
        ],
    },
    Statement { family: "softsdm", verb: "write", head: Head::Method, places: BOTH, args: Args::NameText, words: &[] },
    Statement {
        family: "softsdm",
        verb: "read",
        head: Head::Method,
        places: RUN,
        args: Args::None,
        words: &[
            w("read-address", ANY, Rule("pure noise")),
            w("address-noise", Number, Num(0.3)),
            w("rounds", Number, Num(3.0)),
            w("samples", Number, Num(16.0)),
            w("mode", Symbol(&["pass", "settle"]), Sym("pass")),
            w("burn", Number, Num(10.0)),
            w("seed", Number, RUN_SEED),
        ],
    },
    Statement {
        family: "softsdm",
        verb: "attend",
        head: Head::Method,
        places: RUN,
        args: Args::None,
        words: &[
            w("read-address", ANY, Rule("pure noise")),
            w("address-noise", Number, Num(0.3)),
            w("rounds", Number, Num(3.0)),
            w("seed", Number, RUN_SEED),
        ],
    },
    Statement {
        family: "sdmscale",
        verb: "sdmscale",
        head: Head::Declare,
        places: MODEL,
        args: Args::None,
        words: &[
            w("word-size", Number, Num(256.0)),
            w("hard-locations", Number, Num(100_000.0)),
            w("activation-probability", Number, Num(0.001)),
            w("activation-radius", Number, Rule("the radius for activation-probability, or for tolerate-noise when it is given")),
            w("tolerate-noise", Number, Absent),
            w("seed", Number, Num(1.0)),
        ],
    },
    Statement { family: "sdmscale", verb: "put", head: Head::Method, places: MODEL, args: Args::Name, words: &[] },
    Statement { family: "sdmscale", verb: "fill", head: Head::Method, places: MODEL, args: Args::Count, words: &[] },
    Statement {
        family: "sdmscale",
        verb: "read",
        head: Head::Method,
        places: RUN,
        args: Args::None,
        words: &[
            w("read-address", ANY, Rule("required")),
            w("address-noise", Number, Num(0.2)),
            w("iterated-reads", Number, Num(20.0)),
            w("via", Symbol(&["addresses", "pulls"]), Sym("addresses")),
            w("wake", Symbol(&["fixed", "density", "top"]), Sym("fixed")),
            w("seed", Number, RUN_SEED),
        ],
    },
    Statement {
        family: "refusal",
        verb: "refusal",
        head: Head::Command,
        places: RUN,
        args: Args::None,
        words: &[w("word-size", Number, Num(256.0)), w("load", Number, Num(1000.0)), w("level", Number, Num(0.01))],
    },
    Statement {
        family: "contenttrack",
        verb: "contenttrack",
        head: Head::Command,
        places: RUN,
        args: Args::None,
        words: &[
            w("word-size", Number, Num(256.0)),
            w("hard-locations", Number, Num(100_000.0)),
            w("load", Number, Num(1000.0)),
            w("address-noise", Number, Num(0.3)),
            w("block", Number, Num(0.0)),
            w("samples", Number, Num(200.0)),
        ],
    },
];

/// The statement for a family and a verb (`statement("sdm", "read")`), if the list has it.
pub fn statement(family: &str, verb: &str) -> Option<&'static Statement> {
    STATEMENTS.iter().find(|s| s.family == family && s.verb == verb)
}

/// Words SETTLE used before Kanerva's terms: the old word, the new word, and Kanerva's phrase for it. A retired
/// word is refused with its new word wherever the statement takes the new one.
pub const RETIRED: &[(&str, &str, &str)] = &[
    ("cue", "read-address", "Kanerva's retrieval address"),
    ("damage", "address-noise", "Kanerva's noise in the address"),
    ("locations", "hard-locations", "Kanerva's hard locations"),
    ("radius", "activation-radius", "Kanerva's activation radius"),
    ("fire", "activation-probability", "Kanerva's probability of activation"),
    ("iterations", "iterated-reads", "Kanerva's iterated reading"),
    ("size", "word-size", "Kanerva's word size"),
    ("tolerate", "tolerate-noise", "the address-noise to tolerate"),
];

/// Other spellings that still work: (family, the spelling, the spelling of record). `write_samples:` was the
/// one keyword of the family written with an underscore; `write-samples:` is its spelling of record.
pub const ALIASES: &[(&str, &str, &str)] = &[("softsdm", "write_samples", "write-samples")];

/// The spelling of record of a keyword in a family (an alias resolved; any other word unchanged).
pub fn canonical<'a>(family: &str, word: &'a str) -> &'a str {
    match ALIASES.iter().find(|(f, old, _)| *f == family && *old == word) {
        Some((_, _, new)) => new,
        None => word,
    }
}

/// The statement families as SETTLE's help lists them: the bracketed name, then each help line verbatim.
pub const HELP: &[(&str, &[&str])] = &[
    (
        "sdm",
        &[
            "model: sdm :s, word-size: 256, hard-locations: 2000, activation-radius: 112, seed: 1, fade: 1",
            "model: s.write :cat   /   s.write :note, \"text\"   /   s.write :diary, \"text\", key: \"secret\"",
            "run: s.write ... (as in a model)",
            "run: s.read read-address: :cat, address-noise: 0.3, iterated-reads: 10, via: :addresses, seed: 1   (via: :pulls too)",
            "run: s.read key: \"secret\"   /   s.read   (from noise)",
        ],
    ),
    (
        "softsdm",
        &[
            "model: softsdm :s, word-size: 256, hard-locations: 2000, activation-probability: 0.05, softness: 0.3, gain: 64, seed: 1, write-samples: 16",
            "model: s.write :cat   /   s.write :note, \"some text\"",
            "run: s.write :cat   /   s.write :note, \"some text\"",
            "run: s.read read-address: :cat, address-noise: 0.3, rounds: 3, samples: 16, mode: :pass, seed: 1   (mode :settle adds burn: 10)",
            "run: s.attend read-address: :cat, address-noise: 0.3, rounds: 3, seed: 1",
        ],
    ),
    (
        "sdmscale",
        &[
            "model: sdmscale :k, word-size: 256, hard-locations: 100000, activation-probability: 0.001, seed: 1   (activation-radius: overrides activation-probability; tolerate-noise: 0.3 picks the activation-radius for 30% address-noise)",
            "model: k.put :cat   /   k.fill 500   (random patterns, one fill per store)",
            "run: k.read read-address: :cat, address-noise: 0.2, iterated-reads: 20, via: :addresses, seed: 1   (via: :pulls too, with wake: :fixed | :density | :top)",
        ],
    ),
    ("sdmrefuse", &["run: refusal word-size: 256, load: 3000, level: 0.01   (the travel rule's threshold and the nearest-neighbour ceiling; no memory built)"]),
    (
        "sdmtrack",
        &["run: contenttrack word-size: 256, hard-locations: 100000, load: 3000, address-noise: 0.3, block: 0, samples: 200   (TRACK-C's predicted recall for the content reads; no memory built)"],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_statement_is_found_and_its_words_are_unique() {
        for s in STATEMENTS {
            assert_eq!(statement(s.family, s.verb), Some(s));
            let mut names = s.names();
            names.sort();
            names.dedup();
            assert_eq!(names.len(), s.words.len(), "{} {} lists a keyword twice", s.family, s.verb);
            for name in s.names() {
                assert!(name.chars().all(|c| c.is_ascii_lowercase() || c == '-'), "`{}` is not a hyphenated lowercase word", name);
            }
        }
        // the negative control: a verb the list lacks is not found
        assert_eq!(statement("sdm", "recall"), None);
    }

    #[test]
    fn every_help_line_names_only_listed_keywords() {
        for (ext, lines) in HELP {
            for line in *lines {
                for word in line.split_whitespace().filter_map(|t| t.trim_start_matches('(').strip_suffix(':')) {
                    if matches!(word, "model" | "run") {
                        continue;
                    }
                    assert!(STATEMENTS.iter().any(|s| s.word(word).is_some()), "[{}] help names `{}:`, which no statement takes", ext, word);
                }
            }
        }
    }

    #[test]
    fn the_alias_resolves_and_nothing_else_moves() {
        assert_eq!(canonical("softsdm", "write_samples"), "write-samples");
        assert_eq!(canonical("sdm", "write_samples"), "write_samples");
        assert_eq!(canonical("softsdm", "seed"), "seed");
    }
}
