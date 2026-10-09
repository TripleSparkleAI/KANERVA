//! KANERVA's file face: the sdm-family statements read from text, in SETTLE's own syntax.
//!
//! A `.kanerva` file is a SETTLE program that holds only the sdm family: `sdm`, `softsdm`, `sdmscale`,
//! `refusal` and `contenttrack`, inside `model :name do ... end` and `run :name do ... end` blocks. The
//! `kanerva` command runs one with KANERVA's engine and nothing else. SETTLE mounts this same parser for the
//! same lines in a `.settle` program, so the two commands can never read a line two ways.
//!
//! ```
//! use kanerva::lang::{lex, Family, Names, Stmt};
//! use kanerva::words::Place;
//!
//! struct Nothing;
//! impl Names for Nothing {
//!     fn declared(&self, _: Family, _: &str) -> bool { false }
//! }
//! let toks = lex("sdm :s, word-size: 256, hard-locations: 2000", 1).unwrap();
//! let stmt = Family::Sdm.parse(Place::Model, &toks, 1, &Nothing).unwrap().unwrap();
//! assert!(matches!(stmt, Stmt::SdmDeclare { activation_radius: 112, .. }));
//! ```
//!
//! Layout: [`lex`] the tokens and the argument helpers · [`parse`] the typed statements and the one parser ·
//! [`say`] the printed lines · [`run`] the runner on KANERVA's engine. The keywords themselves live in
//! [`crate::words`], the one word list.
//!
//! <claudes_code_comments>
//! ** Function List **
//! (module root) - re-exports: lex, LangError, Tok, Family, Names, Stmt, From, Via, Wake, Cue, run, Runner
//! HEADS         - the statement heads that begin each family's lines, by place
//!
//! ** Technical Review **
//! - Four submodules: lex (tokens, kwargs, only, suggest, locate), parse (Family::parse and the Stmt types),
//!   say (every printed line), run (the runner over address, soft, store, refuse and track).
//! - HEADS is what a host needs to refuse these statements when it cannot run them (SETTLE built without its
//!   `sdm` feature): the declaration words in a model and the command words in a run.
//!
//! </claudes_code_comments>

pub mod lex;
pub mod parse;
pub mod run;
pub mod say;

pub use lex::{err, kw, kwargs, lex, locate, only, suggest, LangError, Tok};
pub use parse::{Cue, Family, From, Names, Stmt, Via, Wake};
pub use run::{run, Runner};

use crate::words::Place;

/// The words that begin each family's lines, by place: the declarations in a model, the commands in a run.
/// A host that cannot run the sdm family claims lines starting with these to refuse them by name.
pub const HEADS: &[(Place, &str)] =
    &[(Place::Model, "sdm"), (Place::Model, "softsdm"), (Place::Model, "sdmscale"), (Place::Run, "refusal"), (Place::Run, "contenttrack")];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::{Head, STATEMENTS};

    #[test]
    fn heads_are_exactly_the_declarations_and_commands_of_the_word_list() {
        let mut from_words: Vec<(Place, &str)> = STATEMENTS
            .iter()
            .filter(|s| matches!(s.head, Head::Declare | Head::Command))
            .map(|s| (s.places[0], s.verb))
            .collect();
        let mut heads = HEADS.to_vec();
        from_words.sort_by_key(|x| x.1);
        heads.sort_by_key(|x| x.1);
        assert_eq!(heads, from_words);
    }
}
