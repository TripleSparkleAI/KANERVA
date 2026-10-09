//! The Rails face says exactly the words in the word list: every keyword is a method of the same name, and every
//! method is a keyword. Both directions, for every statement and verb.
//!
//! <claudes_code_comments>
//! ** Function List **
//! Face::read(src)                           - the marked types, settings blocks and verb methods in rails.rs
//! problems(src)                             - every difference between the face and crate::words, both ways
//! the_rails_face_says_exactly_the_word_list - the check on the real source
//! a_missing_method_is_caught / an_extra_method_is_caught / a_missing_verb_is_caught / an_unknown_marker_is_caught
//!                                           - the controls: each aimed defect, planted in a copy of the source, is named
//! the_scan_sees_the_whole_face              - the corpus floor: the scan reads at least 60 names
//!
//! ** Technical Review **
//! - rails.rs marks its words: `// statement: <s>` before each type, `// words: <s>` or `// words: <s>.<verb>` before
//!   the impl block of settings (one `pub fn` per keyword, the block ending at a `}` in column 0), and
//!   `// verb: <s>.<verb>` before each method that does a verb (its parameter names are positional words, or a
//!   keyword taken positionally, like write_keyed's `key`).
//! - The list is crate::words::STATEMENTS: one entry per (family, verb). A declaration's settings must equal its
//!   keywords plus `name` (the memory's name, `sdm :s`); a command's settings its keywords; a method's settings
//!   together with its marked methods' parameters must equal its keywords plus its positional arguments
//!   (Args::Name `name`, Args::NameText `name` and `text`, Args::Count `count`). Names compare after
//!   rails::rust_name (hyphens to underscores), so `read-address` is `read_address`.
//! - Each control plants one defect in a copy of the real source and asserts the problem list names it, so a
//!   checker that returned nothing could not pass them.
//!
//! </claudes_code_comments>

use kanerva::rails::rust_name;
use kanerva::words::{Args, Head, STATEMENTS};
use std::collections::{BTreeMap, BTreeSet};

const SRC: &str = include_str!("../src/rails.rs");

#[derive(Default)]
struct Face {
    statements: BTreeSet<String>,
    /// "sdm" or "sdm.read" -> setting method names
    settings: BTreeMap<String, BTreeSet<String>>,
    /// "sdm.write" -> parameter names of every method marked with it
    verbs: BTreeMap<String, BTreeSet<String>>,
}

fn fn_name_and_params(line: &str) -> Option<(String, Vec<String>)> {
    let rest = line.trim_start().strip_prefix("pub fn ")?;
    let open = rest.find('(')?;
    let name = rest[..open].trim().to_string();
    let close = rest.rfind(')')?;
    let params = rest[open + 1..close]
        .split(',')
        .filter_map(|p| p.split_once(':').map(|(n, _)| n.trim().trim_start_matches("mut ").to_string()))
        .collect();
    Some((name, params))
}

impl Face {
    fn read(src: &str) -> Face {
        let mut f = Face::default();
        let lines: Vec<&str> = src.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let l = lines[i].trim();
            if let Some(s) = l.strip_prefix("// statement: ") {
                f.statements.insert(s.trim().to_string());
            } else if let Some(key) = l.strip_prefix("// words: ") {
                let set = f.settings.entry(key.trim().to_string()).or_default();
                i += 1;
                while i < lines.len() && lines[i] != "}" {
                    if let Some((name, _)) = fn_name_and_params(lines[i]) {
                        set.insert(name);
                    }
                    i += 1;
                }
            } else if let Some(key) = l.strip_prefix("// verb: ") {
                let set = f.verbs.entry(key.trim().to_string()).or_default();
                let mut j = i + 1;
                while j < lines.len() && lines[j].trim().starts_with("///") {
                    j += 1;
                }
                if let Some((_, params)) = lines.get(j).and_then(|l| fn_name_and_params(l)) {
                    set.extend(params);
                } else {
                    set.insert(format!("<no method after the marker at line {}>", i + 1));
                }
            }
            i += 1;
        }
        f
    }
}

fn diff(what: &str, want: &BTreeSet<String>, got: &BTreeSet<String>, out: &mut Vec<String>) {
    for w in want.difference(got) {
        out.push(format!("{}: the word `{}` has no method or parameter of that name", what, w));
    }
    for g in got.difference(want) {
        out.push(format!("{}: `{}` is not in the word list", what, g));
    }
}

fn positional(args: Args) -> &'static [&'static str] {
    match args {
        Args::None => &[],
        Args::Name => &["name"],
        Args::NameText => &["name", "text"],
        Args::Count => &["count"],
    }
}

fn problems(src: &str) -> Vec<String> {
    let f = Face::read(src);
    let mut out = Vec::new();
    let families: BTreeSet<String> = STATEMENTS.iter().map(|s| s.family.to_string()).collect();
    diff("statement types", &families, &f.statements, &mut out);
    let mut keys = BTreeSet::new();
    for s in STATEMENTS {
        let key = if s.verb == s.family { s.family.to_string() } else { format!("{}.{}", s.family, s.verb) };
        let mut want: BTreeSet<String> = s.words.iter().map(|w| rust_name(w.name)).collect();
        want.extend(positional(s.args).iter().map(|a| a.to_string()));
        if s.head == Head::Declare {
            want.insert("name".to_string());
        }
        let mut got: BTreeSet<String> = f.settings.get(&key).cloned().unwrap_or_default();
        if s.head == Head::Method {
            match f.verbs.get(&key) {
                Some(params) => got.extend(params.iter().cloned()),
                None => out.push(format!("{}: no method is marked `// verb: {}`", key, key)),
            }
        }
        diff(&key, &want, &got, &mut out);
        keys.insert(key);
    }
    for k in f.settings.keys().chain(f.verbs.keys()) {
        if !keys.contains(k) {
            out.push(format!("`{}` is marked in rails.rs and is no statement or verb in the word list", k));
        }
    }
    out
}

#[test]
fn the_rails_face_says_exactly_the_word_list() {
    let p = problems(SRC);
    assert!(p.is_empty(), "{}", p.join("\n"));
}

#[test]
fn the_scan_sees_the_whole_face() {
    let f = Face::read(SRC);
    let names: usize = f.settings.values().map(|s| s.len()).sum::<usize>() + f.verbs.values().map(|s| s.len()).sum::<usize>();
    assert!(names >= 60, "the scan read only {} names; a wrong file or a broken marker would read too few", names);
    assert_eq!(f.statements.len(), 5);
}

#[test]
fn a_missing_method_is_caught() {
    let src = SRC.replacen("pub fn burn(mut self", "pub fn burnt(mut self", 1);
    assert_ne!(src, SRC, "the mutation must land");
    let p = problems(&src).join("\n");
    assert!(p.contains("softsdm.read: the word `burn` has no method"), "{}", p);
    assert!(p.contains("softsdm.read: `burnt` is not in the word list"), "{}", p);
}

#[test]
fn an_extra_method_is_caught() {
    let src = SRC.replacen("    /// Every write first multiplies every bit-counter by this (1: nothing fades).\n", "    pub fn radius(self, r: usize) -> Sdm { self.activation_radius(r) }\n", 1);
    assert_ne!(src, SRC, "the mutation must land");
    let p = problems(&src).join("\n");
    assert!(p.contains("sdm: `radius` is not in the word list"), "{}", p);
}

#[test]
fn a_missing_verb_is_caught() {
    let src = SRC.replace("// verb: sdmscale.fill\n", "");
    assert_ne!(src, SRC, "the mutation must land");
    let p = problems(&src).join("\n");
    assert!(p.contains("sdmscale.fill: no method is marked"), "{}", p);
}

#[test]
fn an_unknown_marker_is_caught() {
    let src = SRC.replacen("// verb: sdm.read\n", "// verb: sdm.forget\n", 1);
    assert_ne!(src, SRC, "the mutation must land");
    let p = problems(&src).join("\n");
    assert!(p.contains("`sdm.forget` is marked in rails.rs"), "{}", p);
}
