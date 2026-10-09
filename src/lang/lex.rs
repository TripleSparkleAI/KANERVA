//! The tokens of a `.kanerva` line, and the small argument helpers every statement uses.
//!
//! The syntax is SETTLE's (blocks, symbols, keyword arguments), so a line lexes here exactly as SETTLE lexes it.
//! SETTLE keeps its own lexer, because SETTLE also builds without KANERVA; a test on SETTLE's side holds the two
//! equal, token for token, on every line of every example.
//!
//! <claudes_code_comments>
//! ** Function List **
//! LangError            - an error message, `line N: ...`
//! err(ln, msg)         - an error naming its line
//! Tok                  - one token: symbol, label, word, number, string, comma, dot
//! lex(line, ln)        - the tokens of one line (a `#` starts a comment)
//! kwargs(toks, ln)     - `key: value` pairs after the positional part
//! kw(kv, key)          - one keyword's value
//! only(kv, ...)        - refuse a keyword the statement does not take, naming the nearest one
//! list_of(items)       - `a`, `a and b`, `a, b and c`
//! edit_distance(a, b)  - optimal string alignment distance, `-` and `_` equal
//! suggest(word, cands) - the nearest candidate when it is a likely slip
//! locate(src, msg)     - where in the program an error points: line, column, width
//! num / text / sym     - read a number, a quoted string or a symbol, or say what was expected
//!
//! ** Technical Review **
//! - A hyphen joins two words into one name when a letter follows it (`read-address:`); before a digit it starts
//!   a number (`-0.5`). `name:` is a Label, `:name` a Sym, a double-quoted run a Str.
//! - `only` checks RETIRED first (an old word whose new word this statement takes is refused with the new word),
//!   then offers `did you mean` for a near miss, then lists every keyword the statement takes.
//! - Every message here is byte-identical to SETTLE's, so a statement errs the same way under either command.
//!
//! </claudes_code_comments>

use crate::words::RETIRED;
use std::fmt;

/// An error in a program: the message, starting `line N: ` when it belongs to a line.
#[derive(Debug, Clone, PartialEq)]
pub struct LangError(pub String);

impl fmt::Display for LangError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LangError {}

/// An error naming its line.
pub fn err<T>(ln: usize, msg: impl Into<String>) -> Result<T, LangError> {
    Err(LangError(format!("line {}: {}", ln, msg.into())))
}

/// One token. `Sym` is `:name`, `Label` is `name:`, `Str` is a double-quoted string.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// `:name`
    Sym(String),
    /// `name:`
    Label(String),
    /// a bare word: `sdm`, `s`, `read`
    Ident(String),
    /// a number: `256`, `0.3`, `-2`, `1_000`
    Num(f64),
    /// a double-quoted string
    Str(String),
    /// `,`
    Comma,
    /// `.`
    Dot,
}

/// The tokens of one line. A `#` starts a comment that runs to the end of the line.
pub fn lex(line: &str, ln: usize) -> Result<Vec<Tok>, LangError> {
    let c: Vec<char> = line.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    let word = |c: &[char], mut j: usize| {
        let s = j;
        // A hyphen joins two words into one name when a letter follows it: `read-address:`.
        while j < c.len()
            && (c[j].is_alphanumeric() || c[j] == '_' || (c[j] == '-' && j > s && j + 1 < c.len() && c[j + 1].is_alphabetic()))
        {
            j += 1;
        }
        (c[s..j].iter().collect::<String>(), j)
    };
    while i < c.len() {
        let ch = c[i];
        if ch == '#' {
            break;
        } else if ch.is_whitespace() {
            i += 1;
        } else if ch == '"' {
            let s = i + 1;
            let mut j = s;
            while j < c.len() && c[j] != '"' {
                j += 1;
            }
            if j >= c.len() {
                return err(ln, "a string is missing its closing quote");
            }
            out.push(Tok::Str(c[s..j].iter().collect()));
            i = j + 1;
        } else if ch == ',' {
            out.push(Tok::Comma);
            i += 1;
        } else if ch == '.' && !(i + 1 < c.len() && c[i + 1].is_ascii_digit()) {
            out.push(Tok::Dot);
            i += 1;
        } else if ch == ':' {
            let (w, j) = word(&c, i + 1);
            if w.is_empty() {
                return err(ln, "a ':' must start a symbol like :rain");
            }
            out.push(Tok::Sym(w));
            i = j;
        } else if ch.is_ascii_digit() || ch == '-' || ch == '.' {
            let s = i;
            i += 1;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '_' || c[i] == '.' || c[i] == 'e') {
                i += 1;
            }
            let txt: String = c[s..i].iter().filter(|&&x| x != '_').collect();
            match txt.parse::<f64>() {
                Ok(v) => out.push(Tok::Num(v)),
                Err(_) => return err(ln, format!("'{}' is not a number", txt)),
            }
        } else if ch.is_alphabetic() || ch == '_' {
            let (w, j) = word(&c, i);
            if j < c.len() && c[j] == ':' && !(j + 1 < c.len() && c[j + 1] == ':') {
                out.push(Tok::Label(w));
                i = j + 1;
            } else {
                out.push(Tok::Ident(w));
                i = j;
            }
        } else {
            return err(ln, format!("unexpected '{}'", ch));
        }
    }
    Ok(out)
}

/// `key: value` pairs after the positional part of a statement.
pub fn kwargs(toks: &[Tok], ln: usize) -> Result<Vec<(String, Tok)>, LangError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            Tok::Comma => i += 1,
            Tok::Label(k) if i + 1 < toks.len() => {
                out.push((k.clone(), toks[i + 1].clone()));
                i += 2;
            }
            t => return err(ln, format!("expected `key: value`, found {:?}", t)),
        }
    }
    Ok(out)
}

/// Look up a keyword argument by name.
pub fn kw<'a>(kv: &'a [(String, Tok)], key: &str) -> Option<&'a Tok> {
    kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Refuse any keyword not in `allowed`, naming the statement. A retired keyword whose new word this statement
/// takes is refused with the new word, so `cue:` says to write `read-address:`. Any other unknown keyword is
/// refused with the nearest keyword the statement takes when one is close (`did you mean`), and otherwise with
/// the list of keywords it takes.
pub fn only(kv: &[(String, Tok)], allowed: &[&str], what: &str, ln: usize) -> Result<(), LangError> {
    for (k, _) in kv {
        if !allowed.contains(&k.as_str()) {
            if let Some((_, new, why)) = RETIRED.iter().find(|(old, new, _)| old == k && allowed.contains(new)) {
                return err(ln, format!("`{}:` is now `{}:` ({})", k, new, why));
            }
            let hint = match suggest(k, allowed.iter().copied()) {
                Some(s) => format!("; did you mean `{}:`?", s),
                None if allowed.is_empty() => format!("; {} takes no `key: value` arguments", what),
                None => format!("; it takes {}", list_of(allowed.iter().map(|a| format!("`{}:`", a)))),
            };
            return err(ln, format!("{} does not take `{}:`{}", what, k, hint));
        }
    }
    Ok(())
}

/// `a`, `a and b`, `a, b and c`: a list for an error message.
pub fn list_of(items: impl IntoIterator<Item = String>) -> String {
    let v: Vec<String> = items.into_iter().collect();
    match v.len() {
        0 => String::new(),
        1 => v[0].clone(),
        n => format!("{} and {}", v[..n - 1].join(", "), v[n - 1]),
    }
}

/// The edit distance between two words: insertions, deletions, substitutions and swaps of two neighbouring letters
/// cost one each (the optimal string alignment distance), with `-` and `_` counted as the same character.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let norm = |c: char| if c == '-' { '_' } else { c.to_ascii_lowercase() };
    let a: Vec<char> = a.chars().map(norm).collect();
    let b: Vec<char> = b.chars().map(norm).collect();
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut v = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = v;
        }
    }
    d[n][m]
}

/// The candidate nearest to `word` when it is close enough to be a likely slip: at most one edit for a word of up
/// to five letters, at most a third of the length for a longer one, and always fewer edits than the shorter word
/// has letters. Ties go to the first candidate. `None` when nothing is close.
pub fn suggest<'a>(word: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let limit = (word.chars().count() / 3).max(1);
    let mut best: Option<(usize, &'a str)> = None;
    for c in candidates {
        if c == word {
            continue;
        }
        let d = edit_distance(word, c);
        let shorter = word.chars().count().min(c.chars().count());
        if d <= limit && d < shorter && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, c));
        }
    }
    best.map(|(_, c)| c)
}

/// Where in the program an error points: `(line, column, width)`, counted from 1 in characters. The column is the
/// first thing the message's first line quotes that appears on that line (a `backticked` fragment, then a
/// `:symbol`); failing both, the statement itself. `None` when the message names no line inside the program.
pub fn locate(src: &str, msg: &str) -> Option<(usize, usize, usize)> {
    let rest = msg.strip_prefix("line ")?;
    let n_end = rest.find(':')?;
    let ln: usize = rest[..n_end].parse().ok()?;
    let line = src.lines().nth(ln.checked_sub(1)?)?;
    let body = rest[n_end + 1..].lines().next().unwrap_or("");
    let code_end = line.find('#').unwrap_or(line.len());
    let code = &line[..code_end];
    let col_of = |byte: usize| code[..byte].chars().count() + 1;
    let mut parts = body.split('`');
    parts.next();
    while let Some(frag) = parts.next() {
        if !frag.is_empty() {
            if let Some(b) = code.find(frag) {
                return Some((ln, col_of(b), frag.chars().count()));
            }
        }
        parts.next();
    }
    for word in body.split(|c: char| c.is_whitespace() || c == ',' || c == '(' || c == ')' || c == ';') {
        let sym = word.trim_end_matches(['.', '?', '!']);
        if sym.len() > 1 && sym.starts_with(':') {
            let mut from = 0;
            while let Some(b) = code[from..].find(sym) {
                let at = from + b;
                let after = code[at + sym.len()..].chars().next();
                if !after.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                    return Some((ln, col_of(at), sym.chars().count()));
                }
                from = at + sym.len();
            }
        }
    }
    let start = code.len() - code.trim_start().len();
    let width = code.trim().chars().count().max(1);
    Some((ln, col_of(start), width))
}

/// A number, or "a number was expected".
pub fn num(t: &Tok, ln: usize) -> Result<f64, LangError> {
    match t {
        Tok::Num(v) => Ok(*v),
        _ => err(ln, "a number was expected"),
    }
}

/// A quoted string, or "a \"quoted\" string was expected".
pub fn text(t: &Tok, ln: usize) -> Result<String, LangError> {
    match t {
        Tok::Str(s) => Ok(s.clone()),
        _ => err(ln, "a \"quoted\" string was expected"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hyphen_between_letters_joins_one_label_and_a_minus_before_a_digit_is_a_number() {
        let t = lex("s.read read-address: :cat, address-noise: 0.2", 1).unwrap();
        assert!(t.contains(&Tok::Label("read-address".into())), "{:?}", t);
        assert!(t.contains(&Tok::Label("address-noise".into())), "{:?}", t);
        assert_eq!(lex("by: -0.5", 1).unwrap(), vec![Tok::Label("by".into()), Tok::Num(-0.5)]);
        assert_eq!(lex("k.fill 1_000  # a comment", 1).unwrap(), vec![Tok::Ident("k".into()), Tok::Dot, Tok::Ident("fill".into()), Tok::Num(1000.0)]);
        assert_eq!(lex("s.write :n, \"open", 4).unwrap_err().0, "line 4: a string is missing its closing quote");
        assert_eq!(lex("a ; b", 2).unwrap_err().0, "line 2: unexpected ';'");
    }

    #[test]
    fn a_retired_keyword_names_its_new_word_and_a_slip_gets_a_did_you_mean() {
        let kv = kwargs(&lex(concat!("cu", "e: :cat"), 1).unwrap(), 1).unwrap();
        assert_eq!(only(&kv, &["read-address"], "read", 3).unwrap_err().0, "line 3: `cue:` is now `read-address:` (Kanerva's retrieval address)");
        let kv = kwargs(&lex("sed: 1", 1).unwrap(), 1).unwrap();
        assert_eq!(only(&kv, &["seed", "fade"], "sdm", 4).unwrap_err().0, "line 4: sdm does not take `sed:`; did you mean `seed:`?");
        let kv = kwargs(&lex("colour: 1", 1).unwrap(), 1).unwrap();
        assert_eq!(only(&kv, &["seed", "fade"], "sdm", 1).unwrap_err().0, "line 1: sdm does not take `colour:`; it takes `seed:` and `fade:`");
        assert_eq!(only(&kv, &[], "write", 1).unwrap_err().0, "line 1: write does not take `colour:`; write takes no `key: value` arguments");
        // the negative control: a keyword the statement takes is accepted
        assert!(only(&kv, &["colour"], "x", 1).is_ok());
    }

    #[test]
    fn locate_points_at_what_the_message_quotes() {
        let src = "model :m do\n  sdm :s, sed: 1\nend";
        assert_eq!(locate(src, "line 2: sdm does not take `sed:`; did you mean `seed:`?"), Some((2, 11, 4)));
        assert_eq!(locate(src, "line 2: sdm :s is already declared"), Some((2, 7, 2)));
        assert_eq!(locate(src, "line 2: no such thing"), Some((2, 3, 14)));
        assert_eq!(locate(src, "line 9: past the end"), None);
    }
}
