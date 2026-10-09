//! The parser of the sdm family: one line of tokens in, one typed statement out, or an error naming the line.
//!
//! This is the ONE parser of these statements. KANERVA's own runner and the `kanerva` command use it, and SETTLE
//! mounts it, so a `.settle` program and a `.kanerva` program can never read the same line two ways.
//!
//! <claudes_code_comments>
//! ** Function List **
//! Family                  - the five statement families: sdm, softsdm, sdmscale, refusal, contenttrack
//! Family::ALL             - all five, in the order SETTLE registers them
//! Family::ext_name/help   - the bracketed name and the help lines SETTLE prints
//! Family::parse(...)      - claim a line and parse it, or leave it (None) for another family
//! Names                   - what the parser may ask about names already declared
//! Stmt and its parts      - the typed statements: Via, From, Cue, Wake
//! declare_sdm/... (private) - one function per statement, checking in SETTLE's own order
//!
//! ** Technical Review **
//! - `parse` returns None when the line is not this family's, so a host can offer it to the next family. Claims
//!   follow SETTLE exactly: `s.read` belongs to `sdm` only when `:s` is a declared sdm; `s.write` belongs to `sdm`
//!   unless `:s` is a declared softsdm; `k.put`, `k.fill` and `k.read` belong to `sdmscale` only when `:k` is one.
//! - Each statement checks its words in the same order SETTLE always has, so the first error a program meets is
//!   the same error, word for word. The checks that need the memory itself (is the text too long for this word
//!   size, is this name already written) are the runner's, after the parse.
//! - Keyword lists come from `crate::words`, so the parser cannot accept a word the list lacks. An alias
//!   (`write_samples:`) is resolved to its spelling of record before the check.
//! - Defaults that depend on nothing but the line are filled here (the activation-radius for 2% of the address
//!   space; sdmscale's radius for its activation-probability or its tolerated address-noise). Defaults that depend
//!   on the run (the address-noise from a key) are carried as the variant that needs them.
//!
//! </claudes_code_comments>

use super::lex::{err, kw, kwargs, num, only, text, LangError, Tok};
use crate::words::{canonical, statement, Place, HELP};

/// The five statement families, in the order SETTLE registers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// `sdm :s` and its `write` and `read`: Kanerva's hard-location memory.
    Sdm,
    /// `softsdm :s` and its `write`, `read` and `attend`: the memory with a soft activation-radius.
    SoftSdm,
    /// `sdmscale :k` and its `put`, `fill` and `read`: the byte-counter memory for 10^5 to 10^6 hard-locations.
    SdmScale,
    /// `refusal`: the travel rule's threshold, no memory built.
    Refusal,
    /// `contenttrack`: TRACK-C's predicted recall for the content reads, no memory built.
    ContentTrack,
}

/// What the parser may ask about names already declared in the model: is `name` a declared memory of `family`.
pub trait Names {
    /// Whether `name` is a declared memory of `family` (`Sdm`, `SoftSdm` or `SdmScale`).
    fn declared(&self, family: Family, name: &str) -> bool;
}

/// Where a hard read takes its read-address from.
#[derive(Debug, Clone, PartialEq)]
pub enum From {
    /// A stored name, with this address-noise.
    Address {
        /// The name after `read-address:`.
        name: String,
        /// The fraction of bits flipped.
        noise: f64,
    },
    /// A key's read-address, with this address-noise.
    Key {
        /// The key text.
        key: String,
        /// The fraction of bits flipped.
        noise: f64,
    },
    /// Pure noise.
    Noise,
}

/// Which matrix a hard read decodes with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Via {
    /// Kanerva's read: the addresses decide which hard-locations wake.
    Addresses,
    /// The read by content: the bit-counters decide.
    Pulls,
}

/// How an sdmscale read by content wakes hard-locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wake {
    /// A fixed threshold, 0.4 n.
    Fixed,
    /// A threshold scaled with the rows' load.
    Density,
    /// The p M best rows.
    Top,
}

/// The read-address of a soft read: a stored name or pure noise, with an address-noise.
#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    /// The fraction of bits flipped (checked to lie in 0..=1).
    pub noise: f64,
    /// The name after `read-address:`, or None for pure noise.
    pub address: Option<String>,
}

/// One statement of the sdm family, parsed and checked.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `sdm :s, word-size: 256, hard-locations: 2000, ...`
    SdmDeclare {
        /// The memory's name.
        name: String,
        /// Word size n.
        word_size: usize,
        /// Hard-locations M.
        hard_locations: usize,
        /// Activation radius r.
        activation_radius: usize,
        /// Seed of the address matrix.
        seed: u64,
        /// Every counter is multiplied by this before each write (1 keeps everything).
        fade: f64,
    },
    /// `s.write :cat`, `s.write :note, "text"`, `s.write :diary, "text", key: "secret"`
    SdmWrite {
        /// The memory.
        name: String,
        /// The stored name.
        what: String,
        /// Text, masked by the name (or turned by the key).
        text: Option<String>,
        /// The key that turns the text.
        key: Option<String>,
    },
    /// `s.read read-address: :cat, address-noise: 0.2, ...`
    SdmRead {
        /// The memory.
        name: String,
        /// A new seed for the run's random stream.
        seed: Option<u64>,
        /// The most reads to iterate.
        iterated_reads: usize,
        /// Which matrix decodes.
        via: Via,
        /// Where the read starts (a key's read-address makes the answer a keyed note).
        from: From,
    },
    /// `softsdm :s, word-size: 256, ...`
    SoftDeclare {
        /// The memory's name.
        name: String,
        /// Word size n.
        word_size: usize,
        /// Hard-locations M.
        hard_locations: usize,
        /// The hard cut-off's activation-probability.
        activation_probability: f64,
        /// The softness dial.
        softness: f64,
        /// The data field one clean pattern gives.
        gain: f64,
        /// Seed of the addresses.
        seed: u64,
        /// Activation draws a write averages.
        write_samples: usize,
    },
    /// `s.write :cat`, `s.write :note, "text"` on a softsdm
    SoftWrite {
        /// The memory.
        name: String,
        /// The stored name.
        what: String,
        /// Text, masked by the name.
        text: Option<String>,
    },
    /// `s.read read-address: :cat, ...` on a softsdm
    SoftRead {
        /// The memory.
        name: String,
        /// A new seed for the run's random stream.
        seed: Option<u64>,
        /// Read-outs fed back as the next read-address.
        rounds: usize,
        /// Samples per read.
        samples: usize,
        /// Burn-in sweeps of a settle read.
        burn: usize,
        /// `mode: :settle` (true) or `:pass`.
        settle: bool,
        /// Where the read starts.
        cue: Cue,
    },
    /// `s.attend read-address: :cat, ...` on a softsdm
    SoftAttend {
        /// The memory.
        name: String,
        /// A new seed for the run's random stream.
        seed: Option<u64>,
        /// Read-outs fed back.
        rounds: usize,
        /// Where the read starts.
        cue: Cue,
    },
    /// `sdmscale :k, word-size: 256, ...`
    ScaleDeclare {
        /// The store's name.
        name: String,
        /// Word size n.
        word_size: usize,
        /// Hard-locations M.
        hard_locations: usize,
        /// Activation radius r, chosen or derived.
        activation_radius: usize,
        /// Seed, kept as written.
        seed: f64,
    },
    /// `k.put :cat`
    ScalePut {
        /// The store.
        name: String,
        /// The stored name.
        what: String,
    },
    /// `k.fill 500`
    ScaleFill {
        /// The store.
        name: String,
        /// How many random patterns.
        count: usize,
    },
    /// `k.read read-address: :cat, ...`
    ScaleRead {
        /// The store.
        name: String,
        /// A new seed for the run's random stream.
        seed: Option<u64>,
        /// The most reads to iterate.
        iterated_reads: usize,
        /// Which matrix decodes.
        via: Via,
        /// The name after `read-address:`.
        read_address: String,
        /// The fraction of bits flipped.
        address_noise: f64,
        /// How a read by content wakes rows.
        wake: Wake,
    },
    /// `refusal word-size: 256, load: 3000, level: 0.01`
    Refusal {
        /// Word size n.
        word_size: usize,
        /// Stored patterns T.
        load: usize,
        /// The false-acceptance level.
        level: f64,
    },
    /// `contenttrack word-size: 256, hard-locations: 100000, ...`
    ContentTrack {
        /// Word size n.
        word_size: usize,
        /// Hard-locations M.
        hard_locations: usize,
        /// Stored patterns T.
        load: usize,
        /// The fraction of bits flipped.
        address_noise: f64,
        /// The block read (true) or the top-k read.
        block: bool,
        /// Sampled reads.
        samples: usize,
    },
}

impl Family {
    /// All five, in the order SETTLE registers them.
    pub const ALL: [Family; 5] = [Family::Sdm, Family::SoftSdm, Family::SdmScale, Family::Refusal, Family::ContentTrack];

    /// The family word in the word list (`sdm`, `softsdm`, `sdmscale`, `refusal`, `contenttrack`).
    pub fn word(self) -> &'static str {
        match self {
            Family::Sdm => "sdm",
            Family::SoftSdm => "softsdm",
            Family::SdmScale => "sdmscale",
            Family::Refusal => "refusal",
            Family::ContentTrack => "contenttrack",
        }
    }

    /// The bracketed name SETTLE's help prints (`[sdmrefuse] run: refusal ...`).
    pub fn ext_name(self) -> &'static str {
        match self {
            Family::Sdm => "sdm",
            Family::SoftSdm => "softsdm",
            Family::SdmScale => "sdmscale",
            Family::Refusal => "sdmrefuse",
            Family::ContentTrack => "sdmtrack",
        }
    }

    /// The help lines, verbatim as SETTLE prints them.
    pub fn help(self) -> &'static [&'static str] {
        HELP.iter().find(|(e, _)| *e == self.ext_name()).map(|(_, l)| *l).unwrap_or(&[])
    }

    /// Claim and parse one line of tokens standing in `place`. None means "not this family's line".
    pub fn parse(self, place: Place, t: &[Tok], ln: usize, names: &dyn Names) -> Option<Result<Stmt, LangError>> {
        match (self, place) {
            (Family::Sdm, Place::Model) => match t {
                [Tok::Ident(k), Tok::Sym(name), rest @ ..] if k == "sdm" => Some(declare_sdm(name, strip_comma(rest), ln, names)),
                _ => sdm_write(t, ln, names),
            },
            (Family::Sdm, Place::Run) => match t {
                [Tok::Ident(name), Tok::Dot, Tok::Ident(v), rest @ ..] if v == "read" && names.declared(Family::Sdm, name) => {
                    Some(sdm_read(name, rest, ln))
                }
                _ => sdm_write(t, ln, names),
            },
            (Family::SoftSdm, Place::Model) => match t {
                [Tok::Ident(k), Tok::Sym(name), rest @ ..] if k == "softsdm" => Some(declare_soft(name, strip_comma(rest), ln, names)),
                _ => soft_write(t, ln, names),
            },
            (Family::SoftSdm, Place::Run) => match t {
                [Tok::Ident(name), Tok::Dot, Tok::Ident(v), rest @ ..] if v == "read" && names.declared(Family::SoftSdm, name) => {
                    Some(soft_read(name, rest, ln))
                }
                [Tok::Ident(name), Tok::Dot, Tok::Ident(v), rest @ ..] if v == "attend" && names.declared(Family::SoftSdm, name) => {
                    Some(soft_attend(name, rest, ln))
                }
                _ => soft_write(t, ln, names),
            },
            (Family::SdmScale, Place::Model) => match t {
                [Tok::Ident(k), Tok::Sym(name), rest @ ..] if k == "sdmscale" => Some(declare_scale(name, strip_comma(rest), ln, names)),
                [Tok::Ident(name), Tok::Dot, Tok::Ident(v), Tok::Sym(what)] if v == "put" && names.declared(Family::SdmScale, name) => {
                    Some(Ok(Stmt::ScalePut { name: name.clone(), what: what.clone() }))
                }
                [Tok::Ident(name), Tok::Dot, Tok::Ident(v), Tok::Num(k)] if v == "fill" && names.declared(Family::SdmScale, name) => {
                    Some(Ok(Stmt::ScaleFill { name: name.clone(), count: *k as usize }))
                }
                _ => None,
            },
            (Family::SdmScale, Place::Run) => match t {
                [Tok::Ident(name), Tok::Dot, Tok::Ident(v), rest @ ..] if v == "read" && names.declared(Family::SdmScale, name) => {
                    Some(scale_read(name, rest, ln))
                }
                _ => None,
            },
            (Family::Refusal, Place::Run) => match t {
                [Tok::Ident(k), rest @ ..] if k == "refusal" => Some(refusal(rest, ln)),
                _ => None,
            },
            (Family::ContentTrack, Place::Run) => match t {
                [Tok::Ident(k), rest @ ..] if k == "contenttrack" => Some(contenttrack(rest, ln)),
                _ => None,
            },
            (Family::Refusal | Family::ContentTrack, Place::Model) => None,
        }
    }
}

fn strip_comma(rest: &[Tok]) -> &[Tok] {
    if rest.first() == Some(&Tok::Comma) { &rest[1..] } else { rest }
}

/// The keyword names a statement takes, from the word list.
fn words_of(family: &str, verb: &str) -> Vec<&'static str> {
    statement(family, verb).map(|s| s.names()).unwrap_or_default()
}

/// The keyword pairs with every alias of `family` resolved to its spelling of record.
fn canonical_kwargs(family: &str, rest: &[Tok], ln: usize) -> Result<Vec<(String, Tok)>, LangError> {
    Ok(kwargs(rest, ln)?.into_iter().map(|(k, v)| (canonical(family, &k).to_string(), v)).collect())
}

/// A number keyword with its default.
fn get(kv: &[(String, Tok)], k: &str, d: f64, ln: usize) -> Result<f64, LangError> {
    kw(kv, k).map(|v| num(v, ln)).transpose().map(|x| x.unwrap_or(d))
}

fn seed_of(kv: &[(String, Tok)], ln: usize) -> Result<Option<u64>, LangError> {
    kw(kv, "seed").map(|x| num(x, ln).map(|v| v as u64)).transpose()
}

fn via_of(kv: &[(String, Tok)], ln: usize) -> Result<Via, LangError> {
    match kw(kv, "via") {
        None => Ok(Via::Addresses),
        Some(Tok::Sym(s)) if s == "addresses" => Ok(Via::Addresses),
        Some(Tok::Sym(s)) if s == "pulls" => Ok(Via::Pulls),
        Some(_) => err(ln, "via: takes :addresses or :pulls"),
    }
}

fn declare_sdm(name: &str, rest: &[Tok], ln: usize, names: &dyn Names) -> Result<Stmt, LangError> {
    if names.declared(Family::Sdm, name) {
        return err(ln, format!("sdm :{} is already declared", name));
    }
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("sdm", "sdm"), "sdm", ln)?;
    let n = get(&kv, "word-size", 256.0, ln)? as usize;
    let m_loc = get(&kv, "hard-locations", 2000.0, ln)? as usize;
    let seed = get(&kv, "seed", 1.0, ln)? as u64;
    let fade = get(&kv, "fade", 1.0, ln)?;
    if !(16..=4096).contains(&n) {
        return err(ln, "sdm word-size must be between 16 and 4096");
    }
    if !(1..=100_000).contains(&m_loc) || n * m_loc > 20_000_000 {
        return err(ln, "sdm hard-locations must be at least 1, and word-size x hard-locations at most 20 million");
    }
    if !(0.0..=1.0).contains(&fade) || fade == 0.0 {
        return err(ln, "fade must be above 0 and at most 1");
    }
    let radius = get(&kv, "activation-radius", crate::theory::radius_for(n, 0.02) as f64, ln)? as usize;
    if radius > n {
        return err(ln, "activation-radius cannot be larger than word-size");
    }
    Ok(Stmt::SdmDeclare { name: name.to_string(), word_size: n, hard_locations: m_loc, activation_radius: radius, seed, fade })
}

fn sdm_write(t: &[Tok], ln: usize, names: &dyn Names) -> Option<Result<Stmt, LangError>> {
    // `write` is shared with the softsdm family: leave a line alone when its name is a declared softsdm.
    if let [Tok::Ident(name), ..] = t {
        if names.declared(Family::SoftSdm, name) {
            return None;
        }
    }
    match t {
        [Tok::Ident(name), Tok::Dot, Tok::Ident(v), Tok::Sym(what)] if v == "write" => {
            Some(Ok(Stmt::SdmWrite { name: name.clone(), what: what.clone(), text: None, key: None }))
        }
        [Tok::Ident(name), Tok::Dot, Tok::Ident(v), Tok::Sym(what), Tok::Comma, s, rest @ ..] if v == "write" => Some((|| {
            let txt = text(s, ln)?;
            let kv = kwargs(rest, ln)?;
            only(&kv, &words_of("sdm", "write"), "write", ln)?;
            let key = kw(&kv, "key").map(|k| text(k, ln)).transpose()?;
            Ok(Stmt::SdmWrite { name: name.clone(), what: what.clone(), text: Some(txt), key })
        })()),
        _ => None,
    }
}

fn sdm_read(name: &str, rest: &[Tok], ln: usize) -> Result<Stmt, LangError> {
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("sdm", "read"), "read", ln)?;
    let seed = seed_of(&kv, ln)?;
    let iters = kw(&kv, "iterated-reads").map(|x| num(x, ln)).transpose()?.unwrap_or(10.0) as usize;
    let via = via_of(&kv, ln)?;
    let key = kw(&kv, "key").map(|t| text(t, ln)).transpose()?;
    let from = match (kw(&kv, "read-address"), &key) {
        (Some(_), Some(_)) => return err(ln, "read takes read-address: or key:, not both"),
        (Some(Tok::Sym(c)), None) => From::Address { name: c.clone(), noise: get(&kv, "address-noise", 0.3, ln)? },
        (Some(_), None) => return err(ln, "read-address: takes a symbol, like read-address: :cat"),
        (None, Some(k)) => From::Key { key: k.clone(), noise: get(&kv, "address-noise", 0.0, ln)? },
        (None, None) => From::Noise,
    };
    Ok(Stmt::SdmRead { name: name.to_string(), seed, iterated_reads: iters, via, from })
}

fn declare_soft(name: &str, rest: &[Tok], ln: usize, names: &dyn Names) -> Result<Stmt, LangError> {
    if names.declared(Family::SoftSdm, name) {
        return err(ln, format!("softsdm :{} is already declared", name));
    }
    let kv = canonical_kwargs("softsdm", rest, ln)?;
    only(&kv, &words_of("softsdm", "softsdm"), "softsdm", ln)?;
    let n = get(&kv, "word-size", 256.0, ln)? as usize;
    let lm = get(&kv, "hard-locations", 2000.0, ln)? as usize;
    let fire = get(&kv, "activation-probability", 0.05, ln)?;
    let soft = get(&kv, "softness", 0.3, ln)?;
    let gain = get(&kv, "gain", 64.0, ln)?;
    let seed = get(&kv, "seed", 1.0, ln)? as u64;
    let ws = get(&kv, "write-samples", 16.0, ln)? as usize;
    if !(8..=4096).contains(&n) {
        return err(ln, "softsdm size must be between 8 and 4096");
    }
    if lm < 1 || lm * n > 4_000_000 {
        return err(ln, "softsdm needs at least 1 hard location and at most 4,000,000 location-bits (hard-locations x word-size)");
    }
    if !(fire > 0.0 && fire < 1.0) {
        return err(ln, "activation-probability is a fraction of hard locations, above 0 and below 1");
    }
    if soft < 0.0 || gain <= 0.0 {
        return err(ln, "softness must be at least 0 and gain above 0");
    }
    Ok(Stmt::SoftDeclare {
        name: name.to_string(),
        word_size: n,
        hard_locations: lm,
        activation_probability: fire,
        softness: soft,
        gain,
        seed,
        write_samples: ws,
    })
}

fn soft_write(t: &[Tok], ln: usize, names: &dyn Names) -> Option<Result<Stmt, LangError>> {
    match t {
        [Tok::Ident(name), Tok::Dot, Tok::Ident(v), rest @ ..] if v == "write" && names.declared(Family::SoftSdm, name) => match rest {
            [Tok::Sym(what)] => Some(Ok(Stmt::SoftWrite { name: name.clone(), what: what.clone(), text: None })),
            [Tok::Sym(what), Tok::Comma, s] => {
                Some(text(s, ln).map(|txt| Stmt::SoftWrite { name: name.clone(), what: what.clone(), text: Some(txt) }))
            }
            _ => Some(err(ln, "write takes a symbol, and optionally text: s.write :note, \"some text\"")),
        },
        _ => None,
    }
}

fn cue_of(kv: &[(String, Tok)], ln: usize) -> Result<Cue, LangError> {
    let noise = get(kv, "address-noise", 0.3, ln)?;
    if !(0.0..=1.0).contains(&noise) {
        return err(ln, "address-noise is a fraction between 0 and 1");
    }
    match kw(kv, "read-address") {
        Some(Tok::Sym(c)) => Ok(Cue { noise, address: Some(c.clone()) }),
        Some(_) => err(ln, "read-address: takes a symbol, like read-address: :cat"),
        None => Ok(Cue { noise, address: None }),
    }
}

fn soft_read(name: &str, rest: &[Tok], ln: usize) -> Result<Stmt, LangError> {
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("softsdm", "read"), "read", ln)?;
    let seed = seed_of(&kv, ln)?;
    let rounds = get(&kv, "rounds", 3.0, ln)? as usize;
    let samples = get(&kv, "samples", 16.0, ln)? as usize;
    let burn = get(&kv, "burn", 10.0, ln)? as usize;
    let settle = match kw(&kv, "mode") {
        None => false,
        Some(Tok::Sym(x)) if x == "pass" => false,
        Some(Tok::Sym(x)) if x == "settle" => true,
        Some(_) => return err(ln, "mode: is :pass or :settle"),
    };
    let cue = cue_of(&kv, ln)?;
    Ok(Stmt::SoftRead { name: name.to_string(), seed, rounds, samples, burn, settle, cue })
}

fn soft_attend(name: &str, rest: &[Tok], ln: usize) -> Result<Stmt, LangError> {
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("softsdm", "attend"), "attend", ln)?;
    let seed = seed_of(&kv, ln)?;
    let rounds = get(&kv, "rounds", 3.0, ln)? as usize;
    let cue = cue_of(&kv, ln)?;
    Ok(Stmt::SoftAttend { name: name.to_string(), seed, rounds, cue })
}

fn declare_scale(name: &str, rest: &[Tok], ln: usize, names: &dyn Names) -> Result<Stmt, LangError> {
    if names.declared(Family::SdmScale, name) {
        return err(ln, format!("sdmscale :{} is already declared", name));
    }
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("sdmscale", "sdmscale"), "sdmscale", ln)?;
    let n = get(&kv, "word-size", 256.0, ln)? as usize;
    let mm = get(&kv, "hard-locations", 100_000.0, ln)? as usize;
    let seed = get(&kv, "seed", 1.0, ln)?;
    if !(16..=4096).contains(&n) {
        return err(ln, "sdmscale word-size must be between 16 and 4096");
    }
    if !(1..=2_000_000).contains(&mm) || n * mm > 1_100_000_000 {
        return err(ln, "sdmscale hard-locations must be 1 to 2,000,000, and word-size x hard-locations at most 1.1 billion bytes");
    }
    let fire = get(&kv, "activation-probability", 0.001, ln)?;
    if !(fire > 0.0 && fire < 1.0) {
        return err(ln, "activation-probability must be between 0 and 1");
    }
    let tolerate = get(&kv, "tolerate-noise", -1.0, ln)?;
    let by_fire = if tolerate >= 0.0 {
        if tolerate >= 0.5 {
            return err(ln, "tolerate-noise is an address-noise fraction below 0.5");
        }
        // SDMRADIUS: the activation radius whose S-map capacity from a read-address at this address-noise is largest
        let (lo, hi) = crate::smap::search_window(n, mm);
        crate::smap::radius_for_address_noise(n, mm, tolerate, lo, hi).0
    } else {
        crate::theory::radius_for(n, fire)
    };
    let radius = get(&kv, "activation-radius", by_fire as f64, ln)? as usize;
    if radius > n {
        return err(ln, "activation-radius cannot be larger than word-size");
    }
    Ok(Stmt::ScaleDeclare { name: name.to_string(), word_size: n, hard_locations: mm, activation_radius: radius, seed })
}

fn scale_read(name: &str, rest: &[Tok], ln: usize) -> Result<Stmt, LangError> {
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("sdmscale", "read"), "read", ln)?;
    let seed = seed_of(&kv, ln)?;
    let iters = kw(&kv, "iterated-reads").map(|x| num(x, ln)).transpose()?.unwrap_or(20.0) as usize;
    let via = via_of(&kv, ln)?;
    let read_address = match kw(&kv, "read-address") {
        Some(Tok::Sym(c)) => c.clone(),
        _ => return err(ln, "read needs read-address: :name"),
    };
    let address_noise = get(&kv, "address-noise", 0.2, ln)?;
    let wake = match kw(&kv, "wake") {
        None => Wake::Fixed,
        Some(Tok::Sym(s)) if s == "fixed" => Wake::Fixed,
        Some(Tok::Sym(s)) if s == "density" => Wake::Density,
        Some(Tok::Sym(s)) if s == "top" => Wake::Top,
        Some(_) => return err(ln, "wake: takes :fixed (0.4 n), :density (scaled with the rows' load) or :top (the p M best rows)"),
    };
    if via == Via::Addresses && wake != Wake::Fixed {
        return err(ln, "wake: applies to via: :pulls");
    }
    Ok(Stmt::ScaleRead { name: name.to_string(), seed, iterated_reads: iters, via, read_address, address_noise, wake })
}

fn refusal(rest: &[Tok], ln: usize) -> Result<Stmt, LangError> {
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("refusal", "refusal"), "refusal", ln)?;
    let n = get(&kv, "word-size", 256.0, ln)? as usize;
    let load = get(&kv, "load", 1000.0, ln)? as usize;
    let level = get(&kv, "level", 0.01, ln)?;
    if !(16..=4096).contains(&n) {
        return err(ln, "refusal size must be between 16 and 4096");
    }
    if load < 1 || !(level > 0.0 && level < 1.0) {
        return err(ln, "refusal needs load: at least 1 and level: between 0 and 1");
    }
    Ok(Stmt::Refusal { word_size: n, load, level })
}

fn contenttrack(rest: &[Tok], ln: usize) -> Result<Stmt, LangError> {
    let kv = kwargs(rest, ln)?;
    only(&kv, &words_of("contenttrack", "contenttrack"), "contenttrack", ln)?;
    let n = get(&kv, "word-size", 256.0, ln)? as usize;
    let m = get(&kv, "hard-locations", 100_000.0, ln)? as usize;
    let load = get(&kv, "load", 1000.0, ln)? as usize;
    let dmg = get(&kv, "address-noise", 0.3, ln)?;
    let block = get(&kv, "block", 0.0, ln)? != 0.0;
    let smp = get(&kv, "samples", 200.0, ln)? as usize;
    if !(64..=1024).contains(&n) || m < 100 || load < 1 || !(0.0..0.5).contains(&dmg) || smp < 1 {
        return err(ln, "contenttrack needs word-size 64..1024, hard-locations >= 100, load >= 1, address-noise in [0, 0.5), samples >= 1");
    }
    Ok(Stmt::ContentTrack { word_size: n, hard_locations: m, load, address_noise: dmg, block, samples: smp })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::lex::lex;

    /// A model where `s` is an sdm and `f` a softsdm and `k` an sdmscale.
    struct Three;
    impl Names for Three {
        fn declared(&self, family: Family, name: &str) -> bool {
            matches!((family, name), (Family::Sdm, "s") | (Family::SoftSdm, "f") | (Family::SdmScale, "k"))
        }
    }

    fn claim(place: Place, line: &str) -> Vec<(Family, Result<Stmt, LangError>)> {
        let t = lex(line, 1).unwrap();
        Family::ALL.iter().filter_map(|f| f.parse(place, &t, 1, &Three).map(|r| (*f, r))).collect()
    }

    #[test]
    fn each_line_is_claimed_by_the_family_settle_gives_it() {
        assert_eq!(claim(Place::Run, "s.read")[0].0, Family::Sdm);
        assert_eq!(claim(Place::Run, "f.read")[0].0, Family::SoftSdm);
        assert_eq!(claim(Place::Run, "k.read read-address: :a")[0].0, Family::SdmScale);
        // write belongs to sdm unless the name is a softsdm; an undeclared name still goes to sdm
        assert_eq!(claim(Place::Model, "f.write :a").iter().map(|c| c.0).collect::<Vec<_>>(), vec![Family::SoftSdm]);
        assert_eq!(claim(Place::Model, "q.write :a")[0].0, Family::Sdm);
        // the negative controls: a read of an undeclared name, and a model statement in a run, are nobody's
        assert!(claim(Place::Run, "q.read").is_empty());
        assert!(claim(Place::Run, "sdm :x").is_empty());
        assert!(claim(Place::Model, "refusal").is_empty());
    }

    #[test]
    fn the_first_error_is_the_one_settle_meets_first() {
        // already declared comes before a bad keyword
        let e = claim(Place::Model, "sdm :s, colour: 1").remove(0).1.unwrap_err().0;
        assert_eq!(e, "line 1: sdm :s is already declared");
        // a bad keyword comes before a bad value
        let e = claim(Place::Model, "sdm :t, word-size: 8, colour: 1").remove(0).1.unwrap_err().0;
        assert!(e.contains("does not take `colour:`"), "{}", e);
        // read-address and key together are refused before the address-noise is read
        let e = claim(Place::Run, "s.read read-address: :a, key: \"k\", address-noise: :x").remove(0).1.unwrap_err().0;
        assert_eq!(e, "line 1: read takes read-address: or key:, not both");
        // with no read-address, a bad address-noise is never read (as in SETTLE)
        assert!(claim(Place::Run, "s.read address-noise: :x").remove(0).1.is_ok());
    }

    #[test]
    fn defaults_and_the_alias_are_filled_in() {
        match claim(Place::Model, "sdm :t, word-size: 256").remove(0).1.unwrap() {
            Stmt::SdmDeclare { hard_locations, activation_radius, seed, fade, .. } => assert_eq!((hard_locations, activation_radius, seed, fade), (2000, 112, 1, 1.0)),
            s => panic!("{:?}", s),
        }
        for spelling in ["write_samples", "write-samples"] {
            match claim(Place::Model, &format!("softsdm :g, {}: 4", spelling)).remove(0).1.unwrap() {
                Stmt::SoftDeclare { write_samples, .. } => assert_eq!(write_samples, 4),
                s => panic!("{:?}", s),
            }
        }
    }
}
