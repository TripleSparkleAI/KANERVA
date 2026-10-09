//! The runner: a `.kanerva` program executed on KANERVA's engine alone, with no SETTLE.
//!
//! A program is SETTLE's block syntax holding only sdm-family statements:
//!
//! ```
//! let out = kanerva::lang::run(
//!     "model :mind do
//!        sdm :s, word-size: 256, hard-locations: 2000
//!        s.write :cat
//!        s.write :dog
//!      end
//!      run :mind do
//!        s.read read-address: :cat, address-noise: 0.2, seed: 1
//!      end",
//! )
//! .unwrap();
//! assert!(out[0].ends_with("-> :cat"), "{}", out[0]);
//! ```
//!
//! <claudes_code_comments>
//! ** Function List **
//! run(src)               - run a program, returning its printed lines or the first error
//! Runner::exec(src)      - the same, keeping the models for inspection
//! Model (private)        - one model's memories and the names of its things
//! Model::declared        - Names: which memories are declared
//! Runner::stmt (private) - one parsed statement, executed on the engine
//! sdm_* / soft_* / scale_* (private) - each statement's work
//!
//! ** Technical Review **
//! - Blocks follow SETTLE exactly: `model :m do ... end` declares and writes, `run :m do ... end` reads; each run
//!   block starts the random stream at 0x5eed, and `seed:` restarts it. Output is collected and returned only
//!   when the whole program succeeds.
//! - Each memory is held the way SETTLE holds it, so the same program prints the same lines under both commands:
//!   an sdm keeps its bit-counters as f64 rows scaled by 2/n (SETTLE's pull values), faded and added in the same
//!   order, and reads with `address::iterated_read`; a softsdm keeps a `soft::Machine`, written in a model block
//!   with a fresh stream seeded 0x5eed_5d31 per write; an sdmscale keeps its written names and rebuilds a
//!   `store::Store` for each read, its fill drawn from `sdmscale-fill:<name>:<seed>`.
//! - Thing names are tracked as SETTLE's model tracks them, so a softsdm whose things clash is refused with
//!   SETTLE's message. One case needs SETTLE's own machinery and is refused by name rather than approximated:
//!   `via: :pulls` on an sdm (it settles the pulls of a SETTLE model, from a random start over every thing). An
//!   sdm whose thing names clash with things already declared is refused here and in SETTLE in the same words.
//! - A line no sdm-family statement claims is an error that lists the statements kanerva knows; a SETTLE
//!   statement is pointed at `settle`.
//!
//! </claudes_code_comments>

use super::lex::{err, lex, suggest, LangError, Tok};
use super::parse::{Cue, Family, Names, Stmt, Via, Wake, From};
use super::say;
use crate::address::{iterated_read, Addresses};
use crate::codes::{code, pattern, seed_of, with_address_noise};
use crate::keys::{keyed_capacity, keyed_pattern, keyed_read_address};
use crate::rng::Rng;
use crate::soft::{attention_read, sign_of, Attn, Machine};
use crate::store::Store;
use crate::words::Place;
use std::collections::{HashMap, HashSet};

/// The seed SETTLE starts every run block's random stream at.
pub const RUN_SEED: u64 = 0x5eed;
/// The seed of the stream a softsdm write draws from inside a model block (fresh for each write).
pub const MODEL_WRITE_SEED: u64 = 0x5eed_5d31;

/// Run a `.kanerva` program: its printed lines, or the first error.
pub fn run(src: &str) -> Result<Vec<String>, LangError> {
    Runner::default().exec(src)
}

struct SdmMem {
    n: usize,
    m: usize,
    radius: usize,
    fade: f64,
    addr: Addresses,
    /// Bit-counter rows, row-major, scaled by 2/n as SETTLE's pulls are.
    rows: Vec<f64>,
    stored: Vec<(String, String)>,
}

struct SoftMem {
    mach: Machine,
    stored: Vec<(String, Option<String>)>,
}

struct ScaleMem {
    n: usize,
    m: usize,
    radius: usize,
    seed: f64,
    words: Vec<String>,
}

#[derive(Default)]
struct Model {
    things: HashSet<String>,
    sdm: HashMap<String, SdmMem>,
    soft: HashMap<String, SoftMem>,
    scale: HashMap<String, ScaleMem>,
}

impl Names for Model {
    fn declared(&self, family: Family, name: &str) -> bool {
        match family {
            Family::Sdm => self.sdm.contains_key(name),
            Family::SoftSdm => self.soft.contains_key(name),
            Family::SdmScale => self.scale.contains_key(name),
            Family::Refusal | Family::ContentTrack => false,
        }
    }
}

/// A program's models, kept between `exec` calls.
#[derive(Default)]
pub struct Runner {
    models: HashMap<String, Model>,
}

/// Every verb the families list for a place, read from their help lines (as SETTLE reads its own).
fn verbs(place: Place) -> Vec<String> {
    let want = if place == Place::Model { "model" } else { "run" };
    let mut out: Vec<String> = Vec::new();
    for f in Family::ALL {
        for s in f.help() {
            let Some((places, body)) = s.split_once(": ") else { continue };
            if !places.split('/').any(|p| p == want) {
                continue;
            }
            for form in body.split("   /   ") {
                let first = form.split_whitespace().next().unwrap_or("");
                let word = first.rsplit('.').next().unwrap_or(first);
                let word: String = word.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if !word.is_empty() && !out.contains(&word) {
                    out.push(word);
                }
            }
        }
    }
    out
}

fn verb_of(t: &[Tok]) -> Option<&str> {
    match t {
        [Tok::Ident(_), Tok::Dot, Tok::Ident(v), ..] => Some(v),
        [Tok::Ident(v), ..] => Some(v),
        _ => None,
    }
}

fn unknown(place: Place, t: &[Tok], ln: usize) -> LangError {
    let (here, other, word) = match place {
        Place::Model => (Place::Model, Place::Run, ("model", "run")),
        Place::Run => (Place::Run, Place::Model, ("run", "model")),
    };
    let known: Vec<&str> = Family::ALL.iter().flat_map(|f| f.help().iter().copied()).filter_map(|s| s.strip_prefix(&format!("{}: ", word.0))).collect();
    let hint = match verb_of(t) {
        Some(v) if verbs(other).iter().any(|w| w == v) && !verbs(here).iter().any(|w| w == v) => {
            format!(" `{}` is a {} statement; put it inside `{} :name do ... end`.", v, word.1, word.1)
        }
        Some(v) => {
            let vs = verbs(here);
            match suggest(v, vs.iter().map(String::as_str)) {
                Some(s) => format!(" Did you mean `{}`?", s),
                None => " kanerva knows only the sdm family; a SETTLE statement runs with `settle`.".to_string(),
            }
        }
        None => String::new(),
    };
    LangError(format!("line {}: no statement kanerva knows matches this line inside a {}.{} Known:\n    {}", ln, word.0, hint, known.join("\n    ")))
}

impl Runner {
    /// Run a program; its printed lines, or the first error. Models persist across calls.
    pub fn exec(&mut self, src: &str) -> Result<Vec<String>, LangError> {
        let mut out = Vec::new();
        let mut block: Option<(Place, String, usize)> = None;
        let mut rng = Rng::new(RUN_SEED);
        for (n0, raw) in src.lines().enumerate() {
            let ln = n0 + 1;
            let t = lex(raw, ln)?;
            if t.is_empty() {
                continue;
            }
            if let [Tok::Ident(k), Tok::Sym(name), Tok::Ident(d)] = t.as_slice() {
                if (k == "model" || k == "run") && d == "do" {
                    if block.is_some() {
                        return err(ln, "blocks cannot nest; close the previous one with `end`");
                    }
                    let place = if k == "model" {
                        self.models.entry(name.clone()).or_default();
                        Place::Model
                    } else {
                        if !self.models.contains_key(name) {
                            return err(ln, format!("no model :{} to run", name));
                        }
                        rng = Rng::new(RUN_SEED);
                        Place::Run
                    };
                    block = Some((place, name.clone(), ln));
                    continue;
                }
            }
            if t == [Tok::Ident("end".into())] {
                if block.is_none() {
                    return err(ln, "`end` without an open block");
                }
                block = None;
                continue;
            }
            let (place, name) = match &block {
                Some((p, n, _)) => (*p, n.clone()),
                None => return err(ln, "statements live inside `model :name do ... end` or `run :name do ... end`"),
            };
            let model = self.models.get_mut(&name).unwrap();
            let mut parsed = None;
            for f in Family::ALL {
                parsed = f.parse(place, &t, ln, model);
                if parsed.is_some() {
                    break;
                }
            }
            match parsed {
                Some(s) => stmt(model, place, s?, ln, &mut rng, &mut out)?,
                None => return Err(unknown(place, &t, ln)),
            }
        }
        if let Some((_, name, opened)) = block {
            return err(opened, format!("block :{} is never closed with `end`", name));
        }
        Ok(out)
    }
}

/// Run one parsed statement on the engine.
fn stmt(m: &mut Model, place: Place, s: Stmt, ln: usize, rng: &mut Rng, out: &mut Vec<String>) -> Result<(), LangError> {
    match s {
        Stmt::SdmDeclare { name, word_size, hard_locations, activation_radius, seed, fade } => {
            let mut new = Vec::with_capacity(word_size + hard_locations);
            new.extend((0..word_size).map(|j| format!("{}_{}", name, j)));
            new.extend((0..hard_locations).map(|i| format!("{}_loc_{}", name, i)));
            if let Some(clash) = new.iter().find(|x| m.things.contains(*x)) {
                return err(ln, format!("a thing :{} already exists; pick another sdm name", clash));
            }
            m.things.extend(new);
            let addr = Addresses::named(&name, seed, word_size, hard_locations);
            let mem = SdmMem { n: word_size, m: hard_locations, radius: activation_radius, fade, addr, rows: vec![0.0; hard_locations * word_size], stored: Vec::new() };
            m.sdm.insert(name, mem);
            Ok(())
        }
        Stmt::SdmWrite { name, what, text, key } => sdm_write(m, &name, &what, text, key, ln, out),
        Stmt::SdmRead { name, seed, iterated_reads, via, from } => {
            if via == Via::Pulls {
                return err(ln, "via: :pulls settles the pulls of a SETTLE model; kanerva reads via: :addresses, so run this program with settle");
            }
            if let Some(s) = seed {
                *rng = Rng::new(s);
            }
            let v = &m.sdm[&name];
            let noisy = |p: &[f64], noise: f64, rng: &mut Rng| -> Vec<f64> { p.iter().map(|&b| if rng.unit() < noise { -b } else { b }).collect() };
            let start: Vec<f64> = match &from {
                From::Address { name: c, noise } => {
                    let p = say::public(&v.stored, c, v.n).unwrap_or_else(|| code(c, v.n));
                    noisy(&p, *noise, rng)
                }
                From::Key { key, noise } => noisy(&keyed_read_address(key, v.n), *noise, rng),
                From::Noise => (0..v.n).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect(),
            };
            let n = v.n;
            let (got, rounds, awake) = iterated_read(&v.addr, v.radius, &start, iterated_reads, |i, sum| {
                for (j, c) in v.rows[i * n..(i + 1) * n].iter().enumerate() {
                    sum[j] += c;
                }
            });
            out.extend(say::sdm_read(&name, &from, via, rounds, awake, v.m, &got, &v.stored, n));
            Ok(())
        }
        Stmt::SoftDeclare { name, word_size, hard_locations, activation_probability, softness, gain, seed, write_samples } => {
            for nm in [format!("{}_addr_0", name), format!("{}_loc_0", name), format!("{}_data_0", name)] {
                if m.things.contains(&nm) {
                    return err(ln, format!("a thing :{} already exists; pick another softsdm name", nm));
                }
            }
            m.things.extend((0..word_size).map(|i| format!("{}_addr_{}", name, i)));
            m.things.extend((0..hard_locations).map(|l| format!("{}_loc_{}", name, l)));
            m.things.extend((0..word_size).map(|i| format!("{}_data_{}", name, i)));
            let mach = Machine::new(word_size, hard_locations, activation_probability, softness, gain, seed, write_samples);
            m.soft.insert(name, SoftMem { mach, stored: Vec::new() });
            Ok(())
        }
        Stmt::SoftWrite { name, what, text } => {
            let s = m.soft.get_mut(&name).unwrap();
            let n = s.mach.n;
            if let Some(t) = &text {
                if t.len() * 8 > n {
                    return err(ln, format!("{} bytes of text need {} bits; softsdm :{} has size {}", t.len(), t.len() * 8, name, n));
                }
            }
            if s.stored.iter().any(|(x, _)| *x == what) {
                return err(ln, format!(":{} is already written in :{}", what, name));
            }
            let p = pattern(&what, text.as_deref(), n);
            if place == Place::Model {
                s.mach.write(&p, &mut Rng::new(MODEL_WRITE_SEED));
            } else {
                s.mach.write(&p, rng);
            }
            s.stored.push((what, text));
            Ok(())
        }
        Stmt::SoftRead { name, seed, rounds, samples, burn, settle, cue } => {
            if let Some(v) = seed {
                *rng = Rng::new(v);
            }
            let s = &m.soft[&name];
            let (mut z, from) = soft_cue(s, &cue, rng);
            let mut trail = Vec::new();
            for _ in 0..rounds.max(1) {
                let (o, _) = if settle { s.mach.read_settle(&z, burn, samples, rng) } else { s.mach.read_pass(&z, samples, rng) };
                let sc = say::soft_scores(&o, &s.stored, s.mach.n);
                trail.push(sc.first().map(|(_, x)| format!("{:+.2}", x)).unwrap_or_default());
                z = o;
            }
            out.extend(say::soft_read(&name, &from, settle, s.mach.softness, rounds.max(1), &trail, &z, &s.stored, s.mach.n));
            Ok(())
        }
        Stmt::SoftAttend { name, seed, rounds, cue } => {
            if let Some(v) = seed {
                *rng = Rng::new(v);
            }
            let s = &m.soft[&name];
            let (z0, from) = soft_cue(s, &cue, rng);
            let pats: Vec<Vec<f64>> = s.stored.iter().map(|(x, t)| pattern(x, t.as_deref(), s.mach.n)).collect();
            let kern = s.mach.kernel_inf();
            let beta = Machine::fit_beta(&kern);
            out.push(say::soft_attend_head(&name, &from, s.mach.softness, beta));
            for label in ["mean-field read of this machine", "kernel read, infinitely many locations", "softmax attention"] {
                let mut z = z0.clone();
                for _ in 0..rounds.max(1) {
                    z = match label {
                        "mean-field read of this machine" => sign_of(&s.mach.mean_field(&z), &z),
                        "kernel read, infinitely many locations" => attention_read(&z, &pats, &Attn::Kernel(&kern)),
                        _ => attention_read(&z, &pats, &Attn::Softmax(beta)),
                    };
                }
                out.push(say::soft_attend_row(label, &say::soft_scores(&z, &s.stored, s.mach.n)));
            }
            Ok(())
        }
        Stmt::ScaleDeclare { name, word_size, hard_locations, activation_radius, seed } => {
            m.scale.insert(name, ScaleMem { n: word_size, m: hard_locations, radius: activation_radius, seed, words: Vec::new() });
            Ok(())
        }
        Stmt::ScalePut { name, what } => scale_put(m, &name, what, ln),
        Stmt::ScaleFill { name, count } => scale_put(m, &name, format!("#{}", count), ln),
        Stmt::ScaleRead { name, seed, iterated_reads, via, read_address, address_noise, wake } => {
            if let Some(s) = seed {
                *rng = Rng::new(s);
            }
            let k = &m.scale[&name];
            let store = scale_store(&name, k);
            let p: Vec<i8> = code(&read_address, k.n).into_iter().map(|v| if v > 0.0 { 1 } else { -1 }).collect();
            let cue = crate::bits::add_address_noise(&p, address_noise, rng);
            let r = match (via, wake) {
                (Via::Addresses, _) => store.read_addresses(&cue, iterated_reads),
                (Via::Pulls, Wake::Density) => store.read_pulls_at(&cue, iterated_reads, crate::store::density_threshold(&store, 0.1).0),
                (Via::Pulls, Wake::Top) => {
                    store.read_pulls_topk(&cue, iterated_reads, (crate::theory::ball(store.n, store.radius) * store.m as f64).round().max(1.0) as usize)
                }
                (Via::Pulls, Wake::Fixed) => store.read_pulls(&cue, iterated_reads),
            };
            let o = crate::bits::overlap(&r.z, &p);
            let written = k.words.contains(&read_address);
            out.push(say::scale_read(&name, &read_address, address_noise, via, r.rounds, r.awake, store.m, r.nonempty, o, written));
            Ok(())
        }
        Stmt::Refusal { word_size, load, level } => {
            out.push(say::refusal(word_size, load, level));
            Ok(())
        }
        Stmt::ContentTrack { word_size, hard_locations, load, address_noise, block, samples } => {
            out.push(say::contenttrack(word_size, hard_locations, load, address_noise, block, samples));
            Ok(())
        }
    }
}

fn sdm_write(m: &mut Model, name: &str, what: &str, text: Option<String>, key: Option<String>, ln: usize, out: &mut Vec<String>) -> Result<(), LangError> {
    let v = match m.sdm.get_mut(name) {
        Some(v) => v,
        None => return err(ln, format!("no sdm :{} (declare it with: sdm :{}, word-size: 256, hard-locations: 2000)", name, name)),
    };
    if v.stored.iter().any(|(x, _)| x == what) {
        return err(ln, format!(":{} is already written in :{}", what, name));
    }
    let (p, tag) = match (&text, &key) {
        (Some(t), Some(k)) => {
            if t.len() > keyed_capacity(v.n) {
                return err(ln, format!("keyed text of {} bytes is too long; sdm :{} holds at most {}", t.len(), name, keyed_capacity(v.n)));
            }
            (keyed_pattern(k, t, v.n), "#".to_string())
        }
        (Some(t), None) => {
            if t.len() * 8 > v.n {
                return err(ln, format!("{} bytes of text need {} things; sdm :{} has {}", t.len(), t.len() * 8, name, v.n));
            }
            (pattern(what, Some(t), v.n), format!("={}", t))
        }
        (None, _) => (code(what, v.n), String::new()),
    };
    // SETTLE's write, on the pulls' values: fade every counter, then add (2/n) p to each hard-location within
    // the activation-radius of p.
    let half = (4.0 / v.n as f64) / 2.0;
    if v.fade < 1.0 {
        for c in v.rows.iter_mut() {
            *c *= v.fade;
        }
    }
    let act = v.addr.awake(&p, v.radius);
    let n = v.n;
    for &i in &act {
        for j in 0..n {
            v.rows[i * n + j] += half * p[j];
        }
    }
    if act.is_empty() {
        out.push(say::sdm_write_warning(name, v.radius, what));
    }
    v.stored.push((what.to_string(), tag));
    Ok(())
}

fn soft_cue(s: &SoftMem, cue: &Cue, rng: &mut Rng) -> (Vec<f64>, String) {
    let n = s.mach.n;
    match &cue.address {
        Some(c) => {
            let saved = s.stored.iter().find(|(x, _)| x == c).and_then(|(_, t)| t.clone());
            let p = pattern(c, saved.as_deref(), n);
            (with_address_noise(&p, cue.noise, rng), format!("read-address :{} with {:.0}% address-noise", c, 100.0 * cue.noise))
        }
        None => ((0..n).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect(), "pure noise".to_string()),
    }
}

fn scale_put(m: &mut Model, name: &str, what: String, ln: usize) -> Result<(), LangError> {
    let k = m.scale.get_mut(name).unwrap();
    if !what.starts_with('#') && k.words.contains(&what) {
        return err(ln, format!(":{} is already in :{}", what, name));
    }
    if what.starts_with('#') && k.words.iter().any(|w| w.starts_with('#')) {
        return err(ln, format!(":{} is already filled; one fill per store", name));
    }
    k.words.push(what);
    Ok(())
}

/// The store an sdmscale names, rebuilt from its written names and fill, as SETTLE rebuilds it for each read.
fn scale_store(name: &str, k: &ScaleMem) -> Store {
    let seed = k.seed as u64;
    let mut st = Store::new(k.n, k.m, k.radius, seed);
    let mut ps = Vec::new();
    for w in &k.words {
        if let Some(c) = w.strip_prefix('#') {
            let c: usize = c.parse().unwrap_or(0);
            let mut r = Rng::new(seed_of(&format!("sdmscale-fill:{}:{}", name, seed)));
            for _ in 0..c {
                ps.push(crate::bits::random_pattern(k.n, &mut r));
            }
        } else {
            ps.push(code(w, k.n).into_iter().map(|v| if v > 0.0 { 1 } else { -1 }).collect());
        }
    }
    st.write_many(&ps);
    st
}
