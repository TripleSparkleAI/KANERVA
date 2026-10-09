//! The Rails face: sparse distributed memory spoken in Rust with the words of [`crate::words`].
//!
//! Every statement in the word list is a type here, every keyword is a method of the same name (hyphens become
//! underscores), and every verb is a method that writes or reads. A memory is declared with a builder, written
//! by name, and read with a chain of keywords:
//!
//! ```
//! use kanerva::rails::Sdm;
//!
//! let mut s = Sdm::build().word_size(256).hard_locations(2000).seed(1);
//! s.write("cat").unwrap();
//! s.write("dog").unwrap();
//! s.write_text("note", "meet at nine").unwrap();
//! let read = s.read("cat").address_noise(0.2).seed(1).answer().unwrap();
//! assert_eq!(read.recalled.as_deref(), Some("cat"));
//! assert_eq!(
//!     read.to_string(),
//!     "read :s from read-address :cat with 20% address-noise via addresses (2 iterated reads, 48 of 2000 hard \
//!      locations activated): :cat +1.00  :note +0.09  :dog -0.07  -> :cat"
//! );
//! ```
//!
//! The same memory in a `.kanerva` file, or in SETTLE's sdm statements, is written with the same words:
//!
//! ```text
//! sdm :s, word-size: 256, hard-locations: 2000, seed: 1
//! s.write :cat
//! s.read read-address: :cat, address-noise: 0.2, seed: 1
//! ```
//!
//! Each answer prints the very line that file prints: the arithmetic is the same arithmetic, step for step, over
//! the same random streams. A memory here stands alone, so a `via: :pulls` read matches a program in which the
//! memory is the only thing declared (a SETTLE model holding other things settles them too).
//!
//! <claudes_code_comments>
//! ** Function List **
//! Error                                     - a refusal, in the words the file face uses
//! Answer / Written                          - what a read returns / what a write returns
//! Via / Mode / Wake                         - the symbol choices (`via: :pulls`, `mode: :settle`, `wake: :top`)
//! Sdm::build / name / word_size / ...       - the hard memory: settings, then write / write_text / write_keyed,
//!                                             read / read_key / read_from_noise, rewind, shape, stored
//! SdmRead::read_address / key / ... / answer - one read of an Sdm, keyword by keyword
//! SoftSdm::build / ... / write / read / attend - the soft memory, its reads and its attention limit
//! SoftRead / SoftAttend                     - one read / one attend of a SoftSdm
//! SdmScale::build / ... / put / fill / read - the byte-counter memory for 10^5 to 10^6 hard-locations
//! ScaleRead                                 - one read of an SdmScale
//! Refusal::build / ... / answer             - the travel rule's threshold and the nearest-neighbour ceiling
//! ContentTrack::build / ... / answer        - TRACK-C's predicted recall for the content reads
//! rust_name(word)                           - the Rust spelling of a word (hyphens to underscores)
//!
//! ** Technical Review **
//! - This is the WORDS floor over the engine: the lab modules (address, store, soft, smap, refuse, track, keys,
//!   codes) do the arithmetic, this module names it. Defaults are read from crate::words, never repeated.
//! - Settings are set in the builder chain and fixed at the first write or read, when the memory is made and
//!   checked; a setter called after that panics, because changing a memory's shape under its contents is a bug.
//!   A setting out of range is an Error at that first use, with the same words the file face refuses it with.
//! - Every memory carries the read stream a SETTLE run block starts with (`Rng::new(0x5eed)`); a read's `seed`
//!   restarts it there and later reads run on from it, as in a run block. `rewind` starts a fresh run block.
//!   A softsdm write draws from its own stream started at 0x5eed_5d31 per write, as a model block write does.
//! - Sdm keeps its bit-counters scaled as the pulls they are in SETTLE (counter x gain / 2, gain 4 / word-size) and
//!   each data thing's lean, so `via: :pulls` settles exactly the energy SETTLE settles. The read by addresses is
//!   crate::address::iterated_read over Addresses::named(name, seed).
//! - SdmScale rebuilds its crate::store::Store from the written list in order (write_many), as SETTLE does, so the
//!   byte counters saturate in the same order; the store is cached until the next put or fill.
//! - The list is ONEPARSER's crate::words: five families, thirteen statements. SETTLE's `memory` (a Hopfield
//!   memory recalled by SETTLE's own sampler) is not in it, so it has no type here.
//! - Each read builder has `stream(&mut Rng)`, a Rust-only door: memories that share one stream read as the
//!   memories of one SETTLE run block do. It is not a word, so it sits outside the marked blocks.
//! - The parity test (tests/rails_words.rs) reads this file: each settings block is marked `// words: <statement>`
//!   or `// words: <statement>.<verb>`, each verb method `// verb: <statement>.<verb>`, each type
//!   `// statement: <family>`. The marked method and parameter names must equal the word list exactly, both ways:
//!   a declaration's settings are its keywords plus `name`; a method's are its keywords plus its positional
//!   arguments (`name`, `text`, `count`).
//!
//! </claudes_code_comments>

use std::fmt;

use crate::address::{iterated_read, Addresses};
use crate::bits::{add_address_noise, overlap as bits_overlap};
use crate::codes::{bits_text, code, overlap, pattern, seed_of, with_address_noise};
use crate::keys::{keyed_capacity, keyed_pattern, keyed_read, keyed_read_address};
use crate::refuse::{oracle_point, refusal_prob, travel_threshold};
use crate::rng::Rng;
use crate::smap::{radius_for_address_noise, search_window};
use crate::soft::{attention_read, sign_of, Attn, Machine};
use crate::store::{density_threshold, Store, WAKE};
use crate::theory::{ball, radius_for};
use crate::track::{block_theta, Content, Wake as TrackWake};
use crate::words::{statement, DefaultValue};

/// The seed a SETTLE run block starts its random stream with; every memory's reads start here.
pub const RUN_SEED: u64 = 0x5eed;

/// The seed a softsdm write's sampling stream starts at, fresh for each write, as a SETTLE model block write does.
pub const SOFT_WRITE_SEED: u64 = 0x5eed_5d31;

/// The name a memory takes when none is given, the one SETTLE's examples use. With the seed it fixes the addresses.
pub const DEFAULT_NAME: &str = "s";

/// A refusal: the setting, write or read is not one the memory can take, said in the words the file face uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn fail<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error(msg.into()))
}

/// What a read, recall, attend or calculation returned.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    /// The lines the file face prints for it, in order.
    pub lines: Vec<String>,
    /// The word the read ended on, as ±1 (empty for a calculation that builds no memory).
    pub word: Vec<f64>,
    /// Each stored name with its overlap with `word`, best first (the order the line prints them in).
    pub scores: Vec<(String, f64)>,
    /// The stored name the read came back to, when it came back to one.
    pub recalled: Option<String>,
    /// Text read back: a keyed note, or text stored under the recalled name.
    pub text: Option<String>,
    /// Iterated reads, rounds or sweeps made.
    pub steps: usize,
    /// Hard-locations activated on the last read, where the memory has them.
    pub activated: Option<usize>,
}

impl Answer {
    fn calculation(line: String) -> Answer {
        Answer { lines: vec![line], word: Vec::new(), scores: Vec::new(), recalled: None, text: None, steps: 0, activated: None }
    }
}

impl fmt::Display for Answer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.lines.join("\n"))
    }
}

/// What a write did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Written {
    /// How many hard-locations took the word, where the memory counts them.
    pub took: Option<usize>,
    /// Lines the file face prints for it (a warning when nothing took it).
    pub lines: Vec<String>,
}

impl Written {
    fn quiet(took: Option<usize>) -> Written {
        Written { took, lines: Vec::new() }
    }
}

/// `via:` - how a hard read finds its hard-locations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Via {
    /// `:addresses`: Kanerva's read, the hard-locations within the activation-radius of the read-address.
    Addresses,
    /// `:pulls`: the hard-locations woken by what they hold.
    Pulls,
}

/// `mode:` - how a soft read samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `:pass`: one way, address to data.
    Pass,
    /// `:settle`: the data pulls back on the hard-locations while it is sampled.
    Settle,
}

/// `wake:` - which rows a `via: :pulls` read of an SdmScale wakes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wake {
    /// `:fixed`: rows whose match exceeds 0.4 word-size.
    Fixed,
    /// `:density`: a threshold scaled with the rows' measured load.
    Density,
    /// `:top`: the activation-probability x hard-locations best rows.
    Top,
}

macro_rules! choice {
    ($t:ident { $($v:ident => $w:literal),* }) => {
        impl $t {
            /// Every value, in the order the word list gives them.
            pub const ALL: &'static [$t] = &[$($t::$v),*];
            /// The symbol the file face writes for it.
            pub fn word(self) -> &'static str {
                match self { $($t::$v => $w),* }
            }
            /// The value a symbol names.
            pub fn from_word(w: &str) -> Option<$t> {
                match w { $($w => Some($t::$v),)* _ => None }
            }
        }
    };
}
choice!(Via { Addresses => "addresses", Pulls => "pulls" });
choice!(Mode { Pass => "pass", Settle => "settle" });
choice!(Wake { Fixed => "fixed", Density => "density", Top => "top" });

/// The Rust spelling of a word: hyphens become underscores (`word-size` is the method `word_size`).
///
/// ```
/// assert_eq!(kanerva::rails::rust_name("activation-radius"), "activation_radius");
/// assert_eq!(kanerva::rails::rust_name("write-samples"), "write_samples");
/// ```
pub fn rust_name(word: &str) -> String {
    word.replace('-', "_")
}

fn default_of(family: &str, verb: &str, word: &str) -> DefaultValue {
    statement(family, verb)
        .and_then(|s| s.word(word))
        .unwrap_or_else(|| panic!("the word list has no `{}:` on {} {}", word, family, verb))
        .default
}

/// A numeric default, read from the word list rather than repeated here.
fn num(family: &str, verb: &str, word: &str) -> f64 {
    match default_of(family, verb, word) {
        DefaultValue::Num(x) => x,
        d => panic!("`{}:` has no numeric default ({:?})", word, d),
    }
}

/// A symbol default, read from the word list.
fn sym(family: &str, verb: &str, word: &str) -> &'static str {
    match default_of(family, verb, word) {
        DefaultValue::Sym(x) => x,
        d => panic!("`{}:` has no symbol default ({:?})", word, d),
    }
}

fn fixed(made: bool, what: &str) {
    assert!(!made, "{} is set in the builder chain, before the first write or read; the memory is already made", what);
}

fn sorted_abs(mut v: Vec<(String, f64)>) -> Vec<(String, f64)> {
    v.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());
    v
}

fn top(scores: &[(String, f64)], k: usize) -> String {
    scores.iter().take(k).map(|(n, o)| format!(":{} {:+.2}", n, o)).collect::<Vec<_>>().join("  ")
}

// ------------------------------------------------------------------------------------------------ sdm

#[derive(Clone)]
struct SdmCore {
    n: usize,
    m: usize,
    radius: usize,
    fade: f64,
    addr: Addresses,
    /// pull[i][j] = counter C[i][j] x gain / 2, hard-location major
    c: Vec<f64>,
    /// each data thing's lean: the sum of its pulls
    h: Vec<f64>,
    /// (name, tag): "" a code, "=text" masked text, "#" keyed text (not held)
    stored: Vec<(String, String)>,
}

impl SdmCore {
    fn public(&self, name: &str) -> Option<Vec<f64>> {
        let tag = &self.stored.iter().find(|(n, _)| n == name)?.1;
        match tag.chars().next() {
            Some('#') => None,
            Some('=') => Some(pattern(name, Some(&tag[1..]), self.n)),
            _ => Some(code(name, self.n)),
        }
    }

    fn write(&mut self, p: &[f64]) -> usize {
        let (n, m) = (self.n, self.m);
        let half = (4.0 / n as f64) / 2.0;
        if self.fade < 1.0 {
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..m {
                    sum += self.c[i * n + j];
                }
                self.h[j] += (self.fade - 1.0) * sum;
            }
            for v in self.c.iter_mut() {
                *v *= self.fade;
            }
        }
        let act = self.addr.awake(p, self.radius);
        for &i in &act {
            for j in 0..n {
                let d = half * p[j];
                self.c[i * n + j] += d;
                self.h[j] += d;
            }
        }
        act.len()
    }

    fn read_addresses(&self, cue: &[f64], iters: usize) -> (Vec<f64>, usize, usize) {
        let n = self.n;
        iterated_read(&self.addr, self.radius, cue, iters, |i, sum| {
            for (j, e) in self.c[i * n..i * n + n].iter().enumerate() {
                sum[j] += e;
            }
        })
    }

    /// The zero-temperature settle of the pulls, hard-locations then data, from a fresh random arrangement.
    fn read_pulls(&self, rng: &mut Rng, cue: &[f64], iters: usize) -> (Vec<f64>, usize, usize) {
        let (n, m) = (self.n, self.m);
        let all: Vec<f64> = (0..n + m).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect();
        let mut data = cue.to_vec();
        let mut loc = all[n..].to_vec();
        let lean_loc = -(4.0 / n as f64) * WAKE * n as f64 / 2.0;
        let hard = |x: f64, old: f64| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { old };
        let mut rounds = iters.max(1);
        for t in 0..iters.max(1) {
            for i in 0..m {
                let input = lean_loc + self.c[i * n..i * n + n].iter().zip(&data).map(|(&w, &s)| w * s).sum::<f64>();
                loc[i] = hard(input, -1.0);
            }
            let before = data.clone();
            for j in 0..n {
                let input = self.h[j] + (0..m).map(|i| self.c[i * n + j] * loc[i]).sum::<f64>();
                data[j] = hard(input, data[j]);
            }
            if data == before {
                rounds = t + 1;
                break;
            }
        }
        let awake = loc.iter().filter(|&&v| v > 0.0).count();
        (data, rounds, awake)
    }
}

// statement: sdm
/// Kanerva's sparse distributed memory: `hard-locations` fixed random addresses of `word-size` bits, each with a
/// row of bit-counters, activated within an `activation-radius`.
///
/// ```
/// use kanerva::rails::{Sdm, Via};
/// let mut s = Sdm::build().word_size(256).hard_locations(2000);
/// s.write("cat").unwrap();
/// let r = s.read("cat").address_noise(0.2).via(Via::Pulls).seed(1).answer().unwrap();
/// assert_eq!(r.recalled.as_deref(), Some("cat"));
/// ```
#[derive(Clone)]
pub struct Sdm {
    name: String,
    word_size: usize,
    hard_locations: usize,
    activation_radius: Option<usize>,
    seed: u64,
    fade: f64,
    core: Option<SdmCore>,
    stream: Rng,
}

/// The shape a memory was made with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// Bits per address and per word.
    pub word_size: usize,
    /// Hard-locations.
    pub hard_locations: usize,
    /// The activation-radius in bits.
    pub activation_radius: usize,
    /// Names written.
    pub written: usize,
}

// words: sdm
impl Sdm {
    /// The memory's name; with the seed it fixes the hard-locations' addresses.
    pub fn name(mut self, name: &str) -> Sdm {
        fixed(self.core.is_some(), "name");
        self.name = name.to_string();
        self
    }

    /// Bits per address and per word (16 to 4096). Kanerva: "The capacity of a location is referred to as the
    /// memory's word size, U" (K1993 P. 3; KANERVA_TERMS.md).
    pub fn word_size(mut self, word_size: usize) -> Sdm {
        fixed(self.core.is_some(), "word-size");
        self.word_size = word_size;
        self
    }

    /// How many hard-locations: the locations that are built, each with a fixed random address. Kanerva: "the
    /// hard locations are so few compared to the number of possible addresses" (K1993 P. 5).
    pub fn hard_locations(mut self, hard_locations: usize) -> Sdm {
        fixed(self.core.is_some(), "hard-locations");
        self.hard_locations = hard_locations;
        self
    }

    /// The activation-radius in bits (Kanerva's radius of activation H, K1993 P. 5); left out, the smallest whose
    /// ball holds 2% of all addresses (112 at word-size 256).
    pub fn activation_radius(mut self, activation_radius: usize) -> Sdm {
        fixed(self.core.is_some(), "activation-radius");
        self.activation_radius = Some(activation_radius);
        self
    }

    /// With the name, fixes the addresses.
    pub fn seed(mut self, seed: u64) -> Sdm {
        fixed(self.core.is_some(), "seed");
        self.seed = seed;
        self
    }

    /// Every write first multiplies every bit-counter by this (1: nothing fades).
    pub fn fade(mut self, fade: f64) -> Sdm {
        fixed(self.core.is_some(), "fade");
        self.fade = fade;
        self
    }
}

impl Sdm {
    /// A memory with every setting at its default; set them in the chain.
    pub fn build() -> Sdm {
        Sdm {
            name: DEFAULT_NAME.to_string(),
            word_size: num("sdm", "sdm", "word-size") as usize,
            hard_locations: num("sdm", "sdm", "hard-locations") as usize,
            activation_radius: None,
            seed: num("sdm", "sdm", "seed") as u64,
            fade: num("sdm", "sdm", "fade"),
            core: None,
            stream: Rng::new(RUN_SEED),
        }
    }

    fn core(&mut self) -> Result<&mut SdmCore, Error> {
        if self.core.is_none() {
            let (n, m, fade) = (self.word_size, self.hard_locations, self.fade);
            if !(16..=4096).contains(&n) {
                return fail("sdm word-size must be between 16 and 4096");
            }
            if !(1..=100_000).contains(&m) || n * m > 20_000_000 {
                return fail("sdm hard-locations must be at least 1, and word-size x hard-locations at most 20 million");
            }
            if !(0.0..=1.0).contains(&fade) || fade == 0.0 {
                return fail("fade must be above 0 and at most 1");
            }
            let radius = self.activation_radius.unwrap_or_else(|| radius_for(n, 0.02));
            if radius > n {
                return fail("activation-radius cannot be larger than word-size");
            }
            let addr = Addresses::named(&self.name, self.seed, n, m);
            self.core = Some(SdmCore { n, m, radius, fade, addr, c: vec![0.0; n * m], h: vec![0.0; n], stored: Vec::new() });
        }
        Ok(self.core.as_mut().unwrap())
    }

    fn put(&mut self, what: &str, txt: Option<&str>, key: Option<&str>) -> Result<Written, Error> {
        let name = self.name.clone();
        let v = self.core()?;
        if v.stored.iter().any(|(n, _)| n == what) {
            return fail(format!(":{} is already written in :{}", what, name));
        }
        let (p, tag) = match (txt, key) {
            (Some(t), Some(k)) => {
                if t.len() > keyed_capacity(v.n) {
                    return fail(format!("keyed text of {} bytes is too long; sdm :{} holds at most {}", t.len(), name, keyed_capacity(v.n)));
                }
                (keyed_pattern(k, t, v.n), "#".to_string())
            }
            (Some(t), None) => {
                if t.len() * 8 > v.n {
                    return fail(format!("{} bytes of text need {} things; sdm :{} has {}", t.len(), t.len() * 8, name, v.n));
                }
                (pattern(what, Some(t), v.n), format!("={}", t))
            }
            (None, _) => (code(what, v.n), String::new()),
        };
        let took = v.write(&p);
        let mut lines = Vec::new();
        if took == 0 {
            lines.push(format!("warning: no hard location of :{} is within activation-radius {} of :{}, so nothing was written", name, v.radius, what));
        }
        v.stored.push((what.to_string(), tag));
        Ok(Written { took: Some(took), lines })
    }

    // verb: sdm.write
    /// Write the fixed word that belongs to `name`.
    pub fn write(&mut self, name: &str) -> Result<Written, Error> {
        self.put(name, None, None)
    }

    // verb: sdm.write
    /// Write `text` under `name`, masked by the name so it looks random.
    pub fn write_text(&mut self, name: &str, text: &str) -> Result<Written, Error> {
        self.put(name, Some(text), None)
    }

    // verb: sdm.write
    /// Write `text` under `name`, turned by `key`: only the key reads it back.
    pub fn write_keyed(&mut self, name: &str, text: &str, key: &str) -> Result<Written, Error> {
        self.put(name, Some(text), Some(key))
    }

    // verb: sdm.read
    /// Read from the stored word `read_address` (default address-noise 0.3).
    pub fn read(&mut self, read_address: &str) -> SdmRead<'_> {
        SdmRead::new(self).read_address(read_address)
    }

    // verb: sdm.read
    /// Read the note written under `key`.
    pub fn read_key(&mut self, key: &str) -> SdmRead<'_> {
        SdmRead::new(self).key(key)
    }

    // verb: sdm.read
    /// Read from pure noise.
    pub fn read_from_noise(&mut self) -> SdmRead<'_> {
        SdmRead::new(self)
    }

    /// Start the read stream again, as a new run block does.
    pub fn rewind(&mut self) {
        self.stream = Rng::new(RUN_SEED);
    }

    /// The shape the memory was made with (making it, if nothing has been written yet).
    pub fn shape(&mut self) -> Result<Shape, Error> {
        let v = self.core()?;
        Ok(Shape { word_size: v.n, hard_locations: v.m, activation_radius: v.radius, written: v.stored.len() })
    }

    /// The names written, in order.
    pub fn stored(&self) -> Vec<String> {
        self.core.as_ref().map(|v| v.stored.iter().map(|(n, _)| n.clone()).collect()).unwrap_or_default()
    }
}

/// One read of an [`Sdm`], built keyword by keyword and made by [`SdmRead::answer`].
#[must_use = "a read is made by .answer()"]
pub struct SdmRead<'a> {
    sdm: &'a mut Sdm,
    read_address: Option<String>,
    key: Option<String>,
    address_noise: Option<f64>,
    iterated_reads: Option<usize>,
    via: Via,
    seed: Option<u64>,
    stream: Option<&'a mut Rng>,
}

// words: sdm.read
impl<'a> SdmRead<'a> {
    /// The stored word to read from.
    pub fn read_address(mut self, read_address: &str) -> SdmRead<'a> {
        self.read_address = Some(read_address.to_string());
        self
    }

    /// Read the note written under this key instead of a read-address.
    pub fn key(mut self, key: &str) -> SdmRead<'a> {
        self.key = Some(key.to_string());
        self
    }

    /// The share of the read-address's bits flipped first (0.3 from a read-address, 0 from a key). Kanerva:
    /// "Nine noisy words (20% noise) are stored, and the tenth is used as a retrieval cue" (K1993 P. 2).
    pub fn address_noise(mut self, address_noise: f64) -> SdmRead<'a> {
        self.address_noise = Some(address_noise);
        self
    }

    /// The most reads, stopping at a fixed point. Kanerva: "Such recovery typically takes several iterations
    /// (fewer than ten)" (K2009 p. 144).
    pub fn iterated_reads(mut self, iterated_reads: usize) -> SdmRead<'a> {
        self.iterated_reads = Some(iterated_reads);
        self
    }

    /// Kanerva's read by addresses, or the settle of the pulls.
    pub fn via(mut self, via: Via) -> SdmRead<'a> {
        self.via = via;
        self
    }

    /// Restart the read stream here.
    pub fn seed(mut self, seed: u64) -> SdmRead<'a> {
        self.seed = Some(seed);
        self
    }
}

impl<'a> SdmRead<'a> {
    fn new(sdm: &'a mut Sdm) -> SdmRead<'a> {
        let via = Via::from_word(sym("sdm", "read", "via")).unwrap();
        SdmRead { sdm, read_address: None, key: None, address_noise: None, iterated_reads: None, via, seed: None, stream: None }
    }

    /// Draw from `stream` instead of the memory's own read stream, so several memories share one, as the
    /// memories of one SETTLE run block do. Rust only: a file has one run block's stream already.
    pub fn stream(mut self, stream: &'a mut Rng) -> SdmRead<'a> {
        self.stream = Some(stream);
        self
    }

    /// Make the read.
    pub fn answer(self) -> Result<Answer, Error> {
        let sdm = self.sdm;
        let name = sdm.name.clone();
        sdm.core()?;
        let rng: &mut Rng = match self.stream {
            Some(r) => r,
            None => &mut sdm.stream,
        };
        if let Some(s) = self.seed {
            *rng = Rng::new(s);
        }
        let iters = self.iterated_reads.unwrap_or(num("sdm", "read", "iterated-reads") as usize);
        let v = sdm.core.as_ref().unwrap();
        let (start, from) = match (&self.read_address, &self.key) {
            (Some(_), Some(_)) => return fail("read takes read-address: or key:, not both"),
            (Some(c), None) => {
                let damage = self.address_noise.unwrap_or(0.3);
                let p = v.public(c).unwrap_or_else(|| code(c, v.n));
                let z: Vec<f64> = p.iter().map(|&b| if rng.unit() < damage { -b } else { b }).collect();
                (z, format!("read-address :{} with {:.0}% address-noise", c, 100.0 * damage))
            }
            (None, Some(k)) => {
                let damage = self.address_noise.unwrap_or(0.0);
                let p = keyed_read_address(k, v.n);
                (p.iter().map(|&b| if rng.unit() < damage { -b } else { b }).collect(), "a key".to_string())
            }
            (None, None) => ((0..v.n).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect(), "pure noise".to_string()),
        };
        let (got, rounds, awake) = match self.via {
            Via::Pulls => v.read_pulls(rng, &start, iters),
            Via::Addresses => v.read_addresses(&start, iters),
        };
        let head = format!("read :{} from {} via {} ({} iterated reads, {} of {} hard locations activated)", name, from, self.via.word(), rounds, awake, v.m);
        let mut a = Answer { lines: Vec::new(), word: got, scores: Vec::new(), recalled: None, text: None, steps: rounds, activated: Some(awake) };
        if let Some(k) = &self.key {
            a.text = keyed_read(k, &a.word);
            a.lines.push(match &a.text {
                Some(t) => format!("{}: text \"{}\"", head, t),
                None => format!("{}: nothing readable", head),
            });
            return Ok(a);
        }
        a.scores = sorted_abs(v.stored.iter().filter_map(|(n, _)| v.public(n).map(|p| (n.clone(), overlap(&a.word, &p)))).collect());
        let verdict = match a.scores.first() {
            Some((n, o)) if *o >= 0.9 => format!("-> :{}", n),
            Some((n, o)) => format!("-> nothing clear (closest :{} at {:+.2})", n, o),
            None => "-> nothing is written".to_string(),
        };
        a.lines.push(format!("{}: {}  {}", head, top(&a.scores, 3), verdict));
        if let Some((n, o)) = a.scores.first() {
            if *o >= 0.9 {
                a.recalled = Some(n.clone());
                if let Some((_, tag)) = v.stored.iter().find(|(x, _)| x == n) {
                    if let Some(t) = tag.strip_prefix('=') {
                        let mask = code(n, v.n);
                        let bits: Vec<f64> = a.word.iter().zip(&mask).map(|(x, y)| x * y).collect();
                        let text = bits_text(&bits, t.len());
                        a.lines.push(format!("  text: \"{}\"", text));
                        a.text = Some(text);
                    }
                }
            }
        }
        Ok(a)
    }
}

// ------------------------------------------------------------------------------------------------ softsdm

// statement: softsdm
/// The sparse distributed memory with a soft activation-radius: each hard-location is activated with a
/// probability that falls smoothly with distance. `softness` 0 is the hard memory.
///
/// ```
/// use kanerva::rails::{SoftSdm, Mode};
/// let mut s = SoftSdm::build().name("soft").activation_probability(0.05).softness(0.25).seed(2);
/// for w in ["cat", "dog", "owl"] { s.write(w).unwrap(); }
/// let r = s.read("cat").address_noise(0.2).seed(1).answer().unwrap();
/// assert_eq!(r.recalled.as_deref(), Some("cat"));
/// assert!(s.read("cat").mode(Mode::Settle).answer().is_ok());
/// ```
#[derive(Clone)]
pub struct SoftSdm {
    name: String,
    word_size: usize,
    hard_locations: usize,
    activation_probability: f64,
    softness: f64,
    gain: f64,
    seed: u64,
    write_samples: usize,
    mach: Option<Machine>,
    stored: Vec<(String, Option<String>)>,
    stream: Rng,
}

// words: softsdm
impl SoftSdm {
    /// The memory's name.
    pub fn name(mut self, name: &str) -> SoftSdm {
        fixed(self.mach.is_some(), "name");
        self.name = name.to_string();
        self
    }

    /// Bits per address and per word (8 to 4096).
    pub fn word_size(mut self, word_size: usize) -> SoftSdm {
        fixed(self.mach.is_some(), "word-size");
        self.word_size = word_size;
        self
    }

    /// How many hard-locations.
    pub fn hard_locations(mut self, hard_locations: usize) -> SoftSdm {
        fixed(self.mach.is_some(), "hard-locations");
        self.hard_locations = hard_locations;
        self
    }

    /// The expected share of hard-locations one address activates.
    pub fn activation_probability(mut self, activation_probability: f64) -> SoftSdm {
        fixed(self.mach.is_some(), "activation-probability");
        self.activation_probability = activation_probability;
        self
    }

    /// How blurred the edge of the activation-radius is (0: the hard memory). Measured (runs/softsdm,
    /// REPORT_SOFTSDM.md §6 and §9): at softness 0 the machine equals an independent hard SDM on every non-tie
    /// bit; at softness 1 a one-round read agrees with softmax attention on 99.2% of bits and both recall 0%.
    pub fn softness(mut self, softness: f64) -> SoftSdm {
        fixed(self.mach.is_some(), "softness");
        self.softness = softness;
        self
    }

    /// The strength of the pulls in mode :settle.
    pub fn gain(mut self, gain: f64) -> SoftSdm {
        fixed(self.mach.is_some(), "gain");
        self.gain = gain;
        self
    }

    /// Fixes the hard-locations' addresses.
    pub fn seed(mut self, seed: u64) -> SoftSdm {
        fixed(self.mach.is_some(), "seed");
        self.seed = seed;
        self
    }

    /// Samples of the hard-locations per write.
    pub fn write_samples(mut self, write_samples: usize) -> SoftSdm {
        fixed(self.mach.is_some(), "write-samples");
        self.write_samples = write_samples;
        self
    }
}

impl SoftSdm {
    /// A soft memory with every setting at its default; set them in the chain.
    pub fn build() -> SoftSdm {
        SoftSdm {
            name: DEFAULT_NAME.to_string(),
            word_size: num("softsdm", "softsdm", "word-size") as usize,
            hard_locations: num("softsdm", "softsdm", "hard-locations") as usize,
            activation_probability: num("softsdm", "softsdm", "activation-probability"),
            softness: num("softsdm", "softsdm", "softness"),
            gain: num("softsdm", "softsdm", "gain"),
            seed: num("softsdm", "softsdm", "seed") as u64,
            write_samples: num("softsdm", "softsdm", "write-samples") as usize,
            mach: None,
            stored: Vec::new(),
            stream: Rng::new(RUN_SEED),
        }
    }

    fn mach(&mut self) -> Result<&mut Machine, Error> {
        if self.mach.is_none() {
            let (n, lm, fire, soft, gain) = (self.word_size, self.hard_locations, self.activation_probability, self.softness, self.gain);
            if !(8..=4096).contains(&n) {
                return fail("softsdm size must be between 8 and 4096");
            }
            if lm < 1 || lm * n > 4_000_000 {
                return fail("softsdm needs at least 1 hard location and at most 4,000,000 location-bits (hard-locations x word-size)");
            }
            if !(fire > 0.0 && fire < 1.0) {
                return fail("activation-probability is a fraction of hard locations, above 0 and below 1");
            }
            if soft < 0.0 || gain <= 0.0 {
                return fail("softness must be at least 0 and gain above 0");
            }
            self.mach = Some(Machine::new(n, lm, fire, soft, gain, self.seed, self.write_samples));
        }
        Ok(self.mach.as_mut().unwrap())
    }

    fn put(&mut self, what: &str, saved: Option<&str>) -> Result<Written, Error> {
        let name = self.name.clone();
        let n = self.mach()?.n;
        if let Some(t) = saved {
            if t.len() * 8 > n {
                return fail(format!("{} bytes of text need {} bits; softsdm :{} has size {}", t.len(), t.len() * 8, name, n));
            }
        }
        if self.stored.iter().any(|(x, _)| x == what) {
            return fail(format!(":{} is already written in :{}", what, name));
        }
        let p = pattern(what, saved, n);
        let touched = self.mach()?.write(&p, &mut Rng::new(SOFT_WRITE_SEED));
        self.stored.push((what.to_string(), saved.map(str::to_string)));
        Ok(Written::quiet(Some(touched.len())))
    }

    // verb: softsdm.write
    /// Write the fixed word that belongs to `name`.
    pub fn write(&mut self, name: &str) -> Result<Written, Error> {
        self.put(name, None)
    }

    // verb: softsdm.write
    /// Write `text` under `name`, masked by the name.
    pub fn write_text(&mut self, name: &str, text: &str) -> Result<Written, Error> {
        self.put(name, Some(text))
    }

    // verb: softsdm.read
    /// Read from the stored word `read_address` (default address-noise 0.3).
    pub fn read(&mut self, read_address: &str) -> SoftRead<'_> {
        SoftRead::new(self).read_address(read_address)
    }

    // verb: softsdm.read
    /// Read from pure noise.
    pub fn read_from_noise(&mut self) -> SoftRead<'_> {
        SoftRead::new(self)
    }

    // verb: softsdm.attend
    /// The attention limit of a read from the stored word `read_address`, computed outside the sampler.
    pub fn attend(&mut self, read_address: &str) -> SoftAttend<'_> {
        SoftAttend::new(self).read_address(read_address)
    }

    // verb: softsdm.attend
    /// The attention limit of a read from pure noise.
    pub fn attend_from_noise(&mut self) -> SoftAttend<'_> {
        SoftAttend::new(self)
    }

    /// Start the read stream again, as a new run block does.
    pub fn rewind(&mut self) {
        self.stream = Rng::new(RUN_SEED);
    }

    /// The names written, in order.
    pub fn stored(&self) -> Vec<String> {
        self.stored.iter().map(|(n, _)| n.clone()).collect()
    }

}

type SoftStored = [(String, Option<String>)];

fn soft_cue(n: usize, stored: &SoftStored, read_address: &Option<String>, noise: Option<f64>, rng: &mut Rng) -> Result<(Vec<f64>, String), Error> {
    let damage = noise.unwrap_or(num("softsdm", "read", "address-noise"));
    if !(0.0..=1.0).contains(&damage) {
        return fail("address-noise is a fraction between 0 and 1");
    }
    Ok(match read_address {
        Some(c) => {
            let saved = stored.iter().find(|(x, _)| x == c).and_then(|(_, t)| t.clone());
            let p = pattern(c, saved.as_deref(), n);
            (with_address_noise(&p, damage, rng), format!("read-address :{} with {:.0}% address-noise", c, 100.0 * damage))
        }
        None => ((0..n).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect(), "pure noise".to_string()),
    })
}

fn soft_scores(stored: &SoftStored, got: &[f64], n: usize) -> Vec<(String, f64)> {
    let mut v: Vec<(String, f64)> = stored.iter().map(|(nm, t)| (nm.clone(), overlap(got, &pattern(nm, t.as_deref(), n)))).collect();
    v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    v
}

fn soft_verdict(sc: &[(String, f64)]) -> String {
    match sc.first() {
        Some((n, o)) if *o >= 0.9 => format!("-> :{}", n),
        Some((n, o)) => format!("-> nothing clear (closest :{} at {:+.2})", n, o),
        None => "-> nothing is written".to_string(),
    }
}

/// One read of a [`SoftSdm`], built keyword by keyword and made by [`SoftRead::answer`].
#[must_use = "a read is made by .answer()"]
pub struct SoftRead<'a> {
    soft: &'a mut SoftSdm,
    read_address: Option<String>,
    address_noise: Option<f64>,
    rounds: Option<usize>,
    samples: Option<usize>,
    mode: Mode,
    burn: Option<usize>,
    seed: Option<u64>,
    stream: Option<&'a mut Rng>,
}

// words: softsdm.read
impl<'a> SoftRead<'a> {
    /// The stored word to read from.
    pub fn read_address(mut self, read_address: &str) -> SoftRead<'a> {
        self.read_address = Some(read_address.to_string());
        self
    }

    /// The share of the read-address's bits flipped first.
    pub fn address_noise(mut self, address_noise: f64) -> SoftRead<'a> {
        self.address_noise = Some(address_noise);
        self
    }

    /// How many reads, each answer feeding the next.
    pub fn rounds(mut self, rounds: usize) -> SoftRead<'a> {
        self.rounds = Some(rounds);
        self
    }

    /// Samples of the hard-locations per read.
    pub fn samples(mut self, samples: usize) -> SoftRead<'a> {
        self.samples = Some(samples);
        self
    }

    /// One way, or settling. A known limit: in a small probe at softness 0.25 the settling read landed on a
    /// mixture of stored words where the one-way read recalled the word exactly; the cause is not established
    /// (README, Known limits). `examples/rails.rs` shows it (`:owl +0.77`, nothing clear).
    pub fn mode(mut self, mode: Mode) -> SoftRead<'a> {
        self.mode = mode;
        self
    }

    /// Sweeps thrown away before the samples are kept, for mode :settle.
    pub fn burn(mut self, burn: usize) -> SoftRead<'a> {
        self.burn = Some(burn);
        self
    }

    /// Restart the read stream here.
    pub fn seed(mut self, seed: u64) -> SoftRead<'a> {
        self.seed = Some(seed);
        self
    }
}

impl<'a> SoftRead<'a> {
    fn new(soft: &'a mut SoftSdm) -> SoftRead<'a> {
        let mode = Mode::from_word(sym("softsdm", "read", "mode")).unwrap();
        SoftRead { soft, read_address: None, address_noise: None, rounds: None, samples: None, mode, burn: None, seed: None, stream: None }
    }

    /// Draw from `stream` instead of the memory's own read stream (see [`SdmRead::stream`]). Rust only.
    pub fn stream(mut self, stream: &'a mut Rng) -> SoftRead<'a> {
        self.stream = Some(stream);
        self
    }

    /// Make the read.
    pub fn answer(self) -> Result<Answer, Error> {
        let s = self.soft;
        s.mach()?;
        let rng: &mut Rng = match self.stream {
            Some(r) => r,
            None => &mut s.stream,
        };
        if let Some(x) = self.seed {
            *rng = Rng::new(x);
        }
        let rounds = self.rounds.unwrap_or(num("softsdm", "read", "rounds") as usize);
        let samples = self.samples.unwrap_or(num("softsdm", "read", "samples") as usize);
        let burn = self.burn.unwrap_or(num("softsdm", "read", "burn") as usize);
        let settle = self.mode == Mode::Settle;
        let mach = s.mach.as_ref().unwrap();
        let n = mach.n;
        let (mut cue, from) = soft_cue(n, &s.stored, &self.read_address, self.address_noise, rng)?;
        let mut trail = Vec::new();
        for _ in 0..rounds.max(1) {
            let (out, _) = if settle { mach.read_settle(&cue, burn, samples, rng) } else { mach.read_pass(&cue, samples, rng) };
            let sc = soft_scores(&s.stored, &out, n);
            trail.push(sc.first().map(|(_, o)| format!("{:+.2}", o)).unwrap_or_default());
            cue = out;
        }
        let sc = soft_scores(&s.stored, &cue, n);
        let mut lines = vec![format!(
            "read :{} from {} ({}, softness {}, {} rounds, best overlap by round {}): {}  {}",
            s.name,
            from,
            self.mode.word(),
            mach.softness,
            rounds.max(1),
            trail.join(" "),
            top(&sc, 3),
            soft_verdict(&sc)
        )];
        let (mut recalled, mut text) = (None, None);
        if let Some((nm, o)) = sc.first() {
            if *o >= 0.9 {
                recalled = Some(nm.clone());
                if let Some((_, Some(t))) = s.stored.iter().find(|(x, _)| x == nm) {
                    let mask = code(nm, n);
                    let bits: Vec<f64> = cue.iter().zip(&mask).map(|(v, k)| v * k).collect();
                    let tx = bits_text(&bits, t.len());
                    lines.push(format!("  text: \"{}\"", tx));
                    text = Some(tx);
                }
            }
        }
        Ok(Answer { lines, word: cue, scores: sc, recalled, text, steps: rounds.max(1), activated: None })
    }
}

/// One attend of a [`SoftSdm`]: the mean-field read, the infinite-location kernel read and the fitted softmax
/// attention read, side by side. Made by [`SoftAttend::answer`].
#[must_use = "an attend is made by .answer()"]
pub struct SoftAttend<'a> {
    soft: &'a mut SoftSdm,
    read_address: Option<String>,
    address_noise: Option<f64>,
    rounds: Option<usize>,
    seed: Option<u64>,
    stream: Option<&'a mut Rng>,
}

// words: softsdm.attend
impl<'a> SoftAttend<'a> {
    /// The stored word to read from.
    pub fn read_address(mut self, read_address: &str) -> SoftAttend<'a> {
        self.read_address = Some(read_address.to_string());
        self
    }

    /// The share of the read-address's bits flipped first.
    pub fn address_noise(mut self, address_noise: f64) -> SoftAttend<'a> {
        self.address_noise = Some(address_noise);
        self
    }

    /// How many reads, each answer feeding the next.
    pub fn rounds(mut self, rounds: usize) -> SoftAttend<'a> {
        self.rounds = Some(rounds);
        self
    }

    /// Restart the read stream here.
    pub fn seed(mut self, seed: u64) -> SoftAttend<'a> {
        self.seed = Some(seed);
        self
    }
}

impl<'a> SoftAttend<'a> {
    fn new(soft: &'a mut SoftSdm) -> SoftAttend<'a> {
        SoftAttend { soft, read_address: None, address_noise: None, rounds: None, seed: None, stream: None }
    }

    /// Draw from `stream` instead of the memory's own read stream (see [`SdmRead::stream`]). Rust only.
    pub fn stream(mut self, stream: &'a mut Rng) -> SoftAttend<'a> {
        self.stream = Some(stream);
        self
    }

    /// Make the three reads. `word` and `scores` are the softmax attention read's, the last of the three.
    pub fn answer(self) -> Result<Answer, Error> {
        let s = self.soft;
        s.mach()?;
        let rng: &mut Rng = match self.stream {
            Some(r) => r,
            None => &mut s.stream,
        };
        if let Some(x) = self.seed {
            *rng = Rng::new(x);
        }
        let rounds = self.rounds.unwrap_or(num("softsdm", "attend", "rounds") as usize);
        let mach = s.mach.as_ref().unwrap();
        let n = mach.n;
        let (cue0, from) = soft_cue(n, &s.stored, &self.read_address, self.address_noise, rng)?;
        let pats: Vec<Vec<f64>> = s.stored.iter().map(|(nm, t)| pattern(nm, t.as_deref(), n)).collect();
        let kern = mach.kernel_inf();
        let beta = Machine::fit_beta(&kern);
        let mut lines = vec![format!("attend :{} from {} (softness {}, fitted softmax inverse temperature {:.1} on cosine):", s.name, from, mach.softness, beta)];
        let mut last = (Vec::new(), Vec::new());
        for label in ["mean-field read of this machine", "kernel read, infinitely many locations", "softmax attention"] {
            let mut cue = cue0.clone();
            for _ in 0..rounds.max(1) {
                cue = match label {
                    "mean-field read of this machine" => sign_of(&mach.mean_field(&cue), &cue),
                    "kernel read, infinitely many locations" => attention_read(&cue, &pats, &Attn::Kernel(&kern)),
                    _ => attention_read(&cue, &pats, &Attn::Softmax(beta)),
                };
            }
            let sc = soft_scores(&s.stored, &cue, n);
            lines.push(format!("  {:<40} {}  {}", label, top(&sc, 2), soft_verdict(&sc)));
            last = (cue, sc);
        }
        let recalled = last.1.first().filter(|(_, o)| *o >= 0.9).map(|(nm, _)| nm.clone());
        Ok(Answer { lines, word: last.0, scores: last.1, recalled, text: None, steps: rounds.max(1), activated: None })
    }
}

// ------------------------------------------------------------------------------------------------ sdmscale

// statement: sdmscale
/// The byte-counter memory for 10^5 to 10^6 hard-locations ([`crate::store::Store`]).
///
/// ```
/// use kanerva::rails::{SdmScale, Via, Wake};
/// let mut k = SdmScale::build().name("k").hard_locations(20_000).activation_probability(0.01);
/// k.put("cat").unwrap();
/// k.fill(50).unwrap();
/// let r = k.read("cat").address_noise(0.2).seed(1).answer().unwrap();
/// assert_eq!(r.recalled.as_deref(), Some("cat"));
/// assert!(k.read("cat").via(Via::Pulls).wake(Wake::Top).answer().is_ok());
/// ```
#[derive(Clone)]
pub struct SdmScale {
    name: String,
    word_size: usize,
    hard_locations: usize,
    activation_probability: f64,
    activation_radius: Option<usize>,
    tolerate_noise: Option<f64>,
    seed: u64,
    made: Option<(usize, usize, usize)>,
    words: Vec<String>,
    cache: Option<(usize, Store)>,
    stream: Rng,
}

// words: sdmscale
impl SdmScale {
    /// The memory's name.
    pub fn name(mut self, name: &str) -> SdmScale {
        fixed(self.made.is_some(), "name");
        self.name = name.to_string();
        self
    }

    /// Bits per address and per word (16 to 4096).
    pub fn word_size(mut self, word_size: usize) -> SdmScale {
        fixed(self.made.is_some(), "word-size");
        self.word_size = word_size;
        self
    }

    /// How many hard-locations (1 to 2,000,000).
    pub fn hard_locations(mut self, hard_locations: usize) -> SdmScale {
        fixed(self.made.is_some(), "hard-locations");
        self.hard_locations = hard_locations;
        self
    }

    /// The share of all addresses within the activation-radius.
    pub fn activation_probability(mut self, activation_probability: f64) -> SdmScale {
        fixed(self.made.is_some(), "activation-probability");
        self.activation_probability = activation_probability;
        self
    }

    /// The activation-radius in bits; overrides activation-probability and tolerate-noise.
    pub fn activation_radius(mut self, activation_radius: usize) -> SdmScale {
        fixed(self.made.is_some(), "activation-radius");
        self.activation_radius = Some(activation_radius);
        self
    }

    /// Choose the activation-radius whose predicted capacity is largest at this address-noise (below 0.5). Why
    /// it exists (runs/sdmscale, REPORT_SDMSCALE.md §3): with the signal-to-noise activation-radius, capacity at
    /// 30% and 40% address-noise does not grow with hard-locations, because a 40%-noisy read-address shares about
    /// one hard-location with its stored word at every M.
    pub fn tolerate_noise(mut self, tolerate_noise: f64) -> SdmScale {
        fixed(self.made.is_some(), "tolerate-noise");
        self.tolerate_noise = Some(tolerate_noise);
        self
    }

    /// Fixes the hard-locations' addresses.
    pub fn seed(mut self, seed: u64) -> SdmScale {
        fixed(self.made.is_some(), "seed");
        self.seed = seed;
        self
    }
}

impl SdmScale {
    /// A memory with every setting at its default; set them in the chain.
    pub fn build() -> SdmScale {
        SdmScale {
            name: DEFAULT_NAME.to_string(),
            word_size: num("sdmscale", "sdmscale", "word-size") as usize,
            hard_locations: num("sdmscale", "sdmscale", "hard-locations") as usize,
            activation_probability: num("sdmscale", "sdmscale", "activation-probability"),
            activation_radius: None,
            tolerate_noise: None,
            seed: num("sdmscale", "sdmscale", "seed") as u64,
            made: None,
            words: Vec::new(),
            cache: None,
            stream: Rng::new(RUN_SEED),
        }
    }

    fn made(&mut self) -> Result<(usize, usize, usize), Error> {
        if let Some(x) = self.made {
            return Ok(x);
        }
        let (n, mm, fire) = (self.word_size, self.hard_locations, self.activation_probability);
        if !(16..=4096).contains(&n) {
            return fail("sdmscale word-size must be between 16 and 4096");
        }
        if !(1..=2_000_000).contains(&mm) || n * mm > 1_100_000_000 {
            return fail("sdmscale hard-locations must be 1 to 2,000,000, and word-size x hard-locations at most 1.1 billion bytes");
        }
        if !(fire > 0.0 && fire < 1.0) {
            return fail("activation-probability must be between 0 and 1");
        }
        let by_fire = match self.tolerate_noise.filter(|&t| t >= 0.0) {
            Some(t) => {
                if t >= 0.5 {
                    return fail("tolerate-noise is an address-noise fraction below 0.5");
                }
                let (lo, hi) = search_window(n, mm);
                radius_for_address_noise(n, mm, t, lo, hi).0
            }
            None => radius_for(n, fire),
        };
        let radius = self.activation_radius.unwrap_or(by_fire);
        if radius > n {
            return fail("activation-radius cannot be larger than word-size");
        }
        self.made = Some((n, mm, radius));
        Ok((n, mm, radius))
    }

    fn store(&mut self) -> Result<&Store, Error> {
        let (n, mm, radius) = self.made()?;
        if self.cache.as_ref().map(|c| c.0) != Some(self.words.len()) {
            let mut st = Store::new(n, mm, radius, self.seed);
            let mut ps = Vec::new();
            for w in &self.words {
                if let Some(k) = w.strip_prefix('#') {
                    let k: usize = k.parse().unwrap_or(0);
                    let mut r = Rng::new(seed_of(&format!("sdmscale-fill:{}:{}", self.name, self.seed)));
                    for _ in 0..k {
                        ps.push(crate::bits::random_pattern(n, &mut r));
                    }
                } else {
                    ps.push(code(w, n).into_iter().map(|v| if v > 0.0 { 1 } else { -1 }).collect());
                }
            }
            st.write_many(&ps);
            self.cache = Some((self.words.len(), st));
        }
        Ok(&self.cache.as_ref().unwrap().1)
    }

    // verb: sdmscale.put
    /// Write the fixed word that belongs to `name`.
    pub fn put(&mut self, name: &str) -> Result<Written, Error> {
        self.made()?;
        if self.words.iter().any(|w| w == name) {
            return fail(format!(":{} is already in :{}", name, self.name));
        }
        self.words.push(name.to_string());
        Ok(Written::quiet(None))
    }

    // verb: sdmscale.fill
    /// Write `count` random words (one fill per store).
    pub fn fill(&mut self, count: usize) -> Result<Written, Error> {
        self.made()?;
        if self.words.iter().any(|w| w.starts_with('#')) {
            return fail(format!(":{} is already filled; one fill per store", self.name));
        }
        self.words.push(format!("#{}", count));
        Ok(Written::quiet(None))
    }

    // verb: sdmscale.read
    /// Read from `read_address`, written or not (default address-noise 0.2).
    pub fn read(&mut self, read_address: &str) -> ScaleRead<'_> {
        ScaleRead::new(self).read_address(read_address)
    }

    /// Start the read stream again, as a new run block does.
    pub fn rewind(&mut self) {
        self.stream = Rng::new(RUN_SEED);
    }

    /// The shape the memory was made with (making it, if nothing has been written yet).
    pub fn shape(&mut self) -> Result<Shape, Error> {
        let (n, mm, radius) = self.made()?;
        Ok(Shape { word_size: n, hard_locations: mm, activation_radius: radius, written: self.words.len() })
    }
}

/// One read of an [`SdmScale`], built keyword by keyword and made by [`ScaleRead::answer`].
#[must_use = "a read is made by .answer()"]
pub struct ScaleRead<'a> {
    scale: &'a mut SdmScale,
    read_address: Option<String>,
    address_noise: Option<f64>,
    iterated_reads: Option<usize>,
    via: Via,
    wake: Wake,
    seed: Option<u64>,
    stream: Option<&'a mut Rng>,
}

// words: sdmscale.read
impl<'a> ScaleRead<'a> {
    /// The word to read from.
    pub fn read_address(mut self, read_address: &str) -> ScaleRead<'a> {
        self.read_address = Some(read_address.to_string());
        self
    }

    /// The share of the read-address's bits flipped first.
    pub fn address_noise(mut self, address_noise: f64) -> ScaleRead<'a> {
        self.address_noise = Some(address_noise);
        self
    }

    /// The most reads, stopping at a fixed point.
    pub fn iterated_reads(mut self, iterated_reads: usize) -> ScaleRead<'a> {
        self.iterated_reads = Some(iterated_reads);
        self
    }

    /// Kanerva's read by addresses, or the hard-locations woken by what they hold.
    pub fn via(mut self, via: Via) -> ScaleRead<'a> {
        self.via = via;
        self
    }

    /// For via :pulls, which rows wake. Measured (runs/sdmradius, REPORT_SDMRADIUS.md §7): at 1,000,000
    /// hard-locations of 256 bits the top-k read (`via: :pulls, wake: :top`) held 70,000 / 30,000 / 10,000 / 10
    /// words at 10/20/30/40% address-noise (P90), where a Hopfield net with as many pulls held 100 / 70 / 30 / 3.
    pub fn wake(mut self, wake: Wake) -> ScaleRead<'a> {
        self.wake = wake;
        self
    }

    /// Restart the read stream here.
    pub fn seed(mut self, seed: u64) -> ScaleRead<'a> {
        self.seed = Some(seed);
        self
    }
}

impl<'a> ScaleRead<'a> {
    fn new(scale: &'a mut SdmScale) -> ScaleRead<'a> {
        let via = Via::from_word(sym("sdmscale", "read", "via")).unwrap();
        let wake = Wake::from_word(sym("sdmscale", "read", "wake")).unwrap();
        ScaleRead { scale, read_address: None, address_noise: None, iterated_reads: None, via, wake, seed: None, stream: None }
    }

    /// Draw from `stream` instead of the memory's own read stream (see [`SdmRead::stream`]). Rust only.
    pub fn stream(mut self, stream: &'a mut Rng) -> ScaleRead<'a> {
        self.stream = Some(stream);
        self
    }

    /// Make the read.
    pub fn answer(self) -> Result<Answer, Error> {
        let k = self.scale;
        k.made()?;
        let iters = self.iterated_reads.unwrap_or(num("sdmscale", "read", "iterated-reads") as usize);
        let pulls = self.via == Via::Pulls;
        let cue_name = match &self.read_address {
            Some(c) => c.clone(),
            None => return fail("read needs read-address: :name"),
        };
        let damage = self.address_noise.unwrap_or(num("sdmscale", "read", "address-noise"));
        let written = k.words.contains(&cue_name);
        let name = k.name.clone();
        k.store()?;
        let rng: &mut Rng = match self.stream {
            Some(r) => r,
            None => &mut k.stream,
        };
        if let Some(x) = self.seed {
            *rng = Rng::new(x);
        }
        let store = &k.cache.as_ref().unwrap().1;
        let p: Vec<i8> = code(&cue_name, store.n).into_iter().map(|v| if v > 0.0 { 1 } else { -1 }).collect();
        let cue = add_address_noise(&p, damage, rng);
        if !pulls && self.wake != Wake::Fixed {
            return fail("wake: applies to via: :pulls");
        }
        let out = if !pulls {
            store.read_addresses(&cue, iters)
        } else if self.wake == Wake::Density {
            store.read_pulls_at(&cue, iters, density_threshold(store, 0.1).0)
        } else if self.wake == Wake::Top {
            store.read_pulls_topk(&cue, iters, (ball(store.n, store.radius) * store.m as f64).round().max(1.0) as usize)
        } else {
            store.read_pulls(&cue, iters)
        };
        let o = bits_overlap(&out.z, &p);
        let verdict = if o >= 0.95 && written {
            format!("-> :{}", cue_name)
        } else if o >= 0.95 {
            format!("-> back to :{} though it was never written (the read did not move it)", cue_name)
        } else {
            format!("-> nothing clear (overlap with :{} {:+.2})", cue_name, o)
        };
        let line = format!(
            "read :{} from read-address :{} with {:.0}% address-noise via {} ({} iterated reads, {} of {} hard locations activated, {} holding bit-counters): {}",
            name,
            cue_name,
            100.0 * damage,
            if pulls { "pulls" } else { "addresses" },
            out.rounds,
            out.awake,
            store.m,
            out.nonempty,
            verdict
        );
        Ok(Answer {
            lines: vec![line],
            word: out.z.iter().map(|&b| b as f64).collect(),
            scores: vec![(cue_name.clone(), o)],
            recalled: (o >= 0.95 && written).then_some(cue_name),
            text: None,
            steps: out.rounds,
            activated: Some(out.awake),
        })
    }
}

// ------------------------------------------------------------------------------------------------ refusal

// statement: refusal
/// The travel rule's threshold for a memory of `load` random words, and the nearest-neighbour ceiling it leaves.
/// No memory is built.
///
/// Measured limit (runs/sdmrefuse, REPORT_SDMREFUSE.md): in a crowded store a never-stored read-address stops
/// moving, and the fixed threshold refuses only 0.044 of them (top-k read, 10^6 hard-locations, 10,000 words).
/// Calibrating on the memory's own random probes refuses at least 0.98 in 18 of 18 cells (runs/sdmtrack,
/// REPORT_SDMTRACK.md); that rule is an engine call (`crate::calibrated`), and the word list has no keyword
/// for it yet.
///
/// ```
/// use kanerva::rails::Refusal;
/// let r = Refusal::build().word_size(256).load(300).level(0.01).answer().unwrap();
/// assert!(r.to_string().contains("within 95 bits of the cue"));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Refusal {
    word_size: usize,
    load: usize,
    level: f64,
}

// words: refusal
impl Refusal {
    /// Bits per stored word.
    pub fn word_size(mut self, word_size: usize) -> Refusal {
        self.word_size = word_size;
        self
    }

    /// How many random words are stored.
    pub fn load(mut self, load: usize) -> Refusal {
        self.load = load;
        self
    }

    /// The most a never-stored read-address may be accepted.
    pub fn level(mut self, level: f64) -> Refusal {
        self.level = level;
        self
    }
}

impl Refusal {
    /// The calculation with every setting at its default.
    pub fn build() -> Refusal {
        Refusal {
            word_size: num("refusal", "refusal", "word-size") as usize,
            load: num("refusal", "refusal", "load") as usize,
            level: num("refusal", "refusal", "level"),
        }
    }

    /// The travel threshold alone, in bits.
    pub fn threshold(&self) -> Result<usize, Error> {
        let (n, load, level) = (self.word_size, self.load, self.level);
        if !(16..=4096).contains(&n) {
            return fail("refusal size must be between 16 and 4096");
        }
        if load < 1 || !(level > 0.0 && level < 1.0) {
            return fail("refusal needs load: at least 1 and level: between 0 and 1");
        }
        Ok(travel_threshold(n, load, level))
    }

    /// Make the calculation.
    pub fn answer(&self) -> Result<Answer, Error> {
        let h = self.threshold()?;
        let (n, load) = (self.word_size, self.load);
        let recs: Vec<String> = [0.1, 0.2, 0.3, 0.4]
            .iter()
            .map(|&d| {
                let (rc, all, _) = oracle_point(n, d, load, h);
                format!("{:.0}%: {:.3} of {:.3}", 100.0 * d, rc, all)
            })
            .collect();
        Ok(Answer::calculation(format!(
            "refusal for {} patterns of {} bits: accept an answer only if it lies within {} bits of the cue (a never-stored cue is refused with probability {:.4}); the nearest-neighbour ceiling then recalls {}",
            load,
            n,
            h,
            refusal_prob(n, load, h),
            recs.join(", ")
        )))
    }
}

// ------------------------------------------------------------------------------------------------ contenttrack

// statement: contenttrack
/// TRACK-C's predicted recall for the content-woken reads of a memory. No memory is built.
///
/// ```
/// use kanerva::rails::ContentTrack;
/// let a = ContentTrack::build().word_size(64).hard_locations(500).load(20).samples(10).answer().unwrap();
/// assert!(a.to_string().starts_with("contenttrack top-k read, 500 hard locations of 64 bits"));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ContentTrack {
    word_size: usize,
    hard_locations: usize,
    load: usize,
    address_noise: f64,
    block: bool,
    samples: usize,
}

// words: contenttrack
impl ContentTrack {
    /// Bits per word (64 to 1024).
    pub fn word_size(mut self, word_size: usize) -> ContentTrack {
        self.word_size = word_size;
        self
    }

    /// How many hard-locations (at least 100).
    pub fn hard_locations(mut self, hard_locations: usize) -> ContentTrack {
        self.hard_locations = hard_locations;
        self
    }

    /// How many words are stored.
    pub fn load(mut self, load: usize) -> ContentTrack {
        self.load = load;
        self
    }

    /// The read-address's share of flipped bits.
    pub fn address_noise(mut self, address_noise: f64) -> ContentTrack {
        self.address_noise = address_noise;
        self
    }

    /// The block read (true) or the top-k content read (false).
    pub fn block(mut self, block: bool) -> ContentTrack {
        self.block = block;
        self
    }

    /// Sampled reads behind the prediction.
    pub fn samples(mut self, samples: usize) -> ContentTrack {
        self.samples = samples;
        self
    }
}

impl ContentTrack {
    /// The calculation with every setting at its default.
    pub fn build() -> ContentTrack {
        ContentTrack {
            word_size: num("contenttrack", "contenttrack", "word-size") as usize,
            hard_locations: num("contenttrack", "contenttrack", "hard-locations") as usize,
            load: num("contenttrack", "contenttrack", "load") as usize,
            address_noise: num("contenttrack", "contenttrack", "address-noise"),
            block: num("contenttrack", "contenttrack", "block") != 0.0,
            samples: num("contenttrack", "contenttrack", "samples") as usize,
        }
    }

    /// Make the calculation.
    pub fn answer(&self) -> Result<Answer, Error> {
        let (n, m, load, dmg, block, smp) = (self.word_size, self.hard_locations, self.load, self.address_noise, self.block, self.samples);
        if !(64..=1024).contains(&n) || m < 100 || load < 1 || !(0.0..0.5).contains(&dmg) || smp < 1 {
            return fail("contenttrack needs word-size 64..1024, hard-locations >= 100, load >= 1, address-noise in [0, 0.5), samples >= 1");
        }
        let r = radius_for(n, (m as f64 * m as f64 / 10.0).powf(-1.0 / 3.0));
        let c = Content::new(n, m, r);
        let wake = if block { TrackWake::Block(block_theta(n, m, r, load, 0.1, 0.01)) } else { TrackWake::Topk(c.k()) };
        let f = c.p_converge(wake, dmg, load, smp, false, 1);
        let p = c.p_converge(wake, dmg, load, smp, true, 1);
        Ok(Answer::calculation(format!(
            "contenttrack {} read, {} hard locations of {} bits, activation-radius {}, {} patterns, {:.0}% address-noise: TRACK-C predicts recall {:.3} (fresh) {:.3} (persist) over {} sampled reads",
            if block { "block" } else { "top-k" },
            m,
            n,
            r,
            load,
            100.0 * dmg,
            f,
            p,
            smp
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_setting_out_of_range_is_refused_in_the_file_faces_words() {
        let e = Sdm::build().word_size(8).write("cat").unwrap_err();
        assert_eq!(e.0, "sdm word-size must be between 16 and 4096");
        let e = Sdm::build().word_size(64).activation_radius(65).write("cat").unwrap_err();
        assert_eq!(e.0, "activation-radius cannot be larger than word-size");
        let e = SoftSdm::build().softness(-1.0).write("cat").unwrap_err();
        assert_eq!(e.0, "softness must be at least 0 and gain above 0");
    }

    #[test]
    #[should_panic(expected = "set in the builder chain")]
    fn a_setting_after_the_first_write_panics() {
        let mut s = Sdm::build().word_size(64).hard_locations(100);
        s.write("cat").unwrap();
        let _ = s.clone().seed(2);
    }

    #[test]
    fn a_read_takes_a_read_address_or_a_key_not_both() {
        let mut s = Sdm::build().word_size(64).hard_locations(100);
        s.write("cat").unwrap();
        assert_eq!(s.read("cat").key("k").answer().unwrap_err().0, "read takes read-address: or key:, not both");
    }

    #[test]
    fn writing_a_name_twice_is_refused() {
        let mut s = Sdm::build().word_size(64).hard_locations(100);
        s.write("cat").unwrap();
        assert_eq!(s.write("cat").unwrap_err().0, ":cat is already written in :s");
        let mut k = SdmScale::build().name("k").hard_locations(1000);
        k.fill(3).unwrap();
        assert_eq!(k.fill(3).unwrap_err().0, ":k is already filled; one fill per store");
    }

    #[test]
    fn a_seed_restarts_the_stream_and_rewind_starts_a_new_run() {
        let mut s = Sdm::build().word_size(256).hard_locations(2000);
        for w in ["cat", "dog", "owl"] {
            s.write(w).unwrap();
        }
        let a = s.read("cat").address_noise(0.3).answer().unwrap();
        let b = s.read("cat").address_noise(0.3).answer().unwrap();
        s.rewind();
        let c = s.read("cat").address_noise(0.3).answer().unwrap();
        assert_eq!(a.to_string(), c.to_string(), "rewind starts the stream where a run block starts it");
        assert_ne!(a.lines, b.lines, "control: without a rewind the second read draws different noise");
        let d = s.read("cat").address_noise(0.3).seed(7).answer().unwrap();
        let e = s.read("cat").address_noise(0.3).seed(7).answer().unwrap();
        assert_eq!(d, e);
    }

    #[test]
    fn the_choices_spell_the_word_lists_symbols() {
        use crate::words::Value;
        let check = |f: &str, v: &str, k: &str, words: Vec<&str>| match statement(f, v).unwrap().word(k).unwrap().value {
            Value::Symbol(cs) => assert_eq!(cs.to_vec(), words, "{} {} {}", f, v, k),
            _ => panic!("{} is not a choice", k),
        };
        check("sdm", "read", "via", Via::ALL.iter().map(|x| x.word()).collect());
        check("sdmscale", "read", "via", Via::ALL.iter().map(|x| x.word()).collect());
        check("sdmscale", "read", "wake", Wake::ALL.iter().map(|x| x.word()).collect());
        check("softsdm", "read", "mode", Mode::ALL.iter().map(|x| x.word()).collect());
    }

    #[test]
    fn the_defaults_are_the_word_lists() {
        let mut s = Sdm::build();
        let shape = s.shape().unwrap();
        assert_eq!((shape.word_size, shape.hard_locations, shape.activation_radius), (256, 2000, 112));
        assert_eq!(num("softsdm", "softsdm", "write-samples"), 16.0);
        assert_eq!(sym("sdmscale", "read", "wake"), "fixed");
    }
}
