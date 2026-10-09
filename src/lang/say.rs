//! The lines the sdm family prints. KANERVA's runner and SETTLE both print through these functions, so a read
//! that comes out the same says the same, byte for byte.
//!
//! <claudes_code_comments>
//! ** Function List **
//! public(stored, name, n)    - the public pattern of a stored name (None for a keyed note)
//! From::describe()           - "read-address :cat with 20% address-noise", "a key", "pure noise"
//! sdm_write_warning(...)     - the line for a write no hard-location took
//! sdm_read(...)              - the line(s) after a hard read: scoreboard, verdict, text or keyed note
//! soft_scores / soft_verdict - the scoreboard and verdict of a soft read
//! soft_read(...)             - the line(s) after a soft read
//! soft_attend_head/row       - the lines of an attend
//! scale_read(...)            - the line after an sdmscale read
//! refusal(...)               - the refusal statement's line, computed
//! contenttrack(...)          - the contenttrack statement's line, computed
//!
//! ** Technical Review **
//! - A stored name carries a tag: "" for a named code, "=text" for text masked by its name, "#" for a keyed note
//!   (whose text is held nowhere). Only public patterns join a scoreboard.
//! - A hard read's verdict is the best overlap at 0.9 or above; text comes back by unmasking with the name's code.
//! - The soft scoreboard sorts by overlap (signed), the hard one by its absolute value: as SETTLE always has.
//! - refusal and contenttrack compute their numbers here from `crate::refuse` and `crate::track`: they build no
//!   memory, so there is nothing for a host to hold.
//!
//! </claudes_code_comments>

use super::parse::{From, Via, Wake};
use crate::codes::{bits_text, code, overlap, pattern};
use crate::keys::keyed_read;

/// The public pattern of a stored name: its code, or its masked text; None for a keyed note or an unknown name.
pub fn public(stored: &[(String, String)], name: &str, n: usize) -> Option<Vec<f64>> {
    let tag = &stored.iter().find(|(x, _)| x == name)?.1;
    match tag.chars().next() {
        Some('#') => None,
        Some('=') => Some(pattern(name, Some(&tag[1..]), n)),
        _ => Some(code(name, n)),
    }
}

impl From {
    /// How the read is described in its line.
    pub fn describe(&self) -> String {
        match self {
            From::Address { name, noise } => format!("read-address :{} with {:.0}% address-noise", name, 100.0 * noise),
            From::Key { .. } => "a key".to_string(),
            From::Noise => "pure noise".to_string(),
        }
    }
}

impl Via {
    /// `addresses` or `pulls`.
    pub fn word(self) -> &'static str {
        match self {
            Via::Addresses => "addresses",
            Via::Pulls => "pulls",
        }
    }
}

impl Wake {
    /// `fixed`, `density` or `top`.
    pub fn word(self) -> &'static str {
        match self {
            Wake::Fixed => "fixed",
            Wake::Density => "density",
            Wake::Top => "top",
        }
    }
}

/// The line printed when a write reached no hard-location.
pub fn sdm_write_warning(name: &str, radius: usize, what: &str) -> String {
    format!("warning: no hard location of :{} is within activation-radius {} of :{}, so nothing was written", name, radius, what)
}

/// The line(s) after a hard read of sdm `name`: the scoreboard of the three best stored names and the verdict,
/// then the text when the answer is a stored text; for a keyed read, the note or "nothing readable".
#[allow(clippy::too_many_arguments)]
pub fn sdm_read(
    name: &str,
    from: &From,
    via: Via,
    rounds: usize,
    awake: usize,
    hard_locations: usize,
    got: &[f64],
    stored: &[(String, String)],
    n: usize,
) -> Vec<String> {
    let head = format!(
        "read :{} from {} via {} ({} iterated reads, {} of {} hard locations activated)",
        name,
        from.describe(),
        via.word(),
        rounds,
        awake,
        hard_locations
    );
    if let From::Key { key, .. } = from {
        return vec![match keyed_read(key, got) {
            Some(t) => format!("{}: text \"{}\"", head, t),
            None => format!("{}: nothing readable", head),
        }];
    }
    let mut out = Vec::new();
    let mut scores: Vec<(String, f64)> = stored.iter().filter_map(|(x, _)| public(stored, x, n).map(|p| (x.clone(), overlap(got, &p)))).collect();
    scores.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());
    let top: Vec<String> = scores.iter().take(3).map(|(x, o)| format!(":{} {:+.2}", x, o)).collect();
    let verdict = match scores.first() {
        Some((x, o)) if *o >= 0.9 => format!("-> :{}", x),
        Some((x, o)) => format!("-> nothing clear (closest :{} at {:+.2})", x, o),
        None => "-> nothing is written".to_string(),
    };
    out.push(format!("{}: {}  {}", head, top.join("  "), verdict));
    if let Some((x, o)) = scores.first() {
        if *o >= 0.9 {
            if let Some((_, tag)) = stored.iter().find(|(y, _)| y == x) {
                if let Some(t) = tag.strip_prefix('=') {
                    let mask = code(x, n);
                    let bits: Vec<f64> = got.iter().zip(&mask).map(|(a, b)| a * b).collect();
                    out.push(format!("  text: \"{}\"", bits_text(&bits, t.len())));
                }
            }
        }
    }
    out
}

/// A soft memory's scoreboard: each stored name (with its text, if any) against `got`, best overlap first.
pub fn soft_scores(got: &[f64], stored: &[(String, Option<String>)], n: usize) -> Vec<(String, f64)> {
    let mut v: Vec<(String, f64)> = stored.iter().map(|(x, t)| (x.clone(), overlap(got, &pattern(x, t.as_deref(), n)))).collect();
    v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    v
}

/// A soft memory's verdict on a scoreboard.
pub fn soft_verdict(sc: &[(String, f64)]) -> String {
    match sc.first() {
        Some((x, o)) if *o >= 0.9 => format!("-> :{}", x),
        Some((x, o)) => format!("-> nothing clear (closest :{} at {:+.2})", x, o),
        None => "-> nothing is written".to_string(),
    }
}

/// The line(s) after a soft read: the read, its best overlap by round, the scoreboard and the verdict, then the
/// text when the answer is a stored text.
#[allow(clippy::too_many_arguments)]
pub fn soft_read(
    name: &str,
    from: &str,
    settle: bool,
    softness: f64,
    rounds: usize,
    trail: &[String],
    got: &[f64],
    stored: &[(String, Option<String>)],
    n: usize,
) -> Vec<String> {
    let sc = soft_scores(got, stored, n);
    let top: Vec<String> = sc.iter().take(3).map(|(x, o)| format!(":{} {:+.2}", x, o)).collect();
    let mut out = vec![format!(
        "read :{} from {} ({}, softness {}, {} rounds, best overlap by round {}): {}  {}",
        name,
        from,
        if settle { "settle" } else { "pass" },
        softness,
        rounds,
        trail.join(" "),
        top.join("  "),
        soft_verdict(&sc)
    )];
    if let Some((x, o)) = sc.first() {
        if *o >= 0.9 {
            if let Some((_, Some(t))) = stored.iter().find(|(y, _)| y == x) {
                let mask = code(x, n);
                let bits: Vec<f64> = got.iter().zip(&mask).map(|(v, k)| v * k).collect();
                out.push(format!("  text: \"{}\"", bits_text(&bits, t.len())));
            }
        }
    }
    out
}

/// The first line of an attend.
pub fn soft_attend_head(name: &str, from: &str, softness: f64, beta: f64) -> String {
    format!("attend :{} from {} (softness {}, fitted softmax inverse temperature {:.1} on cosine):", name, from, softness, beta)
}

/// One row of an attend: its label, the two best names and the verdict.
pub fn soft_attend_row(label: &str, sc: &[(String, f64)]) -> String {
    let top: Vec<String> = sc.iter().take(2).map(|(x, o)| format!(":{} {:+.2}", x, o)).collect();
    format!("  {:<40} {}  {}", label, top.join("  "), soft_verdict(sc))
}

/// The line after an sdmscale read.
#[allow(clippy::too_many_arguments)]
pub fn scale_read(
    name: &str,
    read_address: &str,
    address_noise: f64,
    via: Via,
    rounds: usize,
    awake: usize,
    hard_locations: usize,
    nonempty: usize,
    o: f64,
    written: bool,
) -> String {
    let verdict = if o >= 0.95 && written {
        format!("-> :{}", read_address)
    } else if o >= 0.95 {
        format!("-> back to :{} though it was never written (the read did not move it)", read_address)
    } else {
        format!("-> nothing clear (overlap with :{} {:+.2})", read_address, o)
    };
    format!(
        "read :{} from read-address :{} with {:.0}% address-noise via {} ({} iterated reads, {} of {} hard locations activated, {} holding bit-counters): {}",
        name,
        read_address,
        100.0 * address_noise,
        via.word(),
        rounds,
        awake,
        hard_locations,
        nonempty,
        verdict
    )
}

/// The refusal statement's line: the travel rule's threshold for `load` patterns of `word_size` bits at
/// `level`, and the nearest-neighbour ceiling at 10% to 40% address-noise.
pub fn refusal(word_size: usize, load: usize, level: f64) -> String {
    use crate::refuse::{oracle_point, refusal_prob, travel_threshold};
    let h = travel_threshold(word_size, load, level);
    let recs: Vec<String> = [0.1, 0.2, 0.3, 0.4]
        .iter()
        .map(|&d| {
            let (rc, all, _) = oracle_point(word_size, d, load, h);
            format!("{:.0}%: {:.3} of {:.3}", 100.0 * d, rc, all)
        })
        .collect();
    format!(
        "refusal for {} patterns of {} bits: accept an answer only if it lies within {} bits of the cue (a never-stored cue is refused with probability {:.4}); the nearest-neighbour ceiling then recalls {}",
        load,
        word_size,
        h,
        refusal_prob(word_size, load, h),
        recs.join(", ")
    )
}

/// The contenttrack statement's line: TRACK-C's predicted recall of the content read, fresh and persistent.
pub fn contenttrack(word_size: usize, hard_locations: usize, load: usize, address_noise: f64, block: bool, samples: usize) -> String {
    use crate::track::{block_theta, Content, Wake as TrackWake};
    let (n, m) = (word_size, hard_locations);
    let r = crate::theory::radius_for(n, (m as f64 * m as f64 / 10.0).powf(-1.0 / 3.0));
    let c = Content::new(n, m, r);
    let wake = if block { TrackWake::Block(block_theta(n, m, r, load, 0.1, 0.01)) } else { TrackWake::Topk(c.k()) };
    let f = c.p_converge(wake, address_noise, load, samples, false, 1);
    let p = c.p_converge(wake, address_noise, load, samples, true, 1);
    format!(
        "contenttrack {} read, {} hard locations of {} bits, activation-radius {}, {} patterns, {:.0}% address-noise: TRACK-C predicts recall {:.3} (fresh) {:.3} (persist) over {} sampled reads",
        if block { "block" } else { "top-k" },
        m,
        n,
        r,
        load,
        100.0 * address_noise,
        f,
        p,
        samples
    )
}
