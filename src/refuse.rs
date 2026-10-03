//! Refusal: when should a sparse distributed memory say "I never stored that"?
//!
//! - `travel_threshold(n, T, level)`: the largest travel h (Hamming distance from read-address to answer) such that a
//!   random read-address has one of T random stored patterns within h with probability at most `level`. A memory
//!   knows how many patterns it holds, so this rule is visible to it. Accept an answer iff travel <= h.
//! - `oracle_point`: the exact recall and refusal of the nearest-neighbour oracle (it sees every stored
//!   pattern, answers the nearest, refuses beyond h). No read can beat it at the same refusal rate.
//! - `Fast`: a store's reads returning diagnostics (`Diag`): the answer, the rounds, the first-round and
//!   last-round votes' agreement with the answer, and the woken rows' mean match. Same answers as the
//!   `store::Store` reads (the content reads keep each row's match current by adding only the flipped bits).
//! - `Track`: predictor TRACK for the ADDRESS read. It keeps the target and every other stored pattern, and at
//!   each read draws the number of hard locations the state shares with each one from its current distance
//!   (Poisson with mean M I(d)); the vote is then exactly sum_mu k_mu x_mu. FRESH redraws the counts every
//!   read; PERSIST keeps the fraction of the old wake set still inside the new one and draws the arrivals.

use crate::bits::hd;
use crate::rng::Rng;
use crate::store::Store;
use crate::theory::{ball, binom_pmf, intersection};

/// P[a random pattern lies within h of a random read-address] = P[Bin(n, 1/2) <= h].
pub fn half_cdf(n: usize, h: usize) -> f64 {
    ball(n, h)
}

/// P[no one of T random stored patterns lies within h of a random read-address].
pub fn refusal_prob(n: usize, t: usize, h: usize) -> f64 {
    let f = half_cdf(n, h).min(1.0);
    (t as f64 * (1.0 - f).ln_1p_safe()).exp()
}

trait Ln1p {
    fn ln_1p_safe(self) -> f64;
}
impl Ln1p for f64 {
    /// ln(x) for x in (0, 1], accurate when x is close to 1 (x = 1 - f with f tiny).
    fn ln_1p_safe(self) -> f64 {
        if self <= 0.0 {
            f64::NEG_INFINITY
        } else {
            (self - 1.0).ln_1p()
        }
    }
}

/// The travel rule's threshold: the largest h with P[some one of T random patterns within h] <= level.
pub fn travel_threshold(n: usize, t: usize, level: f64) -> usize {
    let mut best = 0;
    for h in 0..=n {
        if 1.0 - refusal_prob(n, t, h) <= level {
            best = h;
        } else {
            break;
        }
    }
    best
}

/// The nearest-neighbour oracle, exactly: T stored random patterns, a stored read-address made by flipping each bit
/// of its pattern with probability `dmg`. The oracle answers the nearest stored pattern (ties broken
/// uniformly) and refuses when that distance exceeds h. Returns (recall, recall with no refusal, refusal):
/// recall = P[target nearest and within h], refusal = P[a random read-address has no stored pattern within h].
pub fn oracle_point(n: usize, dmg: f64, t: usize, h: usize) -> (f64, f64, f64) {
    let half = binom_pmf(n, 0.5);
    let b = binom_pmf(n, dmg);
    let mut cdf = 0.0;
    let (mut rec, mut all) = (0.0, 0.0);
    for d in 0..=n {
        let g = half[d];
        let gt = (1.0 - cdf - g).max(0.0); // P[rival > d]
        // P[target wins]: sum over j rivals tied at d of C(T-1, j) g^j gt^(T-1-j) / (j + 1)
        //   = ((gt + g)^T - gt^T) / (T g); the small-g limit is gt^(T-1)
        let tt = t.max(1) as f64;
        let win = if tt * g < 1e-9 {
            gt.powf(tt - 1.0)
        } else {
            ((gt + g).powf(tt) - gt.powf(tt)) / (tt * g)
        };
        all += b[d] * win;
        if d <= h {
            rec += b[d] * win;
        }
        cdf += g;
    }
    (rec.min(1.0), all.min(1.0), refusal_prob(n, t, h))
}

/// What one read did, with the signals a refusal rule may use.
#[derive(Clone, Debug, Default)]
pub struct Diag {
    pub z: Vec<i8>,
    pub rounds: usize,
    pub fixed: bool,
    /// cos(first-round vote, final state): how much of the first vote already pointed at the answer.
    pub cos1: f64,
    /// cos(last-round vote, final state): how cleanly the last vote holds the answer.
    pub cosf: f64,
    /// Content reads: mean C_i . z over the woken rows / n, first and last round. Address read: filled rows
    /// awake on the first and last round.
    pub dot1: f64,
    pub dotf: f64,
}

fn cosine(s: &[i32], z: &[i8]) -> f64 {
    let dot: f64 = s.iter().zip(z).map(|(&a, &b)| a as f64 * b as f64).sum();
    let nn: f64 = s.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt();
    if nn == 0.0 {
        0.0
    } else {
        dot / (nn * (z.len() as f64).sqrt())
    }
}

/// A store with its filled rows listed once, and diagnostic reads.
pub struct Fast<'a> {
    pub st: &'a Store,
    pub rows: Vec<u32>,
}

impl<'a> Fast<'a> {
    pub fn new(st: &'a Store) -> Fast<'a> {
        let rows = (0..st.m).filter(|&i| st.filled[i]).map(|i| i as u32).collect();
        Fast { st, rows }
    }

    /// The top-k pulls read; same answer as `Store::read_pulls_topk`.
    pub fn topk(&self, cue: &[i8], iters: usize, k: usize) -> Diag {
        self.pulls(cue, iters, |rows, dots| {
            let mut v: Vec<(u32, i32)> = rows.iter().copied().zip(dots.iter().copied()).collect();
            let k = k.min(v.len());
            if k == 0 {
                return Vec::new();
            }
            v.select_nth_unstable_by(k - 1, |a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v[..k].iter().map(|x| x.0).collect()
        })
    }

    /// The thresholded pulls read; same answer as `Store::read_pulls_at`.
    pub fn thresh(&self, cue: &[i8], iters: usize, theta: f64) -> Diag {
        self.pulls(cue, iters, |rows, dots| rows.iter().zip(dots).filter(|x| *x.1 as f64 > theta).map(|x| *x.0).collect())
    }

    /// Shared loop: the dots C_i . z of every filled row are kept current by adding only the flipped bits'
    /// terms after each read.
    fn pulls<F: Fn(&[u32], &[i32]) -> Vec<u32>>(&self, cue: &[i8], iters: usize, pick: F) -> Diag {
        let n = self.st.n;
        let mut z = cue.to_vec();
        let mut dots: Vec<i32> = self
            .rows
            .iter()
            .map(|&i| self.st.row(i as usize).iter().zip(&z).map(|(&c, &v)| c as i32 * v as i32).sum())
            .collect();
        let mut sum = vec![0i32; n];
        let mut out = Diag::default();
        let mut s1: Vec<i32> = Vec::new();
        for t in 0..iters.max(1) {
            let awake = pick(&self.rows, &dots);
            sum.iter_mut().for_each(|s| *s = 0);
            let mut dsum = 0.0f64;
            for &i in &awake {
                for (s, &c) in sum.iter_mut().zip(self.st.row(i as usize)) {
                    *s += c as i32;
                }
            }
            // mean dot of the woken rows: recompute from the rows (cheap, woken rows are few)
            for &i in &awake {
                dsum += self.st.row(i as usize).iter().zip(&z).map(|(&c, &v)| c as i32 * v as i32).sum::<i32>() as f64;
            }
            let md = if awake.is_empty() { 0.0 } else { dsum / (awake.len() as f64 * n as f64) };
            if t == 0 {
                s1 = sum.clone();
                out.dot1 = md;
            }
            out.dotf = md;
            let next: Vec<i8> = sum.iter().zip(&z).map(|(&s, &old)| if s > 0 { 1 } else if s < 0 { -1 } else { old }).collect();
            out.rounds = t + 1;
            if next == z {
                out.fixed = true;
                out.cosf = cosine(&sum, &z);
                break;
            }
            let flips: Vec<(usize, i32)> = (0..n).filter(|&j| next[j] != z[j]).map(|j| (j, 2 * next[j] as i32)).collect();
            for (d, &i) in dots.iter_mut().zip(&self.rows) {
                let row = self.st.row(i as usize);
                let mut add = 0i32;
                for &(j, w) in &flips {
                    add += row[j] as i32 * w;
                }
                *d += add;
            }
            out.cosf = cosine(&sum, &next);
            z = next;
        }
        out.cos1 = cosine(&s1, &z);
        out.z = z;
        out
    }

    /// Kanerva's address read; same answer as `Store::read_addresses`.
    pub fn address(&self, cue: &[i8], iters: usize) -> Diag {
        let st = self.st;
        let n = st.n;
        let mut z = cue.to_vec();
        let mut sum = vec![0i32; n];
        let mut out = Diag::default();
        let mut s1: Vec<i32> = Vec::new();
        for t in 0..iters.max(1) {
            let act = st.awake(&z);
            sum.iter_mut().for_each(|s| *s = 0);
            let mut ne = 0;
            for &i in &act {
                let i = i as usize;
                if !st.filled[i] {
                    continue;
                }
                ne += 1;
                for (s, &c) in sum.iter_mut().zip(st.row(i)) {
                    *s += c as i32;
                }
            }
            if t == 0 {
                s1 = sum.clone();
                out.dot1 = ne as f64;
            }
            out.dotf = ne as f64;
            let next: Vec<i8> = sum.iter().zip(&z).map(|(&s, &old)| if s > 0 { 1 } else if s < 0 { -1 } else { old }).collect();
            out.rounds = t + 1;
            out.cosf = cosine(&sum, &next);
            if next == z {
                out.fixed = true;
                break;
            }
            z = next;
        }
        out.cos1 = cosine(&s1, &z);
        out.z = z;
        out
    }
}

/// A Poisson draw (Knuth below mean 30, a rounded normal above).
pub fn poisson(lam: f64, r: &mut Rng) -> u32 {
    if lam <= 0.0 {
        return 0;
    }
    if lam > 30.0 {
        return (lam + lam.sqrt() * r.normal()).round().max(0.0) as u32;
    }
    let l = (-lam).exp();
    let (mut k, mut p) = (0u32, 1.0);
    loop {
        p *= r.unit();
        if p <= l {
            return k;
        }
        k += 1;
    }
}

/// Poisson with e^-lam supplied (the tables below), so the common k = 0 case costs one uniform.
fn poisson_t(lam: f64, elam: f64, r: &mut Rng) -> u32 {
    if lam <= 0.0 {
        return 0;
    }
    if lam > 30.0 {
        return poisson(lam, r);
    }
    let (mut k, mut p) = (0u32, 1.0);
    loop {
        p *= r.unit();
        if p <= elam {
            return k;
        }
        k += 1;
    }
}

/// A binomial draw by counting (k is small here).
fn binomial(k: u32, q: f64, r: &mut Rng) -> u32 {
    if q >= 1.0 {
        return k;
    }
    if k > 200 {
        let m = k as f64 * q;
        return (m + (m * (1.0 - q)).sqrt() * r.normal()).round().clamp(0.0, k as f64) as u32;
    }
    (0..k).filter(|_| r.unit() < q).count() as u32
}

/// Predictor TRACK's tables: lam[d] = M I(d), the expected number of hard locations a state d bits from a
/// pattern shares with that pattern's write set; keep[d] = I(d) / p, the fraction of a wake set that is
/// still awake after the state moves d bits (PERSIST's retention, ignoring where the pattern sits).
pub struct Track {
    pub n: usize,
    pub m: usize,
    pub r: usize,
    pub lam: Vec<f64>,
    pub elam: Vec<f64>,
    pub keep: Vec<f64>,
}

impl Track {
    pub fn new(n: usize, m: usize, r: usize) -> Track {
        let p = ball(n, r);
        let inter: Vec<f64> = (0..=n).map(|d| intersection(n, r, d)).collect();
        let lam = inter.iter().map(|&i| i * m as f64).collect();
        let keep = inter.iter().map(|&i| (i / p).min(1.0)).collect();
        let elam = inter.iter().map(|&i| (-i * m as f64).exp()).collect();
        Track { n, m, r, lam, elam, keep }
    }

    /// One sampled read: a random target and T - 1 random other patterns, the read-address flips each target bit
    /// with probability dmg; up to `iters` reads; returns the final overlap with the target and the rounds.
    /// At each read the vote on bit j is sum_mu k_mu x_mu[j] (exactly what the store's summed awake rows
    /// hold), k_mu drawn from the state's current distance to pattern mu.
    pub fn sample(&self, dmg: f64, t: usize, iters: usize, persist: bool, r: &mut Rng) -> (f64, usize) {
        let n = self.n;
        let w = (n + 63) / 64;
        let tail = if n % 64 == 0 { u64::MAX } else { (1u64 << (n % 64)) - 1 };
        let t = t.max(1);
        let mut pats: Vec<u64> = Vec::with_capacity(t * w);
        for _ in 0..t {
            for k in 0..w {
                let x = r.next_u64();
                pats.push(if k + 1 == w { x & tail } else { x });
            }
        }
        // read-address: flip each bit of pattern 0 with probability dmg
        let mut z: Vec<u64> = pats[..w].to_vec();
        for j in 0..n {
            if r.unit() < dmg {
                z[j / 64] ^= 1 << (j % 64);
            }
        }
        let mut kprev: Vec<u32> = vec![0; t];
        let mut dprev: Vec<usize> = vec![0; t];
        let mut field = vec![0i64; n];
        let mut rounds = 0;
        let mut moved = 0usize;
        for it in 0..iters.max(1) {
            field.iter_mut().for_each(|f| *f = 0);
            for mu in 0..t {
                let x = &pats[mu * w..(mu + 1) * w];
                let d = hd(x, &z);
                let k = if persist && it > 0 {
                    let q = self.keep[moved];
                    let kept = binomial(kprev[mu], q, r);
                    let arr = (self.lam[d] - q * self.lam[dprev[mu]]).max(0.0);
                    kept + poisson(arr, r)
                } else {
                    poisson_t(self.lam[d], self.elam[d], r)
                };
                kprev[mu] = k;
                dprev[mu] = d;
                if k == 0 {
                    continue;
                }
                let k = k as i64;
                for j in 0..n {
                    if (x[j / 64] >> (j % 64)) & 1 == 1 {
                        field[j] += k;
                    } else {
                        field[j] -= k;
                    }
                }
            }
            let mut next = z.clone();
            for j in 0..n {
                if field[j] > 0 {
                    next[j / 64] |= 1 << (j % 64);
                } else if field[j] < 0 {
                    next[j / 64] &= !(1 << (j % 64));
                }
            }
            rounds = it + 1;
            moved = hd(&next, &z);
            if moved == 0 {
                break;
            }
            z = next;
        }
        let d = hd(&pats[..w], &z);
        (1.0 - 2.0 * d as f64 / n as f64, rounds)
    }

    /// Fraction of `samples` sampled reads ending at overlap >= 0.95 (common random numbers across T).
    pub fn p_converge(&self, dmg: f64, t: usize, samples: usize, persist: bool, seed: u64) -> f64 {
        let mut r = Rng::new(seed ^ 0x7EAC_0000_0001);
        let ok = (0..samples).filter(|_| self.sample(dmg, t, 20, persist, &mut r).0 >= 0.95).count();
        ok as f64 / samples.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{add_address_noise, overlap, pack, random_pattern};

    #[test]
    fn refusal_with_one_pattern_by_hand() {
        // n 4, T 1, h 1: a random read-address has the one pattern within 1 bit with probability (1 + 4) / 16
        assert!((refusal_prob(4, 1, 1) - 11.0 / 16.0).abs() < 1e-12);
        // T 2: both patterns must be farther: (11/16)^2
        assert!((refusal_prob(4, 2, 1) - (11.0f64 / 16.0).powi(2)).abs() < 1e-12);
        // level 0.32 admits h = 1 (P[some pattern within 1] = 5/16) but not h = 2 (11/16); 0.3 admits only h = 0
        assert_eq!(travel_threshold(4, 1, 0.32), 1);
        assert_eq!(travel_threshold(4, 1, 0.3), 0);
    }

    #[test]
    fn the_oracle_always_names_a_lone_pattern() {
        let (_, all, _) = oracle_point(64, 0.4, 1, 64);
        assert!((all - 1.0).abs() < 1e-9);
        // with h 0 it accepts only an undamaged read-address
        let (rec, _, _) = oracle_point(64, 0.1, 1, 0);
        assert!((rec - 0.9f64.powi(64)).abs() < 1e-9);
    }

    #[test]
    fn diagnostic_reads_answer_as_the_store_does() {
        let mut rr = Rng::new(3);
        let ps: Vec<Vec<i8>> = (0..300).map(|_| random_pattern(256, &mut rr)).collect();
        let mut st = Store::new(256, 20_000, 105, 3);
        st.write_many(&ps);
        let f = Fast::new(&st);
        let k = (ball(256, 105) * 20_000.0).round() as usize;
        for q in 0..6 {
            let cue = if q % 3 == 2 { random_pattern(256, &mut rr) } else { add_address_noise(&ps[q], 0.2, &mut rr) };
            assert_eq!(f.topk(&cue, 20, k).z, st.read_pulls_topk(&cue, 20, k).z);
            assert_eq!(f.thresh(&cue, 20, 60.0).z, st.read_pulls_at(&cue, 20, 60.0).z);
            assert_eq!(f.address(&cue, 20).z, st.read_addresses(&cue, 20).z);
        }
        // a clean recall travels about the address-noise; the answer is the pattern
        let cue = add_address_noise(&ps[0], 0.2, &mut rr);
        let d = f.topk(&cue, 20, k);
        assert!(overlap(&d.z, &ps[0]) > 0.99);
        assert!(hd(&pack(&cue), &pack(&d.z)) < 80);
    }

    #[test]
    fn track_recalls_a_lone_pattern_and_fails_with_nothing_shared() {
        assert!(Track::new(256, 100_000, 106).p_converge(0.1, 1, 200, false, 1) > 0.99);
        // negative control: a activation radius so small that a 40% read-address shares no hard location
        assert!(Track::new(256, 1000, 90).p_converge(0.4, 1, 200, false, 1) < 0.01);
    }
}
