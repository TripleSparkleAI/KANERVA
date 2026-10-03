//! Predictors for the content-woken reads, and why never-stored read-addresses stop moving.
//!
//! The content reads (top-k: wake the k filled rows whose bit-counters best match the state; block: wake every
//! row whose match exceeds a threshold) choose rows by what they hold, so the number of woken rows holding a
//! pattern depends on the whole stored population. This module gives:
//!
//! - `compound_pmf`: the distribution of a row's match when the row holds a Poisson number of stored
//!   patterns drawn from the current population of overlaps (an FFT of the characteristic function).
//! - `Content` (predictor TRACK-C): keeps the target and every rival pattern; at each read it computes every
//!   pattern's overlap u_mu with the state, the row-match distribution, the top-k cut (or the block
//!   threshold), and each pattern's wake probability q_mu = P[u_mu + rest > cut]; the number of woken rows
//!   holding mu is then Poisson(p M q_mu) (FRESH) or coupled to the previous read's count (PERSIST); the vote
//!   is sum_mu k_mu x_mu. `Content::with_geometry` is TRACK-G, which also places each row's address relative
//!   to the state.
//! - `Member` (reference TRACK-R): an explicit random membership graph (each pattern written to Poisson(p M)
//!   uniformly chosen rows, no addresses), read exactly as the store reads.
//! - `trace_topk` and `census`: the store's own top-k read with every round's woken rows, and which stored
//!   patterns those rows hold.
//! - `TailCal` and `calibrate`: the probe tail fractions and the combined-score refusal rules.

use crate::bits::{hd, pack};
use crate::refuse::poisson;
use crate::rng::Rng;
use crate::store::Store;
use crate::theory::{ball, phi, phi_inv};

/// Grid of the row-match distribution in half units (u = s / 2, s = x . z): values -H .. H-1.
pub const G: usize = 8192;
pub const H: i64 = (G / 2) as i64;

/// In-place iterative radix-2 FFT (length a power of two). `inverse` uses the +i sign and does NOT divide.
pub fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (xr, xi) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
                let t = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = t;
            }
        }
        len <<= 1;
    }
}

/// Forward transform of a severity (weights on half-unit values u, summing to 1).
fn severity_cf(sev: &[(i64, f64)]) -> (Vec<f64>, Vec<f64>) {
    let (mut re, mut im) = (vec![0.0; G], vec![0.0; G]);
    for &(u, w) in sev {
        re[u.rem_euclid(G as i64) as usize] += w;
    }
    fft(&mut re, &mut im, false);
    (re, im)
}

/// pmf over u = -H..H-1 (index u + H) of a compound Poisson sum with rate `lam` and the severity whose
/// transform is (fr, fi).
fn compound_from_cf(fr: &[f64], fi: &[f64], lam: f64) -> Vec<f64> {
    let (mut re, mut im) = (vec![0.0; G], vec![0.0; G]);
    for k in 0..G {
        let a = (lam * (fr[k] - 1.0)).exp();
        let b = lam * fi[k];
        re[k] = a * b.cos();
        im[k] = a * b.sin();
    }
    fft(&mut re, &mut im, true);
    let mut out = vec![0.0; G];
    for u in -H..H {
        out[(u + H) as usize] = (re[u.rem_euclid(G as i64) as usize] / G as f64).max(0.0);
    }
    out
}

/// The distribution of a sum of a Poisson(lam) number of terms, each drawn from `sev` ((u, weight) pairs,
/// weights summing to 1): the match (in half units) of a row holding a Poisson number of stored patterns.
pub fn compound_pmf(sev: &[(i64, f64)], lam: f64) -> Vec<f64> {
    let (fr, fi) = severity_cf(sev);
    compound_from_cf(&fr, &fi, lam)
}

/// `compound_pmf` on a grid of `len` points (a power of two), values -len/2 .. len/2 - 1 (index u + len/2).
pub fn compound_pmf_len(sev: &[(i64, f64)], lam: f64, len: usize) -> Vec<f64> {
    let (mut re, mut im) = (vec![0.0; len], vec![0.0; len]);
    for &(u, w) in sev {
        re[u.rem_euclid(len as i64) as usize] += w;
    }
    fft(&mut re, &mut im, false);
    for k in 0..len {
        let a = (lam * (re[k] - 1.0)).exp();
        let b = lam * im[k];
        re[k] = a * b.cos();
        im[k] = a * b.sin();
    }
    fft(&mut re, &mut im, true);
    let half = (len / 2) as i64;
    let mut out = vec![0.0; len];
    for u in -half..half {
        out[(u + half) as usize] = (re[u.rem_euclid(len as i64) as usize] / len as f64).max(0.0);
    }
    out
}

fn at_len(v: &[f64], u: i64, below: f64, half: i64) -> f64 {
    if u < -half {
        below
    } else if u >= half {
        0.0
    } else {
        v[(u + half) as usize]
    }
}

/// TRACK-G's grid (half the TRACK-C grid; a shell's row match stays well inside +-2048 half units).
pub const GG: usize = 4096;

/// tail[i] = P[X > u] for u = i - H, from a pmf on the same grid.
fn tail_of(pmf: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; pmf.len()];
    let mut acc = 0.0;
    for i in (0..pmf.len()).rev() {
        t[i] = acc;
        acc += pmf[i];
    }
    t
}

fn at(v: &[f64], u: i64, below: f64) -> f64 {
    if u < -H {
        below
    } else if u >= H {
        0.0
    } else {
        v[(u + H) as usize]
    }
}

/// Which rows a content read wakes.
#[derive(Clone, Copy, Debug)]
pub enum Wake {
    /// The k filled rows with the largest match (ties split uniformly).
    Topk(usize),
    /// Every row whose match C_i . z exceeds theta (in s units).
    Block(f64),
}

/// The block threshold SDMRADIUS's `density_threshold_blocks` would set on a store of T random patterns, from
/// its expectations: filled rows M (1 - e^-lam), mean row load L = lam / (1 - e^-lam), lam = p T.
pub fn block_theta(n: usize, m: usize, r: usize, t: usize, eps_row: f64, eps_pat: f64) -> f64 {
    let p = ball(n, r);
    let lam = p * t as f64;
    let filled = (m as f64 * (1.0 - (-lam).exp())).max(1.0);
    let l = if lam > 0.0 { (lam / (1.0 - (-lam).exp())).max(1.0) } else { 1.0 };
    let tail = (eps_row * p * m as f64 / filled).min(0.5);
    let kr = if tail <= 0.0 { 1.0 } else { phi_inv(1.0 - tail).max(1.0) };
    let others = (t.saturating_sub(1)).max(1) as f64;
    let kp = phi_inv(1.0 - (eps_pat / others).min(0.5)).max(1.0);
    kr.max(kp) * (n as f64 * l).sqrt()
}

/// The block threshold with the rows' co-member overlap kappa^2 in the row load (TRACK-G): two patterns that share
/// a row overlap by kappa^2 on average, so L = (lam + lam^2 kappa^2) / (1 - e^-lam).
pub fn block_theta_geo(n: usize, m: usize, r: usize, t: usize, eps_row: f64, eps_pat: f64, kappa: f64) -> f64 {
    let p = ball(n, r);
    let lam = p * t as f64;
    let filled = (m as f64 * (1.0 - (-lam).exp())).max(1.0);
    let l = if lam > 0.0 { ((lam + lam * lam * kappa * kappa) / (1.0 - (-lam).exp())).max(1.0) } else { 1.0 };
    let tail = (eps_row * p * m as f64 / filled).min(0.5);
    let kr = if tail <= 0.0 { 1.0 } else { phi_inv(1.0 - tail).max(1.0) };
    let others = (t.saturating_sub(1)).max(1) as f64;
    let kp = phi_inv(1.0 - (eps_pat / others).min(0.5)).max(1.0);
    kr.max(kp) * (n as f64 * l).sqrt()
}

/// Predictor TRACK-C for the content reads.
pub struct Content {
    pub n: usize,
    pub m: usize,
    pub r: usize,
    pub p: f64,
    pub pm: f64,
    /// TRACK-G (post-hoc): model where each row's address sits relative to the state, so a row's members share
    /// its alignment with the state. None = TRACK-C (members drawn uniformly, no addresses).
    pub geo: Option<Geo>,
}

/// TRACK-G's tables. A row's address a is summarised by its shell m = the number of bits where a disagrees with
/// the state (P(m) = C(n, m) / 2^n). A stored pattern whose half-overlap with the state is u lies within r of an
/// address in shell m with probability h(u, m) (a hypergeometric sum), independently of the other patterns given
/// the address; so a shell's row match is a compound Poisson sum with rate sum_u hist(u) h(u, m).
pub struct Geo {
    /// (m, P(m)) for every shell holding at least 0.01 expected rows.
    pub shells: Vec<(usize, f64)>,
    /// h[u + n/2][s] for the shells in order.
    pub h: Vec<Vec<f64>>,
    /// kappa = E[1 - 2 d / n | d <= r]: a written pattern's mean agreement with the row's address; two patterns
    /// sharing a row overlap by kappa^2 on average.
    pub kappa: f64,
}

impl Geo {
    pub fn new(n: usize, m: usize, r: usize, min_rows: f64) -> Geo {
        let mut lf = vec![0.0f64; n + 1];
        for k in 1..=n {
            lf[k] = lf[k - 1] + (k as f64).ln();
        }
        let lc = |a: usize, b: usize| if b > a { f64::NEG_INFINITY } else { lf[a] - lf[b] - lf[a - b] };
        let l2 = n as f64 * 2f64.ln();
        let shells: Vec<(usize, f64)> = (0..=n).map(|mm| (mm, (lc(n, mm) - l2).exp())).filter(|x| x.1 * m as f64 >= min_rows).collect();
        let half = n / 2;
        let h: Vec<Vec<f64>> = (0..=n)
            .map(|i| {
                let ka = i; // |A| = n/2 + u = i
                let kb = n - ka; // |B| = n/2 - u
                shells
                    .iter()
                    .map(|&(mm, _)| {
                        // k of the m disagreeing bits fall in A; d = 2k + |B| - m
                        let lim = r as i64 + mm as i64 - kb as i64;
                        if lim < 0 {
                            return 0.0;
                        }
                        let kmax = (lim / 2) as usize;
                        let lo = mm.saturating_sub(kb);
                        let hi = ka.min(mm).min(kmax);
                        let mut acc = 0.0;
                        for k in lo..=hi {
                            if hi < lo {
                                break;
                            }
                            acc += (lc(ka, k) + lc(kb, mm - k) - lc(n, mm)).exp();
                        }
                        acc
                    })
                    .collect()
            })
            .collect();
        let _ = half;
        let (mut num, mut den) = (0.0, 0.0);
        for d in 0..=r {
            let w = (lc(n, d) - l2).exp();
            num += w * (1.0 - 2.0 * d as f64 / n as f64);
            den += w;
        }
        Geo { shells, h, kappa: num / den }
    }

    fn wake_probs(&self, c: &Content, u: &[i64], wake: Wake) -> Vec<f64> {
        let n = c.n;
        let off = (n / 2) as i64;
        let mut hist = vec![0.0f64; n + 1];
        for &x in u {
            hist[(x + off) as usize] += 1.0;
        }
        let ns = self.shells.len();
        let mut pmfs: Vec<Vec<f64>> = Vec::with_capacity(ns);
        let mut tails: Vec<Vec<f64>> = Vec::with_capacity(ns);
        let mut lams: Vec<f64> = Vec::with_capacity(ns);
        for s in 0..ns {
            let lam: f64 = hist.iter().enumerate().map(|(i, &c)| c * self.h[i][s]).sum();
            let sev: Vec<(i64, f64)> = if lam > 0.0 {
                hist.iter().enumerate().filter(|x| *x.1 > 0.0).map(|(i, &cnt)| (i as i64 - off, cnt * self.h[i][s] / lam)).filter(|x| x.1 > 0.0).collect()
            } else {
                vec![(0, 1.0)]
            };
            let pmf = compound_pmf_len(&sev, lam, GG);
            tails.push(tail_of(&pmf));
            pmfs.push(pmf);
            lams.push(lam);
        }
        let mf = c.m as f64;
        let cut = match wake {
            Wake::Topk(k) => {
                let filled: f64 = (0..ns).map(|s| mf * self.shells[s].1 * (1.0 - (-lams[s]).exp())).sum();
                if filled <= k as f64 {
                    None
                } else {
                    let mut above = 0.0f64;
                    let mut found = None;
                    let hh0 = (GG / 2) as i64;
                    for i in (0..GG).rev() {
                        let x = i as i64 - hh0;
                        let mut eq = 0.0;
                        for s in 0..ns {
                            eq += mf * self.shells[s].1 * (pmfs[s][i] - if x == 0 { (-lams[s]).exp() } else { 0.0 }).max(0.0);
                        }
                        if above + eq >= k as f64 {
                            let f = if eq > 0.0 { ((k as f64 - above) / eq).clamp(0.0, 1.0) } else { 0.0 };
                            found = Some((x, f));
                            break;
                        }
                        above += eq;
                    }
                    found
                }
            }
            Wake::Block(theta) => Some(((theta / 2.0).floor() as i64, 0.0)),
        };
        let Some((x, f)) = cut else { return vec![1.0; u.len()] };
        // per distinct u: q = sum_s P(s) h(u, s) Q_s(u) / sum_s P(s) h(u, s), where Q_s removes one copy of the
        // pattern itself from the shell's rest by the inverse Poisson series
        let mut qu = vec![f64::NAN; n + 1];
        for i in 0..=n {
            if hist[i] == 0.0 {
                continue;
            }
            let um = i as i64 - off;
            let (mut num, mut den) = (0.0, 0.0);
            for s in 0..ns {
                let w = self.shells[s].1 * self.h[i][s];
                if w <= 0.0 {
                    continue;
                }
                let hh = self.h[i][s];
                let y = x - um;
                let (mut qt, mut qe) = (0.0, 0.0);
                let mut coef = hh.exp();
                for j in 0..40i64 {
                    qt += coef * at_len(&tails[s], y - j * um, 1.0, (GG / 2) as i64);
                    qe += coef * at_len(&pmfs[s], y - j * um, 0.0, (GG / 2) as i64);
                    coef *= -hh / (j + 1) as f64;
                    if coef.abs() < 1e-15 || (um == 0 && j >= 0) {
                        if um == 0 {
                            // removing a zero-valued copy changes nothing
                            qt = at_len(&tails[s], y, 1.0, (GG / 2) as i64);
                            qe = at_len(&pmfs[s], y, 0.0, (GG / 2) as i64);
                        }
                        break;
                    }
                }
                num += w * (qt + f * qe).clamp(0.0, 1.0);
                den += w;
            }
            qu[i] = if den > 0.0 { (num / den).clamp(0.0, 1.0) } else { 0.0 };
        }
        u.iter().map(|&um| qu[(um + off) as usize]).collect()
    }
}

/// What one sampled read did.
#[derive(Clone, Debug, Default)]
pub struct Sample {
    /// Final overlap with the target (pattern 0).
    pub overlap: f64,
    /// Bits between the read-address and the final state.
    pub travel: usize,
    pub rounds: usize,
    /// Distance from the final state to the nearest stored pattern.
    pub nearest: usize,
    /// First read: effective number of patterns (sum k)^2 / sum k^2, and the largest count's share.
    pub neff1: f64,
    pub share1: f64,
    /// First read: distinct patterns with a woken row.
    pub distinct1: usize,
}

impl Content {
    pub fn new(n: usize, m: usize, r: usize) -> Content {
        let p = ball(n, r);
        Content { n, m, r, p, pm: p * m as f64, geo: None }
    }

    /// TRACK-G (post-hoc): the same predictor with the rows' addresses placed relative to the state.
    pub fn with_geometry(n: usize, m: usize, r: usize) -> Content {
        let p = ball(n, r);
        Content { n, m, r, p, pm: p * m as f64, geo: Some(Geo::new(n, m, r, 0.01)) }
    }

    /// The top-k read's k (SDMREFUSE's choice: the expected wake count p M).
    pub fn k(&self) -> usize {
        self.pm.round().max(1.0) as usize
    }

    /// q_mu = P[a row holding pattern mu is woken] for every pattern, from the patterns' half-unit overlaps u.
    pub fn wake_probs(&self, u: &[i64], wake: Wake) -> Vec<f64> {
        if let Some(g) = &self.geo {
            return g.wake_probs(self, u, wake);
        }
        let t = u.len().max(1);
        let lam = self.p * t as f64;
        let mut hist = vec![0.0f64; (self.n + 1) as usize];
        let off = (self.n / 2) as i64;
        for &x in u {
            hist[(x + off) as usize] += 1.0;
        }
        let sev_all: Vec<(i64, f64)> = hist.iter().enumerate().filter(|x| *x.1 > 0.0).map(|(i, &c)| (i as i64 - off, c / t as f64)).collect();
        let (fr, fi) = severity_cf(&sev_all);
        // the cut, in half units: (x, f) = wake iff row sum > x, or == x with probability f
        let cut = match wake {
            Wake::Topk(k) => {
                let d = compound_from_cf(&fr, &fi, lam);
                let e0 = (-lam).exp();
                let filled = self.m as f64 * (1.0 - e0);
                if filled <= k as f64 {
                    None
                } else {
                    let mut above = 0.0f64;
                    let mut found = None;
                    for i in (0..G).rev() {
                        let x = i as i64 - H;
                        let eq = self.m as f64 * (d[i] - if x == 0 { e0 } else { 0.0 }).max(0.0);
                        if above + eq >= k as f64 {
                            let f = if eq > 0.0 { ((k as f64 - above) / eq).clamp(0.0, 1.0) } else { 0.0 };
                            found = Some((x, f));
                            break;
                        }
                        above += eq;
                    }
                    found
                }
            }
            Wake::Block(theta) => Some(((theta / 2.0).floor() as i64, 0.0)),
        };
        let Some((x, f)) = cut else { return vec![1.0; u.len()] };
        if t <= 64 {
            // exclude the pattern itself from its own rows' rest (matters only when T is small); patterns with the
            // same overlap share the same rest distribution, so each distinct overlap is computed once
            let mut cache: std::collections::HashMap<i64, (Vec<f64>, Vec<f64>)> = std::collections::HashMap::new();
            u.iter()
                .map(|&um| {
                    let (rp, rt) = cache.entry(um).or_insert_with(|| {
                        let mut h = hist.clone();
                        h[(um + off) as usize] -= 1.0;
                        let others = (t - 1) as f64;
                        let sev: Vec<(i64, f64)> = if others > 0.0 {
                            h.iter().enumerate().filter(|x| *x.1 > 0.0).map(|(i, &c)| (i as i64 - off, c / others)).collect()
                        } else {
                            vec![(0, 1.0)]
                        };
                        let rp = compound_pmf(&sev, self.p * others);
                        let rt = tail_of(&rp);
                        (rp, rt)
                    });
                    at(rt, x - um, 1.0) + f * at(rp, x - um, 0.0)
                })
                .collect()
        } else {
            let lr = self.p * (t - 1) as f64;
            let rp = compound_from_cf(&fr, &fi, lr);
            let rt = tail_of(&rp);
            u.iter().map(|&um| at(&rt, x - um, 1.0) + f * at(&rp, x - um, 0.0)).collect()
        }
    }

    /// One sampled read: T random patterns, target pattern 0; the read-address is pattern 0 with each bit flipped with
    /// probability `dmg`, or (`never`) a fresh random pattern. Up to `iters` reads.
    pub fn sample(&self, wake: Wake, t: usize, dmg: f64, never: bool, iters: usize, persist: bool, r: &mut Rng) -> Sample {
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
        let mut z: Vec<u64> = if never {
            (0..w).map(|k| {
                let x = r.next_u64();
                if k + 1 == w { x & tail } else { x }
            })
            .collect()
        } else {
            let mut z = pats[..w].to_vec();
            for j in 0..n {
                if r.unit() < dmg {
                    z[j / 64] ^= 1 << (j % 64);
                }
            }
            z
        };
        let cue = z.clone();
        let half = (n / 2) as i64;
        let mut kprev = vec![0u32; t];
        let mut qprev = vec![0.0f64; t];
        let mut field = vec![0i64; n];
        let mut out = Sample::default();
        for it in 0..iters.max(1) {
            let u: Vec<i64> = (0..t).map(|mu| half - hd(&pats[mu * w..(mu + 1) * w], &z) as i64).collect();
            let q = self.wake_probs(&u, wake);
            field.iter_mut().for_each(|f| *f = 0);
            let (mut sk, mut sk2, mut kmax, mut distinct) = (0.0f64, 0.0f64, 0u32, 0usize);
            for mu in 0..t {
                let k = if persist && it > 0 {
                    if q[mu] >= qprev[mu] {
                        kprev[mu] + poisson(self.pm * (q[mu] - qprev[mu]), r)
                    } else if qprev[mu] > 0.0 {
                        let keep = q[mu] / qprev[mu];
                        (0..kprev[mu]).filter(|_| r.unit() < keep).count() as u32
                    } else {
                        0
                    }
                } else {
                    poisson(self.pm * q[mu], r)
                };
                kprev[mu] = k;
                qprev[mu] = q[mu];
                if k == 0 {
                    continue;
                }
                distinct += 1;
                sk += k as f64;
                sk2 += (k as f64) * (k as f64);
                kmax = kmax.max(k);
                let x = &pats[mu * w..(mu + 1) * w];
                let k = k as i64;
                for j in 0..n {
                    if (x[j / 64] >> (j % 64)) & 1 == 1 {
                        field[j] += k;
                    } else {
                        field[j] -= k;
                    }
                }
            }
            if it == 0 {
                out.neff1 = if sk2 > 0.0 { sk * sk / sk2 } else { 0.0 };
                out.share1 = if sk > 0.0 { kmax as f64 / sk } else { 0.0 };
                out.distinct1 = distinct;
            }
            let mut next = z.clone();
            for j in 0..n {
                if field[j] > 0 {
                    next[j / 64] |= 1 << (j % 64);
                } else if field[j] < 0 {
                    next[j / 64] &= !(1 << (j % 64));
                }
            }
            out.rounds = it + 1;
            if next == z {
                break;
            }
            z = next;
        }
        let d = hd(&pats[..w], &z);
        out.overlap = 1.0 - 2.0 * d as f64 / n as f64;
        out.travel = hd(&cue, &z);
        out.nearest = (0..t).map(|mu| hd(&pats[mu * w..(mu + 1) * w], &z)).min().unwrap_or(n);
        out
    }

    /// Fraction of `samples` sampled reads of a noisy read-address ending at overlap >= 0.95.
    pub fn p_converge(&self, wake: Wake, dmg: f64, t: usize, samples: usize, persist: bool, seed: u64) -> f64 {
        let mut r = Rng::new(seed ^ 0xC0_17E7_0000_0001);
        let ok = (0..samples).filter(|_| self.sample(wake, t, dmg, false, 20, persist, &mut r).overlap >= 0.95).count();
        ok as f64 / samples.max(1) as f64
    }
}

/// Reference TRACK-R: a random membership graph with the store's per-pattern write count and no addresses.
pub struct Member {
    pub n: usize,
    pub t: usize,
    w: usize,
    pub pats: Vec<u64>,
    /// Nonempty rows: their members are members[start[i]..start[i+1]].
    start: Vec<u32>,
    members: Vec<u32>,
}

impl Member {
    /// Every pattern goes to Poisson(p M) rows drawn uniformly from M (a repeat inside one pattern is dropped).
    pub fn random(n: usize, m: usize, r: usize, pats: Vec<u64>, rng: &mut Rng) -> Member {
        let w = (n + 63) / 64;
        let t = pats.len() / w;
        let pm = ball(n, r) * m as f64;
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for mu in 0..t {
            let c = poisson(pm, rng) as usize;
            let mut rows: Vec<u32> = (0..c).map(|_| rng.below(m) as u32).collect();
            rows.sort_unstable();
            rows.dedup();
            edges.extend(rows.into_iter().map(|i| (i, mu as u32)));
        }
        Member::from_edges(n, pats, edges)
    }

    /// From explicit (row, pattern) pairs, e.g. the store's own membership.
    pub fn from_edges(n: usize, pats: Vec<u64>, mut edges: Vec<(u32, u32)>) -> Member {
        let w = (n + 63) / 64;
        let t = pats.len() / w;
        edges.sort_unstable();
        let mut start = vec![0u32];
        let mut members = Vec::with_capacity(edges.len());
        let mut cur = None;
        for (row, mu) in edges {
            if cur != Some(row) && cur.is_some() {
                start.push(members.len() as u32);
            }
            cur = Some(row);
            members.push(mu);
        }
        if cur.is_some() {
            start.push(members.len() as u32);
        }
        Member { n, t, w, pats, start, members }
    }

    pub fn rows(&self) -> usize {
        self.start.len() - 1
    }

    pub fn row(&self, i: usize) -> &[u32] {
        &self.members[self.start[i] as usize..self.start[i + 1] as usize]
    }

    /// Mean |C_i|^2 / n over (up to `cap` evenly spaced) nonempty rows: the block threshold's row load.
    pub fn mean_load(&self, cap: usize) -> f64 {
        let nr = self.rows();
        if nr == 0 {
            return 0.0;
        }
        let step = (nr / cap.max(1)).max(1);
        let (mut s, mut c) = (0.0, 0usize);
        let mut v = vec![0i32; self.n];
        for i in (0..nr).step_by(step) {
            v.iter_mut().for_each(|x| *x = 0);
            for &mu in self.row(i) {
                let x = &self.pats[mu as usize * self.w..(mu as usize + 1) * self.w];
                for j in 0..self.n {
                    v[j] += if (x[j / 64] >> (j % 64)) & 1 == 1 { 1 } else { -1 };
                }
            }
            s += v.iter().map(|&a| (a * a) as f64).sum::<f64>() / self.n as f64;
            c += 1;
        }
        s / c as f64
    }

    /// The content read on the graph: returns (final state, rounds, first-read effective number, first-read top
    /// share). Top-k ties go to the lower row index, as in the store.
    pub fn read(&self, cue: &[u64], wake: Wake, iters: usize) -> (Vec<u64>, usize, f64, f64) {
        let (n, w) = (self.n, self.w);
        let mut z = cue.to_vec();
        let half = (n / 2) as i64;
        let nr = self.rows();
        let mut dots: Vec<(u32, i64)> = Vec::with_capacity(nr);
        let mut cnt = vec![0u32; self.t];
        let (mut neff1, mut share1, mut rounds) = (0.0, 0.0, 0);
        for it in 0..iters.max(1) {
            let u: Vec<i64> = (0..self.t).map(|mu| half - hd(&self.pats[mu * w..(mu + 1) * w], &z) as i64).collect();
            dots.clear();
            for i in 0..nr {
                dots.push((i as u32, self.row(i).iter().map(|&mu| u[mu as usize]).sum::<i64>()));
            }
            let woken: Vec<u32> = match wake {
                Wake::Topk(k) => {
                    let k = k.min(dots.len());
                    if k == 0 {
                        Vec::new()
                    } else {
                        dots.select_nth_unstable_by(k - 1, |a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                        dots[..k].iter().map(|x| x.0).collect()
                    }
                }
                Wake::Block(theta) => dots.iter().filter(|x| (2 * x.1) as f64 > theta).map(|x| x.0).collect(),
            };
            let mut field = vec![0i64; n];
            let mut touched: Vec<u32> = Vec::new();
            for &i in &woken {
                for &mu in self.row(i as usize) {
                    if cnt[mu as usize] == 0 {
                        touched.push(mu);
                    }
                    cnt[mu as usize] += 1;
                    let x = &self.pats[mu as usize * w..(mu as usize + 1) * w];
                    for j in 0..n {
                        field[j] += if (x[j / 64] >> (j % 64)) & 1 == 1 { 1 } else { -1 };
                    }
                }
            }
            if it == 0 {
                let sk: f64 = touched.iter().map(|&mu| cnt[mu as usize] as f64).sum();
                let sk2: f64 = touched.iter().map(|&mu| (cnt[mu as usize] as f64).powi(2)).sum();
                let km = touched.iter().map(|&mu| cnt[mu as usize]).max().unwrap_or(0) as f64;
                neff1 = if sk2 > 0.0 { sk * sk / sk2 } else { 0.0 };
                share1 = if sk > 0.0 { km / sk } else { 0.0 };
            }
            for &mu in &touched {
                cnt[mu as usize] = 0;
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
            if next == z {
                break;
            }
            z = next;
        }
        (z, rounds, neff1, share1)
    }
}

/// One round of a traced read: which rows woke and what their vote looked like.
#[derive(Clone, Debug, Default)]
pub struct Round {
    /// The state this round read (packed) and the rows it woke.
    pub state: Vec<u64>,
    pub woken: Vec<u32>,
    /// Bits the read flipped.
    pub flips: usize,
    /// Self-vote: A = v . z / n (the vote's mean component along the state per bit) and its spread
    /// sigma = sqrt(sum_j (v_j - A z_j)^2 / n); the Gaussian estimate of the flips is n Phi(-A / sigma).
    pub a: f64,
    pub sigma: f64,
}

/// The store's top-k read (same answer as `Store::read_pulls_topk`) recording every round's woken rows.
pub fn trace_topk(st: &Store, rows: &[u32], cue: &[i8], iters: usize, k: usize) -> (Vec<i8>, Vec<Round>) {
    let n = st.n;
    let mut z = cue.to_vec();
    let mut dots: Vec<i32> = rows.iter().map(|&i| st.row(i as usize).iter().zip(&z).map(|(&c, &v)| c as i32 * v as i32).sum()).collect();
    let mut sum = vec![0i32; n];
    let mut out = Vec::new();
    for _ in 0..iters.max(1) {
        let mut v: Vec<(u32, i32)> = rows.iter().copied().zip(dots.iter().copied()).collect();
        let kk = k.min(v.len());
        let woken: Vec<u32> = if kk == 0 {
            Vec::new()
        } else {
            v.select_nth_unstable_by(kk - 1, |a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v[..kk].iter().map(|x| x.0).collect()
        };
        sum.iter_mut().for_each(|s| *s = 0);
        for &i in &woken {
            for (s, &c) in sum.iter_mut().zip(st.row(i as usize)) {
                *s += c as i32;
            }
        }
        let a = sum.iter().zip(&z).map(|(&s, &b)| s as f64 * b as f64).sum::<f64>() / n as f64;
        let sigma = (sum.iter().zip(&z).map(|(&s, &b)| (s as f64 - a * b as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
        let next: Vec<i8> = sum.iter().zip(&z).map(|(&s, &old)| if s > 0 { 1 } else if s < 0 { -1 } else { old }).collect();
        let flips: Vec<(usize, i32)> = (0..n).filter(|&j| next[j] != z[j]).map(|j| (j, 2 * next[j] as i32)).collect();
        out.push(Round { state: pack(&z), woken, flips: flips.len(), a, sigma });
        if flips.is_empty() {
            break;
        }
        for (d, &i) in dots.iter_mut().zip(rows) {
            let row = st.row(i as usize);
            let mut add = 0i32;
            for &(j, w) in &flips {
                add += row[j] as i32 * w;
            }
            *d += add;
        }
        z = next;
    }
    (z, out)
}

/// Which stored patterns a set of rows holds.
#[derive(Clone, Debug, Default)]
pub struct Census {
    /// Woken rows, and how many of them hold no stored pattern.
    pub rows: usize,
    pub empty: usize,
    /// Distinct patterns held, total (pattern, row) pairs K, effective number (sum k)^2 / sum k^2.
    pub distinct: usize,
    pub total: usize,
    pub neff: f64,
    /// The pattern with the most rows: its index, its share k_max / K, and its distance to the state.
    pub top: usize,
    pub share: f64,
    pub top_dist: usize,
}

/// Count, for each row in `rows`, the stored patterns (packed) within the store's activation radius of its address.
pub fn census(st: &Store, pats: &[Vec<u64>], rows: &[u32], z: &[u64]) -> Census {
    let mut cnt: Vec<(usize, u32)> = Vec::new();
    let mut idx = std::collections::HashMap::new();
    let mut c = Census { rows: rows.len(), ..Census::default() };
    for &i in rows {
        let mut any = false;
        for (mu, p) in pats.iter().enumerate() {
            if st.dist(i as usize, p) <= st.radius {
                any = true;
                let e = idx.entry(mu).or_insert_with(|| {
                    cnt.push((mu, 0));
                    cnt.len() - 1
                });
                cnt[*e].1 += 1;
            }
        }
        if !any {
            c.empty += 1;
        }
    }
    c.distinct = cnt.len();
    c.total = cnt.iter().map(|x| x.1 as usize).sum();
    let s2: f64 = cnt.iter().map(|x| (x.1 as f64).powi(2)).sum();
    c.neff = if s2 > 0.0 { (c.total as f64).powi(2) / s2 } else { 0.0 };
    if let Some(&(mu, k)) = cnt.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))) {
        c.top = mu;
        c.share = k as f64 / c.total.max(1) as f64;
        c.top_dist = hd(&pats[mu], z);
    }
    c
}

/// Gaussian estimate of the bits one read flips: n Phi(-A / sigma).
pub fn flips_estimate(n: usize, a: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    n as f64 * phi(-a / sigma)
}

/// Probe tail fractions for signals oriented so that SMALLER is more confident: u(x) = (1 + #{probes <= x}) /
/// (P + 1), from the first probe set; the combined scores are then calibrated on a second probe set.
pub struct TailCal {
    /// Sorted probe values per signal.
    pub sorted: Vec<Vec<f64>>,
}

impl TailCal {
    pub fn new(probes: &[Vec<f64>]) -> TailCal {
        let ns = probes.first().map(|p| p.len()).unwrap_or(0);
        let sorted = (0..ns)
            .map(|s| {
                let mut v: Vec<f64> = probes.iter().map(|p| p[s]).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            })
            .collect();
        TailCal { sorted }
    }

    /// u_s(x) for every signal.
    pub fn fractions(&self, x: &[f64]) -> Vec<f64> {
        self.sorted
            .iter()
            .zip(x)
            .map(|(v, &xi)| {
                let below = v.partition_point(|&a| a <= xi);
                (1 + below) as f64 / (v.len() + 1) as f64
            })
            .collect()
    }

    /// Product score (log): sum_s ln u_s(x). Smaller is more confident.
    pub fn product(&self, x: &[f64]) -> f64 {
        self.fractions(x).iter().map(|u| u.ln()).sum()
    }

    /// Min score: min_s u_s(x). Smaller is more confident.
    pub fn min(&self, x: &[f64]) -> f64 {
        self.fractions(x).into_iter().fold(f64::INFINITY, f64::min)
    }
}

/// The acceptance cut from calibration scores: the value at rank floor(alpha P) of the ascending scores; an
/// answer is accepted iff its score is strictly below it, so at most alpha of the calibration probes pass.
pub fn calibrate(scores: &[f64], alpha: f64) -> f64 {
    let mut v = scores.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = ((alpha * v.len() as f64).floor() as usize).min(v.len().saturating_sub(1));
    v[rank]
}

/// Pack a ±1 pattern (re-export for the instrument).
pub fn packed(z: &[i8]) -> Vec<u64> {
    pack(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{add_address_noise, overlap, random_pattern};

    #[test]
    fn compound_poisson_with_one_value_is_the_poisson_law() {
        // every term is +1, so the sum is Poisson(lam) itself
        let lam = 2.5f64;
        let got = compound_pmf(&[(1, 1.0)], lam);
        let mut pk = (-lam).exp();
        for k in 0..20i64 {
            assert!((got[(k + H) as usize] - pk).abs() < 1e-12, "k {}", k);
            pk *= lam / (k + 1) as f64;
        }
        // rate zero: all mass at 0
        let z = compound_pmf(&[(3, 1.0)], 0.0);
        assert!((z[H as usize] - 1.0).abs() < 1e-12 && z.iter().sum::<f64>() < 1.0 + 1e-12);
    }

    #[test]
    fn tail_fractions_and_the_calibrated_cut_by_hand() {
        let tc = TailCal::new(&[vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(tc.fractions(&[0.0]), vec![0.25]);
        assert_eq!(tc.fractions(&[2.0]), vec![0.75]);
        let s: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(calibrate(&s, 0.05), 5.0);
    }

    #[test]
    fn block_wake_probabilities_follow_the_threshold() {
        let c = Content::new(256, 20_000, 104);
        let u: Vec<i64> = (0..100).map(|i| (i % 40) as i64 - 10).collect();
        // a threshold above any possible match wakes nothing; one far below wakes everything
        assert!(c.wake_probs(&u, Wake::Block(1e6)).iter().all(|&q| q < 1e-12));
        assert!(c.wake_probs(&u, Wake::Block(-1e6)).iter().all(|&q| q > 1.0 - 1e-9));
    }

    #[test]
    fn the_traced_read_is_the_store_read() {
        let mut rr = Rng::new(8);
        let ps: Vec<Vec<i8>> = (0..30).map(|_| random_pattern(256, &mut rr)).collect();
        let mut st = Store::new(256, 20_000, 105, 8);
        st.write_many(&ps);
        let rows: Vec<u32> = (0..st.m).filter(|&i| st.filled[i]).map(|i| i as u32).collect();
        let k = (ball(256, 105) * 20_000.0).round() as usize;
        let cue = add_address_noise(&ps[2], 0.2, &mut rr);
        let (z, rounds) = trace_topk(&st, &rows, &cue, 20, k);
        assert_eq!(z, st.read_pulls_topk(&cue, 20, k).z);
        assert!(overlap(&z, &ps[2]) > 0.99);
        let c = census(&st, &crate::bits::pack_all(&ps), &rounds.last().unwrap().woken, &pack(&z));
        assert_eq!(c.top, 2);
    }
}
