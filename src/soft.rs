//! The soft SDM: Kanerva's memory with a SOFT cut-off, as a machine of p-bits.
//!
//! Every hard location m has a fixed random address x_m. For a read-address at Hamming distance d from x_m, the
//! hard location is activated with probability
//!
//!   phi(d) = 1 / (1 + exp(-(t - d) / w)),
//!
//! a logistic step of width w around the threshold t. Softness s sets w = s sqrt(n) / 2 (in units of the
//! spread of a random read-address's distance). At softness 0 the step is hard: a hard location is activated exactly when
//! d <= r (classic SDM). The threshold t is re-fitted for each softness (`calibrate`) so the expected number
//! of activated hard locations for a random read-address stays the number the hard activation radius gives: the dial changes the SHAPE
//! of the cut-off, never how many hard locations fire on average.
//!
//! Write (Hebbian): fire the hard locations `write_samples` times from the pattern and add f_m p to hard location m's
//! counter row, where f_m is the fraction of those draws it was activated in (`write_samples` 0: the exact phi).
//! Read `read_pass`: hard locations fire from the read-address alone, each data bit is drawn from the activated hard locations'
//! vote, repeated `samples` times; each bit is the majority (Kanerva's one-way pass). Read `read_settle`:
//! the whole machine is Gibbs-sampled with the read-address held, so the data bits also pull back on the hard locations
//! through the same symmetric couplings. `recall` feeds each read-out back as the next read-address.
//!
//! The attention limit: `mean_field` is the read with infinitely many samples; `kernel_inf` is the expected
//! co-activated of two points d bits apart for infinitely many hard locations; `fit_beta` fits that kernel to a
//! softmax on cosine similarity; `attention_read` reads the stored patterns weighted by the kernel or by the
//! softmax.

use crate::codes::overlap;
use crate::rng::Rng;
use crate::theory::{binom_log_pmf, hard_radius};

/// Softness below this is treated as exactly hard.
const HARD: f64 = 1e-12;
/// Width used for the model's pulls at softness zero: large enough that every input is saturated.
const HARD_W: f64 = 1e-9;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Firing probability of a hard location at Hamming distance `d`, with threshold `t` and width `w` (w = 0: hard).
pub fn phi(d: f64, t: f64, w: f64) -> f64 {
    if w <= HARD {
        if d < t {
            1.0
        } else {
            0.0
        }
    } else {
        sigmoid((t - d) / w)
    }
}

/// The threshold that keeps the expected activated fraction for a random read-address at `frac`, for width `w`.
pub fn calibrate(n: usize, frac: f64, w: f64, hard_t: f64) -> f64 {
    if w <= HARD {
        return hard_t;
    }
    let lp = binom_log_pmf(n);
    let pk: Vec<f64> = lp.iter().map(|l| l.exp()).collect();
    let mean = |t: f64| pk.iter().enumerate().map(|(d, p)| p * phi(d as f64, t, w)).sum::<f64>();
    let (mut lo, mut hi) = (-20.0 * n as f64, 20.0 * n as f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mean(mid) < frac {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

/// The machine's numbers. `j` is the counter matrix, `locations x size`; the addresses come from the seed.
#[derive(Clone)]
pub struct Machine {
    pub n: usize,
    pub m: usize,
    pub fire: f64,
    pub softness: f64,
    pub gain: f64,
    pub seed: u64,
    pub write_samples: usize,
    pub addr: Vec<f64>,
    pub radius: usize,
    pub frac: f64,
    pub t: f64,
    pub w: f64,
    pub j: Vec<f64>,
}

impl Machine {
    pub fn new(n: usize, m: usize, fire: f64, softness: f64, gain: f64, seed: u64, write_samples: usize) -> Self {
        let mut r = Rng::new(seed ^ 0xA11C_E5ED_50F7_5D30);
        let addr = (0..m * n).map(|_| if r.unit() < 0.5 { -1.0 } else { 1.0 }).collect();
        let (radius, frac) = hard_radius(n, fire);
        let w = softness * (n as f64).sqrt() / 2.0;
        let hard_t = radius as f64 + 0.5;
        let t = calibrate(n, frac, w, hard_t);
        Machine { n, m, fire, softness, gain, seed, write_samples, addr, radius, frac, t, w, j: vec![0.0; m * n] }
    }

    /// Scale of the data pulls: one clean stored pattern gives a data field of about `gain`.
    pub fn g(&self) -> f64 {
        self.gain / (self.frac * self.m as f64)
    }

    /// Width used for pulls: the real width, or a tiny one at softness zero.
    pub fn w_pull(&self) -> f64 {
        if self.w <= HARD {
            HARD_W
        } else {
            self.w
        }
    }

    pub fn dist(&self, loc: usize, cue: &[f64]) -> f64 {
        let x = &self.addr[loc * self.n..(loc + 1) * self.n];
        (self.n as f64 - x.iter().zip(cue).map(|(a, b)| a * b).sum::<f64>()) / 2.0
    }

    /// Gibbs input of every hard location from the address things alone: (t - d) / (2w).
    pub fn loc_input(&self, cue: &[f64]) -> Vec<f64> {
        let w = self.w_pull();
        (0..self.m).map(|l| (self.t - self.dist(l, cue)) / (2.0 * w)).collect()
    }

    /// Exact activated probability of every hard location for a read-address.
    pub fn fire_probs(&self, cue: &[f64]) -> Vec<f64> {
        (0..self.m).map(|l| phi(self.dist(l, cue), self.t, self.w)).collect()
    }

    /// One Gibbs draw of a p-bit with input `u`, by the same rule as `State::sweep`.
    fn draw(u: f64, rng: &mut Rng) -> bool {
        u.tanh() > rng.signed()
    }

    /// The activated weight of each hard location during a write: mean over `write_samples` settles, or exact.
    pub fn write_weights(&self, p: &[f64], rng: &mut Rng) -> Vec<f64> {
        if self.write_samples == 0 {
            return self.fire_probs(p);
        }
        let u = self.loc_input(p);
        let k = self.write_samples;
        u.iter().map(|&x| (0..k).filter(|_| Self::draw(x, rng)).count() as f64 / k as f64).collect()
    }

    /// Hebbian write; returns the change to the bit-counters (for the model's pulls).
    pub fn write(&mut self, p: &[f64], rng: &mut Rng) -> Vec<(usize, f64)> {
        let f = self.write_weights(p, rng);
        let mut touched = Vec::new();
        for (l, &fl) in f.iter().enumerate() {
            if fl > 0.0 {
                for k in 0..self.n {
                    self.j[l * self.n + k] += fl * p[k];
                }
                touched.push((l, fl));
            }
        }
        touched
    }

    /// Mean-field data field: sum over hard locations of activated probability times bit-counters.
    pub fn mean_field(&self, cue: &[f64]) -> Vec<f64> {
        let f = self.fire_probs(cue);
        let mut out = vec![0.0; self.n];
        for (l, &fl) in f.iter().enumerate() {
            if fl > 0.0 {
                for k in 0..self.n {
                    out[k] += fl * self.j[l * self.n + k];
                }
            }
        }
        out
    }

    /// One `:pass` read: returns the majority read-out (ties keep the read-address bit) and the mean data value.
    pub fn read_pass(&self, cue: &[f64], samples: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let u = self.loc_input(cue);
        let g = self.g();
        let mut acc = vec![0.0; self.n];
        let mut field = vec![0.0; self.n];
        for _ in 0..samples.max(1) {
            field.iter_mut().for_each(|v| *v = 0.0);
            for (l, &x) in u.iter().enumerate() {
                if Self::draw(x, rng) {
                    let row = &self.j[l * self.n..(l + 1) * self.n];
                    for k in 0..self.n {
                        field[k] += row[k];
                    }
                }
            }
            // data input is g*field/2, so a data thing is yes with probability sigmoid(g*field)
            for k in 0..self.n {
                acc[k] += if Self::draw(g * field[k] / 2.0, rng) { 1.0 } else { -1.0 };
            }
        }
        let s = samples.max(1) as f64;
        let mean: Vec<f64> = acc.iter().map(|a| a / s).collect();
        let out = mean.iter().zip(cue).map(|(&a, &c)| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { c }).collect();
        (out, mean)
    }

    /// One `:settle` read: Gibbs-sample hard locations and data jointly with the address held, feedback included.
    pub fn read_settle(&self, cue: &[f64], burn: usize, samples: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let (n, m) = (self.n, self.m);
        let u = self.loc_input(cue);
        let q = self.g() / 4.0; // pull between a location and a data thing is q * J
        let col: Vec<f64> = (0..n).map(|k| (0..m).map(|l| self.j[l * n + k]).sum()).collect();
        let mut y: Vec<f64> = (0..m).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect();
        let mut z: Vec<f64> = (0..n).map(|_| if rng.unit() < 0.5 { -1.0 } else { 1.0 }).collect();
        // yz[l] = sum_k J_lk z_k ; jy[k] = sum_l J_lk y_l
        let mut yz: Vec<f64> = (0..m).map(|l| (0..n).map(|k| self.j[l * n + k] * z[k]).sum()).collect();
        let mut jy: Vec<f64> = (0..n).map(|k| (0..m).map(|l| self.j[l * n + k] * y[l]).sum()).collect();
        let mut order: Vec<usize> = (0..m + n).collect();
        let mut acc = vec![0.0; n];
        for sweep in 0..burn + samples.max(1) {
            for i in (1..order.len()).rev() {
                let r = rng.below(i + 1);
                order.swap(i, r);
            }
            for &i in &order {
                if i < m {
                    let inp = u[i] + q * yz[i];
                    let v = if Self::draw(inp, rng) { 1.0 } else { -1.0 };
                    if v != y[i] {
                        let d = v - y[i];
                        y[i] = v;
                        for k in 0..n {
                            jy[k] += d * self.j[i * n + k];
                        }
                    }
                } else {
                    let k = i - m;
                    let inp = q * jy[k] + q * col[k];
                    let v = if Self::draw(inp, rng) { 1.0 } else { -1.0 };
                    if v != z[k] {
                        let d = v - z[k];
                        z[k] = v;
                        for l in 0..m {
                            yz[l] += d * self.j[l * n + k];
                        }
                    }
                }
            }
            if sweep >= burn {
                for k in 0..n {
                    acc[k] += z[k];
                }
            }
        }
        let s = samples.max(1) as f64;
        let mean: Vec<f64> = acc.iter().map(|a| a / s).collect();
        let out = mean.iter().zip(cue).map(|(&a, &c)| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { c }).collect();
        (out, mean)
    }

    /// Iterated read: each read-out becomes the next read-address.
    pub fn recall(&self, cue: &[f64], rounds: usize, samples: usize, settle: bool, burn: usize, rng: &mut Rng) -> Vec<f64> {
        let mut c = cue.to_vec();
        for _ in 0..rounds.max(1) {
            c = if settle { self.read_settle(&c, burn, samples, rng).0 } else { self.read_pass(&c, samples, rng).0 };
        }
        c
    }

    /// Iterated mean-field read (infinitely many samples, data at zero temperature).
    pub fn recall_mean_field(&self, cue: &[f64], rounds: usize) -> Vec<f64> {
        let mut c = cue.to_vec();
        for _ in 0..rounds.max(1) {
            c = sign_of(&self.mean_field(&c), &c);
        }
        c
    }

    /// The kernel for infinitely many hard locations: expected number (per hard location) activated for BOTH of two
    /// points at Hamming distance d, for d = 0..=n. Exact double sum over the binomial splits.
    pub fn kernel_inf(&self) -> Vec<f64> {
        let n = self.n;
        (0..=n)
            .map(|d| {
                let la = binom_log_pmf(n - d);
                let lb = binom_log_pmf(d);
                let mut s = 0.0;
                for (b, lbb) in lb.iter().enumerate() {
                    let pb = lbb.exp();
                    if pb < 1e-300 {
                        continue;
                    }
                    for (a, laa) in la.iter().enumerate() {
                        let pa = laa.exp();
                        if pa < 1e-300 {
                            continue;
                        }
                        s += pa * pb * phi((a + b) as f64, self.t, self.w) * phi((a + d - b) as f64, self.t, self.w);
                    }
                }
                s
            })
            .collect()
    }

    /// Fit ln K(d) = a - c d over the range where K(d) >= 1e-3 K(0); return the softmax inverse temperature
    /// on cosine similarity, beta = c n / 2 (because d = n (1 - cos) / 2).
    pub fn fit_beta(kernel: &[f64]) -> f64 {
        let n = kernel.len() - 1;
        let k0 = kernel[0];
        let pts: Vec<(f64, f64)> =
            (0..=n / 2).filter(|&d| kernel[d] > 1e-3 * k0 && kernel[d] > 0.0).map(|d| (d as f64, kernel[d].ln())).collect();
        if pts.len() < 2 {
            return f64::INFINITY;
        }
        let np = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / np;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / np;
        let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let c = -sxy / sxx;
        c * n as f64 / 2.0
    }
}

/// The two attention-limit reads over the stored patterns (outside the sampler).
pub enum Attn<'a> {
    /// Weight each stored pattern by the infinite-location kernel at its distance from the read-address.
    Kernel(&'a [f64]),
    /// Weight by exp(beta * cosine similarity): softmax attention.
    Softmax(f64),
}

pub fn attention_read(cue: &[f64], stored: &[Vec<f64>], how: &Attn) -> Vec<f64> {
    let n = cue.len();
    let sims: Vec<f64> = stored.iter().map(|p| overlap(cue, p)).collect();
    let wts: Vec<f64> = match how {
        Attn::Kernel(k) => sims.iter().map(|s| k[((n as f64) * (1.0 - s) / 2.0).round() as usize]).collect(),
        Attn::Softmax(beta) => {
            let mx = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            sims.iter().map(|s| (beta * (s - mx)).exp()).collect()
        }
    };
    (0..n)
        .map(|k| {
            let f: f64 = stored.iter().zip(&wts).map(|(p, w)| w * p[k]).sum();
            if f > 0.0 {
                1.0
            } else if f < 0.0 {
                -1.0
            } else {
                cue[k]
            }
        })
        .collect()
}

pub fn sign_of(v: &[f64], tie: &[f64]) -> Vec<f64> {
    v.iter().zip(tie).map(|(&a, &c)| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { c }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::{code, with_address_noise};

    #[test]
    fn the_firing_step_by_hand() {
        assert_eq!(phi(3.0, 4.0, 0.0), 1.0);
        assert_eq!(phi(5.0, 4.0, 0.0), 0.0);
        assert!((phi(4.0, 4.0, 2.0) - 0.5).abs() < 1e-15);
        assert!((phi(2.0, 4.0, 2.0) - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-15);
        // at softness 0 the threshold is the hard activation radius plus a half
        let mc = Machine::new(64, 10, 0.05, 0.0, 8.0, 1, 0);
        assert_eq!(mc.t, mc.radius as f64 + 0.5);
    }

    /// An independent hard SDM: integer Hamming distance, activation radius test, plain bit-counters, majority read.
    fn hard_read(mc: &Machine, c: &[Vec<f64>], a: &[f64]) -> Vec<f64> {
        let act: Vec<usize> = (0..mc.m).filter(|&l| mc.addr[l * mc.n..(l + 1) * mc.n].iter().zip(a).filter(|(x, y)| x != y).count() <= mc.radius).collect();
        (0..mc.n).map(|k| {
            let s: f64 = act.iter().map(|&l| c[l][k]).sum();
            if s > 0.0 { 1.0 } else if s < 0.0 { -1.0 } else { 0.0 }
        }).collect()
    }

    #[test]
    fn softness_zero_is_exactly_hard_sdm() {
        let mut mc = Machine::new(128, 600, 0.05, 0.0, 1e6, 3, 4);
        let mut c = vec![vec![0.0f64; 128]; 600];
        let mut rng = Rng::new(9);
        let pats: Vec<Vec<f64>> = (0..12).map(|i| code(&format!("p{}", i), 128)).collect();
        for p in &pats {
            mc.write(p, &mut rng);
            for l in 0..600 {
                if mc.addr[l * 128..(l + 1) * 128].iter().zip(p).filter(|(x, y)| x != y).count() <= mc.radius {
                    for k in 0..128 {
                        c[l][k] += p[k];
                    }
                }
            }
        }
        for (x, y) in mc.j.chunks(128).zip(&c) {
            assert_eq!(x, &y[..]);
        }
        for (i, p) in pats.iter().enumerate() {
            let cue = with_address_noise(p, 0.1 * (i % 4) as f64, &mut rng);
            let (got, _) = mc.read_pass(&cue, 4, &mut rng);
            let h = hard_read(&mc, &c, &cue);
            assert!(got.iter().zip(&h).all(|(g, h)| *h == 0.0 || g == h), "pattern {}", i);
        }
    }

    #[test]
    fn the_calibration_holds_the_firing_count_across_softness() {
        for soft in [0.25, 1.0, 2.0] {
            let mc = Machine::new(256, 4000, 0.05, soft, 8.0, 1, 0);
            let mut rng = Rng::new(2);
            let mean = (0..20).map(|_| mc.fire_probs(&crate::codes::noise(256, &mut rng)).iter().sum::<f64>()).sum::<f64>() / 20.0 / 4000.0;
            assert!((mean - mc.frac).abs() < 0.006, "softness {} fires {} want {}", soft, mean, mc.frac);
        }
    }

    #[test]
    fn recall_works_and_shuffled_counters_recall_nothing() {
        let mut mc = Machine::new(256, 2000, 0.05, 0.25, 64.0, 1, 16);
        let mut rng = Rng::new(3);
        let pats: Vec<Vec<f64>> = (0..5).map(|i| code(&format!("q{}", i), 256)).collect();
        for p in &pats {
            mc.write(p, &mut rng);
        }
        for p in &pats {
            assert!(overlap(&mc.recall(&with_address_noise(p, 0.15, &mut rng), 3, 16, false, 0, &mut rng), p) > 0.95);
        }
        let mut rows: Vec<Vec<f64>> = mc.j.chunks(256).map(|c| c.to_vec()).collect();
        rows.rotate_left(1);
        mc.j = rows.concat();
        for p in &pats {
            assert!(overlap(&mc.recall(&with_address_noise(p, 0.15, &mut rng), 3, 16, false, 0, &mut rng), p) < 0.9);
        }
    }

    #[test]
    fn softness_flattens_the_kernel() {
        let hard = Machine::new(64, 10, 0.05, 0.0, 8.0, 1, 0).kernel_inf();
        let soft = Machine::new(64, 10, 0.05, 2.0, 8.0, 1, 0).kernel_inf();
        assert!(Machine::fit_beta(&soft) < Machine::fit_beta(&hard));
        // the kernel at distance 0 is the activated fraction itself for the hard step
        let mc = Machine::new(64, 10, 0.05, 0.0, 8.0, 1, 0);
        assert!((hard[0] - mc.frac).abs() < 1e-12);
    }
}
