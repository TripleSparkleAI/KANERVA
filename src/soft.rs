//! The soft SDM: Kanerva's memory with a SOFT cut-off, as a machine of p-bits.
//!
//! Every hard-location m has a fixed random address x_m. For a read-address at Hamming distance d from x_m, the
//! hard-location is activated with probability
//!
//!   phi(d) = 1 / (1 + exp(-(t - d) / w)),
//!
//! a logistic step of width w around the threshold t. Softness s sets w = s sqrt(n) / 2 (in units of the
//! spread of a random read-address's distance). At softness 0 the step is hard: a hard-location is activated exactly when
//! d <= r (classic SDM). The threshold t is re-fitted for each softness (`calibrate`) so the expected number
//! of activated hard-locations for a random read-address stays the number the hard activation-radius gives: the dial changes the SHAPE
//! of the cut-off, never how many hard-locations are activated on average.
//!
//! Write (Hebbian): activate the hard-locations `write_samples` times from the pattern and add f_m p to hard-location m's
//! counter row, where f_m is the fraction of those draws it was activated in (`write_samples` 0: the exact phi).
//! Read `read_pass`: hard-locations are activated from the read-address alone, each data bit is drawn from the activated hard-locations'
//! vote, repeated `samples` times; each bit is the majority (Kanerva's one-way pass). Read `read_settle`:
//! the whole machine is Gibbs-sampled with the read-address held, so the data bits also pull back on the hard-locations
//! through the same symmetric couplings. `recall` feeds each read-out back as the next read-address.
//!
//! The attention limit: `mean_field` is the read with infinitely many samples; `kernel_inf` is the expected
//! share of hard-locations both of two points d bits apart activate, for infinitely many hard-locations; `fit_beta` fits that kernel to a
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

/// Probability of activation of a hard-location at Hamming distance `d`, with threshold `t` and width `w` (w = 0: hard).
///
/// ```
/// use kanerva::soft::phi;
/// assert_eq!(phi(3.0, 5.0, 0.0), 1.0); // w = 0: inside the hard cut-off
/// assert_eq!(phi(7.0, 5.0, 0.0), 0.0); // and outside it
/// assert!((phi(5.0, 5.0, 2.0) - 0.5).abs() < 1e-12); // soft: one half at the threshold
/// ```
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
///
/// ```
/// use kanerva::soft::{calibrate, phi};
/// use kanerva::theory::binom_log_pmf;
/// assert_eq!(calibrate(256, 0.05, 0.0, 113.5), 113.5); // width 0 keeps the hard threshold
/// let t = calibrate(256, 0.05, 8.0, 113.5);
/// let frac: f64 = binom_log_pmf(256).iter().enumerate().map(|(d, l)| l.exp() * phi(d as f64, t, 8.0)).sum();
/// assert!((frac - 0.05).abs() < 1e-9);
/// ```
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
///
/// ```
/// use kanerva::soft::Machine;
/// let mc = Machine::new(256, 2_000, 0.05, 0.25, 64.0, 1, 16);
/// assert_eq!(mc.addr.len(), 2_000 * 256);
/// assert_eq!(mc.j.len(), 2_000 * 256);
/// assert!(mc.frac >= 0.05); // the hard activation-radius activates at least the asked share
/// ```
#[derive(Clone)]
pub struct Machine {
    /// The word-size in bits.
    pub n: usize,
    /// The number of hard-locations M.
    pub m: usize,
    /// The target activation-probability of the hard cut-off (the field keeps its old name, `fire`, so code that reads it
    /// keeps compiling).
    pub fire: f64,
    /// The softness dial s: 0 is the hard cut-off.
    pub softness: f64,
    /// The data field one clean stored pattern gives (see `g`).
    pub gain: f64,
    /// The seed the addresses are drawn from.
    pub seed: u64,
    /// How many activation draws a write averages; 0 uses the exact probability.
    pub write_samples: usize,
    /// The ±1 addresses, row-major: hard-location l's address is `addr[l * n..(l + 1) * n]`.
    pub addr: Vec<f64>,
    /// The hard activation-radius for the activation-probability `fire` (from `theory::hard_radius`).
    pub radius: usize,
    /// The exact fraction of hard-locations that activation-radius activates for a random read-address.
    pub frac: f64,
    /// The threshold t of the logistic cut-off, refitted for the softness (`calibrate`).
    pub t: f64,
    /// The width w = s sqrt(n) / 2 of the logistic cut-off.
    pub w: f64,
    /// The bit-counters, row-major: hard-location l's row is `j[l * n..(l + 1) * n]`.
    pub j: Vec<f64>,
}

impl Machine {
    /// A machine of `m` hard-locations of `n` bits: random addresses from `seed`, the hard activation-radius for `activation_probability`, and the threshold refitted for `softness`.
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let hard = Machine::new(64, 100, 0.05, 0.0, 64.0, 1, 0);
    /// assert_eq!(hard.w, 0.0);
    /// assert_eq!(hard.t, hard.radius as f64 + 0.5);
    /// let soft = Machine::new(64, 100, 0.05, 0.5, 64.0, 1, 0);
    /// assert_eq!(soft.w, 0.5 * 8.0 / 2.0); // w = s sqrt(n) / 2
    /// ```
    pub fn new(n: usize, m: usize, activation_probability: f64, softness: f64, gain: f64, seed: u64, write_samples: usize) -> Self {
        let mut r = Rng::new(seed ^ 0xA11C_E5ED_50F7_5D30);
        let addr = (0..m * n).map(|_| if r.unit() < 0.5 { -1.0 } else { 1.0 }).collect();
        let (radius, frac) = hard_radius(n, activation_probability);
        let w = softness * (n as f64).sqrt() / 2.0;
        let hard_t = radius as f64 + 0.5;
        let t = calibrate(n, frac, w, hard_t);
        Machine { n, m, fire: activation_probability, softness, gain, seed, write_samples, addr, radius, frac, t, w, j: vec![0.0; m * n] }
    }

    /// Scale of the data pulls: one clean stored pattern gives a data field of about `gain`.
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let mc = Machine::new(64, 100, 0.05, 0.25, 64.0, 1, 0);
    /// assert!((mc.g() * mc.frac * mc.m as f64 - mc.gain).abs() < 1e-9);
    /// ```
    pub fn g(&self) -> f64 {
        self.gain / (self.frac * self.m as f64)
    }

    /// Width used for pulls: the real width, or a tiny one at softness zero.
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let hard = Machine::new(64, 100, 0.05, 0.0, 64.0, 1, 0);
    /// assert!(hard.w_pull() > 0.0 && hard.w_pull() < 1e-6);
    /// let soft = Machine::new(64, 100, 0.05, 0.25, 64.0, 1, 0);
    /// assert_eq!(soft.w_pull(), soft.w);
    /// ```
    pub fn w_pull(&self) -> f64 {
        if self.w <= HARD {
            HARD_W
        } else {
            self.w
        }
    }

    /// Hamming distance from hard-location `loc`'s address to a ±1 state.
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let mc = Machine::new(64, 100, 0.05, 0.25, 64.0, 1, 0);
    /// let a: Vec<f64> = mc.addr[..64].to_vec(); // hard-location 0's own address
    /// assert_eq!(mc.dist(0, &a), 0.0);
    /// let flipped: Vec<f64> = a.iter().map(|v| -v).collect();
    /// assert_eq!(mc.dist(0, &flipped), 64.0);
    /// ```
    pub fn dist(&self, loc: usize, read_address: &[f64]) -> f64 {
        let x = &self.addr[loc * self.n..(loc + 1) * self.n];
        (self.n as f64 - x.iter().zip(read_address).map(|(a, b)| a * b).sum::<f64>()) / 2.0
    }

    /// Gibbs input of every hard-location from the address things alone: (t - d) / (2w).
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let mc = Machine::new(64, 100, 0.05, 0.25, 64.0, 1, 0);
    /// let a: Vec<f64> = mc.addr[..64].to_vec();
    /// let u = mc.loc_input(&a);
    /// assert_eq!(u.len(), 100);
    /// assert!(u[0] > 0.0); // a hard-location at distance 0 is pushed toward active
    /// ```
    pub fn loc_input(&self, read_address: &[f64]) -> Vec<f64> {
        let w = self.w_pull();
        (0..self.m).map(|l| (self.t - self.dist(l, read_address)) / (2.0 * w)).collect()
    }

    /// Exact activated probability of every hard-location for a read-address.
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let mc = Machine::new(64, 100, 0.05, 0.0, 64.0, 1, 0);
    /// let a: Vec<f64> = mc.addr[..64].to_vec();
    /// let p = mc.fire_probs(&a);
    /// assert_eq!(p[0], 1.0); // its own address is inside the hard activation-radius
    /// assert!(p.iter().all(|&x| x == 0.0 || x == 1.0)); // softness 0: every probability is 0 or 1
    /// ```
    pub fn fire_probs(&self, read_address: &[f64]) -> Vec<f64> {
        (0..self.m).map(|l| phi(self.dist(l, read_address), self.t, self.w)).collect()
    }

    /// One Gibbs draw of a p-bit with input `u`, by the same rule as `State::sweep`.
    fn draw(u: f64, rng: &mut Rng) -> bool {
        u.tanh() > rng.signed()
    }

    /// The activated weight of each hard-location during a write: mean over `write_samples` settles, or exact.
    ///
    /// ```
    /// use kanerva::{Rng, codes::code, soft::Machine};
    /// let mc = Machine::new(64, 100, 0.05, 0.25, 64.0, 1, 0); // write_samples 0: the exact probability
    /// let p = code("cat", 64);
    /// assert_eq!(mc.write_weights(&p, &mut Rng::new(1)), mc.fire_probs(&p));
    /// ```
    pub fn write_weights(&self, p: &[f64], rng: &mut Rng) -> Vec<f64> {
        if self.write_samples == 0 {
            return self.fire_probs(p);
        }
        let u = self.loc_input(p);
        let k = self.write_samples;
        u.iter().map(|&x| (0..k).filter(|_| Self::draw(x, rng)).count() as f64 / k as f64).collect()
    }

    /// Hebbian write; returns the change to the bit-counters (for the model's pulls).
    ///
    /// ```
    /// use kanerva::{Rng, codes::code, soft::Machine};
    /// let mut mc = Machine::new(256, 2_000, 0.05, 0.25, 64.0, 1, 16);
    /// let touched = mc.write(&code("cat", 256), &mut Rng::new(1));
    /// assert!(!touched.is_empty());
    /// assert!(mc.j.iter().any(|&c| c != 0.0));
    /// ```
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

    /// Mean-field data field: sum over hard-locations of activated probability times bit-counters.
    ///
    /// ```
    /// use kanerva::{Rng, codes::code, soft::Machine};
    /// let mut mc = Machine::new(256, 2_000, 0.05, 0.25, 64.0, 1, 16);
    /// let cat = code("cat", 256);
    /// assert!(mc.mean_field(&cat).iter().all(|&v| v == 0.0)); // nothing written yet
    /// mc.write(&cat, &mut Rng::new(1));
    /// let field = mc.mean_field(&cat);
    /// assert!(field.iter().zip(&cat).all(|(f, c)| f * c > 0.0)); // the field points at the word
    /// ```
    pub fn mean_field(&self, read_address: &[f64]) -> Vec<f64> {
        let f = self.fire_probs(read_address);
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
    ///
    /// ```
    /// use kanerva::{Rng, codes::code, soft::Machine};
    /// let mut mc = Machine::new(256, 2_000, 0.05, 0.25, 64.0, 1, 16);
    /// let mut rng = Rng::new(1);
    /// let words: Vec<Vec<f64>> = ["a", "b", "c"].iter().map(|s| code(s, 256)).collect();
    /// for w in &words { mc.write(w, &mut rng); }
    /// let (out, mean) = mc.read_pass(&words[1], 16, &mut rng);
    /// assert_eq!(out, words[1]);
    /// assert!(mean.iter().all(|m| m.abs() <= 1.0));
    /// ```
    pub fn read_pass(&self, read_address: &[f64], samples: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let u = self.loc_input(read_address);
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
        let out = mean.iter().zip(read_address).map(|(&a, &c)| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { c }).collect();
        (out, mean)
    }

    /// One `:settle` read: Gibbs-sample hard-locations and data jointly with the address held, feedback included.
    ///
    /// ```
    /// use kanerva::{Rng, codes::code, soft::Machine};
    /// let mut mc = Machine::new(128, 400, 0.05, 0.0, 64.0, 1, 16); // softness 0: the hard cut-off
    /// let mut rng = Rng::new(1);
    /// let words: Vec<Vec<f64>> = ["a", "b", "c"].iter().map(|s| code(s, 128)).collect();
    /// for w in &words { mc.write(w, &mut rng); }
    /// let (out, mean) = mc.read_settle(&words[0], 4, 8, &mut rng);
    /// assert_eq!(out, words[0]);
    /// assert!(mean.iter().all(|m| m.abs() <= 1.0));
    /// ```
    pub fn read_settle(&self, read_address: &[f64], burn: usize, samples: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let (n, m) = (self.n, self.m);
        let u = self.loc_input(read_address);
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
        let out = mean.iter().zip(read_address).map(|(&a, &c)| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { c }).collect();
        (out, mean)
    }

    /// Iterated read: each read-out becomes the next read-address.
    ///
    /// ```
    /// use kanerva::{Rng, codes::{code, with_address_noise}, soft::Machine};
    /// let mut mc = Machine::new(256, 2_000, 0.05, 0.25, 64.0, 1, 16);
    /// let mut rng = Rng::new(1);
    /// let words: Vec<Vec<f64>> = ["a", "b", "c", "d", "e"].iter().map(|s| code(s, 256)).collect();
    /// for w in &words { mc.write(w, &mut rng); }
    /// let read_address = with_address_noise(&words[0], 0.2, &mut rng);
    /// assert_eq!(mc.recall(&read_address, 3, 16, false, 0, &mut rng), words[0]);
    /// ```
    pub fn recall(&self, read_address: &[f64], rounds: usize, samples: usize, settle: bool, burn: usize, rng: &mut Rng) -> Vec<f64> {
        let mut c = read_address.to_vec();
        for _ in 0..rounds.max(1) {
            c = if settle { self.read_settle(&c, burn, samples, rng).0 } else { self.read_pass(&c, samples, rng).0 };
        }
        c
    }

    /// Iterated mean-field read (infinitely many samples, data at zero temperature).
    ///
    /// ```
    /// use kanerva::{Rng, codes::{code, with_address_noise}, soft::Machine};
    /// let mut mc = Machine::new(256, 2_000, 0.05, 0.25, 64.0, 1, 16);
    /// let mut rng = Rng::new(1);
    /// let words: Vec<Vec<f64>> = ["a", "b", "c", "d", "e"].iter().map(|s| code(s, 256)).collect();
    /// for w in &words { mc.write(w, &mut rng); }
    /// let read_address = with_address_noise(&words[2], 0.2, &mut rng);
    /// assert_eq!(mc.recall_mean_field(&read_address, 3), words[2]);
    /// ```
    pub fn recall_mean_field(&self, read_address: &[f64], rounds: usize) -> Vec<f64> {
        let mut c = read_address.to_vec();
        for _ in 0..rounds.max(1) {
            c = sign_of(&self.mean_field(&c), &c);
        }
        c
    }

    /// The kernel for infinitely many hard-locations: expected number (per hard-location) activated for BOTH of two
    /// points at Hamming distance d, for d = 0..=n. Exact double sum over the binomial splits.
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let hard = Machine::new(64, 100, 0.05, 0.0, 64.0, 1, 0);
    /// let k = hard.kernel_inf();
    /// assert_eq!(k.len(), 65);
    /// assert!((k[0] - hard.frac).abs() < 1e-9); // a point shares its whole activated set with itself
    /// assert_eq!(k[64], 0.0); // opposite points share no hard-location
    /// ```
    pub fn kernel_inf(&self) -> Vec<f64> {
        let n = self.n;
        // phi at every whole distance 0..=n and every binomial's terms, computed once. The sum reads the same values
        // the direct form computed inside its loops, in the same order, so the kernel is the same to the bit
        // (`the_tabled_kernel_is_the_direct_kernel_bit_for_bit`); only the exp calls moved out of the inner loop.
        let ph: Vec<f64> = (0..=n).map(|x| phi(x as f64, self.t, self.w)).collect();
        let pmf: Vec<Vec<f64>> = (0..=n).map(|k| binom_log_pmf(k).iter().map(|l| l.exp()).collect()).collect();
        (0..=n)
            .map(|d| {
                let (la, lb) = (&pmf[n - d], &pmf[d]);
                let mut s = 0.0;
                for (b, &pb) in lb.iter().enumerate() {
                    if pb < 1e-300 {
                        continue;
                    }
                    for (a, &pa) in la.iter().enumerate() {
                        if pa < 1e-300 {
                            continue;
                        }
                        s += pa * pb * ph[a + b] * ph[a + d - b];
                    }
                }
                s
            })
            .collect()
    }

    /// Fit ln K(d) = a - c d over the range where K(d) >= 1e-3 K(0); return the softmax inverse temperature
    /// on cosine similarity, beta = c n / 2 (because d = n (1 - cos) / 2).
    ///
    /// ```
    /// use kanerva::soft::Machine;
    /// let mc = Machine::new(256, 2_000, 0.05, 0.25, 64.0, 1, 16);
    /// let beta = Machine::fit_beta(&mc.kernel_inf());
    /// assert!((beta - 2.17).abs() < 0.01);
    /// ```
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
///
/// ```
/// use kanerva::soft::Attn;
/// let how = Attn::Softmax(2.0);
/// assert!(matches!(how, Attn::Softmax(b) if b == 2.0));
/// ```
pub enum Attn<'a> {
    /// Weight each stored pattern by the infinite-location kernel at its distance from the read-address.
    Kernel(&'a [f64]),
    /// Weight by exp(beta * cosine similarity): softmax attention.
    Softmax(f64),
}

/// Read the stored patterns weighted by `how` at their similarity to the read-address; each bit is the sign of the weighted sum, a zero sum keeping the read-address's bit.
///
/// ```
/// use kanerva::{codes::code, soft::{attention_read, Attn}};
/// let stored = vec![code("cat", 256), code("owl", 256)];
/// assert_eq!(attention_read(&stored[0], &stored, &Attn::Softmax(50.0)), stored[0]);
/// ```
pub fn attention_read(read_address: &[f64], stored: &[Vec<f64>], how: &Attn) -> Vec<f64> {
    let n = read_address.len();
    let sims: Vec<f64> = stored.iter().map(|p| overlap(read_address, p)).collect();
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
                read_address[k]
            }
        })
        .collect()
}

/// The sign of each entry of `v`; a zero entry takes the entry of `tie`.
///
/// ```
/// use kanerva::soft::sign_of;
/// assert_eq!(sign_of(&[2.0, -1.0, 0.0], &[-1.0, 1.0, -1.0]), vec![1.0, -1.0, -1.0]);
/// ```
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
        // at softness 0 the threshold is the hard activation-radius plus a half
        let mc = Machine::new(64, 10, 0.05, 0.0, 8.0, 1, 0);
        assert_eq!(mc.t, mc.radius as f64 + 0.5);
    }

    /// An independent hard SDM: integer Hamming distance, activation-radius test, plain bit-counters, majority read.
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
            let read_address = with_address_noise(p, 0.1 * (i % 4) as f64, &mut rng);
            let (got, _) = mc.read_pass(&read_address, 4, &mut rng);
            let h = hard_read(&mc, &c, &read_address);
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

    /// The kernel as it was first written: every exp and phi computed inside the loops.
    fn kernel_direct(mc: &Machine) -> Vec<f64> {
        let n = mc.n;
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
                        s += pa * pb * phi((a + b) as f64, mc.t, mc.w) * phi((a + d - b) as f64, mc.t, mc.w);
                    }
                }
                s
            })
            .collect()
    }

    #[test]
    fn the_tabled_kernel_is_the_direct_kernel_bit_for_bit() {
        for (n, soft, fire) in [(64, 0.0, 0.05), (64, 0.5, 0.05), (128, 0.25, 0.01), (256, 0.25, 0.05), (256, 2.0, 0.05)] {
            let mc = Machine::new(n, 10, fire, soft, 8.0, 1, 0);
            let (a, b) = (mc.kernel_inf(), kernel_direct(&mc));
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "n {} softness {}", n, soft);
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
