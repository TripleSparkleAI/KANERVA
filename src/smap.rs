//! Capacity theory: the Bricken-Pehlevan signal-to-noise map (the S-map) for an iterated read, its
//! critical-distance and capacity, and the choice of activation-radius for the address-noise a read-address will carry.
//!
//! A read-address d bits from its stored pattern shares M I(d) hard-locations with it (I from `theory::intersection`).
//! Those hard-locations vote for the pattern; the T - 1 other stored patterns add noise. One read maps d to
//!
//!   SNR(d) = M I(d) / sqrt(M I(d) + (T - 1)(M I_o + (M I_o)^2)),   I_o = I(n/2),
//!   d_next = n Phi(-SNR(d)).
//!
//! Iterating the map from d0 says whether the read converges. `SMap` is the plain map with every I(d)
//! precomputed; `Lazy` computes I(d) on demand and adds three refinements measured in the SETTLE campaign:
//! noise averaged over the other patterns' distances (`averaged`), the mirror field (`mirrored`), the first
//! read conditioned on a Poisson number of shared hard-locations (`p_converge`), address-noise as independent bit flips, so the distance is binomial (`p_converge_flips`),
//! and RACE, where rival patterns compete as whole patterns (`p_converge_race`).

use crate::rng::Rng;
use crate::theory::{intersection, phi, radius_for};
use std::cell::RefCell;

/// The Bricken-Pehlevan signal-to-noise S-map for an iterated read (wiki WIKI_SDR 85 section 2, 21 section 5):
/// SNR(d) = M I(d) / sqrt(M I(d) + (T-1)(M I_o + (M I_o)^2)), next d = n Phi(-SNR(d)).
///
/// ```
/// use kanerva::smap::{SMap, goal};
/// let s = SMap::new(256, 2_000, 112);
/// assert_eq!(s.inter.len(), 257);
/// assert!(s.capacity(26, goal(256)) > 50); // about 10% address-noise
/// ```
pub struct SMap {
    /// The word-size in bits.
    pub n: usize,
    /// The number of hard-locations M, as a float.
    pub m: f64,
    /// I(d) for d = 0..=n.
    pub inter: Vec<f64>,
}

impl SMap {
    /// Precompute I(d) for d = 0..=n at word-size `n`, `m` hard-locations and activation-radius `r`.
    ///
    /// ```
    /// let s = kanerva::smap::SMap::new(256, 2_000, 112);
    /// assert!(s.inter[0] > s.inter[26]); // nearer points share more hard-locations
    /// ```
    pub fn new(n: usize, m: usize, r: usize) -> SMap {
        SMap { n, m: m as f64, inter: (0..=n).map(|d| intersection(n, r, d)).collect() }
    }
    /// SNR(d) with `t` stored patterns (t below 1 is treated as 1).
    ///
    /// ```
    /// let s = kanerva::smap::SMap::new(256, 2_000, 112);
    /// assert!(s.snr(26, 1) > s.snr(26, 20)); // more stored patterns, more noise
    /// assert!((s.snr(26, 1) - (2_000.0 * s.inter[26]).sqrt()).abs() < 1e-9); // one pattern: SNR = sqrt(M I(d))
    /// ```
    pub fn snr(&self, d: usize, t: usize) -> f64 {
        let sig = self.m * self.inter[d];
        let io = self.m * self.inter[self.n / 2];
        sig / (sig + (t.max(1) - 1) as f64 * (io + io * io)).sqrt()
    }
    /// Expected next distance.
    ///
    /// ```
    /// let s = kanerva::smap::SMap::new(256, 2_000, 112);
    /// assert!(s.next(26, 20) < 26.0); // a read from 26 bits away moves closer
    /// ```
    pub fn next(&self, d: usize, t: usize) -> f64 {
        self.n as f64 * phi(-self.snr(d, t))
    }
    /// Iterate from d0; true if it reaches a distance at most `goal` within 50 steps.
    ///
    /// ```
    /// use kanerva::smap::{SMap, goal};
    /// let s = SMap::new(256, 2_000, 112);
    /// assert!(s.converges(26, 20, goal(256)));
    /// assert!(!s.converges(26, 1_000, goal(256)));
    /// ```
    pub fn converges(&self, d0: usize, t: usize, goal: f64) -> bool {
        let mut d = d0 as f64;
        for _ in 0..50 {
            if d <= goal {
                return true;
            }
            let nd = self.next(d.round() as usize, t);
            if nd >= d {
                return false;
            }
            d = nd;
        }
        d <= goal
    }
    /// Largest T (by doubling then bisection) whose S-map from d0 reaches `goal`.
    ///
    /// ```
    /// use kanerva::smap::{SMap, goal};
    /// let s = SMap::new(256, 2_000, 112);
    /// let cap = s.capacity(26, goal(256));
    /// assert!(s.converges(26, cap, goal(256)) && !s.converges(26, cap + 1, goal(256)));
    /// ```
    pub fn capacity(&self, d0: usize, goal: f64) -> usize {
        if !self.converges(d0, 1, goal) {
            return 0;
        }
        let mut hi = 1usize;
        while self.converges(d0, hi * 2, goal) && hi < SEARCH_CAP {
            hi *= 2;
        }
        let (mut lo, mut hi) = (hi, hi * 2);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.converges(d0, mid, goal) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
    /// Critical distance: the smallest d at which the S-map no longer moves d toward 0 (the unstable fixed
    /// point), scanning upward (bisection finds the wrong root, wiki 85 section 2).
    ///
    /// Read the result with the load in mind. Under light load the first such d is the unstable fixed point,
    /// Kanerva's critical-distance; a lightly loaded map pulls in from every distance, so the scan returns n/2.
    /// Under heavy load a read that starts next to its pattern drifts out to a residual error, so the map already
    /// fails to move d = 1 toward 0 and the scan returns 1, although reads from farther out may still settle at
    /// that residual. The value is then "nothing is pulled all the way in", not "nothing is pulled in".
    ///
    /// ```
    /// let s = kanerva::smap::SMap::new(256, 2_000, 112);
    /// assert_eq!(s.critical(10), 128); // lightly loaded: the map pulls in from every distance up to n/2
    /// assert!(s.critical(100) < 10);   // overloaded: no distance is pulled all the way to 0
    /// ```
    pub fn critical(&self, t: usize) -> usize {
        for d in 1..=self.n / 2 {
            if self.next(d, t) >= d as f64 {
                return d;
            }
        }
        self.n / 2
    }
}

/// The S-map at one activation-radius, intersections memoised on demand.
///
/// ```
/// use kanerva::smap::{Lazy, goal};
/// let l = Lazy::new(256, 2_000, 112);
/// assert_eq!((l.n, l.r, l.avg, l.mirror), (256, 112, false, false));
/// assert_eq!(l.capacity(26, goal(256)), kanerva::smap::SMap::new(256, 2_000, 112).capacity(26, goal(256)));
/// ```
pub struct Lazy {
    /// The word-size in bits.
    pub n: usize,
    /// The number of hard-locations M, as a float.
    pub m: f64,
    /// The activation-radius r.
    pub r: usize,
    /// Noise from other patterns averaged over their distance to the read-address (d ~ Bin(n, 1/2)) instead of
    /// taken at d = n/2. See `noise_var`.
    pub avg: bool,
    /// Add the mirror field: the other stored patterns' mean pull toward the current state (see `mirror`).
    pub mirror: bool,
    memo: RefCell<Vec<Option<f64>>>,
    moments: RefCell<Option<(f64, f64, f64)>>,
}

impl Lazy {
    /// The plain S-map: no averaged noise and no mirror field.
    ///
    /// ```
    /// let l = kanerva::smap::Lazy::new(256, 2_000, 112);
    /// assert!(!l.avg && !l.mirror);
    /// ```
    pub fn new(n: usize, m: usize, r: usize) -> Lazy {
        Lazy { n, m: m as f64, r, avg: false, mirror: false, memo: RefCell::new(vec![None; n + 1]), moments: RefCell::new(None) }
    }
    /// The same map with the noise of other patterns averaged over their distance to the read-address.
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let (plain, avg) = (Lazy::new(256, 2_000, 112), Lazy::averaged(256, 2_000, 112));
    /// assert!(avg.avg && !avg.mirror);
    /// assert!(avg.noise_var(20) > plain.noise_var(20)); // averaging over distances adds noise here
    /// assert!(avg.capacity(26, goal(256)) <= plain.capacity(26, goal(256)));
    /// ```
    pub fn averaged(n: usize, m: usize, r: usize) -> Lazy {
        let mut l = Lazy::new(n, m, r);
        l.avg = true;
        l
    }
    /// Averaged noise plus the mirror field (SDMRADIUS, found after the controls run).
    ///
    /// ```
    /// let m = kanerva::smap::Lazy::mirrored(256, 2_000, 112);
    /// assert!(m.avg && m.mirror);
    /// assert!(m.mirror_field(20) > 0.0);
    /// ```
    pub fn mirrored(n: usize, m: usize, r: usize) -> Lazy {
        let mut l = Lazy::averaged(n, m, r);
        l.mirror = true;
        l
    }
    /// (E\[I(d)\], E\[I(d)^2\], E[I(d)(1 - 2d/n)]) over d ~ Bin(n, 1/2): moments of the fraction of space a random
    /// other pattern shares with the current state. E\[I\] = p^2 exactly; E\[I^2\] >= p^4; the third is the mean
    /// alignment of that pattern with the state, weighted by how much it shares, and is positive because I(d)
    /// falls with d (nearer patterns share more hard-locations).
    ///
    /// ```
    /// use kanerva::{smap::Lazy, theory::ball};
    /// let l = Lazy::new(256, 2_000, 112);
    /// let (e1, e2, em) = l.moments();
    /// let p = ball(256, 112);
    /// assert!((e1 - p * p).abs() < 1e-6); // E[I] = p^2
    /// assert!(e2 >= p.powi(4) && em > 0.0);
    /// ```
    pub fn moments(&self) -> (f64, f64, f64) {
        if let Some(v) = *self.moments.borrow() {
            return v;
        }
        let n = self.n;
        let lc: Vec<f64> = {
            let mut lg = vec![0.0f64; n + 1];
            for k in 1..=n {
                lg[k] = lg[k - 1] + (k as f64).ln();
            }
            (0..=n).map(|k| lg[n] - lg[k] - lg[n - k] - n as f64 * 2f64.ln()).collect()
        };
        let (mut e1, mut e2, mut em) = (0.0, 0.0, 0.0);
        for d in 0..=n {
            let w = lc[d].exp();
            if w < 1e-18 {
                continue;
            }
            let i = self.inter(d);
            e1 += w * i;
            e2 += w * i * i;
            em += w * i * (1.0 - 2.0 * d as f64 / n as f64);
        }
        *self.moments.borrow_mut() = Some((e1, e2, em));
        (e1, e2, em)
    }
    /// I(d): fraction of all addresses within r of two points d bits apart.
    ///
    /// ```
    /// let l = kanerva::smap::Lazy::new(256, 2_000, 112);
    /// assert_eq!(l.inter(0), kanerva::theory::ball(256, 112)); // a point shares its whole ball with itself
    /// assert!(l.inter(26) > l.inter(128));
    /// ```
    pub fn inter(&self, d: usize) -> f64 {
        let d = d.min(self.n);
        if let Some(v) = self.memo.borrow()[d] {
            return v;
        }
        let v = intersection(self.n, self.r, d);
        self.memo.borrow_mut()[d] = Some(v);
        v
    }
    /// Expected shared hard-locations of a read-address d bits from its pattern.
    ///
    /// ```
    /// let l = kanerva::smap::Lazy::new(256, 2_000, 112);
    /// assert!((l.shared(26) - 2_000.0 * l.inter(26)).abs() < 1e-9); // about 23 hard-locations
    /// ```
    pub fn shared(&self, d: usize) -> f64 {
        self.m * self.inter(d)
    }
    /// Variance the T - 1 other stored patterns add to one bit's vote: (T-1)(M I_o + (M I_o)^2) with
    /// I_o = I(n/2) (Bricken-Pehlevan Eq. 25), or with `avg` (T-1)(M E\[I\] + M^2 E\[I^2\]).
    ///
    /// ```
    /// let l = kanerva::smap::Lazy::new(256, 2_000, 112);
    /// assert_eq!(l.noise_var(1), 0.0); // one stored pattern: no other patterns to add noise
    /// assert!(l.noise_var(20) > l.noise_var(10));
    /// ```
    pub fn noise_var(&self, t: usize) -> f64 {
        let k = (t.max(1) - 1) as f64;
        if self.avg {
            let (e1, e2, _) = self.moments();
            return k * (self.m * e1 + self.m * self.m * e2);
        }
        let io = self.m * self.inter(self.n / 2);
        k * (io + io * io)
    }
    /// Bricken-Pehlevan Eq. 25.
    ///
    /// ```
    /// use kanerva::smap::{Lazy, SMap};
    /// let (l, s) = (Lazy::new(256, 2_000, 112), SMap::new(256, 2_000, 112));
    /// assert!((l.snr(26, 20) - s.snr(26, 20)).abs() < 1e-12); // the plain map is the same as SMap
    /// ```
    pub fn snr(&self, d: usize, t: usize) -> f64 {
        let sig = self.shared(d);
        let v = sig + self.noise_var(t);
        if v <= 0.0 {
            0.0
        } else {
            sig / v.sqrt()
        }
    }
    /// The mirror field mu = (T-1) M E[I(d)(1 - 2d/n)]: summed over the other stored patterns, the
    /// hard-locations the current state wakes vote for the state itself by mu per bit on average (0 if off).
    ///
    /// ```
    /// use kanerva::smap::Lazy;
    /// assert_eq!(Lazy::new(256, 2_000, 112).mirror_field(20), 0.0); // off unless mirrored
    /// let m = Lazy::mirrored(256, 2_000, 112);
    /// assert_eq!(m.mirror_field(1), 0.0); // no other patterns
    /// assert!(m.mirror_field(20) > 0.0);
    /// ```
    pub fn mirror_field(&self, t: usize) -> f64 {
        if !self.mirror {
            return 0.0;
        }
        (t.max(1) - 1) as f64 * self.m * self.moments().2
    }
    /// Next distance after one read from distance d, the target's own vote being s (its expected value
    /// M I(d), or a realised count), the rest noise with variance `var`: with no mirror n Phi(-s/sigma);
    /// with it, the d bits the state has wrong see s - mu and the n - d it has right see s + mu.
    ///
    /// ```
    /// let l = kanerva::smap::Lazy::new(256, 2_000, 112);
    /// let s = l.shared(26);
    /// let v = s + l.noise_var(20);
    /// assert_eq!(l.step(26, s, v, 20), l.next(26, 20)); // next is step at the expected vote
    /// assert_eq!(l.step(26, 0.0, 0.0, 20), 26.0);       // no vote and no noise: the read does not move
    /// ```
    pub fn step(&self, d: usize, s: f64, var: f64, t: usize) -> f64 {
        let n = self.n as f64;
        if var <= 0.0 {
            return if s > 0.0 { 0.0 } else { d as f64 };
        }
        let sd = var.sqrt();
        if !self.mirror {
            return n * phi(-s / sd);
        }
        let mu = self.mirror_field(t);
        d as f64 * phi(-(s - mu) / sd) + (n - d as f64) * phi(-(s + mu) / sd)
    }
    /// The expected next distance from distance `d` with `t` stored patterns; with the mirror field on it goes through `step`.
    ///
    /// ```
    /// let l = kanerva::smap::Lazy::new(256, 2_000, 112);
    /// assert!(l.next(26, 20) < 26.0);
    /// assert!(l.next(26, 5_000) > 26.0); // overloaded: the read moves away
    /// ```
    pub fn next(&self, d: usize, t: usize) -> f64 {
        if !self.mirror {
            return self.n as f64 * phi(-self.snr(d, t));
        }
        let s = self.shared(d);
        self.step(d, s, s + self.noise_var(t), t)
    }
    /// The S-map from d0 reaches a distance <= goal within 50 steps.
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// assert!(l.converges(26.0, 20, goal(256)));
    /// assert!(!l.converges(26.0, 5_000, goal(256)));
    /// ```
    pub fn converges(&self, d0: f64, t: usize, goal: f64) -> bool {
        let mut d = d0;
        for _ in 0..50 {
            if d <= goal {
                return true;
            }
            let nd = self.next(d.round() as usize, t);
            if nd >= d {
                return false;
            }
            d = nd;
        }
        d <= goal
    }
    /// Largest T whose S-map from d0 reaches goal (0 if even T = 1 does not).
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// let cap = l.capacity(26, goal(256));
    /// assert!(l.converges(26.0, cap, goal(256)) && !l.converges(26.0, cap + 1, goal(256)));
    /// assert!(l.capacity(51, goal(256)) < cap); // more address-noise, fewer patterns
    /// ```
    pub fn capacity(&self, d0: usize, goal: f64) -> usize {
        largest_t(|t| self.converges(d0 as f64, t, goal))
    }
    /// Smallest d at which the map stops moving d toward 0 (the unstable fixed point), scanning upward.
    ///
    /// Read the result with the load in mind. Under light load the first such d is the unstable fixed point,
    /// Kanerva's critical-distance; a lightly loaded map pulls in from every distance, so the scan returns n/2.
    /// Under heavy load a read that starts next to its pattern drifts out to a residual error, so the map already
    /// fails to move d = 1 toward 0 and the scan returns 1, although reads from farther out may still settle at
    /// that residual. The value is then "nothing is pulled all the way in", not "nothing is pulled in".
    ///
    /// ```
    /// let l = kanerva::smap::Lazy::new(256, 2_000, 112);
    /// assert!(l.critical(20) > l.critical(80)); // more stored patterns, a smaller critical-distance
    /// ```
    pub fn critical(&self, t: usize) -> usize {
        for d in 1..=self.n / 2 {
            if self.next(d, t) >= d as f64 {
                return d;
            }
        }
        self.n / 2
    }
    /// Probability that the iterated read from d0 converges, with the first read conditioned on the
    /// realised number of shared hard-locations k ~ Poisson(M I(d0)): SNR_1 = k / sqrt(noise), d_1 = n Phi(-SNR_1),
    /// then the plain S-map from d_1. k = 0 is a failure (no hard-location holds the pattern for the read-address).
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// assert_eq!(l.p_converge(2, 20, goal(256)), 1.0); // already inside the goal
    /// let p = l.p_converge(26, 20, goal(256));
    /// assert!(p > 0.99 && p <= 1.0);
    /// assert!(l.p_converge(26, 200, goal(256)) < p);
    /// ```
    pub fn p_converge(&self, d0: usize, t: usize, goal: f64) -> f64 {
        if d0 as f64 <= goal {
            return 1.0;
        }
        let lam = self.shared(d0);
        let nv = self.noise_var(t);
        let kmax = (lam + 12.0 * lam.sqrt() + 30.0) as usize;
        let mut lp = -lam; // log Poisson(0)
        let mut p = 0.0;
        for k in 0..=kmax {
            if k > 0 {
                lp += lam.ln() - (k as f64).ln();
            }
            // k = 0 without the mirror gives n/2: no hard-location holds the pattern, a failure
            let d1 = self.step(d0, k as f64, nv, t);
            if d1 < d0 as f64 && self.converges(d1, t, goal) {
                p += lp.exp();
            }
        }
        p.min(1.0)
    }
    /// The convergence probability for a read-address made by flipping each bit with probability `address_noise` (so its
    /// distance is Bin(n, address_noise), not exactly address_noise n): p_converge averaged over that distance.
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// let p = l.p_converge_flips(0.1, 20, goal(256));
    /// assert!(p > 0.99 && p <= 1.0);
    /// assert!(l.p_converge_flips(0.3, 20, goal(256)) < p);
    /// ```
    pub fn p_converge_flips(&self, address_noise: f64, t: usize, goal: f64) -> f64 {
        let n = self.n as f64;
        let (mu, sd) = (n * address_noise, (n * address_noise * (1.0 - address_noise)).sqrt());
        let (lo, hi) = ((mu - 6.0 * sd).floor().max(0.0) as usize, (mu + 6.0 * sd).ceil().min(n) as usize);
        let lw = |d: usize| -> f64 {
            let mut lg = 0.0;
            for k in 1..=self.n {
                lg += (k as f64).ln();
            }
            let lf = |k: usize| (1..=k).map(|x| (x as f64).ln()).sum::<f64>();
            lg - lf(d) - lf(self.n - d) + d as f64 * address_noise.ln() + (n - d as f64) * (1.0 - address_noise).ln()
        };
        (lo..=hi).map(|d| lw(d).exp() * self.p_converge(d, t, goal)).sum::<f64>().min(1.0)
    }
    /// Largest T whose flip-address-noise convergence probability is at least q.
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// let t = l.f_capacity(0.1, goal(256), 0.9);
    /// assert!(l.p_converge_flips(0.1, t, goal(256)) >= 0.9);
    /// assert!(l.p_converge_flips(0.1, t + 1, goal(256)) < 0.9);
    /// ```
    pub fn f_capacity(&self, address_noise: f64, goal: f64, q: f64) -> usize {
        largest_t(|t| self.p_converge_flips(address_noise, t, goal) >= q)
    }
    /// Largest T whose Poisson-conditioned convergence probability from d0 is at least q.
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// let t = l.p_capacity(26, goal(256), 0.9);
    /// assert!(t > 0 && t <= l.capacity(26, goal(256))); // asking for 90% recall costs capacity
    /// ```
    pub fn p_capacity(&self, d0: usize, goal: f64, q: f64) -> usize {
        largest_t(|t| self.p_converge(d0, t, goal) >= q)
    }
}

/// A Poisson draw (Knuth for small means, a rounded normal above 40).
fn poisson(lam: f64, r: &mut Rng) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    if lam > 40.0 {
        return (lam + lam.sqrt() * r.normal()).round().max(0.0);
    }
    let l = (-lam).exp();
    let (mut k, mut p) = (0.0, 1.0);
    loop {
        p *= r.unit();
        if p <= l {
            return k;
        }
        k += 1.0;
    }
}

impl Lazy {
    /// SDMRADIUS post-hoc predictor RACE: the other stored patterns interfere as WHOLE patterns. Traced
    /// failures at 40% address-noise (M 10^5, r 107, T 2 and 5; M 10^6, r 104, T 5) were all captures by one rival
    /// or a mixture of three, decided by one number: the target's shared-location count minus the best
    /// rival's (every read-address with the rival ahead failed, every read-address with the target 4 ahead recalled). So: the
    /// read-address sits d0 ~ Bin(n, address_noise) from its pattern and shares k ~ Poisson(M I(d0)) hard-locations with it; rival mu
    /// sits d_mu ~ Bin(n, 1/2) away and shares k_mu ~ Poisson(M I(d_mu)). The read fails if a rival shares
    /// more (a tie counts half); otherwise its bits see k plus the rivals' summed vote (mean sum k_mu a_mu
    /// toward the read-address, a_mu = 1 - 2 d_mu / n; variance sum k_mu^2 (1 - a_mu^2)) and later reads follow the
    /// averaged S-map. Rivals are sampled one by one up to 20,000 of them (above that, their mean field).
    /// Returns the fraction of `samples` sampled reads that converge.
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// let p = l.p_converge_race(0.1, 20, goal(256), 200, 1);
    /// assert!((0.0..=1.0).contains(&p));
    /// assert_eq!(p, l.p_converge_race(0.1, 20, goal(256), 200, 1)); // the same seed gives the same estimate
    /// ```
    pub fn p_converge_race(&self, address_noise: f64, t: usize, goal: f64, samples: usize, seed: u64) -> f64 {
        let n = self.n as f64;
        let mut rr = Rng::new(seed ^ 0xC0E7);
        let later = Lazy::averaged(self.n, self.m as usize, self.r);
        let rivals = t.max(1) - 1;
        let mut ok = 0.0;
        for _ in 0..samples {
            let d0 = (0..self.n).filter(|_| rr.unit() < address_noise).count();
            if d0 as f64 <= goal {
                ok += 1.0;
                continue;
            }
            let k = poisson(self.shared(d0), &mut rr);
            let (mut mean, mut var, mut best) = (0.0, 0.0, 0.0f64);
            if rivals <= 20_000 {
                for _ in 0..rivals {
                    let dm = ((n / 2.0 + (n / 4.0).sqrt() * rr.normal()).round().max(0.0) as usize).min(self.n);
                    let km = poisson(self.shared(dm), &mut rr);
                    let a = 1.0 - 2.0 * dm as f64 / n;
                    mean += km * a;
                    var += km * km * (1.0 - a * a);
                    best = best.max(km);
                }
            } else {
                let (e1, e2, em) = self.moments();
                mean = rivals as f64 * self.m * em;
                var = rivals as f64 * (self.m * e1 + self.m * self.m * e2);
            }
            let weight = if k < best || k == 0.0 {
                0.0
            } else if k == best {
                0.5
            } else {
                1.0
            };
            if weight == 0.0 {
                continue;
            }
            let d1 = if var <= 0.0 {
                0.0
            } else {
                let sd = var.sqrt();
                d0 as f64 * phi(-(k - mean) / sd) + (n - d0 as f64) * phi(-(k + mean) / sd)
            };
            if d1 < d0 as f64 && later.converges(d1, t, goal) {
                ok += weight;
            }
        }
        ok / samples.max(1) as f64
    }
    /// Largest T whose RACE convergence probability is at least q (common random numbers across T).
    ///
    /// ```
    /// use kanerva::smap::{Lazy, goal};
    /// let l = Lazy::new(256, 2_000, 112);
    /// let t = l.r_capacity(0.1, goal(256), 0.9, 100);
    /// assert!(t > 0);
    /// assert!(l.p_converge_race(0.1, t, goal(256), 100, 77) >= 0.9);
    /// ```
    pub fn r_capacity(&self, address_noise: f64, goal: f64, q: f64, samples: usize) -> usize {
        largest_t(|t| self.p_converge_race(address_noise, t, goal, samples, 77) >= q)
    }
}

/// The doubling searches (`SMap::capacity`, `largest_t`) stop at 2^40 on a 64-bit target and at a quarter of the
/// address space on a 32-bit one (wasm32), where 2^40 does not fit in a `usize`. On 64-bit the cap is exactly the
/// 2^40 it always was, so every output there is unchanged.
///
/// ```
/// #[cfg(target_pointer_width = "64")]
/// assert_eq!(kanerva::smap::SEARCH_CAP, 1 << 40);
/// assert!(kanerva::smap::SEARCH_CAP.checked_mul(2).is_some());
/// ```
pub const SEARCH_CAP: usize = {
    let cap = 1u64 << 40;
    let quarter = (usize::MAX / 4) as u64;
    (if cap < quarter { cap } else { quarter }) as usize
};

/// Largest t >= 1 with ok(t) (ok assumed monotone, true then false); 0 if ok(1) is false.
///
/// ```
/// use kanerva::smap::largest_t;
/// assert_eq!(largest_t(|t| t <= 37), 37);
/// assert_eq!(largest_t(|_| false), 0);
/// ```
pub fn largest_t<F: Fn(usize) -> bool>(ok: F) -> usize {
    if !ok(1) {
        return 0;
    }
    let mut hi = 1usize;
    while hi < SEARCH_CAP && ok(hi * 2) {
        hi *= 2;
    }
    let (mut lo, mut hi) = (hi, hi * 2);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The S-map goal used throughout: final overlap >= 0.95, i.e. distance <= 0.025 n.
///
/// ```
/// assert_eq!(kanerva::smap::goal(256), 6.4); // 2.5% of the bits: overlap at least 0.95
/// ```
pub fn goal(n: usize) -> f64 {
    0.025 * n as f64
}

/// The activation-radius in \[lo, hi\] whose plain S-map capacity from a read-address at fractional address-noise `address_noise` is largest
/// (ties to the smaller activation-radius). Returns (activation-radius, S-map capacity). This is the critical-distance-optimal
/// activation-radius for target address-noise address_noise: the activation-radius at which address_noise n stays inside the critical-distance up to the
/// largest load.
///
/// ```
/// use kanerva::smap::{radius_for_address_noise, search_window, Lazy, goal};
/// let (lo, hi) = search_window(256, 2_000);
/// let (r, cap) = radius_for_address_noise(256, 2_000, 0.1, lo, hi);
/// assert!((lo..=hi).contains(&r));
/// assert_eq!(cap, Lazy::new(256, 2_000, r).capacity(26, goal(256)));
/// ```
pub fn radius_for_address_noise(n: usize, m: usize, address_noise: f64, lo: usize, hi: usize) -> (usize, usize) {
    let d0 = (address_noise * n as f64).round() as usize;
    let mut best = (lo, 0usize);
    for r in lo..=hi {
        let c = Lazy::new(n, m, r).capacity(d0, goal(n));
        if c > best.1 {
            best = (r, c);
        }
    }
    best
}

/// The same choice under the Poisson-conditioned prediction at recall probability q.
///
/// ```
/// use kanerva::smap::{radius_for_address_noise_poisson, search_window};
/// let (lo, hi) = search_window(256, 2_000);
/// let (r, cap) = radius_for_address_noise_poisson(256, 2_000, 0.1, lo, hi, 0.9);
/// assert!((lo..=hi).contains(&r) && cap > 0);
/// ```
pub fn radius_for_address_noise_poisson(n: usize, m: usize, address_noise: f64, lo: usize, hi: usize, q: f64) -> (usize, usize) {
    let d0 = (address_noise * n as f64).round() as usize;
    let mut best = (lo, 0usize);
    for r in lo..=hi {
        let c = Lazy::new(n, m, r).p_capacity(d0, goal(n), q);
        if c > best.1 {
            best = (r, c);
        }
    }
    best
}

/// Bricken-Pehlevan's d*_CD: the activation-radius in \[lo, hi\] with the largest critical-distance at load t.
/// Returns (activation-radius, critical-distance).
///
/// ```
/// use kanerva::smap::{cd_optimal_radius, search_window, Lazy};
/// let (lo, hi) = search_window(256, 2_000);
/// let (r, cd) = cd_optimal_radius(256, 2_000, 20, lo, hi);
/// assert_eq!(cd, Lazy::new(256, 2_000, r).critical(20));
/// assert!((lo..=hi).all(|x| Lazy::new(256, 2_000, x).critical(20) <= cd));
/// ```
pub fn cd_optimal_radius(n: usize, m: usize, t: usize, lo: usize, hi: usize) -> (usize, usize) {
    let mut best = (lo, 0usize);
    for r in lo..=hi {
        let c = Lazy::new(n, m, r).critical(t);
        if c > best.1 {
            best = (r, c);
        }
    }
    best
}

/// An activation-radius search window around the SNR-optimal activation-radius for M patterns at design load M/20: the bottom is
/// a few bits under it, the top is the activation-radius waking 5% of hard-locations.
///
/// ```
/// let (lo, hi) = kanerva::smap::search_window(256, 2_000);
/// assert!(lo < hi);
/// assert_eq!(hi, kanerva::theory::radius_for(256, 0.05)); // the top activates 5% of hard-locations
/// ```
pub fn search_window(n: usize, m: usize) -> (usize, usize) {
    let p = (m as f64 * m as f64 / 10.0).powf(-1.0 / 3.0);
    let r0 = radius_for(n, p);
    (r0.saturating_sub(4), radius_for(n, 0.05).max(r0 + 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::ball;

    #[test]
    fn with_one_pattern_the_snr_is_the_root_of_the_shared_count() {
        // T = 1: no noise from other patterns, so SNR(d) = M I(d) / sqrt(M I(d)) = sqrt(M I(d))
        let s = SMap::new(256, 20_000, 104);
        for d in [0usize, 30, 77] {
            assert!((s.snr(d, 1) - (20_000.0 * intersection(256, 104, d)).sqrt()).abs() < 1e-12);
        }
        // and at d = 0, I(0) is the ball
        assert!((s.inter[0] - ball(256, 104)).abs() < 1e-15);
    }

    #[test]
    fn largest_t_by_hand() {
        assert_eq!(largest_t(|t| t <= 37), 37);
        assert_eq!(largest_t(|t| t <= 1), 1);
        assert_eq!(largest_t(|_| false), 0);
    }

    #[test]
    fn capacity_is_finite_and_a_crowded_store_does_not_converge() {
        let l = Lazy::new(256, 100_000, 103);
        let c = l.capacity(26, goal(256));
        assert!(c > 100 && c < 1_000_000, "{}", c);
        // negative control: far past capacity the map does not reach the goal
        assert!(!l.converges(26.0, c * 50, goal(256)));
        assert!(l.converges(26.0, c, goal(256)));
    }

    #[test]
    fn lazy_equals_the_full_map() {
        let (full, lazy) = (SMap::new(256, 30_000, 105), Lazy::new(256, 30_000, 105));
        for t in [1usize, 10, 300] {
            for d in [0usize, 26, 77, 128] {
                assert!((full.snr(d, t) - lazy.snr(d, t)).abs() < 1e-12);
            }
            assert_eq!(full.critical(t), lazy.critical(t));
        }
    }

    #[test]
    fn a_radius_for_more_damage_is_wider() {
        let (lo, hi) = search_window(256, 100_000);
        let (r10, _) = radius_for_address_noise(256, 100_000, 0.1, lo, hi);
        let (r30, _) = radius_for_address_noise(256, 100_000, 0.3, lo, hi);
        assert!(r30 > r10, "{} {}", r30, r10);
    }

    #[test]
    fn bricken_pehlevan_table_1_and_the_critical_distance_at_kanerva_s_point() {
        // Bricken and Pehlevan 2021, Table 1 (p. 38), canonical SDM n 1000, M 10^6, T 10^4: d*_SNR 447, d*_CD 448
        assert_eq!(radius_for(1000, (2e10f64).powf(-1.0 / 3.0)), 447);
        assert_eq!(cd_optimal_radius(1000, 1_000_000, 10_000, 440, 460).0, 448);
        // The critical-distance at r 451, T 10^4. Three numbers, recorded, not reconciled:
        // - this S-map: 165 (pinned here, so a change to the map is seen);
        // - Bricken and Pehlevan Fig. 17a, their reproduction of Kanerva 1988 Fig. 7.3: 188;
        // - measured on a real store (SETTLE lane SDMSCALE, runs/sdmscale/anchor_kanerva.txt): 204.3.
        // The S-map is the most pessimistic of the three; the cause of the gap is not established.
        assert_eq!(Lazy::new(1000, 1_000_000, 451).critical(10_000), 165);
    }
}
