//! Capacity theory: the Bricken-Pehlevan signal-to-noise map (the S-map) for an iterated read, its
//! critical distance and capacity, and the choice of activation radius for the address-noise a read-address will carry.
//!
//! A read-address d bits from its stored pattern shares M I(d) hard locations with it (I from `theory::intersection`).
//! Those hard locations vote for the pattern; the T - 1 other stored patterns add noise. One read maps d to
//!
//!   SNR(d) = M I(d) / sqrt(M I(d) + (T - 1)(M I_o + (M I_o)^2)),   I_o = I(n/2),
//!   d_next = n Phi(-SNR(d)).
//!
//! Iterating the map from d0 says whether the read converges. `SMap` is the plain map with every I(d)
//! precomputed; `Lazy` computes I(d) on demand and adds three refinements measured in the SETTLE campaign:
//! noise averaged over the other patterns' distances (`averaged`), the mirror field (`mirrored`), the first
//! read conditioned on a Poisson number of shared hard locations (`p_converge`), flip add_address_noise (`p_converge_flips`),
//! and RACE, where rival patterns compete as whole patterns (`p_converge_race`).

use crate::rng::Rng;
use crate::theory::{intersection, phi, radius_for};
use std::cell::RefCell;

/// The Bricken-Pehlevan signal-to-noise S-map for an iterated read (wiki WIKI_SDR 85 section 2, 21 section 5):
/// SNR(d) = M I(d) / sqrt(M I(d) + (T-1)(M I_o + (M I_o)^2)), next d = n Phi(-SNR(d)).
pub struct SMap {
    pub n: usize,
    pub m: f64,
    /// I(d) for d = 0..=n.
    pub inter: Vec<f64>,
}

impl SMap {
    pub fn new(n: usize, m: usize, r: usize) -> SMap {
        SMap { n, m: m as f64, inter: (0..=n).map(|d| intersection(n, r, d)).collect() }
    }
    pub fn snr(&self, d: usize, t: usize) -> f64 {
        let sig = self.m * self.inter[d];
        let io = self.m * self.inter[self.n / 2];
        sig / (sig + (t.max(1) - 1) as f64 * (io + io * io)).sqrt()
    }
    /// Expected next distance.
    pub fn next(&self, d: usize, t: usize) -> f64 {
        self.n as f64 * phi(-self.snr(d, t))
    }
    /// Iterate from d0; true if it reaches a distance at most `goal` within 50 steps.
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
    pub fn capacity(&self, d0: usize, goal: f64) -> usize {
        if !self.converges(d0, 1, goal) {
            return 0;
        }
        let mut hi = 1usize;
        while self.converges(d0, hi * 2, goal) && hi < 1 << 40 {
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
    pub fn critical(&self, t: usize) -> usize {
        for d in 1..=self.n / 2 {
            if self.next(d, t) >= d as f64 {
                return d;
            }
        }
        self.n / 2
    }
}

/// The S-map at one activation radius, intersections memoised on demand.
pub struct Lazy {
    pub n: usize,
    pub m: f64,
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
    pub fn new(n: usize, m: usize, r: usize) -> Lazy {
        Lazy { n, m: m as f64, r, avg: false, mirror: false, memo: RefCell::new(vec![None; n + 1]), moments: RefCell::new(None) }
    }
    /// The same map with the noise of other patterns averaged over their distance to the read-address.
    pub fn averaged(n: usize, m: usize, r: usize) -> Lazy {
        let mut l = Lazy::new(n, m, r);
        l.avg = true;
        l
    }
    /// Averaged noise plus the mirror field (SDMRADIUS, found after the controls run).
    pub fn mirrored(n: usize, m: usize, r: usize) -> Lazy {
        let mut l = Lazy::averaged(n, m, r);
        l.mirror = true;
        l
    }
    /// (E[I(d)], E[I(d)^2], E[I(d)(1 - 2d/n)]) over d ~ Bin(n, 1/2): moments of the fraction of space a random
    /// other pattern shares with the current state. E[I] = p^2 exactly; E[I^2] >= p^4; the third is the mean
    /// alignment of that pattern with the state, weighted by how much it shares, and is positive because I(d)
    /// falls with d (nearer patterns share more hard locations).
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
    pub fn inter(&self, d: usize) -> f64 {
        let d = d.min(self.n);
        if let Some(v) = self.memo.borrow()[d] {
            return v;
        }
        let v = intersection(self.n, self.r, d);
        self.memo.borrow_mut()[d] = Some(v);
        v
    }
    /// Expected shared hard locations of a read-address d bits from its pattern.
    pub fn shared(&self, d: usize) -> f64 {
        self.m * self.inter(d)
    }
    /// Variance the T - 1 other stored patterns add to one bit's vote: (T-1)(M I_o + (M I_o)^2) with
    /// I_o = I(n/2) (Bricken-Pehlevan Eq. 25), or with `avg` (T-1)(M E[I] + M^2 E[I^2]).
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
    /// hard locations the current state wakes vote for the state itself by mu per bit on average (0 if off).
    pub fn mirror_field(&self, t: usize) -> f64 {
        if !self.mirror {
            return 0.0;
        }
        (t.max(1) - 1) as f64 * self.m * self.moments().2
    }
    /// Next distance after one read from distance d, the target's own vote being s (its expected value
    /// M I(d), or a realised count), the rest noise with variance `var`: with no mirror n Phi(-s/sigma);
    /// with it, the d bits the state has wrong see s - mu and the n - d it has right see s + mu.
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
    pub fn next(&self, d: usize, t: usize) -> f64 {
        if !self.mirror {
            return self.n as f64 * phi(-self.snr(d, t));
        }
        let s = self.shared(d);
        self.step(d, s, s + self.noise_var(t), t)
    }
    /// The S-map from d0 reaches a distance <= goal within 50 steps.
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
    pub fn capacity(&self, d0: usize, goal: f64) -> usize {
        largest_t(|t| self.converges(d0 as f64, t, goal))
    }
    /// Smallest d at which the map stops moving d toward 0 (the unstable fixed point), scanning upward.
    pub fn critical(&self, t: usize) -> usize {
        for d in 1..=self.n / 2 {
            if self.next(d, t) >= d as f64 {
                return d;
            }
        }
        self.n / 2
    }
    /// Probability that the iterated read from d0 converges, with the first read conditioned on the
    /// realised number of shared hard locations k ~ Poisson(M I(d0)): SNR_1 = k / sqrt(noise), d_1 = n Phi(-SNR_1),
    /// then the plain S-map from d_1. k = 0 is a failure (no hard location holds the pattern for the read-address).
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
            // k = 0 without the mirror gives n/2: no hard location holds the pattern, a failure
            let d1 = self.step(d0, k as f64, nv, t);
            if d1 < d0 as f64 && self.converges(d1, t, goal) {
                p += lp.exp();
            }
        }
        p.min(1.0)
    }
    /// The convergence probability for a read-address made by flipping each bit with probability `dmg` (so its
    /// distance is Bin(n, dmg), not exactly dmg n): p_converge averaged over that distance.
    pub fn p_converge_flips(&self, dmg: f64, t: usize, goal: f64) -> f64 {
        let n = self.n as f64;
        let (mu, sd) = (n * dmg, (n * dmg * (1.0 - dmg)).sqrt());
        let (lo, hi) = ((mu - 6.0 * sd).floor().max(0.0) as usize, (mu + 6.0 * sd).ceil().min(n) as usize);
        let lw = |d: usize| -> f64 {
            let mut lg = 0.0;
            for k in 1..=self.n {
                lg += (k as f64).ln();
            }
            let lf = |k: usize| (1..=k).map(|x| (x as f64).ln()).sum::<f64>();
            lg - lf(d) - lf(self.n - d) + d as f64 * dmg.ln() + (n - d as f64) * (1.0 - dmg).ln()
        };
        (lo..=hi).map(|d| lw(d).exp() * self.p_converge(d, t, goal)).sum::<f64>().min(1.0)
    }
    /// Largest T whose flip-address-noise convergence probability is at least q.
    pub fn f_capacity(&self, dmg: f64, goal: f64, q: f64) -> usize {
        largest_t(|t| self.p_converge_flips(dmg, t, goal) >= q)
    }
    /// Largest T whose Poisson-conditioned convergence probability from d0 is at least q.
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
    /// failures at 40% add_address_noise (M 10^5, r 107, T 2 and 5; M 10^6, r 104, T 5) were all captures by one rival
    /// or a mixture of three, decided by one number: the target's shared-location count minus the best
    /// rival's (every read-address with the rival ahead failed, every read-address with the target 4 ahead recalled). So: the
    /// read-address sits d0 ~ Bin(n, dmg) from its pattern and shares k ~ Poisson(M I(d0)) hard locations with it; rival mu
    /// sits d_mu ~ Bin(n, 1/2) away and shares k_mu ~ Poisson(M I(d_mu)). The read fails if a rival shares
    /// more (a tie counts half); otherwise its bits see k plus the rivals' summed vote (mean sum k_mu a_mu
    /// toward the read-address, a_mu = 1 - 2 d_mu / n; variance sum k_mu^2 (1 - a_mu^2)) and later reads follow the
    /// averaged S-map. Rivals are sampled one by one up to 20,000 of them (above that, their mean field).
    /// Returns the fraction of `samples` sampled reads that converge.
    pub fn p_converge_race(&self, dmg: f64, t: usize, goal: f64, samples: usize, seed: u64) -> f64 {
        let n = self.n as f64;
        let mut rr = Rng::new(seed ^ 0xC0E7);
        let later = Lazy::averaged(self.n, self.m as usize, self.r);
        let rivals = t.max(1) - 1;
        let mut ok = 0.0;
        for _ in 0..samples {
            let d0 = (0..self.n).filter(|_| rr.unit() < dmg).count();
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
    pub fn r_capacity(&self, dmg: f64, goal: f64, q: f64, samples: usize) -> usize {
        largest_t(|t| self.p_converge_race(dmg, t, goal, samples, 77) >= q)
    }
}

/// Largest t >= 1 with ok(t) (ok assumed monotone, true then false); 0 if ok(1) is false.
pub fn largest_t<F: Fn(usize) -> bool>(ok: F) -> usize {
    if !ok(1) {
        return 0;
    }
    let mut hi = 1usize;
    while hi < 1 << 40 && ok(hi * 2) {
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
pub fn goal(n: usize) -> f64 {
    0.025 * n as f64
}

/// The activation radius in [lo, hi] whose plain S-map capacity from a read-address at fractional address-noise `dmg` is largest
/// (ties to the smaller activation radius). Returns (activation radius, S-map capacity). This is the critical-distance-optimal
/// activation radius for target address-noise dmg: the activation radius at which dmg n stays inside the critical distance up to the
/// largest load.
pub fn radius_for_address_noise(n: usize, m: usize, dmg: f64, lo: usize, hi: usize) -> (usize, usize) {
    let d0 = (dmg * n as f64).round() as usize;
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
pub fn radius_for_address_noise_poisson(n: usize, m: usize, dmg: f64, lo: usize, hi: usize, q: f64) -> (usize, usize) {
    let d0 = (dmg * n as f64).round() as usize;
    let mut best = (lo, 0usize);
    for r in lo..=hi {
        let c = Lazy::new(n, m, r).p_capacity(d0, goal(n), q);
        if c > best.1 {
            best = (r, c);
        }
    }
    best
}

/// Bricken-Pehlevan's d*_CD: the activation radius in [lo, hi] with the largest critical distance at load t.
/// Returns (activation radius, critical distance).
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

/// A activation radius search window around the SNR-optimal activation radius for M patterns at design load M/20: the bottom is
/// a few bits under it, the top is the activation radius waking 5% of hard locations.
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
}
