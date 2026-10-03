//! The counting behind every SDM: binomial tables, the Hamming ball, the intersection of two balls, the
//! activation radius that wakes a given fraction of hard locations, and the standard normal CDF and its inverse.
//!
//! For n-bit addresses drawn uniformly, the distance from any fixed point to a random address is
//! Binomial(n, 1/2). So the fraction of hard locations inside a ball of activation radius r is
//! `ball(n, r) = P[Bin(n, 1/2) <= r]`, and the fraction inside two balls whose centres are d bits apart is
//! `intersection(n, r, d)` (Bricken and Pehlevan 2021, Eq. 2).

/// log C(n, k) for k = 0..=n (by summed logs of the factorials).
pub fn log_choose(n: usize) -> Vec<f64> {
    let mut lg = vec![0.0f64; n + 1];
    for k in 1..=n {
        lg[k] = lg[k - 1] + (k as f64).ln();
    }
    (0..=n).map(|k| lg[n] - lg[k] - lg[n - k]).collect()
}

/// Fraction of all addresses within activation radius r of one point: P[Bin(n, 1/2) <= r].
pub fn ball(n: usize, r: usize) -> f64 {
    let lc = log_choose(n);
    let l2 = n as f64 * 2f64.ln();
    (0..=r.min(n)).map(|k| (lc[k] - l2).exp()).sum()
}

/// Exact fraction of all addresses within r of BOTH of two points at Hamming distance d (Bricken and
/// Pehlevan 2021, Eq. 2): sum over b bits changed among the d differing positions and c among the rest.
pub fn intersection(n: usize, r: usize, d: usize) -> f64 {
    let (la, lb) = (log_choose(d), log_choose(n - d));
    let l2 = n as f64 * 2f64.ln();
    let mut s = 0.0;
    for b in 0..=d {
        for c in 0..=(n - d) {
            if b + c <= r && d - b + c <= r {
                s += (la[b] + lb[c] - l2).exp();
            } else if b + c > r && d - b + c > r {
                break;
            }
        }
    }
    s
}

/// Standard normal CDF (Abramowitz-Stegun 7.1.26 through erf, error under 1.5e-7).
pub fn phi(x: f64) -> f64 {
    let z = x.abs() / 2f64.sqrt();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-z * z).exp();
    if x >= 0.0 {
        0.5 * (1.0 + y)
    } else {
        0.5 * (1.0 - y)
    }
}

/// Inverse of `phi` by bisection on [-40, 40] (200 halvings).
pub fn phi_inv(p: f64) -> f64 {
    let (mut lo, mut hi) = (-40.0, 40.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if phi(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Smallest activation radius r with P[Bin(n, 1/2) <= r] >= p: the activation radius at which about a fraction p of hard locations wake.
/// This is the activation radius rule the in-pulls `sdm` and `sdmscale` families use.
pub fn radius_for(n: usize, p: f64) -> usize {
    let mut lg = vec![0.0f64; n + 1];
    for k in 1..=n {
        lg[k] = lg[k - 1] + (k as f64).ln();
    }
    let mut cdf = 0.0;
    for r in 0..=n {
        cdf += (lg[n] - lg[r] - lg[n - r] - n as f64 * 2f64.ln()).exp();
        if cdf >= p {
            return r;
        }
    }
    n
}

/// ln P(d = k) for d ~ Binomial(n, 1/2), k = 0..=n.
pub fn binom_log_pmf(n: usize) -> Vec<f64> {
    let mut lf = vec![0.0f64; n + 1];
    for k in 1..=n {
        lf[k] = lf[k - 1] + (k as f64).ln();
    }
    (0..=n).map(|k| lf[n] - lf[k] - lf[n - k] - n as f64 * std::f64::consts::LN_2).collect()
}

/// Smallest activation radius r with P(Bin(n, 1/2) <= r) >= fire, and that probability. The same activation radius as
/// `radius_for` (computed from `binom_log_pmf`); the soft machine uses this form because it also needs the
/// exact activated fraction.
pub fn hard_radius(n: usize, fire: f64) -> (usize, f64) {
    let lp = binom_log_pmf(n);
    let mut c = 0.0;
    for (r, l) in lp.iter().enumerate() {
        c += l.exp();
        if c >= fire {
            return (r, c);
        }
    }
    (n, 1.0)
}

/// P[Bin(n, q) = k] for k = 0..=n.
pub fn binom_pmf(n: usize, q: f64) -> Vec<f64> {
    let lc = log_choose(n);
    (0..=n)
        .map(|k| {
            if q <= 0.0 {
                return if k == 0 { 1.0 } else { 0.0 };
            }
            if q >= 1.0 {
                return if k == n { 1.0 } else { 0.0 };
            }
            (lc[k] + k as f64 * q.ln() + (n - k) as f64 * (1.0 - q).ln()).exp()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ball_and_intersection_by_brute_force_at_n_16() {
        // count all 2^16 addresses
        let (n, r, d) = (16usize, 6usize, 5usize);
        let (mut one, mut both) = (0u32, 0u32);
        for a in 0u32..(1 << 16) {
            let dx = a.count_ones() as usize;
            let dy = (a ^ 0b11111).count_ones() as usize;
            one += (dx <= r) as u32;
            both += (dx <= r && dy <= r) as u32;
        }
        assert!((ball(n, r) - one as f64 / 65536.0).abs() < 1e-12);
        assert!((intersection(n, r, d) - both as f64 / 65536.0).abs() < 1e-12);
        assert!((intersection(n, r, 0) - ball(n, r)).abs() < 1e-12);
    }

    #[test]
    fn intersection_falls_with_distance() {
        // negative control on the formula's direction: two far-apart centres share less of the space
        let v: Vec<f64> = (0..=64).map(|d| intersection(64, 26, d)).collect();
        for d in 1..=64 {
            assert!(v[d] <= v[d - 1] + 1e-15, "d {}", d);
        }
        assert!(v[40] < 0.1 * v[0]);
    }

    #[test]
    fn kanerva_and_default_radii() {
        assert_eq!(radius_for(256, 0.02), 112);
        assert_eq!(radius_for(1000, 0.001), 451); // Kanerva 1988's example: n 1000, p 0.001
    }

    #[test]
    fn both_radius_rules_agree_everywhere_tested() {
        for n in [64usize, 128, 256, 1000] {
            for p in [0.001, 0.01, 0.02, 0.05, 0.3] {
                assert_eq!(radius_for(n, p), hard_radius(n, p).0, "n {} p {}", n, p);
            }
        }
    }

    #[test]
    fn normal_cdf_and_inverse() {
        assert!((phi(0.0) - 0.5).abs() < 1e-9);
        assert!((phi(1.0) - 0.841344746).abs() < 1e-6);
        assert!((phi_inv(0.975) - 1.959964).abs() < 1e-4);
        assert!((phi(-2.0) + phi(2.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn binomial_pmf_sums_to_one_and_matches_by_hand() {
        let b = binom_pmf(4, 0.5);
        for (k, want) in [1.0, 4.0, 6.0, 4.0, 1.0].iter().enumerate() {
            assert!((b[k] - want / 16.0).abs() < 1e-12);
        }
        assert!((binom_pmf(300, 0.3).iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(binom_pmf(5, 0.0)[0], 1.0);
    }
}
