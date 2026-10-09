//! The counting behind every SDM: binomial tables, the Hamming ball, the intersection of two balls, the
//! activation-radius that wakes a given fraction of hard-locations, and the standard normal CDF and its inverse.
//!
//! For n-bit addresses drawn uniformly, the distance from any fixed point to a random address is
//! Binomial(n, 1/2). So the fraction of hard-locations inside a ball of activation-radius r is
//! `ball(n, r) = P[Bin(n, 1/2) <= r]`, and the fraction inside two balls whose centres are d bits apart is
//! `intersection(n, r, d)` (Bricken and Pehlevan 2021, Eq. 2).

/// log C(n, k) for k = 0..=n (by summed logs of the factorials).
///
/// ```
/// use kanerva::theory::log_choose;
/// let c: Vec<f64> = log_choose(4).iter().map(|l| l.exp().round()).collect();
/// assert_eq!(c, vec![1.0, 4.0, 6.0, 4.0, 1.0]);
/// ```
pub fn log_choose(n: usize) -> Vec<f64> {
    let mut lg = vec![0.0f64; n + 1];
    for k in 1..=n {
        lg[k] = lg[k - 1] + (k as f64).ln();
    }
    (0..=n).map(|k| lg[n] - lg[k] - lg[n - k]).collect()
}

/// Fraction of all addresses within activation-radius r of one point: P[Bin(n, 1/2) <= r].
///
/// ```
/// use kanerva::theory::ball;
/// assert!((ball(4, 1) - 5.0 / 16.0).abs() < 1e-12); // C(4,0) + C(4,1) of 16 addresses
/// assert!((ball(256, 256) - 1.0).abs() < 1e-9);
/// assert!((ball(256, 107) - 0.00513).abs() < 0.00001);
/// ```
pub fn ball(n: usize, r: usize) -> f64 {
    let lc = log_choose(n);
    let l2 = n as f64 * 2f64.ln();
    (0..=r.min(n)).map(|k| (lc[k] - l2).exp()).sum()
}

/// Exact fraction of all addresses within r of BOTH of two points at Hamming distance d (Bricken and
/// Pehlevan 2021, Eq. 2): sum over b bits changed among the d differing positions and c among the rest.
///
/// ```
/// use kanerva::theory::{ball, intersection};
/// assert!((intersection(256, 110, 0) - ball(256, 110)).abs() < 1e-12); // a point shares its whole ball with itself
/// assert!(intersection(256, 110, 20) > intersection(256, 110, 60)); // farther apart, less shared
/// ```
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
///
/// ```
/// use kanerva::theory::phi;
/// assert!((phi(0.0) - 0.5).abs() < 1e-7);
/// assert!((phi(1.96) - 0.975).abs() < 1e-4);
/// ```
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

/// Inverse of `phi` by bisection on \[-40, 40\] (200 halvings).
///
/// ```
/// use kanerva::theory::{phi, phi_inv};
/// assert!((phi_inv(0.975) - 1.96).abs() < 1e-3);
/// assert!((phi(phi_inv(0.3)) - 0.3).abs() < 1e-9);
/// ```
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

/// Smallest activation-radius r with P[Bin(n, 1/2) <= r] >= p: the activation-radius at which about a fraction p of hard-locations wake.
/// This is the activation-radius rule the in-pulls `sdm` and `sdmscale` families use.
///
/// ```
/// use kanerva::theory::{ball, radius_for};
/// assert_eq!(radius_for(1000, 0.001), 451); // Kanerva 1988's example, n 1000, p about 1/1000 (as Jaeckel 1989 p. 7 quotes it)
/// assert_eq!(radius_for(1000, 0.000368), 447); // Kanerva 1993's sample memory: "H = 447" (P. 10)
/// let r = radius_for(256, 0.005);
/// assert_eq!(r, 107);
/// assert!(ball(256, r) >= 0.005 && ball(256, r - 1) < 0.005);
/// ```
pub fn radius_for(n: usize, p: f64) -> usize {
    // one code path for the radius: the same terms, summed in the same order, as hard_radius
    hard_radius(n, p).0
}

/// ln P(d = k) for d ~ Binomial(n, 1/2), k = 0..=n.
///
/// ```
/// use kanerva::theory::binom_log_pmf;
/// let lp = binom_log_pmf(4);
/// assert!((lp[2].exp() - 6.0 / 16.0).abs() < 1e-12);
/// assert!((binom_log_pmf(256).iter().map(|l| l.exp()).sum::<f64>() - 1.0).abs() < 1e-9);
/// ```
pub fn binom_log_pmf(n: usize) -> Vec<f64> {
    let mut lf = vec![0.0f64; n + 1];
    for k in 1..=n {
        lf[k] = lf[k - 1] + (k as f64).ln();
    }
    (0..=n).map(|k| lf[n] - lf[k] - lf[n - k] - n as f64 * std::f64::consts::LN_2).collect()
}

/// Smallest activation-radius r with P(Bin(n, 1/2) <= r) >= `activation_probability`, and that probability. The same activation-radius as
/// `radius_for` (computed from `binom_log_pmf`); the soft machine uses this form because it also needs the
/// exact activated fraction.
///
/// ```
/// use kanerva::theory::{hard_radius, radius_for};
/// let (r, p) = hard_radius(256, 0.05);
/// assert_eq!(r, radius_for(256, 0.05));
/// assert!(p >= 0.05);
/// ```
pub fn hard_radius(n: usize, activation_probability: f64) -> (usize, f64) {
    // the terms of binom_log_pmf, summed in order, stopping at the radius (no table is built)
    let mut lf = vec![0.0f64; n + 1];
    for k in 1..=n {
        lf[k] = lf[k - 1] + (k as f64).ln();
    }
    let mut c = 0.0;
    for r in 0..=n {
        c += (lf[n] - lf[r] - lf[n - r] - n as f64 * std::f64::consts::LN_2).exp();
        if c >= activation_probability {
            return (r, c);
        }
    }
    (n, 1.0)
}

/// P[Bin(n, q) = k] for k = 0..=n.
///
/// ```
/// use kanerva::theory::binom_pmf;
/// assert_eq!(binom_pmf(2, 0.5).iter().map(|x| (x * 4.0).round()).collect::<Vec<_>>(), vec![1.0, 2.0, 1.0]);
/// assert!((binom_pmf(100, 0.3).iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// assert_eq!(binom_pmf(3, 0.0), vec![1.0, 0.0, 0.0, 0.0]);
/// ```
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

/// Kanerva's best probability of activation for M hard-locations holding T words read from their exact
/// addresses: p = (2 M T)^(-1/3), the p that maximises the signal-to-noise ratio
/// rho^2 = pM / (1 + pT(1 + p^2 M)) (Kanerva 1993, P. 15; Bricken and Pehlevan 2021, p*_SNR).
///
/// ```
/// use kanerva::theory::{optimal_activation_probability, radius_for};
/// let p = optimal_activation_probability(1_000_000, 10_000);
/// assert!((p - 0.000368).abs() < 0.0000005); // Kanerva 1993's sample memory: "optimal p is 0.000368" (P. 10)
/// assert_eq!(radius_for(1000, p), 447);       // and its radius, H = 447
/// ```
pub fn optimal_activation_probability(m: usize, t: usize) -> f64 {
    (2.0 * m as f64 * t.max(1) as f64).powf(-1.0 / 3.0)
}

/// Kanerva's asymptotic memory capacity tau = T_max / M as M grows without bound, at bit-fidelity `fidelity`
/// (the probability a stored bit is read back correctly from its exact address): tau = 1 / [Phi^-1(fidelity)]^2
/// (Kanerva 1993, P. 16).
///
/// ```
/// use kanerva::theory::asymptotic_capacity;
/// assert!((asymptotic_capacity(0.999) - 0.105).abs() < 0.0005); // "the asymptotic capacity is tau = 0.105" (P. 17)
/// assert!((asymptotic_capacity(0.995) - 0.15).abs() < 0.001);   // the Hopfield net's 0.15 N "corresponds to 0.995"
/// ```
pub fn asymptotic_capacity(fidelity: f64) -> f64 {
    1.0 / phi_inv(fidelity).powi(2)
}

/// Kanerva's memory capacity tau = T_max / M of a memory with `m` hard-locations at bit-fidelity `fidelity`,
/// with the probability of activation at its optimum for that load, p = (2 tau M^2)^(-1/3): the tau solving
/// [Phi^-1(fidelity)]^2 = 1 / (1 / (pM) + tau (1 + p^2 M)) (Kanerva 1993, P. 16-17), found by bisection.
/// Exact read-addresses are assumed; read-addresses with address-noise hold fewer words.
///
/// ```
/// use kanerva::theory::{capacity_fraction, asymptotic_capacity};
/// // "the capacity of the million-location sample memory is 0.096" (Kanerva 1993, P. 17)
/// assert!((capacity_fraction(1_000_000, 0.999) - 0.096).abs() < 0.0005);
/// // a larger memory comes closer to the asymptote
/// assert!(capacity_fraction(1_000_000_000, 0.999) > capacity_fraction(1_000_000, 0.999));
/// assert!(capacity_fraction(1_000_000_000, 0.999) < asymptotic_capacity(0.999));
/// ```
pub fn capacity_fraction(m: usize, fidelity: f64) -> f64 {
    let z2 = phi_inv(fidelity).powi(2);
    let m = m as f64;
    let rho2 = |tau: f64| {
        let p = (2.0 * tau * m * m).powf(-1.0 / 3.0);
        1.0 / (1.0 / (p * m) + tau * (1.0 + p * p * m))
    };
    let (mut lo, mut hi) = (1e-12, 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if rho2(mid) > z2 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
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
        assert_eq!(radius_for(1000, 0.001), 451); // Kanerva 1988's example, as Jaeckel 1989 p. 7 quotes it
    }

    #[test]
    fn both_radius_rules_agree_everywhere_tested() {
        for n in [64usize, 128, 256, 1000] {
            for p in [0.001, 0.01, 0.02, 0.05, 0.3] {
                assert_eq!(radius_for(n, p), hard_radius(n, p).0, "n {} p {}", n, p);
                // the share it returns is the binom_log_pmf terms summed in order, to the last bit
                let (r, c) = hard_radius(n, p);
                let want: f64 = binom_log_pmf(n)[..=r].iter().fold(0.0, |a, l| a + l.exp());
                assert_eq!(c.to_bits(), want.to_bits(), "n {} p {}", n, p);
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

    // ---- checks against the papers on disk (wikis/WIKI_SDR/papers/) ----

    #[test]
    fn kanerva_1993_sample_memory() {
        // K1993 P. 10: "optimal p is 0.000368 ... Radius H = 446 captures 354 locations, and H = 447 captures 445
        // locations, on the average", M = 1,000,000, N = 1,000; and D = N - 2H gives 108 and 106.
        assert_eq!(radius_for(1000, 0.000368), 447);
        assert_eq!((1e6 * ball(1000, 447)).round(), 445.0);
        assert_eq!((1e6 * ball(1000, 446)).round(), 354.0);
        assert_eq!((1000 - 2 * 447, 1000 - 2 * 446), (106, 108));
        // K1993 also quotes p = 0.000445 for H = 447; ball(1000, 447) is 0.00044499, just under it, so the
        // smallest radius reaching 0.000445 exactly is 448 (the quoted p is the rounded share, not a threshold)
        assert_eq!(radius_for(1000, 0.000445), 448);
    }

    #[test]
    fn kanerva_1988_table_7_1_as_jaeckel_quotes_it() {
        // Jaeckel 1989 (RIACS TR 89.28) p. 22, Table 1, "Kanerva's design", from Kanerva (1988) Table 7.1 p. 63:
        // n 1000, M 10^6, r 451, the expected hard-locations activated by two addresses d bits apart. The table is
        // normalised to 1,000 at d = 0 (the exact share at r 451 is 0.00107, 1,072 locations).
        let table = [(0, 1000.0), (1, 894.0), (10, 743.0), (50, 445.0), (100, 267.0), (150, 162.0), (200, 97.0), (300, 30.0), (400, 7.0), (500, 1.0)];
        let i0 = intersection(1000, 451, 0);
        for (d, want) in table {
            let got = 1000.0 * intersection(1000, 451, d) / i0;
            assert!((got - want).abs() <= 0.6, "d {}: {} against {}", d, got, want);
        }
        assert!((1e6 * ball(1000, 451) - 1072.0).abs() < 1.0);
    }

    #[test]
    fn kanerva_2009_far_points_are_rare() {
        // Kanerva 2009 p. 143, n 10,000: "less than a millionth ... closer than 0.476", "a thousand-millionth ...
        // closer than 0.47" (as shares of n)
        assert!(ball(10_000, 4_759) < 1e-6);
        assert!(ball(10_000, 4_699) < 1e-9);
    }

    #[test]
    fn two_balls_farther_apart_than_twice_the_radius_share_nothing() {
        // FKB 1989 p. 15: the critical-distance "is necessarily less than the sum of the Hamming radii"
        for (n, r) in [(64usize, 20usize), (256, 112), (1000, 451)] {
            assert_eq!(intersection(n, r, 2 * r + 1), 0.0);
            assert!(intersection(n, r, 2 * r) > 0.0);
        }
    }
}
