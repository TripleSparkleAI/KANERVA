//! Sizing a memory: the activation-radius to build with, and how many words it will hold, from the rules the
//! SETTLE campaign measured.
//!
//! - [`snr_activation_radius`]: the activation-radius that wakes p = (2 M T)^(-1/3) of the hard-locations at the
//!   design load T = M / 20 (Kanerva's write budget). SDMSCALE measured that the address read's capacity at 10%
//!   address-noise grows about as M^0.96 under it (P90 50 at M 2,000 to 20,000 at M 10^6), where a fixed
//!   activation-radius saturates near 50 (`runs/sdmscale/REPORT_SDMSCALE.md` §3).
//! - [`noise_activation_radius`]: the activation-radius with the largest signal-to-noise capacity from a given
//!   address-noise. SDMRADIUS measured that, chosen for 30% address-noise, it makes the 30% capacity grow with M
//!   (P90 10 to 300 from M 10^4 to 10^6) where the SNR radius stays at 1 to 70; the SNR radius holds 1.4x to 3x
//!   more at 10% (`runs/sdmradius/REPORT_SDMRADIUS.md` §3). A radius is chosen for the noise you expect.
//! - [`Sdm::predicted_failure_rate`]: predictor TRACK (PERSIST) for the address read. SDMREFUSE measured its mean
//!   absolute error in the failure fraction at 0.015 over 12 sealed cells and 0.013 over 8 post-hoc cells
//!   (`runs/sdmrefuse/REPORT_SDMREFUSE.md` §5), where the plain signal-to-noise map behind
//!   [`Sdm::predicted_capacity`] erred by 0.120.
//!
//! <claudes_code_comments>
//! ** Function List **
//! snr_activation_radius(n, m)                       - the SNR policy's activation-radius (design load M/20)
//! noise_activation_radius(n, m, address_noise)      - the activation-radius tuned for an address-noise
//! impl Sdm: with_snr_radius / for_address_noise     - front-door constructors over those radii
//! impl Sdm: predicted_failure_rate                  - TRACK-P failure fraction for this memory
//! impl Sdm: predicted_capacity_track                - the largest load TRACK-P predicts at recall >= q
//!
//! ** Technical Review **
//! - snr_activation_radius is exactly the r0 inside smap::search_window and the instruments' snr_radius:
//!   radius_for(n, (M^2 / 10)^(-1/3)).
//! - noise_activation_radius is the instruments' cd_radius: smap::radius_for_address_noise over
//!   smap::search_window(n, m).
//! - predicted_failure_rate averages refuse::Track::p_converge(.., persist = true, ..) and returns 1 - success;
//!   predicted_capacity_track runs smap::largest_t over it with common random numbers (one seed for every T), so
//!   the curve is close to monotone. SDMREFUSE validated the failure fraction cell by cell; a capacity read off the
//!   curve inherits that accuracy but was summarised there on a coarse checkpoint grid, not by bisection.
//!
//! </claudes_code_comments>

use crate::refuse::Track;
use crate::sdm::Sdm;
use crate::smap::{largest_t, radius_for_address_noise, search_window};
use crate::theory::radius_for;

/// The signal-to-noise policy's activation-radius: the smallest radius waking at least p = (2 M T)^(-1/3) of
/// the hard-locations at design load T = M / 20, that is p = (M^2 / 10)^(-1/3).
///
/// ```
/// use kanerva::{sizing::snr_activation_radius, smap::search_window, theory::radius_for};
/// let r = snr_activation_radius(256, 100_000);
/// assert_eq!(r, radius_for(256, (1e10f64 / 10.0).powf(-1.0 / 3.0)));
/// assert_eq!(search_window(256, 100_000).0, r - 4); // the search window starts four bits under it
/// assert!(snr_activation_radius(256, 1_000_000) < r); // more hard-locations, a smaller radius
/// ```
pub fn snr_activation_radius(n: usize, m: usize) -> usize {
    radius_for(n, (m as f64 * m as f64 / 10.0).powf(-1.0 / 3.0))
}

/// The activation-radius in the search window with the largest signal-to-noise capacity from read-addresses
/// with this share of address-noise.
///
/// ```
/// use kanerva::sizing::{noise_activation_radius, snr_activation_radius};
/// // tuned for heavy noise, the radius is wider than the SNR radius
/// assert!(noise_activation_radius(256, 100_000, 0.3) > snr_activation_radius(256, 100_000));
/// ```
pub fn noise_activation_radius(n: usize, m: usize, address_noise: f64) -> usize {
    let (lo, hi) = search_window(n, m);
    radius_for_address_noise(n, m, address_noise, lo, hi).0
}

impl Sdm {
    /// A memory with the signal-to-noise activation-radius ([`snr_activation_radius`]): the measured choice for
    /// light address-noise (about 10%) at Kanerva's design load.
    ///
    /// ```
    /// use kanerva::{Sdm, sizing::snr_activation_radius};
    /// let sdm = Sdm::with_snr_radius(256, 10_000, 1);
    /// assert_eq!(sdm.activation_radius(), snr_activation_radius(256, 10_000));
    /// ```
    pub fn with_snr_radius(word_size: usize, hard_locations: usize, seed: u64) -> Sdm {
        Sdm::new(word_size, hard_locations, snr_activation_radius(word_size, hard_locations), seed)
    }

    /// A memory whose activation-radius is tuned for read-addresses with this share of address-noise
    /// ([`noise_activation_radius`]).
    ///
    /// ```
    /// use kanerva::Sdm;
    /// let noisy = Sdm::for_address_noise(256, 10_000, 0.3, 1);
    /// let light = Sdm::with_snr_radius(256, 10_000, 1);
    /// assert!(noisy.activation_radius() > light.activation_radius());
    /// ```
    pub fn for_address_noise(word_size: usize, hard_locations: usize, address_noise: f64, seed: u64) -> Sdm {
        Sdm::new(word_size, hard_locations, noise_activation_radius(word_size, hard_locations, address_noise), seed)
    }

    /// The share of address reads predicted to miss (end below overlap 0.95) when this memory holds `stored`
    /// random words and the read-address carries `address_noise`: predictor TRACK in its PERSIST form, from
    /// `samples` sampled reads. A prediction from the memory's shape (n, M, r), not a read of its contents.
    ///
    /// ```
    /// let sdm = kanerva::Sdm::with_snr_radius(256, 2_000, 1);
    /// let light = sdm.predicted_failure_rate(0.1, 10, 200, 1);
    /// let heavy = sdm.predicted_failure_rate(0.1, 400, 200, 1);
    /// assert!(light < heavy);
    /// assert!((0.0..=1.0).contains(&heavy));
    /// ```
    pub fn predicted_failure_rate(&self, address_noise: f64, stored: usize, samples: usize, seed: u64) -> f64 {
        let tr = Track::new(self.word_size(), self.hard_locations(), self.activation_radius());
        1.0 - tr.p_converge(address_noise, stored, samples, true, seed)
    }

    /// The largest number of stored words at which TRACK predicts at least `recall` of address reads succeed from
    /// this share of address-noise. `samples` sampled reads per load tried; one seed for every load.
    ///
    /// ```
    /// let sdm = kanerva::Sdm::with_snr_radius(256, 2_000, 1);
    /// let c10 = sdm.predicted_capacity_track(0.1, 0.9, 200, 1);
    /// let c30 = sdm.predicted_capacity_track(0.3, 0.9, 200, 1);
    /// assert!(c10 > c30);
    /// ```
    pub fn predicted_capacity_track(&self, address_noise: f64, recall: f64, samples: usize, seed: u64) -> usize {
        let tr = Track::new(self.word_size(), self.hard_locations(), self.activation_radius());
        largest_t(|t| tr.p_converge(address_noise, t, samples, true, seed) >= recall)
    }
}
