//! xorshift64* random numbers: uniform in [0, 1) and in [-1, 1), and a bounded integer. No dependencies.

/// The xorshift64* generator every module draws from; the same seed gives the same stream as the SETTLE interpreter.
/// A clone continues the same stream from the same point.
///
/// ```
/// use kanerva::Rng;
/// let (mut a, mut b) = (Rng::new(42), Rng::new(42));
/// assert_eq!(a.next_u64(), b.next_u64());
/// let mut c = a.clone();
/// assert_eq!(a.unit(), c.unit());
/// ```
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator from a seed; the seed is mixed by a multiply and made odd, so seed 0 is valid.
    ///
    /// ```
    /// use kanerva::Rng;
    /// let (mut a, mut b) = (Rng::new(0), Rng::new(1));
    /// assert_ne!(a.next_u64(), b.next_u64()); // seed 0 is valid and differs from seed 1
    /// ```
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    /// The next raw 64-bit output.
    ///
    /// ```
    /// use kanerva::Rng;
    /// let mut a = Rng::new(9);
    /// let mut b = Rng::new(9);
    /// assert!((0..10).all(|_| a.next_u64() == b.next_u64()));
    /// ```
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// A uniform draw in [0, 1), from the top 53 bits.
    ///
    /// ```
    /// use kanerva::Rng;
    /// let mut r = Rng::new(1);
    /// let xs: Vec<f64> = (0..10_000).map(|_| r.unit()).collect();
    /// assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
    /// assert!((xs.iter().sum::<f64>() / 10_000.0 - 0.5).abs() < 0.02);
    /// ```
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A uniform draw in [-1, 1).
    ///
    /// ```
    /// use kanerva::Rng;
    /// let mut r = Rng::new(1);
    /// assert!((0..1000).map(|_| r.signed()).all(|x| (-1.0..1.0).contains(&x)));
    /// ```
    pub fn signed(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }
    /// An integer in 0..n by modulo (`n` 0 is treated as 1, so it returns 0).
    ///
    /// ```
    /// use kanerva::Rng;
    /// let mut r = Rng::new(1);
    /// assert!((0..1000).all(|_| r.below(7) < 7));
    /// assert_eq!(r.below(0), 0);
    /// ```
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    /// A standard normal draw (Box-Muller), for the smooth-number things.
    ///
    /// ```
    /// use kanerva::Rng;
    /// let mut r = Rng::new(1);
    /// let xs: Vec<f64> = (0..20_000).map(|_| r.normal()).collect();
    /// let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    /// let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    /// assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05);
    /// ```
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
