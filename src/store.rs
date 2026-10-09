//! Kanerva's sparse distributed memory at Kanerva's own scale: 10^5 to 10^6 hard-locations, one byte per
//! counter in a flat array, bit-packed addresses, and reads that are popcount scans.
//!
//! Write p: every hard-location whose address is within `radius` of p adds p to its counter row,
//! C\[i\] += p (bit-counters clamp at +-127; any clamp is counted in `overflow`).
//!
//! Three reads, all iterated to a fixed point, all taking z_j <- sign(sum over awake i of C\[i\]\[j\]) with a zero
//! sum keeping the old bit:
//! - `read_addresses`: Kanerva's read, awake = { i : hamming(a_i, z) <= activation-radius }.
//! - `read_pulls` / `read_pulls_at`: the content-woken read, awake = { filled i : C\[i\] . z > theta }
//!   (theta = WAKE n = 0.4 n by default). Addresses are used only when writing.
//! - `read_pulls_topk`: the content-woken read waking the k filled rows with the largest C\[i\] . z.
//!
//! Also: the row and entry shuffles (negative controls), and the density-scaled wake thresholds.

use crate::address::Addresses;
use crate::bits::pack;
use crate::rng::Rng;
use crate::theory::{ball, phi_inv};

/// Fraction of n a hard-location's counter overlap C\[i\] . z must pass to wake in the content-woken (`pulls`) read.
///
/// ```
/// assert_eq!(kanerva::store::WAKE, 0.4);
/// let n = 256;
/// assert_eq!(kanerva::store::WAKE * n as f64, 102.4); // the default threshold for 256-bit words
/// ```
pub const WAKE: f64 = 0.4;

/// Thread count for the parallel parts: SDMSCALE_THREADS, else 8. On wasm32, which has no threads, always 1, so
/// every parallel part runs inline there. The count changes speed only, never a result.
///
/// ```
/// let t = kanerva::store::threads();
/// assert!(t >= 1); // 8 unless SDMSCALE_THREADS says otherwise
/// ```
pub fn threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        return 1;
    }
    std::env::var("SDMSCALE_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8).max(1)
}

/// What one read did.
///
/// ```
/// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
/// let mut rng = Rng::new(3);
/// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
/// let mut st = Store::new(256, 2_000, 112, 1);
/// st.write_many(&pats);
/// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
/// let out: kanerva::store::ReadOut = st.read_addresses(&read_address, 20);
/// assert!(out.fixed && out.rounds <= 20);
/// assert_eq!(out.z, pats[0]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct ReadOut {
    /// The final state of the read.
    pub z: Vec<i8>,
    /// The number of reads the iterated read made.
    pub rounds: usize,
    /// How many hard-locations voted on the last round: those activated by address in `read_addresses` (empty
    /// rows included), those woken by content in the pulls reads.
    pub awake: usize,
    /// How many of the hard-locations activated on the last round hold any nonzero counter.
    pub nonempty: usize,
    /// True if the last round reached a fixed point (z unchanged).
    pub fixed: bool,
}

/// Kanerva's hard-location memory with byte bit-counters.
///
/// ```
/// use kanerva::store::Store;
/// let st = Store::new(256, 2_000, 112, 1);
/// assert_eq!((st.n, st.m, st.radius), (256, 2_000, 112));
/// assert_eq!(st.ctr.len(), 2_000 * 256);
/// assert_eq!(st.filled_rows(), 0);
/// ```
#[derive(Clone)]
pub struct Store {
    /// The word-size in bits.
    pub n: usize,
    /// The number of hard-locations M.
    pub m: usize,
    /// The activation-radius r.
    pub radius: usize,
    wpl: usize,
    addr: Vec<u64>,
    /// Counters, row-major: hard-location i's row is `ctr[i * n..(i + 1) * n]`. Code that edits them directly
    /// must keep `filled` current.
    pub ctr: Vec<i8>,
    /// Rows with at least one nonzero counter (kept current by write and the shuffles).
    pub filled: Vec<bool>,
    /// Counter updates that would have left \[-127, 127\] (clamped). Any run with overflow > 0 is flagged.
    pub overflow: u64,
    /// The number of patterns written.
    pub writes: usize,
}

impl Store {
    /// Addresses from a seed, 64 random bits at a time (the last word masked to n bits).
    ///
    /// ```
    /// use kanerva::store::Store;
    /// let (a, b) = (Store::new(256, 500, 112, 7), Store::new(256, 500, 112, 7));
    /// let probe = vec![1i8; 256];
    /// assert_eq!(a.awake(&probe), b.awake(&probe)); // the same seed gives the same addresses
    /// ```
    pub fn new(n: usize, m: usize, radius: usize, seed: u64) -> Store {
        let wpl = n.div_ceil(64);
        let mut r = Rng::new(seed ^ 0x5D5C_A1E0_0000_0001);
        let tail = if n.is_multiple_of(64) { u64::MAX } else { (1u64 << (n % 64)) - 1 };
        let mut addr = Vec::with_capacity(m * wpl);
        for _ in 0..m {
            for k in 0..wpl {
                let w = r.next_u64();
                addr.push(if k + 1 == wpl { w & tail } else { w });
            }
        }
        Store { n, m, radius, wpl, addr, ctr: vec![0; m * n], filled: vec![false; m], overflow: 0, writes: 0 }
    }

    /// A store over the named addresses `Addresses::named(name, seed, n, m)`: the same matrix the SETTLE `sdm`
    /// family (bit-counters in pulls) builds, so the two can be checked against each other bit for bit.
    ///
    /// ```
    /// use kanerva::{store::Store, address::Addresses};
    /// let st = Store::like_view("s", 256, 500, 112, 1);
    /// let a = Addresses::named("s", 1, 256, 500);
    /// let z = vec![1i8; 256];
    /// let zf = vec![1.0f64; 256];
    /// let from_store: Vec<usize> = st.awake(&z).into_iter().map(|i| i as usize).collect();
    /// assert_eq!(from_store, a.awake(&zf, 112)); // the same addresses as the named matrix
    /// ```
    pub fn like_view(name: &str, n: usize, m: usize, radius: usize, seed: u64) -> Store {
        Store::from_addresses(Addresses::named(name, seed, n, m), radius)
    }

    /// An empty store over an address matrix the caller already holds (for example one drawn by another program
    /// from its own seed), with the given activation-radius. The word-size and the number of hard-locations come
    /// from the matrix.
    ///
    /// ```
    /// use kanerva::{store::Store, address::Addresses};
    /// let a = Addresses::named("s", 1, 256, 500);
    /// let st = Store::from_addresses(a.clone(), 112);
    /// assert_eq!((st.n, st.m, st.radius, st.filled_rows()), (256, 500, 112, 0));
    /// let z = vec![1i8; 256];
    /// let woken: Vec<usize> = st.awake(&z).into_iter().map(|i| i as usize).collect();
    /// assert_eq!(woken, a.awake(&vec![1.0; 256], 112)); // the same hard-locations wake as in the matrix
    /// ```
    pub fn from_addresses(a: Addresses, radius: usize) -> Store {
        let (n, m) = (a.n, a.m);
        Store { n, m, radius, wpl: a.wpl, addr: a.words, ctr: vec![0; m * n], filled: vec![false; m], overflow: 0, writes: 0 }
    }

    /// Hard-location `i`'s bit-counter for bit `j`.
    ///
    /// ```
    /// use kanerva::store::Store;
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// let p = vec![1i8; 256];
    /// let i = st.awake(&p)[0] as usize;
    /// st.write(&p);
    /// assert_eq!(st.counter(i, 0), 1);
    /// ```
    pub fn counter(&self, i: usize, j: usize) -> i8 {
        self.ctr[i * self.n + j]
    }

    /// Hard-location `i`'s row of bit-counters.
    ///
    /// ```
    /// use kanerva::store::Store;
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// let p = vec![-1i8; 256];
    /// let i = st.awake(&p)[0] as usize;
    /// st.write(&p);
    /// assert!(st.row(i).iter().all(|&c| c == -1));
    /// ```
    pub fn row(&self, i: usize) -> &[i8] {
        &self.ctr[i * self.n..(i + 1) * self.n]
    }

    /// Hamming distance from hard-location i's address to a packed pattern.
    ///
    /// ```
    /// use kanerva::{store::Store, bits::pack};
    /// let st = Store::new(256, 2_000, 112, 1);
    /// let p = vec![1i8; 256];
    /// let q = pack(&p);
    /// for i in st.awake(&p) {
    ///     assert!(st.dist(i as usize, &q) <= 112); // every activated hard-location is inside the activation-radius
    /// }
    /// ```
    pub fn dist(&self, i: usize, q: &[u64]) -> usize {
        self.addr[i * self.wpl..(i + 1) * self.wpl].iter().zip(q).map(|(a, b)| (a ^ b).count_ones() as usize).sum()
    }

    /// Hard-locations whose address is within the activation-radius of z.
    ///
    /// ```
    /// use kanerva::{store::Store, theory::ball};
    /// let st = Store::new(256, 2_000, 112, 1);
    /// let n = st.awake(&vec![1i8; 256]).len();
    /// let expected = ball(256, 112) * 2_000.0; // about 52.5
    /// assert!(n > 20 && n < 100, "{} near {}", n, expected);
    /// ```
    pub fn awake(&self, z: &[i8]) -> Vec<u32> {
        let q = pack(z);
        self.awake_packed(&q)
    }

    /// Hard-locations whose address is within the activation-radius of a packed state, in index order.
    ///
    /// ```
    /// use kanerva::{store::Store, bits::pack};
    /// let st = Store::new(256, 2_000, 112, 1);
    /// let z = vec![1i8; 256];
    /// assert_eq!(st.awake_packed(&pack(&z)), st.awake(&z));
    /// ```
    pub fn awake_packed(&self, q: &[u64]) -> Vec<u32> {
        let (w, r) = (self.wpl, self.radius as u32);
        let mut out = Vec::new();
        for (i, a) in self.addr.chunks_exact(w).enumerate() {
            let mut d = 0u32;
            for k in 0..w {
                d += (a[k] ^ q[k]).count_ones();
            }
            if d <= r {
                out.push(i as u32);
            }
        }
        out
    }

    fn add_rows(&mut self, act: &[u32], p: &[i8]) {
        let n = self.n;
        for &i in act {
            let i = i as usize;
            let row = &mut self.ctr[i * n..(i + 1) * n];
            // branch-free so the compiler vectorizes it: the clamp count is summed, not branched on
            // (the same counters and the same overflow count as the branching loop, about 3x faster; DISCOVERIES.md)
            let mut clamped = 0u64;
            for (c, &x) in row.iter_mut().zip(p) {
                let v = *c as i16 + x as i16;
                clamped += u64::from(!(-127..=127).contains(&v));
                *c = v.clamp(-127, 127) as i8;
            }
            self.overflow += clamped;
            self.filled[i] = row.iter().any(|&c| c != 0);
        }
        self.writes += 1;
    }

    /// Write p: C\[i\] += p for every hard-location within the activation-radius. Returns how many hard-locations took it.
    ///
    /// ```
    /// use kanerva::{store::Store, Rng, bits::random_pattern};
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// let p = random_pattern(256, &mut Rng::new(1));
    /// let took = st.write(&p);
    /// assert_eq!(took, st.awake(&p).len());
    /// assert_eq!((st.writes, st.filled_rows()), (1, took));
    /// ```
    pub fn write(&mut self, p: &[i8]) -> usize {
        let act = self.awake(p);
        self.add_rows(&act, p);
        act.len()
    }

    /// Write many patterns in order, finding their wake sets in parallel. Same result as writing one by one.
    ///
    /// ```
    /// use kanerva::{store::Store, Rng, bits::random_pattern};
    /// let mut rng = Rng::new(2);
    /// let pats: Vec<Vec<i8>> = (0..10).map(|_| random_pattern(256, &mut rng)).collect();
    /// let (mut a, mut b) = (Store::new(256, 2_000, 112, 1), Store::new(256, 2_000, 112, 1));
    /// let total = a.write_many(&pats);
    /// let one_by_one: usize = pats.iter().map(|p| b.write(p)).sum();
    /// assert_eq!((total, &a.ctr), (one_by_one, &b.ctr));
    /// ```
    pub fn write_many(&mut self, ps: &[Vec<i8>]) -> usize {
        let th = threads().min(ps.len().max(1));
        let chunk = (ps.len() + th - 1) / th.max(1);
        let acts: Vec<Vec<u32>> = if ps.len() < 4 || th == 1 {
            ps.iter().map(|p| self.awake(p)).collect()
        } else {
            let me = &*self;
            std::thread::scope(|s| {
                let hs: Vec<_> = ps.chunks(chunk.max(1)).map(|c| s.spawn(move || c.iter().map(|p| me.awake(p)).collect::<Vec<_>>())).collect();
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            })
        };
        let mut total = 0;
        for (p, a) in ps.iter().zip(&acts) {
            total += a.len();
            self.add_rows(a, p);
        }
        total
    }

    /// Kanerva's read: z <- sign(sum over awake rows), a zero sum keeps the bit; repeat to a fixed point.
    ///
    /// ```
    /// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
    /// let mut rng = Rng::new(3);
    /// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&pats);
    /// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
    /// let out = st.read_addresses(&read_address, 20);
    /// assert_eq!(overlap(&out.z, &pats[0]), 1.0); // 10% address-noise read back exactly
    /// assert!(out.fixed);
    /// ```
    pub fn read_addresses(&self, read_address: &[i8], iters: usize) -> ReadOut {
        let n = self.n;
        let mut z = read_address.to_vec();
        let mut sum = vec![0i32; n];
        let mut out = ReadOut::default();
        for t in 0..iters.max(1) {
            let act = self.awake(&z);
            sum.iter_mut().for_each(|s| *s = 0);
            let mut ne = 0;
            for &i in &act {
                let i = i as usize;
                if !self.filled[i] {
                    continue;
                }
                ne += 1;
                for (s, &c) in sum.iter_mut().zip(&self.ctr[i * n..(i + 1) * n]) {
                    *s += c as i32;
                }
            }
            let next: Vec<i8> = sum.iter().zip(&z).map(|(&s, &old)| if s > 0 { 1 } else if s < 0 { -1 } else { old }).collect();
            out.rounds = t + 1;
            out.awake = act.len();
            out.nonempty = ne;
            if next == z {
                out.fixed = true;
                break;
            }
            z = next;
        }
        out.z = z;
        out
    }

    /// The sdm family's `via: :pulls` read at zero temperature: a hard-location wakes when C\[i\].z > 0.4 n, then
    /// z <- sign(sum over awake rows), a zero sum keeps the bit; repeat to a fixed point. Addresses unused.
    ///
    /// ```
    /// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
    /// let mut rng = Rng::new(3);
    /// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&pats);
    /// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
    /// let out = st.read_pulls(&read_address, 20);
    /// assert_eq!(out.z, pats[0]);
    /// ```
    pub fn read_pulls(&self, read_address: &[i8], iters: usize) -> ReadOut {
        self.read_pulls_at(read_address, iters, WAKE * self.n as f64)
    }

    /// The pulls read with a given wake threshold (SDMRADIUS scales it with the store's density).
    ///
    /// ```
    /// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
    /// let mut rng = Rng::new(3);
    /// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&pats);
    /// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
    /// // the default threshold, 0.4 n, gives the same read as read_pulls
    /// let a = st.read_pulls_at(&read_address, 20, 0.4 * 256.0);
    /// assert_eq!(a.z, st.read_pulls(&read_address, 20).z);
    /// ```
    pub fn read_pulls_at(&self, read_address: &[i8], iters: usize, thresh: f64) -> ReadOut {
        self.pulls_loop(read_address, iters, |dots| dots.iter().map(|&(i, d)| (i, d as f64 > thresh)).filter(|x| x.1).map(|x| x.0).collect())
    }

    /// The pulls read waking the k filled rows with the largest C\[i\].z (ties by row index), k fixed.
    ///
    /// ```
    /// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
    /// let mut rng = Rng::new(3);
    /// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&pats);
    /// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
    /// let out = st.read_pulls_topk(&read_address, 20, 52);
    /// assert!(overlap(&out.z, &pats[0]) > 0.95);
    /// assert!(out.awake <= 52);
    /// ```
    pub fn read_pulls_topk(&self, read_address: &[i8], iters: usize, k: usize) -> ReadOut {
        self.pulls_loop(read_address, iters, |dots| {
            let mut v: Vec<(usize, i32)> = dots.to_vec();
            let k = k.min(v.len());
            if k == 0 {
                return Vec::new();
            }
            v.select_nth_unstable_by(k - 1, |a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v[..k].iter().map(|x| x.0).collect()
        })
    }

    /// Shared loop of the content-woken reads: `pick` chooses the awake rows from (row, C\[i\].z) of every
    /// filled row; z <- sign(sum over awake rows), a zero sum keeps the bit; repeat to a fixed point.
    fn pulls_loop<F: Fn(&[(usize, i32)]) -> Vec<usize>>(&self, read_address: &[i8], iters: usize, pick: F) -> ReadOut {
        let n = self.n;
        let mut z = read_address.to_vec();
        let mut sum = vec![0i32; n];
        let mut out = ReadOut::default();
        let mut dots: Vec<(usize, i32)> = Vec::with_capacity(self.m);
        for t in 0..iters.max(1) {
            sum.iter_mut().for_each(|s| *s = 0);
            dots.clear();
            for i in 0..self.m {
                if !self.filled[i] {
                    continue;
                }
                let row = &self.ctr[i * n..(i + 1) * n];
                let dot: i32 = row.iter().zip(&z).map(|(&c, &v)| (c as i16 * v as i16) as i32).sum();
                dots.push((i, dot));
            }
            let awake = pick(&dots);
            for &i in &awake {
                for (s, &c) in sum.iter_mut().zip(&self.ctr[i * n..(i + 1) * n]) {
                    *s += c as i32;
                }
            }
            let next: Vec<i8> = sum.iter().zip(&z).map(|(&s, &old)| if s > 0 { 1 } else if s < 0 { -1 } else { old }).collect();
            out.rounds = t + 1;
            out.awake = awake.len();
            out.nonempty = awake.len();
            if next == z {
                out.fixed = true;
                break;
            }
            z = next;
        }
        out.z = z;
        out
    }

    /// Negative control: deal whole counter rows to other hard-locations (a random permutation of rows).
    /// Returns the permutation: row now at hard-location i came from hard-location perm\[i\].
    ///
    /// ```
    /// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
    /// let mut rng = Rng::new(3);
    /// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&pats);
    /// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
    /// let mut control = st.clone();
    /// let perm = control.shuffle_rows(&mut Rng::new(5));
    /// assert_eq!(perm.len(), 2_000);
    /// assert!(overlap(&control.read_addresses(&read_address, 20).z, &pats[0]) < 0.5); // the memory is gone
    /// ```
    pub fn shuffle_rows(&mut self, r: &mut Rng) -> Vec<usize> {
        let n = self.n;
        let mut perm: Vec<usize> = (0..self.m).collect();
        for i in (1..self.m).rev() {
            let k = r.below(i + 1);
            perm.swap(i, k);
        }
        let old = self.ctr.clone();
        for i in 0..self.m {
            let src = perm[i];
            self.ctr[i * n..(i + 1) * n].copy_from_slice(&old[src * n..(src + 1) * n]);
        }
        let of = self.filled.clone();
        for i in 0..self.m {
            self.filled[i] = of[perm[i]];
        }
        perm
    }

    /// Negative control: permute every counter entry across the whole matrix (M x n entries).
    ///
    /// ```
    /// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
    /// let mut rng = Rng::new(3);
    /// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// st.write_many(&pats);
    /// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
    /// let mut control = st.clone();
    /// control.shuffle_entries(&mut Rng::new(5));
    /// assert!(overlap(&control.read_addresses(&read_address, 20).z, &pats[0]) < 0.5);
    /// ```
    pub fn shuffle_entries(&mut self, r: &mut Rng) {
        let len = self.ctr.len();
        for i in (1..len).rev() {
            let k = (r.next_u64() % (i as u64 + 1)) as usize;
            self.ctr.swap(i, k);
        }
        let n = self.n;
        for i in 0..self.m {
            self.filled[i] = self.ctr[i * n..(i + 1) * n].iter().any(|&c| c != 0);
        }
    }

    /// Number of rows holding any nonzero counter.
    ///
    /// ```
    /// use kanerva::store::Store;
    /// let mut st = Store::new(256, 2_000, 112, 1);
    /// assert_eq!(st.filled_rows(), 0);
    /// let took = st.write(&vec![1i8; 256]);
    /// assert_eq!(st.filled_rows(), took);
    /// ```
    pub fn filled_rows(&self) -> usize {
        self.filled.iter().filter(|&&f| f).count()
    }
}

/// Mean load of a filled row, measured from the bit-counters: sum_j C_ij^2 / n (a row holding L random
/// patterns has E\[C_ij^2\] = L), averaged over filled rows.
///
/// ```
/// use kanerva::{store::{Store, mean_row_load}, Rng, bits::random_pattern};
/// let mut st = Store::new(256, 2_000, 112, 1);
/// assert_eq!(mean_row_load(&st), 0.0); // no filled rows
/// st.write(&random_pattern(256, &mut Rng::new(1)));
/// assert_eq!(mean_row_load(&st), 1.0); // each filled row holds one pattern: every C_ij^2 is 1
/// ```
pub fn mean_row_load(st: &Store) -> f64 {
    let (mut s, mut k) = (0.0, 0usize);
    for i in 0..st.m {
        if st.filled[i] {
            s += st.row(i).iter().map(|&c| (c as f64) * (c as f64)).sum::<f64>() / st.n as f64;
            k += 1;
        }
    }
    if k == 0 {
        0.0
    } else {
        s / k as f64
    }
}

/// Density-scaled wake threshold for the pulls read: theta = kappa sqrt(n L), where L is the mean row load
/// and kappa = Phi^-1(1 - eps p M / M_f) puts the expected number of rows woken by noise alone
/// (a row not holding the pattern has C_i.z ~ N(0, n L)) at eps times the p M rows that hold the pattern.
/// kappa is floored at 1. Returns (theta, kappa, L).
///
/// ```
/// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
/// let mut rng = Rng::new(3);
/// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
/// let mut st = Store::new(256, 2_000, 112, 1);
/// st.write_many(&pats);
/// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
/// use kanerva::store::density_threshold;
/// let (theta, kappa, load) = density_threshold(&st, 0.1);
/// assert!(kappa >= 1.0 && load >= 1.0);
/// assert!((theta - kappa * (256.0 * load).sqrt()).abs() < 1e-9);
/// ```
pub fn density_threshold(st: &Store, eps: f64) -> (f64, f64, f64) {
    let l = mean_row_load(st).max(1.0);
    let mf = st.filled_rows().max(1) as f64;
    let pm = ball(st.n, st.radius) * st.m as f64;
    let tail = (eps * pm / mf).min(0.5);
    let kappa = if tail <= 0.0 { 1.0 } else { phi_inv(1.0 - tail).max(1.0) };
    (kappa * (st.n as f64 * l).sqrt(), kappa, l)
}

/// The pattern-level repair of `density_threshold` (SDMRADIUS follow-up, sealed after the first pulls run):
/// rows that hold the same pattern wake together, so a false wake is a whole block of about p M rows, not one
/// row. kappa_pat = Phi^-1(1 - eps_pat / (T - 1)) bounds the chance that any of the T - 1 other patterns wakes
/// as a block; theta = max(kappa_row, kappa_pat) sqrt(n L). Returns (theta, kappa, L).
///
/// ```
/// use kanerva::{Rng, bits::{random_pattern, add_address_noise, overlap}, store::Store};
/// let mut rng = Rng::new(3);
/// let pats: Vec<Vec<i8>> = (0..20).map(|_| random_pattern(256, &mut rng)).collect();
/// let mut st = Store::new(256, 2_000, 112, 1);
/// st.write_many(&pats);
/// let read_address = add_address_noise(&pats[0], 0.1, &mut rng);
/// use kanerva::store::{density_threshold, density_threshold_blocks};
/// let (row, _, _) = density_threshold(&st, 0.1);
/// let (block, _, _) = density_threshold_blocks(&st, 0.1, 0.01);
/// assert!(block >= row); // the block rule never wakes on a lower threshold
/// ```
pub fn density_threshold_blocks(st: &Store, eps_row: f64, eps_pat: f64) -> (f64, f64, f64) {
    let (_, kr, l) = density_threshold(st, eps_row);
    let others = st.writes.saturating_sub(1).max(1) as f64;
    let kp = phi_inv(1.0 - (eps_pat / others).min(0.5)).max(1.0);
    let k = kr.max(kp);
    (k * (st.n as f64 * l.max(1.0)).sqrt(), k, l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{add_address_noise, overlap, random_pattern};
    use crate::theory::radius_for;

    #[test]
    fn a_two_location_store_by_hand() {
        // n 4; build the addresses by hand: hard-location 0 at 0000, hard-location 1 at 1111; activation-radius 1
        let mut st = Store::new(4, 2, 1, 0);
        st.addr = vec![0b0000, 0b1111];
        // p = [+1, -1, -1, -1] packs to 0b0001: 1 bit from hard-location 0, 3 from hard-location 1
        assert_eq!(st.write(&[1, -1, -1, -1]), 1);
        assert_eq!(st.row(0), &[1, -1, -1, -1]);
        assert_eq!(st.row(1), &[0, 0, 0, 0]);
        assert_eq!((st.filled_rows(), st.writes, st.overflow), (1, 1, 0));
        // an all -1 read-address wakes hard-location 0 only; the vote is [1, -1, -1, -1]: bit 0 flips, the rest hold
        let o = st.read_addresses(&[-1, -1, -1, -1], 5);
        assert_eq!(o.z, vec![1, -1, -1, -1]);
        assert_eq!((o.rounds, o.awake, o.nonempty, o.fixed), (2, 1, 1, true));
        // the content read: C_0 . z = 1 + 1 + 1 + 1 = 4 for z = p; a threshold of 3.9 wakes it, 4 does not
        assert_eq!(st.read_pulls_at(&[1, -1, -1, -1], 3, 3.9).awake, 1);
        assert_eq!(st.read_pulls_at(&[1, -1, -1, -1], 3, 4.0).awake, 0);
    }

    #[test]
    fn counters_clamp_and_count_the_overflow() {
        let mut st = Store::new(8, 1, 8, 0); // radius n: every write reaches the one location
        let p = vec![1i8; 8];
        for _ in 0..130 {
            st.write(&p);
        }
        assert_eq!(st.counter(0, 0), 127);
        assert_eq!(st.overflow, 3 * 8);
    }

    #[test]
    fn the_branch_free_counter_update_matches_the_branching_reference() {
        // the reference is the loop the update replaced: branch on every clamp, count it, then clamp
        fn reference(ctr: &mut [i8], p: &[i8], overflow: &mut u64) {
            for j in 0..ctr.len() {
                let v = ctr[j] as i16 + p[j] as i16;
                if !(-127..=127).contains(&v) {
                    *overflow += 1;
                }
                ctr[j] = v.clamp(-127, 127) as i8;
            }
        }
        // radius n: every write reaches the one location, so 300 writes drive the counters into both clamps;
        // the patterns are general i8 (not only ±1), so the clamp is reached from far outside the range too
        let mut st = Store::new(64, 1, 64, 0);
        let (mut want, mut want_of) = (vec![0i8; 64], 0u64);
        let mut r = Rng::new(17);
        for _ in 0..300 {
            let p: Vec<i8> = (0..64).map(|_| (r.below(255) as i16 - 127) as i8).collect();
            st.write(&p);
            reference(&mut want, &p, &mut want_of);
        }
        assert!(want_of > 1_000, "the clamp was exercised: {}", want_of);
        assert_eq!((st.row(0), st.overflow), (&want[..], want_of));
    }

    #[test]
    fn written_patterns_come_back_and_both_shuffles_and_never_written_cues_do_not() {
        let mut st = Store::new(256, 20_000, radius_for(256, 0.005), 7);
        let mut r = Rng::new(9);
        let ps: Vec<Vec<i8>> = (0..100).map(|_| random_pattern(256, &mut r)).collect();
        st.write_many(&ps);
        let ok = |s: &Store, r: &mut Rng| ps.iter().take(20).filter(|p| overlap(&s.read_addresses(&add_address_noise(p, 0.1, r), 20).z, p) >= 0.95).count();
        assert_eq!(ok(&st, &mut r), 20);
        let mut e = st.clone();
        e.shuffle_entries(&mut Rng::new(4));
        assert_eq!(ok(&e, &mut r), 0);
        let mut w = st.clone();
        w.shuffle_rows(&mut Rng::new(4));
        assert_eq!(ok(&w, &mut r), 0);
        let never = random_pattern(256, &mut r);
        assert!(overlap(&st.read_addresses(&add_address_noise(&never, 0.1, &mut r), 20).z, &never) < 0.95);
    }

    #[test]
    fn write_many_equals_one_by_one() {
        let mut a = Store::new(256, 3000, radius_for(256, 0.01), 5);
        let mut b = a.clone();
        let mut r = Rng::new(2);
        let ps: Vec<Vec<i8>> = (0..40).map(|_| random_pattern(256, &mut r)).collect();
        for p in &ps {
            a.write(p);
        }
        b.write_many(&ps);
        assert!(a.ctr == b.ctr && a.filled == b.filled && a.overflow == 0);
    }

    #[test]
    fn the_density_threshold_rises_with_load() {
        let mut st = Store::new(256, 20_000, 106, 2);
        let mut rr = Rng::new(5);
        st.write_many(&(0..5).map(|_| random_pattern(256, &mut rr)).collect::<Vec<_>>());
        let (t1, _, l1) = density_threshold(&st, 0.1);
        st.write_many(&(0..400).map(|_| random_pattern(256, &mut rr)).collect::<Vec<_>>());
        let (t2, _, l2) = density_threshold(&st, 0.1);
        assert!(l2 > l1 && t2 > t1);
        assert!(density_threshold_blocks(&st, 0.1, 0.01).0 >= t2);
    }
}
