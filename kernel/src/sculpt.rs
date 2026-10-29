use crate::intrinsics::sqrt;
use crate::perlin::{Fbm, Lattice, Perm};
use crate::real::Real;
use crate::rng::Rng;
use crate::split::Split;

pub const NUM_CRATERS: usize = 10;
pub const NUM_LOBES: usize = 4;

const RIM_STEEPNESS: f32 = 0.5;
const MIN_RADIUS: f32 = 0.3;
const CRATER_MIN: f64 = 0.15;
const CRATER_MAX: f64 = 0.45;

#[repr(C)]
struct Lobe {
    dir: [Split; 3],
    /// Signed amplitude, negative for a dent.
    amp: Split,
    sharp: Split,
}

#[repr(C)]
struct Crater {
    center: [Split; 3],
    inv_radius: Split,
    depth: f32,
    rim_width: f32,
}

/// Everything the per vertex loop needs, drawn from the seed before any vertex
/// is touched. Each nonce has a seed of its own, so a draw belongs to one
/// scanning block on a GPU and one scan on a CPU core.
#[repr(C)]
pub struct Params {
    lobes: [Lobe; NUM_LOBES],
    craters: [Crater; NUM_CRATERS],
    fbm: Fbm,
    noise_amplitude: Split,
    axis_scale: [Split; 3],
}

impl Params {
    pub fn draw(seed: [u8; 32]) -> Self {
        // Plain f32 and u16 fields, so all zero bits are a valid value to fill.
        let mut params: Params = unsafe { core::mem::zeroed() };
        params.fill(seed, &mut [0u8; 256]);
        params
    }

    /// Draw in place, shuffling the permutation in `perm`. A GPU fills its
    /// output directly rather than building the draw in local memory and
    /// copying it out.
    ///
    /// Draw order is protocol. Shape parameters first, then the fBm stream
    /// seed, then the craters. The draw itself runs in f64, as the chain's
    /// does, and only its results are narrowed.
    #[inline(always)]
    pub fn fill(&mut self, seed: [u8; 32], perm: &mut impl Perm) {
        let mut rng = Rng::new(seed);

        let lam31 = rng.range(0.18, 0.90);
        let lam21 = rng.range(lam31, (1.0 + lam31) * 0.5);
        let mut axis_scale = [1.0, sqrt(lam21), sqrt(lam31)];
        shuffle3(&mut axis_scale, &mut rng);

        for lobe in &mut self.lobes {
            let dir = rng.unit_vector();
            let amp = rng.range(0.08, 0.20);
            let sharp = rng.range(1.0, 2.0);
            let sign = if rng.unit() < 0.25 { -1.0 } else { 1.0 };
            *lobe = Lobe { dir: dir.map(Split::new), amp: Split::new(sign * amp), sharp: Split::new(sharp) };
        }

        self.noise_amplitude = Split::new(rng.range(0.16, 0.30));
        let noise_frequency = rng.range(1.1, 2.4);

        self.fbm.fill(rng.next_seed(), noise_frequency, perm);
        for crater in &mut self.craters {
            let center = rng.unit_vector();
            let radius = rng.range(CRATER_MIN, CRATER_MAX);
            let depth = rng.range(0.4, 0.9);
            let rim_width = rng.range(0.1, 0.3);
            *crater = Crater {
                center: center.map(Split::new),
                inv_radius: Split::new(1.0 / radius),
                depth: depth as f32,
                rim_width: rim_width as f32,
            };
        }
        self.axis_scale = axis_scale.map(Split::new);
    }

    /// The noise lattice drawn with the rest.
    pub fn fbm(&self) -> &Fbm {
        &self.fbm
    }

    /// Displace blank vertices, one per lane, with the noise lattice read
    /// through `lattice`, which must hold this draw's permutation. Only the
    /// lobes and craters set in `near`, as `near` reports them, are looked at.
    /// The rest must add exactly zero at these vertices.
    #[inline(always)]
    pub fn displacement_with<T: Real>(&self, dir: [Split<T>; 3], lattice: &impl Lattice<T>, near: u32) -> T {
        let zero = T::splat(0.0);
        let mut lobes = zero;
        for (k, lobe) in self.lobes.iter().enumerate() {
            if near & 1 << k == 0 {
                continue;
            }
            let dot = dot_split(dir, lobe.dir);
            let up = dot.gt(zero);
            if T::any(up) {
                lobes = lobes + T::select(up, lobe.amp.scale(dot.pow(lobe.sharp)), zero);
            }
        }

        let mut craters = zero;
        for (k, c) in self.craters.iter().enumerate() {
            if near & 1 << (NUM_LOBES + k) == 0 {
                continue;
            }
            // Past 1.5 radii the cavity has risen past both blends and the rim
            // is flat, so the chain adds exactly zero here. Most vertices are
            // that far out, and the squared distance with a margin far wider
            // than its rounding says so without a square root. Only the
            // uncertain band pays for the exact test.
            let d2 = dist2(dir, c.center);
            let close = (d2 * T::splat(c.inv_radius.hi * c.inv_radius.hi)).le(T::splat(2.25 * (1.0 + 1e-5)));
            if !T::any(close) {
                continue;
            }
            let x = c.inv_radius.scale(d2.sqrt());
            let inside = close & x.lt(T::splat(1.5));
            if !T::any(inside) {
                continue;
            }
            let cavity = x * x - T::splat(1.0);
            let rim_x = x - T::splat(1.0) - T::splat(c.rim_width);
            let rim = T::select(rim_x.lt(zero), T::splat(RIM_STEEPNESS) * rim_x * rim_x, zero);
            let blend = smin(smax(cavity, T::splat(-c.depth), &FLOOR), rim, &RIM);
            craters = craters + T::select(inside, blend, zero);
        }

        let e = self.noise_amplitude.scale(self.fbm.get(dir, lattice) + craters) + lobes;
        let least = T::splat(MIN_RADIUS - 1.0);
        T::select(e.gt(least), e, least)
    }

    /// One coordinate at displacement `e`. Keeping this operation separate
    /// lets a GPU retain a compact vertex record and reconstruct only the
    /// coordinates a later pass did not store.
    #[inline(always)]
    pub fn position_component<T: Real>(&self, dir: Split<T>, axis: usize, e: T) -> T {
        let s = dir * self.axis_scale[axis];
        s.hi + s.hi.fma(e, s.lo * (T::splat(1.0) + e))
    }

    #[inline(always)]
    pub fn displace_with<T: Real>(&self, dir: [Split<T>; 3], lattice: &impl Lattice<T>, near: u32) -> [T; 3] {
        let e = self.displacement_with(dir, lattice, near);
        core::array::from_fn(|k| {
            self.position_component(dir[k], k, e)
        })
    }

    /// The lobes and craters that may reach a direction within `radius` of
    /// `center`, as a `displace_with` mask, bit k for lobe k and bit
    /// `NUM_LOBES + k` for crater k. The others add exactly zero anywhere in
    /// that ball, which a warp's or a vector's neighbouring vertices share, so
    /// they skip them together. The margins dwarf every rounding in the tests.
    pub fn near(&self, center: [f32; 3], radius: f32) -> u32 {
        let mut near = 0;
        for (k, lobe) in self.lobes.iter().enumerate() {
            // A lobe adds only where its dot with the direction is positive.
            let dot = center[0] * lobe.dir[0].hi + center[1] * lobe.dir[1].hi + center[2] * lobe.dir[2].hi;
            if dot + radius > -1e-4 {
                near |= 1 << k;
            }
        }
        for (k, c) in self.craters.iter().enumerate() {
            // A crater adds only within 1.5 radii of its center.
            let reach = 1.5 / c.inv_radius.hi * (1.0 + 1e-4) + radius + 1e-4;
            let d: [f32; 3] = core::array::from_fn(|q| center[q] - c.center[q].hi);
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= reach * reach {
                near |= 1 << (NUM_LOBES + k);
            }
        }
        near
    }
}

/// Byte offset of the permutation pairs within `Params`.
#[cfg(any(target_arch = "nvptx64", target_arch = "amdgpu"))]
pub(crate) const PAIRS: usize = core::mem::offset_of!(Params, fbm) + crate::perlin::PAIRS;

fn shuffle3(a: &mut [f64; 3], rng: &mut Rng) {
    for i in (1..3).rev() {
        let j = ((rng.unit() * (i as f64 + 1.0)) as usize).min(i);
        a.swap(i, j);
    }
}

/// Dot of two split vectors. The low halves add in as one correction, which
/// keeps the rounding of a lobe's direction out of every vertex it shapes.
#[inline(always)]
fn dot_split<T: Real>(a: [Split<T>; 3], b: [Split; 3]) -> T {
    let b: [Split<T>; 3] = core::array::from_fn(|k| Split { hi: T::splat(b[k].hi), lo: T::splat(b[k].lo) });
    let lo = a[0].hi * b[0].lo + a[0].lo * b[0].hi + a[1].hi * b[1].lo + a[1].lo * b[1].hi + a[2].hi * b[2].lo
        + a[2].lo * b[2].hi;
    a[0].hi * b[0].hi + a[1].hi * b[1].hi + a[2].hi * b[2].hi + lo
}

#[inline(always)]
fn dist2<T: Real>(a: [Split<T>; 3], b: [Split; 3]) -> T {
    let d: [T; 3] = core::array::from_fn(|k| (a[k].hi - T::splat(b[k].hi)) + (a[k].lo - T::splat(b[k].lo)));
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// A smooth blend of width `k`, with `slope` holding 0.5 / k.
struct Blend {
    k: Split,
    slope: Split,
}

impl Blend {
    const fn new(k: f64) -> Blend {
        Blend { k: Split::new(k), slope: Split::new(0.5 / k) }
    }
}

/// Flattens the crater floor.
const FLOOR: Blend = Blend::new(0.5);

/// Rounds the rim into the surrounding surface.
const RIM: Blend = Blend::new(0.3);

#[inline(always)]
fn smin<T: Real>(a: T, b: T, blend: &Blend) -> T {
    let h = (T::splat(0.5) + blend.slope.scale(b - a)).clamp(0.0, 1.0);
    b * (T::splat(1.0) - h) + a * h - blend.k.scale(h * (T::splat(1.0) - h))
}

#[inline(always)]
fn smax<T: Real>(a: T, b: T, blend: &Blend) -> T {
    -smin(-a, -b, blend)
}
