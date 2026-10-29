use crate::pipeline::HIST_BINS;
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
use crate::pipeline::{Frame, N_FEATURES, assemble};
use crate::sculpt::Params;
use crate::work::Nonce;
use crate::topology::{N_FACES, N_VERTS};

/// Threads in a scanning block, all on one nonce so the body fits in shared
/// memory and the sums in registers. A lane per nonce would stall on DRAM.
pub const THREADS: u32 = 128;

pub const WARPS: usize = THREADS as usize / 32;

/// Harmonics from l = 1 to 6, weighted by area and then by area times radius.
pub const SH: usize = 96;

/// What every nonce of a launch shares. It rides in the launch by value,
/// since a copy from pageable host memory waits for the stream to drain and
/// would stall each launch behind the one before it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Task {
    pub pre_hash: [u8; 32],
    /// The launch's first nonce. Nonce `i` of the launch adds `i`.
    pub nonce: Nonce,
    pub threshold: [u8; 32],
}

/// What one nonce's scan hands to the finish kernel.
#[repr(C)]
pub struct Sums {
    /// Area, then six times the volume, 24 times the first moments and 120
    /// times the second.
    pub body: [f64; 11],
    /// Radius weighted by area.
    pub radius: f64,
    pub sh: [[f32; SH]; WARPS],
    pub hist: [[f64; HIST_BINS]; WARPS],
}

#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
impl Sums {
    /// The nonce's raw features from sums read back off the device, or `None`
    /// when its body came out unmeasurable, worked out as `poscan_finish` does.
    /// The kernel keeps its own copy, since sharing this one reorders its AMD
    /// code.
    pub fn features(&self) -> Option<[f64; N_FEATURES]> {
        let frame = Frame::new(&self.body)?;
        let area = self.body[0];
        let warps = |get: &dyn Fn(usize) -> f64| (0..WARPS).fold(0.0, |s, w| s + get(w));
        Some(assemble(
            &frame,
            area,
            self.radius / area,
            |i| (warps(&|w| self.sh[w][i - 1] as f64), warps(&|w| self.sh[w][SH / 2 + i - 1] as f64)),
            |j| warps(&|w| self.hist[w][j]),
        ))
    }
}

/// Shared memory, the sculpted vertices, the block reduction slots, and then
/// the nonce's parameters while sculpting or each warp's list of split faces
/// while sampling.
const VERTS: u32 = 0;
const RED: u32 = VERTS + (N_VERTS * 12) as u32;
const PARAMS: u32 = RED + (WARPS * 11 * 8) as u32;
#[cfg(any(target_arch = "nvptx64", target_arch = "amdgpu"))]
const LIST: u32 = PARAMS;
const LIST_BYTES: usize = WARPS * 32 * 8;

/// The noise gradient table, 16 rows of 16 bytes.
const GRADS: u32 = PARAMS + if size_of::<Params>() > LIST_BYTES { size_of::<Params>() } else { LIST_BYTES } as u32;
const _: () = assert!(GRADS.is_multiple_of(16));

/// Which lobes and craters may reach each patch, a u16 per patch.
const NEAR: u32 = GRADS + 256;

/// Shared bytes one scanning block takes.
pub const SHARED: u32 = NEAR + (PATCHES * 2).next_multiple_of(16) as u32;

/// Runs of 32 consecutive blank vertices, one warp's share of a sculpt step.
/// Subdivision numbers neighbours together, so a run covers one small patch
/// of the sphere.
pub const PATCHES: usize = N_VERTS.div_ceil(32);

/// Samples a warp may stream from its walk to its sums. A nonce takes about
/// 7300 over its four warps, and no body measured has come near this.
pub const STREAM: usize = 4096;

/// Bytes of sample stream each resident scanning block owns, a 16 byte
/// direction and radius record and a weight per sample.
pub const STREAM_BYTES: usize = WARPS * STREAM * 20;

/// Threads per block of the draw and finish kernels, a nonce each. Both run
/// serial f64 code, which in the scan would keep a block waiting on one thread.
pub const SERIAL_BLOCK: u32 = 128;

/// Hits retained from one launch. Pool difficulty normally produces far less
/// than one, but a large batch at a low starting difficulty can contain
/// several. The original single slot silently discarded all but the first.
pub const MAX_HITS: usize = 16;

/// Shared bytes the draw kernel takes per thread, its permutation strip.
pub const DRAW_SHARED: u32 = 256;

/// The blank as the scan kernel reads it.
///
/// Vertices come renumbered so that each run of 32, one warp's share of a
/// sculpt step, sits in one small patch of the sphere, which lets the warp
/// skip together the lobes and craters that cannot reach it. Renumbering moves
/// no value, so the scan comes out bit for bit the same. A direction is its
/// high halves and then its low halves, each padded to one 16 byte load, a
/// face is three u16 vertex indices in one u64, and a patch is the center and
/// chord radius of the ball around its run.
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
pub struct Packed {
    pub dirs: alloc::vec::Vec<f32>,
    pub tris: alloc::vec::Vec<u64>,
    pub patches: alloc::vec::Vec<f32>,
}

#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
pub fn pack(base: &[f32], faces: &[u32]) -> Packed {
    use alloc::vec::Vec;

    let dir = |i: usize| -> [f64; 3] {
        core::array::from_fn(|k| base[3 * i + k] as f64 + base[3 * (N_VERTS + i) + k] as f64)
    };
    let mut order: Vec<usize> = (0..N_VERTS).collect();
    bisect(&mut order, &dir);
    let mut renamed = alloc::vec![0u64; N_VERTS];
    for (new, &old) in order.iter().enumerate() {
        renamed[old] = new as u64;
    }

    let dirs = order
        .iter()
        .flat_map(|&i| {
            let (hi, lo) = (&base[3 * i..3 * i + 3], &base[3 * (N_VERTS + i)..3 * (N_VERTS + i) + 3]);
            [hi[0], hi[1], hi[2], 0.0, lo[0], lo[1], lo[2], 0.0]
        })
        .collect();
    let tris = faces[..3 * N_FACES]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|f| renamed[f[0] as usize] | renamed[f[1] as usize] << 16 | renamed[f[2] as usize] << 32)
        .collect();
    let patches = order
        .chunks(32)
        .flat_map(|run| {
            let n = run.len() as f64;
            let mut c = [0.0f64; 3];
            for &i in run {
                let d = dir(i);
                for k in 0..3 {
                    c[k] += d[k] / n;
                }
            }
            let r = run
                .iter()
                .map(|&i| {
                    let d = dir(i);
                    libm::sqrt((0..3).map(|k| (d[k] - c[k]) * (d[k] - c[k])).sum())
                })
                .fold(0.0, f64::max);
            [c[0] as f32, c[1] as f32, c[2] as f32, (r * (1.0 + 1e-6)) as f32]
        })
        .collect();
    Packed { dirs, tris, patches }
}

/// Order `ids` into runs of 32 by cutting at the median of the widest axis,
/// over and over, with every cut on a multiple of 32.
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
fn bisect(ids: &mut [usize], dir: &impl Fn(usize) -> [f64; 3]) {
    let runs = ids.len().div_ceil(32);
    if runs <= 1 {
        return;
    }
    let extent = |k: usize| {
        let (lo, hi) = ids.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &i| (lo.min(dir(i)[k]), hi.max(dir(i)[k])));
        hi - lo
    };
    let axis = (0..3).max_by(|&a, &b| extent(a).total_cmp(&extent(b))).unwrap_or(0);
    ids.sort_by(|&a, &b| dir(a)[axis].total_cmp(&dir(b)[axis]));
    let (left, right) = ids.split_at_mut(32 * (runs / 2));
    bisect(left, dir);
    bisect(right, dir);
}

#[cfg(any(target_arch = "nvptx64", target_arch = "amdgpu"))]
mod kernels {
    use super::*;
    use crate::intrinsics::{ceilf, fmaf, sqrtf};
    use crate::pipeline::{
        Body, Frame, HIST_STEP, Sum, TARGET_SAMPLES, assemble, centroid, cross, dot, real_sh6, sub,
    };
    use crate::simt;
    use crate::split::Split;
    use crate::work::{derive_seed, hash_buckets, meets, nonce_add, quantize};
    use crate::perlin::{self, Lattice, Perm};
    use crate::sculpt::PAIRS;

    /// The noise lattice of the nonce in shared memory, its permutation
    /// inside the parameters and its gradients in their own table, each read
    /// by 32 bit address.
    struct Shared;

    impl Lattice<f32> for Shared {
        #[inline(always)]
        fn pair(&self, i: u32) -> (u32, u32) {
            let v = unsafe { simt::ld_u16(PARAMS + PAIRS as u32 + 2 * (i & 255)) } as u32;
            (v & 0xff, v >> 8)
        }

        #[inline(always)]
        fn grad(&self, hash: u32, x: f32, y: f32, z: f32) -> f32 {
            let [gx, gy, gz, _] = unsafe { simt::ld_v4(GRADS + 16 * (hash & 15)) };
            fmaf(gx, x, fmaf(gy, y, gz * z))
        }
    }

    /// A thread's permutation strip, starting at its thread index, its bytes
    /// interleaved with the block's other threads so a warp touching the same
    /// entry hits distinct banks.
    struct Strip(u32);

    impl Perm for Strip {
        #[inline(always)]
        fn get(&self, i: usize) -> u8 {
            unsafe { simt::ld_u8(self.0 + i as u32 * SERIAL_BLOCK) }
        }

        #[inline(always)]
        fn set(&mut self, i: usize, v: u8) {
            unsafe { simt::st_u8(self.0 + i as u32 * SERIAL_BLOCK, v) };
        }
    }

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Quad([f32; 4]);

    #[inline(always)]
    fn dir(dirs: *const Quad, i: usize) -> [Split; 3] {
        let (hi, lo) = unsafe { (*dirs.add(2 * i), *dirs.add(2 * i + 1)) };
        core::array::from_fn(|k| Split { hi: hi.0[k], lo: lo.0[k] })
    }

    #[inline(always)]
    fn tri(tris: *const u64, t: usize) -> [usize; 3] {
        unpack(unsafe { *tris.add(t) })
    }

    #[inline(always)]
    fn global_id() -> usize {
        (simt::ctaid() * SERIAL_BLOCK + simt::tid()) as usize
    }

    #[inline(always)]
    fn vert(i: usize) -> [f32; 3] {
        let a = VERTS + 12 * i as u32;
        unsafe { [simt::ld_f32(a), simt::ld_f32(a + 4), simt::ld_f32(a + 8)] }
    }

    #[inline(always)]
    fn set_vert(i: usize, v: [f32; 3]) {
        let a = VERTS + 12 * i as u32;
        unsafe {
            simt::st_f32(a, v[0]);
            simt::st_f32(a + 4, v[1]);
            simt::st_f32(a + 8, v[2]);
        }
    }

    /// One thread per nonce, parameters drawn into `params`.
    #[unsafe(no_mangle)]
    #[allow(improper_gpu_kernel_arg)]
    pub unsafe extern "gpu-kernel" fn poscan_draw(task: Task, count: u32, params: *mut Params) {
        let id = global_id();
        if id >= count as usize {
            return;
        }
        let seed = derive_seed(&task.pre_hash, &nonce_add(&task.nonce, id as u64));
        unsafe { &mut *params.add(id) }.fill(seed, &mut Strip(simt::tid()));
    }

    /// Resident blocks scanning a nonce each, round robin over the launch. A
    /// block owns stream slot `ctaid` for its whole life, which is why it
    /// loops rather than retiring after one nonce.
    #[unsafe(no_mangle)]
    pub unsafe extern "gpu-kernel" fn poscan_scan(
        dirs: *const f32,
        tris: *const u64,
        patches: *const [f32; 4],
        params: *const Params,
        sums: *mut Sums,
        stream: *mut u8,
        count: u32,
    ) {
        let t = simt::tid() as usize;
        let dirs = dirs as *const Quad;
        let warp = t / 32;
        let lane = simt::laneid() as usize;
        let mine = unsafe { stream.add((simt::ctaid() as usize * WARPS + warp) * STREAM * 20) };
        let stream = Stream { quads: mine as *mut Quad, weights: unsafe { mine.add(STREAM * 16) } as *mut f32 };

        if t < 64 {
            let row = &perlin::GRADS[t / 4].0;
            unsafe { simt::st_f32(GRADS + 4 * t as u32, row[t % 4]) };
        }

        let mut n = simt::ctaid() as usize;
        let blocks = simt::nctaid() as usize;
        while n < count as usize {
            let out = unsafe { &mut *sums.add(n) };

            // Parameters go to shared memory, since every vertex rereads them
            // and L1 holds little once the body takes most of the SM's shared
            // window.
            let words = unsafe { params.add(n) } as *const u32;
            let mut w = t;
            while w < size_of::<Params>() / 4 {
                unsafe { simt::st_u32(PARAMS + 4 * w as u32, *words.add(w)) };
                w += THREADS as usize;
            }
            simt::sync();
            let params = unsafe { &*simt::shared::<Params>(PARAMS) };
            let mut p = t;
            while p < PATCHES {
                let [x, y, z, r] = unsafe { *patches.add(p) };
                unsafe { simt::st_u16(NEAR + 2 * p as u32, params.near([x, y, z], r) as u16) };
                p += THREADS as usize;
            }
            simt::sync();

            // Each loop loads the next item before working on this one, which
            // takes the L2 round trip for the blank off the critical path.
            let mut i = t;
            let mut next = dir(dirs, i);
            while i < N_VERTS {
                let d = next;
                if i + (THREADS as usize) < N_VERTS {
                    next = dir(dirs, i + THREADS as usize);
                }
                let near = unsafe { simt::ld_u16(NEAR + 2 * (i / 32) as u32) } as u32;
                set_vert(i, params.displace_with(d, &Shared, near));
                i += THREADS as usize;
            }
            simt::sync();

            // Two faces a step, which gives the scheduler two independent load
            // and square root chains to overlap. The sums still take the faces
            // in the same order.
            const STEP: usize = 2 * THREADS as usize;
            const _: () = assert!(N_FACES % STEP == 0);
            let mut body = Body::zero();
            let mut f = t;
            let mut next = [tri(tris, f), tri(tris, f + THREADS as usize)];
            while f < N_FACES {
                let [p, q] = next.map(|ix| ix.map(vert));
                if f + STEP < N_FACES {
                    next = [tri(tris, f + STEP), tri(tris, f + STEP + THREADS as usize)];
                }
                body.add(p[0], p[1], p[2]);
                body.add(q[0], q[1], q[2]);
                f += STEP;
            }
            let body = block_sum(body.get());
            if t == 0 {
                out.body = body;
            }

            let area = body[0];
            let c = centroid(&body);
            let per = (area / TARGET_SAMPLES as f64) as f32;

            // The walk lays each warp's samples out in its stream, and the sums
            // replay them. Apart, the walk's face latency and the sums' 96
            // running totals no longer share one register budget, and the sums
            // read the stream in lock step with every lane busy.
            let streamed = walk(tris, per, c, stream);
            simt::syncwarp();

            let mut acc = [0.0f32; SH];
            let mut radius = Sum::zero();
            let mut s = lane;
            while s < streamed {
                let Quad([x, y, z, r]) = unsafe { *stream.quads.add(s) };
                let w = unsafe { *stream.weights.add(s) };
                let wr = w * r;
                radius.add(wr);
                let y = real_sh6([x, y, z]);
                for i in 0..SH / 2 {
                    acc[i] = fmaf(w, y[1 + i], acc[i]);
                    acc[SH / 2 + i] = fmaf(wr, y[1 + i], acc[SH / 2 + i]);
                }
                s += 32;
            }
            let sh = warp_sum96(acc);
            for k in 0..3 {
                out.sh[warp][3 * lane + k] = sh[k];
            }
            let [radius] = block_sum([radius.get()]);
            if t == 0 {
                out.radius = radius;
            }

            let rbar = radius / area;
            let bins = Split::new(1.0 / (HIST_STEP * rbar));
            let mut hist = [0.0f32; HIST_BINS];
            let mut s = lane;
            while s < streamed {
                let r = unsafe { (*stream.quads.add(s)).0[3] };
                let w = unsafe { *stream.weights.add(s) };
                let x = (bins.scale(r) - 0.5).clamp(0.0, (HIST_BINS - 1) as f32);
                for b in 0..HIST_BINS {
                    hist[b] = fmaf(w, (1.0 - (x - b as f32).abs()).max(0.0), hist[b]);
                }
                s += 32;
            }
            let h = warp_sum8(hist);
            if lane & 3 == 0 {
                out.hist[warp][lane >> 2] = h;
            }
            n += blocks;
        }
    }

    /// One thread per nonce, sums into features, work and the target test.
    #[unsafe(no_mangle)]
    #[allow(improper_gpu_kernel_arg)]
    pub unsafe extern "gpu-kernel" fn poscan_finish(
        sums: *const Sums,
        task: Task,
        count: u32,
        out_nonce: *mut u64,
        out_work: *mut u8,
        out_count: *mut u32,
    ) {
        let id = global_id();
        if id >= count as usize {
            return;
        }
        let s = unsafe { &*sums.add(id) };
        let Some(frame) = Frame::new(&s.body) else {
            return;
        };
        let area = s.body[0];
        let f = assemble(
            &frame,
            area,
            s.radius / area,
            |i| (warps(|w| s.sh[w][i - 1] as f64), warps(|w| s.sh[w][SH / 2 + i - 1] as f64)),
            |j| warps(|w| s.hist[w][j]),
        );

        let w = hash_buckets(&quantize(&f), &task.pre_hash);
        if !meets(&w, &task.threshold) {
            return;
        }
        let nonce = nonce_add(&task.nonce, id as u64);
        unsafe {
            let slot = simt::reserve(out_count) as usize;
            if slot < MAX_HITS {
                for (i, limb) in nonce.into_iter().enumerate() {
                    *out_nonce.add(4 * slot + i) = limb;
                }
                for (i, b) in w.into_iter().enumerate() {
                    *out_work.add(32 * slot + i) = b;
                }
            }
        }
    }

    #[inline(always)]
    fn warps(mut get: impl FnMut(usize) -> f64) -> f64 {
        let mut s = 0.0;
        for w in 0..WARPS {
            s += get(w);
        }
        s
    }

    /// A face relative to `c`, its first vertex and two edges, and its area.
    #[inline(always)]
    fn face_at(ix: [usize; 3], c: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3], f32) {
        let [p0, p1, p2] = ix.map(vert);
        let (e1, e2) = (sub(p1, p0), sub(p2, p0));
        let n = cross(e1, e2);
        (sub(p0, c), e1, e2, 0.5 * sqrtf(dot(n, n)))
    }

    /// Where a warp's samples go, direction and radius as one record and the
    /// weight beside it.
    #[derive(Clone, Copy)]
    struct Stream {
        quads: *mut Quad,
        weights: *mut f32,
    }

    /// Walk this warp's samples relative to `c` into its stream and return how
    /// many it wrote. Faces go round robin over the block, so a warp takes 32
    /// neighbouring faces at a time.
    ///
    /// Every face's first sample runs in step across the warp. The other three
    /// samples of each face split in two per side, about one face in seven,
    /// then run flattened over the warp's lanes rather than each lane walking
    /// its own face while the rest idle. Those faces pass through shared
    /// memory as vertex indices and are rebuilt.
    #[inline(always)]
    fn walk(tris: *const u64, per: f32, c: [f32; 3], stream: Stream) -> usize {
        let lane = simt::laneid();
        let mut streamed = 0;
        let mut t = simt::tid() as usize;
        let mut next = unsafe { *tris.add(t) };
        while t < N_FACES {
            let packed = next;
            if t + (THREADS as usize) < N_FACES {
                next = unsafe { *tris.add(t + THREADS as usize) };
            }
            let (a, e1, e2, area) = face_at(unpack(packed), c);
            // k = ceil(sqrt(area / per)), the subdivision per side, by compares
            // for the two sizes every measured face takes.
            let k = if area <= per {
                1
            } else if area <= 4.0 * per {
                2
            } else {
                (ceilf(sqrtf(area / per)) as u32).clamp(1, 64)
            };

            let split = simt::ballot(k == 2);
            if k == 2 {
                unsafe { simt::st_u64(list() + 8 * (split & simt::lanemask_lt()).count_ones(), packed) };
            }
            let big = simt::ballot(k > 2);
            let (w, step) = match k {
                1 => (area, 1.0 / 3.0),
                2 => (area * 0.25, 1.0 / 6.0),
                _ => (area / (k * k) as f32, 1.0 / (3 * k) as f32),
            };
            streamed = put(stream, streamed, bary(a, e1, e2, step, 0, 0, false), w, true);

            simt::syncwarp();
            let extras = 3 * split.count_ones();
            let mut s = lane;
            while s < extras.next_multiple_of(32) {
                let live = s < extras;
                let packed = unsafe { simt::ld_u64(list() + 8 * if live { s / 3 } else { 0 }) };
                let (a, e1, e2, area) = face_at(unpack(packed), c);
                let (i, j, down) = SPLIT_TWO[(s % 3) as usize];
                streamed = put(stream, streamed, bary(a, e1, e2, 1.0 / 6.0, i, j, down), area * 0.25, live);
                s += 32;
            }
            simt::syncwarp();

            // A face split three or more ways per side, which no measured body
            // has had, takes the whole warp to itself.
            let mut big = big;
            while big != 0 {
                let o = big.trailing_zeros();
                big &= big - 1;
                let ko = simt::shfl_idx(k, o);
                let face = simt::shfl_idx(t as u32, o) as usize;
                let (a, e1, e2, area) = face_at(unpack(unsafe { *tris.add(face) }), c);
                let mut q = 1 + lane;
                while q < 1 + (ko * ko - 1).next_multiple_of(32) {
                    let live = q < ko * ko;
                    let (i, j, down) = subsample(if live { q } else { 0 }, ko);
                    streamed = put(stream, streamed, bary(a, e1, e2, 1.0 / (3 * ko) as f32, i, j, down),
                                   area / (ko * ko) as f32, live);
                    q += 32;
                }
            }
            t += THREADS as usize;
        }
        streamed.min(STREAM)
    }

    /// Append one sample to the warp's stream where `live`, as its direction,
    /// radius and weight. Every lane of the warp calls it together.
    #[inline(always)]
    fn put(stream: Stream, streamed: usize, p: [f32; 3], w: f32, live: bool) -> usize {
        let taken = simt::ballot(live);
        let slot = streamed + (taken & simt::lanemask_lt()).count_ones() as usize;
        if live && slot < STREAM {
            let r = sqrtf(dot(p, p));
            let inv = simt::recip(r);
            unsafe {
                *stream.quads.add(slot) = Quad([p[0] * inv, p[1] * inv, p[2] * inv, r]);
                *stream.weights.add(slot) = w;
            }
        }
        streamed + taken.count_ones() as usize
    }

    /// This warp's list of split faces.
    #[inline(always)]
    fn list() -> u32 {
        LIST + 8 * 32 * (simt::tid() / 32)
    }

    #[inline(always)]
    fn unpack(v: u64) -> [usize; 3] {
        [(v & 0xffff) as usize, (v >> 16 & 0xffff) as usize, (v >> 32 & 0xffff) as usize]
    }

    /// Samples one to three of a face split in two per side.
    const SPLIT_TWO: [(u32, u32, bool); 3] = [(0, 0, true), (0, 1, false), (1, 0, false)];

    /// Centroid of sub triangle `(i, j)`, the downward one when `down`, of a
    /// face split `1 / (3 step)` ways per side.
    #[inline(always)]
    fn bary(a: [f32; 3], e1: [f32; 3], e2: [f32; 3], step: f32, i: u32, j: u32, down: bool) -> [f32; 3] {
        let d = if down { 2 } else { 1 };
        let u = (3 * i + d) as f32 * step;
        let v = (3 * j + d) as f32 * step;
        core::array::from_fn(|q| fmaf(v, e2[q], fmaf(u, e1[q], a[q])))
    }

    /// Sub triangle `q` of a face split `k` ways per side, in the chain's
    /// order, upward triangles row by row with each one's downward neighbour
    /// right after it.
    #[inline(always)]
    fn subsample(q: u32, k: u32) -> (u32, u32, bool) {
        let (mut i, mut j, mut down) = (0, 0, false);
        for _ in 0..q {
            if !down && i + j + 1 < k {
                down = true;
            } else {
                down = false;
                j += 1;
                if j == k - i {
                    j = 0;
                    i += 1;
                }
            }
        }
        (i, j, down)
    }

    /// Sum each value over the block. Every thread gets the same totals, added
    /// in the same order.
    #[inline(always)]
    fn block_sum<const N: usize>(mut v: [f64; N]) -> [f64; N] {
        let mut m = 16;
        while m > 0 {
            for i in 0..N {
                v[i] += shfl_xor_f64(v[i], m);
            }
            m >>= 1;
        }
        let warp = simt::tid() / 32;
        if simt::laneid() == 0 {
            for (i, x) in v.iter().enumerate() {
                unsafe { simt::st_f64(RED + 8 * (warp * N as u32 + i as u32), *x) };
            }
        }
        simt::sync();
        let total = core::array::from_fn(|i| {
            let mut s = 0.0;
            for w in 0..WARPS {
                s += unsafe { simt::ld_f64(RED + 8 * (w * N + i) as u32) };
            }
            s
        });
        simt::sync();
        total
    }

    /// Halve a warp's values. Lanes with `mask` set keep the upper half, the
    /// others the lower, and each adds in its partner's copy of the half it
    /// kept.
    #[inline(always)]
    fn fold<const N: usize, const H: usize>(v: [f32; N], mask: u32) -> [f32; H] {
        let upper = simt::laneid() & mask != 0;
        core::array::from_fn(|i| {
            let (keep, send) = if upper { (v[H + i], v[i]) } else { (v[i], v[H + i]) };
            keep + f32::from_bits(simt::shfl_xor(send.to_bits(), mask))
        })
    }

    /// Sum 96 values over the warp. Lane `l` comes out holding the totals of
    /// values 3l, 3l + 1 and 3l + 2.
    ///
    /// All in f32. An f64 tail over the last three halvings gains 0.2% of
    /// retention and costs 2.3% of throughput on an RTX 5070 Ti.
    #[inline(always)]
    fn warp_sum96(v: [f32; SH]) -> [f32; 3] {
        let v: [f32; 48] = fold(v, 16);
        let v: [f32; 24] = fold(v, 8);
        let v: [f32; 12] = fold(v, 4);
        let v: [f32; 6] = fold(v, 2);
        fold(v, 1)
    }

    /// Sum the histogram over the warp. Lane `l` comes out holding bin l / 4.
    ///
    /// In f64, since the last few adds of an f32 tree join halves of a total
    /// and round at that size, which on the fullest bins costs more than every
    /// lane's own running sum.
    #[inline(always)]
    fn warp_sum8(v: [f32; HIST_BINS]) -> f64 {
        let v = v.map(|x| x as f64);
        let v: [f64; 4] = fold64(v, 16);
        let v: [f64; 2] = fold64(v, 8);
        let [x]: [f64; 1] = fold64(v, 4);
        let x = x + shfl_xor_f64(x, 2);
        x + shfl_xor_f64(x, 1)
    }

    /// `fold` in f64.
    #[inline(always)]
    fn fold64<const N: usize, const H: usize>(v: [f64; N], mask: u32) -> [f64; H] {
        let upper = simt::laneid() & mask != 0;
        core::array::from_fn(|i| {
            let (keep, send) = if upper { (v[H + i], v[i]) } else { (v[i], v[H + i]) };
            keep + shfl_xor_f64(send, mask)
        })
    }

    #[inline(always)]
    fn shfl_xor_f64(v: f64, mask: u32) -> f64 {
        let bits = v.to_bits();
        let lo = simt::shfl_xor(bits as u32, mask) as u64;
        let hi = simt::shfl_xor((bits >> 32) as u32, mask) as u64;
        f64::from_bits(hi << 32 | lo)
    }
}
