use anyhow::{Context, Result, bail};
use cudarc::driver::sys::{self, CUdevice_attribute, CUfunction_attribute};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PinnedHostSlice, PushKernelArg, result,
};
use cudarc::nvrtc::Ptx;
use include_bytes_aligned::include_bytes_aligned;
use libloading::Library;
use numen_kernel::coop::{self, MAX_HITS, Packed, SERIAL_BLOCK, Sums};
use numen_kernel::pipeline::N_FEATURES;
use numen_kernel::sculpt::Params;
use nvml_wrapper::Nvml;
use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::ptr::null_mut;
use std::sync::Arc;
use std::time::Duration;

use crate::autotune;
use crate::engine::{Candidate, Job, Meter, Shared};
use crate::telemetry::{AmdSensors, Sensor};

/// Nonces per launch per SM or CU. The draw and finish kernels run a thread
/// per nonce and wait on latency when a launch is small, while a launch at
/// this size still turns over in about 25 ms on an RTX 5070 Ti.
const NONCES_PER_UNIT: u32 = 480;

const MODERN_PTX: &str = include_str!("../ptx/modern.ptx");
const LEGACY_PTX: &str = include_str!("../ptx/legacy.ptx");

#[cfg(windows)]
const LIBRARIES: &[&str] = &["amdhip64_8.dll", "amdhip64_7.dll", "amdhip64_6.dll"];
#[cfg(not(windows))]
const LIBRARIES: &[&str] = &["libamdhip64.so.8", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5"];

const MULTIPROCESSOR_COUNT: c_int = 63;
const PCI_BUS_ID: c_int = 67;
const PCI_DEVICE_ID: c_int = 68;
const PCI_DOMAIN_ID: c_int = 69;

const FUNC_LOCAL_SIZE_BYTES: c_int = 3;
const FUNC_NUM_REGS: c_int = 4;

macro_rules! code_objects {
    ($($gfx:literal),* $(,)?) => {
        [$(($gfx, include_bytes_aligned!(8, concat!("../gfx/", $gfx, ".hsaco")) as &[u8])),*]
    };
}

const CODE_OBJECTS: [(&str, &[u8]); 24] = code_objects!(
    "gfx1010", "gfx1011", "gfx1012",
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    "gfx1200", "gfx1201",
    "gfx803", "gfx900", "gfx906", "gfx90c",
);

type Launch = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

pub enum Gpu {
    Cuda(CudaDevice),
    Hip(HipDevice),
}

/// Every GPU, NVIDIA first and each vendor in PCI order. Positions in this
/// list are the numbers `--devices` takes.
pub fn gpus() -> Vec<Gpu> {
    let cuda = cuda_devices().into_iter().map(Gpu::Cuda);
    let hip = hip_devices().into_iter().map(Gpu::Hip);
    cuda.chain(hip).collect()
}

/// Nonces per launch for a GPU with this many multiprocessors or compute
/// units, when `--batch` leaves it out.
fn default_lanes(units: u32) -> u32 {
    NONCES_PER_UNIT * units
}

/// Batch sizes worth measuring around the architecture-based starting point.
/// Keep the largest allocation below a quarter of VRAM so tuning cannot crowd
/// out the desktop or other miner state. The base is always retained.
fn autotune_lanes(base: u32, memory: usize, resident: u32) -> Vec<u32> {
    let mut lanes = [base / 2, base, base.saturating_mul(2), base.saturating_mul(4)]
        .into_iter()
        .filter(|&candidate| candidate != 0)
        .filter(|&candidate| candidate == base || state_bytes(candidate, resident) <= memory / 4)
        .collect::<Vec<_>>();
    lanes.sort_unstable();
    lanes.dedup();
    lanes
}

fn cuda_driver_version() -> i32 {
    let mut version = 0;
    let _ = unsafe { sys::cuDriverGetVersion(&mut version) };
    version
}

/// PCI location as `bus:device.function`, the way nvidia-smi and lspci print it.
fn bus(pci: [i32; 3]) -> String {
    format!("{:02x}:{:02x}.0", pci[1], pci[2])
}

fn gib(bytes: usize) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn mib(bytes: usize) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// Device bytes a launch of `lanes` nonces keeps, with `resident` scanning
/// blocks.
fn state_bytes(lanes: u32, resident: u32) -> usize {
    lanes as usize * (size_of::<Params>() + size_of::<Sums>()) + resident as usize * coop::STREAM_BYTES
}

pub struct CudaDevice {
    ordinal: usize,
    name: String,
    compute: (i32, i32),
    sms: u32,
    memory: usize,
    pci: [i32; 3],
}

impl CudaDevice {
    pub fn default_lanes(&self) -> u32 {
        default_lanes(self.sms)
    }

    pub fn bus(&self) -> String {
        bus(self.pci)
    }

    pub fn sensor(&self, nvml: &Arc<Nvml>) -> Sensor {
        let [domain, bus, device] = self.pci;
        Sensor::Nvml { nvml: nvml.clone(), bus: format!("{domain:x}:{bus:x}:{device:x}.0") }
    }

    /// Cells for one row of the `numen-miner devices` table.
    pub fn row(&self, number: usize) -> [String; 7] {
        let (major, minor) = self.compute;
        [
            number.to_string(),
            self.name.clone(),
            bus(self.pci),
            format!("{major}.{minor}"),
            self.sms.to_string(),
            gib(self.memory),
            self.default_lanes().to_string(),
        ]
    }
}

impl fmt::Display for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (major, minor) = self.compute;
        write!(
            f,
            "{}, bus {}, compute {major}.{minor}, {} SMs, {}",
            self.name,
            bus(self.pci),
            self.sms,
            gib(self.memory)
        )
    }
}

/// Every CUDA device in PCI order, the order nvidia-smi lists them in.
fn cuda_devices() -> Vec<CudaDevice> {
    if !unsafe { sys::is_culib_present() } {
        log::info!("no NVIDIA driver, skipped");
        return Vec::new();
    }
    let count = match CudaContext::device_count() {
        Ok(count) => count,
        Err(e) => {
            log::warn!("CUDA: {e}");
            return Vec::new();
        }
    };
    let mut devices: Vec<CudaDevice> = (0..count)
        .filter_map(|ordinal| match probe_cuda(ordinal) {
            Ok(device) => Some(device),
            Err(e) => {
                log::warn!("CUDA device {ordinal}: {e}");
                None
            }
        })
        .collect();
    devices.sort_by_key(|device| device.pci);
    devices
}

fn probe_cuda(ordinal: i32) -> Result<CudaDevice, result::DriverError> {
    let dev = result::device::get(ordinal)?;
    let attribute = |attr| unsafe { result::device::get_attribute(dev, attr) };
    Ok(CudaDevice {
        ordinal: ordinal as usize,
        name: result::device::get_name(dev)?,
        compute: (
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ),
        sms: attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32,
        memory: unsafe { result::device::total_mem(dev)? },
        pci: [
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?,
        ],
    })
}

/// A block per nonce, see `numen_kernel::coop`.
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    lanes: u32,
    draw: CudaFunction,
    scan: CudaFunction,
    finish: CudaFunction,
    /// Scanning blocks the device holds at once, the scan kernel's grid.
    resident: u32,
    readback: [Readback; 2],
    /// Readback slots of the launches still in flight, oldest first.
    queued: VecDeque<usize>,
    d_dirs: CudaSlice<f32>,
    d_tris: CudaSlice<u64>,
    d_patches: CudaSlice<f32>,
    d_params: CudaSlice<u8>,
    d_sums: CudaSlice<u8>,
    /// Sample stream, one slot per resident scanning block.
    d_stream: CudaSlice<u8>,
    d_out_nonce: CudaSlice<u64>,
    d_out_work: CudaSlice<u8>,
    d_out_count: CudaSlice<u32>,
}

/// Host copies of one launch's hit, which the stream fills while the next
/// launch runs.
struct Readback {
    count: PinnedHostSlice<u32>,
    nonce: PinnedHostSlice<u64>,
    work: PinnedHostSlice<u8>,
    /// Recorded behind the copies. The pinned slices carry events of their
    /// own, but cudarc records those only when it manages several streams, so
    /// on one stream they never hold the host back.
    copied: CudaEvent,
}

impl Readback {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        // Every field is written by a device copy before it is read.
        unsafe {
            Ok(Readback {
                count: ctx.alloc_pinned(1)?,
                nonce: ctx.alloc_pinned(4 * MAX_HITS)?,
                work: ctx.alloc_pinned(32 * MAX_HITS)?,
                copied: ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?,
            })
        }
    }

    fn hits(&self) -> Result<Vec<Candidate>> {
        self.copied.synchronize()?;
        let found = self.count.as_slice()?[0] as usize;
        if found > MAX_HITS {
            log::debug!("GPU hit buffer kept {MAX_HITS} of {found} hits; lower --batch or raise pool difficulty");
        }
        let nonces = self.nonce.as_slice()?;
        let works = self.work.as_slice()?;
        Ok((0..found.min(MAX_HITS))
            .map(|slot| {
                let mut hit = Candidate { nonce: [0; 4], work: [0; 32] };
                hit.nonce.copy_from_slice(&nonces[4 * slot..4 * slot + 4]);
                hit.work.copy_from_slice(&works[32 * slot..32 * slot + 32]);
                hit
            })
            .collect())
    }
}

#[repr(transparent)]
struct TaskArg(coop::Task);

// A plain C struct the kernels take by value, as `.param .align 8 .b8 [96]`.
unsafe impl DeviceRepr for TaskArg {}

impl CudaBackend {
    pub fn new(number: usize, device: &CudaDevice, lanes: Option<u32>, retune: bool) -> Result<Self> {
        let (major, minor) = device.compute;
        if major < 3 {
            bail!("{} is compute {major}.{minor} and CUDA mining needs 3.0 or newer", device.name);
        }
        let ctx = CudaContext::new(device.ordinal).context(
            "open CUDA device. GPU mining needs an NVIDIA driver from the 470 \
             series of 2021 on",
        )?;
        let requested_lanes = lanes;
        let (blob, src) = if major >= 7 { ("modern", MODERN_PTX) } else { ("legacy", LEGACY_PTX) };
        let module = ctx.load_module(Ptx::from_src(src)).context("load PTX")?;
        let stream = ctx.default_stream();

        let scan = module.load_function("poscan_scan").context("find scan kernel")?;
        scan.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100)?;
        let resident = device.sms
            * scan.occupancy_max_active_blocks_per_multiprocessor(coop::THREADS, coop::SHARED as usize, None)?;
        let candidates = requested_lanes.map_or_else(
            || autotune_lanes(device.default_lanes(), device.memory, resident),
            |lanes| vec![lanes],
        );
        let bus = device.bus();
        let architecture = format!("sm_{major}{minor}");
        let key = autotune::Key::new(autotune::Identity {
            backend: "cuda",
            bus: &bus,
            name: &device.name,
            architecture: &architecture,
            units: device.sms,
            memory: device.memory,
            runtime: cuda_driver_version(),
            kernel: autotune::fingerprint(&[src.as_bytes()]),
        });
        let cached = (requested_lanes.is_none() && !retune).then(|| autotune::cached(&key, &candidates)).flatten();
        let allocated_lanes = cached.as_ref().map(|tuning| tuning.lanes).unwrap_or(
            *candidates.iter().max().context("no CUDA batch-size candidate")?,
        );

        let Packed { dirs, tris, patches } = blank();
        let mut backend = CudaBackend {
            lanes: allocated_lanes,
            draw: module.load_function("poscan_draw").context("find draw kernel")?,
            finish: module.load_function("poscan_finish").context("find finish kernel")?,
            scan,
            resident,
            readback: [Readback::new(&ctx)?, Readback::new(&ctx)?],
            queued: VecDeque::new(),
            d_dirs: stream.clone_htod(&dirs).context("upload blank directions")?,
            d_tris: stream.clone_htod(&tris).context("upload blank faces")?,
            d_patches: stream.clone_htod(&patches).context("upload blank patches")?,
            d_params: stream
                .alloc_zeros::<u8>(allocated_lanes as usize * size_of::<Params>())
                .context("allocate parameters")?,
            d_sums: stream
                .alloc_zeros::<u8>(allocated_lanes as usize * size_of::<Sums>())
                .context("allocate sums")?,
            d_stream: stream
                .alloc_zeros::<u8>(resident as usize * coop::STREAM_BYTES)
                .context("allocate sample stream")?,
            d_out_nonce: stream.alloc_zeros::<u64>(4 * MAX_HITS)?,
            d_out_work: stream.alloc_zeros::<u8>(32 * MAX_HITS)?,
            d_out_count: stream.alloc_zeros::<u32>(1)?,
            stream,
        };
        if let Some(tuning) = cached {
            backend.lanes = tuning.lanes;
        } else if requested_lanes.is_none() {
            let choice = backend.autotune(number, device.default_lanes(), &candidates)?;
            if let Err(e) = autotune::save(key, &choice) {
                log::warn!("gpu{number} could not save autotune result: {e:#}");
            }
        }
        log::info!(
            "gpu{number} {device}, {blob} kernel, {} nonces per launch, {resident} resident blocks, \
             {} registers, {} B spilled, {} state allocation",
            backend.lanes,
            backend.scan.num_regs()?,
            backend.scan.get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            mib(state_bytes(allocated_lanes, resident))
        );
        Ok(backend)
    }

    pub fn lanes(&self) -> u32 {
        self.lanes
    }

    fn autotune(&mut self, number: usize, base: u32, candidates: &[u32]) -> Result<autotune::Choice> {
        let pre_hash = [0x5a; 32];
        let job = dump_job(&pre_hash, 0);
        let choice = autotune::run(number, base, candidates, |lanes| {
            self.lanes = lanes;
            self.launch(&job)?;
            self.stream.synchronize()?;
            Ok(())
        })?;
        self.lanes = choice.lanes;
        Ok(choice)
    }

    /// Queue a launch over `job` with its hit's readback behind it. The stream
    /// runs launches in order, so one queued behind another starts the moment
    /// the first ends, with no host round trip between them.
    fn enqueue(&mut self, job: &Job) -> Result<()> {
        let slot = self.queued.back().map_or(0, |s| 1 - s);
        self.launch(job)?;
        let readback = &mut self.readback[slot];
        self.stream.memcpy_dtoh(&self.d_out_count, &mut readback.count)?;
        self.stream.memcpy_dtoh(&self.d_out_nonce, &mut readback.nonce)?;
        self.stream.memcpy_dtoh(&self.d_out_work, &mut readback.work)?;
        readback.copied.record(&self.stream)?;
        self.queued.push_back(slot);
        // Under WDDM the driver batches launches until something flushes them,
        // so a queued launch can sit on the host while the GPU drains the one
        // before it. A stream query flushes without waiting.
        let _ = unsafe { sys::cuStreamQuery(self.stream.cu_stream()) };
        Ok(())
    }

    /// Wait for the oldest queued launch and return its hit.
    fn collect(&mut self) -> Result<Vec<Candidate>> {
        let slot = self.queued.pop_front().context("no launch in flight")?;
        self.readback[slot].hits()
    }

    fn launch(&mut self, job: &Job) -> Result<()> {
        self.stream.memset_zeros(&mut self.d_out_count)?;
        let task = TaskArg(coop::Task { pre_hash: job.pre_hash, nonce: job.nonce_start, threshold: job.threshold });
        let serial = LaunchConfig {
            grid_dim: (self.lanes.div_ceil(SERIAL_BLOCK), 1, 1),
            block_dim: (SERIAL_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut b = self.stream.launch_builder(&self.draw);
        b.arg(&task).arg(&self.lanes).arg(&mut self.d_params);
        unsafe { b.launch(LaunchConfig { shared_mem_bytes: coop::DRAW_SHARED * SERIAL_BLOCK, ..serial }) }
            .context("launch draw kernel")?;

        let blocks = LaunchConfig {
            grid_dim: (self.lanes.min(self.resident), 1, 1),
            block_dim: (coop::THREADS, 1, 1),
            shared_mem_bytes: coop::SHARED,
        };
        let mut b = self.stream.launch_builder(&self.scan);
        b.arg(&self.d_dirs)
            .arg(&self.d_tris)
            .arg(&self.d_patches)
            .arg(&self.d_params)
            .arg(&mut self.d_sums)
            .arg(&mut self.d_stream)
            .arg(&self.lanes);
        unsafe { b.launch(blocks) }.context("launch scan kernel")?;

        let mut b = self.stream.launch_builder(&self.finish);
        b.arg(&self.d_sums)
            .arg(&task)
            .arg(&self.lanes)
            .arg(&mut self.d_out_nonce)
            .arg(&mut self.d_out_work)
            .arg(&mut self.d_out_count);
        unsafe { b.launch(serial) }.context("launch finish kernel")?;
        Ok(())
    }

    /// Raw features of one launch's nonces from `start`, taken on the host
    /// from the scan's sums just as the finish kernel takes them.
    pub fn features(&mut self, pre_hash: &[u8; 32], start: u64) -> Result<Vec<Option<[f64; N_FEATURES]>>> {
        self.launch(&dump_job(pre_hash, start))?;
        Ok(features_from(&self.stream.clone_dtoh(&self.d_sums).context("read back sums")?))
    }
}

/// The blank as the scan kernel takes it.
fn blank() -> Packed {
    let (base, faces) = numen_kernel::topology::build();
    coop::pack(&base, &faces)
}

/// A launch over the nonces from `start`. A dump reads the sums, so the
/// threshold never matters.
fn dump_job(pre_hash: &[u8; 32], start: u64) -> Job {
    Job { pre_hash: *pre_hash, threshold: [0; 32], nonce_start: [start, 0, 0, 0], generation: 0 }
}

/// Features of every nonce in a launch, from its sums as they came off the
/// device.
fn features_from(sums: &[u8]) -> Vec<Option<[f64; N_FEATURES]>> {
    sums.as_chunks::<{ size_of::<Sums>() }>()
        .0
        .iter()
        // Sums is plain floats, so any bytes make a valid one.
        .map(|bytes| unsafe { bytes.as_ptr().cast::<Sums>().read_unaligned() }.features())
        .collect()
}

pub fn cuda_worker(shared: Arc<Shared>, meter: Arc<Meter>, mut backend: CudaBackend) {
    let lanes = backend.lanes as u64;
    // One launch stays queued behind the one being collected, so the GPU never
    // waits on the host.
    let mut pending: Option<Job> = None;
    while !shared.stopped() {
        let paused = meter.paused();
        let next = if paused { None } else { shared.next_job(lanes) };
        if let Some(job) = &next
            && let Err(e) = backend.enqueue(job)
        {
            log::error!("{}: {e:#}", meter.name);
            backend.queued.clear();
            pending = None;
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        if let Some(job) = pending.take() {
            match backend.collect() {
                Ok(hits) => {
                    meter.add(lanes);
                    for hit in hits {
                        shared.found(&meter, &job, hit);
                    }
                }
                Err(e) => log::error!("{}: {e:#}", meter.name),
            }
        }
        pending = next;
        if pending.is_none() {
            std::thread::sleep(Duration::from_millis(if paused { 1000 } else { 100 }));
        }
    }
}

/// The HIP runtime, bound at run time.
struct Hip {
    get_device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    runtime_get_version: unsafe extern "C" fn(*mut c_int) -> c_int,
    set_device: unsafe extern "C" fn(c_int) -> c_int,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int,
    device_total_mem: unsafe extern "C" fn(*mut usize, c_int) -> c_int,
    module_load_data: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int,
    module_get_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
    func_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> c_int,
    occupancy: unsafe extern "C" fn(*mut c_int, *mut c_void, c_int, usize) -> c_int,
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    host_malloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int,
    memcpy_htod: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int,
    memcpy_htod_async: unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> c_int,
    memcpy_dtoh_async: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *mut c_void) -> c_int,
    memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int,
    memset_async: unsafe extern "C" fn(*mut c_void, c_int, usize, *mut c_void) -> c_int,
    stream_create: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    event_create: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    event_record: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    event_synchronize: unsafe extern "C" fn(*mut c_void) -> c_int,
    launch_kernel: Launch,
    synchronize: unsafe extern "C" fn() -> c_int,
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    _library: Library,
}

impl Hip {
    fn load(name: &str) -> Result<Self, libloading::Error> {
        unsafe {
            let library = Library::new(name)?;
            Ok(Hip {
                get_device_count: *library.get(b"hipGetDeviceCount\0")?,
                runtime_get_version: *library.get(b"hipRuntimeGetVersion\0")?,
                set_device: *library.get(b"hipSetDevice\0")?,
                device_get_name: *library.get(b"hipDeviceGetName\0")?,
                device_get_attribute: *library.get(b"hipDeviceGetAttribute\0")?,
                device_total_mem: *library.get(b"hipDeviceTotalMem\0")?,
                module_load_data: *library.get(b"hipModuleLoadData\0")?,
                module_get_function: *library.get(b"hipModuleGetFunction\0")?,
                func_get_attribute: *library.get(b"hipFuncGetAttribute\0")?,
                occupancy: *library.get(b"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor\0")?,
                malloc: *library.get(b"hipMalloc\0")?,
                host_malloc: *library.get(b"hipHostMalloc\0")?,
                memcpy_htod: *library.get(b"hipMemcpyHtoD\0")?,
                memcpy_dtoh: *library.get(b"hipMemcpyDtoH\0")?,
                memcpy_htod_async: *library.get(b"hipMemcpyHtoDAsync\0")?,
                memcpy_dtoh_async: *library.get(b"hipMemcpyDtoHAsync\0")?,
                memset: *library.get(b"hipMemset\0")?,
                memset_async: *library.get(b"hipMemsetAsync\0")?,
                stream_create: *library.get(b"hipStreamCreate\0")?,
                event_create: *library.get(b"hipEventCreate\0")?,
                event_record: *library.get(b"hipEventRecord\0")?,
                event_synchronize: *library.get(b"hipEventSynchronize\0")?,
                launch_kernel: *library.get(b"hipModuleLaunchKernel\0")?,
                synchronize: *library.get(b"hipDeviceSynchronize\0")?,
                error_string: *library.get(b"hipGetErrorString\0")?,
                _library: library,
            })
        }
    }

    fn check(&self, code: c_int, call: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let message = unsafe { CStr::from_ptr((self.error_string)(code)) };
        bail!("{call}: {}", message.to_string_lossy())
    }

    fn device_count(&self) -> Result<c_int> {
        let mut count = 0;
        self.check(unsafe { (self.get_device_count)(&mut count) }, "hipGetDeviceCount")?;
        Ok(count)
    }

    fn runtime_version(&self) -> c_int {
        let mut version = 0;
        if unsafe { (self.runtime_get_version)(&mut version) } == 0 { version } else { 0 }
    }

    fn attribute(&self, attribute: c_int, ordinal: c_int) -> Result<c_int> {
        let mut value = 0;
        self.check(
            unsafe { (self.device_get_attribute)(&mut value, attribute, ordinal) },
            "hipDeviceGetAttribute",
        )?;
        Ok(value)
    }

    fn function(&self, module: *mut c_void, name: &CStr) -> Result<*mut c_void> {
        let mut func = null_mut();
        self.check(unsafe { (self.module_get_function)(&mut func, module, name.as_ptr()) }, "hipModuleGetFunction")?;
        Ok(func)
    }

    fn function_attribute(&self, func: *mut c_void, attribute: c_int) -> Result<c_int> {
        let mut value = 0;
        self.check(unsafe { (self.func_get_attribute)(&mut value, attribute, func) }, "hipFuncGetAttribute")?;
        Ok(value)
    }

    fn alloc(&self, bytes: usize) -> Result<*mut c_void> {
        let mut ptr = null_mut();
        self.check(unsafe { (self.malloc)(&mut ptr, bytes) }, "hipMalloc")?;
        Ok(ptr)
    }

    fn host_alloc<T>(&self) -> Result<*mut T> {
        let mut ptr = null_mut();
        self.check(unsafe { (self.host_malloc)(&mut ptr, size_of::<T>(), 0) }, "hipHostMalloc")?;
        unsafe { ptr.cast::<u8>().write_bytes(0, size_of::<T>()) };
        Ok(ptr.cast())
    }

    fn stream(&self) -> Result<*mut c_void> {
        let mut stream = null_mut();
        self.check(unsafe { (self.stream_create)(&mut stream) }, "hipStreamCreate")?;
        Ok(stream)
    }

    fn event(&self) -> Result<*mut c_void> {
        let mut event = null_mut();
        self.check(unsafe { (self.event_create)(&mut event) }, "hipEventCreate")?;
        Ok(event)
    }

    fn upload<T: Copy>(&self, dst: *mut c_void, src: &[T]) -> Result<()> {
        let bytes = size_of_val(src);
        self.check(unsafe { (self.memcpy_htod)(dst, src.as_ptr().cast(), bytes) }, "hipMemcpyHtoD")
    }

    fn clone_htod<T: Copy>(&self, src: &[T]) -> Result<*mut c_void> {
        let dst = self.alloc(size_of_val(src))?;
        self.upload(dst, src)?;
        Ok(dst)
    }

    fn download<T: Copy>(&self, dst: &mut [T], src: *mut c_void) -> Result<()> {
        let bytes = size_of_val(dst);
        self.check(unsafe { (self.memcpy_dtoh)(dst.as_mut_ptr().cast(), src, bytes) }, "hipMemcpyDtoH")
    }

    fn upload_async<T: Copy>(&self, dst: *mut c_void, src: *const T, count: usize, stream: *mut c_void) -> Result<()> {
        self.check(
            unsafe { (self.memcpy_htod_async)(dst, src.cast(), count * size_of::<T>(), stream) },
            "hipMemcpyHtoDAsync",
        )
    }

    fn download_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *mut c_void,
        count: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        self.check(
            unsafe { (self.memcpy_dtoh_async)(dst.cast(), src, count * size_of::<T>(), stream) },
            "hipMemcpyDtoHAsync",
        )
    }

    fn launch(&self, func: *mut c_void, grid: u32, block: u32, shared: u32, args: &mut [*mut c_void]) -> Result<()> {
        self.launch_on(func, grid, block, shared, args, null_mut())
    }

    fn launch_on(
        &self,
        func: *mut c_void,
        grid: u32,
        block: u32,
        shared: u32,
        args: &mut [*mut c_void],
        stream: *mut c_void,
    ) -> Result<()> {
        let code = unsafe {
            (self.launch_kernel)(func, grid, 1, 1, block, 1, 1, shared, stream, args.as_mut_ptr(), null_mut())
        };
        self.check(code, "hipModuleLaunchKernel")
    }
}

/// The first HIP runtime on this machine that sees a GPU.
fn load_hip() -> Option<(Arc<Hip>, c_int)> {
    LIBRARIES.iter().find_map(|name| {
        let hip = Hip::load(name).ok()?;
        let count = hip.device_count().ok().filter(|&count| count > 0)?;
        Some((Arc::new(hip), count))
    })
}

pub struct HipDevice {
    hip: Arc<Hip>,
    ordinal: c_int,
    name: String,
    arch: Option<&'static str>,
    module: *mut c_void,
    kernel: u64,
    cus: u32,
    memory: usize,
    pci: [i32; 3],
}

impl HipDevice {
    pub fn default_lanes(&self) -> u32 {
        default_lanes(self.cus)
    }

    pub fn bus(&self) -> String {
        bus(self.pci)
    }

    pub fn sensor(&self, sensors: &AmdSensors) -> Option<Sensor> {
        sensors.sensor(self.pci)
    }

    /// Cells for one row of the `numen-miner devices` table.
    pub fn row(&self, number: usize) -> [String; 7] {
        [
            number.to_string(),
            self.name.clone(),
            bus(self.pci),
            self.arch.unwrap_or("unsupported").to_string(),
            self.cus.to_string(),
            gib(self.memory),
            self.default_lanes().to_string(),
        ]
    }
}

impl fmt::Display for HipDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}, bus {}, {}, {} CUs, {}",
            self.name,
            bus(self.pci),
            self.arch.unwrap_or("unsupported"),
            self.cus,
            gib(self.memory)
        )
    }
}

/// Every AMD GPU the HIP runtime sees, in PCI order.
fn hip_devices() -> Vec<HipDevice> {
    let Some((hip, count)) = load_hip() else {
        log::info!("no HIP runtime sees a GPU, skipped");
        return Vec::new();
    };
    let mut devices: Vec<HipDevice> = (0..count)
        .filter_map(|ordinal| match probe_hip(&hip, ordinal) {
            Ok(device) => Some(device),
            Err(e) => {
                log::warn!("HIP device {ordinal}: {e:#}");
                None
            }
        })
        .collect();
    devices.sort_by_key(|device| device.pci);
    devices
}

/// Reads the device and loads the one embedded code object built for its
/// architecture. The runtime refuses every other one.
fn probe_hip(hip: &Arc<Hip>, ordinal: c_int) -> Result<HipDevice> {
    let mut name = [0 as c_char; 256];
    hip.check(
        unsafe { (hip.device_get_name)(name.as_mut_ptr(), name.len() as c_int, ordinal) },
        "hipDeviceGetName",
    )?;
    let mut memory = 0;
    hip.check(unsafe { (hip.device_total_mem)(&mut memory, ordinal) }, "hipDeviceTotalMem")?;
    hip.check(unsafe { (hip.set_device)(ordinal) }, "hipSetDevice")?;
    let loaded = CODE_OBJECTS.iter().find_map(|&(arch, image)| {
        let mut module = null_mut();
        let code = unsafe { (hip.module_load_data)(&mut module, image.as_ptr().cast()) };
        (code == 0).then_some((arch, module, autotune::fingerprint(&[image])))
    });
    Ok(HipDevice {
        hip: hip.clone(),
        ordinal,
        name: unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned(),
        arch: loaded.map(|(arch, _, _)| arch),
        module: loaded.map_or(null_mut(), |(_, module, _)| module),
        kernel: loaded.map_or(0, |(_, _, kernel)| kernel),
        cus: hip.attribute(MULTIPROCESSOR_COUNT, ordinal)? as u32,
        memory,
        pci: [
            hip.attribute(PCI_DOMAIN_ID, ordinal)?,
            hip.attribute(PCI_BUS_ID, ordinal)?,
            hip.attribute(PCI_DEVICE_ID, ordinal)?,
        ],
    })
}

#[repr(C)]
struct HipStaging {
    task: coop::Task,
    count: u32,
    nonces: [u64; 4 * MAX_HITS],
    works: [u8; 32 * MAX_HITS],
}

struct HipSlot {
    staging: *mut HipStaging,
    event: *mut c_void,
}

impl HipSlot {
    fn new(hip: &Hip) -> Result<Self> {
        Ok(HipSlot { staging: hip.host_alloc()?, event: hip.event()? })
    }
}

/// A block per nonce, the kernels `CudaBackend` runs built for AMD.
pub struct HipBackend {
    hip: Arc<Hip>,
    ordinal: c_int,
    lanes: u32,
    draw: *mut c_void,
    scan: *mut c_void,
    finish: *mut c_void,
    resident: u32,
    /// The launch's task. Rust hands a struct argument to an AMD kernel by
    /// pointer, so the task rides in device memory rather than in the launch.
    d_task: *mut c_void,
    d_dirs: *mut c_void,
    d_tris: *mut c_void,
    d_patches: *mut c_void,
    d_params: *mut c_void,
    d_sums: *mut c_void,
    d_stream: *mut c_void,
    d_out_nonce: *mut c_void,
    d_out_work: *mut c_void,
    d_out_count: *mut c_void,
    stream: *mut c_void,
    slots: [HipSlot; 2],
    queued: VecDeque<usize>,
}

// HIP handles and device pointers belong to the process, not to the thread
// that created them.
unsafe impl Send for HipBackend {}

impl HipBackend {
    pub fn new(number: usize, device: &HipDevice, lanes: Option<u32>, retune: bool) -> Result<Self> {
        if device.arch.is_none() {
            bail!("{} has no embedded kernel for its architecture", device.name);
        }
        let hip = device.hip.clone();
        hip.check(unsafe { (hip.set_device)(device.ordinal) }, "hipSetDevice")?;
        let requested_lanes = lanes;
        let scan = hip.function(device.module, c"poscan_scan")?;
        let mut per_unit = 0;
        hip.check(
            unsafe { (hip.occupancy)(&mut per_unit, scan, coop::THREADS as c_int, coop::SHARED as usize) },
            "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
        )?;
        // HIP counts an RDNA workgroup processor as one multiprocessor but
        // sizes the occupancy for one of its two compute units. Twice what it
        // reports is then the true count, or on a runtime that counts every
        // unit, a second full round that costs nothing.
        let rdna = device.arch.is_some_and(|arch| arch.starts_with("gfx1"));
        let resident = device.cus * per_unit as u32 * if rdna { 2 } else { 1 };
        if resident == 0 {
            bail!("{} cannot hold a scanning block", device.name);
        }
        let candidates = requested_lanes.map_or_else(
            || autotune_lanes(device.default_lanes(), device.memory, resident),
            |lanes| vec![lanes],
        );
        let bus = device.bus();
        let key = autotune::Key::new(autotune::Identity {
            backend: "hip",
            bus: &bus,
            name: &device.name,
            architecture: device.arch.unwrap_or("unsupported"),
            units: device.cus,
            memory: device.memory,
            runtime: hip.runtime_version(),
            kernel: device.kernel,
        });
        let cached = (requested_lanes.is_none() && !retune).then(|| autotune::cached(&key, &candidates)).flatten();
        let allocated_lanes = cached.as_ref().map(|tuning| tuning.lanes).unwrap_or(
            *candidates.iter().max().context("no HIP batch-size candidate")?,
        );

        let Packed { dirs, tris, patches } = blank();
        let alloc = |bytes: usize, what: &str| hip.alloc(bytes).with_context(|| format!("allocate {what}"));
        let mut backend = HipBackend {
            ordinal: device.ordinal,
            lanes: allocated_lanes,
            draw: hip.function(device.module, c"poscan_draw")?,
            finish: hip.function(device.module, c"poscan_finish")?,
            scan,
            resident,
            d_task: alloc(size_of::<coop::Task>(), "task")?,
            d_dirs: hip.clone_htod(&dirs).context("upload blank directions")?,
            d_tris: hip.clone_htod(&tris).context("upload blank faces")?,
            d_patches: hip.clone_htod(&patches).context("upload blank patches")?,
            d_params: alloc(allocated_lanes as usize * size_of::<Params>(), "parameters")?,
            d_sums: alloc(allocated_lanes as usize * size_of::<Sums>(), "sums")?,
            d_stream: alloc(resident as usize * coop::STREAM_BYTES, "sample stream")?,
            d_out_nonce: alloc(32 * MAX_HITS, "hit nonces")?,
            d_out_work: alloc(32 * MAX_HITS, "hit work")?,
            d_out_count: alloc(size_of::<u32>(), "hit count")?,
            stream: hip.stream()?,
            slots: [HipSlot::new(&hip)?, HipSlot::new(&hip)?],
            queued: VecDeque::with_capacity(2),
            hip,
        };
        if let Some(tuning) = cached {
            backend.lanes = tuning.lanes;
        } else if requested_lanes.is_none() {
            let choice = backend.autotune(number, device.default_lanes(), &candidates)?;
            if let Err(e) = autotune::save(key, &choice) {
                log::warn!("gpu{number} could not save autotune result: {e:#}");
            }
        }
        log::info!(
            "gpu{number} {device}, {} nonces per launch, {resident} resident blocks, {} registers, \
             {} B scratch, {} state allocation",
            backend.lanes,
            backend.hip.function_attribute(scan, FUNC_NUM_REGS)?,
            backend.hip.function_attribute(scan, FUNC_LOCAL_SIZE_BYTES)?,
            mib(state_bytes(allocated_lanes, resident))
        );
        Ok(backend)
    }

    pub fn lanes(&self) -> u32 {
        self.lanes
    }

    fn autotune(&mut self, number: usize, base: u32, candidates: &[u32]) -> Result<autotune::Choice> {
        let pre_hash = [0x5a; 32];
        let job = dump_job(&pre_hash, 0);
        let choice = autotune::run(number, base, candidates, |lanes| {
            self.lanes = lanes;
            self.launch(&job)?;
            Ok(())
        })?;
        self.lanes = choice.lanes;
        Ok(choice)
    }

    /// Queue a complete launch and pinned readback. Two host slots let the
    /// next launch sit behind the current one before the worker waits for and
    /// validates the current result, removing the host round trip between
    /// batches without allowing two scan kernels to overlap.
    fn enqueue(&mut self, job: &Job) -> Result<()> {
        let slot = self.queued.back().map_or(0, |slot| 1 - slot);
        let staging = unsafe { &mut *self.slots[slot].staging };
        staging.task = coop::Task { pre_hash: job.pre_hash, nonce: job.nonce_start, threshold: job.threshold };

        let hip = &self.hip;
        hip.check(unsafe { (hip.set_device)(self.ordinal) }, "hipSetDevice")?;
        hip.upload_async(self.d_task, &raw const staging.task, 1, self.stream)?;
        hip.check(
            unsafe { (hip.memset_async)(self.d_out_count, 0, size_of::<u32>(), self.stream) },
            "hipMemsetAsync",
        )?;

        let serial = self.lanes.div_ceil(SERIAL_BLOCK);
        hip.launch_on(
            self.draw,
            serial,
            SERIAL_BLOCK,
            coop::DRAW_SHARED * SERIAL_BLOCK,
            &mut [arg(&self.d_task), arg(&self.lanes), arg(&self.d_params)],
            self.stream,
        )?;
        hip.launch_on(
            self.scan,
            self.lanes.min(self.resident),
            coop::THREADS,
            coop::SHARED,
            &mut [
                arg(&self.d_dirs),
                arg(&self.d_tris),
                arg(&self.d_patches),
                arg(&self.d_params),
                arg(&self.d_sums),
                arg(&self.d_stream),
                arg(&self.lanes),
            ],
            self.stream,
        )?;
        hip.launch_on(
            self.finish,
            serial,
            SERIAL_BLOCK,
            0,
            &mut [
                arg(&self.d_sums),
                arg(&self.d_task),
                arg(&self.lanes),
                arg(&self.d_out_nonce),
                arg(&self.d_out_work),
                arg(&self.d_out_count),
            ],
            self.stream,
        )?;
        hip.download_async(&raw mut staging.count, self.d_out_count, 1, self.stream)?;
        hip.download_async(staging.nonces.as_mut_ptr(), self.d_out_nonce, staging.nonces.len(), self.stream)?;
        hip.download_async(staging.works.as_mut_ptr(), self.d_out_work, staging.works.len(), self.stream)?;
        hip.check(unsafe { (hip.event_record)(self.slots[slot].event, self.stream) }, "hipEventRecord")?;
        self.queued.push_back(slot);
        Ok(())
    }

    fn collect(&mut self) -> Result<Vec<Candidate>> {
        let slot = self.queued.pop_front().context("collect HIP launch with no queued work")?;
        let hip = &self.hip;
        hip.check(unsafe { (hip.event_synchronize)(self.slots[slot].event) }, "hipEventSynchronize")?;
        let staging = unsafe { &*self.slots[slot].staging };
        let found = staging.count as usize;
        if found > MAX_HITS {
            log::debug!("GPU hit buffer kept {MAX_HITS} of {found} hits; lower --batch or raise pool difficulty");
        }
        let kept = found.min(MAX_HITS);
        Ok((0..kept)
            .map(|hit| Candidate {
                nonce: staging.nonces[4 * hit..4 * hit + 4].try_into().expect("four nonce limbs"),
                work: staging.works[32 * hit..32 * hit + 32].try_into().expect("32 work bytes"),
            })
            .collect())
    }

    /// Run a launch over `job` to completion.
    fn launch(&self, job: &Job) -> Result<()> {
        let hip = &self.hip;
        hip.check(unsafe { (hip.set_device)(self.ordinal) }, "hipSetDevice")?;
        let task = coop::Task { pre_hash: job.pre_hash, nonce: job.nonce_start, threshold: job.threshold };
        hip.upload(self.d_task, &[task])?;
        hip.check(unsafe { (hip.memset)(self.d_out_count, 0, size_of::<u32>()) }, "hipMemset")?;

        let serial = self.lanes.div_ceil(SERIAL_BLOCK);
        hip.launch(
            self.draw,
            serial,
            SERIAL_BLOCK,
            coop::DRAW_SHARED * SERIAL_BLOCK,
            &mut [arg(&self.d_task), arg(&self.lanes), arg(&self.d_params)],
        )?;
        hip.launch(
            self.scan,
            self.lanes.min(self.resident),
            coop::THREADS,
            coop::SHARED,
            &mut [
                arg(&self.d_dirs),
                arg(&self.d_tris),
                arg(&self.d_patches),
                arg(&self.d_params),
                arg(&self.d_sums),
                arg(&self.d_stream),
                arg(&self.lanes),
            ],
        )?;
        hip.launch(
            self.finish,
            serial,
            SERIAL_BLOCK,
            0,
            &mut [
                arg(&self.d_sums),
                arg(&self.d_task),
                arg(&self.lanes),
                arg(&self.d_out_nonce),
                arg(&self.d_out_work),
                arg(&self.d_out_count),
            ],
        )?;
        hip.check(unsafe { (hip.synchronize)() }, "hipDeviceSynchronize")
    }

    /// Raw features of one launch's nonces from `start`, as
    /// `CudaBackend::features` takes them.
    pub fn features(&self, pre_hash: &[u8; 32], start: u64) -> Result<Vec<Option<[f64; N_FEATURES]>>> {
        self.launch(&dump_job(pre_hash, start))?;
        let mut sums = vec![0u8; self.lanes as usize * size_of::<Sums>()];
        self.hip.download(&mut sums, self.d_sums)?;
        Ok(features_from(&sums))
    }
}

/// One kernel argument as HIP takes it, a pointer to the value.
fn arg<T>(value: &T) -> *mut c_void {
    (value as *const T).cast_mut().cast()
}

pub fn hip_worker(shared: Arc<Shared>, meter: Arc<Meter>, mut backend: HipBackend) {
    let lanes = backend.lanes as u64;
    let mut pending: Option<Job> = None;
    while !shared.stopped() {
        let paused = meter.paused();
        let mut next = if paused { None } else { shared.next_job(lanes) };
        if let Some(job) = &next
            && let Err(e) = backend.enqueue(job)
        {
            log::error!("{}: {e:#}", meter.name);
            let _ = backend.hip.check(
                unsafe { (backend.hip.synchronize)() },
                "hipDeviceSynchronize(recover enqueue)",
            );
            backend.queued.clear();
            pending = None;
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        if let Some(job) = pending.take() {
            match backend.collect() {
                Ok(hits) => {
                    meter.add(lanes);
                    for hit in hits {
                        shared.found(&meter, &job, hit);
                    }
                }
                Err(e) => {
                    log::error!("{}: {e:#}", meter.name);
                    let _ = backend.hip.check(
                        unsafe { (backend.hip.synchronize)() },
                        "hipDeviceSynchronize(recover collect)",
                    );
                    backend.queued.clear();
                    next = None;
                }
            }
        }
        pending = next;
        if pending.is_none() {
            std::thread::sleep(Duration::from_millis(if paused { 1000 } else { 100 }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autotune_candidates_span_the_base() {
        assert_eq!(autotune_lanes(480, usize::MAX, 1), [240, 480, 960, 1920]);
    }

    #[test]
    fn autotune_candidates_respect_the_memory_cap() {
        let memory = 4 * state_bytes(960, 1);
        assert_eq!(autotune_lanes(480, memory, 1), [240, 480, 960]);
    }
}
