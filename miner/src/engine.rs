use numen_kernel::Nonce;
use primitive_types::U256;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use crate::telemetry::{self, Reading, Sensor, TempLimit};

pub struct Candidate {
    pub nonce: Nonce,
    pub work: [u8; 32],
}

/// One slice of work handed to a backend.
pub struct Job {
    pub pre_hash: [u8; 32],
    pub threshold: [u8; 32],
    pub nonce_start: Nonce,
    pub generation: u64,
}

pub struct ActiveTask {
    pub pre_hash: [u8; 32],
    pub threshold: [u8; 32],
    pub extranonce: u64,
}

/// How the live link sends a hit upstream.
pub type Submit = Box<dyn Fn(&Arc<Shared>, Share) + Send + Sync>;

/// A hit on its way upstream.
pub struct Share {
    pub meter: Arc<Meter>,
    pub pre_hash: [u8; 32],
    pub nonce: Nonce,
    pub work: [u8; 32],
}

/// One sample a second for the last minute.
const SAMPLES: usize = 60;

/// Nonces one device has scanned, what the upstream made of its shares, and
/// how hot it runs.
pub struct Meter {
    pub name: String,
    pub bus: String,
    pub batch: Option<u32>,
    scanned: AtomicU64,
    accepted: AtomicU64,
    rejected: AtomicU64,
    samples: Mutex<Samples>,
    sensor: Option<Sensor>,
    reading: Mutex<Reading>,
    hot_since: Mutex<Option<Instant>>,
    paused: AtomicBool,
}

/// A ring of running counts, oldest at `next`.
struct Samples {
    ring: [(Instant, u64); SAMPLES],
    next: usize,
}

impl Meter {
    fn new(name: String, bus: String, batch: Option<u32>, sensor: Option<Sensor>, now: Instant) -> Self {
        Meter {
            name,
            bus,
            batch,
            scanned: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            samples: Mutex::new(Samples { ring: [(now, 0); SAMPLES], next: 0 }),
            sensor,
            reading: Mutex::new(Reading::default()),
            hot_since: Mutex::new(None),
            paused: AtomicBool::new(false),
        }
    }

    pub fn add(&self, n: u64) {
        self.scanned.fetch_add(n, Ordering::Relaxed);
    }

    pub fn scanned(&self) -> u64 {
        self.scanned.load(Ordering::Relaxed)
    }

    pub fn accepted(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }

    pub fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    pub fn sample(&self, now: Instant) {
        let mut samples = self.samples.lock().expect("sample lock");
        let next = samples.next;
        samples.ring[next] = (now, self.scanned());
        samples.next = (next + 1) % SAMPLES;
    }

    /// Read the sensor and pause or resume the device against `limit`.
    pub fn sense(&self, now: Instant, limit: Option<TempLimit>) {
        let Some(sensor) = &self.sensor else {
            return;
        };
        let reading = sensor.read();
        *self.reading.lock().expect("reading lock") = reading;
        let (Some(limit), Some(temp)) = (limit, reading.temp) else {
            return;
        };
        let mut hot_since = self.hot_since.lock().expect("heat lock");
        let was_paused = self.paused();
        let (paused, since) = telemetry::throttle(was_paused, *hot_since, temp, limit, now);
        *hot_since = since;
        self.paused.store(paused, Ordering::Relaxed);
        if paused && !was_paused {
            log::warn!("{} paused at {temp:.0} C", self.name);
        }
        if was_paused && !paused {
            log::info!("{} resumed at {temp:.0} C", self.name);
        }
    }

    pub fn reading(&self) -> Reading {
        *self.reading.lock().expect("reading lock")
    }

    pub fn paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Hashes per second over the last minute, or since startup in the first
    /// minute.
    pub fn rate(&self, now: Instant) -> f64 {
        let samples = self.samples.lock().expect("sample lock");
        let (then, count) = samples.ring[samples.next];
        (self.scanned() - count) as f64 / now.duration_since(then).as_secs_f64()
    }
}

pub struct Shared {
    task: RwLock<Option<ActiveTask>>,
    /// The low nonce limb handed out next. Claimed under the task's read lock
    /// and reset under its write lock, so no job pairs a task with nonces
    /// counted for another.
    cursor: AtomicU64,
    /// Changes whenever the active task or link changes. Results from an
    /// already queued GPU launch can then be discarded locally instead of
    /// being submitted as guaranteed-stale work.
    generation: AtomicU64,
    /// Nonce limbs 1 and 2, drawn once per process.
    limbs: [u64; 2],
    submit: RwLock<Option<Submit>>,
    meters: Mutex<Vec<Arc<Meter>>>,
    shutdown: AtomicBool,
}

impl Shared {
    pub fn new() -> Self {
        Shared {
            task: RwLock::new(None),
            cursor: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            limbs: [random_limb(), random_limb()],
            submit: RwLock::new(None),
            meters: Mutex::new(Vec::new()),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Mine `pre_hash` at `difficulty` with `extranonce` as the nonce's top
    /// limb.
    pub fn set_task(&self, pre_hash: [u8; 32], difficulty: U256, extranonce: u64) {
        let threshold = threshold(difficulty);
        let mut guard = self.task.write().expect("task lock");
        if guard.as_ref().is_none_or(|t| t.pre_hash != pre_hash) {
            self.cursor.store(0, Ordering::Relaxed);
        }
        *guard = Some(ActiveTask { pre_hash, threshold, extranonce });
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn next_job(&self, count: u64) -> Option<Job> {
        let guard = self.task.read().expect("task lock");
        let task = guard.as_ref()?;
        let start = self.cursor.fetch_add(count, Ordering::Relaxed);
        let [a, b] = self.limbs;
        Some(Job {
            pre_hash: task.pre_hash,
            threshold: task.threshold,
            nonce_start: [start, a, b, task.extranonce],
            generation: self.generation.load(Ordering::Relaxed),
        })
    }

    pub fn meter(&self, name: String, bus: String, batch: Option<u32>, sensor: Option<Sensor>) -> Arc<Meter> {
        let meter = Arc::new(Meter::new(name, bus, batch, sensor, Instant::now()));
        self.meters.lock().expect("meter lock").push(meter.clone());
        meter
    }

    pub fn meters(&self) -> Vec<Arc<Meter>> {
        self.meters.lock().expect("meter lock").clone()
    }

    /// Hashes per second across every device.
    pub fn rate(&self, now: Instant) -> f64 {
        self.meters().iter().map(|meter| meter.rate(now)).sum()
    }

    /// Shares the upstream accepted and rejected, over every device.
    pub fn verdicts(&self) -> (u64, u64) {
        self.meters().iter().fold((0, 0), |(a, r), m| (a + m.accepted(), r + m.rejected()))
    }

    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    pub fn stopped(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Route hits to a link until it drops.
    pub fn attach(&self, submit: Submit) {
        *self.submit.write().expect("submit lock") = Some(submit);
    }

    /// Drop the link and its task, since no hit on that task can reach the
    /// upstream any more. The devices idle until the next link sets a task.
    pub fn detach(&self) {
        *self.submit.write().expect("submit lock") = None;
        *self.task.write().expect("task lock") = None;
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Send a hit upstream from the worker that found it. With no link up the
    /// hit is lost, as it would be stale by the time one came back.
    pub fn found(self: &Arc<Self>, meter: &Arc<Meter>, job: &Job, hit: Candidate) {
        if self.generation.load(Ordering::Relaxed) != job.generation {
            log::debug!("{} stale hit dropped after a task change", meter.name);
            return;
        }
        let share = Share { meter: meter.clone(), pre_hash: job.pre_hash, nonce: hit.nonce, work: hit.work };
        match &*self.submit.read().expect("submit lock") {
            Some(submit) => submit(self, share),
            None => log::debug!("{} hit dropped with no upstream", meter.name),
        }
    }

    /// Count what the upstream made of a share and log it with the running
    /// totals, plus the pool's reason when it gives one.
    pub fn verdict(&self, meter: &Meter, sent: Instant, accepted: bool, reason: Option<&str>) {
        let (counter, verdict) = if accepted {
            (&meter.accepted, "accepted")
        } else {
            (&meter.rejected, "rejected")
        };
        counter.fetch_add(1, Ordering::Relaxed);
        let (a, r) = self.verdicts();
        let reason = reason.map(|reason| format!(", {reason}")).unwrap_or_default();
        log::info!("{} {verdict} ({a}/{r}) {} ms{reason}", meter.name, sent.elapsed().as_millis());
    }
}

/// Largest work value that still clears `difficulty`.
///
/// The chain asks whether `work * difficulty` overflows 256 bits, which is
/// the same question as `work <= U256::MAX / difficulty`. Dividing once per
/// task turns the inner loop test into a byte comparison.
fn threshold(difficulty: U256) -> [u8; 32] {
    U256::MAX.checked_div(difficulty).unwrap_or(U256::MAX).to_big_endian()
}

/// Rigs mining through one node receive the same pre hash, so each scans
/// under random limbs of its own.
pub fn random_limb() -> u64 {
    let mut bytes = [0u8; 8];
    getrandom::getrandom(&mut bytes).expect("OS random source");
    u64::from_le_bytes(bytes)
}
