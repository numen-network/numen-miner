use anyhow::{Result, ensure};
use clap::builder::BoolishValueParser;
use clap::{ArgGroup, Args, Parser, Subcommand};
use std::num::ParseIntError;
use std::ops::RangeInclusive;
use std::path::PathBuf;

use crate::telemetry::TempLimit;
use crate::upstream::Endpoint;

pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("VERGEN_GIT_SHA"), ")");

#[derive(Parser)]
#[command(
    name = "numen-miner",
    about = "Numen PoScan miner",
    version = VERSION,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// URLs to mine through, tried in order for failover. `ws` and `wss` reach
    /// a node, `http` a pool serving the node's RPC such as DiffPool, and
    /// `stratum+tcp` and `stratum+ssl` a stratum pool. Repeat the flag or
    /// separate them with commas.
    #[arg(long, short = 'o', env = "NUMEN_MINER_URL", value_delimiter = ',', required = true)]
    pub url: Vec<Endpoint>,

    /// Stratum login, your wallet address and a worker name as
    /// `WALLET.WORKER`.
    #[arg(long, short = 'u', env = "NUMEN_MINER_USER", value_name = "WALLET.WORKER")]
    pub user: Option<String>,

    /// Stratum password. rplant reads `start=N` here as the starting
    /// difficulty.
    #[arg(long, short = 'p', env = "NUMEN_MINER_PASS", default_value = "x")]
    pub pass: String,

    #[command(flatten)]
    pub rig: Rig,

    /// Seconds between speed reports.
    #[arg(long, short = 'i', env = "NUMEN_MINER_REPORT_INTERVAL", default_value_t = 60)]
    pub report_interval: u64,

    /// Port for the read-only `GET /summary` API on 127.0.0.1.
    #[arg(long, env = "NUMEN_MINER_API_PORT", default_value_t = 4068)]
    pub api_port: u16,

    /// Pause a GPU that stays at or above this many degrees Celsius for ten
    /// seconds. Off unless set.
    #[arg(long, env = "NUMEN_MINER_TEMP_STOP")]
    pub temp_stop: Option<u32>,

    /// Resume a paused GPU at or below this many degrees Celsius. Defaults to
    /// ten under `--temp-stop`.
    #[arg(long, env = "NUMEN_MINER_TEMP_START", requires = "temp_stop")]
    pub temp_start: Option<u32>,
}

impl Cli {
    pub fn temp_limit(&self) -> Result<Option<TempLimit>> {
        let Some(stop) = self.temp_stop else {
            return Ok(None);
        };
        let start = self.temp_start.unwrap_or(stop.saturating_sub(10));
        ensure!(start < stop, "--temp-start {start} must be below --temp-stop {stop}");
        Ok(Some(TempLimit { stop: f64::from(stop), start: f64::from(start) }))
    }
}

/// Device flags for mining and `bench`. They aren't global, since `devices`
/// takes none.
#[derive(Args)]
pub struct Rig {
    /// Mine on the CPU as well.
    #[arg(long, env = "NUMEN_MINER_CPU", value_parser = BoolishValueParser::new())]
    pub cpu: bool,

    #[command(flatten)]
    pub cpus: CpuChoice,

    /// Skip the GPUs entirely, without even opening one.
    #[arg(long, env = "NUMEN_MINER_NO_GPU", value_parser = BoolishValueParser::new())]
    pub no_gpu: bool,

    #[command(flatten)]
    pub gpus: GpuChoice,
}

#[derive(Args)]
pub struct CpuChoice {
    /// CPU worker threads when `--cpu` is on. Defaults to every core.
    #[arg(long, short = 't', env = "NUMEN_MINER_THREADS")]
    pub threads: Option<usize>,

    /// Logical CPUs to pin CPU worker threads to, one each, as in `0-7,16`.
    /// The list sets the thread count, so it can't go with `--threads`.
    #[arg(
        long,
        env = "NUMEN_MINER_CPU_AFFINITY",
        value_delimiter = ',',
        value_parser = cpu_range,
        conflicts_with = "threads"
    )]
    pub cpu_affinity: Vec<RangeInclusive<usize>>,
}

impl CpuChoice {
    /// One entry per CPU worker thread, the logical CPU it's pinned to or
    /// `None` to let the OS place it.
    pub fn cpu_threads(&self) -> Vec<Option<usize>> {
        if self.cpu_affinity.is_empty() {
            let threads = self.threads.unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
            });
            return vec![None; threads];
        }
        self.cpu_affinity.iter().cloned().flatten().map(Some).collect()
    }
}

#[derive(Args)]
pub struct GpuChoice {
    /// GPUs to mine on, comma separated, by the numbers `numen-miner devices`
    /// prints. Leave it out to mine on every GPU.
    #[arg(long, short = 'd', env = "NUMEN_MINER_DEVICES", value_delimiter = ',')]
    pub devices: Option<Vec<usize>>,

    /// Nonces per launch for each mining GPU in order, comma separated. GPUs
    /// past the end of the list take the last value. When omitted, load the
    /// saved per-GPU result or benchmark several sizes around 480 per SM or CU
    /// and save the stable winner.
    #[arg(
        long,
        short = 'b',
        env = "NUMEN_MINER_BATCH",
        value_delimiter = ',',
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub batch: Vec<u32>,

    /// Ignore a saved automatic batch and measure this machine again. Has no
    /// meaning with an explicit `--batch`.
    #[arg(
        long,
        env = "NUMEN_MINER_RETUNE",
        value_parser = BoolishValueParser::new(),
        conflicts_with = "batch"
    )]
    pub retune: bool,
}

impl GpuChoice {
    pub fn mines_on_gpu(&self, number: usize) -> bool {
        self.devices.as_ref().is_none_or(|devices| devices.contains(&number))
    }

    /// Nonces per launch for the mining GPU at `position`, or `None` to
    /// autotune it.
    pub fn lanes_for(&self, position: usize) -> Option<u32> {
        self.batch.get(position).or(self.batch.last()).copied()
    }
}

/// One logical CPU, or a range of them as in `0-7`.
fn cpu_range(s: &str) -> Result<RangeInclusive<usize>, ParseIntError> {
    let (first, last) = s.split_once('-').unwrap_or((s, s));
    Ok(first.parse()?..=last.parse()?)
}

#[derive(Subcommand)]
pub enum Command {
    /// Run a reproducible fixed-task benchmark. No node needed.
    Bench(Bench),
    /// List every GPU with the number `--devices` takes, then exit.
    Devices,
    /// Write one device's raw features for a run of nonces, for
    /// `numen-harness retention --from` to grade against the chain.
    Dump(Dump),
}

#[derive(Args)]
pub struct Bench {
    #[command(flatten)]
    pub rig: Rig,

    /// Seconds to measure after device setup and autotuning. Zero runs until
    /// Ctrl+C.
    #[arg(long, default_value_t = 60)]
    pub duration: u64,

    /// Seconds between progress reports.
    #[arg(long, short = 'i', default_value_t = 10)]
    pub report_interval: u64,

    /// Also write the final machine-readable report to this JSON file.
    #[arg(long)]
    pub json: Option<PathBuf>,
}

#[derive(Args)]
#[command(group(ArgGroup::new("target").required(true).args(["cpu", "device"])))]
pub struct Dump {
    /// Dump the CPU, on the widest path it runs.
    #[arg(long)]
    pub cpu: bool,

    /// GPU to dump, by the number `numen-miner devices` prints.
    #[arg(long, short = 'd')]
    pub device: Option<usize>,

    /// Nonces to dump, counting up from zero.
    #[arg(long, default_value_t = 512)]
    pub count: u64,

    /// Pre hash as a big endian number in hex, so `0x2a` fills only the last
    /// byte.
    #[arg(long, default_value = "0x2a", value_parser = pre_hash)]
    pub pre_hash: [u8; 32],

    /// File to write.
    #[arg(long)]
    pub out: PathBuf,
}

/// The chain builds a pre hash from a big endian 256 bit number, so a short
/// one lands in the trailing bytes.
fn pre_hash(s: &str) -> Result<[u8; 32], hex::FromHexError> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(format!("{:0>64}", s.strip_prefix("0x").unwrap_or(s)), &mut out)?;
    Ok(out)
}
