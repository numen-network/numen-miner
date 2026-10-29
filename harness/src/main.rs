mod attribute;
mod dump;
mod reference;
mod report;

use clap::{Parser, Subcommand};
use numen_kernel::host::Scanner;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "numen-harness", about = "Price a PoScan kernel variant")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compare the kernel against the canonical pipeline nonce by nonce.
    Retention {
        #[arg(long, default_value_t = 512)]
        count: u64,
        #[arg(long, default_value = "0x2a", value_parser = pre_hash)]
        pre_hash: [u8; 32],
        /// Report which feature dimensions moved, not just how many nonces did.
        #[arg(long)]
        attribute: bool,
        /// Grade a `numen-miner dump` file instead of the CPU kernel here. The
        /// file brings its own pre hash and nonces.
        #[arg(long, conflicts_with_all = ["count", "pre_hash"])]
        from: Option<PathBuf>,
    },
    /// Measure throughput on both sides.
    Throughput {
        #[arg(long, default_value_t = 256)]
        count: u64,
        #[arg(long, default_value = "0x2a", value_parser = pre_hash)]
        pre_hash: [u8; 32],
        #[arg(long, default_value_t = 1)]
        threads: usize,
    },
    /// Retention and throughput together, reported as effective hash power.
    Power {
        #[arg(long, default_value_t = 512)]
        count: u64,
        #[arg(long, default_value = "0x2a", value_parser = pre_hash)]
        pre_hash: [u8; 32],
        #[arg(long, default_value_t = 1)]
        threads: usize,
    },
    /// Combine a GPU dump's retention with a finite miner benchmark.
    GpuPower {
        /// Raw feature dump produced by `numen-miner dump`.
        #[arg(long)]
        from: PathBuf,
        /// JSON report produced by `numen-miner bench --json`.
        #[arg(long)]
        benchmark: PathBuf,
        /// Device name in the benchmark report, such as gpu0.
        #[arg(long, default_value = "gpu0")]
        device: String,
        /// Report which feature dimensions moved.
        #[arg(long)]
        attribute: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Retention { count, pre_hash, attribute, from } => {
            let retention = match from {
                Some(path) => attribute::measure_dump(&dump::read(&path), attribute),
                None => attribute::measure(&pre_hash, count, attribute),
            };
            report::retention(&retention);
        }
        Command::Throughput { count, pre_hash, threads } => {
            report::throughput(&throughput(&pre_hash, count, threads));
        }
        Command::Power { count, pre_hash, threads } => {
            let ret = attribute::measure(&pre_hash, count, false);
            let thr = throughput(&pre_hash, count, threads);
            report::retention(&ret);
            report::throughput(&thr);
            report::power(&ret, &thr);
        }
        Command::GpuPower { from, benchmark, device, attribute } => {
            let retention = attribute::measure_dump(&dump::read(&from), attribute);
            let text = std::fs::read_to_string(&benchmark).expect("read benchmark report");
            let benchmark: Benchmark = serde_json::from_str(&text).expect("parse benchmark report");
            let measured = benchmark
                .devices
                .iter()
                .find(|measured| measured.name == device)
                .unwrap_or_else(|| panic!("benchmark has no device {device}"));
            report::retention(&retention);
            report::gpu_power(&retention, &device, measured.hashrate);
        }
    }
}

#[derive(Deserialize)]
struct Benchmark {
    devices: Vec<BenchmarkDevice>,
}

#[derive(Deserialize)]
struct BenchmarkDevice {
    name: String,
    hashrate: f64,
}

pub struct Throughput {
    pub kernel_per_sec: f64,
    pub canonical_per_sec: f64,
    pub threads: usize,
    pub count: u64,
}

fn throughput(pre_hash: &[u8; 32], count: u64, threads: usize) -> Throughput {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");

    let kernel_per_sec = pool.install(|| {
        let start = Instant::now();
        run_parallel(count, threads, |lo, hi| {
            let mut scanner = Scanner::new();
            for n in lo..hi {
                std::hint::black_box(scanner.work(pre_hash, &[n, 0, 0, 0]));
            }
        });
        count as f64 / start.elapsed().as_secs_f64()
    });

    let canonical_per_sec = pool.install(|| {
        let start = Instant::now();
        run_parallel(count, threads, |lo, hi| {
            for n in lo..hi {
                std::hint::black_box(reference::work(pre_hash, n));
            }
        });
        count as f64 / start.elapsed().as_secs_f64()
    });

    Throughput { kernel_per_sec, canonical_per_sec, threads, count }
}

/// Split `count` nonces into one contiguous range per thread.
fn run_parallel<F>(count: u64, threads: usize, body: F)
where
    F: Fn(u64, u64) + Sync + Send,
{
    use rayon::prelude::*;
    let threads = threads.max(1) as u64;
    let chunk = count.div_ceil(threads);
    (0..threads).into_par_iter().for_each(|t| {
        let lo = t * chunk;
        let hi = (lo + chunk).min(count);
        if lo < hi {
            body(lo, hi);
        }
    });
}

/// The chain builds a pre hash from a big endian 256 bit number, so a short
/// one lands in the trailing bytes.
fn pre_hash(s: &str) -> Result<[u8; 32], hex::FromHexError> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(format!("{:0>64}", s.strip_prefix("0x").unwrap_or(s)), &mut out)?;
    Ok(out)
}
