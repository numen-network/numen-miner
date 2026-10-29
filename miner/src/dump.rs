use anyhow::{Context, Result};
use numen_kernel::host::Scanner;
use numen_kernel::pipeline::N_FEATURES;
use numen_kernel::work::derive_seed;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::cli::Dump;
use crate::gpu::{self, CudaBackend, Gpu, HipBackend};

/// Raw features of the nonces from zero up, `None` where the body came out
/// unmeasurable.
type Features = Vec<Option<[f64; N_FEATURES]>>;

pub fn dump(args: &Dump) -> Result<()> {
    let features = match args.device {
        Some(number) => gpu_features(number, &args.pre_hash, args.count)?,
        None => cpu_features(&args.pre_hash, args.count),
    };
    write(&args.out, &args.pre_hash, &features).with_context(|| format!("write {}", args.out.display()))?;
    log::info!("wrote {} nonces to {}", features.len(), args.out.display());
    Ok(())
}

/// One line per nonce under a pre hash header, each feature as the shortest
/// decimal that parses back to the same bits.
fn write(path: &Path, pre_hash: &[u8; 32], features: &Features) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "pre_hash {}", hex::encode(pre_hash))?;
    for (nonce, row) in features.iter().enumerate() {
        write!(out, "{nonce}")?;
        match row {
            Some(row) => row.iter().try_for_each(|x| write!(out, " {x}"))?,
            None => write!(out, " -")?,
        }
        writeln!(out)?;
    }
    out.flush()
}

fn cpu_features(pre_hash: &[u8; 32], count: u64) -> Features {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let chunk = count.div_ceil(threads);
    std::thread::scope(|scope| {
        let parts: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut scanner = Scanner::new();
                    (t * chunk..count.min((t + 1) * chunk))
                        .map(|nonce| scanner.features(derive_seed(pre_hash, &[nonce, 0, 0, 0])))
                        .collect::<Features>()
                })
            })
            .collect();
        parts.into_iter().flat_map(|part| part.join().expect("dump thread")).collect()
    })
}

fn gpu_features(number: usize, pre_hash: &[u8; 32], count: u64) -> Result<Features> {
    let gpu = gpu::gpus().into_iter().nth(number).with_context(|| format!("no gpu{number}"))?;
    match gpu {
        Gpu::Cuda(device) => {
            let mut backend = CudaBackend::new(number, &device, None, false)?;
            launches(count, |start| backend.features(pre_hash, start))
        }
        Gpu::Hip(device) => {
            let backend = HipBackend::new(number, &device, None, false)?;
            launches(count, |start| backend.features(pre_hash, start))
        }
    }
}

/// Launch after launch until `count` nonces are in, the last one cut short.
fn launches(count: u64, mut launch: impl FnMut(u64) -> Result<Features>) -> Result<Features> {
    let mut features = Features::new();
    while (features.len() as u64) < count {
        features.extend(launch(features.len() as u64)?);
    }
    features.truncate(count as usize);
    Ok(features)
}
