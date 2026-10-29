mod api;
mod cli;
mod cpu;
mod dump;
mod engine;
mod gpu;
mod rpc;
mod stratum;
mod telemetry;
mod upstream;

use clap::Parser;
use cli::{Command, Cli, GpuChoice, Rig};
use comfy_table::Table;
use engine::Shared;
use nvml_wrapper::Nvml;
use primitive_types::U256;
use std::sync::Arc;
use std::thread::JoinHandle;
use telemetry::{AmdSensors, TempLimit};
use std::time::{Duration, Instant};
use upstream::Upstream;

use crate::cpu::spawn_cpu_workers;
use crate::gpu::{Gpu, cuda_worker, hip_worker};

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    match &cli.command {
        Some(Command::Devices) => list_devices(),
        Some(Command::Bench(rig)) => bench(rig, cli.report_interval).await,
        Some(Command::Dump(args)) => {
            if let Err(e) = dump::dump(args) {
                log::error!("{e:#}");
                std::process::exit(1);
            }
        }
        None => work(&cli).await,
    }
}

fn list_devices() {
    let gpus = gpu::gpus();
    if gpus.is_empty() {
        println!("no GPUs found");
        return;
    }
    let mut table = Table::new();
    table.set_header(["Device", "Name", "Bus", "Compute", "SMs/CUs", "Memory", "Autotune base"]);
    for (number, gpu) in gpus.iter().enumerate() {
        table.add_row(match gpu {
            Gpu::Cuda(device) => device.row(number),
            Gpu::Hip(device) => device.row(number),
        });
    }
    println!("{table}");
}

async fn bench(rig: &Rig, report_interval: u64) {
    let shared = Arc::new(Shared::new());
    shared.set_task([0; 32], U256::MAX, 0);
    let workers = spawn_workers(rig, &shared);

    tokio::select! {
        _ = report_loop(shared.clone(), report_interval, None) => {}
        _ = tokio::signal::ctrl_c() => log::info!("shutting down"),
    }

    shared.stop();
    for w in workers {
        let _ = w.join();
    }
}

async fn work(cli: &Cli) {
    let setup = Upstream::new(cli.url.clone(), cli.user.clone(), cli.pass.clone())
        .and_then(|upstream| Ok((upstream, cli.temp_limit()?)));
    let (upstream, limit) = match setup {
        Ok(setup) => setup,
        Err(e) => {
            log::error!("{e:#}");
            std::process::exit(1);
        }
    };
    let shared = Arc::new(Shared::new());
    let workers = spawn_workers(&cli.rig, &shared);

    tokio::spawn(api::serve(shared.clone(), cli.api_port));

    tokio::select! {
        _ = upstream.run(&shared) => {}
        _ = report_loop(shared.clone(), cli.report_interval, limit) => {}
        _ = tokio::signal::ctrl_c() => log::info!("shutting down"),
    }

    shared.stop();
    for w in workers {
        let _ = w.join();
    }
}

fn spawn_workers(rig: &Rig, shared: &Arc<Shared>) -> Vec<JoinHandle<()>> {
    let mut workers = Vec::new();
    if rig.cpu {
        workers.extend(spawn_cpu_workers(rig.cpus.cpu_threads(), shared));
    }
    if !rig.no_gpu {
        workers.extend(spawn_gpu_workers(&rig.gpus, shared));
    }
    if workers.is_empty() {
        log::warn!("no device to mine on, add --cpu to mine on the CPU");
    }
    workers
}

fn spawn_gpu_workers(choice: &GpuChoice, shared: &Arc<Shared>) -> Vec<JoinHandle<()>> {
    let nvml = if std::env::var_os("NUMEN_NO_NVML").is_some() { None } else { Nvml::init().ok().map(Arc::new) };
    let amd_sensors = AmdSensors::new();
    let gpus = gpu::gpus()
        .into_iter()
        .enumerate()
        .filter(|(number, _)| choice.mines_on_gpu(*number));
    let mut workers = Vec::new();
    for (position, (number, gpu)) in gpus.enumerate() {
        let lanes = choice.lanes_for(position);
        let shared = shared.clone();
        let worker = match gpu {
            Gpu::Cuda(device) => gpu::CudaBackend::new(number, &device, lanes).map(|backend| {
                let sensor = nvml.as_ref().map(|nvml| device.sensor(nvml));
                let meter = shared.meter(format!("gpu{number}"), device.bus(), sensor);
                std::thread::spawn(move || cuda_worker(shared, meter, backend))
            }),
            Gpu::Hip(device) => gpu::HipBackend::new(number, &device, lanes).map(|backend| {
                let meter = shared.meter(format!("gpu{number}"), device.bus(), device.sensor(&amd_sensors));
                std::thread::spawn(move || hip_worker(shared, meter, backend))
            }),
        };
        match worker {
            Ok(worker) => workers.push(worker),
            Err(e) => log::error!("gpu{number}: {e:#}"),
        }
    }
    workers
}

async fn report_loop(shared: Arc<Shared>, interval: u64, limit: Option<TempLimit>) {
    let every = Duration::from_secs(interval.max(1));
    let mut sample = tokio::time::interval(Duration::from_secs(1));
    let mut report = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
    loop {
        tokio::select! {
            _ = sample.tick() => {
                let now = Instant::now();
                for meter in shared.meters() {
                    meter.sample(now);
                    meter.sense(now, limit);
                }
            }
            _ = report.tick() => log::info!("{}", speed_report(&shared)),
        }
    }
}

fn speed_report(shared: &Shared) -> String {
    let now = Instant::now();
    let mut table = Table::new();
    table.set_header(["Device", "Bus", "Speed", "Accepted", "Rejected", "Temp", "Fan", "Power"]);
    for meter in shared.meters() {
        let reading = meter.reading();
        table.add_row([
            meter.name.clone(),
            meter.bus.clone(),
            hashrate(meter.rate(now)),
            meter.accepted().to_string(),
            meter.rejected().to_string(),
            reading.temp.map_or("n/a".into(), |temp| format!("{temp:.0} C")),
            reading.fan.map_or("n/a".into(), |fan| fan.to_string()),
            reading.power.map_or("n/a".into(), |power| format!("{power:.0} W")),
        ]);
    }
    let (accepted, rejected) = shared.verdicts();
    format!("speed {}, accepted {accepted}, rejected {rejected}\n{table}", hashrate(shared.rate(now)))
}

fn hashrate(rate: f64) -> String {
    let (rate, unit) = match rate {
        r if r < 1e4 => (r, "H/s"),
        r if r < 1e7 => (r / 1e3, "KH/s"),
        r if r < 1e10 => (r / 1e6, "MH/s"),
        r => (r / 1e9, "GH/s"),
    };
    format!("{rate:.2} {unit}")
}
