mod api;
mod autotune;
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
use cli::{Bench, Command, Cli, GpuChoice, Rig, VERSION};
use comfy_table::Table;
use engine::{Farm, Shared};
use nvml_wrapper::Nvml;
use primitive_types::U256;
use serde::Serialize;
use std::path::Path;
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
        Some(Command::Bench(args)) => bench(args).await,
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

async fn bench(args: &Bench) {
    let mut running = spawn_workers(&args.rig, true);
    running.farm.set_task([0; 32], U256::MAX, 0);
    let started = Instant::now();
    running.farm.reset_scanned(started);

    tokio::select! {
        _ = report_loop(running.farm.clone(), args.report_interval, None) => {}
        _ = tokio::signal::ctrl_c() => log::info!("shutting down"),
        _ = benchmark_deadline(args.duration) => log::info!("benchmark duration complete"),
    }

    running.stop();
    let report = benchmark_report(&running.farm, started.elapsed());
    log::info!("final benchmark\n{}", speed_report(&running.farm));
    if let Some(path) = &args.json
        && let Err(e) = write_benchmark(path, &report)
    {
        log::error!("write benchmark {}: {e:#}", path.display());
    }
}

async fn benchmark_deadline(seconds: u64) {
    if seconds == 0 {
        std::future::pending::<()>().await;
    } else {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkReport {
    version: &'static str,
    duration_seconds: f64,
    hashrate: f64,
    scanned: u64,
    devices: Vec<BenchmarkDevice>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkDevice {
    name: String,
    bus: String,
    batch: Option<u32>,
    hashrate: f64,
    scanned: u64,
    temp: Option<f64>,
    fan: Option<String>,
    power: Option<f64>,
}

fn benchmark_report(farm: &Farm, duration: Duration) -> BenchmarkReport {
    let seconds = duration.as_secs_f64();
    let devices = farm
        .meters()
        .iter()
        .map(|meter| {
            let reading = meter.reading();
            let scanned = meter.scanned();
            BenchmarkDevice {
                name: meter.name.clone(),
                bus: meter.bus.clone(),
                batch: meter.batch,
                hashrate: if seconds > 0.0 { scanned as f64 / seconds } else { 0.0 },
                scanned,
                temp: reading.temp,
                fan: reading.fan.map(|fan| fan.to_string()),
                power: reading.power,
            }
        })
        .collect::<Vec<_>>();
    BenchmarkReport {
        version: VERSION,
        duration_seconds: duration.as_secs_f64(),
        hashrate: devices.iter().map(|device| device.hashrate).sum(),
        scanned: devices.iter().map(|device| device.scanned).sum(),
        devices,
    }
}

fn write_benchmark(path: &Path, report: &BenchmarkReport) -> anyhow::Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)?;
    log::info!("wrote benchmark to {}", path.display());
    Ok(())
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
    let per_device = cli.worker_mode.per_device(&cli.url);
    log::info!(
        "worker connections: {}",
        if per_device { "one per device" } else { "one for the rig" }
    );
    let mut running = spawn_workers(&cli.rig, per_device);
    let connections = connections(&upstream, &running.farm, per_device);

    tokio::spawn(api::serve(running.farm.clone(), cli.api_port));

    tokio::select! {
        _ = run_upstreams(connections) => {}
        _ = report_loop(running.farm.clone(), cli.report_interval, limit) => {}
        _ = tokio::signal::ctrl_c() => log::info!("shutting down"),
    }

    running.stop();
}

struct RunningRig {
    farm: Farm,
    workers: Vec<JoinHandle<()>>,
}

impl RunningRig {
    fn stop(&mut self) {
        self.farm.stop();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn spawn_workers(rig: &Rig, per_device: bool) -> RunningRig {
    if !per_device {
        let shared = Arc::new(Shared::new());
        let workers = spawn_grouped_workers(rig, &shared);
        if workers.is_empty() {
            log::warn!("no device to mine on, add --cpu to mine on the CPU");
        }
        return RunningRig { farm: Farm::new(vec![shared]), workers };
    }

    let mut groups = Vec::new();
    let mut workers = Vec::new();
    if rig.cpu {
        let shared = Arc::new(Shared::new());
        let cpu = spawn_cpu_workers(rig.cpus.cpu_threads(), &shared);
        if !cpu.is_empty() {
            groups.push(shared);
            workers.extend(cpu);
        }
    }
    if !rig.no_gpu {
        let (gpu_groups, gpu_workers) = spawn_gpu_groups(&rig.gpus);
        groups.extend(gpu_groups);
        workers.extend(gpu_workers);
    }
    if workers.is_empty() {
        log::warn!("no device to mine on, add --cpu to mine on the CPU");
        groups.push(Arc::new(Shared::new()));
    }
    RunningRig { farm: Farm::new(groups), workers }
}

fn spawn_grouped_workers(rig: &Rig, shared: &Arc<Shared>) -> Vec<JoinHandle<()>> {
    let mut workers = Vec::new();
    if rig.cpu {
        workers.extend(spawn_cpu_workers(rig.cpus.cpu_threads(), shared));
    }
    if !rig.no_gpu {
        workers.extend(spawn_gpu_workers(&rig.gpus, shared));
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
        let worker = spawn_gpu_worker(
            number,
            gpu,
            choice.lanes_for(position),
            choice.retune,
            shared.clone(),
            nvml.as_ref(),
            &amd_sensors,
        );
        match worker {
            Ok(worker) => workers.push(worker),
            Err(e) => log::error!("gpu{number}: {e:#}"),
        }
    }
    workers
}

fn spawn_gpu_groups(choice: &GpuChoice) -> (Vec<Arc<Shared>>, Vec<JoinHandle<()>>) {
    let nvml = if std::env::var_os("NUMEN_NO_NVML").is_some() { None } else { Nvml::init().ok().map(Arc::new) };
    let amd_sensors = AmdSensors::new();
    let gpus = gpu::gpus()
        .into_iter()
        .enumerate()
        .filter(|(number, _)| choice.mines_on_gpu(*number));
    let mut groups = Vec::new();
    let mut workers = Vec::new();
    for (position, (number, gpu)) in gpus.enumerate() {
        let shared = Arc::new(Shared::new());
        match spawn_gpu_worker(
            number,
            gpu,
            choice.lanes_for(position),
            choice.retune,
            shared.clone(),
            nvml.as_ref(),
            &amd_sensors,
        ) {
            Ok(worker) => {
                groups.push(shared);
                workers.push(worker);
            }
            Err(e) => log::error!("gpu{number}: {e:#}"),
        }
    }
    (groups, workers)
}

fn spawn_gpu_worker(
    number: usize,
    gpu: Gpu,
    lanes: Option<u32>,
    retune: bool,
    shared: Arc<Shared>,
    nvml: Option<&Arc<Nvml>>,
    amd_sensors: &AmdSensors,
) -> anyhow::Result<JoinHandle<()>> {
    match gpu {
        Gpu::Cuda(device) => gpu::CudaBackend::new(number, &device, lanes, retune).map(|backend| {
            let sensor = nvml.map(|nvml| device.sensor(nvml));
            let meter = shared.meter(format!("gpu{number}"), device.bus(), Some(backend.lanes()), sensor);
            std::thread::spawn(move || cuda_worker(shared, meter, backend))
        }),
        Gpu::Hip(device) => gpu::HipBackend::new(number, &device, lanes, retune).map(|backend| {
            let meter = shared.meter(
                format!("gpu{number}"),
                device.bus(),
                Some(backend.lanes()),
                device.sensor(amd_sensors),
            );
            std::thread::spawn(move || hip_worker(shared, meter, backend))
        }),
    }
}

fn connections(upstream: &Upstream, farm: &Farm, per_device: bool) -> Vec<(Upstream, Arc<Shared>)> {
    farm.groups()
        .iter()
        .map(|shared| {
            let device = shared
                .meters()
                .first()
                .map(|meter| meter.name.clone())
                .unwrap_or_else(|| "rig".into());
            let upstream = if per_device { upstream.for_device(&device) } else { upstream.clone() };
            shared.set_worker(upstream.worker());
            (upstream, shared.clone())
        })
        .collect()
}

async fn run_upstreams(connections: Vec<(Upstream, Arc<Shared>)>) {
    let mut links = tokio::task::JoinSet::new();
    for (upstream, shared) in connections {
        links.spawn(async move { upstream.run(&shared).await });
    }
    while let Some(result) = links.join_next().await {
        if let Err(e) = result {
            log::error!("upstream task failed: {e}");
        }
    }
}

async fn report_loop(farm: Farm, interval: u64, limit: Option<TempLimit>) {
    let every = Duration::from_secs(interval.max(1));
    let mut sample = tokio::time::interval(Duration::from_secs(1));
    let mut report = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
    loop {
        tokio::select! {
            _ = sample.tick() => {
                let now = Instant::now();
                for meter in farm.meters() {
                    meter.sample(now);
                    meter.sense(now, limit);
                }
            }
            _ = report.tick() => log::info!("{}", speed_report(&farm)),
        }
    }
}

fn speed_report(farm: &Farm) -> String {
    let now = Instant::now();
    let mut table = Table::new();
    table.set_header(["Device", "Bus", "Speed", "Accepted", "Rejected", "Temp", "Fan", "Power"]);
    for meter in farm.meters() {
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
    let (accepted, rejected) = farm.verdicts();
    format!("speed {}, accepted {accepted}, rejected {rejected}\n{table}", hashrate(farm.rate(now)))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn group(name: &str) -> Arc<Shared> {
        let shared = Arc::new(Shared::new());
        shared.meter(name.into(), String::new(), None, None);
        shared
    }

    #[test]
    fn device_connections_get_independent_workers_and_tasks() {
        let cpu = group("cpu");
        let gpu = group("gpu0");
        let farm = Farm::new(vec![cpu.clone(), gpu.clone()]);
        let endpoint = "stratum+ssl://pool.example:1234".parse().unwrap();
        let upstream = Upstream::new(vec![endpoint], Some("wallet.rig".into()), "x".into()).unwrap();

        let links = connections(&upstream, &farm, true);
        assert_eq!(links.len(), 2);
        assert_eq!(cpu.meters()[0].worker().as_deref(), Some("rig-cpu"));
        assert_eq!(gpu.meters()[0].worker().as_deref(), Some("rig-gpu0"));

        cpu.set_task([1; 32], 11.into(), 101);
        gpu.set_task([2; 32], 29.into(), 202);
        assert_eq!(cpu.difficulty(), Some(11.into()));
        assert_eq!(gpu.difficulty(), Some(29.into()));
        assert_eq!(cpu.next_job(1).unwrap().nonce_start[3], 101);
        assert_eq!(gpu.next_job(1).unwrap().nonce_start[3], 202);
    }
}
