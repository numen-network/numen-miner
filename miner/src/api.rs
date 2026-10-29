use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use std::net::SocketAddr;
use std::time::Instant;

use crate::cli::VERSION;
use crate::engine::Farm;
use crate::telemetry::Fan;

#[derive(Clone)]
struct Api {
    farm: Farm,
    started: Instant,
}

#[derive(Serialize)]
struct Summary {
    version: &'static str,
    uptime: u64,
    hashrate: f64,
    accepted: u64,
    rejected: u64,
    devices: Vec<Device>,
}

#[derive(Serialize)]
struct Device {
    name: String,
    bus: String,
    worker: Option<String>,
    difficulty: Option<String>,
    batch: Option<u32>,
    hashrate: f64,
    accepted: u64,
    rejected: u64,
    temp: Option<f64>,
    fan: Option<Fan>,
    power: Option<f64>,
    paused: bool,
}

/// Serve `GET /summary` on 127.0.0.1 for as long as the process runs. A busy
/// port costs the API and never the mining.
pub async fn serve(farm: Farm, port: u16) {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            log::warn!("api on {addr}: {e}");
            return;
        }
    };
    log::info!("api on http://{addr}/summary");
    if let Err(e) = axum::serve(listener, router(farm)).await {
        log::warn!("api on {addr}: {e}");
    }
}

fn router(farm: Farm) -> Router {
    Router::new().route("/summary", get(summary)).with_state(Api { farm, started: Instant::now() })
}

async fn summary(State(api): State<Api>) -> Json<Summary> {
    let farm = &api.farm;
    let now = Instant::now();
    let (accepted, rejected) = farm.verdicts();
    let devices = farm
        .groups()
        .iter()
        .flat_map(|group| {
            let difficulty = group.difficulty().map(|difficulty| difficulty.to_string());
            group.meters().into_iter().map(move |meter| {
                let reading = meter.reading();
                Device {
                    name: meter.name.clone(),
                    bus: meter.bus.clone(),
                    worker: meter.worker(),
                    difficulty: difficulty.clone(),
                    batch: meter.batch,
                    hashrate: meter.rate(now),
                    accepted: meter.accepted(),
                    rejected: meter.rejected(),
                    temp: reading.temp,
                    fan: reading.fan,
                    power: reading.power,
                    paused: meter.paused(),
                }
            })
        })
        .collect();
    Json(Summary {
        version: VERSION,
        uptime: api.started.elapsed().as_secs(),
        hashrate: farm.rate(now),
        accepted,
        rejected,
        devices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Shared;
    use primitive_types::U256;
    use std::sync::Arc;

    fn group(name: &str, worker: &str, difficulty: u64) -> Arc<Shared> {
        let shared = Arc::new(Shared::new());
        shared.meter(name.into(), String::new(), None, None);
        shared.set_worker(Some(worker.into()));
        shared.set_task([difficulty as u8; 32], U256::from(difficulty), difficulty);
        shared
    }

    #[tokio::test]
    async fn summary_keeps_worker_and_difficulty_per_device() {
        let farm = Farm::new(vec![group("cpu", "rig-cpu", 11), group("gpu0", "rig-gpu0", 29)]);
        let Json(summary) = summary(State(Api { farm, started: Instant::now() })).await;
        assert_eq!(summary.devices.len(), 2);
        assert_eq!(summary.devices[0].worker.as_deref(), Some("rig-cpu"));
        assert_eq!(summary.devices[0].difficulty.as_deref(), Some("11"));
        assert_eq!(summary.devices[1].worker.as_deref(), Some("rig-gpu0"));
        assert_eq!(summary.devices[1].difficulty.as_deref(), Some("29"));
    }
}
