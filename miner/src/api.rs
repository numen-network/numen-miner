use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use crate::cli::VERSION;
use crate::engine::Shared;
use crate::telemetry::Fan;

#[derive(Clone)]
struct Api {
    shared: Arc<Shared>,
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
pub async fn serve(shared: Arc<Shared>, port: u16) {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            log::warn!("api on {addr}: {e}");
            return;
        }
    };
    log::info!("api on http://{addr}/summary");
    if let Err(e) = axum::serve(listener, router(shared)).await {
        log::warn!("api on {addr}: {e}");
    }
}

fn router(shared: Arc<Shared>) -> Router {
    Router::new().route("/summary", get(summary)).with_state(Api { shared, started: Instant::now() })
}

async fn summary(State(api): State<Api>) -> Json<Summary> {
    let shared = &api.shared;
    let now = Instant::now();
    let (accepted, rejected) = shared.verdicts();
    let devices = shared
        .meters()
        .iter()
        .map(|meter| {
            let reading = meter.reading();
            Device {
                name: meter.name.clone(),
                bus: meter.bus.clone(),
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
        .collect();
    Json(Summary {
        version: VERSION,
        uptime: api.started.elapsed().as_secs(),
        hashrate: shared.rate(now),
        accepted,
        rejected,
        devices,
    })
}
