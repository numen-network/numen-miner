use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CACHE_SCHEMA: u32 = 1;
const TUNER_REVISION: u32 = 2;
const WARMUP: Duration = Duration::from_secs(10);
const WINDOW: Duration = Duration::from_secs(5);
const PASSES: usize = 3;
const CLOSE_MARGIN: f64 = 0.03;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Key {
    backend: String,
    bus: String,
    name: String,
    architecture: String,
    units: u32,
    memory: u64,
    runtime: i32,
    kernel: u64,
    tuner: u32,
}

impl Key {
    pub fn new(identity: Identity<'_>) -> Self {
        Key {
            backend: identity.backend.into(),
            bus: identity.bus.into(),
            name: identity.name.into(),
            architecture: identity.architecture.into(),
            units: identity.units,
            memory: identity.memory as u64,
            runtime: identity.runtime,
            kernel: identity.kernel,
            tuner: TUNER_REVISION,
        }
    }
}

pub struct Identity<'a> {
    pub backend: &'a str,
    pub bus: &'a str,
    pub name: &'a str,
    pub architecture: &'a str,
    pub units: u32,
    pub memory: usize,
    pub runtime: i32,
    pub kernel: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Cached {
    pub lanes: u32,
    pub mh_s: f64,
    #[serde(default)]
    pub scores: Vec<Score>,
    saved_at: u64,
}

#[derive(Default, Deserialize, Serialize)]
struct Cache {
    schema: u32,
    entries: Vec<Entry>,
}

#[derive(Deserialize, Serialize)]
struct Entry {
    key: Key,
    tuning: Cached,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Score {
    pub lanes: u32,
    pub mh_s: f64,
    pub spread_percent: f64,
}

pub struct Choice {
    pub lanes: u32,
    pub mh_s: f64,
    pub scores: Vec<Score>,
}

/// A deterministic fingerprint for embedded GPU code. It deliberately does
/// not use `DefaultHasher`, whose output is not a stable storage contract.
pub fn fingerprint(chunks: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for chunk in chunks {
        for &byte in *chunk {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

pub fn cached(key: &Key, candidates: &[u32]) -> Option<Cached> {
    let path = cache_path()?;
    let text = fs::read_to_string(&path).ok()?;
    let cache: Cache = serde_json::from_str(&text).ok()?;
    if cache.schema != CACHE_SCHEMA {
        return None;
    }
    let tuning = cache.entries.into_iter().find(|entry| &entry.key == key)?.tuning;
    if !candidates.contains(&tuning.lanes) || !tuning.mh_s.is_finite() || tuning.mh_s <= 0.0 {
        return None;
    }
    log::info!(
        "autotune cache {}: batch {}, {:.3} MH/s",
        path.display(),
        tuning.lanes,
        tuning.mh_s
    );
    Some(tuning)
}

pub fn save(key: Key, choice: &Choice) -> Result<()> {
    let path = cache_path().context("no persistent cache directory")?;
    let mut cache = fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Cache>(&text).ok())
        .filter(|cache| cache.schema == CACHE_SCHEMA)
        .unwrap_or(Cache { schema: CACHE_SCHEMA, entries: Vec::new() });
    cache.entries.retain(|entry| entry.key != key);
    cache.entries.push(Entry {
        key,
        tuning: Cached {
            lanes: choice.lanes,
            mh_s: choice.mh_s,
            scores: choice.scores.clone(),
            saved_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        },
    });
    let parent = path.parent().context("autotune cache has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&cache)?)
        .with_context(|| format!("write {}", temporary.display()))?;
    if path.exists() {
        fs::remove_file(&path).with_context(|| format!("replace {}", path.display()))?;
    }
    fs::rename(&temporary, &path).with_context(|| format!("commit {}", path.display()))?;
    log::info!("saved autotune result to {}", path.display());
    Ok(())
}

pub fn run(
    number: usize,
    base: u32,
    candidates: &[u32],
    mut launch: impl FnMut(u32) -> Result<()>,
) -> Result<Choice> {
    let warm = *candidates.iter().min_by_key(|&&lanes| lanes.abs_diff(base)).context("no autotune candidate")?;
    log::info!("gpu{number} autotune warm-up: batch {warm}, {} s", WARMUP.as_secs());
    run_for(warm, WARMUP, &mut launch)?;

    let mut samples = candidates.iter().map(|&lanes| (lanes, Vec::new())).collect::<Vec<_>>();
    for pass in 0..PASSES {
        let mut order = candidates.to_vec();
        match pass {
            1 => order.reverse(),
            2 if order.len() > 1 => order.rotate_left(1),
            _ => {}
        }
        for lanes in order {
            let rate = run_for(lanes, WINDOW, &mut launch)?;
            log::info!("gpu{number} autotune pass {} batch {lanes}: {rate:.3} MH/s", pass + 1);
            samples.iter_mut().find(|sample| sample.0 == lanes).expect("candidate sample").1.push(rate);
        }
    }

    let mut initial = scores(&samples);
    initial.sort_by(|a, b| b.mh_s.total_cmp(&a.mh_s));
    for candidate in initial.iter().take(2).rev() {
        let rate = run_for(candidate.lanes, WINDOW, &mut launch)?;
        log::info!("gpu{number} autotune confirmation batch {}: {rate:.3} MH/s", candidate.lanes);
        samples.iter_mut().find(|sample| sample.0 == candidate.lanes).expect("candidate sample").1.push(rate);
    }

    let final_scores = scores(&samples);
    let selected = select(base, &final_scores).context("autotune produced no score")?;
    for score in &final_scores {
        log::info!(
            "gpu{number} autotune result batch {}: median {:.3} MH/s, spread {:.2}%",
            score.lanes,
            score.mh_s,
            score.spread_percent
        );
    }
    log::info!(
        "gpu{number} autotune selected stable batch {} at {:.3} MH/s ({}% close-result margin)",
        selected.lanes,
        selected.mh_s,
        CLOSE_MARGIN * 100.0
    );
    Ok(Choice { lanes: selected.lanes, mh_s: selected.mh_s, scores: final_scores })
}

fn run_for(lanes: u32, duration: Duration, launch: &mut impl FnMut(u32) -> Result<()>) -> Result<f64> {
    let started = Instant::now();
    let mut hashes = 0u64;
    loop {
        launch(lanes)?;
        hashes = hashes.saturating_add(u64::from(lanes));
        if started.elapsed() >= duration {
            break;
        }
    }
    Ok(hashes as f64 / started.elapsed().as_secs_f64() / 1_000_000.0)
}

fn scores(samples: &[(u32, Vec<f64>)]) -> Vec<Score> {
    samples
        .iter()
        .filter_map(|(lanes, values)| {
            let mh_s = median(values)?;
            let lo = values.iter().copied().min_by(f64::total_cmp)?;
            let hi = values.iter().copied().max_by(f64::total_cmp)?;
            Some(Score {
                lanes: *lanes,
                mh_s,
                spread_percent: if mh_s == 0.0 { 0.0 } else { (hi - lo) / mh_s * 100.0 },
            })
        })
        .collect()
}

fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.iter().copied().filter(|value| value.is_finite()).collect::<Vec<_>>();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn select(base: u32, scores: &[Score]) -> Option<&Score> {
    let fastest = scores.iter().max_by(|a, b| a.mh_s.total_cmp(&b.mh_s))?;
    let floor = fastest.mh_s * (1.0 - CLOSE_MARGIN);
    scores
        .iter()
        .filter(|score| score.mh_s >= floor)
        .min_by_key(|score| (score.lanes.abs_diff(base), score.lanes))
}

fn cache_path() -> Option<PathBuf> {
    #[cfg(windows)]
    let root = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let root = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")));
    root.map(|root| root.join("numen-miner").join("autotune.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(lanes: u32, mh_s: f64) -> Score {
        Score { lanes, mh_s, spread_percent: 0.0 }
    }

    #[test]
    fn close_results_prefer_the_architecture_base() {
        let values = [score(10_080, 1.02), score(20_160, 1.00), score(40_320, 0.98)];
        assert_eq!(select(20_160, &values).unwrap().lanes, 20_160);
    }

    #[test]
    fn clear_gain_beats_the_base() {
        let values = [score(10_080, 1.10), score(20_160, 1.00), score(40_320, 0.95)];
        assert_eq!(select(20_160, &values).unwrap().lanes, 10_080);
    }

    #[test]
    fn median_rejects_one_outlier() {
        assert_eq!(median(&[1.0, 9.0, 1.1]), Some(1.1));
    }
}
