use anyhow::{Context, Result, anyhow, bail, ensure};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use crate::engine::Shared;
use crate::rpc;
use crate::stratum::{self, Login, Pool};

/// Pause before a retry or before reopening a link that dropped.
const BACKOFF: Duration = Duration::from_secs(1);

/// Retries an endpoint gets before the next one takes over.
const RETRIES: usize = 3;

/// Longest wait for one attempt at a link. A peer that takes the connection
/// and never answers would otherwise hold the attempt for good.
const ATTEMPT: Duration = Duration::from_secs(10);

/// How often a backup checks whether the first endpoint is back.
const FAILBACK: Duration = Duration::from_secs(300);

/// A URL to mine through, told apart by its scheme.
#[derive(Clone)]
pub enum Endpoint {
    /// A node's mining RPC over a websocket.
    Node(String),
    /// The same RPC over HTTP, the way DiffPool serves it.
    Http(String),
    Stratum(Pool),
}

impl FromStr for Endpoint {
    type Err = anyhow::Error;

    fn from_str(url: &str) -> Result<Self> {
        let (scheme, address) = url.split_once("://").context("expected a URL such as ws://127.0.0.1:9944")?;
        Ok(match scheme {
            "ws" | "wss" => Endpoint::Node(url.to_string()),
            "http" => Endpoint::Http(url.to_string()),
            "stratum+tcp" => Endpoint::Stratum(Pool::new(address, false)?),
            "stratum+ssl" => Endpoint::Stratum(Pool::new(address, true)?),
            _ => bail!("{scheme} is none of ws, wss, http, stratum+tcp and stratum+ssl"),
        })
    }
}

impl fmt::Display for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Endpoint::Node(url) | Endpoint::Http(url) => f.write_str(url),
            Endpoint::Stratum(pool) => pool.fmt(f),
        }
    }
}

impl Endpoint {
    pub fn is_stratum(&self) -> bool {
        matches!(self, Endpoint::Stratum(_))
    }
}

/// A live connection to one endpoint.
enum Link {
    Node(rpc::Node),
    Http(rpc::Http),
    Stratum(stratum::Session),
}

impl Link {
    /// Mine through the link until it breaks, and say why it did.
    async fn run(self, shared: &Arc<Shared>) -> anyhow::Error {
        match self {
            Link::Node(node) => node.run(shared).await,
            Link::Http(http) => http.run(shared).await,
            Link::Stratum(session) => session.run(shared).await,
        }
    }
}

#[derive(Clone)]
pub struct Upstream {
    endpoints: Vec<Endpoint>,
    login: Login,
    label: String,
}

impl Upstream {
    pub fn new(endpoints: Vec<Endpoint>, user: Option<String>, pass: String) -> Result<Self> {
        let stratum = endpoints.iter().any(|endpoint| matches!(endpoint, Endpoint::Stratum(_)));
        ensure!(user.is_some() || !stratum, "stratum pools need --user WALLET.WORKER");
        let login = Login { user: user.unwrap_or_default(), pass };
        Ok(Upstream { endpoints, login, label: "rig".into() })
    }

    pub fn for_device(&self, device: &str) -> Self {
        let mut upstream = self.clone();
        if upstream.endpoints.iter().any(Endpoint::is_stratum) {
            upstream.login = upstream.login.for_device(device);
        }
        upstream.label = device.into();
        upstream
    }

    pub fn worker(&self) -> Option<String> {
        if self.endpoints.iter().any(Endpoint::is_stratum) {
            self.login.worker()
        } else {
            None
        }
    }

    /// Mine through the first endpoint that answers. A link that drops is
    /// opened again, one that stays down hands over to the next endpoint, and
    /// a backup gives way to the first endpoint once it answers again.
    pub async fn run(&self, shared: &Arc<Shared>) {
        let mut index = 0;
        let mut ready = None;
        loop {
            let endpoint = &self.endpoints[index];
            let link = match ready.take() {
                Some(link) => link,
                None => match self.connect(endpoint).await {
                    Some(link) => link,
                    None => {
                        index = (index + 1) % self.endpoints.len();
                        continue;
                    }
                },
            };
            log::info!("{} connected to {endpoint}", self.label);
            let error = tokio::select! {
                error = link.run(shared) => Some(error),
                first = self.failback(), if index > 0 => {
                    ready = Some(first);
                    index = 0;
                    None
                }
            };
            shared.detach();
            if let Some(error) = error {
                log::warn!("{} {endpoint}: {error:#}", self.label);
                tokio::time::sleep(BACKOFF).await;
            }
        }
    }

    /// Open a link, retrying the attempts that fail and logging why each did.
    async fn connect(&self, endpoint: &Endpoint) -> Option<Link> {
        for _ in 0..=RETRIES {
            match self.open(endpoint).await {
                Ok(link) => return Some(link),
                Err(e) => log::warn!("{} connect to {endpoint}: {e:#}", self.label),
            }
            tokio::time::sleep(BACKOFF).await;
        }
        None
    }

    /// One attempt at a link, given up after `ATTEMPT`.
    async fn open(&self, endpoint: &Endpoint) -> Result<Link> {
        let open = async {
            anyhow::Ok(match endpoint {
                Endpoint::Node(url) => Link::Node(rpc::Node::open(url).await?),
                Endpoint::Http(url) => Link::Http(rpc::Http::open(url).await?),
                Endpoint::Stratum(pool) => Link::Stratum(stratum::Session::open(pool, &self.login).await?),
            })
        };
        tokio::time::timeout(ATTEMPT, open).await.map_err(|_| anyhow!("no answer in {} s", ATTEMPT.as_secs()))?
    }

    /// A link to the first endpoint, once it answers again.
    async fn failback(&self) -> Link {
        loop {
            tokio::time::sleep(FAILBACK).await;
            if let Some(link) = self.connect(&self.endpoints[0]).await {
                return link;
            }
        }
    }
}
