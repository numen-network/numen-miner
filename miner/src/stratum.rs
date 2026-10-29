use anyhow::{Context, Result, anyhow, bail, ensure};
use hex::FromHex;
use primitive_types::{H256, U256};
use rustls_platform_verifier::ConfigVerifierExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader, Lines, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::runtime::Handle;
use tokio_rustls::TlsConnector;
use tokio_rustls::rustls::ClientConfig;
use tokio_rustls::rustls::pki_types::ServerName;

use crate::engine::{Meter, Share, Shared, Submit};

const AGENT: &str = concat!("numen-miner/", env!("CARGO_PKG_VERSION"));

/// Jobs a share may still name. A pool that sends `clean` false keeps its
/// older jobs open, and a share found on one of them still counts.
const JOBS: usize = 8;

/// Silence that marks a connection dead. Both pools push a job every few
/// seconds, while a link that died without a word never errors.
const QUIET: Duration = Duration::from_secs(120);

/// Defensive bounds for a pool connection. A peer that keeps sending jobs but
/// never answers shares must not grow the pending map forever.
const MAX_PENDING: usize = 4096;
const PENDING_TTL: Duration = Duration::from_secs(120);
const MAX_LINE: usize = 64 * 1024;

/// Where a stratum pool listens.
#[derive(Clone)]
pub struct Pool {
    host: String,
    port: u16,
    tls: bool,
}

impl Pool {
    /// `address` is the `host:port` after the scheme.
    pub fn new(address: &str, tls: bool) -> Result<Pool> {
        let (host, port) = address.rsplit_once(':').context("expected host:port")?;
        let port = port.parse().with_context(|| format!("bad port {port}"))?;
        Ok(Pool { host: host.to_string(), port, tls })
    }
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scheme = if self.tls { "stratum+ssl" } else { "stratum+tcp" };
        write!(f, "{scheme}://{}:{}", self.host, self.port)
    }
}

/// The worker a pool credits, as `WALLET.WORKER`, and its password.
pub struct Login {
    pub user: String,
    pub pass: String,
}

/// A pool connection, plain or TLS.
trait Stream: AsyncRead + AsyncWrite + Unpin + Send {}

impl<T: AsyncRead + AsyncWrite + Unpin + Send> Stream for T {}

type Reader = Lines<BufReader<ReadHalf<Box<dyn Stream>>>>;
type Writer = WriteHalf<Box<dyn Stream>>;

/// One line from the pool, a push or the answer to one of our requests.
#[derive(Deserialize)]
struct Message {
    id: Option<u64>,
    method: Option<String>,
    #[serde(default)]
    params: Value,
    #[serde(default)]
    result: Value,
    #[serde(default)]
    error: Value,
}

impl Message {
    /// The pool's answer, or its reason for refusing.
    fn outcome(self) -> Result<Value> {
        if self.error.is_null() {
            return Ok(self.result);
        }
        // rplant sends [code, message, data] and ninjaraider {code, message}.
        let reason = self.error.get(1).or_else(|| self.error.get("message"));
        bail!("{}", reason.and_then(Value::as_str).map_or_else(|| self.error.to_string(), str::to_string))
    }
}

/// A pool connection past its login.
pub struct Session {
    reader: Reader,
    difficulty: U256,
    out: Arc<Outbox>,
}

/// The half of a session that workers submit through.
struct Outbox {
    writer: tokio::sync::Mutex<Writer>,
    user: String,
    extranonce: u64,
    book: Mutex<Book>,
}

struct Book {
    next_id: u64,
    /// Oldest first, so the last is the job to mine.
    jobs: VecDeque<([u8; 32], String)>,
    /// Submits awaiting the pool's verdict, by request id.
    pending: HashMap<u64, (Arc<Meter>, Instant)>,
}

impl Session {
    /// Connect, subscribe and log in.
    pub async fn open(pool: &Pool, login: &Login) -> Result<Session> {
        let tcp = TcpStream::connect((pool.host.as_str(), pool.port)).await?;
        let stream: Box<dyn Stream> = if pool.tls {
            let config = Arc::new(ClientConfig::with_platform_verifier());
            let name = ServerName::try_from(pool.host.clone())?;
            Box::new(TlsConnector::from(config).connect(name, tcp).await?)
        } else {
            Box::new(tcp)
        };
        let (reader, mut writer) = tokio::io::split(stream);
        let mut reader = BufReader::new(reader).lines();

        // The pool pushes nothing before it answers, knowing nothing of us.
        send(&mut writer, 0, "mining.subscribe", json!([AGENT])).await?;
        let subscribed = read(&mut reader).await?.outcome()?;
        let (_, extranonce, _): (Value, String, Value) = serde_json::from_value(subscribed)?;
        let extranonce = <[u8; 8]>::from_hex(&extranonce).with_context(|| format!("extranonce {extranonce}"))?;

        let out = Outbox {
            writer: tokio::sync::Mutex::new(writer),
            user: login.user.clone(),
            extranonce: u64::from_be_bytes(extranonce),
            book: Mutex::new(Book { next_id: 1, jobs: VecDeque::new(), pending: HashMap::new() }),
        };
        let mut session = Session {
            reader,
            // Stratum's default until the pool sets one.
            difficulty: U256::one(),
            out: Arc::new(out),
        };
        let id = session.out.request("mining.authorize", json!([login.user, login.pass])).await?;
        ensure!(session.answer(id).await? == true, "login {} refused", login.user);
        Ok(session)
    }

    /// Mine the pool's jobs until the connection breaks.
    pub async fn run(mut self, shared: &Shared) -> anyhow::Error {
        shared.attach(self.submitter());
        self.resume(shared);
        let mut heard = tokio::time::Instant::now();
        loop {
            let step = tokio::select! {
                message = read(&mut self.reader) => {
                    heard = tokio::time::Instant::now();
                    message.and_then(|message| self.receive(message, shared))
                }
                () = tokio::time::sleep_until(heard + QUIET) => Err(anyhow!("pool went quiet")),
            };
            if let Err(e) = step {
                return e;
            }
        }
    }

    /// Workers are plain threads, so they hand each submit to the runtime.
    fn submitter(&self) -> Submit {
        let out = self.out.clone();
        let runtime = Handle::current();
        Box::new(move |_, share| {
            let out = out.clone();
            runtime.spawn(async move {
                let name = share.meter.name.clone();
                if let Err(e) = out.submit(share).await {
                    log::warn!("{name} submit failed: {e:#}");
                }
            });
        })
    }

    /// Take a line from the pool, a verdict on a share or a push.
    fn receive(&mut self, message: Message, shared: &Shared) -> Result<()> {
        let pending = message.id.and_then(|id| self.out.book().pending.remove(&id));
        if let Some((meter, sent)) = pending {
            match message.outcome() {
                Ok(result) => shared.verdict(&meter, sent, result == true, None),
                Err(reason) => shared.verdict(&meter, sent, false, Some(&reason.to_string())),
            }
        } else if self.take(message)? {
            self.resume(shared);
        }
        Ok(())
    }

    /// Take in a push. True when it brings a job.
    fn take(&mut self, message: Message) -> Result<bool> {
        match message.method.as_deref() {
            Some("mining.set_difficulty") => {
                let (difficulty,): (u64,) = serde_json::from_value(message.params)?;
                self.difficulty = difficulty.into();
                Ok(false)
            }
            Some("mining.notify") => {
                let (id, pre_hash, clean): (String, H256, bool) = serde_json::from_value(message.params)?;
                self.out.book().open(pre_hash.0, id, clean);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Mine the newest job at the latest difficulty. A new difficulty applies
    /// from the next job on, which both pools send right behind it.
    fn resume(&self, shared: &Shared) {
        if let Some((pre_hash, id)) = self.out.book().jobs.back() {
            log::info!("new job {id}, difficulty {}", self.difficulty);
            shared.set_task(*pre_hash, self.difficulty, self.out.extranonce);
        }
    }

    /// Read until the pool answers request `id`, taking in its pushes
    /// meanwhile.
    async fn answer(&mut self, id: u64) -> Result<Value> {
        loop {
            let message = read(&mut self.reader).await?;
            if message.id == Some(id) {
                return message.outcome();
            }
            self.take(message)?;
        }
    }
}

impl Outbox {
    fn book(&self) -> MutexGuard<'_, Book> {
        self.book.lock().expect("book lock")
    }

    /// Send a share on, unless its job is closed or its extranonce belongs to
    /// an earlier connection.
    async fn submit(&self, share: Share) -> Result<()> {
        let Some((id, params)) = self.book().file(share, &self.user, self.extranonce) else {
            return Ok(());
        };
        send(&mut *self.writer.lock().await, id, "mining.submit", params).await
    }

    async fn request(&self, method: &str, params: Value) -> Result<u64> {
        let id = self.book().take_id();
        send(&mut *self.writer.lock().await, id, method, params).await?;
        Ok(id)
    }
}

impl Book {
    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn open(&mut self, pre_hash: [u8; 32], id: String, clean: bool) {
        if clean {
            self.jobs.clear();
        }
        self.jobs.retain(|(known, _)| *known != pre_hash);
        self.jobs.push_back((pre_hash, id));
        if self.jobs.len() > JOBS {
            self.jobs.pop_front();
        }
    }

    /// Book a share as pending before it goes out, so its verdict always
    /// finds it. None when its job is closed.
    fn file(&mut self, share: Share, user: &str, extranonce: u64) -> Option<(u64, Value)> {
        let now = Instant::now();
        self.pending.retain(|_, (_, sent)| now.duration_since(*sent) < PENDING_TTL);
        if self.pending.len() >= MAX_PENDING {
            log::warn!("stratum pending-share limit reached; dropping a share until the pool answers");
            return None;
        }
        let job = self.jobs.iter().find(|(pre_hash, _)| *pre_hash == share.pre_hash);
        let Some((_, job)) = job.filter(|_| share.nonce[3] == extranonce) else {
            log::debug!("{} share for a closed job dropped", share.meter.name);
            return None;
        };
        let [a, b, c, d] = share.nonce;
        let nonce = format!("{d:016x}{c:016x}{b:016x}{a:016x}");
        let params = json!([user, job, nonce, hex::encode(share.work)]);
        let id = self.take_id();
        self.pending.insert(id, (share.meter, now));
        Some((id, params))
    }
}

async fn send(writer: &mut Writer, id: u64, method: &str, params: Value) -> Result<()> {
    let mut line = json!({"id": id, "method": method, "params": params}).to_string();
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    // TLS holds bytes back until flushed.
    writer.flush().await?;
    Ok(())
}

async fn read(reader: &mut Reader) -> Result<Message> {
    let line = reader.next_line().await?.context("pool closed the connection")?;
    ensure!(line.len() <= MAX_LINE, "pool line exceeds {MAX_LINE} bytes");
    serde_json::from_str(&line).with_context(|| format!("pool sent {line}"))
}
