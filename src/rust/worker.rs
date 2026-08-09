use std::fmt;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::protocol;

#[derive(Debug)]
pub enum WorkerError {
    Io(String),
    Protocol(String),
    Unavailable(String),
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::Io(e) | WorkerError::Protocol(e) | WorkerError::Unavailable(e) => {
                f.write_str(e)
            }
        }
    }
}

pub struct WorkerResponse {
    pub http_status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
    pub restart_worker: bool,
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

/// One slot in the worker pool. `proc` is None when the slot is empty (never
/// spawned yet, or its process is currently leased out for an in-flight
/// request). `leased` marks a request that has checked the process out.
struct WorkerSlot {
    proc: Option<WorkerProcess>,
    pid: u32,
    leased: bool,
    current: Option<CurrentRequest>,
}

pub struct Worker {
    launcher: String,
    version: String,
    request_timeout: Duration,
    pool_size: usize,
    next_id: AtomicU32,
    restart_count: AtomicU32,
    timeout_count: AtomicU32,
    waiting_count: AtomicU32,
    pool: Mutex<Vec<WorkerSlot>>,
    state: Mutex<WorkerState>,
}

struct WorkerState {
    last_error: Option<String>,
    last_restart_reason: Option<String>,
}

#[derive(Clone)]
struct CurrentRequest {
    id: u32,
    opcode: u16,
    started: Instant,
}

struct WaitTracker<'a> {
    worker: &'a Worker,
}

/// A worker process checked out of the pool for the duration of one request.
/// On drop it is either returned to its slot (healthy) or killed and the slot
/// left empty for a fresh spawn (discard / dead child). Because I/O happens on
/// the leased process without holding the pool lock, concurrent requests run
/// on their own workers in parallel.
struct LeasedWorker<'a> {
    worker: &'a Worker,
    slot_idx: usize,
    proc: Option<WorkerProcess>,
    pid: u32,
    discard: bool,
}

impl Worker {
    pub fn new(launcher: &str, version: String) -> Self {
        Self {
            launcher: launcher.to_string(),
            version,
            request_timeout: worker_timeout(),
            pool_size: pool_size(),
            next_id: AtomicU32::new(1),
            restart_count: AtomicU32::new(0),
            timeout_count: AtomicU32::new(0),
            waiting_count: AtomicU32::new(0),
            pool: Mutex::new(Vec::new()),
            state: Mutex::new(WorkerState {
                last_error: None,
                last_restart_reason: None,
            }),
        }
    }

    pub fn ensure_started(&self) -> Result<(), WorkerError> {
        let mut pool = self.pool.lock().map_err(|_| {
            WorkerError::Unavailable("worker pool mutex poisoned".to_string())
        })?;
        if pool.is_empty() {
            *pool = (0..self.pool_size)
                .map(|_| WorkerSlot {
                    proc: None,
                    pid: 0,
                    leased: false,
                    current: None,
                })
                .collect();
        }
        // Eagerly spawn the first worker so /health works immediately; the
        // rest of the pool spawns on demand as concurrent requests arrive.
        if pool[0].proc.is_none() {
            let (proc, pid) = spawn_worker(&self.launcher)?;
            pool[0].proc = Some(proc);
            pool[0].pid = pid;
        }
        Ok(())
    }

    pub fn health(&self) -> Result<WorkerResponse, WorkerError> {
        self.request_json(protocol::OP_HEALTH, Value::Null)
    }

    pub fn snapshot(&self) -> Value {
        let (pids, current, workers_running, active_requests) = match self.pool.lock() {
            Ok(pool) => {
                let pids: Vec<u32> = pool.iter().filter(|s| s.pid != 0).map(|s| s.pid).collect();
                let current = pool
                    .iter()
                    .find(|s| s.leased)
                    .and_then(|s| s.current.clone());
                let workers_running = pool.iter().filter(|s| s.proc.is_some()).count();
                let active = pool.iter().filter(|s| s.leased).count();
                (pids, current, workers_running, active)
            }
            Err(_) => (Vec::new(), None, 0, 0),
        };
        let (last_error, last_restart_reason) = self
            .state
            .lock()
            .map(|s| (s.last_error.clone(), s.last_restart_reason.clone()))
            .unwrap_or((None, None));
        json!({
            "pids": pids,
            "pid": pids.first().copied().unwrap_or(0),
            "pool_size": self.pool_size,
            "workers_running": workers_running,
            "active_requests": active_requests,
            "request_timeout_secs": self.request_timeout.as_secs(),
            "restart_count": self.restart_count.load(Ordering::Relaxed),
            "timeout_count": self.timeout_count.load(Ordering::Relaxed),
            "waiting_count": self.waiting_count.load(Ordering::Relaxed),
            "current_request": current.map(|r| json!({
                "id": r.id,
                "opcode": r.opcode,
                "elapsed_ms": r.started.elapsed().as_millis(),
            })),
            "last_error": last_error,
            "last_restart_reason": last_restart_reason,
        })
    }

    pub fn request_json(&self, opcode: u16, payload: Value) -> Result<WorkerResponse, WorkerError> {
        let bytes = if payload.is_null() {
            Vec::new()
        } else {
            serde_json::to_vec(&payload).map_err(|e| WorkerError::Protocol(e.to_string()))?
        };
        let frame = self.request(opcode, bytes)?;
        parse_worker_response(frame)
    }

    pub fn decrypt_batch(
        &self,
        adam: &str,
        uri: &str,
        samples: Vec<Vec<u8>>,
    ) -> Result<Vec<Vec<u8>>, WorkerError> {
        let payload = protocol::decrypt_batch_payload(adam, uri, &samples)
            .map_err(|e| WorkerError::Protocol(e.to_string()))?;
        let frame = self.request(protocol::OP_DECRYPT_BATCH, payload)?;
        if frame.flags & 1 == 1 {
            return protocol::parse_decrypt_samples_payload(&frame.payload)
                .map_err(|e| WorkerError::Protocol(e.to_string()));
        }
        let r = parse_worker_response(frame)?;
        if r.restart_worker {
            self.restart_after_delay();
        }
        Err(WorkerError::Unavailable(
            String::from_utf8_lossy(&r.body).to_string(),
        ))
    }

    fn request(&self, opcode: u16, payload: Vec<u8>) -> Result<protocol::Frame, WorkerError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let deadline = Instant::now() + self.request_timeout;
        let mut lw = self.acquire(id, opcode)?;

        let req = protocol::Frame {
            kind: protocol::KIND_REQUEST,
            request_id: id,
            opcode,
            flags: 0,
            payload,
        };

        if let Err(e) = write_frame_timeout(&mut lw.proc_mut().stdin, &req, deadline) {
            if e.kind() == io::ErrorKind::TimedOut {
                eprintln!(
                    "wrapperd: worker request opcode={opcode} timed out while writing after {:?}; discarding worker",
                    self.request_timeout
                );
                self.timeout_count.fetch_add(1, Ordering::Relaxed);
            } else {
                eprintln!(
                    "wrapperd: worker request opcode={opcode} ipc write error: {e}; discarding worker"
                );
            }
            self.record_error(e.to_string());
            lw.discard();
            return Err(WorkerError::Io(e.to_string()));
        }

        let resp = match read_frame_timeout(&mut lw.proc_mut().stdout, deadline) {
            Ok(frame) => frame,
            Err(e) => {
                if e.kind() == io::ErrorKind::TimedOut {
                    eprintln!(
                        "wrapperd: worker request opcode={opcode} timed out after {:?}; discarding worker",
                        self.request_timeout
                    );
                    self.timeout_count.fetch_add(1, Ordering::Relaxed);
                } else {
                    eprintln!(
                        "wrapperd: worker request opcode={opcode} ipc read error: {e}; discarding worker"
                    );
                }
                self.record_error(e.to_string());
                lw.discard();
                return Err(WorkerError::Io(e.to_string()));
            }
        };

        if resp.kind != protocol::KIND_RESPONSE || resp.request_id != id || resp.opcode != opcode {
            self.record_error("mismatched ipc response");
            lw.discard();
            return Err(WorkerError::Protocol("mismatched ipc response".to_string()));
        }

        drop(lw);
        Ok(resp)
    }

    /// Check a worker process out of the pool for one request. Reuses an idle
    /// live worker, or spawns into an empty slot, or (after the deadline) fails
    /// with "worker pool busy timed out". Only the slot bookkeeping holds the
    /// pool lock — the actual IPC runs on the leased process, so other requests
    /// can lease other workers concurrently.
    fn acquire(&self, id: u32, opcode: u16) -> Result<LeasedWorker<'_>, WorkerError> {
        let deadline = Instant::now() + self.request_timeout;
        let _wait = self.track_wait();
        loop {
            {
                let mut pool = self.pool.lock().map_err(|_| {
                    WorkerError::Unavailable("worker pool mutex poisoned".to_string())
                })?;
                if pool.is_empty() {
                    *pool = (0..self.pool_size)
                        .map(|_| WorkerSlot {
                            proc: None,
                            pid: 0,
                            leased: false,
                            current: None,
                        })
                        .collect();
                }

                // 1. Reuse an idle live worker.
                for (idx, slot) in pool.iter_mut().enumerate() {
                    if slot.leased {
                        continue;
                    }
                    let mut proc = match slot.proc.take() {
                        Some(p) => p,
                        None => continue,
                    };
                    let pid = slot.pid;
                    match proc.child.try_wait() {
                        Ok(Some(_)) => {
                            // Died while idle; reap and skip this slot.
                            let _ = proc.child.wait();
                            slot.pid = 0;
                            continue;
                        }
                        Ok(None) => {}
                        Err(_) => {
                            slot.pid = 0;
                            continue;
                        }
                    }
                    slot.leased = true;
                    slot.current = Some(CurrentRequest {
                        id,
                        opcode,
                        started: Instant::now(),
                    });
                    return Ok(LeasedWorker {
                        worker: self,
                        slot_idx: idx,
                        proc: Some(proc),
                        pid,
                        discard: false,
                    });
                }

                // 2. Fill an empty slot with a fresh worker.
                for (idx, slot) in pool.iter_mut().enumerate() {
                    if slot.leased || slot.proc.is_some() {
                        continue;
                    }
                    let (proc, pid) = spawn_worker(&self.launcher)?;
                    slot.leased = true;
                    slot.current = Some(CurrentRequest {
                        id,
                        opcode,
                        started: Instant::now(),
                    });
                    return Ok(LeasedWorker {
                        worker: self,
                        slot_idx: idx,
                        proc: Some(proc),
                        pid,
                        discard: false,
                    });
                }

                // 3. All workers busy.
                if Instant::now() >= deadline {
                    self.timeout_count.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "wrapperd: worker request opcode={opcode} timed out waiting for a free worker after {:?}",
                        self.request_timeout
                    );
                    return Err(WorkerError::Unavailable(
                        "worker pool busy timed out".to_string(),
                    ));
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn restart_after_delay(&self) {
        // The worker response asked for a restart (e.g. auth/session state).
        // Discard one idle worker so the next request spawns a fresh process.
        let old = {
            let mut pool = match self.pool.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            for slot in pool.iter_mut() {
                if !slot.leased && slot.proc.is_some() {
                    let proc = slot.proc.take();
                    slot.pid = 0;
                    return proc;
                }
            }
            None
        };
        if let Some(proc) = old {
            self.record_restart("restart requested by worker response");
            reap_worker(proc, "restart requested by worker response");
        }
    }

    fn track_wait(&self) -> WaitTracker<'_> {
        self.waiting_count.fetch_add(1, Ordering::Relaxed);
        WaitTracker { worker: self }
    }

    fn record_error(&self, error: impl Into<String>) {
        if let Ok(mut state) = self.state.lock() {
            state.last_error = Some(error.into());
        }
    }

    fn record_restart(&self, reason: impl Into<String>) {
        self.restart_count.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut state) = self.state.lock() {
            state.last_restart_reason = Some(reason.into());
        }
    }
}

fn set_nonblocking<T: AsRawFd>(fd: &T) -> Result<(), WorkerError> {
    let raw = fd.as_raw_fd();
    let flags = unsafe { libc::fcntl(raw, libc::F_GETFL) };
    if flags < 0 {
        return Err(io_err(io::Error::last_os_error()));
    }
    let rc = unsafe { libc::fcntl(raw, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if rc < 0 {
        return Err(io_err(io::Error::last_os_error()));
    }
    Ok(())
}

impl<'a> LeasedWorker<'a> {
    fn proc_mut(&mut self) -> &mut WorkerProcess {
        self.proc.as_mut().expect("leased worker process is present")
    }

    /// Mark this worker for discard: on drop it is killed (not returned to
    /// the pool) and the slot is left empty for a fresh spawn.
    fn discard(mut self) {
        self.discard = true;
    }
}

impl<'a> Drop for LeasedWorker<'a> {
    fn drop(&mut self) {
        let mut pool = match self.worker.pool.lock() {
            Ok(g) => g,
            Err(_) => {
                // Pool poisoned; just reap whatever process we hold.
                if let Some(proc) = self.proc.take() {
                    reap_worker(proc, "pool poisoned");
                }
                return;
            }
        };
        let slot = match pool.get_mut(self.slot_idx) {
            Some(s) => s,
            None => {
                if let Some(proc) = self.proc.take() {
                    reap_worker(proc, "slot missing");
                }
                return;
            }
        };
        if let Some(proc) = self.proc.take() {
            let mut proc = proc;
            match proc.child.try_wait() {
                Ok(None) if !self.discard => {
                    // Healthy and not being discarded: return to the pool.
                    slot.proc = Some(proc);
                    slot.pid = self.pid;
                }
                _ => {
                    // Dead, or being discarded: reap and leave the slot empty
                    // so the next request spawns a fresh worker.
                    slot.pid = 0;
                    if self.discard {
                        self.worker.record_restart("discarded worker");
                    }
                    reap_worker(proc, "discarded worker");
                }
            }
        } else {
            slot.pid = 0;
        }
        slot.leased = false;
        slot.current = None;
    }
}

impl Drop for WaitTracker<'_> {
    fn drop(&mut self) {
        self.worker.waiting_count.fetch_sub(1, Ordering::Relaxed);
    }
}

fn worker_timeout() -> Duration {
    std::env::var("WRAPPER_WORKER_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(60))
}

/// Number of independent Android worker processes for this wrapper instance.
/// Default 3 (matches the addon's 3-concurrent-rip semaphore). Each worker
/// serializes its own decrypt traffic, so concurrent rips each get their own
/// worker instead of queuing behind a single process.
fn pool_size() -> usize {
    std::env::var("WRAPPER_WORKER_POOL")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (1..=8).contains(v))
        .unwrap_or(3)
}

fn io_err(e: io::Error) -> WorkerError {
    WorkerError::Io(e.to_string())
}

/// Spawn a fresh Android worker process. Does not touch the pool (the caller
/// holds the pool lock) — returns the process and its pid.
fn spawn_worker(launcher: &str) -> Result<(WorkerProcess, u32), WorkerError> {
    eprintln!("wrapperd: starting ipc worker {launcher}");
    let mut child = Command::new(launcher)
        .env("WRAPPER_MODE", "ipc-worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(io_err)?;
    let pid = child.id();
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| WorkerError::Io("worker stdin unavailable".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| WorkerError::Io("worker stdout unavailable".to_string()))?;
    set_nonblocking(&stdin)?;
    set_nonblocking(&stdout)?;
    Ok((WorkerProcess { child, stdin, stdout }, pid))
}

/// Kill + reap a worker process in the background so a wedged child never
/// blocks the supervisor.
fn reap_worker(proc: WorkerProcess, reason: &'static str) {
    thread::spawn(move || {
        eprintln!("wrapperd: cleaning up worker: {reason}");
        let _ = proc.child.kill();
        let _ = proc.child.wait();
    });
}

fn write_frame_timeout(
    stdin: &mut ChildStdin,
    frame: &protocol::Frame,
    deadline: Instant,
) -> io::Result<()> {
    if frame.payload.len() > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "payload too large",
        ));
    }
    let mut bytes = Vec::with_capacity(20 + frame.payload.len());
    bytes.extend_from_slice(&protocol::MAGIC.to_be_bytes());
    bytes.extend_from_slice(&protocol::VERSION.to_be_bytes());
    bytes.extend_from_slice(&frame.kind.to_be_bytes());
    bytes.extend_from_slice(&frame.request_id.to_be_bytes());
    bytes.extend_from_slice(&frame.opcode.to_be_bytes());
    bytes.extend_from_slice(&frame.flags.to_be_bytes());
    bytes.extend_from_slice(&(frame.payload.len() as u32).to_be_bytes());
    bytes.extend_from_slice(&frame.payload);
    write_all_timeout(stdin, &bytes, deadline)?;
    stdin.flush()
}

fn read_frame_timeout(stdout: &mut ChildStdout, deadline: Instant) -> io::Result<protocol::Frame> {
    let mut h = [0u8; 20];
    read_exact_timeout(stdout, &mut h, deadline)?;
    let magic = u32::from_be_bytes([h[0], h[1], h[2], h[3]]);
    let version = u16::from_be_bytes([h[4], h[5]]);
    if magic != protocol::MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad ipc magic"));
    }
    if version != protocol::VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad ipc version",
        ));
    }
    let kind = u16::from_be_bytes([h[6], h[7]]);
    let request_id = u32::from_be_bytes([h[8], h[9], h[10], h[11]]);
    let opcode = u16::from_be_bytes([h[12], h[13]]);
    let flags = u16::from_be_bytes([h[14], h[15]]);
    let payload_len = u32::from_be_bytes([h[16], h[17], h[18], h[19]]) as usize;
    let mut payload = vec![0u8; payload_len];
    read_exact_timeout(stdout, &mut payload, deadline)?;
    Ok(protocol::Frame {
        kind,
        request_id,
        opcode,
        flags,
        payload,
    })
}

fn write_all_timeout<W: Write + AsRawFd>(
    writer: &mut W,
    mut buf: &[u8],
    deadline: Instant,
) -> io::Result<()> {
    while !buf.is_empty() {
        wait_writable(writer.as_raw_fd(), deadline)?;
        match writer.write(buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "worker stdin closed",
                ))
            }
            Ok(n) => buf = &buf[n..],
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn read_exact_timeout<R: Read + AsRawFd>(
    reader: &mut R,
    mut buf: &mut [u8],
    deadline: Instant,
) -> io::Result<()> {
    while !buf.is_empty() {
        wait_readable(reader.as_raw_fd(), deadline)?;
        match reader.read(buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "worker stdout closed",
                ))
            }
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn wait_readable(fd: i32, deadline: Instant) -> io::Result<()> {
    wait_fd(fd, libc::POLLIN, "worker stdout closed", deadline)
}

fn wait_writable(fd: i32, deadline: Instant) -> io::Result<()> {
    wait_fd(fd, libc::POLLOUT, "worker stdin closed", deadline)
}

fn wait_fd(fd: i32, events: i16, closed_message: &str, deadline: Instant) -> io::Result<()> {
    let mut pfd = libc::pollfd {
        fd,
        events,
        revents: 0,
    };
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "worker ipc response timed out",
            ));
        }
        let remaining = deadline.saturating_duration_since(now);
        let timeout_ms = remaining.as_millis().min(i32::MAX as u128) as i32;
        pfd.revents = 0;
        let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if n > 0 {
            if pfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, closed_message));
            }
            return Ok(());
        }
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "worker ipc response timed out",
            ));
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "worker ipc response timed out",
            ));
        }
    }
}

fn parse_worker_response(frame: protocol::Frame) -> Result<WorkerResponse, WorkerError> {
    let v: Value =
        serde_json::from_slice(&frame.payload).map_err(|e| WorkerError::Protocol(e.to_string()))?;
    let http_status = v.get("http_status").and_then(|v| v.as_u64()).unwrap_or(502) as u16;
    let content_type = v
        .get("content_type")
        .and_then(|v| v.as_str())
        .unwrap_or("application/json")
        .to_string();
    let restart_worker = v
        .get("restart_worker")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let body = match v.get("body") {
        Some(Value::String(s)) => s.as_bytes().to_vec(),
        Some(other) => serde_json::to_vec(other).unwrap_or_else(|_| {
            json!({"error":"invalid_worker_body"})
                .to_string()
                .into_bytes()
        }),
        None => Vec::new(),
    };
    Ok(WorkerResponse {
        http_status,
        content_type,
        body,
        restart_worker,
    })
}
