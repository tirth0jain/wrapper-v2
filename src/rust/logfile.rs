// Logfile: tees wrapperd output to a file (WRAPPER_LOG_FILE) in addition to
// stderr, so full debug history survives container restarts and can be
// inspected without docker logs. The file is truncated once at startup
// (each build starts clean). All functions are best-effort — logging never
// fails a request.
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);

/// Initialize the file logger. Opens (and truncates) WRAPPER_LOG_FILE when
/// set. Call once at startup.
pub fn init() {
    if let Ok(path) = std::env::var("WRAPPER_LOG_FILE") {
        if path.is_empty() {
            return;
        }
        // Ensure parent dir exists.
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path);
        match f {
            Ok(file) => {
                let mut guard = LOG_FILE.lock().unwrap_or_else(|p| p.into_inner());
                *guard = Some(file);
                eprintln!("wrapperd: logging to file {path} (truncated)");
            }
            Err(e) => eprintln!("wrapperd: cannot open log file {path}: {e}"),
        }
    }
}

/// Whether verbose per-request tracing is enabled (WRAPPER_RUNTIME_TRACE=1).
pub fn trace_enabled() -> bool {
    std::env::var("WRAPPER_RUNTIME_TRACE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Write a line to the log file (if configured) AND stderr. Timestamped.
pub fn log(msg: &str) {
    let line = format!("[{}] {msg}\n", now_ms());
    eprint!("{line}");
    if let Ok(mut guard) = LOG_FILE.lock() {
        if let Some(f) = guard.as_mut() {
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
    }
}
