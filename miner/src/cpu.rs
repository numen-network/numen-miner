use std::io;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use numen_kernel::host::Scanner;
use numen_kernel::meets;
use thread_priority::{ThreadPriority, set_current_thread_priority};

use crate::engine::{Candidate, Meter, Shared};

pub fn spawn_cpu_workers(threads: Vec<Option<usize>>, shared: &Arc<Shared>) -> Vec<JoinHandle<()>> {
    if threads.is_empty() {
        return Vec::new();
    }
    let meter = shared.meter("cpu".into(), String::new(), None, None);
    log::info!("cpu threads {}", threads.len());
    threads
        .into_iter()
        .map(|cpu| {
            let shared = shared.clone();
            let meter = meter.clone();
            std::thread::spawn(move || cpu_worker(shared, meter, cpu))
        })
        .collect()
}

fn cpu_worker(shared: Arc<Shared>, meter: Arc<Meter>, cpu: Option<usize>) {
    if let Some(cpu) = cpu
        && let Err(e) = pin(cpu)
    {
        log::warn!("{} can't pin a thread to CPU {cpu}: {e}", meter.name);
    }
    let _ = set_current_thread_priority(ThreadPriority::Min);
    let mut scanner = Scanner::new();
    while !shared.stopped() {
        let Some(job) = shared.next_job(1) else {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        };
        let work = scanner.work(&job.pre_hash, &job.nonce_start);
        meter.add(1);
        if let Some(work) = work
            && meets(&work, &job.threshold)
        {
            shared.found(&meter, &job, Candidate { nonce: job.nonce_start, work });
        }
    }
}

/// Keep the calling thread on one logical CPU, numbered as the OS numbers
/// them.
#[cfg(target_os = "linux")]
fn pin(cpu: usize) -> io::Result<()> {
    // CPU_SET aborts on a CPU past the end of the set.
    if cpu >= libc::CPU_SETSIZE as usize {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_SET(cpu, &mut set) };
    if unsafe { libc::sched_setaffinity(0, size_of_val(&set), &set) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn pin(cpu: usize) -> io::Result<()> {
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadGroupAffinity};

    let (group, index) = processor_group(cpu).ok_or(io::ErrorKind::InvalidInput)?;
    let affinity = GROUP_AFFINITY { Mask: 1 << index, Group: group, Reserved: [0; 3] };
    if unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, std::ptr::null_mut()) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The processor group holding `cpu` and its index there. Windows splits more
/// than 64 CPUs into groups, and a CPU's number counts through the groups in
/// order.
#[cfg(windows)]
fn processor_group(cpu: usize) -> Option<(u16, usize)> {
    use windows_sys::Win32::System::Threading::{GetActiveProcessorCount, GetActiveProcessorGroupCount};

    let mut index = cpu;
    for group in 0..unsafe { GetActiveProcessorGroupCount() } {
        let count = unsafe { GetActiveProcessorCount(group) } as usize;
        if index < count {
            return Some((group, index));
        }
        index -= count;
    }
    None
}

#[cfg(not(any(target_os = "linux", windows)))]
fn pin(_: usize) -> io::Result<()> {
    Err(io::ErrorKind::Unsupported.into())
}
