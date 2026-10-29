use crate::attribute::Retention;
use crate::Throughput;

const FEATURE_NAMES: [&str; 23] = [
    "lam21", "lam31", "rbar",
    "p1", "p2", "p3", "p4", "p5", "p6",
    "q1", "q2", "q3", "q4", "q5", "q6",
    "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7",
];

pub fn retention(r: &Retention) {
    println!("retention");
    println!("  nonces           {}", r.count);
    println!("  matched          {} ({:.4}%)", r.matched, r.rate() * 100.0);
    if r.kernel_rejected > 0 {
        println!("  kernel rejected  {}", r.kernel_rejected);
    }
    if r.canonical_rejected > 0 {
        println!("  chain rejected   {}", r.canonical_rejected);
    }
    if !r.attributed {
        println!();
        return;
    }

    println!("  drift by dimension, in bucket widths");
    let mut any = false;
    for (d, name) in FEATURE_NAMES.iter().enumerate() {
        if r.worst_drift[d] == 0.0 && r.bucket_flips[d] == 0 {
            continue;
        }
        any = true;
        println!(
            "    {:<6} worst {:>10.3e}   flips {}",
            name, r.worst_drift[d], r.bucket_flips[d]
        );
    }
    if !any {
        println!("    every dimension bit identical");
    }
    println!();
}

pub fn throughput(t: &Throughput) {
    println!("throughput  ({} nonces, {} thread(s))", t.count, t.threads);
    println!("  kernel     {:>10.1} H/s", t.kernel_per_sec);
    println!("  canonical  {:>10.1} H/s", t.canonical_per_sec);
    println!("  speedup    {:>10.2}x", t.kernel_per_sec / t.canonical_per_sec);
    println!();
}

/// Throughput alone is a half answer. This is the number that decides.
pub fn power(r: &Retention, t: &Throughput) {
    let kernel = t.kernel_per_sec * r.rate();
    let canonical = t.canonical_per_sec;
    println!("effective hash power");
    println!(
        "  kernel     {:>10.1} H/s x {:.2}% = {:.1}",
        t.kernel_per_sec,
        r.rate() * 100.0,
        kernel
    );
    println!("  canonical  {:>10.1} H/s x 100.00% = {:.1}", canonical, canonical);
    println!("  net        {:>10.2}x", kernel / canonical);
    println!();
}

pub fn gpu_power(r: &Retention, device: &str, raw: f64) {
    let effective = raw * r.rate();
    println!("effective GPU hash power ({device})");
    println!("  raw         {:>10.1} H/s", raw);
    println!("  retention   {:>9.4}%", r.rate() * 100.0);
    println!("  effective   {:>10.1} H/s", effective);
    println!();
}
