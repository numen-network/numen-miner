# numen-miner

Miner for [Numen](https://github.com/numen-network/numen)'s PoScan proof of work. It mines on NVIDIA and AMD GPUs, and on the CPU if you ask, solo through a node you run or at a pool.

## Quick start

Start a node with your reward address. `--miner` tells the node where rewards go and opens the mining RPC.

```bash
./numen --chain mainnet-raw.json --miner <YOUR_ADDRESS>
```

Download the archive for your system from [Releases](https://github.com/numen-network/numen-miner/releases) and point the miner at the node.

```bash
./numen-miner --url ws://127.0.0.1:9944
```

For rigs on other machines, add `--rpc-external --rpc-cors all` and open port 9944 to your rigs only.

```bash
./numen --chain mainnet-raw.json --miner <YOUR_ADDRESS> --rpc-external --rpc-cors all
```

On Linux

```bash
./numen-miner --url ws://<NODE_IP>:9944
```

On Windows (PowerShell)

```bash
./numen-miner.exe --url ws://<NODE_IP>:9944
```

Windows Defender may flag `numen-miner.exe` as a trojan. Exclude the miner's folder from an administrator PowerShell.

```powershell
Add-MpPreference -ExclusionPath <MINER_FOLDER>
```

The miner uses every GPU it finds. Add `--cpu` to put the CPU to work as well, and `--no-gpu` to leave the GPUs out. Rewards always go to the node's `--miner` address.

### Pools

To mine at a pool instead, pick one and follow its setup. No node needed.

- [rplant](https://pool.rplant.xyz)
  ```
  ./numen-miner -o stratum+ssl://eu.rplant.xyz:17189 -u <YOUR_ADDRESS>.<RIG_NAME>
  ```
- [ninjaraider](https://ninjaraider.com/numen-pplns/guide)
  ```
  ./numen-miner -o stratum+ssl://numen.ninjaraider.com:44961 -u <YOUR_ADDRESS>.<RIG_NAME>
  ```
- [DiffPool](https://diffpool.xyz/numen/mine)
  ```
  ./numen-miner --url http://stratum.diffpool.xyz:8010/rpc/<YOUR_ADDRESS>/<RIG_NAME>
  ```
- [Nitopool](https://pool.nitopool.online/pools/numenpplnsbf)
  ```
  ./numen-miner -o stratum+ssl://stratum.nitopool.online:3027 -u <YOUR_ADDRESS>.<RIG_NAME>
  ```

The release archives carry each of these as a script named after its pool, `.bat` on Windows and `.sh` on Linux. Set `WALLET` in yours and run it.

With a Stratum pool, the miner automatically opens one connection per device so the pool can tune VarDiff separately. The base `WALLET.RIG_NAME` login becomes `WALLET.RIG_NAME-cpu`, `WALLET.RIG_NAME-gpu0`, `WALLET.RIG_NAME-gpu1`, and so on. Every device still uses the same pool port.

## GPU support

| Vendor | Cards                           | Needs                                |
| ------ | ------------------------------- | ------------------------------------ |
| NVIDIA | Kepler (compute 3.0) and newer  | Driver 470 or newer, no CUDA toolkit |
| AMD    | Polaris, Vega, RDNA 1 to RDNA 4 | HIP runtime                          |

On Windows the Adrenalin driver ships the HIP runtime as `amdhip64_6.dll` or `amdhip64_7.dll`. On Linux, install the HIP runtime from ROCm so that `libamdhip64.so.7`, `.so.6` or `.so.5` loads, and add its directory to `LD_LIBRARY_PATH` if the loader can't find it. HiveOS rigs need the same library.

The binary carries AMD code objects for gfx803, gfx900, gfx906, gfx90c, gfx1010 to gfx1012, gfx1030 to gfx1036, gfx1100 to gfx1103, gfx1150 to gfx1153, gfx1200 and gfx1201. `numen-miner devices` shows the one each card loaded, or `unsupported`.

Kepler and Maxwell cards can't be preempted in the middle of a launch. On Windows a launch that outlasts the driver's two-second timeout resets the driver, and that GPU logs errors until the miner restarts. Lower its `--batch` until the resets stop.

## Usage

`numen-miner` mines, and its subcommands look after the devices it mines on.

| Command               | Effect                                                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `numen-miner`         | Mine through a node or a pool                                                                                        |
| `numen-miner bench`   | Measure the chosen devices on a fixed task for a repeatable duration. No node needed                                 |
| `numen-miner devices` | List every GPU with the number `--devices` takes, then exit                                                          |
| `numen-miner dump`    | Write one device's raw features for a run of nonces, for `numen-harness retention --from` to grade against the chain |

### numen-miner

```bash
./numen-miner --url ws://10.0.0.2:9944,ws://10.0.0.3:9944
```

| Option                       | Default                 | Effect                                                                                                                                                                             |
| ---------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-o, --url <URL>`            | required                | Node websockets (`ws`, `wss`), DiffPool style endpoints (`http`) or stratum pools (`stratum+tcp`, `stratum+ssl`) in failover order. Repeat the option or separate URLs with commas |
| `-u, --user <WALLET.WORKER>` | none                    | Stratum login, needed for a stratum pool                                                                                                                                           |
| `-p, --pass <PASS>`          | `x`                     | Stratum password                                                                                                                                                                   |
| `--worker-mode <MODE>`       | `auto`                  | Connection layout: automatic per-device Stratum workers, one shared `rig` worker, or forced per-`device` connections                                                              |
| `--cpu`                      | off                     | Mine on the CPU as well                                                                                                                                                            |
| `-t, --threads <N>`          | every core              | CPU threads when `--cpu` is on                                                                                                                                                     |
| `--cpu-affinity <LIST>`      | off                     | Logical CPUs to pin one CPU thread each to, as in `0-7,16`, in place of `--threads`                                                                                                |
| `--no-gpu`                   | off                     | Skip the GPUs without opening any                                                                                                                                                  |
| `-d, --devices <LIST>`       | every GPU               | GPUs to mine on, by the numbers `numen-miner devices` prints                                                                                                                       |
| `-b, --batch <LIST>`         | cached autotune         | Override the automatically selected nonces per launch for each mining GPU                                                                                                          |
| `--retune`                   | off                     | Ignore saved automatic batches and measure the selected GPUs again                                                                                                                 |
| `--temp-stop <C>`            | off                     | Pause a GPU that stays at or above this temperature for ten seconds                                                                                                                |
| `--temp-start <C>`           | ten under `--temp-stop` | Resume a paused GPU at or below this temperature                                                                                                                                   |
| `-i, --report-interval <S>`  | 60                      | Seconds between speed tables                                                                                                                                                       |
| `--api-port <PORT>`          | 4068                    | Port of the `/summary` API on 127.0.0.1                                                                                                                                            |

Several URLs in `--url` fail over in order. The miner mines through the first one that answers, and the list may mix nodes and pools. A connection that drops is opened again, and one that stays down through three retries a second apart hands over to the next URL. An attempt with no answer for ten seconds counts as failed, and every failed attempt logs its reason. While on a backup, the miner tries the first URL again every five minutes. The devices idle while no URL is connected.

`--worker-mode auto` gives the CPU group and every GPU its own upstream whenever the failover list contains a Stratum pool. Each connection then has its own VarDiff state, difficulty, jobs, extranonce and reconnect cycle. One failing device connection does not stop the others. Use `--worker-mode rig` to keep one shared connection, or `--worker-mode device` to force separate connections for every endpoint type.

`--cpu` runs a thread per logical CPU, or `--threads` of them, and leaves it to the OS to place them. `--cpu-affinity 0-7,16` runs one thread on each listed CPU instead and pins it there. CPUs go by the numbers the OS gives them. Windows numbers the two threads of a core side by side, while Linux usually puts them a core count apart, as `lscpu -e` shows, so on a 16 core CPU `0-15` means both threads of eight cores on Windows and one thread on each of the sixteen on Linux. A CPU the OS refuses, such as one past the last, costs a warning and its thread runs unpinned.

`--no-gpu` mines without the GPUs and never opens one, which leaves them free for another miner.

`--batch` sets how many nonces each mining GPU takes per launch, one value per GPU in the order `devices` lists them. `--devices 0,2 --batch 8192,4096` gives GPU 0 8192 nonces and GPU 2 4096. GPUs past the end of the list take its last value, so `--batch 4096` alone sets every GPU.

When `--batch` is omitted, a GPU without a valid saved result gets a ten-second warm-up, three five-second measurements of four architecture-scaled sizes in changing order, then confirmation measurements of the best two. A candidate within three percent of the fastest is treated as equivalent and the size closest to the architecture baseline wins, which avoids chasing clock noise. The cache is invalidated when the GPU identity, driver or HIP runtime, embedded kernel or tuner revision changes. Use `--retune` after changing clocks, power limits or cooling. An explicit `--batch` skips both the cache and tuning entirely. The tuning allocation is capped at one quarter of VRAM.

The cache is `autotune.json` beside the running `numen-miner` executable on every platform. Each copied installation therefore keeps its own tuning result. Move that file with the miner to reuse its tuning, delete it to start fresh, or use `--retune` to replace its saved result explicitly.

`--temp-stop 85` pauses a GPU that has stayed at 85 C or hotter for ten seconds and resumes it at 75 C, or at `--temp-start` if you set one. Readings come from NVML on NVIDIA cards, and on AMD cards from the amdgpu driver's sysfs on Linux and ADLX on Windows. A card without readings never pauses.

Every option also comes from an environment variable, `NUMEN_MINER_` plus the long name in capitals with underscores for dashes, as in `NUMEN_MINER_TEMP_STOP` for `--temp-stop`. Lists there take commas, as in `NUMEN_MINER_URL=ws://10.0.0.2:9944,ws://10.0.0.3:9944`, and `NUMEN_MINER_CPU=1` turns on CPU mining.

### numen-miner bench

```bash
./numen-miner bench --devices 0 --duration 60 --json benchmark.json
```

| Option                  | Default          | Effect                                                                              |
| ----------------------- | ---------------- | ----------------------------------------------------------------------------------- |
| `--cpu`                 | off              | Mine on the CPU as well                                                             |
| `-t, --threads <N>`     | every core       | CPU threads when `--cpu` is on                                                      |
| `--cpu-affinity <LIST>` | off              | Logical CPUs to pin one CPU thread each to, as in `0-7,16`, in place of `--threads` |
| `--no-gpu`              | off              | Skip the GPUs without opening any                                                   |
| `-d, --devices <LIST>`  | every GPU        | GPUs to mine on, by the numbers `numen-miner devices` prints                        |
| `-b, --batch <LIST>`    | cached autotune  | Override the automatically selected nonces per launch for each mining GPU           |
| `--retune`              | off              | Ignore saved automatic batches and measure again                                    |
| `--duration <S>`        | 60               | Measured seconds after setup and autotuning; zero runs until Ctrl+C                  |
| `-i, --report-interval` | 10               | Seconds between progress tables                                                     |
| `--json <FILE>`         | none             | Write the final duration, rate, work count and per-device measurements as JSON       |

`bench` needs no node. Setup and autotuning happen before its duration starts. Without `--batch` it uses the same persistent autotune result as normal mining; pass an explicit value to compare a fixed size or `--retune` to refresh the saved choice. Its device options come from the same environment variables as mining.

### numen-miner devices

```bash
./numen-miner devices
```

It takes no options.

`devices` puts the number `--devices` takes in the first column. NVIDIA cards come first, then AMD, each in PCI order. An AMD row shows its architecture under `Compute`.

### numen-miner dump

```bash
./numen-miner dump --device 0 --count 33600 --out gpu0.txt
```

| Option             | Default  | Effect                                                                     |
| ------------------ | -------- | -------------------------------------------------------------------------- |
| `--cpu`            | off      | Dump the CPU, on the widest path it runs                                   |
| `-d, --device <N>` | none     | GPU to dump, by the number `numen-miner devices` prints                    |
| `--count <N>`      | 512      | Nonces to dump, counting up from zero                                      |
| `--pre-hash <HEX>` | `0x2a`   | Pre hash as a big endian number in hex, so `0x2a` fills only the last byte |
| `--out <FILE>`     | required | File to write                                                              |

`dump` takes exactly one of `--cpu` and `--device` and needs no node. It writes a line per nonce for [`numen-harness retention`](#numen-harness-retention) to grade, so a rig without the harness can still be graded.

## API

The miner serves `GET /summary` on `127.0.0.1:4068` for dashboards and rig managers. A busy port costs a warning and never the mining.

```bash
curl -s http://127.0.0.1:4068/summary
```

```json
{
  "version": "0.1.0 (1a2b3c4)",
  "uptime": 3600,
  "hashrate": 15298.0,
  "accepted": 12,
  "rejected": 1,
  "devices": [
    {
      "name": "gpu0",
      "bus": "2b:00.0",
      "worker": "rig-gpu0",
      "difficulty": "4096",
      "batch": 40320,
      "hashrate": 15298.0,
      "accepted": 12,
      "rejected": 1,
      "temp": 64.0,
      "fan": { "percent": 55 },
      "power": 245.0,
      "paused": false
    }
  ]
}
```

`hashrate` is hashes per second over the last minute, or since startup in the first minute. `worker` is the pool worker without the wallet and is `null` outside Stratum. `difficulty` is a decimal string because node difficulties may exceed JSON's exact integer range, and is `null` while the device has no active task. `batch` is the selected GPU launch size and is `null` for the CPU. `temp` is in degrees Celsius and `power` in watts. `fan` is `{"percent": n}` from NVML and `{"rpm": n}` on AMD. A reading the driver doesn't report is `null`. The CPU shows up as `cpu` with an empty `bus`.

## HiveOS

Create a flight sheet with any coin name such as `NUMN` and your wallet. Set the pool to **Configure in miner** and the miner to **Custom**, then fill in the miner setup.

| Field                      | Value                                                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Installation URL           | Link to `numen-miner-<version>_hiveos.tar.gz` from Releases                                                |
| Wallet and worker template | `%WAL%.%WORKER_NAME%` for a stratum pool                                                                   |
| Pool URL                   | Node websocket such as `ws://192.168.1.10:9944` or a pool URL. Separate more URLs with spaces for failover |
| Pass                       | `x`, or what the pool's setup asks for                                                                     |
| Extra config arguments     | Any other options, such as `--cpu` or `--temp-stop 85`                                                     |

The template goes to `--user` and the pass to `--pass`. Solo mining leaves both unused, since the node's `--miner` address gets the rewards. The dashboard reads GPU rates and share counts from the `/summary` API, so keep `--api-port` at 4068. The log lands in `/var/log/miner/custom/custom.log`.

## mmpOS

Create a miner profile, choose the custom miner and paste the link to `numen-miner-<version>_mmpos.tar.gz` from Releases. Under the advanced settings, set the arguments to `./numen-miner --url ws://192.168.1.10:9944` for solo or `./numen-miner --url stratum+ssl://<POOL_HOST>:<PORT> --user <YOUR_ADDRESS>.rig1` for a pool, plus any other options.

The profile's pool and wallet go unused, since the arguments carry them. The dashboard reads rates and share counts from the `/summary` API, so keep `--api-port` at 4068.

## How it works

A node or pool hands out tasks, each a pre-hash and a difficulty. Each nonce seeds a procedurally generated 3D body, which the miner scans into a spectral signature and hashes into a work value. A work value that clears the difficulty seals a block. A pool sets its difficulty far lower, so its shares measure the work done.

The low 64 bits of the nonce count up from zero for each pre-hash. The next 128 bits are random per device group. The top 64 bits are that device connection's pool extranonce, or random per node connection, so devices and rigs never scan the same nonces.

Consumer GPUs run f64 at 1/16 to 1/64 of their f32 rate, so the kernels compute in f32 and miss the chain's answer on a few percent of nonces. Hits go out unchecked, so the few that miss come back rejected.

## Performance

Effective hash rate is hash rate times retention, the share of nonces where the kernel agrees with the chain. Retention is a price, not a gate. A variant that trades some for more speed wins when the product grows.

The optimized GPU kernel keeps the same arithmetic while storing compact scalar vertex displacements and sharing exact coordinate reconstruction across fixed 16-face icosphere families. On the reference RX 7900 XT this increased measured end-to-end throughput by 6.375% with byte-identical GPU feature dumps. AMD mining also pipelines dispatch and pinned readback on one ordered HIP stream; scan kernels never overlap.

The gfx1100 kernel also evaluates one statistically ranked adjacent quantization bucket when a feature lies close to a measured boundary. GPU candidates are deduplicated and recomputed canonically before submission. Three RX 7900 XT samples improved canonical coverage from 94.1–94.6% to 99.35–99.49%. This calibrated profile is enabled only for gfx1100; every other AMD architecture and NVIDIA keep the single-bucket kernel.

## Building from source

```bash
cargo build --release
```

Stable Rust builds the miner and the harness with no CUDA toolkit or ROCm. The PTX kernels in `miner/ptx` and the AMD code objects in `miner/gfx` are checked in and embedded at build time, so only kernel changes need the GPU toolchains below.

### PTX

```bash
rustup toolchain install nightly --component rust-src --component llvm-bitcode-linker
./scripts/build-ptx.sh
```

The script stages `modern.ptx` for Volta and newer and `legacy.ptx`, built for sm_30, for older cards.

### AMD code objects

```bash
rustup toolchain install nightly --component rust-src --component llvm-tools
./scripts/build-gfx.sh
```

The script builds one code object per architecture into `miner/gfx`. Name architectures, as in `./scripts/build-gfx.sh gfx1036`, to build only those.

## Harness

`numen-harness` grades the kernel against the chain's own crates.

| Command                    | Effect                                                              |
| -------------------------- | ------------------------------------------------------------------- |
| `numen-harness retention`  | Compare the kernel against the canonical pipeline nonce by nonce    |
| `numen-harness throughput` | Measure throughput on both sides                                    |
| `numen-harness power`      | Retention and throughput together, reported as effective hash power |
| `numen-harness gpu-power`  | Combine a GPU feature dump with a miner benchmark JSON               |

### numen-harness retention

```bash
./target/release/numen-harness retention --count 512 --attribute
./target/release/numen-harness retention --from gpu0.txt --attribute
```

| Option             | Default | Effect                                                                                                      |
| ------------------ | ------- | ----------------------------------------------------------------------------------------------------------- |
| `--count <N>`      | 512     | Nonces to run, counting up from zero                                                                        |
| `--pre-hash <HEX>` | `0x2a`  | Pre hash in hex, a short one landing in the trailing bytes                                                  |
| `--attribute`      | off     | Report which feature dimensions moved, not just how many nonces did                                         |
| `--from <FILE>`    | none    | Grade a `numen-miner dump` file instead of the CPU kernel here. The file brings its own pre hash and nonces |

`--attribute` reports per-dimension drift in bucket widths, the unit the protocol quantizes in. A dimension at 1e-9 has margin to spare and one at 0.4 is about to start flipping buckets.

`--from` is how a GPU gets graded. A GPU sums in its own order and takes the hardware's approximate log2 and exp2, so the CPU kernel run here says nothing about it, while a [`numen-miner dump`](#numen-miner-dump) file carries the GPU's own features.

### numen-harness throughput

```bash
./target/release/numen-harness throughput --count 256 --threads 8
```

| Option             | Default | Effect                                                     |
| ------------------ | ------- | ---------------------------------------------------------- |
| `--count <N>`      | 256     | Nonces to run, counting up from zero                       |
| `--pre-hash <HEX>` | `0x2a`  | Pre hash in hex, a short one landing in the trailing bytes |
| `--threads <N>`    | 1       | Threads to time each side on                               |

`throughput` times both sides without checking retention.

### numen-harness power

```bash
./target/release/numen-harness power --count 512 --threads 8
```

| Option             | Default | Effect                                                     |
| ------------------ | ------- | ---------------------------------------------------------- |
| `--count <N>`      | 512     | Nonces to run, counting up from zero                       |
| `--pre-hash <HEX>` | `0x2a`  | Pre hash in hex, a short one landing in the trailing bytes |
| `--threads <N>`    | 1       | Threads to time each side on                               |

`power` is the number that decides. It reports throughput and retention separately and then multiplies them. Its retention half runs on every core whatever `--threads` says.

### Reproducible GPU power benchmark

Build the workspace in release mode, make sure no miner is running, then on Windows run:

```powershell
cargo build --release --workspace
powershell -ExecutionPolicy Bypass -File scripts/benchmark-effective.ps1 -Device 0 -Duration 60 -RetentionCount 512
```

The script refuses to start while another `numen-miner` process exists and runs its benchmark and feature dump sequentially. It creates a timestamped directory under `benchmark-results` containing the raw benchmark JSON, the GPU feature dump and `effective.txt`. Effective hash power is the measured raw GPU rate multiplied by the dump's chain retention.

The same final calculation can be run directly on every platform:

```bash
./target/release/numen-harness gpu-power --from gpu0.txt --benchmark benchmark.json --device gpu0 --attribute
```

## Layout

| Directory   | Holds                                                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| `kernel`    | The pipeline. It compiles for the host, and `no_std` for `nvptx64-nvidia-cuda` and `amdgcn-amd-amdhsa`                      |
| `miner`     | Node and pool clients, nonce handout, backends, telemetry and API, with the embedded kernels in `miner/ptx` and `miner/gfx` |
| `harness`   | The scale. It links the real chain crates rather than a second copy, which is the whole point of grading against it         |
| `packaging` | Windows and Linux start scripts plus HiveOS and mmpOS extras that `scripts/package.sh` bundles into the release archives    |

## License

MIT
