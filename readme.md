# HASH Miner — CPU + GPU hybrid

Mines HASH token on Ethereum mainnet using **both CPU (rayon) and GPU (CUDA)**
simultaneously. The first thread (CPU or GPU) to find a valid nonce wins;
the result is submitted on-chain immediately.

## Hardware this was tuned for

| Component | Spec |
|-----------|------|
| GPU | RTX PRO 6000 S (Blackwell, sm_100, 95.6 GB VRAM, 93.6 TFLOPS) |
| CPU | AMD EPYC 9554 — 64 cores / 128 threads (using 32 alloc'd) |

## Requirements

| Tool | Version |
|------|---------|
| Rust | 1.78+ |
| CUDA Toolkit | 12.x (`nvcc` in PATH) |
| `cudarc` crate | 0.9 (pinned in Cargo.toml) |

## Project layout

```
.
├── Cargo.toml        ← dependencies + "gpu" feature flag
├── build.rs          ← compiles hash_kernel.cu → PTX at build time
├── hash_kernel.cu    ← CUDA Keccak-256 kernel
└── src/
    └── main.rs       ← hybrid CPU+GPU miner
```

## Setup

```bash
cp .env.example .env
# Edit .env with your keys
```

**.env**
```env
PRIVATE_KEY=0xyour_private_key
INFURA_URL=https://mainnet.infura.io/v3/YOUR_KEY

# Optional Telegram notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# CPU threads (default: num_cpus)
THREADS=32

# GPU nonces per kernel launch — tune for your GPU
# RTX PRO 6000 S: 1<<22 (4 M) is a safe starting point
# Increase to 1<<23 or 1<<24 if GPU utilisation < 90%
GPU_BATCH=4194304

# Set to false to disable GPU (CPU-only mode)
USE_GPU=true
```

## Build

```bash
# With GPU (default)
cargo build --release

# Override SM target for Blackwell RTX PRO 6000 S:
CUDA_ARCH=sm_100 cargo build --release

# CPU-only (no CUDA required)
cargo build --release --no-default-features
```

## Run

```bash
./target/release/hash-miner
```

## How nonce space is partitioned

```
Nonce 0        → CPU thread 0
Nonce 1        → CPU thread 1
...
Nonce N-1      → CPU thread N-1   (N = THREADS)
Nonce N        → GPU batch 0      (GPU_BATCH nonces per launch)
Nonce N+stride → GPU batch 1
...
```

`stride = THREADS + 1` so CPU and GPU never hash the same nonce.

## Performance expectations

| Mode | Approximate throughput |
|------|----------------------|
| CPU only (32 threads) | ~80–120 MH/s |
| GPU only (RTX PRO 6000 S) | ~800–1200 MH/s |
| **CPU + GPU combined** | **~900–1320 MH/s** |

GPU throughput scales with `GPU_BATCH`; increase it until GPU utilisation
(check with `nvidia-smi`) stays above 95%.

## Live stats output

```
[STATS] CPU 95.40 MH/s | GPU 1024.00 MH/s | Total 1119.40 MH/s | Runtime: 42s
```

## Tuning tips

1. **GPU_BATCH** — start at `4194304` (2²²). Double it until `nvidia-smi`
   shows ~99% GPU utilisation.
2. **THREADS** — set to your available vCPU count. On EPYC 9554 with 32
   alloc'd cores, `32` is optimal.
3. **CUDA_ARCH** — use `sm_100` for RTX PRO 6000 S (Blackwell).
   `sm_89` is Ada and works but misses Blackwell-specific optimisations.