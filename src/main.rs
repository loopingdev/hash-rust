use anyhow::Result;
use dotenv::dotenv;
use ethers::abi::{encode, Token};
use ethers::contract::abigen;
use ethers::core::types::{Address, BlockNumber, U256};
use ethers::middleware::SignerMiddleware;
use ethers::providers::{Http, Middleware, Provider};
use ethers::signers::{LocalWallet, Signer};

use rayon::prelude::*;
use sha3::{Digest, Keccak256};

use reqwest;
use serde_json;

use std::env;
use std::str::FromStr;

use std::sync::{
    Arc,
    Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use std::time::{Duration, Instant};

abigen!(
    HashContract,
    r#"[
        function getChallenge(address miner) view returns (bytes32)
        function miningState() view returns (uint256,uint256,uint256,uint256,uint256,uint256,uint256)
        function mine(uint256 nonce)
    ]"#
);

const CONTRACT: &str = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc";

// Separate atomic counters so we can report CPU vs GPU hashrates independently
static CPU_HASH_COUNTER: AtomicU64 = AtomicU64::new(0);
static GPU_HASH_COUNTER: AtomicU64 = AtomicU64::new(0);

// ─── GPU kernel source (PTX via nvcc or inline CUDA C) ───────────────────────
//
// We embed the CUDA kernel as a PTX string compiled at build time.
// See build.rs for how to compile hash_kernel.cu → hash_kernel.ptx.
//
// The kernel hashes batches of nonces with Keccak-256 (ABI-encoded as
// abi.encode(bytes32 challenge, uint256 nonce)) and writes the first nonce
// whose hash < difficulty into *result.
//
// PTX is loaded at runtime via cudarc; no linking against libcuda at compile
// time is required beyond what cudarc handles.
// ─────────────────────────────────────────────────────────────────────────────

/// Spawns a CPU mining thread-pool (rayon) that searches nonces in the range
/// [cpu_start, ∞) stepping by `total_threads`.
fn spawn_cpu_miner(
    challenge: [u8; 32],
    difficulty_bytes: [u8; 32],
    cpu_threads: usize,
    total_threads: usize, // cpu_threads + gpu_virtual_threads
    cpu_offset: u64,      // first nonce assigned to CPU block
    found_nonce: Arc<Mutex<Option<u64>>>,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let difficulty = U256::from_big_endian(&difficulty_bytes);

        (0..cpu_threads).into_par_iter().for_each(|tid| {
            let mut nonce: u64 = cpu_offset + tid as u64;

            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let encoded = encode(&[
                    Token::FixedBytes(challenge.to_vec()),
                    Token::Uint(U256::from(nonce)),
                ]);

                let mut hasher = Keccak256::new();
                hasher.update(&encoded);
                let result = hasher.finalize();
                let hash_u256 = U256::from_big_endian(&result);

                CPU_HASH_COUNTER.fetch_add(1, Ordering::Relaxed);

                if hash_u256 < difficulty {
                    let mut found = found_nonce.lock().unwrap();
                    if found.is_none() {
                        *found = Some(nonce);
                        println!("\n[CPU FOUND] Nonce = {}", nonce);
                    }
                    stop.store(true, Ordering::Relaxed);
                    break;
                }

                nonce += total_threads as u64;
            }
        });
    })
}

/// Spawns a GPU mining thread that launches CUDA kernels in batches.
///
/// Requires the compiled PTX embedded via `include_str!` from build.rs.
/// Falls back gracefully (logs a warning) if no CUDA device is found.
fn spawn_gpu_miner(
    challenge: [u8; 32],
    difficulty_bytes: [u8; 32],
    gpu_batch: u64,       // nonces per kernel launch (e.g. 1 << 22 ≈ 4 M)
    total_threads: u64,   // stride between GPU batches
    gpu_offset: u64,      // first nonce assigned to GPU block
    found_nonce: Arc<Mutex<Option<u64>>>,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        // ── Try to initialise CUDA ──────────────────────────────────────────
        #[cfg(feature = "gpu")]
        {
            use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

            let dev = match CudaDevice::new(0) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[GPU] CUDA init failed: {e}. GPU mining disabled.");
                    return;
                }
            };

            // Load PTX compiled by build.rs from hash_kernel.cu
            let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/hash_kernel.ptx"));
            dev.load_ptx(ptx_src.into(), "hash_kernel", &["mine_kernel"])
                .expect("Failed to load PTX");

            let f = dev
                .get_func("hash_kernel", "mine_kernel")
                .expect("kernel not found");

            // Allocate result buffer: [found: u32, nonce_lo: u32, nonce_hi: u32]
            let mut result_host = vec![0u32; 3];
            let mut result_dev: CudaSlice<u32> = dev.htod_sync_copy(&result_host).unwrap();

            // Flatten challenge + difficulty to u32 arrays for the kernel
            let mut chall_u32 = [0u32; 8];
            let mut diff_u32 = [0u32; 8];
            for i in 0..8 {
                chall_u32[i] = u32::from_be_bytes(challenge[i * 4..i * 4 + 4].try_into().unwrap());
                diff_u32[i] = u32::from_be_bytes(difficulty_bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
            let chall_dev: CudaSlice<u32> = dev.htod_sync_copy(&chall_u32).unwrap();
            let diff_dev: CudaSlice<u32> = dev.htod_sync_copy(&diff_u32).unwrap();

            let threads_per_block: u32 = 256;
            let blocks: u32 = ((gpu_batch as u32) + threads_per_block - 1) / threads_per_block;

            let mut base_nonce: u64 = gpu_offset;

            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                // Reset result flag
                result_host[0] = 0;
                dev.htod_sync_copy_into(&result_host, &mut result_dev).unwrap();

                unsafe {
                    f.clone().launch(
                        LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (threads_per_block, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &chall_dev,
                            &diff_dev,
                            base_nonce,
                            gpu_batch,
                            &mut result_dev,
                        ),
                    )
                    .expect("kernel launch failed");
                }

                dev.dtoh_sync_copy_into(&result_dev, &mut result_host).unwrap();

                // Update GPU hash counter
                GPU_HASH_COUNTER.fetch_add(gpu_batch, Ordering::Relaxed);

                if result_host[0] == 1 {
                    let nonce_lo = result_host[1] as u64;
                    let nonce_hi = result_host[2] as u64;
                    let nonce = nonce_lo | (nonce_hi << 32);

                    let mut found = found_nonce.lock().unwrap();
                    if found.is_none() {
                        *found = Some(nonce);
                        println!("\n[GPU FOUND] Nonce = {}", nonce);
                    }
                    stop.store(true, Ordering::Relaxed);
                    break;
                }

                base_nonce += total_threads;
            }
        }

        // ── Stub when compiled without "gpu" feature ───────────────────────
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!("[GPU] Compiled without 'gpu' feature — GPU mining disabled.");
            let _ = (challenge, difficulty_bytes, gpu_batch, total_threads, gpu_offset, found_nonce, stop);
        }
    })
}

async fn telegram(token: &str, chat_id: &str, text: &str) {
    if token.is_empty() || chat_id.is_empty() {
        return;
    }
    let url = format!("https://api.telegram.org/bot{}/sendMessage", token);
    let client = reqwest::Client::new();
    let _ = client
        .post(url)
        .json(&serde_json::json!({ "chat_id": chat_id, "text": text }))
        .send()
        .await;
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let private_key = env::var("PRIVATE_KEY")?;
    let infura = env::var("INFURA_URL")?;
    let bot_token = env::var("TELEGRAM_BOT_TOKEN").unwrap_or_default();
    let chat_id = env::var("TELEGRAM_CHAT_ID").unwrap_or_default();

    // CPU thread count (default: physical core count)
    let cpu_threads: usize = env::var("THREADS")
        .unwrap_or_else(|_| num_cpus::get().to_string())
        .parse()
        .unwrap();

    // GPU batch size: how many nonces per kernel launch (tune for your GPU)
    // RTX PRO 6000 S has 48 SM × 128 CUDA cores = 6144 cores; 1<<22 ≈ 4 M is safe
    let gpu_batch: u64 = env::var("GPU_BATCH")
        .unwrap_or_else(|_| (1u64 << 22).to_string())
        .parse()
        .unwrap();

    // Enable/disable GPU (default: true when compiled with "gpu" feature)
    let use_gpu: bool = env::var("USE_GPU")
        .unwrap_or_else(|_| "true".to_string())
        .parse()
        .unwrap_or(true);

    let provider = Provider::<Http>::try_from(infura)?;
    let wallet: LocalWallet = private_key
        .parse::<LocalWallet>()?
        .with_chain_id(1u64);
    let address = wallet.address();

    let client = Arc::new(SignerMiddleware::new(provider, wallet));
    let contract_addr = Address::from_str(CONTRACT)?;
    let contract = HashContract::new(contract_addr, client.clone());

    println!("========================================");
    println!("HASH-RUST ETH MAINNET MINER  [CPU+GPU]");
    println!("========================================");
    println!("Wallet      : {:?}", address);
    println!("CPU threads : {}", cpu_threads);
    println!("GPU batch   : {} hashes/launch", gpu_batch);
    println!("GPU enabled : {}", use_gpu);

    telegram(
        &bot_token,
        &chat_id,
        &format!(
            "🚀 HASH miner started (CPU+GPU)\nWallet: {:?}\nCPU Threads: {}\nGPU Batch: {}",
            address, cpu_threads, gpu_batch
        ),
    )
    .await;

    loop {
        CPU_HASH_COUNTER.store(0, Ordering::Relaxed);
        GPU_HASH_COUNTER.store(0, Ordering::Relaxed);

        // ── Fetch on-chain state ─────────────────────────────────────────────
        let state = contract.mining_state().call().await?;
        let difficulty = state.2;
        let epoch = state.5;
        let reward = state.1;

        println!("\nDifficulty : {}", difficulty);
        println!("Epoch      : {}", epoch);
        println!("Reward     : {} HASH", reward / U256::exp10(18));

        let challenge = contract.get_challenge(address).call().await?;
        let challenge_bytes: [u8; 32] = challenge.into();
        println!("Challenge  : 0x{}", hex::encode(challenge_bytes));

        // Serialize difficulty to bytes for GPU kernel
        let mut difficulty_bytes = [0u8; 32];
        difficulty.to_big_endian(&mut difficulty_bytes);

        // ── Shared state ─────────────────────────────────────────────────────
        let found_nonce: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));

        // ── Nonce space partitioning ─────────────────────────────────────────
        //
        // We assign nonce ranges so CPU and GPU never overlap:
        //   CPU: 0, 1, …, cpu_threads-1  (stepping by total_stride)
        //   GPU: cpu_threads, …           (stepping by total_stride in gpu_batch chunks)
        //
        // total_stride = cpu_threads + gpu_virtual_threads
        // We treat each GPU batch launch as 1 "virtual thread" for stride purposes.
        let gpu_virtual_threads: usize = if use_gpu { 1 } else { 0 };
        let total_stride = cpu_threads + gpu_virtual_threads;

        let start = Instant::now();
        println!("\nMining started (CPU + GPU simultaneously)…\n");

        // ── Live hashrate monitor ─────────────────────────────────────────────
        {
            let stats_start = Instant::now();
            std::thread::spawn(move || {
                let mut last_cpu = 0u64;
                let mut last_gpu = 0u64;
                loop {
                    std::thread::sleep(Duration::from_secs(1));

                    let cpu_now = CPU_HASH_COUNTER.load(Ordering::Relaxed);
                    let gpu_now = GPU_HASH_COUNTER.load(Ordering::Relaxed);

                    let cpu_rate = cpu_now - last_cpu;
                    let gpu_rate = gpu_now - last_gpu;

                    last_cpu = cpu_now;
                    last_gpu = gpu_now;

                    let total_rate = cpu_rate + gpu_rate;
                    let elapsed = stats_start.elapsed().as_secs();

                    println!(
                        "[STATS] CPU {:.2} MH/s | GPU {:.2} MH/s | Total {:.2} MH/s | Runtime: {}s",
                        cpu_rate as f64 / 1_000_000.0,
                        gpu_rate as f64 / 1_000_000.0,
                        total_rate as f64 / 1_000_000.0,
                        elapsed
                    );
                }
            });
        }

        // ── GPU miner thread ──────────────────────────────────────────────────
        let gpu_handle = if use_gpu {
            let fn_arc = found_nonce.clone();
            let stop_arc = stop.clone();
            Some(spawn_gpu_miner(
                challenge_bytes,
                difficulty_bytes,
                gpu_batch,
                total_stride as u64,
                cpu_threads as u64, // GPU starts right after CPU's slice
                fn_arc,
                stop_arc,
            ))
        } else {
            None
        };

        // ── CPU miner thread ──────────────────────────────────────────────────
        let cpu_handle = {
            let fn_arc = found_nonce.clone();
            let stop_arc = stop.clone();
            spawn_cpu_miner(
                challenge_bytes,
                difficulty_bytes,
                cpu_threads,
                total_stride,
                0, // CPU starts at nonce 0
                fn_arc,
                stop_arc,
            )
        };

        // ── Wait for either to find a nonce ───────────────────────────────────
        cpu_handle.join().expect("CPU thread panicked");
        if let Some(h) = gpu_handle {
            h.join().expect("GPU thread panicked");
        }

        let nonce = found_nonce.lock().unwrap().unwrap();
        let elapsed = start.elapsed().as_secs_f64();

        println!("\nElapsed : {:.2} sec", elapsed);
        println!("\nSubmitting tx…");

        // ── Dynamic gas ───────────────────────────────────────────────────────
        let block = client
            .get_block(BlockNumber::Latest)
            .await?
            .expect("latest block not found");

        let base_fee = block
            .base_fee_per_gas
            .unwrap_or(U256::from(163_000_000u64));

        let priority_fee = if base_fee < U256::from(1_000_000_000u64) {
            U256::from(100_000_000u64)
        } else {
            U256::from(2_000_000_000u64)
        };

        let gas_price =
            (base_fee * U256::from(120u64) / U256::from(100u64)) + priority_fee;

        println!("Base Fee     : {} gwei", base_fee / U256::exp10(9));
        println!("Priority Fee : {} gwei", priority_fee / U256::exp10(9));
        println!("Gas Price    : {} gwei", gas_price / U256::exp10(9));

        let estimated_cost = gas_price * U256::from(300_000u64);
        println!(
            "Estimated Gas Cost : {:.8} ETH",
            estimated_cost.as_u128() as f64 / 1e18
        );

        let call = contract
            .mine(U256::from(nonce))
            .gas(300_000u64)
            .gas_price(gas_price);
        let pending_tx = call.send().await?;

        println!("\nTX Sent : {:?}", pending_tx.tx_hash());

        match pending_tx.await? {
            Some(receipt) => {
                println!("SUCCESS : {:?}", receipt.transaction_hash);
                telegram(
                    &bot_token,
                    &chat_id,
                    &format!(
                        "✅ HASH mined\n\nTX: {:?}\nNonce: {}\nGas: {:.8} ETH",
                        receipt.transaction_hash,
                        nonce,
                        estimated_cost.as_u128() as f64 / 1e18
                    ),
                )
                .await;
            }
            None => {
                println!("TX dropped");
                telegram(&bot_token, &chat_id, "⚠️ TX dropped").await;
            }
        }

        println!("\nRefreshing challenge…\n");
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}