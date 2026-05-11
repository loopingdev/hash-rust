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
    atomic::{AtomicU64, Ordering},
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

const CONTRACT: &str =
    "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc";

static HASH_COUNTER: AtomicU64 =
    AtomicU64::new(0);

async fn telegram(
    token: &str,
    chat_id: &str,
    text: &str,
) {

    if token.is_empty() || chat_id.is_empty() {
        return;
    }

    let url = format!(
        "https://api.telegram.org/bot{}/sendMessage",
        token
    );

    let client = reqwest::Client::new();

    let _ = client
        .post(url)
        .json(&serde_json::json!({
            "chat_id": chat_id,
            "text": text
        }))
        .send()
        .await;
}

#[tokio::main]
async fn main() -> Result<()> {

    dotenv().ok();

    let private_key =
        env::var("PRIVATE_KEY")?;

    let infura =
        env::var("INFURA_URL")?;

    let bot_token =
        env::var("TELEGRAM_BOT_TOKEN")
            .unwrap_or_default();

    let chat_id =
        env::var("TELEGRAM_CHAT_ID")
            .unwrap_or_default();

    let threads: usize =
        env::var("THREADS")
            .unwrap_or("8".to_string())
            .parse()
            .unwrap();

    let provider =
        Provider::<Http>::try_from(infura)?;

    let wallet: LocalWallet =
        private_key
            .parse::<LocalWallet>()?
            .with_chain_id(1u64);

    let address = wallet.address();

    let client = Arc::new(
        SignerMiddleware::new(
            provider,
            wallet
        )
    );

    let contract_addr =
        Address::from_str(CONTRACT)?;

    let contract =
        HashContract::new(
            contract_addr,
            client.clone(),
        );

    println!("========================================");
    println!("HASH-RUST ETH MAINNET MINER");
    println!("========================================");

    println!("Wallet  : {:?}", address);
    println!("Threads : {}", threads);

    telegram(
        &bot_token,
        &chat_id,
        &format!(
            "🚀 HASH miner started\nWallet: {:?}\nThreads: {}",
            address,
            threads
        )
    ).await;

    loop {

        HASH_COUNTER.store(
            0,
            Ordering::Relaxed
        );

        let state = contract
            .mining_state()
            .call()
            .await?;

        let difficulty = state.2;
        let epoch = state.5;
        let reward = state.1;

        println!(
            "Difficulty : {}",
            difficulty
        );

        println!(
            "Epoch      : {}",
            epoch
        );

        println!(
            "Reward     : {} HASH",
            reward / U256::exp10(18)
        );

        let challenge = contract
            .get_challenge(address)
            .call()
            .await?;

        println!(
            "Challenge  : 0x{}",
            hex::encode(challenge)
        );

        let found_nonce =
            Arc::new(
                Mutex::new(None::<u64>)
            );

        let start =
            Instant::now();

        println!("\nMining started...\n");

        // =====================================
        // LIVE HASHRATE MONITOR
        // =====================================

        let stats_start =
            Instant::now();

        std::thread::spawn(move || {

            let mut last = 0u64;

            loop {

                std::thread::sleep(
                    Duration::from_secs(1)
                );

                let current =
                    HASH_COUNTER.load(
                        Ordering::Relaxed
                    );

                let rate =
                    current - last;

                last = current;

                let mh =
                    rate as f64 / 1_000_000.0;

                let elapsed =
                    stats_start
                        .elapsed()
                        .as_secs();

                println!(
                    "[STATS] {:.2} MH/s | Total Hashes: {} | Runtime: {} sec",
                    mh,
                    current,
                    elapsed
                );
            }
        });

        // =====================================
        // MINING THREADS
        // =====================================

        (0..threads)
            .into_par_iter()
            .for_each(|thread_id| {

            let mut nonce: u64 =
                thread_id as u64;

            loop {

                {
                    if found_nonce
                        .lock()
                        .unwrap()
                        .is_some()
                    {
                        break;
                    }
                }

                let encoded = encode(&[
                    Token::FixedBytes(
                        challenge.to_vec()
                    ),
                    Token::Uint(
                        U256::from(nonce)
                    ),
                ]);

                let mut hasher =
                    Keccak256::new();

                hasher.update(encoded);

                let result =
                    hasher.finalize();

                let hash_u256 =
                    U256::from_big_endian(
                        &result
                    );

                HASH_COUNTER.fetch_add(
                    1,
                    Ordering::Relaxed
                );

                if hash_u256 < difficulty {

                    let mut found =
                        found_nonce
                            .lock()
                            .unwrap();

                    if found.is_none() {

                        *found =
                            Some(nonce);

                        println!(
                            "\n[FOUND] Nonce = {}",
                            nonce
                        );
                    }

                    break;
                }

                nonce +=
                    threads as u64;
            }
        });

        let nonce = {

            let guard =
                found_nonce
                    .lock()
                    .unwrap();

            guard.unwrap()
        };

        let elapsed =
            start
                .elapsed()
                .as_secs_f64();

        println!(
            "\nElapsed : {:.2} sec",
            elapsed
        );

        println!("\nSubmitting tx...");

        // =====================================
        // DYNAMIC GAS
        // =====================================

        let block = client
            .get_block(
                BlockNumber::Latest
            )
            .await?
            .expect(
                "latest block not found"
            );

        let base_fee = block
            .base_fee_per_gas
            .unwrap_or(
                U256::from(
                    163_000_000u64
                )
            );

        let priority_fee =
            if base_fee <
                U256::from(
                    1_000_000_000u64
                )
            {
                U256::from(
                    100_000_000u64
                )
            } else {
                U256::from(
                    2_000_000_000u64
                )
            };

        let gas_price =
            (base_fee
                * U256::from(120u64)
                / U256::from(100u64))
            + priority_fee;

        println!(
            "Base Fee     : {} gwei",
            base_fee
                / U256::exp10(9)
        );

        println!(
            "Priority Fee : {} gwei",
            priority_fee
                / U256::exp10(9)
        );

        println!(
            "Gas Price    : {} gwei",
            gas_price
                / U256::exp10(9)
        );

        let estimated_cost =
            gas_price
                * U256::from(300000u64);

        println!(
            "Estimated Gas Cost : {:.8} ETH",
            estimated_cost
                .as_u128() as f64
                / 1e18
        );

        let call = contract
            .mine(U256::from(nonce))
            .gas(300000u64)
            .gas_price(gas_price);

        let pending_tx = call
            .send()
            .await?;

        println!(
            "\nTX Sent : {:?}",
            pending_tx.tx_hash()
        );

        match pending_tx.await? {

            Some(receipt) => {

                println!(
                    "SUCCESS : {:?}",
                    receipt.transaction_hash
                );

                telegram(
                    &bot_token,
                    &chat_id,
                    &format!(
                        "✅ HASH mined\n\nTX: {:?}\nNonce: {}\nGas: {:.8} ETH",
                        receipt.transaction_hash,
                        nonce,
                        estimated_cost.as_u128() as f64 / 1e18
                    )
                ).await;
            }

            None => {

                println!(
                    "TX dropped"
                );

                telegram(
                    &bot_token,
                    &chat_id,
                    "⚠️ TX dropped"
                ).await;
            }
        }

        println!(
            "\nRefreshing challenge...\n"
        );

        tokio::time::sleep(
            Duration::from_secs(2)
        ).await;
    }
}