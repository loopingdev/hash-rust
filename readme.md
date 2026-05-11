# Rust HASH256 Ethereum Mainnet Miner

* Ethereum Mainnet
* Infura RPC
* HASH256 mining contract

Contract:

urlHASH256 Contract on Etherscan[https://etherscan.io/address/0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc#code](https://etherscan.io/address/0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc#code)

The miner:

* uses all CPU cores
* multi-threaded mining
* optimized Keccak hashing
* auto-submits transactions
* supports Infura RPC

---

# 1. Recommended VPS

Recommended:

```bash
8 vCPU
16GB RAM
Ubuntu 22.04
```

Better:

```bash
16–64 dedicated cores
```

---

# 2. Install Rust

## Install dependencies

```bash
sudo apt update
sudo apt install build-essential curl pkg-config libssl-dev -y
```

---

## Install Rust

```bash
curl https://sh.rustup.rs -sSf | sh
```

Choose:

```text
1. Proceed with installation
```

Reload shell:

```bash
source $HOME/.cargo/env
```

Verify:

```bash
rustc --version
cargo --version
```

---

# 3. Create Project

```bash
cargo new hash-rust
cd hash-rust
```

---

# 4. Replace Cargo.toml

Open:

```bash
nano Cargo.toml
```

Replace with:

```toml
[package]
name = "hash-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
ethers = "2"
tokio = { version = "1", features = ["full"] }
hex = "0.4"
rayon = "1.10"
num_cpus = "1.16"
sha3 = "0.10"
anyhow = "1"
serde_json = "1"
```

Save.

---

# 5. Create .env

```bash
nano .env
```

Put:

```env
PRIVATE_KEY=YOUR_PRIVATE_KEY
INFURA_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
```

IMPORTANT:

Use dedicated mining wallet only.

Never use your main wallet.

---


# 6. Build Miner

```bash
cargo build --release
```

Compiled binary:

```bash
./target/release/hash-rust
```

---

# 7. Run Miner

```bash
./target/release/hash-rust
```

---

# Example Output

```text
========================================
HASH256 RUST ETH MAINNET MINER
========================================
Wallet : 0x123...
Difficulty : 9283749823749823
Epoch      : 8291
Reward     : 100 HASH
Challenge  : 0xabcd...
Threads    : 16
Mining started...

[FOUND] Nonce = 291928122

Submitting tx...
TX Sent : 0xabc123...
SUCCESS : 0xabc123...
```

---

# 8. Run In Background

Install tmux:

```bash
sudo apt install tmux -y
```

Start:

```bash
tmux
```

Run miner:

```bash
./target/release/hash-rust
```

Detach:

```text
CTRL+B then D
```

Reattach:

```bash
tmux attach
```

---
