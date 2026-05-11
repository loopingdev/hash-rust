// build.rs
// Compiles hash_kernel.cu → hash_kernel.ptx at build time using nvcc.
// The PTX is placed in OUT_DIR and embedded in main.rs via include_str!.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile the CUDA kernel when the "gpu" feature is active.
    if env::var("CARGO_FEATURE_GPU").is_err() {
        println!("cargo:warning=gpu feature not set — skipping CUDA kernel compilation");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_out = out_dir.join("hash_kernel.ptx");

    // CARGO_MANIFEST_DIR is always the project root (where Cargo.toml lives).
    // nvcc runs from an arbitrary working directory, so we must use an absolute path.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_src = manifest_dir.join("hash_kernel.cu");

    println!("cargo:rerun-if-changed={}", kernel_src.display());

    // SM target: override with CUDA_ARCH env var if needed.
    // RTX PRO 6000 S (Blackwell) → sm_100  (requires CUDA 12.8+)
    // Ada Lovelace  (RTX 4xxx)   → sm_89
    // Ampere        (RTX 3xxx)   → sm_86
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());

    let status = Command::new("nvcc")
        .args([
            "-ptx",
            "-arch", &arch,
            "-O3",
            "-o", ptx_out.to_str().unwrap(),
            kernel_src.to_str().unwrap(),   // <- absolute path, not relative
        ])
        .status()
        .expect(
            "nvcc not found. Install CUDA toolkit or disable the gpu feature:\n\
             cargo build --release --no-default-features"
        );

    if !status.success() {
        panic!("nvcc failed to compile {}", kernel_src.display());
    }

    println!("cargo:warning=CUDA kernel compiled → {}", ptx_out.display());
}