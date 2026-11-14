# cs-backend-cuda (`NVIDIA/cuda-rust` / `cuda-oxide`)

NVIDIA CUDA SIMT GPU compute backend implementing `cs_solver::ComputeBackend<T>` in pure Rust via [`NVIDIA/cuda-rust`](https://github.com/NVIDIA/cuda-rust).

This crate is intentionally isolated (`[workspace]`, `edition = "2024"`, `RUSTC_BOOTSTRAP = "1"`, `rust-toolchain 1.99.1`) and excluded from the root CPU workspace so CPU CI and headless environments build cleanly without a CUDA Toolkit or NVIDIA driver.

## Quick Run (Native Windows 11 `x86_64-pc-windows-msvc`)

Inside **Developer PowerShell for VS 2022 (x64)** after completing [../../docs/WINDOWS_CUDA_OXIDE.md](../../docs/WINDOWS_CUDA_OXIDE.md):

```powershell
cd crates\cs-backend-cuda
cargo oxide doctor
cargo oxide setup
cargo oxide build
cargo oxide run --bin gpu_parity
```

## Quick Run (WSL2 Ubuntu 24.04 / Linux)

```bash
cd crates/cs-backend-cuda
cargo oxide doctor
cargo oxide setup
cargo oxide build
cargo oxide run --bin gpu_parity
```

See:
- **Native Windows 11 + RTX 5070 Ti (`sm_120`) Guide**: [../../docs/WINDOWS_CUDA_OXIDE.md](../../docs/WINDOWS_CUDA_OXIDE.md)
- **WSL2 Ubuntu 24.04 + RTX 5070 Ti (`sm_120`) Guide**: [../../docs/WSL2_CUDA_OXIDE.md](../../docs/WSL2_CUDA_OXIDE.md)
