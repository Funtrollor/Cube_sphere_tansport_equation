# Contributing to cubesphere

Thanks for your interest! Bug reports, verification cases, and pull requests are all welcome.
Issues and discussions can be written in English or Traditional Chinese.

## Reporting a bug

Please include:
- the exact `cubesphere` command (or a minimal Rust snippet) that reproduces the problem,
- the mesh and method settings (`--k`, `--order`, `--projection`, `--flux`, `--precision`),
- what you expected and what you got (error norms, mass error, panic message),
- your OS and `rustc --version`.

## Development setup

You need stable Rust 1.87 or newer. The CPU workspace needs no GPU or CUDA toolkit.

```bash
cargo build --workspace
cargo test --workspace --release   # release mode: the convergence tests are much faster
```

The CUDA backend in `crates/cubesphere-backend-cuda/` is a separate workspace with its own toolchain.
See [docs/WINDOWS_CUDA_OXIDE.md](docs/WINDOWS_CUDA_OXIDE.md) or [docs/WSL2_CUDA_OXIDE.md](docs/WSL2_CUDA_OXIDE.md).

## Before opening a pull request

CI runs these checks, so please run them locally first:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
cargo doc --workspace --no-deps
```

Guidelines:
- **Numerical changes need evidence.** If you touch the discretization, the fluxes, or the time integration, add or update a test in `crates/cubesphere-backend-cpu/tests/solver_verification.rs` (convergence order, mass conservation, or energy behaviour) and put the before/after error norms in the PR description.
- **Public items need docs.** Library crates use `#![warn(missing_docs)]`, so CI fails on undocumented public items.
- **Keep the CPU and CUDA backends in sync.** A change to the RHS must be mirrored in `cubesphere-backend-cuda` and checked with `gpu_parity` if you have an NVIDIA GPU. If you don't, say so in the PR and a maintainer will run it.
- Update `CHANGELOG.md` under `[Unreleased]`.

## Plotting scripts

`scripts/` needs only `numpy`, `matplotlib` and `pillow`. Keep the scripts dependency-free beyond these, and fail loudly on bad input rather than skipping it.
