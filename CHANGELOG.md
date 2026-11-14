# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may contain breaking API changes.

## [Unreleased]

### Added
- `NumericalFlux` (upwind / central / blended α) shared by the CPU and CUDA backends; every α conserves mass to machine precision.
- Discrete energy diagnostic `E = ∫ φ² dS` (`compute_energy`) and `energy_initial` / `energy_final` / `energy_rel_change` in `ErrorMetrics`.
- `--flux` CLI option and an `energy_history` array in the JSON report.
- `EquidistantProjection::analytical_jacobian`.
- Verification tests for the semi-discrete energy structure, energy evolution per flux, and equidistant geometry.
- Plotting scripts in `scripts/` (`plot_convergence.py`, `plot_grids.py`, `pvd_to_gif.py`) and README figures in `docs/images/`.
- MIT `LICENSE` file, CI workflow (fmt, clippy, tests on Linux/Windows/macOS, docs, MSRV, scripts), `CONTRIBUTING.md`, `CITATION.cff`.
- API documentation for every public item, with `#![warn(missing_docs)]` and a runnable crate-level example per library crate.
- crates.io metadata (`repository`, `readme`, `keywords`, `categories`, `rust-version = "1.87"`).

### Changed
- **Breaking:** crates renamed from `cs-*` to `cubesphere-*` (`cubesphere-core`, `cubesphere-equations`, `cubesphere-solver`, `cubesphere-backend-cpu`, `cubesphere-io`, `cubesphere-cli`, `cubesphere-backend-cuda`). The CLI binary is now `cubesphere`.
- Repository renamed to `Funtrollor/cubesphere`.
- The original prototype files (`P2_new.ipynb`, `NPDE.pdf`, `spherical_advection_clean.gif`) moved to `archive/prototype/`.
- `Cargo.lock` is now committed for reproducible builds of the CLI and the CUDA backend.
- The README is now in English; the Traditional Chinese version is `README.zh-TW.md`.

### Fixed
- Stale L∞ values (p ≥ 12) in the README p-refinement table.

## [0.2.0] - 2026-11-13

### Added
- First modular Rust workspace, refactored from the `P2_new.ipynb` prototype: LGL spectral elements on a K×K-per-face cubed sphere, equiangular and equidistant projections, split-form divergence with conservative SBP-SAT coupling, and Carpenter–Kennedy LSRK45 with Neumaier-compensated time accumulation.
- Solid-body rotation and Nair & Lauritzen (2010) Case 4 deformational flow, plus Gaussian, cosine-bell, twin-Gaussian and constant initial fields.
- Rayon CPU backend, generic `f64`/`f32` precision, VTK/CSV/JSON output, and the `info` / `run` / `converge` / `bench` CLI.
- Experimental `cuda-oxide` GPU backend with a GPU–CPU parity checker.

[Unreleased]: https://github.com/Funtrollor/cubesphere/compare/f99de7f...HEAD
[0.2.0]: https://github.com/Funtrollor/cubesphere/commit/f99de7f
