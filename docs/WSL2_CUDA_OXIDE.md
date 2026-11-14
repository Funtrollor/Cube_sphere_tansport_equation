# Windows 11 + WSL2 + RTX 5070 Ti (`cuda-oxide`) GPU 工具鏈設定與驗證指南

本指南說明如何在 **Windows 11 + WSL2 (Ubuntu 24.04) + NVIDIA GeForce RTX 5070 Ti (`sm_120`, Consumer Blackwell)** 環境中，安裝 [`NVIDIA/cuda-rust`](https://github.com/NVIDIA/cuda-rust)（`cuda-oxide`）工具鏈，並編譯與驗證本專案的純 Rust GPU 後端 [`crates/cubesphere-backend-cuda`](../crates/cubesphere-backend-cuda/src/lib.rs)。

> **原生 Windows vs WSL2 說明**：
> - 上游 `NVIDIA/cuda-rust` 目前**已正式支援原生 Windows (`x86_64-pc-windows-msvc`)**！若希望直接在 Developer PowerShell for VS 2022 編譯執行、不經過 WSL2 虛擬層，請優先參閱 [`docs/WINDOWS_CUDA_OXIDE.md`](WINDOWS_CUDA_OXIDE.md)。
> - 若偏好透過 Ubuntu 24.04 `apt` 管理 `llvm-21`/`libclang-21`，或未來 Phase 3 需測試依賴 Linux `procfs` 的 `.so` PyO3 擴充模組，可繼續使用本篇 **WSL2 Ubuntu 24.04** 方案。

---

## 1. 事前檢查：Windows 主機端 NVIDIA 驅動

WSL2 會透過 `/usr/lib/wsl/lib/libcuda.so` 直接橋接 Windows Host 的 NVIDIA 驅動程式：

1. 在 **Windows 11** 下載並安裝 **R580 或更新版** 的 NVIDIA Game Ready / Studio Driver（搭配 CUDA Toolkit 13.x 使用）。
2. 在 Windows PowerShell 更新並啟動 WSL2 (Ubuntu 24.04)：
   ```powershell
   wsl --update
   wsl -d Ubuntu-24.04
   ```
3. 進入 WSL2 後先測試 GPU 透傳是否正常：
   ```bash
   nvidia-smi
   ```
   確認能正確看到 **`NVIDIA GeForce RTX 5070 Ti`** 與驅動版本（例如 `580.xx` 以上）。

> **重要警告（絕對不要踩的坑）：**
> **切勿在 WSL2 Ubuntu 內執行 `sudo apt install nvidia-driver-*`！**
> Linux 套件版的顯示驅動會覆蓋 WSL2 專用的半虛擬化驅動橋接，導致 `cuda_bindings` 無法載入 `libcuda.so`。在 WSL2 內只安裝 **CUDA Toolkit（使用者空間編譯工具與標頭檔）** 即可。

4. **I/O 效能建議**：請將專案 `git clone` 至 WSL2 原生 Linux 檔案系統（如 `~/workspace/cubesphere`），**避免放在 `/mnt/c/...`**，否則 Cargo + `cuda-oxide` 編譯速度與 `cargo oxide` 快取 I/O 會慢數倍。

---

## 2. 在 WSL2 (Ubuntu 24.04) 安裝編譯環境

### 2.1 安裝 CUDA Toolkit 13.0+ (WSL-Ubuntu)

依 [NVIDIA CUDA Downloads (WSL-Ubuntu x86_64)](https://developer.nvidia.com/cuda-downloads) 安裝 **CUDA Toolkit 13.0+**（必須包含 `nvcc`、`cuda.h`、`curand.h`、`libnvvm` 與 `nvJitLink`）：

```bash
# 確認 Toolkit 已就緒
ls /usr/local/cuda/include/cuda.h /usr/local/cuda/include/curand.h
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
```

> **提示**：若同時安裝多套 CUDA，`cargo-oxide` 會依序尋找 `/usr/local/cuda-13.3` → `/usr/local/cuda-13.2` → `/usr/local/cuda-13` → `/usr/local/cuda`。將 `export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0`（或你的實際版本路徑）寫入 `~/.bashrc` 最為明確。

### 2.2 安裝 LLVM 21+ 與 Clang 21 開發標頭檔

`cuda-bindings` 在主機端會透過 `bindgen` 解析 `cuda.h`，需要 `libclang` 共享函式庫與 Clang 內建標頭檔（如 `stddef.h`）：

```bash
sudo apt update
sudo apt install -y \
  llvm-21 \
  libclang-21-dev \
  libclang-cpp21-dev \
  libclang-common-21-dev \
  clang \
  libclang-dev \
  pkg-config \
  build-essential
```

若系統只有 `clang-21` 而缺乏無版本號的 `clang` 指令，使用 `update-alternatives` 補上軟連結：

```bash
sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-21   100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100
```

驗證 `llvm-21` 已含 NVPTX 後端：

```bash
llc-21 --version | grep nvptx
# 應看到 nvptx / nvptx64
```

### 2.3 安裝 Rust 1.99.1 工具鏈與 `cargo-oxide`

本專案在 [`crates/cubesphere-backend-cuda/rust-toolchain.toml`](../crates/cubesphere-backend-cuda/rust-toolchain.toml) 已鎖定與 `cuda-oxide` 相容的 Rust 版本 `1.99.1`，並在 [`.cargo/config.toml`](../crates/cubesphere-backend-cuda/.cargo/config.toml) 自動設定 `RUSTC_BOOTSTRAP = "1"`：

```bash
rustup toolchain install 1.99.1
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain 1.99.1

# 安裝 cargo-oxide CLI（首次執行會編譯並快取 rustc_codegen_cuda 至 ~/.cargo/cuda-oxide/）
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
```

---

## 3. 環境體檢與常見錯誤排查 (`cargo oxide doctor`)

進入獨立隔離的 CUDA 後端目錄執行健康檢查：

```bash
cd ~/workspace/cubesphere/crates/cubesphere-backend-cuda
cargo oxide doctor
```

常見錯誤對照表：

| 症狀 / 錯誤訊息 | 原因 | 解法 |
| --- | --- | --- |
| `'stddef.h' file not found` | 缺少 Clang resource dir 或 `clang` 無版本號連結 | 安裝 `libclang-common-21-dev clang libclang-dev` 並設定 `update-alternatives` |
| `No working llc found` | 未安裝 `llvm-tools` 元件或 `llvm-21` 不在 `PATH` | 執行 `rustup component add llvm-tools --toolchain 1.99.1` 或設定 `CUDA_OXIDE_LLC=/usr/bin/llc-21` |
| `cuda.h not found` | `CUDA_TOOLKIT_PATH` 未指向安裝根目錄 | `export CUDA_TOOLKIT_PATH=/usr/local/cuda` |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)` | CUDA Toolkit（如 13.3 的 PTX 9.3）產生的 PTX 版本高於 Windows 驅動支援範圍 | 更新 Windows 11 NVIDIA 驅動至最新版本，或切換至較低小版的 CUDA Toolkit（如 `CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0`） |
| `CUDA_ERROR_NOT_INITIALIZED` | WSL2 內找不到 `libcuda.so` 或誤裝 Linux GPU 驅動 | 確認 Windows PowerShell 已 `wsl --update`，且 `/usr/lib/wsl/lib/libcuda.so` 存在、未安裝 `nvidia-driver-*` |

---

## 4. 編譯與執行 GPU vs CPU 對拍驗證 (`gpu_parity`)

先跑純 CPU Workspace 測試確認數學與幾何核心在本地全數通過：

```bash
cd ~/workspace/cubesphere
cargo test --release --workspace
```

接著編譯並執行 GPU 後端驗證程式 [`gpu_parity.rs`](../crates/cubesphere-backend-cuda/src/bin/gpu_parity.rs)：

```bash
cd crates/cubesphere-backend-cuda
cargo oxide build
cargo oxide run --bin gpu_parity
```

`gpu_parity` 會透過完全相同的 [`ComputeBackend<T>`](../crates/cubesphere-solver/src/backend.rs) 與 `run_lsrk45_simulation` 介面，自動執行以下四項驗收：

1. **Solid-Body Rotation (`f64`, `K=4, p=4`, 1 圈)**：驗證 GPU 與 CPU Rayon 逐點最大誤差 $\max |\phi_{\text{GPU}} - \phi_{\text{CPU}}| < 10^{-13}$，且全域質量誤差 $< 10^{-14}$。
2. **Nair & Lauritzen (2010) Case 4 變形流 (`f64`, `K=4, p=5`, 1 週期)**：驗證動態時變風場的 **無超越函數三角加倍角 GPU 核心**（於主機只計算 $c_{2\omega t}, s_{2\omega t}, k_t$ 三個純量，裝置端零 `sin`/`cos` 呼叫，完全避免 `libdevice` 版本或精度漂移）與 CPU 誤差 $< 10^{-12}$，且質量守恆 $< 10^{-14}$。
3. **單精度對拍 (`f32`, `K=8, p=4`)**：驗證 `CudaBackend<f32>` 與 `CpuBackend<f32>` 誤差 $< 2 \times 10^{-5}$。
4. **高解析度吞吐量基準測試 (`f64` 與 `f32`，`K=16/32, p=4`)**：輸出 `MDOF-stages/s` 與相對於 CPU Rayon 的實際加速比。

---

## 5. PTX 檢視與 RTX 5070 Ti (`sm_120`) 效能調校指南

### 5.1 檢查產生的 PTX 與暫存器 / Shared Memory 用量

```bash
cd crates/cubesphere-backend-cuda
cargo oxide inspect --release --bin gpu_parity
```

取得建置產生的 `.ptx` 後，可使用 `ptxas` 針對 RTX 5070 Ti（Compute Capability `12.0` / `sm_120`）檢查暫存器壓力（Register Spills / Occupancy）：

```bash
ptxas -v -arch=sm_120 target/cuda-oxide/<artifact>.ptx -o /tmp/test_sm120.cubin
rm -f /tmp/test_sm120.cubin
```

### 5.2 GPU 核心設計與 RTX 5070 Ti 架構特性

1. **單執行緒單自由度（1-thread-per-DOF）融合核心，徹底無資料競爭**：
   `split_form_and_sat_{f64,f32}` 將體積項分裂散度（3 項）與 4 條邊的迎風 SAT 邊界懲罰融合成單次 Kernel 啟動；每個執行緒僅寫入自身對應的 `*rhs_slot`，相鄰元素／立方體角點僅透過只讀切片 `&[T]` 與 `u64` 位元封裝鄰接表（`ld.global.u64`）讀取，不需要原子操作（`atomicAdd`）也不受多項式階數 $p$ 的 Shared Memory 容量限制。
2. **無 `sqrt_g` SAT 修正 + 零動態主機複製**：
   幾何因子 `sqrt_g`、微分矩陣 `diff_mat`、鄰接表 `neighbors_dev` 與風場緩衝區全部常駐 VRAM，LSRK45 時間步迴圈中不進行任何 Host ↔ Device 資料拷貝。
3. **RTX 5070 Ti (`sm_120`) FP32 vs FP64 特性**：
   - Consumer Blackwell (`sm_120` / `sm_120a`) 專攻圖形與 AI/FP32，FP64 與 FP32 理論計算單元比例極懸殊（約 `1 : 64`），且沒有 Datacenter Blackwell (`sm_100a`) 才有的 `tcgen_05` TMEM 指令（本 crate 使用純 SIMT FP32/FP64 算子，可原生無障礙編譯執行）。
   - 在記憶體頻寬受限的高階數譜元素運算（大 $K$、中等 $p=4..8$）中，`f32` 不僅 VRAM 流量減半、L2 Cache 命中率倍增，且可完全釋放 `sm_120` 龐大的 FP32 ALU 管線。建議用 `gpu_parity` 實測 `f64`（嚴格守恆）與 `f32`（大網格高吞吐）的差距，挑選最符合任務需求的精度。
4. **記憶體除錯（Compute Sanitizer）**：
   若未來新增自訂邊界或方程核心，可直接透過 `cuda-oxide` 整合的 NVIDIA Compute Sanitizer 檢查越界或同步問題：
   ```bash
   cargo oxide sanitize --bin gpu_parity
   ```
