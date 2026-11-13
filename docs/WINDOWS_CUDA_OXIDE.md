# 原生 Windows 11 + RTX 5070 Ti (`cuda-oxide`) GPU 工具鏈設定與驗證指南

本指南說明如何在 **原生 Windows 11 (`x86_64-pc-windows-msvc`) + NVIDIA GeForce RTX 5070 Ti (`sm_120`, Consumer Blackwell)** 環境中，安裝 [`NVIDIA/cuda-rust`](https://github.com/NVIDIA/cuda-rust)（`cuda-oxide`）工具鏈，並直接於 Developer PowerShell 編譯與驗證本專案的純 Rust GPU 後端 [`crates/cs-backend-cuda`](../crates/cs-backend-cuda/src/lib.rs)。

> **原生 Windows vs WSL2 選擇建議**：
> - 上游 `NVIDIA/cuda-rust` 已正式支援原生 `x86_64-pc-windows-msvc`（見官方 [`windows.md`](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/cuda-oxide-book/getting-started/windows.md) 與 MSVC 修補如 [#1473](https://github.com/NVIDIA/cuda-rust/commit/d9430b0)），無需安裝 WSL2 即可直接編譯、執行與除錯（含 WDDM Compute Sanitizer）。
> - 若偏好 Linux（`apt` 一鍵管理 `llvm-21` 與未來 Phase 3 PyO3 `.so` 共享函式庫的 `/proc/self/maps` anchor 自動載入），亦可使用 [`docs/WSL2_CUDA_OXIDE.md`](WSL2_CUDA_OXIDE.md)。

---

## 1. 系統與工具鏈需求

1. **NVIDIA 驅動程式（Windows 11）**
   - 安裝與 **CUDA Toolkit 13.x 相容**的 NVIDIA Game Ready / Studio Driver（**R580 或更新版本**）。
   - 於一般 PowerShell 執行 `nvidia-smi`，確認可辨識 **`NVIDIA GeForce RTX 5070 Ti`**。

2. **CUDA Toolkit 13.0+ (Windows x86_64)**
   - 下載並安裝 [CUDA Toolkit 13.0+ for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)（建議 `13.2` 或 `13.3`；須包含 `nvcc`、`libNVVM`、`nvJitLink` 與 `cuda.h` / `curand.h` 標頭檔）。
   - `cuda-bindings` 預設會偵測環境變數 `CUDA_TOOLKIT_PATH`、`CUDA_HOME`，或探測 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` 與 `v13.2`。

3. **Visual Studio 2022（或更新版本）+ C++ Clang 工具**
   - 於 Visual Studio Installer 勾選 **使用 C++ 的桌面開發（Desktop development with C++）**，並確認已安裝：
     - **MSVC x64/x86 建置工具（Build Tools）**
     - **Windows 10/11 SDK**
     - **適用于 Windows 的 C++ Clang 工具**（需具備 `libclang.dll` 與 `lld-link.exe`；亦可另外安裝官方 LLVM Windows 發行版至 `C:\Program Files\LLVM`）。

4. **`vcpkg` 與 `libffi:x64-windows`**
   - 若尚未安裝 `vcpkg`，可在工作目錄外（例如 `C:\vcpkg`）執行：
     ```powershell
     git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
     C:\vcpkg\bootstrap-vcpkg.bat
     ```
   - 安裝 64 位元 `libffi`：
     ```powershell
     $env:VCPKG_ROOT = "C:\vcpkg"
     & "$env:VCPKG_ROOT\vcpkg.exe" install libffi:x64-windows
     ```
   - `cargo-oxide` 會依序從 `LIBFFI_LIB_DIR`、`LIB` 與 `VCPKG_ROOT\installed\x64-windows\lib` 自動搜尋 `ffi.lib` 與對應 DLL 目錄。

5. **Rust 工具鏈（CPU 穩定版 + GPU 鎖定版本 `1.99.1`）**
   - 根 Workspace（純 CPU Crates）使用標準穩定版 Rust (`stable-x86_64-pc-windows-msvc`)。
   - GPU Crate [`crates/cs-backend-cuda/rust-toolchain.toml`](../crates/cs-backend-cuda/rust-toolchain.toml) 已自動鎖定 `cuda-oxide` 相容通道 `1.99.1`（並於 [`.cargo/config.toml`](../crates/cs-backend-cuda/.cargo/config.toml) 自動帶入 `RUSTC_BOOTSTRAP = "1"`），請先安裝該通道與必備元件：
     ```powershell
     rustup toolchain install stable
     rustup toolchain install 1.99.1
     rustup component add rust-src rustc-dev rust-analyzer rustfmt clippy llvm-tools --toolchain 1.99.1
     ```

---

## 2. 開啟 Developer PowerShell 與安裝 `cargo-oxide`

### 2.1 初始化 Developer PowerShell 環境變數

務必開啟 **Developer PowerShell for VS 2022**（或 **x64 Native Tools Command Prompt** 後切換至 PowerShell）以載入 MSVC x64 `link.exe` 與 Windows SDK 環境，再設定 CUDA 與 Clang 路徑（請依本機實際 CUDA／LLVM 安裝路徑調整）：

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_TOOLKIT_PATH = $env:CUDA_PATH
# 若使用 VS 內建 Clang，路徑通常為：
# "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:VCPKG_ROOT = "C:\vcpkg"
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:LIBCLANG_PATH;$env:PATH"
```

> **提示**：若希望永久保留，可透過 Windows「編輯系統環境變數」設定 `CUDA_TOOLKIT_PATH`、`LIBCLANG_PATH` 與 `VCPKG_ROOT`，並將 `%CUDA_PATH%\bin`、`%LIBCLANG_PATH%` 加入系統 `Path`。

### 2.2 安裝 `cargo-oxide` CLI

```powershell
cargo +1.99.1 install --locked --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
```

<details>
<summary>備用方案：若需要對照或本地鎖定 <code>NVIDIA/cuda-rust</code> 原始碼編譯安裝</summary>

```powershell
git clone https://github.com/NVIDIA/cuda-rust.git C:\dev\cuda-rust
cd C:\dev\cuda-rust\cuda-oxide
cargo install --locked --path crates/cargo-oxide
```

</details>

---

## 3. 取得最新程式碼與完成 CPU + GPU 雙重驗證

### 3.1 同步倉庫並驗證純 CPU Workspace（6 個 Crates）

```powershell
git clone https://github.com/Funtrollor/Cube_sphere_tansport_equation.git
# 若本機已有倉庫：git checkout main; git pull origin main
cd Cube_sphere_tansport_equation

# 1. 靜態檢查（零警告）
cargo clippy --workspace --all-targets -- -D warnings

# 2. 全套單元與數值收斂驗證（23 項測試，含 O(h^4)、O(dt^4) 與機器精度檢驗）
cargo test --release --workspace

# 3. 編譯 CLI 並試跑基準算例
cargo build --release -p cs-cli
.\target\release\cs-cli.exe info --k 8 --order 4
```

### 3.2 環境體檢與執行 GPU vs CPU 對拍（`gpu_parity`）

在同一個 **Developer PowerShell for VS 2022** 中進入獨立隔離的 GPU Crate 目錄：

```powershell
cd crates\cs-backend-cuda

# 1. 檢查 CUDA Toolkit、libclang、lld-link、vcpkg libffi 與 Rust 1.99.1 是否全數就緒
cargo oxide doctor

# 2. 依本 crate 的 NVIDIA/cuda-rust git 相依編譯並快取 rustc_codegen_cuda
cargo oxide setup

# 3. 編譯 release GPU 後端與執行對拍基準
cargo oxide build --release
cargo oxide run --release --bin gpu_parity
```

`gpu_parity` 會透過完全相同的 [`ComputeBackend<T>`](../crates/cs-solver/src/backend.rs) 與 [`run_lsrk45_simulation`](../crates/cs-solver/src/lsrk45.rs#L112) 介面執行以下 4 項自動驗收：

1. **Solid-Body Rotation (`f64`, `K=4, p=4`, 1 圈)**：驗證 GPU 與 CPU Rayon 逐點誤差 $\max |\phi_{\text{GPU}} - \phi_{\text{CPU}}| < 10^{-13}$、質量相對誤差 $< 10^{-14}$。
2. **Nair & Lauritzen (2010) Case 4 變形流 (`f64`, `K=4, p=5`, 1 週期)**：驗證無超越函數加倍角 GPU 風場核心與 CPU 誤差 $< 10^{-12}$、質量誤差 $< 10^{-14}$。
3. **單精度對拍 (`f32`, `K=8, p=4`)**：驗證 `CudaBackend<f32>` 與 `CpuBackend<f32>` 誤差 $< 2 \times 10^{-5}$。
4. **高解析度吞吐量基準（`f64` 與 `f32`，`K=16/32, p=4`）**：實測 RTX 5070 Ti 的 `MDOF-stages/s` 與相對於 CPU Rayon 的實際加速比。

---

## 4. PTX 檢視、RTX 5070 Ti (`sm_120`) 調校與 Compute Sanitizer

### 4.1 檢視生成的 PTX 與暫存器使用率

```powershell
cd crates\cs-backend-cuda
cargo oxide inspect --release --bin gpu_parity
```

取得建置目錄輸出的 `.ptx` 後，使用 `ptxas` 針對 RTX 5070 Ti（Compute Capability `12.0` / `sm_120`）檢查暫存器壓力（Register Spills / Occupancy）：

```powershell
ptxas -v -arch=sm_120 <path-to-artifact>.ptx -o "$env:TEMP\test_sm120.cubin"
Remove-Item "$env:TEMP\test_sm120.cubin" -ErrorAction SilentlyContinue
```

### 4.2 在 Windows WDDM 啟用 NVIDIA Compute Sanitizer

在執行 `cargo oxide sanitize` 之前，須從 **系統管理員權限的 PowerShell / 命令提示字元** 執行一次 WDDM 除錯介面啟用腳本（只需設定一次）：

```powershell
& "$env:CUDA_PATH\compute-sanitizer\EnableDebuggerInterface.bat"
```

啟用後即可於一般 Developer PowerShell 檢查越界或競態條件：

```powershell
cd crates\cs-backend-cuda
cargo oxide sanitize --bin gpu_parity
```

### 4.3 RTX 5070 Ti (`sm_120`) 架構特性與設計重點

- **1-thread-per-DOF 融合核心（零原子競爭）**：`split_form_and_sat_{f64,f32}` 將分裂散度與 4 邊 SAT 懲罰融合成單一 Kernel，每位執行緒只寫入自身的 `rhs[tid]`，透過 `DisjointSlice<T>` 與只讀鄰接表讀取跨面板節點，無需 `atomicAdd` 亦不受高階 $p$ 的 Shared Memory 容量限制。
- **FP32 vs FP64 特性**：Consumer Blackwell (`sm_120`) 的 FP32 ALU 與記憶體頻寬優勢極大，而 FP64 比例約為 `1 : 64`。系統預設以 `f64` 保證嚴格譜精度與機器精度質量守恆，同時透過 [`CudaScalar`](../crates/cs-backend-cuda/src/lib.rs#L564) 提供一鍵切換 `f32` 高吞吐測試。
- **Phase 3 PyO3 擴充提醒**：本專案目前已驗證之可執行檔（`gpu_parity.exe`）透過 EXE 內嵌 `.oxart` 載入 PTX，在 Windows 可直接運作。未來若在 Phase 3 新增 Python C 擴充模組（`.pyd` / `cdylib`），因 `cuda-host` 的 `load_embedded_module_from_anchor` 目前僅支援 Linux/Android `procfs`，在原生 Windows 建置 `.pyd` 時需改用顯式 PTX/cubin 路徑或將 `.oxart` 獨立載入，而 Linux / WSL2 `.so` 則已直接支援 anchor 自動載入。

---

## 5. 原生 Windows 常見問題排查表

| 症狀 / 錯誤訊息 | 可能原因 | 解法 |
| --- | --- | --- |
| `cargo oxide doctor` 找不到 `link.exe` / Windows SDK | 未從 Developer PowerShell for VS 2022 開啟終端機 | 改從開始功能表的 **Developer PowerShell for VS 2022** 執行，或先呼叫 `vcvars64.bat` |
| 無法找到 `libclang.dll` 或 Clang 內建標頭 | 未勾選 C++ Clang 工具或未設 `$env:LIBCLANG_PATH` | 設定 `$env:LIBCLANG_PATH` 指向含 `libclang.dll` 與 `lld-link.exe` 的 `bin` 目錄並加入 `PATH` |
| 找不到 `ffi.lib` / `libffi` | `vcpkg` 未安裝 `libffi:x64-windows` 或找不到 `VCPKG_ROOT` | 確認 `$env:VCPKG_ROOT = "C:\vcpkg"`，執行 `& "$env:VCPKG_ROOT\vcpkg.exe" install libffi:x64-windows` |
| `cuda.h not found` / Toolkit version &lt; 13.0 | `CUDA_TOOLKIT_PATH` 未設或安裝版次過舊 | 安裝 CUDA Toolkit 13.0+，確認 `$env:CUDA_TOOLKIT_PATH` 指向安裝目錄且 `$env:PATH` 含 `bin\x64` 與 `bin` |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)` | Toolkit 產出的 PTX 版本高於目前 Windows 驅動程式上限 | 更新 Windows NVIDIA 驅動至最新版本，或改指較低小版次的 CUDA 13.x Toolkit |
| `cargo oxide sanitize` 無法附加 GPU 行程 | WDDM 驅動未啟用除錯介面 | 以系統管理員執行 `& "$env:CUDA_PATH\compute-sanitizer\EnableDebuggerInterface.bat"` |
