# CUDA Practice Guide (NVIDIA GPUs: H100, A100, Consumer GPUs)

Note: exact tool names, flags, and defaults vary across CUDA toolkit versions and GPU architectures. If a command is not found, check `nvcc --version`, your compute capability, and the CUDA toolkit docs.

---

## GPU architecture primer: NVIDIA Hopper H100 SXM (SM 9.0)

Numbers are from NVIDIA's public specs. Verify on your machine with the commands in the next subsection.

### Big picture

- **Monolithic die** (unlike AMD's chiplet design): single GH100 die with 8 GPCs (Graphics Processing Clusters).
- **132 SMs** (Streaming Multiprocessors): GH100 has 144 SMs, H100 SXM enables 132.
- **80 GB HBM3** with **3.35 TB/s** peak memory bandwidth (H100 SXM5). The H100 PCIe variant has 2 TB/s.
- **50 MB L2 cache** shared across all SMs.
- **Peak compute (dense)**: ~990 TFLOP/s FP16/BF16 (with Tensor Cores), ~1979 TFLOP/s FP8, ~67 TFLOP/s FP32, ~34 TFLOP/s FP64.
- **Sparsity**: 2:4 structured sparsity doubles the effective Tensor Core throughput for supported formats.

### Comparison: A100 (Ampere, SM 8.0)

| Spec | A100 SXM (80 GB) | H100 SXM |
|---|---|---|
| SMs | 108 | 132 |
| HBM | 80 GB HBM2e | 80 GB HBM3 |
| Bandwidth | 2.0 TB/s | 3.35 TB/s |
| FP16 Tensor | 312 TFLOP/s | 990 TFLOP/s |
| FP32 | 19.5 TFLOP/s | 67 TFLOP/s |
| L2 Cache | 40 MB | 50 MB |
| TDP | 400W | 700W |

### Visual map (ASCII diagrams, three zoom levels)

**Zoom 1: the whole GPU.** The GH100 die has 8 GPCs, each containing multiple TPCs, each containing 2 SMs. All SMs share a single L2 cache backed by HBM memory controllers.

```
+---------------------------------------------------------------+
| GH100 (H100 SXM)                                             |
|                                                               |
|  +------+ +------+ +------+ +------+                          |
|  | GPC0 | | GPC1 | | GPC2 | | GPC3 |                          |
|  | 18SM | | 18SM | | 18SM | | 18SM |                          |
|  +------+ +------+ +------+ +------+                          |
|  +------+ +------+ +------+ +------+                          |
|  | GPC4 | | GPC5 | | GPC6 | | GPC7 |                          |
|  | 18SM | | 18SM | | 18SM | | 18SM |  (132 of 144 enabled)    |
|  +------+ +------+ +------+ +------+                          |
|                                                               |
|  +----------------------------------------------------------+ |
|  | L2 Cache, 50 MB, shared by all 132 SMs                    | |
|  +----------------------------------------------------------+ |
|  +----------------------------------------------------------+ |
|  | Memory Controllers -> HBM3 (80 GB, 3.35 TB/s)            | |
|  +----------------------------------------------------------+ |
|  | NVLink 4.0 (900 GB/s), PCIe Gen5                          | |
|  +----------------------------------------------------------+ |
+---------------------------------------------------------------+
```

**Zoom 2: inside one SM.** Each SM contains 4 processing blocks (quadrants), each with its own warp scheduler, register file partition, CUDA cores, and Tensor Cores. Shared memory / L1 is unified per SM.

```
+---------------------------------------------------------------+
| SM (Streaming Multiprocessor)                                 |
|  +----------------------------+ +----------------------------+ |
|  | Quadrant 0                 | | Quadrant 1                 | |
|  | Warp Scheduler             | | Warp Scheduler             | |
|  | 16384 registers (32-bit)   | | 16384 registers (32-bit)   | |
|  | 32 FP32 cores              | | 32 FP32 cores              | |
|  | 16 FP64 cores              | | 16 FP64 cores              | |
|  | 1 Tensor Core (4th gen)    | | 1 Tensor Core (4th gen)    | |
|  | 16 INT32 cores             | | 16 INT32 cores             | |
|  | 4 SFUs + 4 LD/ST units     | | 4 SFUs + 4 LD/ST units     | |
|  +----------------------------+ +----------------------------+ |
|  +----------------------------+ +----------------------------+ |
|  | Quadrant 2                 | | Quadrant 3                 | |
|  | (same layout)              | | (same layout)              | |
|  +----------------------------+ +----------------------------+ |
|  +----------------------------------------------------------+ |
|  | Shared Memory / L1 Cache (228 KB combined, configurable)  | |
|  +----------------------------------------------------------+ |
+---------------------------------------------------------------+

Per SM totals: 128 FP32 cores, 64 FP64 cores, 4 Tensor Cores,
               65536 registers, 228 KB shared/L1
Max threads per SM: 2048 (64 warps)
Max threads per block: 1024
```

**Zoom 3: inside one SM quadrant.** One warp scheduler issues instructions to its execution units. Threads run in warps of 32.

```
+--------------------------------------------------+
| SM Quadrant: runs 32-thread warps                |
|  +-----------------------------------------+     |
|  | Warp Scheduler + Dispatch Unit          |     |
|  +-----------------------------------------+     |
|  +----------+ +----------+ +----------+          |
|  | 32 FP32  | | 16 FP64  | | 16 INT32 |          |
|  | cores    | | cores    | | cores    |          |
|  +----------+ +----------+ +----------+          |
|  +----------+ +----------+                       |
|  | 1 Tensor | | 4 SFUs   |                       |
|  | Core     | | (sin,cos,|                       |
|  | (4th gen)| |  rcp,etc)|                       |
|  +----------+ +----------+                       |
|  +------------------------------------------+   |
|  | 16384 x 32-bit registers                 |   |
|  +------------------------------------------+   |
+--------------------------------------------------+
```

### Where everything physically lives

Understanding what is **on the SM** vs **off the SM** is critical:

```
+---------------------------------------------------------------+
| GPU Die                                                       |
|                                                               |
|  +-------------------------+  +-------------------------+     |
|  | SM 0 (on-chip)          |  | SM 1 (on-chip)          |     |
|  |  Registers (~1 cycle)   |  |  Registers (~1 cycle)   |     |
|  |  Warp Schedulers        |  |  Warp Schedulers        |     |
|  |  CUDA Cores, Tensor     |  |  CUDA Cores, Tensor     |     |
|  |  Cores, SFUs, LD/ST     |  |  Cores, SFUs, LD/ST     |     |
|  |  +-----------+          |  |  +-----------+          |     |
|  |  | Unified   |          |  |  | Unified   |          |     |
|  |  | SRAM pool |          |  |  | SRAM pool |          |     |
|  |  | 228 KB    |          |  |  | 228 KB    |          |     |
|  |  | ┌───────┐ |          |  |  | ┌───────┐ |          |     |
|  |  | │Shared │ |          |  |  | │Shared │ |          |     |
|  |  | │Memory │<-- ON SM   |  |  | │Memory │ |          |     |
|  |  | ├───────┤ |          |  |  | ├───────┤ |          |     |
|  |  | │L1     │ |          |  |  | │L1     │ |          |     |
|  |  | │Cache  │<-- ON SM   |  |  | │Cache  │ |          |     |
|  |  | └───────┘ |          |  |  | └───────┘ |          |     |
|  |  +-----------+          |  |  +-----------+          |     |
|  +-------------------------+  +-------------------------+     |
|                ... (132 SMs total on H100) ...                |
|                                                               |
|  +----------------------------------------------------------+ |
|  | L2 Cache (50 MB) — shared, NOT on any SM    <-- OFF SM    | |
|  +----------------------------------------------------------+ |
|  +----------------------------------------------------------+ |
|  | Memory Controllers → HBM (80 GB)           <-- OFF CHIP  | |
|  +----------------------------------------------------------+ |
+---------------------------------------------------------------+
```

**Key point:** Shared memory and L1 cache are the **same physical SRAM** on the SM. You configure the split — more shared memory means less L1, and vice versa. Both are on-chip, on the SM itself, which is why they're fast (~20-30 cycles). L2 is on the GPU die but is a separate structure shared by all SMs (~200 cycles). HBM is off-chip entirely (~400-600 cycles).

Path from a thread to memory: `Thread -> Registers (on SM) -> Shared Mem/L1 (on SM) -> L2 (on die, off SM) -> HBM (off chip)`.

### Glossary of GPU hardware units

| Abbreviation | Full Name | What It Is |
|---|---|---|
| **GPC** | Graphics Processing Cluster | Top-level grouping on the die. H100 has 8 GPCs. Contains multiple TPCs. Determines Thread Block Cluster co-scheduling in Hopper. |
| **TPC** | Texture Processing Cluster | Contains 2 SMs + shared texture units. The grouping level for texture filtering hardware. Each GPC has ~9 TPCs (H100). |
| **SM** | Streaming Multiprocessor | The core compute unit. All your kernel code runs here. Contains warp schedulers, CUDA cores, Tensor Cores, registers, shared memory, L1 cache. |
| **SFU** | Special Function Unit | Hardware unit inside each SM quadrant (4 SFUs per quadrant, 16 per SM) that computes transcendental functions: `sinf()`, `cosf()`, `expf()`, `logf()`, `rsqrtf()`, reciprocal (`1/x`). Much faster than computing these in software, but lower throughput than CUDA cores — only 4 SFUs vs 32 FP32 cores per quadrant. Using many SFUs in a kernel can become a bottleneck. |
| **LD/ST** | Load/Store Unit | Issues memory load and store operations. 4 per SM quadrant (16 per SM). Handles address generation and feeds data to/from the cache hierarchy. |
| **FP32 Core** | CUDA Core (FP32) | Scalar floating-point unit. Executes one FP32 operation per clock per core. 32 per quadrant, 128 per SM. |
| **FP64 Core** | Double-Precision Core | Scalar double-precision unit. 16 per quadrant, 64 per SM on H100. Half the FP32 count, so FP64 throughput is ~half of FP32 (not always — depends on architecture). |
| **INT32 Core** | Integer Core | Executes integer operations. 16 per quadrant, 64 per SM. On Volta+ can execute INT32 concurrently with FP32. |
| **Tensor Core** | Tensor Core | Matrix multiply-accumulate unit. 1 per quadrant, 4 per SM. Operates on tiles (e.g., 16×16×16) of FP16/BF16/TF32/FP8/INT8, accumulating into FP32. Orders of magnitude faster than FP32 cores for matrix math. |

### GPC vs TPC vs SM hierarchy

```
GPU Die
├── GPC 0 (Graphics Processing Cluster)
│   ├── TPC 0 (Texture Processing Cluster)
│   │   ├── SM 0 (Streaming Multiprocessor) ← your code runs here
│   │   └── SM 1
│   ├── TPC 1
│   │   ├── SM 2
│   │   └── SM 3
│   ├── ... (~9 TPCs per GPC)
│   └── Raster Engine (graphics, not used in compute)
├── GPC 1
│   └── ... (same structure)
├── ... (8 GPCs total on H100, 132 of 144 SMs enabled)
├── L2 Cache (50 MB, shared by all SMs across all GPCs)
├── Memory Controllers → HBM3
├── GigaThread Engine (global thread block scheduler)
├── NVLink 4.0 interface
└── PCIe Gen5 interface
```

**GPC (Graphics Processing Cluster)** is the top-level grouping. H100 has 8 GPCs, each containing ~9 TPCs (~18 SMs). GPCs are mostly transparent for compute workloads, but they matter for:
- **Thread Block Clusters (Hopper):** blocks in a cluster are co-scheduled on SMs within the same GPC
- **L1 → L2 routing:** SMs within the same GPC share closer routing to their L2 partition

**TPC (Texture Processing Cluster)** contains 2 SMs plus shared texture filtering hardware. In compute workloads, the TPC grouping is mostly invisible — you interact with SMs. The texture units in the TPC are used when you access texture memory or use `__ldg()` (read-only cache path).

**SM (Streaming Multiprocessor)** is the actual compute unit. All your kernel code runs on SMs. Each SM:
- Has its own register file (65536 x 32-bit), shared memory, L1 cache, warp schedulers
- Can run up to 2048 concurrent threads (64 warps)
- Is assigned one or more thread blocks by the hardware scheduler
- Has 4 Tensor Cores (Hopper) for matrix operations
- Contains 128 FP32 cores, 64 FP64 cores, 64 INT32 cores, 16 SFUs, 16 LD/ST units

Why it matters for kernels:
- **L2 is global across all SMs.** Unlike AMD's per-XCD L2, NVIDIA's single L2 means any SM can hit data cached by any other SM. This simplifies data placement reasoning.
- **Thread block scheduling is hardware-managed.** The GigaThread engine distributes blocks to SMs. You don't control which SM a block lands on (but can influence it with cluster launches in Hopper).
- **Hopper introduced Thread Block Clusters** (SM 9.0): groups of thread blocks guaranteed to be co-scheduled on SMs in the same GPC, enabling efficient shared memory access across blocks via distributed shared memory.

### Inside one SM (detailed)

- **4 warp schedulers** per SM (one per quadrant), each managing up to 16 warps. Each scheduler can issue 1 instruction per clock to its execution units.
- **Warp size = 32 threads** (AMD wavefront = 64). This affects reductions, shuffles, and ballot masks (32-bit masks).
- **CUDA Cores (FP32)**: 128 per SM (32 per quadrant). Each executes one single-precision floating-point or integer operation per clock. These are the workhorses for element-wise operations (vec_add, activation functions, etc.).
- **FP64 Cores**: 64 per SM (16 per quadrant). Double-precision. Half the throughput of FP32 on H100.
- **INT32 Cores**: 64 per SM. Can execute concurrently with FP32 cores (since Volta) — the scheduler can issue an FP32 instruction and an INT32 instruction in the same clock to different execution units.
- **Tensor Cores** (4th gen on Hopper): 4 per SM (1 per quadrant). Execute MMA (matrix multiply-accumulate) instructions for FP16/BF16/FP8/TF32/INT8 tiles via `wmma` or `mma.sync` PTX instructions. This is where the peak TFLOP/s numbers come from — FP32 cores alone cannot approach Tensor Core throughput.
- **SFUs (Special Function Units)**: 16 per SM (4 per quadrant). Handle transcendental math: `sinf`, `cosf`, `expf`, `logf`, `rsqrtf`, reciprocal. These are **lower throughput** than CUDA cores — if your kernel is heavy on `expf()` (like softmax), SFU throughput can become the bottleneck. The profiler metric `sm__inst_executed_pipe_sfu` shows SFU utilization.
- **LD/ST (Load/Store) Units**: 16 per SM (4 per quadrant). Generate addresses and issue memory transactions. Each unit handles one address per clock. If your kernel does many loads per instruction (e.g., gather operations), LD/ST units can limit throughput.
- **Shared memory / L1 cache**: up to 228 KB per SM on H100. This is a **single unified SRAM pool** physically on the SM. You configure the split between shared memory (programmer-managed) and L1 cache (hardware-managed):
  ```cpp
  // Request more shared memory, less L1
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); // percent for shared
  ```
  Organized in 32 banks of 4 bytes each. Watch for bank conflicts.
- **Register file**: 65536 x 32-bit registers per SM, partitioned across all resident warps. More registers per thread = fewer warps per SM (lower occupancy). Example: 128 registers per thread × 32 threads per warp = 4096 registers per warp → 65536/4096 = 16 warps max (25% occupancy if max is 64).
- **Max threads per block**: 1024.
- **Max blocks per SM**: 32 (Hopper), 16–32 depending on architecture.

### NVIDIA vs AMD vocabulary

| NVIDIA / CUDA | AMD / HIP |
|---|---|
| SM | CU (Compute Unit) |
| Warp (32) | Wavefront (64 on MI300X) |
| Shared memory | LDS (Local Data Share) |
| Tensor Cores | Matrix Cores (MFMA instructions) |
| `__shfl_sync` | `__shfl` (HIP, 64-wide) / DPP ops |
| nvcc / PTX / SASS | hipcc / LLVM IR / GCN-CDNA ISA |
| cuBLAS / cuDNN | rocBLAS, hipBLASLt / MIOpen |
| Nsight Compute / Systems | rocprof-compute / rocprofv3 |
| compute-sanitizer | no direct equivalent |
| cuda-gdb | rocgdb |

### What the architecture means for your kernels

1. **Block size should be a multiple of 32** (the warp size). A 48-thread block has 2 warps, one of which is only half active.
2. **Memory coalescing**: consecutive threads should access consecutive addresses. A warp reading 32 consecutive floats = one 128-byte transaction. Strided or scattered access = multiple transactions, wasted bandwidth.
3. **Occupancy**: limited by registers, shared memory, and block size. Use `cudaOccupancyMaxPotentialBlockSize` or Nsight Compute's occupancy analysis. Higher occupancy helps hide latency but isn't always the goal.
4. **Shared memory**: tile your data into shared memory to exploit reuse. Pad shared arrays (`[TILE][TILE+1]`) to avoid bank conflicts.
5. **Use Tensor Cores** for peak GEMM performance (via cuBLAS, CUTLASS, or `wmma`/`mma` intrinsics in your own kernels). Plain FMA loops cannot approach Tensor Core peak.
6. **L2 is shared across all SMs.** Data cached by one SM's load can be hit by any other SM, unlike AMD where L2 is per-XCD. This makes global data sharing easier but also means L2 contention from unrelated blocks.
7. **Roofline thinking**: vec_add is bandwidth-bound (compare to peak HBM bandwidth), large GEMM should be compute-bound (compare to peak TFLOP/s for the dtype).

### Warp divergence

**What it is:** A warp (32 threads on NVIDIA) must execute the same instruction in lockstep. When threads in a warp take different sides of an `if/else` or loop with varying trip counts, the hardware serializes both paths — threads not on the active path are masked off and do no useful work. This is warp divergence.

**Why it happens:**
- **Data-dependent branching**: `if (A[tid] > 0)` where some elements are positive and others aren't.
- **Thread-ID-dependent branching**: `if (threadIdx.x % 2 == 0)` splits every warp 50/50.
- **Variable-length loops**: `while (data[tid])` where each thread iterates a different number of times.

**Example — divergent reduction (bad):**
```cpp
// threads with odd stride sit idle every other iteration
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)          // <-- divergent: interleaved active/inactive
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

**Example — non-divergent reduction (good):**
```cpp
// active threads are contiguous → no divergence until warp partially empties
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)                      // <-- first s threads are active, rest are off
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

**How to handle / avoid it:**
1. **Restructure branches so threads diverge across warps, not within them.** If threads 0–31 all take the same path and threads 32–63 take the other, there is zero divergence — each warp is uniform.
2. **Use predication / branchless math**: replace `if (x > 0) y = x; else y = 0;` with `y = max(x, 0);` or `y = x * (x > 0);`. The compiler often does this automatically for simple cases.
3. **Sort or bucket data** so threads in the same warp hit the same branch.
4. **Reduce-then-compact**: compact active elements into a dense buffer (stream compaction) and launch a second kernel on just those.
5. **Check with the profiler**: Nsight Compute reports branch efficiency and divergent branch metrics. Look for high "divergent branch" counts.

### Memory coalescing deep dive

Coalescing is one of the most important performance concepts in CUDA. When a warp executes a load/store instruction, the hardware combines (coalesces) individual thread requests into as few memory transactions as possible.

**Coalesced access (good):**
```cpp
// 32 threads read 32 consecutive floats = one 128-byte transaction
float val = data[blockIdx.x * blockDim.x + threadIdx.x];
```

**Strided access (bad):**
```cpp
// Column-major access in a row-major matrix — stride of N between thread reads
float val = matrix[threadIdx.x * N + col];  // stride = N elements apart
```
This generates up to 32 separate transactions (one per thread in the worst case).

**Struct of Arrays vs Array of Structs:**
```cpp
// AoS (bad for coalescing): threads in a warp read x fields scattered by sizeof(Particle)
struct Particle { float x, y, z, w; };
Particle particles[N];
float px = particles[tid].x;  // stride = 16 bytes

// SoA (good for coalescing): threads in a warp read consecutive x values
struct Particles { float *x, *y, *z, *w; };
float px = particles.x[tid];  // consecutive access
```

**Rules of thumb:**
- Align data to 128 bytes (use `cudaMallocPitch` for 2D arrays).
- Use vector types (`float4`, `int4`) for 128-bit loads per thread when possible.
- If you must access a matrix column-wise, load a tile into shared memory first (shared memory has no coalescing requirement, only bank conflicts).
- The L1 cache (since Volta) helps with some non-coalesced patterns, but don't rely on it.

### Shared memory bank conflicts

Shared memory is divided into 32 banks (matching the warp size). Each bank can serve one 4-byte word per clock. When multiple threads in a warp access different words in the same bank, the accesses are serialized — this is a bank conflict.

**How banks are mapped:**
```
Address 0   -> Bank 0
Address 4   -> Bank 1
Address 8   -> Bank 2
...
Address 124 -> Bank 31
Address 128 -> Bank 0   (wraps around)
```

**No conflict (each thread hits a different bank):**
```cpp
__shared__ float s[256];
float val = s[threadIdx.x];          // thread 0 -> bank 0, thread 1 -> bank 1, etc.
```

**32-way bank conflict (all threads hit the same bank):**
```cpp
__shared__ float s[256];
float val = s[threadIdx.x * 32];     // all threads hit bank 0
```

**Common fix — padding:**
```cpp
__shared__ float tile[32][32];       // column access: tile[row][threadIdx.x] has no conflict
                                     // but tile[threadIdx.x][col] → 32-way conflict

__shared__ float tile[32][32 + 1];   // padding fixes column access conflicts
```

**Broadcast exception:** if all threads read the *same* address, it's a broadcast, not a conflict.

### Roofline model

A roofline graph plots:

- **X axis:** arithmetic intensity (AI) = FLOPs performed per byte moved from memory (FLOP/byte)
- **Y axis:** achieved performance (FLOP/s)

Both axes are usually log scale.

```
perf
 |            ______________  <- compute roof (peak FLOP/s)
 |          /
 |        /   <- memory roof (slope = peak bandwidth)
 |      /
 |    /
 |  /
 +------------------------------ arithmetic intensity
        ^ ridge point
```

- **Sloped part (left):** memory bound. Performance = bandwidth × AI.
- **Flat part (right):** compute bound. Performance is capped by peak FLOP/s.
- **Ridge point:** peak FLOP/s ÷ peak bandwidth. Kernels left of it are memory bound; right of it are compute bound.
- Each kernel is one dot. The farther below the roof, the more performance is left on the table.

Approximate H100 SXM numbers:

| Case | Calculation | Result |
|---|---|---|
| vec_add | 1 FLOP per 12 bytes (2 loads + 1 store of 4-byte floats) | AI ≈ 0.08, far left, memory bound |
| Ridge point, FP16 Tensor | ~990 TFLOP/s / ~3.35 TB/s | ~295 FLOP/byte |
| Ridge point, FP32 | ~67 TFLOP/s / ~3.35 TB/s | ~20 FLOP/byte |

Approximate A100 SXM numbers:

| Case | Calculation | Result |
|---|---|---|
| Ridge point, FP16 Tensor | ~312 TFLOP/s / ~2.0 TB/s | ~156 FLOP/byte |
| Ridge point, FP32 | ~19.5 TFLOP/s / ~2.0 TB/s | ~10 FLOP/byte |

Estimating AI for sample kernels:
- **vec_add:** ~0.08 FLOP/byte, memory bound. Best case approaches peak bandwidth.
- **matmul_naive:** no data reuse, each output needs 2N FLOPs and reads ~8N bytes → AI ≈ 0.25 FLOP/byte. Memory bound.
- **matmul_tiled:** each value loaded into shared memory is reused TILE times → AI ≈ TILE/4 FLOP/byte (~4 for TILE=16, ~8 for TILE=32). Better but still well below the compute ridge.
- Reaching the compute roof requires Tensor Cores + deep register tiling (what cuBLAS/CUTLASS do).

Generate a roofline with Nsight Compute:
```bash
ncu --set roofline -o matmul_roofline ./matmul
ncu-ui matmul_roofline.ncu-rep   # open the GUI to see the roofline chart
```

### Instruction scheduling and latency hiding

The GPU hides memory and instruction latency by running many warps concurrently on each SM. When one warp stalls (waiting for a memory load, for example), the warp scheduler instantly switches to another ready warp — no context-switch cost because each warp has its own registers.

**How warp schedulers work (Hopper):**
- Each SM has **4 warp schedulers**, one per quadrant.
- Each scheduler can issue **1 instruction per clock** to its execution units.
- Each scheduler manages a pool of warps (up to 16 warps per scheduler, 64 total per SM).
- Scheduling is **greedy**: the scheduler picks a ready warp (one whose operands are available) and issues its next instruction. If no warp is ready, the scheduler stalls — this is wasted cycles.

```
Clock 0:  Scheduler 0 issues WARP_A instruction (memory load, ~400 cycle latency)
Clock 1:  Scheduler 0 issues WARP_B instruction (WARP_A still waiting for data)
Clock 2:  Scheduler 0 issues WARP_C instruction
...
Clock N:  WARP_A's data arrives, scheduler issues WARP_A's next instruction
```

**Latency numbers (approximate, H100):**
- Register access: ~1 cycle
- Shared memory: ~20–30 cycles
- L1 cache hit: ~30 cycles
- L2 cache hit: ~200 cycles
- HBM (global memory): ~400–600 cycles

**Instruction-level parallelism (ILP):** Within a single warp, independent instructions can be pipelined. If instruction B doesn't depend on instruction A's result, the hardware can issue B before A completes:

```cpp
// Low ILP: each line depends on the previous
float a = x[i];
float b = a * 2.0f;     // depends on a
float c = b + 1.0f;     // depends on b

// High ILP: independent work interleaved
float a = x[i];
float d = y[i];          // independent of a — can issue in parallel
float b = a * 2.0f;
float e = d * 3.0f;      // independent of b
```

**Why occupancy matters for latency hiding:**
- If you have 64 warps per SM and each warp stalls for 400 cycles, but each issues 1 instruction before stalling, you need at least 400/1 = 400 warps to keep the scheduler busy. You can't get 400, so you also need ILP within each warp.
- In practice, 50%+ occupancy is usually sufficient for memory-bound kernels. Compute-bound kernels can sometimes do well at lower occupancy if ILP is high.
- The occupancy calculator (Nsight Compute or `cudaOccupancy*` APIs) tells you the theoretical maximum; the profiler tells you actual achieved occupancy.

**What the profiler shows:**
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed` — achieved occupancy
- `smsp__warp_issue_stalled_*` — why warps are stalling (memory, compute, synchronization, etc.)
- `sm__inst_executed_pipe_*` — instructions issued per pipe (FP32, FP64, tensor, etc.)

### Cache hierarchy details

**Cache line sizes:**
- **L1 cache**: 128-byte cache lines. A warp load that spans one 128-byte line is one transaction. Misaligned or scattered access causes multiple transactions.
- **L2 cache**: 128-byte cache lines, unified across all SMs.

**L1 cache behavior (since Volta):**
- L1 and shared memory share a unified on-chip memory pool. On H100, this pool is 228 KB per SM.
- By default, global loads go through L1. You can control this with `__ldg()` (read-only, uses texture cache path) or compiler hints.
- The L1 is **write-through** for global memory writes on most architectures — writes go to L2, not cached in L1.
- Shared memory operations bypass the cache entirely (they use the same SRAM but as explicitly managed memory).

**L2 cache:**
- Shared by all SMs, **write-back** (writes are cached and flushed later).
- On H100: 50 MB. On A100: 40 MB.
- **L2 persistence control** (Ampere+): you can hint that certain data should stay in L2 longer:
  ```cpp
  cudaAccessPropertyWindow window = {};
  window.base_ptr = d_data;
  window.num_bytes = bytes;
  window.hitRatio = 1.0f;      // try to keep all of this in L2
  window.hitProp = cudaAccessPropertyPersisting;
  window.missProp = cudaAccessPropertyStreaming;
  cudaCtxSetAccessPolicyWindow(&window);
  ```
  This is useful when a small buffer is read repeatedly (e.g., an embedding table).

**L2 partitioning (Hopper):**
- H100 can partition L2 between SMs for better isolation in MIG (Multi-Instance GPU) mode.

**Cache coherence:**
- Within a kernel launch, L1 caches across SMs are **not coherent** with each other. Thread block A on SM 0 and thread block B on SM 1 cannot communicate through L1.
- L2 provides coherence for global memory: if block A writes to global memory and a `__threadfence()` is issued, subsequent reads by other blocks will see the update (after they miss L1 and hit L2).
- `__threadfence()` ensures writes are visible to other blocks. `__threadfence_block()` ensures writes are visible within the block. `__threadfence_system()` ensures writes are visible to the host and other GPUs.

### Constant memory and texture memory

**Constant memory:**
- 64 KB total, read-only from the device, cached in a dedicated constant cache per SM.
- Optimized for **broadcast access**: when all threads in a warp read the same address, it's served in one cycle. When threads read different addresses, accesses are serialized.
- Best for: lookup tables, kernel parameters, coefficients that all threads share.

```cpp
__constant__ float filter[256];  // declare in global scope

// Host side: copy to constant memory
cudaMemcpyToSymbol(filter, h_filter, 256 * sizeof(float));

// Device side: all threads reading filter[0] = one cycle (broadcast)
// Threads reading filter[threadIdx.x] = serialized if > 1 unique address per warp
```

**Texture memory:**
- Uses the **texture cache** (separate from L1), optimized for 2D spatial locality.
- Read-only from kernels (bindless texture objects since CUDA 5.0).
- Benefits: hardware interpolation, automatic boundary handling (clamp/wrap), and the 2D cache is better than L1 for spatially local 2D access patterns.
- Modern CUDA code mostly uses `__ldg()` for read-only global loads (uses the same hardware path) rather than explicit texture objects, unless 2D interpolation is needed.

```cpp
// __ldg: read-only load through texture cache path (no 2D features, just caching)
float val = __ldg(&input[idx]);  // hints the compiler that input[idx] is read-only
```

### Virtual memory and TLB

GPUs use virtual memory with their own page tables and TLBs (Translation Lookaside Buffers), similar to CPUs but with GPU-specific details.

**Page sizes:**
- **Small pages**: 4 KB (like CPU)
- **Large pages**: 2 MB (used by default for `cudaMalloc` allocations on most architectures)
- Large pages reduce TLB pressure: a 2 MB page covers 512x more memory than a 4 KB page, so fewer TLB entries are needed.

**TLB hierarchy:**
```
Thread → L1 TLB (per SM) → L2 TLB (shared) → Page Table Walk (very slow)
```

- **L1 TLB**: small, per-SM, fast. Covers the most recently used pages for that SM.
- **L2 TLB**: larger, shared across SMs. Covers more pages but slower.
- **Page table walk**: if both TLBs miss, the GPU must walk the page table in memory. This adds hundreds of cycles to every memory access until the TLB is refilled.

**When TLB matters:**
- Accessing many small, scattered allocations (each may be on a different page).
- Unified memory page faults: when a GPU thread accesses a page that hasn't been migrated to GPU memory yet, it causes a page fault → the page is migrated from CPU → very expensive (microseconds, not cycles).
- Very large working sets that exceed TLB coverage.

**Unified Memory and page faults:**
```
Thread accesses page not in GPU memory
  → TLB miss → Page fault → Driver interrupt
  → Driver migrates page from CPU to GPU (or from another GPU)
  → TLB entry updated → Thread resumes
```
This is why `cudaMemPrefetchAsync` is important for Unified Memory: it pre-migrates pages to avoid runtime faults.

### Thread block scheduling

The **GigaThread Engine** (global scheduler) distributes thread blocks to SMs at kernel launch time and during execution.

**How it works:**
1. Kernel launch specifies grid dimensions (total blocks) and block dimensions.
2. The GigaThread engine assigns blocks to SMs based on resource availability.
3. Each SM can hold multiple blocks simultaneously (limited by registers, shared memory, and max-blocks-per-SM).
4. When a block finishes, the SM becomes available for a new block (if any remain in the grid).

**Scheduling order:**
- Blocks are generally dispatched in **linear order** (block 0 first, then 1, 2, ...), distributed round-robin across SMs.
- But this is **not guaranteed** by the CUDA programming model. You cannot assume any specific execution order of blocks.
- In practice, blocks are often assigned in 3D grid order: X varies fastest, then Y, then Z.

**Why it matters:**
- **Tail effect**: if you have 1000 blocks and 132 SMs, the first 132 blocks run in parallel. The last wave may have only a few blocks, leaving most SMs idle. This is the "tail effect" or "wave quantization."
  ```
  1000 blocks / 132 SMs = 7 full waves + 76 blocks in the last wave
  Last wave: 76 SMs busy, 56 idle → wasted compute
  ```
  Mitigation: make blocks smaller (more of them, smoother tail) or use persistent kernels.

- **Persistent kernels**: launch exactly as many blocks as SMs can hold, each block loops over its share of work:
  ```cpp
  __global__ void persistent_kernel(float* data, int total_work, int items_per_block) {
      int block_start = blockIdx.x * items_per_block;
      for (int i = block_start; i < total_work; i += gridDim.x * items_per_block) {
          // process item i
      }
  }
  ```

- **Hopper Thread Block Clusters**: for the first time, you can control co-scheduling. Blocks in a cluster are guaranteed to run on SMs in the same GPC, enabling distributed shared memory access across blocks.

### NVLink and interconnect

**NVLink** is NVIDIA's high-bandwidth GPU-to-GPU interconnect, bypassing PCIe.

| Generation | Per-link BW (bidirectional) | Links per GPU | Total BW | GPUs |
|---|---|---|---|---|
| NVLink 3.0 | 50 GB/s | 12 | 600 GB/s | A100 |
| NVLink 4.0 | 50 GB/s | 18 | 900 GB/s | H100 |

**NVSwitch**: a crossbar switch that connects all GPUs in a node with full bisection bandwidth. In an 8-GPU H100 DGX node, every GPU can talk to every other GPU at full NVLink speed simultaneously.

```
Without NVSwitch:         With NVSwitch:
GPU0 -- GPU1              GPU0 --|
GPU2 -- GPU3              GPU1 --|-- NVSwitch -- full mesh
(limited topology)        GPU2 --|
                          ...    |
```

**PCIe Gen5**: 64 GB/s bidirectional (per x16 slot). Used for CPU-GPU communication and for GPUs without NVLink.

**How to check topology:**
```bash
nvidia-smi topo --matrix
# Output shows NV18 (18 NVLink connections), PHB (PCIe host bridge), etc.
```

**Why it matters for kernels:**
- **Multi-GPU kernels**: data transfer between GPUs via NVLink is ~15x faster than PCIe. Use `cudaMemcpyPeerAsync` or NCCL for GPU-to-GPU transfers.
- **Gradient all-reduce in training**: NCCL uses NVLink rings/trees for collective operations. The interconnect bandwidth determines the communication overhead in data-parallel training.
- **Tensor parallelism**: splits a single layer across GPUs, requiring frequent all-reduce of activations. Only viable with NVLink bandwidth — too slow over PCIe.

### Power management and clock domains

GPUs have multiple clock domains and dynamic frequency scaling:

**Clock domains:**
- **Graphics/SM clock**: controls SM execution speed. This is what you lock for benchmarking.
- **Memory clock**: controls HBM bandwidth. Usually not user-adjustable.
- **Video clock**: for encode/decode engines (not relevant for compute).

**Dynamic clocking (GPU Boost):**
- The GPU automatically adjusts SM clock based on temperature, power draw, and utilization.
- Under heavy load, the clock may throttle down if the GPU hits its power or thermal limit.
- This causes **variable kernel runtime** — the same kernel may run faster on a cool GPU than a hot one.

**For consistent benchmarking:**
```bash
# Lock clocks to a fixed frequency
sudo nvidia-smi -pm 1                        # persistence mode
sudo nvidia-smi -lgc 1980,1980               # lock SM clock at 1980 MHz
sudo nvidia-smi -lmc 1593,1593               # lock memory clock (if supported)

# Run benchmarks...

sudo nvidia-smi -rgc                          # reset to default
sudo nvidia-smi -rmc                          # reset memory clock
```

**Power monitoring during a kernel:**
```bash
nvidia-smi dmon -s p -d 1                    # power draw every 1 second
# Or programmatically:
nvmlDeviceGetPowerUsage(device, &power_mW);  # NVML API
```

**Thermal throttling thresholds (typical):**
- **Slowdown**: ~83°C — GPU reduces clocks to stay below this.
- **Shutdown**: ~90°C — GPU shuts down to prevent damage.
- Check with: `nvidia-smi -q | grep -i "Temperature\|Throttle"`

### Query the architecture yourself

```bash
nvidia-smi                                    # GPU visible? temp, power, VRAM
nvidia-smi -q | grep -E "Product Name|Compute Mode|GPU Current Temp|Power Draw"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw --format=csv
nvidia-smi -q | grep -E "Product Name|Compute Mode|GPU Current Temp|Power Draw"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw --format=csv
```

Save as `devprops.cu` to print properties from the runtime:

```cpp
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    for (int d = 0; d < n; d++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        printf("Device %d: %s\n", d, p.name);
        printf("  compute capability  : %d.%d\n", p.major, p.minor);
        printf("  SMs                 : %d\n", p.multiProcessorCount);
        printf("  warp size           : %d\n", p.warpSize);
        printf("  max threads/block   : %d\n", p.maxThreadsPerBlock);
        printf("  max threads/SM      : %d\n", p.maxThreadsPerMultiProcessor);
        printf("  shared mem / block  : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  shared mem / SM     : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
        printf("  regs / block        : %d\n", p.regsPerBlock);
        printf("  regs / SM           : %d\n", p.regsPerMultiprocessor);
        printf("  global mem          : %.1f GB\n", p.totalGlobalMem / 1e9);
        printf("  L2 cache            : %d KB\n", p.l2CacheSize / 1024);
        printf("  clock rate          : %d MHz\n", p.clockRate / 1000);
        printf("  mem clock           : %d MHz\n", p.memoryClockRate / 1000);
        printf("  mem bus width       : %d bits\n", p.memoryBusWidth);
        printf("  async engines       : %d\n", p.asyncEngineCount);
        printf("  concurrent kernels  : %s\n", p.concurrentKernels ? "yes" : "no");
    }
    return 0;
}
```

```bash
nvcc -arch=sm_90 devprops.cu -o devprops && ./devprops
```

Exercise: compute the theoretical peak bandwidth and FLOP/s for your GPU from these numbers, then compare to what your `vec_add` and `matmul` actually achieve.

---

## 0. Setup

```bash
# Verify CUDA installation
nvcc --version
which nvcc

# Environment (usually set by the CUDA installer)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

mkdir -p ~/cuda-practice && cd ~/cuda-practice
```

---

## 1. Verify the stack (do this first)

```bash
nvidia-smi                            # GPU visible? driver version, CUDA version
nvidia-smi -L                         # list all GPUs with UUIDs
nvcc --version                        # toolkit version
ls /usr/local/cuda/bin | head -30     # see what tools you have

# Compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

PyTorch sanity check:

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Device:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
"
```

---

## 2. Sample kernel 1: vector add (with error checking + timing)

Save as `vec_add.cu`:

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                            \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);           \
        exit(1);                                                      \
    }                                                                 \
} while (0)

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 24;                  // 16M elements
    const size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = i * 0.5f; h_b[i] = i * 2.0f; }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());           // catches launch config errors
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));   // catches execution errors

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify against CPU
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-3f) {
            errors++;
            if (errors < 5) printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], h_a[i]+h_b[i]);
        }
    }
    double gbps = (3.0 * bytes) / (ms * 1e-3) / 1e9;
    printf("%s | time: %.3f ms | bandwidth: %.1f GB/s | errors: %d\n",
           errors ? "FAIL" : "PASS", ms, gbps, errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return errors != 0;
}
```

Build and run:

```bash
nvcc -arch=sm_90 -O3 vec_add.cu -o vec_add    # sm_90 for H100, sm_80 for A100
./vec_add
```

Expected: `PASS`. Compare bandwidth to the GPU's peak HBM bandwidth. A single unwarmed launch will underreport; try a warmup launch plus averaging.

---

## 3. Sample kernel 2: tiled matrix multiply (shared memory)

Save as `matmul.cu`:

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                            \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);           \
        exit(1);                                                      \
    }                                                                 \
} while (0)

#define TILE 16

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.f;
        for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

static float time_kernel(void (*launch)(dim3, dim3, const float*, const float*, float*, int),
                         dim3 g, dim3 b, const float* A, const float* B, float* C, int N) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    launch(g, b, A, B, C, N);                 // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < 10; i++) launch(g, b, A, B, C, N);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    return ms / 10.f;
}

static void launch_naive(dim3 g, dim3 b, const float* A, const float* B, float* C, int N) {
    matmul_naive<<<g, b>>>(A, B, C, N); CUDA_CHECK(cudaGetLastError());
}
static void launch_tiled(dim3 g, dim3 b, const float* A, const float* B, float* C, int N) {
    matmul_tiled<<<g, b>>>(A, B, C, N); CUDA_CHECK(cudaGetLastError());
}

int main() {
    const int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    float *ref = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) { hA[i] = (i % 7) * 0.1f; hB[i] = (i % 5) * 0.2f; }

    // CPU reference
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            float s = 0; for (int k = 0; k < N; k++) s += hA[r * N + k] * hB[k * N + c];
            ref[r * N + c] = s;
        }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes)); CUDA_CHECK(cudaMalloc(&dB, bytes)); CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    struct { const char* name; void (*fn)(dim3, dim3, const float*, const float*, float*, int); } ks[] = {
        {"naive", launch_naive}, {"tiled", launch_tiled}};
    for (auto& k : ks) {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        float ms = time_kernel(k.fn, grid, block, dA, dB, dC, N);
        CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        int errs = 0; float maxdiff = 0;
        for (int i = 0; i < N * N; i++) {
            float d = fabsf(hC[i] - ref[i]);
            if (d > maxdiff) maxdiff = d;
            if (d > 1e-2f * fmaxf(1.f, fabsf(ref[i]))) errs++;
        }
        double gflops = 2.0 * N * N * N / (ms * 1e-3) / 1e9;
        printf("%-6s | %.3f ms | %.1f GFLOP/s | max diff %.5f | %s\n",
               k.name, ms, gflops, maxdiff, errs ? "FAIL" : "PASS");
    }
    return 0;
}
```

```bash
nvcc -arch=sm_90 -O3 matmul.cu -o matmul
./matmul
```

Expected: both PASS, tiled should be faster. Then try: `TILE=32`, register blocking (each thread computes a 2x2 or 4x4 output tile), vectorized loads (`float4`), and compare against cuBLAS.

---

## 4. Sample kernel 3: parallel reduction (sum)

Save as `reduce.cu`:

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                            \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);           \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// Warp-level reduction using shuffle
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block-level reduction: shared memory + warp shuffle
__global__ void reduce_sum(const float* input, float* output, int n) {
    __shared__ float warp_sums[32];  // max 32 warps per block (1024/32)

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element (or 0 if out of bounds)
    float val = (gid < n) ? input[gid] : 0.0f;

    // Intra-warp reduction
    val = warp_reduce_sum(val);

    // First thread of each warp writes to shared memory
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // First warp reduces across all warps
    int num_warps = (blockDim.x + 31) / 32;
    val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    // Block result
    if (tid == 0) atomicAdd(output, val);
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    double cpu_sum = 0;
    for (int i = 0; i < N; i++) { h_input[i] = (i % 1000) * 0.001f; cpu_sum += h_input[i]; }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_sum<<<blocks, threads>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    double rel_err = fabs(gpu_sum - cpu_sum) / fabs(cpu_sum);
    double gbps = bytes / (ms * 1e-3) / 1e9;
    printf("CPU sum: %.6f | GPU sum: %.6f | rel error: %.2e | %.3f ms | %.1f GB/s | %s\n",
           cpu_sum, (double)gpu_sum, rel_err, ms, gbps,
           rel_err < 1e-3 ? "PASS" : "FAIL (check FP accumulation order)");

    cudaFree(d_input); cudaFree(d_output);
    free(h_input);
    return 0;
}
```

```bash
nvcc -arch=sm_90 -O3 reduce.cu -o reduce
./reduce
```

Note: the GPU sum may differ slightly from the CPU sum due to floating-point accumulation order. This is expected. The relative error should be small (< 1e-3 for this input).

---

## 5. Sample kernel 4: softmax

Save as `softmax.cu`:

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                            \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);           \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// Online softmax: one block per row, each thread handles multiple elements
// Uses the numerically stable online algorithm (find max and sum in one pass)
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    extern __shared__ float sdata[];  // [0..31] for max, [32..63] for sum

    // Phase 1: find row max (each thread scans a strided range)
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        thread_max = fmaxf(thread_max, in_row[j]);

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) sdata[warp_id] = thread_max;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < num_warps) ? sdata[threadIdx.x] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        if (threadIdx.x == 0) sdata[0] = val;
    }
    __syncthreads();
    float row_max = sdata[0];

    // Phase 2: compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = expf(in_row[j] - row_max);
        out_row[j] = e;  // store intermediate
        thread_sum += e;
    }

    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    if (lane == 0) sdata[warp_id] = thread_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        float val = (threadIdx.x < num_warps) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x == 0) sdata[0] = val;
    }
    __syncthreads();
    float row_sum = sdata[0];

    // Phase 3: normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        out_row[j] /= row_sum;
}

int main() {
    const int ROWS = 1024;
    const int COLS = 4096;
    const size_t bytes = (size_t)ROWS * COLS * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < ROWS * COLS; i++) h_in[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    // CPU reference
    for (int r = 0; r < ROWS; r++) {
        float mx = -FLT_MAX;
        for (int c = 0; c < COLS; c++) mx = fmaxf(mx, h_in[r * COLS + c]);
        float s = 0;
        for (int c = 0; c < COLS; c++) { h_ref[r * COLS + c] = expf(h_in[r * COLS + c] - mx); s += h_ref[r * COLS + c]; }
        for (int c = 0; c < COLS; c++) h_ref[r * COLS + c] /= s;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int shared_bytes = 32 * sizeof(float);  // for warp-level partial results

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel<<<ROWS, threads, shared_bytes>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0;
    int errs = 0;
    for (int i = 0; i < ROWS * COLS; i++) {
        float d = fabsf(h_out[i] - h_ref[i]);
        if (d > max_err) max_err = d;
        if (d > 1e-5f) errs++;
    }
    double gbps = (2.0 * bytes) / (ms * 1e-3) / 1e9;  // read + write
    printf("softmax %dx%d | %.3f ms | %.1f GB/s | max err: %.2e | %s\n",
           ROWS, COLS, ms, gbps, max_err, errs ? "FAIL" : "PASS");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
```

```bash
nvcc -arch=sm_90 -O3 softmax.cu -o softmax
./softmax
```

---

## 6. Tools to check correctness and performance

### 6.1 Correctness (does it work?)

| What | Command |
|---|---|
| Check every CUDA call | `CUDA_CHECK` macro (above), plus `cudaGetLastError()` after each launch |
| Synchronous mode (find which kernel fails) | `CUDA_LAUNCH_BLOCKING=1 ./vec_add` |
| Memory checker (out-of-bounds, leaks) | `compute-sanitizer ./vec_add` |
| Race condition checker | `compute-sanitizer --tool racecheck ./matmul` |
| Uninitialized memory checker | `compute-sanitizer --tool initcheck ./vec_add` |
| Synchronization checker | `compute-sanitizer --tool synccheck ./matmul` |
| Compare to CPU or reference | built into the samples; also try `torch.allclose` against PyTorch |
| Debug crashes or hangs | `cuda-gdb ./vec_add` then `run`, `bt`, `info threads` |
| Compile with debug info | `nvcc -arch=sm_90 -G -g -O0 vec_add.cu -o vec_add_dbg` |
| Compiler warnings | add `-Xcompiler -Wall,-Wextra` to nvcc |

compute-sanitizer quick start:

```bash
# Basic memory check (default tool = memcheck)
compute-sanitizer ./vec_add

# Race condition detection
compute-sanitizer --tool racecheck ./matmul

# All tools in one pass (slower)
compute-sanitizer --tool memcheck --tool racecheck --tool synccheck --tool initcheck ./matmul
```

cuda-gdb quick start:

```bash
nvcc -arch=sm_90 -G -g vec_add.cu -o vec_add_dbg
cuda-gdb ./vec_add_dbg
(gdb) break vec_add
(gdb) run
(gdb) info cuda threads              # list GPU threads
(gdb) cuda thread (0,0,0)            # switch to a specific thread
(gdb) print i                        # inspect thread-local variable
(gdb) bt
(gdb) continue
```

### 6.2 Hardware and system health

```bash
nvidia-smi                           # overall GPU status
nvidia-smi dmon -s pucvmet           # live monitoring (power, utilization, clock, VRAM, memory, ECC, temp)
watch -n 1 nvidia-smi               # auto-refresh in another terminal
nvidia-smi topo --matrix             # NVLink / PCIe topology between GPUs
nvidia-smi nvlink --status           # NVLink status per GPU

# Power and clock management (for consistent benchmarking)
sudo nvidia-smi -pm 1                          # enable persistence mode
sudo nvidia-smi -lgc <min_clk>,<max_clk>       # lock GPU clock (e.g. 1980,1980)
sudo nvidia-smi -rgc                           # reset GPU clocks
```

### 6.3 Profiling (is it fast, and why or why not?)

NVIDIA has two main profilers: **Nsight Systems** (system-wide timeline) and **Nsight Compute** (per-kernel deep dive).

```bash
# Nsight Systems: timeline of kernels, memcpys, API calls, CPU activity
nsys profile -o vec_add_profile ./vec_add
nsys stats vec_add_profile.nsys-rep          # text summary
nsys-ui vec_add_profile.nsys-rep             # GUI timeline (needs display)

# Nsight Compute: detailed kernel analysis (one kernel at a time)
ncu ./matmul                                  # basic metrics for all kernels
ncu --set full -o matmul_profile ./matmul     # comprehensive analysis
ncu-ui matmul_profile.ncu-rep                 # GUI (needs display)

# Roofline analysis
ncu --set roofline -o matmul_roofline ./matmul
ncu-ui matmul_roofline.ncu-rep

# Target a specific kernel
ncu --kernel-name matmul_tiled --launch-count 1 --set full ./matmul

# Key metrics to look at
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed \
./matmul
```

Things to look for in profiler output:
- **Kernel duration**: compare naive vs tiled.
- **Memory throughput vs peak**: vec_add should approach peak bandwidth.
- **Compute throughput vs peak**: matmul should improve with tiling.
- **Occupancy**: limited by registers? shared memory? block size?
- **Roofline position**: memory bound or compute bound?
- **Stall reasons**: memory dependency? execution dependency? synchronization?
- **Warp divergence**: branch efficiency metric.
- **Bank conflicts**: shared memory bank conflict count.

### 6.4 Inspect what the compiler generated

```bash
# Resource usage per kernel (registers, shared memory, spills)
nvcc -arch=sm_90 -O3 --ptxas-options=-v matmul.cu -o matmul

# Generate PTX (virtual ISA, human-readable)
nvcc -arch=sm_90 -O3 -ptx matmul.cu       # outputs matmul.ptx

# Generate SASS (actual GPU assembly)
nvcc -arch=sm_90 -O3 matmul.cu -o matmul
cuobjdump -sass matmul                     # disassemble the binary

# Generate both PTX and SASS with source correlation
nvcc -arch=sm_90 -O3 -lineinfo matmul.cu -o matmul
cuobjdump -sass -ptx matmul

# List CUDA functions in binary
cuobjdump -symbols matmul

# Extract resource info
cuobjdump -res-usage matmul
```

Check for register spills (`local memory` in the resource report or `STL`/`LDL` in SASS). Spills usually mean a slow kernel.

### 6.5 Occupancy analysis

```bash
# Programmatic occupancy query
nvcc -arch=sm_90 -O3 occupancy_check.cu -o occupancy_check
```

Add this to your kernel code to query occupancy at runtime:

```cpp
int minGridSize, optBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optBlockSize, vec_add, 0, 0);
printf("Optimal block size: %d, min grid size: %d\n", optBlockSize, minGridSize);

// Or check occupancy for a specific block size:
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vec_add, 256, 0);
printf("Max active blocks per SM with 256 threads: %d\n", maxActiveBlocks);
```

---

## 7. Streams and concurrency

Streams enable overlap of kernel execution with memory transfers and overlap of multiple kernels.

### Key concepts

```cpp
// Create streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Launch kernels on different streams (can run concurrently)
kernel_A<<<grid, block, 0, stream1>>>(d_a, N);
kernel_B<<<grid, block, 0, stream2>>>(d_b, N);

// Async memory transfers (host memory must be pinned!)
float *h_pinned;
cudaMallocHost(&h_pinned, bytes);  // pinned (page-locked) host memory
cudaMemcpyAsync(d_a, h_pinned, bytes, cudaMemcpyHostToDevice, stream1);

// Synchronize
cudaStreamSynchronize(stream1);   // wait for one stream
cudaDeviceSynchronize();          // wait for all streams

// Destroy streams
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### Overlap pattern (pipeline)

```cpp
// Double-buffered pipeline: copy chunk i while computing chunk i-1
for (int i = 0; i < num_chunks; i++) {
    int s = i % 2;  // alternate between 2 streams
    cudaMemcpyAsync(d_buf[s], h_buf + i*chunk, chunk_bytes, cudaMemcpyHostToDevice, streams[s]);
    kernel<<<grid, block, 0, streams[s]>>>(d_buf[s], chunk);
    cudaMemcpyAsync(h_out + i*chunk, d_buf[s], chunk_bytes, cudaMemcpyDeviceToHost, streams[s]);
}
```

### Events for inter-stream synchronization

```cpp
cudaEvent_t event;
cudaEventCreate(&event);

// Stream 1 records an event after it finishes
kernel_A<<<grid, block, 0, stream1>>>(...);
cudaEventRecord(event, stream1);

// Stream 2 waits for that event before starting
cudaStreamWaitEvent(stream2, event);
kernel_B<<<grid, block, 0, stream2>>>(...);  // guaranteed to start after kernel_A
```

Profile streams with Nsight Systems to verify overlap:
```bash
nsys profile -o streams_profile ./my_streams_app
nsys-ui streams_profile.nsys-rep    # look at the timeline for overlapping bars
```

---

## 8. Memory management patterns

### Pinned (page-locked) memory

```cpp
float *h_pinned;
cudaMallocHost(&h_pinned, bytes);  // pinned, enables async transfers
// ... use it ...
cudaFreeHost(h_pinned);
```

Pinned memory is faster for host-device transfers and required for `cudaMemcpyAsync`. But it's a limited resource — don't pin everything.

### Unified memory

```cpp
float *data;
cudaMallocManaged(&data, bytes);  // accessible from both host and device

// Use from host
for (int i = 0; i < N; i++) data[i] = i;

// Use from device (driver handles page migration)
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Read results on host (pages migrate back automatically)
printf("%f\n", data[0]);

cudaFree(data);
```

**Prefetch hints** (avoid page faults during kernel execution):
```cpp
int device;
cudaGetDevice(&device);
cudaMemPrefetchAsync(data, bytes, device);      // prefetch to GPU
cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId); // prefetch to CPU

cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, device);  // hint: mostly read
cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, device);
```

### Memory pools (CUDA 11.2+)

```cpp
// Allocate from the default memory pool (fast, cached)
float *d_temp;
cudaMallocAsync(&d_temp, bytes, stream);
// ... use in same stream ...
cudaFreeAsync(d_temp, stream);  // returned to pool, not freed immediately
```

---

## 9. Tensor Core usage (WMMA)

For interview/practice purposes, here's a simple WMMA (Warp Matrix Multiply-Accumulate) example.

Save as `wmma_gemm.cu`:

```cpp
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define CUDA_CHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                            \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);           \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// WMMA dimensions (16x16x16 is the most common starting point)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Simple WMMA GEMM: C = A * B (half precision input, float accumulation)
// Each warp computes one 16x16 output tile
__global__ void wmma_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;  // wrong
    int warpN = blockIdx.x * WMMA_N;

    // Better: use warp ID directly
    int warp_id = (blockIdx.x + blockIdx.y * gridDim.x);
    int warp_row = (warp_id / (N / WMMA_N)) * WMMA_M;
    int warp_col = (warp_id % (N / WMMA_N)) * WMMA_N;

    if (warp_row >= M || warp_col >= N) return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}

int main() {
    const int M = 256, N = 256, K = 256;

    // Allocate and init (half precision)
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((i % 7) * 0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half((i % 5) * 0.1f);

    half *d_A, *d_B; float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Launch: one warp per output tile
    // Grid: (N/WMMA_N, M/WMMA_M), Block: (32, 1) — one warp per block for simplicity
    dim3 grid(N / WMMA_N, M / WMMA_M);
    dim3 block(32, 1);

    wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements against CPU
    int errs = 0;
    for (int r = 0; r < M && errs < 5; r++) {
        for (int c = 0; c < N && errs < 5; c++) {
            float ref = 0;
            for (int k = 0; k < K; k++) ref += __half2float(h_A[r * K + k]) * __half2float(h_B[k * N + c]);
            if (fabsf(h_C[r * N + c] - ref) > 1.0f) {
                printf("Mismatch at (%d,%d): got %f, expected %f\n", r, c, h_C[r*N+c], ref);
                errs++;
            }
        }
    }
    printf("WMMA GEMM %dx%dx%d: %s\n", M, N, K, errs ? "FAIL" : "PASS");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

```bash
nvcc -arch=sm_80 -O3 wmma_gemm.cu -o wmma_gemm   # sm_70+ for WMMA
./wmma_gemm
```

Note: this is a minimal example to show the WMMA API. Production Tensor Core code (cuBLAS, CUTLASS) uses tiling, pipelining, and shared memory staging for orders-of-magnitude better performance.

---

## 10. Triton on CUDA (quick check)

Save as `triton_add.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1 << 24
x = torch.rand(n, device="cuda")
y = torch.rand(n, device="cuda")
out = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
print("match:", torch.allclose(out, x + y))

ms = triton.testing.do_bench(lambda: add_kernel[grid](x, y, out, n, BLOCK=1024))
print(f"{ms:.3f} ms, {3 * n * 4 / (ms * 1e-3) / 1e9:.1f} GB/s")
```

```bash
python3 triton_add.py
```

---

## 11. PyTorch checks

```bash
# Quick GEMM benchmark
python3 - <<'EOF'
import torch, time
N = 4096
a = torch.randn(N, N, device="cuda", dtype=torch.float16)
b = torch.randn(N, N, device="cuda", dtype=torch.float16)
for _ in range(5): a @ b
torch.cuda.synchronize()
t = time.time()
for _ in range(50): a @ b
torch.cuda.synchronize()
dt = (time.time() - t) / 50
print(f"{2*N**3/dt/1e12:.1f} TFLOP/s (fp16 GEMM via cuBLAS)")
EOF

# Profile PyTorch ops
python3 - <<'EOF'
import torch
from torch.profiler import profile, ProfilerActivity
a = torch.randn(4096, 4096, device="cuda")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    a @ a
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=5))
EOF
```

---

## 12. Hopper-specific features (SM 9.0)

These are advanced features for H100 and beyond:

### Thread Block Clusters

```cpp
// Clusters: groups of blocks guaranteed to be co-scheduled on the same GPC
// Enables distributed shared memory across blocks in the cluster
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float* data) {
    // Can access shared memory of other blocks in the cluster
    // via cluster.map_shared_rank()
}
```

### TMA (Tensor Memory Accelerator)

```cpp
// TMA offloads address calculation and data movement to a dedicated hardware unit
// Instead of threads computing addresses, TMA does bulk copies between
// global memory <-> shared memory with a single instruction
// Primarily used via CUTLASS 3.x or PTX
```

### Asynchronous copy (cp.async)

```cpp
// Available since Ampere (SM 80), refined in Hopper
// Copy global -> shared without going through registers
#include <cuda/pipeline>

__shared__ float smem[256];
cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
pipe.producer_acquire();
cuda::memcpy_async(&smem[threadIdx.x], &global[idx], sizeof(float), pipe);
pipe.producer_commit();
pipe.consumer_wait();
// smem[threadIdx.x] is now ready
```

---

## 13. Common pitfalls and interview traps

### 1. Forgetting error checking
Every CUDA API call can fail silently. Always use the `CUDA_CHECK` macro. Kernel launches are asynchronous — `cudaGetLastError()` only catches configuration errors, not runtime errors. You need `cudaDeviceSynchronize()` or event synchronization to catch execution errors.

### 2. Race conditions in shared memory
```cpp
// BUG: threads may read stale values
sdata[tid] = input[gid];
float val = sdata[tid + 1];  // WRONG: thread tid+1 may not have written yet

// FIX: sync before reading
sdata[tid] = input[gid];
__syncthreads();
float val = sdata[tid + 1];  // OK: all threads have written
```

### 3. Warp divergence in reductions
Use the contiguous-threads-first pattern (see warp divergence section above).

### 4. Not warming up before benchmarking
The first kernel launch includes lazy initialization overhead (context creation, module loading). Always do warmup launches before timing.

### 5. Measuring wall-clock time instead of GPU time
```cpp
// WRONG: includes launch overhead, other CPU work
auto start = std::chrono::high_resolution_clock::now();
kernel<<<grid, block>>>(args);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();

// RIGHT: GPU events measure only GPU execution
cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

### 6. Misunderstanding occupancy
High occupancy is not always better. A kernel using 255 registers per thread has low occupancy but might be faster than one using 32 registers if it avoids memory stalls through register reuse. Occupancy is a tool for hiding latency, not a goal in itself.

### 7. Using global atomics when block-level reduction suffices
Atomic operations to global memory serialize across all SMs. Do block-level reduction in shared memory first, then one atomic per block (as in the reduction kernel above).

### 8. Ignoring alignment
`cudaMalloc` returns 256-byte aligned pointers, but if you index into the middle of an array, the effective alignment may be wrong for vector loads. Use `__align__(16)` for shared memory arrays when doing `float4` loads.

---

## 14. Compute capability quick reference

| CC | Architecture | Key GPU | Notable Features |
|---|---|---|---|
| 7.0 | Volta | V100 | Tensor Cores (1st gen), independent thread scheduling |
| 7.5 | Turing | RTX 2080 | INT8/INT4 Tensor Cores, RT cores |
| 8.0 | Ampere | A100 | TF32 Tensor Cores, async copy (cp.async), 3rd gen NVLink |
| 8.6 | Ampere | RTX 3090 | Same as 8.0 but with different SM count/config |
| 8.9 | Ada Lovelace | RTX 4090 | FP8 Tensor Cores, shader execution reordering |
| 9.0 | Hopper | H100 | Thread Block Clusters, TMA, DPX, FP8 training, 4th gen Tensor Cores |

Key limits by compute capability:

| Resource | 7.x | 8.x | 9.0 |
|---|---|---|---|
| Max threads/block | 1024 | 1024 | 1024 |
| Max threads/SM | 2048 | 2048 | 2048 |
| Max blocks/SM | 32 | 32 | 32 |
| Registers/SM | 65536 | 65536 | 65536 |
| Max shared mem/SM | 96 KB | 164 KB | 228 KB |
| Max shared mem/block | 48 KB (96 KB opt-in) | 164 KB (opt-in) | 228 KB (opt-in) |

To opt in to larger shared memory per block:
```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1024);
kernel<<<grid, block, 164 * 1024>>>(args);
```

---

## 15. Practice ladder (do in this order)

1. **vec_add**: change `N` and block size (32, 64, 128, 256, 512, 1024). Note bandwidth vs block size. Compute effective bandwidth as a fraction of peak.
2. **Break it on purpose**: remove the `if (i < n)` guard and run with `compute-sanitizer`. Launch with 2048 threads per block and read the error message.
3. **matmul**: compare naive vs tiled. Profile both with `ncu`. Use `ncu --set roofline` to see where each sits.
4. **Optimize matmul**: `TILE=32`, register blocking (2x2, 4x4 output per thread), vectorized loads (`float4`), then compare against cuBLAS.
5. **reduction**: run the sum kernel. Try different block sizes. Add a multi-pass version (kernel reduces to partial sums, second kernel reduces those).
6. **softmax**: verify against PyTorch (`torch.softmax`). Profile and compare bandwidth.
7. **WMMA**: run the Tensor Core matmul. Compare GFLOP/s against your FP32 tiled matmul and against cuBLAS FP16.
8. **Write a layernorm kernel** and verify against `torch.nn.LayerNorm`.
9. **Streams**: implement the double-buffered pipeline pattern. Profile with `nsys` to verify overlap.
10. **Port a kernel to Triton** and compare the runtime against your CUDA version.

---

## 16. cuBLAS comparison (how far are you from the library?)

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;
// C = alpha * A * B + beta * C
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
// Note: cuBLAS uses column-major, so arguments are "reversed" for row-major

cublasDestroy(handle);
```

Link with: `nvcc -lcublas ...`

For FP16 with Tensor Cores:
```cpp
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             N, N, N, &alpha,
             dB, CUDA_R_16F, N,
             dA, CUDA_R_16F, N,
             &beta, dC, CUDA_R_32F, N,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

---

## 17. Multi-GPU basics

```cpp
int num_gpus;
cudaGetDeviceCount(&num_gpus);

// Simple pattern: split work across GPUs
for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice(g);
    cudaMalloc(&d_buf[g], chunk_bytes);
    cudaMemcpyAsync(d_buf[g], h_buf + g * chunk, chunk_bytes, cudaMemcpyHostToDevice, streams[g]);
    kernel<<<grid, block, 0, streams[g]>>>(d_buf[g], chunk);
}

// Peer-to-peer access (NVLink/PCIe)
int can_access;
cudaDeviceCanAccessPeer(&can_access, 0, 1);  // can GPU 0 access GPU 1's memory?
if (can_access) {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);         // GPU 0 can now read/write GPU 1's memory
    // Direct GPU-to-GPU copy
    cudaMemcpyPeerAsync(d_buf_gpu0, 0, d_buf_gpu1, 1, bytes, stream);
}
```

Check topology:
```bash
nvidia-smi topo --matrix
```

---

## 18. NCCL for collective communications

**NCCL** (NVIDIA Collective Communications Library, pronounced "nickel") is the standard library for multi-GPU and multi-node communication. It implements collective operations optimized for NVIDIA's interconnect topology (NVLink, NVSwitch, InfiniBand).

### Collective operations

These are the fundamental communication patterns. Every distributed training framework (PyTorch DDP, DeepSpeed, Megatron) uses these under the hood.

```
AllReduce — most common in data-parallel training (gradient sync)
Each GPU has a value. After AllReduce, every GPU has the sum (or max, min, etc.)

Before:  GPU0=[1]  GPU1=[2]  GPU2=[3]  GPU3=[4]
After:   GPU0=[10] GPU1=[10] GPU2=[10] GPU3=[10]  (sum = 10)
```

```
Broadcast — one GPU sends its data to all others

Before:  GPU0=[DATA]  GPU1=[?]  GPU2=[?]  GPU3=[?]
After:   GPU0=[DATA]  GPU1=[DATA]  GPU2=[DATA]  GPU3=[DATA]
```

```
Reduce — all GPUs contribute, only one GPU gets the result

Before:  GPU0=[1]  GPU1=[2]  GPU2=[3]  GPU3=[4]
After:   GPU0=[10] GPU1=[2]  GPU2=[3]  GPU3=[4]  (only root has sum)
```

```
AllGather — each GPU has a piece, after AllGather every GPU has all pieces

Before:  GPU0=[A]  GPU1=[B]  GPU2=[C]  GPU3=[D]
After:   GPU0=[ABCD]  GPU1=[ABCD]  GPU2=[ABCD]  GPU3=[ABCD]
```

```
ReduceScatter — reduce + scatter: reduce across GPUs, each gets a different chunk

Before:  GPU0=[a0,a1,a2,a3]  GPU1=[b0,b1,b2,b3]  GPU2=[c0,c1,c2,c3]  GPU3=[d0,d1,d2,d3]
After:   GPU0=[a0+b0+c0+d0]  GPU1=[a1+b1+c1+d1]  GPU2=[a2+b2+c2+d2]  GPU3=[a3+b3+c3+d3]
```

```
AllToAll — each GPU sends a different piece to each other GPU (transpose)

Before:  GPU0=[a0,a1,a2,a3]  GPU1=[b0,b1,b2,b3]
After:   GPU0=[a0,b0,c0,d0]  GPU1=[a1,b1,c1,d1]  (each gets its "column")
```

### How NCCL implements AllReduce

NCCL uses **ring** and **tree** algorithms depending on message size and topology.

**Ring AllReduce (large messages):**
GPUs are arranged in a logical ring. The operation happens in two phases:

```
Phase 1: ReduceScatter (N-1 steps)
Each GPU sends a chunk to the next GPU in the ring, which adds it to its own chunk.
After N-1 steps, each GPU has the fully reduced version of one chunk.

  Step 0: GPU0 → GPU1 → GPU2 → GPU3 → GPU0
  Step 1: shift and accumulate
  Step 2: shift and accumulate
  ...

Phase 2: AllGather (N-1 steps)
Each GPU sends its fully reduced chunk around the ring.
After N-1 steps, every GPU has all chunks.
```

Total data transferred per GPU: `2 * (N-1)/N * data_size` — nearly 2x the data regardless of GPU count. This is **bandwidth-optimal**.

**Tree AllReduce (small messages):**
Uses a binary tree for lower latency (fewer steps: log₂N instead of 2(N-1)):

```
        GPU0 (root)
       /    \
    GPU1    GPU2
    /
  GPU3

Reduce up the tree → Broadcast down the tree
Fewer steps but less bandwidth utilization
```

NCCL automatically picks the best algorithm based on message size and topology.

### Code example

```cpp
#include <nccl.h>

ncclComm_t comms[num_gpus];
ncclCommInitAll(comms, num_gpus, dev_list);

// AllReduce across GPUs
for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice(g);
    ncclAllReduce(d_send[g], d_recv[g], count, ncclFloat, ncclSum, comms[g], streams[g]);
}

// Synchronize all
for (int g = 0; g < num_gpus; g++) {
    cudaSetDevice(g);
    cudaStreamSynchronize(streams[g]);
}
```

Link with: `nvcc -lnccl ...`

### Where each collective is used in training

| Collective | Where it's used |
|---|---|
| **AllReduce** | Data parallelism — sync gradients across GPUs after backward pass |
| **AllGather** | FSDP (Fully Sharded Data Parallel) — gather full parameters before forward pass |
| **ReduceScatter** | FSDP — shard gradients after backward pass; ZeRO optimizer stage 2 |
| **AllToAll** | Expert parallelism in MoE (Mixture of Experts) — route tokens to experts |
| **Broadcast** | Distribute model weights from rank 0 at initialization |
| **Reduce** | Collect metrics/losses to a single rank for logging |

### Debugging NCCL

```bash
# Verbose NCCL logging
NCCL_DEBUG=INFO ./my_training_script

# Show which algorithm/protocol NCCL chose
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL ./my_training_script

# Common issues:
# - NCCL timeout → GPU hang, usually a deadlock (not all ranks call the collective)
# - NCCL_P2P_DISABLE=1 → fallback to shared memory if NVLink has issues
```

---

## 19. Save your work

```bash
cd ~/cuda-practice
git init && git add . && git commit -m "CUDA practice kernels"
git remote add origin git@github.com:<you>/cuda-practice.git
git push -u origin main
```