# ROCm Practice Guide (AMD Developer Cloud, MI300X)

Note: tool names and flags shift between ROCm versions (especially the profilers). If a command is not found, run `<tool> --help` or check `ls /opt/rocm/bin`.

---

## GPU architecture primer: AMD Instinct MI300X (CDNA 3, gfx942)

Numbers are from AMD's public specs as I know them. Verify on your machine with the commands in the next subsection.

### Big picture

- **Chiplet design**: 8 XCDs (compute dies) stacked on 4 IODs (I/O dies), connected by Infinity Fabric.
- **304 Compute Units (CUs)**: 38 CUs per XCD x 8 XCDs.
- **192 GB HBM3** on 8 stacks, roughly **5.3 TB/s** peak memory bandwidth.
- **256 MB Infinity Cache** (last-level cache) shared across the package, plus **4 MB L2 per XCD**.
- **Peak compute (approx, dense)**: about 1.3 PFLOP/s FP16/BF16, about 2.6 PFLOP/s FP8, about 163 TFLOP/s FP32 (vector and matrix), about 81.7 TFLOP/s FP64 vector (about 163 matrix).

### Visual map (ASCII diagrams, three zoom levels)

Figures are the MI300X numbers used elsewhere in this guide. The SIMD detail is a simplified view.

**Zoom 1: the whole package.** XCDs (compute dies) are stacked on top of the IODs (base dies). HBM3 stacks sit beside the IODs. Each IOD holds the same four blocks: memory controllers, Infinity Cache (about 64 MB each, 256 MB total), Infinity Fabric, and external links.

```
+------+ +----------------------------+ +----------------------------+ +------+
| HBM3 | | IOD 0                      | | IOD 1                      | | HBM3 |
| 24GB | |  +-------+    +-------+    | |  +-------+    +-------+    | | 24GB |
+------+ |  | XCD   |    | XCD   |    | |  | XCD   |    | XCD   |    | +------+
+------+ |  | 38 CU |    | 38 CU |    | |  | 38 CU |    | 38 CU |    | +------+
| HBM3 | |  +-------+    +-------+    | |  +-------+    +-------+    | | HBM3 |
| 24GB | | [Mem ctrl] [Inf. Cache]    | | [Mem ctrl] [Inf. Cache]    | | 24GB |
+------+ | [Fabric]   [PCIe, IF links]| | [Fabric]   [PCIe, IF links]| +------+
         +----------------------------+ +----------------------------+
         <======== Infinity Fabric links all four IODs ========>
+------+ +----------------------------+ +----------------------------+ +------+
| HBM3 | | IOD 2                      | | IOD 3                      | | HBM3 |
| 24GB | |  +-------+    +-------+    | |  +-------+    +-------+    | | 24GB |
+------+ |  | XCD   |    | XCD   |    | |  | XCD   |    | XCD   |    | +------+
+------+ |  | 38 CU |    | 38 CU |    | |  | 38 CU |    | 38 CU |    | +------+
| HBM3 | |  +-------+    +-------+    | |  +-------+    +-------+    | | HBM3 |
| 24GB | | [Mem ctrl] [Inf. Cache]    | | [Mem ctrl] [Inf. Cache]    | | 24GB |
+------+ | [Fabric]   [PCIe, IF links]| | [Fabric]   [PCIe, IF links]| +------+
         +----------------------------+ +----------------------------+

Totals: 8 XCDs x 38 CUs = 304 CUs | 4 IODs | 8 HBM3 stacks x 24 GB = 192 GB
```

**Zoom 2: inside one XCD.** 38 CUs share one 4 MB L2 cache.

```
+--------------------------------------------------+
| XCD (compute die)                                |
|  [CU][CU][CU][CU][CU][CU][CU][CU][CU][CU]        |
|  [CU][CU][CU][CU][CU][CU][CU][CU][CU][CU]        |
|  [CU][CU][CU][CU][CU][CU][CU][CU][CU][CU]        |
|  [CU][CU][CU][CU][CU][CU][CU][CU]   (38 total)   |
|  +--------------------------------------------+  |
|  | L2 cache, 4 MB, shared by all 38 CUs       |  |
|  +--------------------------------------------+  |
+--------------------------------------------------+
```

**Zoom 3: inside one CU.** 4 SIMD units, each with vector and matrix (MFMA) hardware, plus LDS and an L1 cache. Threads run in wavefronts of 64.

```
+-------------------------------------------------------+
| CU (compute unit): runs 64-thread wavefronts          |
|  +-------------------------------------------------+  |
|  | Wavefront scheduler                             |  |
|  +-------------------------------------------------+  |
|  +----------+ +----------+ +----------+ +----------+  |
|  | SIMD 0   | | SIMD 1   | | SIMD 2   | | SIMD 3   |  |
|  | vector   | | vector   | | vector   | | vector   |  |
|  | + MFMA   | | + MFMA   | | + MFMA   | | + MFMA   |  |
|  +----------+ +----------+ +----------+ +----------+  |
|  +-----------------------+ +-----------------------+  |
|  | LDS, 64 KB            | | L1 vector cache       |  |
|  | shared memory         | | per-CU cache          |  |
|  +-----------------------+ +-----------------------+  |
+-------------------------------------------------------+
```

Path from a CU to memory: `CU -> L1 -> L2 (per XCD) -> Infinity Cache (on the IODs) -> HBM`.

### XCD vs IOD vs L2 (what each term means)

**XCD (Accelerator Complex Die)** is a physical compute chiplet. It contains:
- 38 CUs (each CU has its own L1 cache, LDS, and matrix cores)
- one shared **4 MB L2 cache** used by all 38 of its CUs

MI300X has 8 XCDs, so it has 8 separate L2 caches (8 x 4 MB = 32 MB total). They are **not shared** with each other.

**L2 is a cache, not a chip.** It lives inside an XCD. "The XCD's L2" means the L2 on that particular XCD.

**IOD (I/O Die)** is the base chiplet that the XCDs sit on. It has no compute units and never runs your kernel. MI300X has 4 IODs, and each holds:
- **Memory controllers** for the HBM3 stacks (2 stacks per IOD, 8 total)
- **Infinity Cache** (the 256 MB last-level cache, about 64 MB per IOD), shared by all XCDs
- **Infinity Fabric**, the on-package network linking the 4 IODs so any XCD can reach any HBM stack or cache slice
- **External links**: PCIe to the host CPU, and Infinity Fabric links to other GPUs in an 8-GPU server

Physical layout (2 XCDs stacked on each IOD, HBM beside the IODs):

```
 XCD  XCD        <- compute (2 per IOD)
 [   IOD   ]     <- Infinity Cache, memory controllers, fabric
   HBM HBM       <- memory next to the IOD
```

Data path from a kernel to memory:

```
CU -> L1 (per CU) -> L2 (per XCD) -> Infinity Cache (on IODs, shared) -> HBM
```

Why AMD splits it this way: compute dies benefit from the newest, most expensive process node, while I/O circuitry does not, so I/O goes on an older, cheaper node and the GPU is built from smaller dies instead of one huge one.

Why it matters for kernels:
- **L2 is per XCD.** Workgroups are spread across XCDs (often round-robin). Two workgroups that read the same data but land on different XCDs each fetch it into their own L2. If they run on the same XCD, the second can hit in the shared L2. Advanced tuning remaps workgroup IDs (for example, swizzling program IDs in Triton) so workgroups that share data run on the same XCD.
- **Memory is not uniformly close.** An XCD reaching an HBM stack on its own IOD takes a shorter path than reaching one on another IOD. How much this shows up depends on the GPU's partition mode.
- For your first kernels you can ignore all of this. It matters once you are tuning GEMM for the last bit of performance.

### Inside one Compute Unit

- **4 SIMD units** per CU, each executing a 64-wide **wavefront**.
- **Wavefront size = 64 threads** (NVIDIA warp = 32). This changes reductions, shuffles, and ballot masks (64-bit masks).
- **Matrix cores** execute MFMA instructions (matrix fused multiply-add) for FP16/BF16/FP8/INT8/FP32/FP64 tiles.
- **LDS (shared memory)**: 64 KB per CU, organized in banks. Watch for bank conflicts.
- **Vector L1 cache**: small per-CU cache (32 KB class), then L2, then Infinity Cache, then HBM.
- **Register files**: VGPRs (per-lane vector registers) and SGPRs (scalar, shared across the wave). On CDNA 3 the vector register file is unified with the accumulation registers used by MFMA. More registers per thread means fewer waves per SIMD (lower occupancy).
- **Max threads per block**: 1024.

### CUDA to ROCm vocabulary

| CUDA | AMD / HIP |
|---|---|
| SM | CU (Compute Unit) |
| Warp (32) | Wavefront (64 on MI300X) |
| Shared memory | LDS (Local Data Share) |
| Tensor Cores | Matrix Cores (MFMA instructions) |
| `__shfl_*` | `__shfl_*` (HIP, 64-wide) / DPP ops |
| nvcc / PTX / SASS | hipcc / LLVM IR / GCN-CDNA ISA (gfx942) |
| cuBLAS / cuDNN | rocBLAS, hipBLASLt / MIOpen |
| Nsight Compute / Systems | rocprof-compute / rocprofv3 (and rocprof-sys) |
| compute-sanitizer | no direct equivalent; use rocgdb, ASAN, and error checking |
| cuda-gdb | rocgdb |

### What the architecture means for your kernels

1. **Block size should be a multiple of 64** (a 32-thread block wastes half a wavefront).
2. **Memory coalescing**: have consecutive lanes read consecutive addresses; vector loads (`float4`) help a lot.
3. **Occupancy**: check VGPR count in the resource-usage report (section 4.4). High VGPR use limits waves per SIMD.
4. **LDS**: tile your data, pad shared arrays to avoid bank conflicts (e.g. `[TILE][TILE+1]`).
5. **Use MFMA** for real GEMM performance (via rocBLAS/hipBLASLt, Triton `tl.dot`, or intrinsics like `__builtin_amdgcn_mfma_f32_16x16x16f16`). Plain FMA loops cannot reach matrix-core peak.
6. **Chiplet effects**: work spread across XCDs has separate L2s, so data placement and workgroup-to-XCD mapping can affect cache hit rates. This is an advanced tuning topic (swizzling program IDs in Triton, for example).
7. **Roofline thinking**: vec_add is bandwidth bound (compare to about 5.3 TB/s), large GEMM should become compute bound (compare to peak TFLOP/s for the dtype).

### Warp (wavefront) divergence

**What it is:** A wavefront (64 threads on AMD, 32 on NVIDIA) must execute the same instruction in lockstep. When threads in a wavefront take different sides of an `if/else` or loop with varying trip counts, the hardware serializes both paths — threads not on the active path are masked off and do no useful work. This is warp divergence.

**Why it happens:**
- **Data-dependent branching**: `if (A[tid] > 0)` where some elements are positive and others aren't.
- **Thread-ID-dependent branching**: `if (threadIdx.x % 2 == 0)` splits every wavefront 50/50.
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
// active threads are contiguous → no divergence until wavefront partially empties
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)                      // <-- first s threads are active, rest are off
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

**How to handle / avoid it:**
1. **Restructure branches so threads diverge across wavefronts, not within them.** If threads 0–63 all take the same path and threads 64–127 take the other, there is zero divergence — each wavefront is uniform.
2. **Use predication / branchless math**: replace `if (x > 0) y = x; else y = 0;` with `y = max(x, 0);` or `y = x * (x > 0);`. The compiler often does this automatically for simple cases.
3. **Sort or bucket data** so threads in the same wavefront hit the same branch. Common in particle simulations and sparse workloads.
4. **Reduce-then-compact**: instead of branching per-thread, compact the active elements into a dense buffer (stream compaction) and launch a second kernel on just those.
5. **Check with the profiler**: `rocprof-compute` / `rocprof` reports branch divergence metrics. Look for high "branch taken" vs "branch not taken" divergence within wavefronts.

### Roofline model (AI = arithmetic intensity, not artificial intelligence)

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

- **Sloped part (left):** memory bound. Performance = bandwidth x AI.
- **Flat part (right):** compute bound. Performance is capped by the GPU's peak FLOP/s.
- **Ridge point:** peak FLOP/s divided by peak bandwidth. Kernels left of it are memory bound, kernels right of it are compute bound.
- Each kernel is one dot. The farther below the roof it sits, the more performance is unused, and which part of the roof is above it tells you what to fix (memory access patterns vs. compute efficiency).

Approximate MI300X numbers (from the spec figures above, so verify them):

| Case | Calculation | Result |
|---|---|---|
| vec_add | 1 FLOP per 12 bytes (2 loads + 1 store of 4-byte floats) | AI about 0.08, far left, memory bound |
| Ridge point, FP16 matrix | about 1,300 TFLOP/s / about 5.3 TB/s | about 250 FLOP/byte |
| Ridge point, FP32 | about 163 TFLOP/s / about 5.3 TB/s | about 30 FLOP/byte |

Estimating AI for the sample kernels:
- **vec_add:** about 0.08 FLOP/byte, so the best possible result is to approach peak memory bandwidth. Compare your printed GB/s to about 5.3 TB/s.
- **matmul_naive:** with no data reuse, each output needs 2N FLOPs and reads about 8N bytes, so AI is about 0.25 FLOP/byte (caches help in practice, so measured numbers differ).
- **matmul_tiled:** each value loaded into shared memory is reused TILE times, so AI is roughly TILE/4 FLOP/byte (about 4 for TILE=16, about 8 for TILE=32). Bigger tiles and register blocking move the dot to the right.
- Even tiled FP32 code like this sits well left of the ridge points above. Reaching the compute roof needs matrix cores (MFMA) and much more reuse, which is what rocBLAS/hipBLASLt do.

Generate the roofline plot for your own kernel:

```bash
rocprof-compute profile -n matmul_run -- ./matmul
rocprof-compute analyze -p workloads/matmul_run/MI300
```

Look for the roofline section in the analyze output (or its HTML/GUI view if available in your version), find where naive and tiled land, and note whether each is under the memory roof or the compute roof.

Exercise: for each kernel, compute AI by hand, predict its position on the roofline, then compare with what the profiler shows.

### Instruction scheduling and latency hiding

The GPU hides memory and instruction latency by running many wavefronts concurrently on each CU. When one wavefront stalls (waiting for a memory load, for example), the scheduler switches to another ready wavefront — no context-switch cost because each wavefront has its own registers.

**How wavefront scheduling works (CDNA 3 / MI300X):**
- Each CU has **4 SIMD units**, each with its own scheduler.
- Each SIMD can manage multiple wavefronts (up to 10 wavefronts per SIMD on CDNA 3, so up to 40 per CU).
- Each scheduler issues one instruction per clock from a ready wavefront.
- Scheduling is **round-robin** among ready wavefronts on each SIMD: the scheduler picks a wavefront whose operands are available. If no wavefront is ready, the SIMD stalls.

```
Clock 0:  SIMD 0 issues WAVE_A instruction (memory load, ~400 cycle latency)
Clock 1:  SIMD 0 issues WAVE_B instruction (WAVE_A still waiting for data)
Clock 2:  SIMD 0 issues WAVE_C instruction
...
Clock N:  WAVE_A's data arrives, scheduler issues WAVE_A's next instruction
```

**Latency numbers (approximate, MI300X):**
- Register access: ~1 cycle
- LDS (shared memory): ~20–30 cycles
- L1 cache hit: ~30 cycles
- L2 cache hit (same XCD): ~100–200 cycles
- Infinity Cache hit: ~200–300 cycles
- HBM (global memory): ~400–600 cycles

**Instruction-level parallelism (ILP):** Within a single wavefront, independent instructions can be pipelined. If instruction B doesn't depend on instruction A's result, the hardware can issue B before A completes:

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
- If each wavefront stalls for ~400 cycles on a global load and issues 1 instruction before stalling, you'd need many wavefronts to keep the SIMD busy during those 400 cycles.
- In practice, 50%+ occupancy is usually sufficient for memory-bound kernels. Compute-bound kernels can sometimes do well at lower occupancy if ILP is high.
- More VGPRs per wavefront = fewer wavefronts per SIMD (lower occupancy). Check VGPR usage with `-Rpass-analysis=kernel-resource-usage`.

**What the profiler shows:**
- `rocprof-compute analyze` shows occupancy metrics: achieved occupancy, register usage, LDS usage, and what limits occupancy.
- Look for wavefront-level stall reasons: memory dependency, execution dependency, LDS conflicts, etc.

### Cache hierarchy details

**Cache line sizes:**
- **L1 cache**: 64-byte cache lines per CU. Smaller than NVIDIA's 128B lines.
- **L2 cache**: 128-byte cache lines, per XCD (not shared across XCDs — this is a key difference from NVIDIA).
- **Infinity Cache**: 256 MB across 4 IODs, acts as a last-level cache before HBM.

**L1 cache behavior:**
- The L1 vector cache is per-CU, small (~16–32 KB class on CDNA 3), and caches global memory reads.
- **LDS is not a cache** — it's explicitly managed shared memory (64 KB per CU). The programmer controls what goes in and out.
- Global loads can bypass L1 with GLC (globally coherent) and SLC (system-level coherent) modifiers in the ISA, but this is usually handled by the compiler based on memory scope.

**L2 cache (per XCD):**
- Each XCD has its own **4 MB L2 cache**, shared by the 38 CUs on that XCD.
- **Not shared across XCDs.** This is a fundamental difference from NVIDIA where L2 is global. Two workgroups on different XCDs that read the same data will each fetch it into their own L2.
- Write-back policy: writes are cached in L2 and flushed to Infinity Cache / HBM later.

**Infinity Cache:**
- 256 MB total across 4 IODs (~64 MB per IOD).
- Acts as a victim cache or last-level cache between L2 and HBM.
- Reduces HBM traffic for workloads with moderate locality that exceeds L2 capacity but fits in Infinity Cache.
- Hit rate matters: workloads that fit in 256 MB see significantly lower effective memory latency.

**Cache coherence:**
- Within an XCD, the L2 provides coherence for all 38 CUs.
- Across XCDs, Infinity Fabric maintains coherence, but at higher latency. A write by one XCD that needs to be visible to another XCD must propagate through the fabric.
- `__threadfence()` in HIP ensures writes are visible to other workgroups (maps to cache flush/invalidate operations in the ISA).
- `__threadfence_block()` ensures writes are visible within the workgroup (only needs LDS/L1 fence).
- `__threadfence_system()` ensures writes are visible to the host CPU and other GPUs.

### Constant memory and texture memory

**Constant memory:**
- AMD GPUs have a **scalar data cache** per CU that serves a similar role to NVIDIA's constant cache.
- SGPRs (Scalar General Purpose Registers) hold values shared across all lanes of a wavefront — these act as constant-like storage.
- The compiler automatically places uniform values (same for all threads) in SGPRs.
- HIP supports `__constant__` declarations, but the underlying hardware implementation differs from NVIDIA: it uses the scalar cache path rather than a dedicated constant memory segment.

```cpp
__constant__ float filter[256];  // declare in global scope

// Host side
hipMemcpyToSymbol(HIP_SYMBOL(filter), h_filter, 256 * sizeof(float));

// Device side: all threads reading the same address → served via scalar cache (fast)
// Divergent reads → falls back to vector cache path
```

**Texture memory:**
- CDNA architectures (MI300X, MI250, etc.) are compute-focused and have **limited or no texture hardware** compared to RDNA (gaming) GPUs.
- The texture cache path exists but is not a primary optimization target for HPC/ML workloads.
- For read-only global data, use `__builtin_nontemporal_load()` or rely on the compiler's read-only detection (similar to NVIDIA's `__ldg()`).

### Virtual memory and TLB

GPUs use virtual memory with their own page tables and TLBs (Translation Lookaside Buffers).

**Page sizes on AMD GPUs:**
- **Small pages**: 4 KB
- **Large pages**: 2 MB (used by default for VRAM allocations via the driver)
- Large pages reduce TLB pressure: a 2 MB page covers 512x more memory than a 4 KB page.

**TLB hierarchy:**
```
Thread → L1 TLB (per CU) → L2 TLB (per XCD or shared) → Page Table Walk (very slow)
```

- **L1 TLB**: per-CU, small, fast. Covers recently used pages for that CU.
- **L2 TLB**: larger, shared across CUs (per XCD). Covers more pages but higher latency.
- **Page table walk**: if both TLBs miss, the GPU walks the page table in memory — hundreds of extra cycles per access.

**When TLB matters:**
- Accessing many small, scattered allocations across different pages.
- Very large working sets that exceed TLB coverage (~2 GB with 1024 L2 TLB entries × 2 MB pages).
- **XNACK mode** (available on some AMD GPUs): enables GPU page faults for unified memory (like NVIDIA's Unified Memory). When a thread accesses an unmapped page, it faults and the driver migrates the page. This is controlled by the target architecture flag:
  ```bash
  hipcc --offload-arch=gfx942:xnack+ ...   # enable XNACK (page fault retry)
  hipcc --offload-arch=gfx942:xnack- ...   # disable XNACK (default, no page faults)
  ```
  XNACK is typically disabled for performance-critical code because page faults are expensive.

**`hipMallocManaged` (unified memory):**
- Works on AMD GPUs with XNACK support.
- Pages are migrated on demand between host and device.
- Use `hipMemPrefetchAsync` to pre-migrate pages and avoid runtime faults.

### Workgroup (thread block) scheduling

The **Command Processor (CP)** on each XCD distributes workgroups to CUs.

**How it works:**
1. Kernel launch specifies grid dimensions (total workgroups) and block dimensions.
2. The CP assigns workgroups to CUs based on resource availability (VGPRs, LDS, max-waves-per-CU).
3. Each CU can hold multiple workgroups simultaneously.
4. When a workgroup finishes, the CU becomes available for a new one.

**Scheduling across XCDs:**
- Workgroups are distributed across XCDs, typically in a round-robin or block-interleaved fashion.
- You cannot directly control which XCD a workgroup lands on (without advanced swizzling techniques).
- **Workgroup ID to XCD mapping** matters for L2 locality: workgroups with adjacent IDs may land on different XCDs, each with a separate 4 MB L2.

**Why it matters:**

- **Tail effect**: same as NVIDIA — the last wave of workgroups may not fill all CUs, leaving some idle. With 304 CUs, the quantization is coarser.
  ```
  1000 workgroups / 304 CUs = 3 full waves + 88 in the last wave
  Last wave: 88 CUs busy, 216 idle
  ```

- **XCD-aware scheduling (advanced)**:
  ```
  Without swizzling: WG 0→XCD0, WG 1→XCD1, ..., WG 7→XCD7, WG 8→XCD0
  Adjacent workgroups that share data end up on different XCDs → separate L2s → no cache reuse

  With swizzling (e.g., in Triton): remap program IDs so WG 0–37 → XCD0, WG 38–75 → XCD1, ...
  Adjacent workgroups share the same L2 → better cache hit rate for tiled GEMM
  ```
  This is one reason Triton on ROCm includes XCD-aware `program_id` swizzling for GEMM kernels.

- **Persistent kernels**: launch exactly as many workgroups as CUs can hold, each loops over work:
  ```cpp
  __global__ void persistent_kernel(float* data, int total_work, int items_per_wg) {
      int wg_start = blockIdx.x * items_per_wg;
      for (int i = wg_start; i < total_work; i += gridDim.x * items_per_wg) {
          // process item i
      }
  }
  ```

### Infinity Fabric and interconnect

**Infinity Fabric** is AMD's interconnect, used both within the GPU package (between IODs/XCDs) and between GPUs.

**On-package (MI300X):**
- Connects the 4 IODs so any XCD can reach any HBM stack or Infinity Cache slice.
- Bandwidth within the package is very high (multiple TB/s aggregate) but accessing HBM on a remote IOD adds latency compared to the local IOD.

**GPU-to-GPU (inter-node):**
- MI300X servers use **Infinity Fabric links** between GPUs, similar in concept to NVIDIA's NVLink.
- In an 8-GPU MI300X node (e.g., the AMD Instinct platform), GPUs are connected via xGMI (a variant of Infinity Fabric for inter-GPU communication).
- Aggregate bandwidth depends on the platform: typically **400–800 GB/s** per GPU bidirectional to the other GPUs.

**Comparison with NVLink:**

| Feature | Infinity Fabric (MI300X) | NVLink 4.0 (H100) |
|---|---|---|
| GPU-to-GPU BW per GPU | ~400–800 GB/s | 900 GB/s |
| Switch | Infinity Fabric Switch (in OAM platforms) | NVSwitch |
| CPU-GPU | Infinity Fabric (CXL on some) | PCIe Gen5 |

**How to check topology:**
```bash
rocm-smi --showtopo                         # show GPU interconnect topology
rocm-smi --showtopoweight                   # show link weights (latency indicator)
rocm-bandwidth-test                          # measure actual bandwidth between GPUs
```

**Why it matters for kernels:**
- **Multi-GPU kernels**: GPU-to-GPU transfers via Infinity Fabric are much faster than PCIe. Use `hipMemcpyPeerAsync` or RCCL for GPU-to-GPU communication.
- **Gradient all-reduce in training**: RCCL (AMD's equivalent of NCCL) uses the Infinity Fabric topology for collective operations.
- **Tensor parallelism**: splits layers across GPUs, requiring high-bandwidth interconnect for activation all-reduce.

### Power management and clock domains

GPUs have multiple clock domains and dynamic frequency scaling:

**Clock domains:**
- **SCLK (shader/compute clock)**: controls CU execution speed.
- **MCLK (memory clock)**: controls HBM bandwidth.
- **FCLK (fabric clock)**: controls Infinity Fabric speed.

**Dynamic clocking:**
- The GPU adjusts SCLK based on workload, temperature, and power budget.
- Under heavy load, the clock may throttle if the GPU hits its TDP (700W for MI300X).
- This causes **variable kernel runtime** — the same kernel may run faster on a cool GPU.

**For consistent benchmarking:**
```bash
# Check current clocks
rocm-smi --showclocks

# Lock clocks (requires root on some systems)
rocm-smi --setsclk <level>                  # set compute clock to a specific level
rocm-smi --setmclk <level>                  # set memory clock level
rocm-smi --setperflevel high                # force high-performance mode

# Or use amd-smi if available
amd-smi set --gpu 0 --perf-level high
```

**Power monitoring during a kernel:**
```bash
rocm-smi --showpower                        # current power draw
watch -n 1 'rocm-smi --showpower'           # live power monitoring
rocm-smi -d 0 --showclkfrq                  # show clock frequencies for GPU 0
```

**Thermal throttling thresholds (MI300X, typical):**
- **Edge temperature limit**: ~100°C
- **Junction (hotspot) temperature limit**: ~110°C
- The GPU throttles clocks when approaching these limits.
- Check with: `rocm-smi --showtemp`

### Query the architecture yourself

```bash
rocminfo | grep -E "Marketing Name|gfx|Compute Unit|Wavefront|Max Waves|LDS|Cache|Workgroup Max"
rocm-smi --showproductname --showmeminfo vram
amd-smi static          # if installed: detailed static device info
```

Save as `devprops.cpp` to print the same from the HIP runtime:

```cpp
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int n = 0;
    hipGetDeviceCount(&n);
    for (int d = 0; d < n; d++) {
        hipDeviceProp_t p;
        hipGetDeviceProperties(&p, d);
        printf("Device %d: %s\n", d, p.name);
        printf("  arch                : %s\n", p.gcnArchName);
        printf("  compute units       : %d\n", p.multiProcessorCount);
        printf("  warp (wave) size    : %d\n", p.warpSize);
        printf("  max threads/block   : %d\n", p.maxThreadsPerBlock);
        printf("  shared mem / block  : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  regs / block        : %d\n", p.regsPerBlock);
        printf("  global mem          : %.1f GB\n", p.totalGlobalMem / 1e9);
        printf("  L2 cache            : %d KB\n", p.l2CacheSize / 1024);
        printf("  clock               : %d MHz\n", p.clockRate / 1000);
        printf("  mem bus width       : %d bits\n", p.memoryBusWidth);
    }
    return 0;
}
```

```bash
hipcc --offload-arch=gfx942 devprops.cpp -o devprops && ./devprops
```

Exercise: compute the theoretical peak bandwidth and FLOP/s for your GPU from these numbers, then compare to what your `vec_add` and `matmul` actually achieve.

---

## 0. Setup on the GPU Droplet

```bash
ssh root@<droplet-ip>

# Make sure ROCm is on PATH
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

mkdir -p ~/rocm-practice && cd ~/rocm-practice
```

## 1. Verify the stack (do this first)

```bash
rocm-smi                      # GPU visible? temp, power, VRAM
rocm-smi --showproductname
amd-smi list                  # newer alternative to rocm-smi (if installed)
rocminfo | grep -E "Name:|gfx"   # MI300X should show gfx942
hipcc --version               # compiler + HIP version
hipconfig --full              # platform, paths, versions
ls /opt/rocm/bin | head -50   # see what tools you have
```

PyTorch sanity check:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 2. Sample kernel 1: vector add (with error checking + timing)

Save as `vec_add.cpp`:

```cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(cmd) do {                                            \
    hipError_t e = (cmd);                                              \
    if (e != hipSuccess) {                                             \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                    \
                hipGetErrorString(e), __FILE__, __LINE__);             \
        exit(1);                                                       \
    }                                                                  \
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
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipGetLastError());           // catches launch errors
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));   // catches execution errors

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

    // Verify against CPU
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-3f) { errors++; if (errors < 5) printf("Mismatch at %d\n", i); }
    }
    double gbps = (3.0 * bytes) / (ms * 1e-3) / 1e9;
    printf("%s | time: %.3f ms | bandwidth: %.1f GB/s | errors: %d\n",
           errors ? "FAIL" : "PASS", ms, gbps, errors);

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return errors != 0;
}
```

Build and run:

```bash
hipcc --offload-arch=gfx942 -O3 vec_add.cpp -o vec_add
./vec_add
```

Expected: `PASS`. Compare the bandwidth number to MI300X peak HBM bandwidth (several TB/s) to see how close you are. A single un-warmed launch will underreport, so try a warmup launch plus averaging over many runs as an exercise.

---

## 3. Sample kernel 2: tiled matrix multiply (shared memory)

Save as `matmul.cpp`:

```cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(cmd) do {                                            \
    hipError_t e = (cmd);                                              \
    if (e != hipSuccess) {                                             \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                    \
                hipGetErrorString(e), __FILE__, __LINE__);             \
        exit(1);                                                       \
    }                                                                  \
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
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&e));
    launch(g, b, A, B, C, N);                 // warmup
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(s));
    for (int i = 0; i < 10; i++) launch(g, b, A, B, C, N);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    return ms / 10.f;
}

static void launch_naive(dim3 g, dim3 b, const float* A, const float* B, float* C, int N) {
    matmul_naive<<<g, b>>>(A, B, C, N); HIP_CHECK(hipGetLastError());
}
static void launch_tiled(dim3 g, dim3 b, const float* A, const float* B, float* C, int N) {
    matmul_tiled<<<g, b>>>(A, B, C, N); HIP_CHECK(hipGetLastError());
}

int main() {
    const int N = 1024;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    float *ref = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) { hA[i] = (i % 7) * 0.1f; hB[i] = (i % 5) * 0.2f; }

    // CPU reference (slow but simple)
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            float s = 0; for (int k = 0; k < N; k++) s += hA[r * N + k] * hB[k * N + c];
            ref[r * N + c] = s;
        }

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, bytes)); HIP_CHECK(hipMalloc(&dB, bytes)); HIP_CHECK(hipMalloc(&dC, bytes));
    HIP_CHECK(hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice));

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    struct { const char* name; void (*fn)(dim3, dim3, const float*, const float*, float*, int); } ks[] = {
        {"naive", launch_naive}, {"tiled", launch_tiled}};
    for (auto& k : ks) {
        HIP_CHECK(hipMemset(dC, 0, bytes));
        float ms = time_kernel(k.fn, grid, block, dA, dB, dC, N);
        HIP_CHECK(hipMemcpy(hC, dC, bytes, hipMemcpyDeviceToHost));
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
hipcc --offload-arch=gfx942 -O3 matmul.cpp -o matmul
./matmul
```

Expected: both PASS, and tiled should be faster than naive. Then try changing `TILE` to 32, adding register blocking (each thread computes a 2x2 or 4x4 output), and comparing against rocBLAS/hipBLAS.

---

## 4. Tools to check correctness and performance

### 4.1 Correctness (does it work?)

| What | Command |
|---|---|
| Check every HIP call | `HIP_CHECK` macro (above), plus `hipGetLastError()` after each launch |
| Force synchronous launches (find which kernel fails) | `AMD_SERIALIZE_KERNEL=3 ./vec_add` |
| Verbose runtime logging | `AMD_LOG_LEVEL=3 ./vec_add` (levels 1 to 4, higher is noisier) |
| Compare to CPU or reference | built into the samples; also try `torch.allclose` against PyTorch |
| Debug crashes or hangs | `rocgdb ./vec_add` then `run`, `bt`, `info threads` |
| Compile with debug info | `hipcc --offload-arch=gfx942 -g -O0 vec_add.cpp -o vec_add_dbg` |
| Compiler warnings | add `-Wall -Wextra` to hipcc |
| Address sanitizer (host and device, optional, needs XNACK-capable setup) | `hipcc -fsanitize=address -shared-libsan --offload-arch=gfx942:xnack+ ...` (see ROCm docs; may not work on every setup) |

rocgdb quick start:

```bash
rocgdb ./vec_add_dbg
(gdb) break vec_add
(gdb) run
(gdb) info threads
(gdb) bt
(gdb) continue
```

### 4.2 Hardware and system health

```bash
rocm-smi --showuse            # GPU utilization while your kernel runs
rocm-smi --showmeminfo vram   # VRAM used
watch -n 1 rocm-smi           # live view (run in a second terminal)
rocm-bandwidth-test           # host/device and device/device bandwidth (if installed)
```

### 4.3 Profiling (is it fast, and why or why not?)

Tool names depend on version: `rocprofv3` is current, `rocprof` is legacy, and `rocprof-compute` (formerly Omniperf) gives deep per-kernel analysis.

```bash
# Quick kernel timing and HIP API trace with summary stats
rocprofv3 --hip-trace --kernel-trace --stats -d prof_out -- ./vec_add

# Hardware counters (list what is available first)
rocprofv3 --list-avail | head -100

# Deep analysis (roofline, memory, occupancy), best for the matmul
rocprof-compute profile -n matmul_run -- ./matmul
rocprof-compute analyze -p workloads/matmul_run/MI300
```

Things to read in the output:
- **Kernel duration**: compare naive vs tiled.
- **Memory bandwidth vs peak**: vec_add should be memory bound.
- **Occupancy and register or LDS usage**: why is the kernel limited?
- **Roofline**: is the matmul compute bound or memory bound?

### 4.4 Inspect what the compiler generated

```bash
# Resource usage per kernel (VGPRs, SGPRs, LDS, spills)
hipcc --offload-arch=gfx942 -O3 -Rpass-analysis=kernel-resource-usage matmul.cpp -o matmul

# Keep intermediate files (LLVM IR, GCN assembly)
hipcc --offload-arch=gfx942 -O3 -save-temps matmul.cpp -o matmul
ls *.s *.ll 2>/dev/null          # look at the generated gfx942 ISA

# List embedded device code objects
roc-obj-ls ./matmul
```

Check for register spills (scratch usage) in the resource report. Spills usually mean a slow kernel.

---

## 5. Practice ladder (do in this order)

1. Run `vec_add`, change `N` and the block size (64, 128, 256, 512, 1024), and note the effect on bandwidth.
2. Break it on purpose: remove the `if (i < n)` guard and run with `AMD_SERIALIZE_KERNEL=3`; launch with 2048 threads per block and watch the error message.
3. Run `matmul` and compare naive vs tiled. Profile both with `rocprofv3`.
4. Optimize matmul: `TILE=32`, register blocking, vectorized loads (`float4`), then compare against rocBLAS.
5. Write a reduction (sum) kernel using shared memory and warp-level ops (wavefront size is **64** on MI300X, not 32).
6. Write a softmax or layernorm kernel and verify against PyTorch.
7. Hipify an existing CUDA kernel of yours:

```bash
hipify-perl my_kernel.cu > my_kernel.hip.cpp
# or, more accurate (needs clang):
hipify-clang my_kernel.cu -o my_kernel.hip.cpp
hipcc --offload-arch=gfx942 -O3 my_kernel.hip.cpp -o my_kernel
```

8. Rebuild a kernel in Triton and compare the runtime against your HIP version.

---

## 6. Triton on ROCm (quick check)

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
x = torch.rand(n, device="cuda")   # "cuda" maps to the AMD GPU under ROCm
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

## 7. PyTorch checks

```bash
# Quick GEMM benchmark vs your kernel
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
print(f"{2*N**3/dt/1e12:.1f} TFLOP/s (fp16 GEMM via hipBLASLt/rocBLAS)")
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

## 8. RCCL for collective communications

**RCCL** (ROCm Collective Communications Library, pronounced "rockle") is AMD's equivalent of NVIDIA's NCCL. It provides the same collective operations, optimized for AMD's Infinity Fabric topology. The API is nearly identical to NCCL — most NCCL code works with RCCL by just changing the header and link flags.

### Collective operations

These are the same fundamental communication patterns used by NCCL. Every distributed training framework uses these under the hood.

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

### How AllReduce is implemented

RCCL (like NCCL) uses **ring** and **tree** algorithms depending on message size and topology.

**Ring AllReduce (large messages):**
GPUs are arranged in a logical ring. Two phases:

```
Phase 1: ReduceScatter (N-1 steps)
Each GPU sends a chunk to the next GPU in the ring, which adds it to its own chunk.
After N-1 steps, each GPU has the fully reduced version of one chunk.

Phase 2: AllGather (N-1 steps)
Each GPU sends its fully reduced chunk around the ring.
After N-1 steps, every GPU has all chunks.
```

Total data transferred per GPU: `2 * (N-1)/N * data_size` — bandwidth-optimal.

**Tree AllReduce (small messages):**
Binary tree for lower latency (log₂N steps instead of 2(N-1)):

```
        GPU0 (root)
       /    \
    GPU1    GPU2
    /
  GPU3

Reduce up the tree → Broadcast down the tree
Fewer steps but less bandwidth utilization
```

RCCL automatically picks the best algorithm based on message size and topology.

### Code example

```cpp
#include <rccl/rccl.h>

ncclComm_t comms[num_gpus];   // API uses ncclComm_t (same types as NCCL)
ncclCommInitAll(comms, num_gpus, dev_list);

// AllReduce across GPUs
for (int g = 0; g < num_gpus; g++) {
    hipSetDevice(g);
    ncclAllReduce(d_send[g], d_recv[g], count, ncclFloat, ncclSum, comms[g], streams[g]);
}

// Synchronize all
for (int g = 0; g < num_gpus; g++) {
    hipSetDevice(g);
    hipStreamSynchronize(streams[g]);
}
```

Build with: `hipcc -lrccl ...`

### NCCL vs RCCL comparison

| Feature | NCCL (NVIDIA) | RCCL (AMD) |
|---|---|---|
| Full name | NVIDIA Collective Communications Library | ROCm Collective Communications Library |
| API | `nccl.h` | `rccl/rccl.h` (same function names) |
| Link flag | `-lnccl` | `-lrccl` |
| Interconnect | NVLink, NVSwitch, InfiniBand | Infinity Fabric, xGMI |
| Algorithms | Ring, tree, collnet | Ring, tree (similar set) |
| Used by | PyTorch (default), DeepSpeed, Megatron | PyTorch (auto-detected on ROCm), DeepSpeed |

### Where each collective is used in training

| Collective | Where it's used |
|---|---|
| **AllReduce** | Data parallelism — sync gradients across GPUs after backward pass |
| **AllGather** | FSDP (Fully Sharded Data Parallel) — gather full parameters before forward pass |
| **ReduceScatter** | FSDP — shard gradients after backward pass; ZeRO optimizer stage 2 |
| **AllToAll** | Expert parallelism in MoE (Mixture of Experts) — route tokens to experts |
| **Broadcast** | Distribute model weights from rank 0 at initialization |
| **Reduce** | Collect metrics/losses to a single rank for logging |

### Debugging RCCL

```bash
# Verbose RCCL logging (same env vars as NCCL)
NCCL_DEBUG=INFO ./my_training_script

# Show which algorithm/protocol RCCL chose
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL ./my_training_script

# Common issues:
# - Timeout → usually a deadlock (not all ranks call the collective)
# - NCCL_P2P_DISABLE=1 → fallback if Infinity Fabric has issues
# - HSA_FORCE_FINE_GRAIN_PCIE=1 → may help with host-device transfers
```

---

## 9. Save your work (important with the credit deadline)

```bash
cd ~/rocm-practice
git init && git add . && git commit -m "ROCm practice kernels"
git remote add origin git@github.com:<you>/rocm-practice.git
git push -u origin main
```

## 9. Cost control

- Droplets bill hourly **even when powered off**. Destroy the Droplet when finished.
- Snapshot first if you want to keep the environment.
- Your credit expires October 14, 2026, so use the Droplet before then and push code off the machine.