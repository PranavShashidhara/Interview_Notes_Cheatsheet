# AMD GPU Concepts, Terms, Issues & Tools Reference

## 1. AMD GPU Architecture Terminology

| AMD Term | CUDA Equivalent | Description |
|----------|----------------|-------------|
| Wavefront | Warp | Group of threads executing in lockstep. **64 lanes on CDNA (MI-series)**, **32 lanes on RDNA (consumer)** |
| Workgroup | Thread Block | Group of wavefronts that share LDS and can synchronize |
| Compute Unit (CU) | Streaming Multiprocessor (SM) | Contains SIMD units, LDS, scheduler, register files |
| Workgroup Processor (WGP) | — | RDNA concept: 2 CUs paired together sharing resources |
| LDS (Local Data Share) | Shared Memory | Fast on-chip scratchpad per CU, 64 KB typical |
| VGPR (Vector General Purpose Register) | Register | Per-lane register. Each wavefront lane has its own VGPR |
| SGPR (Scalar General Purpose Register) | — | Shared across all lanes in a wavefront (uniform values) |
| VALU (Vector ALU) | CUDA Core | Executes per-lane vector operations |
| SALU (Scalar ALU) | — | Executes uniform ops shared by all lanes (branching, indexing) |
| GDS (Global Data Share) | — | Small global scratchpad shared across all CUs |
| Shader Engine (SE) | GPC (Graphics Processing Cluster) | Top-level partition containing multiple CUs |
| Shader Array (SA) | TPC | Sub-partition within a Shader Engine |
| HBM (High Bandwidth Memory) | HBM | MI210: HBM2e, MI300X: HBM3 (192 GB) |
| Infinity Fabric | NVLink (conceptually) | AMD's high-speed interconnect between chiplets/GPUs |
| XGMI | NVLink | GPU-to-GPU interconnect used in MI-series multi-GPU |

### Architecture Hierarchy
```
GPU
 └── Shader Engine (SE)
      └── Shader Array (SA) / CU Array
           └── Compute Unit (CU)
                ├── 4x SIMD units (each 16-wide on CDNA → 4 cycles to execute wavefront64)
                ├── 1x Scalar Unit (SALU)
                ├── LDS (64 KB)
                ├── VGPRs (register file)
                ├── SGPRs (scalar register file)
                ├── Texture/Load units
                └── Scheduler
```

### Key Specs (CDNA vs RDNA)

| | CDNA2 (MI250X) | CDNA3 (MI300X) | RDNA3 (RX 7900) |
|---|---|---|---|
| Wavefront size | 64 | 64 | 32 (native) |
| CUs | 110 per GCD | 304 (across chiplets) | 96 (WGPs=48) |
| VGPRs per CU | 512 × 32-bit (per SIMD) | 512 × 32-bit (per SIMD) | 1536 per SIMD |
| LDS per CU | 64 KB | 64 KB | 128 KB per WGP |
| L2 Cache | 8 MB per GCD | 256 MB (Infinity Cache) | 96 MB (Infinity Cache) |
| HBM | 128 GB HBM2e | 192 GB HBM3 | GDDR6 24 GB |

---

## 2. Wavefront Divergence (Warp Divergence on AMD)

### What Is It?
When threads in the same wavefront take **different paths** at a branch (`if/else`), the hardware must execute **both paths serially**, masking off inactive lanes.

### How It Works on AMD
```
// Pseudocode causing divergence
if (threadIdx.x % 2 == 0) {
    // Path A — even lanes active, odd lanes masked
    do_something();
} else {
    // Path B — odd lanes active, even lanes masked
    do_other();
}
// Reconvergence point — all lanes active again
```

- CDNA (wavefront64): **64 threads** must agree → higher divergence probability than CUDA's 32-wide warps
- RDNA (wavefront32): **32 threads** → divergence profile similar to CUDA
- The SALU handles branch evaluation. If the scalar condition is **uniform** (same for all lanes), no divergence occurs — the SALU jumps directly

### AMD-Specific: Scalar vs Vector Branching
- **Scalar branch (`s_cbranch_*`)**: Condition is the same for all lanes → no divergence, fast
- **Vector branch (`v_cmp_*` + `s_cbranch_execz`)**: Condition varies per lane → EXEC mask set, divergent path
- The **EXEC mask** (64-bit on CDNA, 32-bit on RDNA) controls which lanes are active

### Performance Impact
- Divergence doesn't "slow down" — it **wastes SIMD lanes**. If only 1 of 64 lanes is active, you're at 1/64 utilization
- Nested divergence compounds: each nesting level can further reduce active lanes

### Mitigation Strategies
1. **Sort work by branch outcome** — group threads so entire wavefronts take the same path
2. **Use predication** for short branches — compiler may use `v_cndmask` instead of branching
3. **Restructure with `select`/ternary** — `result = cond ? a : b` avoids actual branch
4. **Reduce wavefront size** (RDNA: native 32, or use `__attribute__((amdgpu_flat_work_group_size(X,Y)))`)
5. **Move uniform conditions to scalar path** — ensure loop bounds and flags are SGPRs

---

## 3. Memory Coalescing

### What It Means
When threads in a wavefront access **contiguous, aligned** global memory addresses, the hardware merges these into fewer memory transactions. Uncoalesced access wastes bandwidth.

### Coalescing Rules on AMD
- L2 cache line: **128 bytes** (CDNA), **64 bytes** (RDNA)
- L1 cache line: **64 bytes** typically
- A wavefront64 accessing consecutive 4-byte elements = 256 bytes = 2–4 cache lines → **coalesced** (2 transactions)
- Same wavefront accessing scattered addresses → potentially **64 separate transactions**

### Good vs Bad Patterns
```cpp
// COALESCED: threads access consecutive addresses
float val = input[threadIdx.x];  // stride-1 access ✓

// UNCOALESCED: strided access
float val = input[threadIdx.x * N];  // stride-N access ✗

// UNCOALESCED: random/indirect
float val = input[indices[threadIdx.x]];  // gather ✗ (but may hit L1/L2)
```

### AoS vs SoA
```cpp
// Array of Structures (AoS) — BAD for coalescing
struct Particle { float x, y, z, w; };
Particle particles[N];
// Thread i reads particles[i].x → stride = sizeof(Particle) = 16 bytes

// Structure of Arrays (SoA) — GOOD for coalescing
float x[N], y[N], z[N], w[N];
// Thread i reads x[i] → stride = 4 bytes ✓
```

### Tips
- Prefer **SoA** over AoS
- Use `__builtin_nontemporal_load/store` for streaming access that shouldn't pollute cache
- Align allocations to **256 bytes** (`hipMalloc` does this by default)
- For matrix operations, ensure the **fastest-varying dimension** maps to consecutive threads

---

## 4. LDS Bank Conflicts

### LDS Structure
- **32 banks**, each **4 bytes wide** (32-bit word)
- Successive 4-byte words map to successive banks: address `4*k` → bank `k % 32`
- All 32/64 threads in a wavefront can access LDS simultaneously **if they hit different banks**

### What Causes Conflicts
```
Bank 0   Bank 1   Bank 2   ...   Bank 31
word[0]  word[1]  word[2]  ...   word[31]
word[32] word[33] word[34] ...   word[63]
```

- **Stride-1 access**: Thread `i` reads word `i` → each hits a different bank → **no conflict** ✓
- **Stride-2 access**: Thread `i` reads word `2*i` → only 16 of 32 banks used → **2-way conflict**
- **Stride-32 access**: All threads hit bank 0 → **32-way conflict** (serialized)

### Conflict Types
- **N-way bank conflict**: N threads access different addresses in the same bank → serialized into N accesses
- **Broadcast**: Multiple threads read the **same address** → **no conflict** (hardware broadcasts)

### Mitigation
```cpp
// BAD: column-major access with stride = 32
__shared__ float tile[32][32];
float val = tile[threadIdx.x][col];  // stride-32 → all hit same bank

// FIX: pad the inner dimension
__shared__ float tile[32][33];       // +1 padding breaks the stride pattern
float val = tile[threadIdx.x][col];  // now stride-33 → different banks ✓
```

- **Padding**: Add 1 element to the inner dimension of 2D shared arrays
- **Swizzle access patterns**: XOR-based index remapping to spread bank usage
- **Use 64-bit or 128-bit loads** when possible to reduce number of bank accesses

---

## 5. Occupancy

### Definition
**Occupancy** = (active wavefronts per CU) / (max wavefronts per CU)

High occupancy helps hide memory latency through wavefront switching. The scheduler switches between wavefronts on stalls (memory waits).

### Limiting Factors

| Resource | CDNA2 Limit per CU | How It Limits |
|----------|-------------------|---------------|
| Wavefronts | Max 32 per CU (8 per SIMD × 4 SIMDs) | Hard cap |
| VGPRs | 512 per SIMD (each 32-bit) | More VGPRs per wavefront → fewer wavefronts fit |
| SGPRs | 800 per CU | Shared across wavefronts on the CU |
| LDS | 64 KB per CU | Shared across workgroups on the CU |
| Workgroups | Max 40 per CU | Each workgroup reserves LDS, barrier slots |

### VGPR → Occupancy Table (CDNA2, wavefront64)

| VGPRs per wavefront | Max wavefronts per SIMD | Occupancy |
|---------------------|------------------------|-----------|
| ≤ 64 | 8 | 100% |
| 65–96 | 5 | 62.5% |
| 97–128 | 4 | 50% |
| 129–256 | 2 | 25% |
| 257–512 | 1 | 12.5% |

### How to Check Occupancy
```bash
# Use rocprof to get VGPR/SGPR usage
rocprof --stats kernel.hip

# Use launch bounds to hint the compiler
__global__ __attribute__((amdgpu_flat_work_group_size(64, 256)))
__attribute__((amdgpu_waves_per_eu(4)))
void myKernel() { ... }
```

### Tuning
- Reduce register pressure: reuse variables, use `__launch_bounds__`
- Reduce LDS usage per workgroup
- Sometimes **lower occupancy is fine** if compute-bound — register-heavy kernels with high ILP can still saturate ALUs
- Use `rocprof` counters: `SQ_WAVES`, `SQ_WAIT_ANY` to see if latency-bound

---

## 6. Other Common GPU Issues

### Register Spilling
- When a kernel uses more VGPRs than available → compiler **spills** to "scratch" memory (global memory backed)
- Scratch is **extremely slow** — looks like a register but costs 100x
- Check: `rocprof --stats` shows `ScratchSize`; if nonzero, you're spilling
- Fix: simplify kernel, split into multiple passes, use `__launch_bounds__`

### Instruction-Level Parallelism (ILP)
- AMD GPUs interleave instructions from the same wavefront — the compiler schedules VALU, VMEM, LDS ops in parallel
- Manual ILP: unroll loops, compute multiple output elements per thread

### `s_waitcnt` — Wait States
- AMD ISA requires explicit waits for memory operations:
  - `s_waitcnt vmcnt(N)` — wait until outstanding VMEM loads ≤ N
  - `s_waitcnt lgkmcnt(N)` — wait until outstanding LDS/GDS/scalar-mem ops ≤ N
  - `s_waitcnt expcnt(N)` — wait until outstanding exports ≤ N
- The compiler inserts these, but understanding them helps debug ISA-level performance

### Memory Fence & Atomics
- `__threadfence()` → ensures visibility across workgroups (maps to buffer_wbinvl1 on older, more complex on newer)
- Atomic ops: `atomicAdd`, `atomicCAS`, etc. — same API as CUDA via HIP
- **GDS atomics**: global scope atomics, slower but guaranteed cross-CU ordering

---

## 7. ROCm Tools & Commands

### rocm-smi — System Management Interface
```bash
# Show all GPUs, temp, power, utilization
rocm-smi

# Show detailed info for GPU 0
rocm-smi -d 0 --showtemp --showuse --showpower --showmeminfo vram

# Set power cap (watts)
rocm-smi -d 0 --setpoweroverdrive 300

# Reset clocks
rocm-smi --resetclocks

# Show topology (xGMI links, NUMA)
rocm-smi --showtopo

# Show firmware versions
rocm-smi --showfwinfo

# Monitor in a loop (like nvidia-smi dmon)
watch -n 1 rocm-smi
```

### rocminfo — Device Query
```bash
# List all agents (CPUs and GPUs), their properties
rocminfo

# Key fields to look for:
#   Name:             gfx90a (MI210), gfx942 (MI300X)
#   Max Wavefronts:   2048
#   Max Work Group:   1024
#   VGPRs:            512
#   SGPRs:            104
#   LDS size:         65536
```

### amdgpu-arch — Quick Architecture Query
```bash
# Print GPU architecture name(s)
amdgpu-arch
# Output: gfx90a   (or gfx942, gfx1100, etc.)
```

### hipcc — HIP Compiler
```bash
# Basic compilation
hipcc -o mykernel mykernel.hip

# Specify target architecture
hipcc --offload-arch=gfx90a -o mykernel mykernel.hip

# Multiple architectures (fat binary)
hipcc --offload-arch=gfx90a --offload-arch=gfx942 -o mykernel mykernel.hip

# Optimization flags
hipcc -O3 -o mykernel mykernel.hip

# Show compiler commands (debug)
hipcc --verbose -o mykernel mykernel.hip

# Use amdclang++ directly (more control)
amdclang++ -x hip --offload-arch=gfx90a -O3 -o mykernel mykernel.hip

# Save intermediate IR (for debugging)
hipcc -save-temps --offload-arch=gfx90a -o mykernel mykernel.hip

# Print resource usage (registers, LDS)
hipcc --offload-arch=gfx90a -Rpass-analysis=kernel-resource-usage -o mykernel mykernel.hip
```

### rocprof — Profiling
```bash
# Basic kernel stats (duration, grid size, registers)
rocprof --stats myapp

# Collect specific hardware counters
rocprof -i counters.txt -o output.csv myapp

# counters.txt example:
#   pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM SQ_WAIT_ANY
#   pmc: TCP_TCC_READ_REQ_sum TCP_TCC_WRITE_REQ_sum
#   pmc: TCC_HIT_sum TCC_MISS_sum

# HIP API trace (shows hipMemcpy, hipLaunchKernel, etc.)
rocprof --hip-trace myapp

# HSA API trace (lower level)
rocprof --hsa-trace myapp

# System trace (combined)
rocprof --sys-trace myapp

# Output goes to results.json (viewable in Chrome tracing: chrome://tracing)

# Key hardware counters:
#   SQ_WAVES           — total wavefronts launched
#   SQ_INSTS_VALU      — vector ALU instructions
#   SQ_INSTS_SALU      — scalar ALU instructions
#   SQ_INSTS_LDS       — LDS instructions
#   SQ_WAIT_ANY        — cycles wavefronts waited (latency indicator)
#   SQ_ACTIVE_INST_ANY — active instruction slots
#   TCP_TCC_READ_REQ   — L1→L2 read requests
#   TCC_HIT/TCC_MISS   — L2 cache hit/miss
#   FETCH_SIZE          — bytes fetched from memory
#   WRITE_SIZE          — bytes written to memory
```

### Omniperf — Advanced Profiling (MI-series)
```bash
# Profile an application
omniperf profile -n my_profile -- ./myapp

# Analyze results (interactive TUI)
omniperf analyze -p my_profile/

# Analyze specific kernel
omniperf analyze -p my_profile/ --kernel-names

# Key sections in Omniperf:
#   Speed-of-Light (SOL) — % of peak FLOPS, bandwidth achieved
#   Compute Unit metrics — VALU/SALU utilization, occupancy
#   L1/L2 cache metrics — hit rates, bandwidth
#   LDS metrics — bank conflicts, throughput
#   Instruction mix — VALU vs VMEM vs LDS breakdown

# Web UI (Grafana-based)
omniperf analyze -p my_profile/ --gui
```

### Omnitrace — Tracing & Visualization
```bash
# Instrument binary
omnitrace-instrument -o myapp.inst -- ./myapp

# Run instrumented binary (generates traces)
omnitrace-run ./myapp.inst

# Or runtime instrumentation (no recompile)
omnitrace-run --trace -- ./myapp

# Output: perfetto traces (open in ui.perfetto.dev)
# Shows: kernel timelines, HIP API calls, CPU-GPU overlap, memory copies
```

### rocgdb — GPU Debugger
```bash
# Compile with debug info
hipcc -g -O0 --offload-arch=gfx90a -o mykernel mykernel.hip

# Launch debugger
rocgdb ./mykernel

# Key commands inside rocgdb:
(gdb) break myKernel                  # breakpoint on kernel
(gdb) run                             # start execution
(gdb) info threads                    # show all wavefronts
(gdb) thread 2                        # switch to wavefront 2
(gdb) print $vgpr0                    # print vector register
(gdb) print $sgpr0                    # print scalar register
(gdb) print $exec                     # print EXEC mask (active lanes)
(gdb) set print lane-mask on          # show per-lane values
(gdb) print/x (int[64])$vgpr0        # all 64 lane values of vgpr0
(gdb) x/16xw 0x<lds_addr>            # examine LDS memory
(gdb) info rocm-devices               # list GPUs
(gdb) info rocm-waves                 # list active wavefronts
```

### roc-obj-ls / roc-obj-extract — Binary Inspection
```bash
# List code objects in a binary
roc-obj-ls myapp

# Extract code objects (ISA)
roc-obj-extract myapp

# Disassemble extracted object
llvm-objdump -d extracted.co --mcpu=gfx90a

# Or use amdgpu-dis
# Gives you the actual ISA (s_waitcnt, v_fma_f32, etc.)
```

### rocm-bandwidth-test — Bandwidth Testing
```bash
# Run full bidirectional bandwidth test
rocm-bandwidth-test

# Host-to-device only
rocm-bandwidth-test -t 1

# Device-to-device (P2P)
rocm-bandwidth-test -t 2

# Specific device pair
rocm-bandwidth-test -s 0 -d 1
```

### hipify-clang — CUDA to HIP Conversion
```bash
# Convert a single CUDA file
hipify-clang cuda_kernel.cu -o hip_kernel.hip

# Convert in-place
hipify-clang --inplace cuda_kernel.cu

# Convert entire project
hipify-clang --project-path=./cuda_project --output-path=./hip_project

# Dry run (show what would change)
hipify-clang --print-stats cuda_kernel.cu
```

---

## 8. ISA-Level Concepts

### Instruction Categories
| Category | Prefix | What It Does | Latency |
|----------|--------|-------------|---------|
| VALU | `v_` | Vector ALU (per-lane math) | 1–4 cycles |
| SALU | `s_` | Scalar ALU (uniform ops, branching) | 1 cycle |
| VMEM | `buffer_`/`global_`/`flat_` | Vector memory (global/generic loads/stores) | 100s of cycles |
| SMEM | `s_load_`/`s_buffer_` | Scalar memory (uniform address loads) | ~20 cycles |
| LDS | `ds_` | Local Data Share read/write | ~20 cycles |
| Export | `exp` | Output (graphics) | — |

### Wait Counts (s_waitcnt)
```asm
; Load from global memory — non-blocking
global_load_dword v1, v[2:3], off

; Must wait before using v1
s_waitcnt vmcnt(0)  ; wait for all VMEM loads to complete

; Now safe to use v1
v_add_f32 v4, v1, v5

; LDS operations use lgkmcnt
ds_read_b32 v6, v7
s_waitcnt lgkmcnt(0)
v_add_f32 v8, v6, v9
```

### EXEC Mask
- 64-bit register (CDNA) controlling which lanes participate
- Divergent branches set EXEC to mask off lanes
- `s_mov_b64 exec, <mask>` — manually set active lanes
- `v_cmp_*` instructions write comparison results to VCC or an SGPR pair, then branch based on EXEC

### Key ISA Instructions
```asm
v_fma_f32 v0, v1, v2, v3     ; v0 = v1 * v2 + v3 (fused multiply-add)
v_mad_f32 v0, v1, v2, v3     ; v0 = v1 * v2 + v3 (multiply-add, less precise)
v_dot2_f32_f16 v0, v1, v2, v3 ; 2x FP16 dot product (MI-series)
v_mfma_f32_32x32x8f16        ; Matrix FMA (CDNA MFMA instruction for tensor ops)
v_pk_fma_f16 v0, v1, v2, v3  ; Packed FP16 FMA (2 FP16 ops per instruction)
s_barrier                      ; Workgroup barrier (sync all wavefronts in workgroup)
s_endpgm                      ; End program
```

---

## 9. Quick Debugging Checklist

| Symptom | Likely Cause | Check With |
|---------|-------------|-----------|
| Kernel runs but wrong results | Race condition, missing sync | `rocgdb`, add `__syncthreads()` |
| Kernel very slow | Low occupancy, register spilling | `rocprof --stats`, check ScratchSize and VGPR count |
| Kernel very slow | Uncoalesced memory | `omniperf` → L1/L2 metrics, check access patterns |
| Kernel very slow | LDS bank conflicts | `omniperf` → LDS bank conflict counter |
| `hipErrorNoBinaryForGpu` | Wrong `--offload-arch` | `amdgpu-arch`, recompile with correct target |
| `hipErrorOutOfMemory` | VRAM exhaustion | `rocm-smi --showmeminfo vram` |
| GPU not detected | Driver issue | `rocminfo`, `dmesg | grep amdgpu` |
| Poor multi-GPU scaling | XGMI/PCIe bottleneck | `rocm-bandwidth-test`, `rocm-smi --showtopo` |
| Numerical differences vs CUDA | FMA contraction, rounding | Compile with `-ffp-contract=off`, compare |

---

## 10. Environment Variables

```bash
# Select specific GPU(s)
export HIP_VISIBLE_DEVICES=0,1

# Force specific platform
export HIP_PLATFORM=amd

# Enable kernel serialization (debugging)
export HIP_LAUNCH_BLOCKING=1

# Verbose HIP API logging
export AMD_LOG_LEVEL=4

# GPU memory allocation strategy
export HSA_TOOLS_LIB=librocprofiler64.so  # for profiling

# Disable SDMA (use shader copy instead — sometimes faster for small copies)
export HSA_ENABLE_SDMA=0

# Set GPU memory limit
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
```
