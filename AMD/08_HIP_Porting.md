# HIP Porting Foundations

## What is HIP Porting
- HIP = C++ **marshalling** interface that maps to ROCm (AMD) or CUDA (NVIDIA)
- Same source compiled per platform — device binaries are not interchangeable
- AMD: `.hsaco` / `.hip_fatbin` vs NVIDIA: `.ptx` / `.cubin` / `.fatbin`
- Unified Runtime APIs: CUDA splits between `cu*` driver vs `cuda*` runtime; HIP mirrors both through unified interface
- Mirroring of CUDA driver/runtime concepts: modules, contexts, streams, events

## Porting Workflow
1. **Scan**: `hipify --examine` to quantify API references, LOC, warnings — shows which files can be changed, which APIs can be converted
2. **Convert**: `hipify-clang` (AST-based, recommended) or `hipify-perl` (text substitution, quick)
3. **Validate on NVIDIA**: `hipcc → nvcc` to stabilize functionality and performance
4. **Platform guards**: `#ifdef __HIP_PLATFORM_AMD__` / `__HIP_PLATFORM_NVIDIA__` — compiler skips guarded code
5. **Build for AMD**: HIP-Clang/amdclang++, validate on ROCm, tune occupancy
6. **Iterate**: scan → convert → build → resolve nested includes, wrappers, context/module corner cases

## HIPIFY Tools
| | hipify-clang (Production) | hipify-perl (Quick) |
|---|---|---|
| Method | Clang-based AST parsing | Pattern-based text substitution |
| Accuracy | High (handles complex code) | Lower |
| CUDA install | Required | Not required |

### Modes
- `--examine`: estimate conversion coverage (files, APIs, warnings)
- Directory scripts: `hipexamine.sh` / `hipexamine-perl.sh` for large-scale source tree analysis
- `--inplace` + `hipconvertinplace.sh`: modify source files directly, preserves `.prehip` backups
- Reports: detailed stats (APIs, counts, LOC, warnings) for identifying manual intervention points

## Build & Toolchain
| | AMD | NVIDIA |
|---|---|---|
| Compiler | amdclang++ (via `HIP_CLANG_PATH`) | nvcc |
| Wrapper | hipcc | hipcc → nvcc |
| Binaries | Not portable — compile for AMD target | Not portable — compile for NVIDIA target |
| Config tool | `hipconfig` — queries platform, compiler, include paths, runtime | Same |
| Headers | `hip/hip_runtime.h` (superset, preferred) over `hip/hip_runtime_api.h` (subset) | nvcc often auto-includes CUDA headers |
| PTX JIT | **Not implemented** — `hipModuleLoadDataEx` behaves like `hipModuleLoadData` | Supported |

## Platform Detection Macros
| Macro | amdclang++ | nvcc | Other (GCC/MSVC) |
|-------|-----------|------|-------------------|
| `__HIP_PLATFORM_AMD__` | Defined | - | Must define explicitly |
| `__HIP_PLATFORM_NVIDIA__` | - | Defined | Must define explicitly |
| `__HIPCC__` / `__HIP__` | Defined (HIP inputs) | Defined (HIP inputs) | Must define explicitly |
| `__HIP_DEVICE_COMPILE__` | 1 on device pass | 1 on device pass | - |
| `__CUDACC__` / `__NVCC__` | - | Defined | - |
| `__CUDA_ARCH__` | - | Defined (device only) | - |

- **amdclang++** performs multiple device passes (one per GPU target)
- **nvcc** performs a host pass + a single device pass
- Use `__HIP_DEVICE_COMPILE__` instead of `__CUDA_ARCH__`; HIP feature-specific macros recommended

## Code Objects & Fat Binaries
| Artifact | APIs | nvcc | HIP-Clang |
|----------|------|------|-----------|
| Code object | `hipModuleLoad`, `hipModuleLoadData` | .cubin / PTX | .hsaco |
| Fat binary | `hipModuleLoadFatBin` | .fatbin | .hip_fatbin |

- **Code object**: compiled GPU code for one specific architecture (e.g., `gfx942`)
- **Fat binary**: single file bundling compiled GPU code for several architectures — runtime picks the match
- No cross-vendor fatbin — build separately for AMD and NVIDIA
- Both backends support embedding code objects in executable and dynamic loading via `hipModuleLoad*`
- NVCC PTX JIT: `hipModuleLoadDataEx` passes PTX-JIT options on NVIDIA, ignored on AMD (HSACO is fully compiled)

## HIP-Clang Runtime: What Happens at Load
1. Device code linked across translation units — per-target code bundled into `__hip_fatbin` ELF section
2. Auto-generated host init calls: `__hipRegisterFatBinary`, `__hipRegisterFunction`, `__hipRegisterVar`
3. Termination: `__hipUnregisterFatBinary`
4. Single register guard: global `__hip_gpubin_handle` (linkonce, void**=0) ensures one-time registration
5. Result: stubs associated with kernels, ready for `<<<>>>` or `hipModuleLaunchKernel`

## Kernel Launch Forms
1. `<<<>>>` triple chevron syntax
2. `hipLaunchKernel`
3. `hipLaunchKernelGGL` (macro, expands to `<<<>>>` by default — can become template via `HIP_TEMPLATE_KERNEL_LAUNCH`)

**Launch sequence**: `hipConfigureCall` (sets grid, block, shared mem, stream) → `hipSetupArgument*` (packs args) → `hipLaunchByPtr` (dispatches)

## Contexts (HIP vs CUDA)
| | CUDA | HIP |
|---|---|---|
| Contexts | Explicit, multiple per device — each has own streams, events, virtual address space, memory pointers. Can create several "sessions" on one GPU | Manual creation avoided — effectively one context per GPU. Use `hipSetDevice()` + streams |
| Address spaces | Per-context — pointer from `cudaMalloc` in context A means nothing in context B | Process-wide — pointers shareable across contexts |
| Deprecations | - | `hipCtx*` exists for porting but deprecated for new development |

Context = grouping of one process's working state in a GPU.
**Action**: favor runtime API and streams; avoid new `hipCtx*`; review mixed API code during migration. Primary contexts created on first use — mixing runtime and driver calls can differ from CUDA.

## Memory Copies, Pointer Attributes & Directionality

### Memory Copies
- Direction-specific: `hipMemcpyHtoD`, `hipMemcpyDtoH`, `hipMemcpyDtoD`
- Generic: `hipMemcpy(ptr, kind)`
- Explicit direction calls reduce overhead (transfer path known at compile time)

### Pointer Attributes
- Query with `hipPointerGetAttributes(&attr, ptr)` → `attr.type` reports host/device/array/managed

| HIP | CUDA |
|-----|------|
| `hipMemoryTypeHost` | `cudaMemoryTypeHost` |
| `hipMemoryTypeDevice` | `cudaMemoryTypeDevice` |
| `hipMemoryTypeManaged` | `cudaMemoryTypeManaged` |
| - | `cudaMemoryTypeUnregistered` |

- NVIDIA platform: HIP maps enums via `nvidia_hip_runtime_api.h`
- HIP enums aren't identical on AMD and NVIDIA — `switch` on them can fail to compile on one platform. Wrap in `#ifdef __HIP_PLATFORM_AMD__` / `__HIP_PLATFORM_NVIDIA__`

### Directionality
- Identify correct pointer type to avoid misclassifying memory
- Prevents incorrect copy directions or unexpected transfer behavior
- Ensures runtime sees correct memory type — avoids direction errors, simplifies debugging

## Warp Size Portability
- NVIDIA: always 32 lanes. AMD: 32 or 64 depending on architecture (including RDNA GPUs)
- Device code: use `warpSize` built-in variable
- Host: query device properties at runtime
- Update lane-mask logic to width covering both cases (`uint64_t`)

```c
#if defined(__GFX8__) || defined(__GFX9__)
typedef uint64_t lane_mask_t;
#else
typedef uint32_t lane_mask_t;
#endif

unsigned int laneId = threadIdx.x % warpSize;
// Use lane ID for bit-shift
val & ((lane_mask_t{1} << (threadIdx.x % warpSize)) - 1);
// Shift with val variable
WarpReduce::sum((val < warpSize) ? (lane_mask_t{1} << val) : 0);
```

## Portable Feature Gating
| Device Side Macros | Host Side |
|-------------------|-----------|
| `__HIP_ARCH_HAS_DOUBLES__` | `hipGetDeviceProperties()` |
| `__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__` | `hipDeviceGetAttribute()` |
| `__HIP_ARCH_HAS_WARP_SHUFFLE__` | `hasWarpShuffle` |
| `__HIP_ARCH_HAS_3DGRID__` | `has3DGrid` |
| `__HIP_ARCH_HAS_DYNAMIC_PARALLEL__` | `hasSharedInt32Atomics` |

- Device: use `__HIP_ARCH_HAS_<FEATURE>__` macros to include/exclude hardware-specific code
- Host: don't rely on device macros (actual GPUs may differ at runtime) — query device properties
- Never use `__CUDA_ARCH__` — non-portable outside NVIDIA

## Libraries: hip* (Portable) vs roc* (AMD-Optimized)
| hip* | roc* |
|------|------|
| Portable marshalling layers — route to roc* on AMD or CUDA libs on NVIDIA | AMD optimized, may include ISA-specific kernels or alternate APIs, not intended as portable abstractions |

**Strategy:**
1. Cross-vendor deployment: standardize on `hip*` interfaces
2. AMD-only solutions: use `roc*` where available
3. AMD-exclusive: use `hipify --roc` to bias mapping toward `roc*`

## Runtime vs Driver Launches
- Prefer runtime `<<<>>>` launches for simplicity and readability

| Driver-Style APIs (`hipModule*`) | NVIDIA Backend |
|---|---|
| `hipModuleLoad` — dynamic code loading | Flows may mirror CUDA driver init: `hipInit` → `hipCtxCreate` before module ops |
| `hipModuleGetFunction` — retrieve kernel symbols | |
| `hipModuleLaunchKernel` — explicit parameter buffers | |

**Note**: ensure code object (.ptx/.cubin vs .hsaco) matches backend and kernel names passed to `hipModuleGetFunction` are correct.

## Launch Bounds
```c
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
```
- HIP uses launch bounds for occupancy control (not `--maxregcount` compiler flag)

| CU Mode (CDNA) | WGP Mode (RDNA) |
|----------------|-----------------|
| `MIN_WARPS_PER_EU = (MIN_BLOCKS * MAX_THREADS) / (warpSize * 2)` | `MIN_WARPS_PER_EU = (MIN_BLOCKS * MAX_THREADS) / (warpSize * 4)` |

## Driver Entry Points
- `hipGetProcAddress`: obtain runtime function pointers dynamically
  - Use cases: feature probing, forward compatibility, plugin architectures
- `hipGetDriverEntryPoint` / `cudaGetDriverEntryPoint`: expose runtime API entry points
- Aligns with driver-level resolution

## Per-Thread Default Stream
- `hipStreamPerThread`: reduces contention, simplifies async command submission
- Pair with `hipMemsetAsync` and `hipStreamSynchronize` on same per-thread stream for ordering

## Memory Type Introspection
- Use `hipPointerGetAttributes(&attr, ptr)` and branch on `attr.type`
- Integrate pointer classification into memory management utilities

| Memory Type | HIP | CUDA |
|-------------|-----|------|
| Host | `hipMemoryTypeHost` | `cudaMemoryTypeHost` |
| Device | `hipMemoryTypeDevice` | `cudaMemoryTypeDevice` |
| Managed | `hipMemoryTypeManaged` | `cudaMemoryTypeManaged` |
| Unregistered | - | `cudaMemoryTypeUnregistered` |

Use `#ifdef` guards for platform-specific enum values.

## Common Pitfalls & Debugging Aids
1. **Header gaps**: HIP requires explicit includes after HIPIFY (nvcc auto-includes CUDA headers)
2. **CUDA_ARCH usage**: replace with HIP feature macros and host queries
3. **Warp assumptions**: never hardcode 32; use `warpSize` and 64-bit masks
4. **Context mixing**: account for HIP-Clang primary context push; test runtime/driver paths
5. **Module mismatches**: match .hsaco vs .ptx; verify symbols with `hipModuleGetFunction`
6. **Memory transfers**: use `hipPointerAttributes` and assertions in debug builds

## Migration Checklist
1. Run HIPIFY (`--examine`), capture references, LOC, warnings
2. Convert with hipify-clang/perl, rescan and rebuild
3. Validate on NVIDIA (hipcc → nvcc); fix logic/performance, introduce platform guards
4. Compile on AMD with HIP-Clang; validate kernels, memory paths, occupancy
5. Feature-gate device code with `HIP_ARCH_HAS_*`; query device properties on host
6. Choose library strategy (`hip*` for portability, `roc*` for AMD optimization)
7. Hand off module format documentation, build configs, and CI matrix per platform (.ptx/.cubin/.fatbin for NVIDIA; .hsaco/.hip_fatbin for AMD)
