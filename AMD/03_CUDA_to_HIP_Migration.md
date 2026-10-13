# CUDA to HIP/ROCm Migration

## Platform Comparison
| | CUDA (NVIDIA) | HIP/ROCm (AMD) |
|---|---|---|
| API | Runtime (`cuda*`) + Driver (`cu*`), can mix both | Runtime (`hip*`) only, no exposed driver API; lower-level ROCm abstracted |
| Device code | PTX → cubin, bundled in fatbins | AMDGPU ISA → HSACO code objects |
| Code selection | Driver selects cubin or JIT-compiles PTX | Runtime selects matching HSACO |
| Compiler | nvcc | hipcc (Clang-based) |
| Baseline | Compilation/execution model is baseline for many GPU apps | Compilation/runtime behavior differs from CUDA, must be considered during migration |

- Both platforms support math/communication libraries, profiling tools, and require explicit host-device management and kernel launches
- Core programming abstractions (kernels, threads, memory spaces) map directly across platforms

## Performance Expectations After Porting
- **Functional correctness ≠ performance parity** — correct results can still have different performance
- Initial HIP ports typically underperform until retuned; CUDA-specific optimizations may not map directly
- AMD execution width varies: RDNA GPUs support 32 or 64 lane wavefronts
- Register usage (VGPR pressure) can affect occupancy more aggressively on AMD
- LDS differs from CUDA shared memory in bank structure patterns

## API Layers: Runtime vs Driver
| CUDA | HIP/ROCm |
|------|----------|
| Two public APIs: Runtime + Driver | Runtime API modeled after CUDA Runtime |
| Can use either independently or mix both | No CUDA-style driver API; lower-level ROCm abstracted from app code |

Migration notes:
- HIPIFY translates CUDA Runtime API calls; refactor CUDA Driver API usage manually
- Avoid mixing CUDA runtime and driver APIs — rely on consistent runtime interface for portability
- Validate module loading and kernel dispatch behavior after porting
- Update CUDA-specific error handling to use HIP equivalents
- Verify no CUDA driver headers remain in the codebase

## API Mapping
| CUDA | HIP |
|------|-----|
| `cudaError_t` | `hipError_t` |
| `cudaStream_t` | `hipStream_t` |
| `cudaEvent_t` | `hipEvent_t` |
| `cudaMalloc/Free/Memcpy` | `hipMalloc/Free/Memcpy` |
| `<<<>>>` launch syntax | Same `<<<>>>` or `hipLaunchKernelGGL` |
| `cudaGetLastError()` | `hipGetLastError()` |
| `cudaPeekAtLastError()` | `hipPeekAtLastError()` |
| `cudaGetErrorString()` | `hipGetErrorString()` |

Migration notes:
- Verify all `cuda*` APIs replaced with `hip*` after HIPIFY
- Review struct and handle type differences for compatibility
- Check kernel launch arguments for correct preservation during translation
- Validate error-handling semantics using HIP runtime
- Recheck locations where HIPIFY inserted TODO markers for manual intervention

## Execution Model
| CUDA | HIP/ROCm |
|------|----------|
| Warp = 32 threads always | Wavefront = 64 (CDNA) or 32/64 (RDNA) |

- Both use same hierarchical model: Grid → Block → Thread; supports 1D, 2D, 3D indexing
- Built-in variables: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `warpSize`

Migration notes:
- **Never hardcode warpSize** — query it at runtime for portability
- Adapt warp-level operations for 32 or 64 lane wavefronts on AMD
- Retune block sizes based on compute unit occupancy for target architecture
- Reassess register usage (VGPR pressure) to avoid limiting parallelism
- Adjust shared memory/LDS layouts to align with AMD bank structures and avoid conflicts
- Reprofile memory access patterns for efficient coalescing on target hardware

## Memory Model
| Type | Notes |
|------|-------|
| Global | Large, off-chip (same both platforms) |
| Shared / LDS | On-chip, per-block; LDS bank structure differs; used for tiling and improving data locality |
| Local | Per-thread storage |
| Constant/Texture | Specialized access paths |

- Global memory access should be coalesced and aligned to achieve high bandwidth
- CUDA: UVA and Unified Memory with prefetch/memory advice
- HIP: managed and host-mapped memory with different controls and constraints

Migration notes:
- Retune memory usage patterns to match AMD GPU behavior
- Adjust LDS tile sizes and ensure bank-friendly layouts to avoid conflicts
- Avoid large per-thread local arrays — prevents register spilling into global memory
- Revalidate global memory access patterns for proper coalescing and alignment
- Reevaluate kernel configurations originally tuned for 32-thread warps
- Verify managed and host-mapped memory usage aligns with ROCm semantics
- Validate alignment constraints to prevent unexpected behavior

## Memory Coherency
- Coherence has a cost and should not be assumed implicitly
- Write visibility may lag during kernel execution — rely on explicit synchronization
- Coherence guaranteed at kernel boundaries (both platforms)
- Correctness inside a running kernel depends on appropriate synchronization primitives
- Explicit synchronization used to avoid subtle data races and maintain portability

Migration notes:
- Use explicit synchronization (barriers, memory fences, atomics) for memory visibility
- Avoid implicit assumptions about intra-kernel coherence
- Validate cross-block communication using streams and events
- Reassess coherence guarantees for managed and host-mapped memory under ROCm
- Recheck ordering assumptions from original CUDA implementation after migration

## Host-Device Interactions
- Streams enable asynchronous kernel execution and memory operations (both platforms)
- Events used to express dependencies and measure execution timing
- Support for peer access depending on hardware topology
- Graph-based APIs define and execute sequences of operations (features may vary)

Migration notes:
- Preserve stream-ordered execution to maintain correct operation sequencing
- Map CUDA streams/events to HIP equivalents (`cudaStream_t` → `hipStream_t`)
- Replace NVTX profiling markers with roctx for ROCm tool compatibility
- Revalidate asynchronous memory copy behavior on target hardware
- Verify peer access support based on system topology and AMD GPU capabilities

## Library Mapping
| Area | CUDA | ROCm |
|------|------|------|
| BLAS | cuBLAS / cuBLASLt | rocBLAS / hipBLAS |
| FFT | cuFFT / cuFFTDx | rocFFT / hipFFT |
| RNG | cuRAND | rocRAND / hipRAND |
| Sparse | cuSPARSE / cuSPARSELt | rocSPARSE / hipSPARSE / hipSPARSELt |
| Solvers | cuSOLVER | rocSOLVER / hipSOLVER |
| Multi-GPU | NCCL | RCCL |

Migration notes:
- **Strategy**: `hip*` wrapper APIs for minimal changes and portability; `roc*` for advanced features and AMD-specific performance
- Validate numerical correctness across CUDA and ROCm library implementations
- Verify handle and descriptor compatibility after migration
- Benchmark library kernels on AMD GPUs to evaluate performance

## Unsupported / Different CUDA Features
- **Dynamic parallelism**: kernels launching other kernels must be refactored to host-side or alternative strategies
- Certain CUDA intrinsics may require alternative implementations
- Some CUDA libraries may require replacement with HIP wrappers or native ROCm libraries
- `hipify-clang` reports unsupported or partially supported constructs during translation

## Error Handling
| CUDA | HIP |
|------|-----|
| `cudaError_t` | `hipError_t` |
| `cudaGetLastError()` / `cudaPeekAtLastError()` | `hipGetLastError()` / `hipPeekAtLastError()` |
| `cudaGetErrorString()` | `hipGetErrorString()` |

- Kernel launches are asynchronous — errors may surface later
- Errors may require explicit synchronization to surface reliably

## Debugging & Profiling
| CUDA | ROCm |
|------|------|
| cuda-gdb | rocgdb |
| Nsight Compute (kernel metrics) | rocprof / rocprofv2 (kernel metrics) |
| Nsight Systems (system timelines) | roctracer (system tracing) |
| NVTX markers | roctx markers |

Migration notes:
- Insert synchronization points to isolate kernel launch errors and make debugging predictable
- Learn ROCm debugging and profiling tools (rocgdb, rocprof, roctracer)
- Replace NVTX markers with roctx for timeline annotation
- Retune kernels after profiling due to different performance characteristics on AMD GPUs

## Build System
| CUDA | HIP/ROCm |
|------|----------|
| nvcc generates PTX + cubin | hipcc (Clang-based) generates AMDGPU ISA + HSACO |
| PTX/cubins bundled into fatbins | Multiple `--offload-arch=gfx*` targets bundled in single binary |
| Driver selects cubin or JIT-compiles PTX | Runtime selects appropriate code object |

Migration notes:
- Replace nvcc with hipcc (or Clang-based HIP flow)
- Specify offload targets with `--offload-arch=gfx*`
- Update include paths and linking configuration for ROCm libraries
- Validate generated HSACO code objects load and execute correctly
- Compare compiler warnings before/after migration for compatibility issues

## Common Migration Pitfalls
1. Hardcoding warp/wavefront size as 32 (AMD may use 32 or 64)
2. Relying on implicit memory coherence without proper synchronization
3. Expecting performance parity without retuning block sizes, LDS usage, access patterns
4. Forgetting to update compiler flags and gfx targets or mixing CUDA/HIP headers
5. Skipping validation after replacing libraries — causes subtle numerical errors

## Validation Strategy (ordered)
1. Verify functional correctness — unit and integration tests
2. Validate numerical accuracy and floating point tolerance
3. Begin performance profiling and optimization
4. Maintain dual builds (CUDA + HIP) during transition
5. Revalidate after compiler, driver, or ROCm stack updates

## Migration Checklist
1. Run `hipify-clang` and review translation logs for unmapped/partially supported constructs
2. Replace CUDA libraries with `hip*` wrappers or native `roc*` libraries
3. Refactor device-side kernel launches (dynamic parallelism) into host-driven workflows
4. Audit warp-level code for correct adaptation to 32 and 64 lane wavefronts
5. Update build system to use hipcc with appropriate gfx targets
6. Profile application, retune occupancy and memory behavior, verify numerical correctness and benchmark performance
