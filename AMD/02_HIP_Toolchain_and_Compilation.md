# HIP Toolchain & Compilation

## Compilation Pipeline
```
HIP Source → Clang/hipcc → LLVM IR → AMDGPU ISA → HSACO (ELF binary)
```

1. Host and device code written in C++
2. Device declaration identification — HIP construct parsing and GPU code preparation
3. LLVM IR: architecture-agnostic optimization before backend generates AMDGPU-specific instructions
4. HSACO output: ELF-based binary with GPU kernel code and resource metadata
- Toolchain embeds code variants to enable applications to run on multiple AMD GPU generations without source changes

## Why Understanding It Matters
- **Host/device separation**: diagnose incorrect device annotations, unsupported language constructs, mismatched compile flags
- **Multi-architecture workflows**: produce binaries for diverse GPU cluster/cloud environments
- **Runtime loader behavior**: anticipate startup performance, mitigate HSACO load latency, confirm correct architecture selection
- **LLVM/backend insights**: reason about register pressure, wavefront utilization, memory coalescing

## Host vs Device Compilation
| Host | Device |
|------|--------|
| Standard C++ compilation | GPU-specific compilation |
| Linked with HIP runtime libraries | IR transformations → backend lowering |
| Manages kernel launches, memory transfers, sync | HSACO generation |

## Compilation Modes

### AOT (Ahead of Time)
- Compiles at build time using hipcc/amdclang++ in a single step
- Device code embedded into host object file
- Faster startup — kernels precompiled and ready to run
- Offers performance predictability, build system integration, early validation
- Trade-off: increased build time and binary sizes when targeting multiple architectures

### hipRTC (Runtime Compilation)
- Compiles kernels at runtime from source strings
- Useful for development environments, distributing to unknown GPUs
- Uses `hiprtc*` API for runtime compilation
- Cannot access filesystem — all headers provided as strings
- Kernel names must be unmangled or retrieved via hipRTC APIs
- **Security caveat**: embedded source strings can be extracted by determined users — avoid if code must remain secret

### Hybrid Approach
- Performance-critical kernels: AOT (optimal runtime performance, minimal overhead)
- Specialized/rare kernels: hipRTC (runtime specializations for FFT, sorting, ML)

Multiple compilation paths exist because modern GPU applications vary in when kernel variants are known.

## Compiler Tools

### hipcc (Standard Entry Point)
- Wrapper around Clang and backend tools — unified command for host and device compilation
- Auto-configures HIP runtime include paths and injects required compiler flags
- Determines offload architecture from environment or user input
- Supports multi-architecture builds — multiple HSACO variants in single executable
- Abstracts complex Clang/LLVM flags, reduces misconfiguration risk and cognitive overhead

### amdclang++ (Advanced Control)
- Enables precise control over frontend behavior, IR generation, and device compilation flags
- Requires explicit `--offload-arch` flags for device targeting
- For experimenting with specific optimization passes, debugging IR-generation issues, analyzing backend transformations
- Ideal for research, profiling, and low-level debugging
- Complements hipcc for specialized workflows

## Device Code Identification (amdclang++)
- Scans HIP source for functions marked `__global__`, `__device__`, or `__host__ __device__`
- Builds AST subtree capturing GPU functions, intrinsic calls, thread hierarchy expressions
- Queues identified kernel entry points for LLVM transformation and backend lowering
- Uses Clang's HIP integration for precise host/device analysis — avoids misclassification causing runtime or linking errors

## LLVM IR Generation for Device Code
- Captures GPU logic in SSA form with memory hierarchy, threading models, math operations
- **Address space lowering**: global, local, shared, and constant memory represented in IR
- **Architecture-agnostic**: IR remains target-independent until AMDGPU optimization passes applied
- Thread and block index intrinsics lowered into IR-level expressions referencing special registers

## Key IR Optimization Passes
| Pass | Purpose |
|------|---------|
| SROA | Breaks structs/small arrays into individual scalars for efficient register allocation — avoids unnecessary memory traffic, critical for register-constrained GPU kernels |
| Aggressive inlining | Substitutes called functions at call sites — reduces call overhead in nested GPU kernels (costlier than on CPUs), exposes more code to optimization |
| Constant propagation | Reduces unnecessary computation by propagating known constant values |
| Loop unrolling/vectorization | Exploits GPU ALU throughput |

## AMDGPU Backend Compilation
1. **SelectionDAG lowering**: maps IR instructions to hardware-supportable operations
2. **Wavefront scheduling**: reorders instructions based on dependencies to minimize pipeline stalls and hazards
3. **Register allocation**: assigns to physical SGPRs and VGPRs with architecture constraints
4. **Instruction selection**: fine-tunes ALU, memory, and control flow operations
5. **Code emission**: assembles ISA into HSACO `.text` section

## HSACO Code Object Structure (ELF-based)
| Section | Contents |
|---------|----------|
| `.text` | Assembled kernel ISA (instructions) |
| `.rodata` | Immutable constants, lookup tables |
| `.note` | Metadata: kernel arguments, register usage, implicit parameters, dispatch requirements, alignment rules |
| Symbol tables | Entry point names resolved by runtime loader at launch |

### Kernel Metadata (guides runtime dispatch packets)
- **Registers**: ensure SGPR and VGPR counts available at launch
- **LDS**: enforce hardware limits for Local Data Share usage
- **Argument descriptors**: specify memory offsets, pointer qualifiers, hidden parameters
- **Execution config**: encode wavefront configuration and implicit arguments

### Inspection Tools
- `llvm-objdump`: disassemble kernel ISA to inspect low-level instructions and behavior
- `readelf`: examine ELF headers/sections for code-object structure
- **Metadata viewers**: display SGPR/VGPR/LDS usage to diagnose register pressure and confirm memory-instruction lowering

## TargetIDs
| ID | Meaning |
|----|---------|
| gfx90a | CDNA2 |
| gfx942 | CDNA3 |
| xnack+ | Page migration enabled |
| sramecc+ | SRAM ECC enabled |

**Why they encode**: microarchitectural model + feature toggles that influence code generation and behavior
**Why they matter**: allows compiler to generate optimized device code and create multi-target binaries with multiple code object variants

## Multi-Architecture / Fat Binaries
- Single executable bundles multiple HSACO variants for different AMD GPU architectures
- Fat binary aggregates HSACOs with metadata — runtime selects the most appropriate code object
- Enhances portability across heterogeneous clusters
- Ensures efficient execution without rebuilding per GPU model
- Trade-off: increases binary size and build time but simplifies deployment and long-term maintenance

## hipRTC Workflow

### Steps
```
Source string → hiprtcCreateProgram → hiprtcCompileProgram (+ flags)
→ hiprtcGetProgramLog (diagnostics) → hiprtcGetCodeSize → hiprtcGetCode
→ hipModuleLoadData → hipModuleLaunchKernel → hiprtcDestroyProgram
```

1. **Obtain source**: embed as string or generate at runtime
2. **Create program**: `hiprtcCreateProgram` — encapsulate compilation session, pass source and optional headers
3. **Compile**: `hiprtcCompileProgram` with flags; retrieve logs via `hiprtcGetProgramLog` / `hiprtcGetProgramLogSize` for warnings/errors
4. **Get code**: `hiprtcGetCodeSize` + `hiprtcGetCode` to obtain HSACO code object
5. **Load and launch**: `hipModuleLoadData` → extract kernel handle → `hipModuleLaunchKernel`
6. **Cleanup**: `hiprtcDestroyProgram`

### hipRTC Compilation Options
- **Target architecture selection**: ensures generated code is optimized for current GPU
- **Optimization levels**: control instruction selection, register usage, and inlining decisions
- **Language standard configuration**: configures C++ version to enable newer features in kernels

### Source Requirements
- **Explicit header management**: include all headers beyond built-in HIP headers manually (no filesystem access)
- **Self-contained source**: all code needed as string inputs, no external dependencies
- **Kernel name resolution**: ensure names are unmangled or retrievable via hipRTC APIs for correct linking

## Runtime Loading (HSACO)
1. Parses HSACO ELF sections and reads metadata for kernel dispatch
2. Resolves kernel symbol using metadata — allows application to reference compiled GPU function
3. Constructs dispatch packet: grid/block sizes, kernel arguments, resource requirements
4. Submits dispatch packet to GPU queue → launches kernel

### Dynamic Module Loading APIs
- `hipModuleLoad`: dynamic loading of kernel modules at runtime
- `hipModuleGetFunction`: retrieves kernel functions from loaded modules
- `hipModuleLaunchKernel`: launches kernels with runtime execution pipeline
