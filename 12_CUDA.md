# CUDA Complete Cheatsheet

> **Covers:** CUDA C/C++ Programming, Memory Management, Synchronization, Streams, Profiling, NVIDIA System Tools, and Best Practices

---

## Table of Contents

1. [GPU Architecture Concepts](#1-gpu-architecture-concepts)
2. [Compilation & Setup](#2-compilation--setup)
3. [Kernel Basics](#3-kernel-basics)
4. [Thread Hierarchy & Indexing](#4-thread-hierarchy--indexing)
5. [Memory Hierarchy](#5-memory-hierarchy)
6. [Memory Management APIs](#6-memory-management-apis)
7. [Unified Memory](#7-unified-memory)
8. [Synchronization](#8-synchronization)
9. [Streams & Concurrency](#9-streams--concurrency)
10. [Events & Timing](#10-events--timing)
11. [Atomic Operations](#11-atomic-operations)
12. [Warp-Level Primitives](#12-warp-level-primitives)
13. [Dynamic Parallelism](#13-dynamic-parallelism)
14. [Cooperative Groups](#14-cooperative-groups)
15. [CUDA Libraries](#15-cuda-libraries)
16. [Error Handling](#16-error-handling)
17. [Profiling: Nsight & nvprof](#17-profiling-nsight--nvprof)
18. [NVIDIA System Management (nvidia-smi)](#18-nvidia-system-management-nvidia-smi)
19. [Performance Optimization Checklist](#19-performance-optimization-checklist)
20. [Compute Capability Reference](#20-compute-capability-reference)

---

## 1. GPU Architecture Concepts

| Term | Description |
|------|-------------|
| **SM (Streaming Multiprocessor)** | Core compute unit; each GPU has many SMs |
| **CUDA Core** | Scalar FP32/INT32 execution unit inside an SM |
| **Tensor Core** | Matrix multiply unit (FP16/BF16/INT8); present since Volta (CC 7.0) |
| **Warp** | Group of 32 threads that execute in lockstep (SIMT) |
| **Block** | Programmer-defined group of threads; assigned to one SM |
| **Grid** | Collection of all blocks for a kernel launch |
| **L1 Cache / Shared Mem** | Fast on-chip memory shared per SM (configurable split) |
| **L2 Cache** | Shared across all SMs |
| **Global Memory** | Main GPU DRAM (HBM2/GDDR6); largest, slowest |
| **Register File** | Per-thread storage; fastest but limited (65536 per SM) |
| **Occupancy** | Ratio of active warps / max warps per SM |
| **Bank Conflict** | Multiple threads accessing same shared memory bank → serialized |

### SM Execution Model

```
GPU
└── N × Streaming Multiprocessors (SM)
    ├── Warp Schedulers (2–4 per SM)
    ├── Register File (65536 × 32-bit registers)
    ├── Shared Memory / L1 Cache (unified pool)
    ├── CUDA Cores (FP32, INT32)
    ├── Tensor Cores (Volta+)
    └── Special Function Units (SFU)
```

---

## 2. Compilation & Setup

### nvcc Compiler

```bash
# Basic compilation
nvcc -o program program.cu

# Specify GPU architecture (always recommended!)
nvcc -arch=sm_86 -o program program.cu          # Ampere (RTX 30xx)
nvcc -arch=sm_89 -o program program.cu          # Ada (RTX 40xx)
nvcc -arch=sm_90 -o program program.cu          # Hopper (H100)

# Generate PTX for multiple architectures (fat binary)
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -o program program.cu

# Optimization flags
nvcc -O3 -arch=sm_86 -o program program.cu

# Debug build
nvcc -G -g -arch=sm_86 -o program program.cu   # -G enables device debug info

# Line info (profiler-friendly, minimal overhead)
nvcc -lineinfo -arch=sm_86 -o program program.cu

# C++17 standard
nvcc -std=c++17 -arch=sm_86 -o program program.cu

# Enable fast math
nvcc --use_fast_math -arch=sm_86 -o program program.cu

# Verbose PTX output
nvcc -ptx -arch=sm_86 program.cu                # outputs program.ptx

# Show register usage
nvcc --ptxas-options=-v -arch=sm_86 program.cu
```

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCudaProject CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")  # Multi-arch

add_executable(program main.cu kernel.cu)
target_compile_options(program PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
)
```

### Include / Link

```cpp
#include <cuda_runtime.h>       // Runtime API
#include <cuda.h>               // Driver API
#include <device_launch_parameters.h>
#include <cuda_fp16.h>          // Half precision
#include <cooperative_groups.h> // Cooperative groups
#include <cuda/atomic>          // C++20-style atomics (libcu++)
```

---

## 3. Kernel Basics

### Function Qualifiers

| Qualifier | Callable From | Executes On |
|-----------|--------------|-------------|
| `__global__` | Host (or device w/ Dynamic Parallelism) | Device |
| `__device__` | Device only | Device |
| `__host__` | Host only | Host |
| `__host__ __device__` | Both | Both |
| `__noinline__` | Device | Device (prevents inlining) |
| `__forceinline__` | Device | Device (forces inlining) |

### Kernel Launch Syntax

```cpp
// Basic launch
kernel<<<gridDim, blockDim>>>(args...);

// With shared memory and stream
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);

// Example
dim3 block(256);
dim3 grid((N + block.x - 1) / block.x);
myKernel<<<grid, block>>>(d_data, N);
```

### Simple Kernel Example

```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch
int N = 1 << 20;  // 1M elements
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

### Kernel Variable Specifiers

```cpp
__shared__ float tile[32][32];    // Shared memory (block scope)
__constant__ float coeff[256];    // Constant memory (read-only, cached)
__device__ int globalCounter;     // Device global variable
__managed__ int sharedVar;        // Unified/managed variable
```

---

## 4. Thread Hierarchy & Indexing

### Built-in Variables

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | `dim3` | Thread index within block (x, y, z) |
| `blockIdx` | `dim3` | Block index within grid (x, y, z) |
| `blockDim` | `dim3` | Dimensions of each block |
| `gridDim` | `dim3` | Dimensions of grid |
| `warpSize` | `int` | Always 32 on current hardware |

### 1D, 2D, 3D Index Patterns

```cpp
// 1D grid of 1D blocks
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid of 2D blocks (matrix indexing)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;

// 3D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;

// Grid-stride loop (handles N > total threads)
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
    process(i);
}
```

### Limits per Compute Capability

| Resource | Limit |
|----------|-------|
| Max threads per block | 1024 |
| Max block dimensions | 1024 × 1024 × 64 |
| Max grid dimensions X | 2³¹ - 1 |
| Max grid dimensions Y, Z | 65535 |
| Max warps per SM | 64 (Ampere) |
| Max blocks per SM | 32 (Ampere) |
| Shared memory per SM | 48–164 KB (configurable) |
| Registers per SM | 65536 |

---

## 5. Memory Hierarchy

| Type | Scope | Lifetime | Speed | Size |
|------|-------|----------|-------|------|
| **Register** | Thread | Kernel | ~1 cycle | ~256 KB/SM |
| **Shared Memory** | Block | Kernel | ~5 cycles | 48–164 KB/SM |
| **L1 Cache** | SM | Automatic | ~5 cycles | Part of shared mem pool |
| **L2 Cache** | GPU | Application | ~30 cycles | 2–80 MB |
| **Constant Memory** | Grid | Application | ~5 cycles (cached) | 64 KB |
| **Texture Memory** | Grid | Application | ~600 cycles (uncached) | Up to global mem |
| **Global Memory** | Grid | Application | ~600 cycles | GBs (DRAM) |
| **Local Memory** | Thread | Kernel | ~600 cycles | Part of global |
| **Unified Memory** | CPU+GPU | Application | Variable | System RAM + GPU VRAM |

### Shared Memory Usage

```cpp
// Static allocation
__global__ void kernel() {
    __shared__ float s_data[1024];
    // use s_data...
}

// Dynamic allocation (specify size at launch)
__global__ void kernel(int n) {
    extern __shared__ float s_data[];  // extern keyword required
    // ...
}
// Launch: kernel<<<grid, block, n * sizeof(float)>>>(n);

// 2D tile (matrix multiply pattern)
__global__ void matMul(float* A, float* B, float* C, int N) {
    const int TILE = 32;
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

### Constant Memory

```cpp
__constant__ float d_filter[256];

// Copy to constant memory (host side)
cudaMemcpyToSymbol(d_filter, h_filter, 256 * sizeof(float));

// Read in kernel (broadcast to all threads in warp = very fast)
__global__ void applyFilter(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] *= d_filter[i % 256];
}
```

### Texture Memory

```cpp
// 1D texture (legacy API — still useful for spatial locality)
texture<float, 1, cudaReadModeElementType> tex;

cudaBindTexture(0, tex, d_data, N * sizeof(float));

__global__ void kernel(int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = tex1Dfetch(tex, i);
}

cudaUnbindTexture(tex);

// Modern texture object API
cudaTextureObject_t texObj = 0;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.sizeInBytes = N * sizeof(float);
resDesc.res.linear.desc = cudaCreateChannelDesc<float>();

cudaTextureDesc texDesc = {};
texDesc.readMode = cudaReadModeElementType;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

// In kernel: float val = tex1Dfetch<float>(texObj, i);
cudaDestroyTextureObject(texObj);
```

---

## 6. Memory Management APIs

### Basic Allocation

```cpp
// Device memory
float* d_ptr;
cudaMalloc(&d_ptr, N * sizeof(float));
cudaFree(d_ptr);

// Pinned (page-locked) host memory — enables async transfers
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));
cudaFreeHost(h_pinned);

// Or via cudaHostAlloc with flags
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocDefault);
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocWriteCombined); // write-only from host
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocMapped);        // zero-copy
```

### Memory Copy

```cpp
// Synchronous copies
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);

// Asynchronous (requires pinned memory)
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);

// 2D pitched copy
cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);

// Memset
cudaMemset(d_ptr, 0, N * sizeof(float));
cudaMemsetAsync(d_ptr, 0, N * sizeof(float), stream);
```

### Pitched Memory (2D arrays)

```cpp
size_t pitch;
float* d_matrix;
cudaMallocPitch(&d_matrix, &pitch, width * sizeof(float), height);

// Access element [row][col] in kernel
float* row_ptr = (float*)((char*)d_matrix + row * pitch);
float val = row_ptr[col];

// Copy 2D host array → pitched device array
cudaMemcpy2D(d_matrix, pitch,
             h_matrix, width * sizeof(float),
             width * sizeof(float), height,
             cudaMemcpyHostToDevice);
```

### 3D Memory

```cpp
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr d_vol;
cudaMalloc3D(&d_vol, extent);

cudaMemcpy3DParms p = {};
p.srcPtr = make_cudaPitchedPtr(h_data, width*sizeof(float), width, height);
p.dstPtr = d_vol;
p.extent = extent;
p.kind = cudaMemcpyHostToDevice;
cudaMemcpy3D(&p);
```

---

## 7. Unified Memory

```cpp
// Allocate accessible from both CPU and GPU
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Use from host
for (int i = 0; i < N; i++) data[i] = i;

// Use from device
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Read back on host
printf("%f\n", data[0]);

cudaFree(data);

// Prefetch to GPU (avoids page faults during kernel)
int device;
cudaGetDevice(&device);
cudaMemPrefetchAsync(data, N * sizeof(float), device, stream);

// Prefetch back to CPU
cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId, stream);

// Memory advice hints
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device);     // Cache on GPU
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device);
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, device);
```

---

## 8. Synchronization

### Host–Device Synchronization

```cpp
cudaDeviceSynchronize();           // Wait for all GPU work to complete
cudaStreamSynchronize(stream);     // Wait for specific stream
cudaEventSynchronize(event);       // Wait for a specific event
```

### Thread Block Synchronization (Kernel)

```cpp
__syncthreads();          // Sync all threads in a block (barrier)
__syncwarp();             // Sync all threads in a warp (Volta+)
__syncwarp(mask);         // Sync subset of warp using 32-bit mask

// Synchronize and test predicate
int __syncthreads_count(int predicate);   // Returns number of threads with true predicate
int __syncthreads_and(int predicate);     // 1 if ALL threads true
int __syncthreads_or(int predicate);      // 1 if ANY thread true
```

### Thread Fences (Memory Ordering)

```cpp
__threadfence();          // Ensure memory writes visible to all threads on device
__threadfence_block();    // Ensure visible to threads in same block
__threadfence_system();   // Ensure visible to CPU and all GPU threads (Unified Memory)
```

---

## 9. Streams & Concurrency

### Stream Basics

```cpp
// Create / destroy streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamDestroy(stream1);

// Non-blocking stream (doesn't sync with default stream)
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

// Priority streams (lower number = higher priority)
int loPri, hiPri;
cudaDeviceGetStreamPriorityRange(&loPri, &hiPri);
cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, hiPri);

// Synchronize
cudaStreamSynchronize(stream1);

// Query without blocking
cudaError_t status = cudaStreamQuery(stream1);
// cudaSuccess = complete, cudaErrorNotReady = still running
```

### Overlapping Transfers and Kernels

```cpp
// Double buffering pattern (overlap H2D, compute, D2H)
for (int i = 0; i < nChunks; i++) {
    int curr = i & 1, next = 1 - curr;

    // Async copy current chunk
    cudaMemcpyAsync(d_buf[curr], h_buf + i * chunkSize,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice, streams[curr]);

    // Launch kernel on current
    kernel<<<grid, block, 0, streams[curr]>>>(d_buf[curr], chunkSize);

    // Copy result back
    cudaMemcpyAsync(h_out + i * chunkSize, d_buf[curr],
                    chunkSize * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[curr]);
}
cudaDeviceSynchronize();
```

### Stream Callbacks

```cpp
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    printf("Stream %p done\n", (void*)stream);
}

cudaStreamAddCallback(stream, myCallback, nullptr, 0);
```

### CUDA Graphs (capture & replay)

```cpp
// Capture a graph
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

for (int i = 0; i < 100; i++) {
    kernel<<<grid, block, 0, stream>>>(d_data, N);
}

cudaStreamEndCapture(stream, &graph);

// Instantiate and launch
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

for (int iter = 0; iter < 1000; iter++) {
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}

cudaGraphExecDestroy(graphExec);
cudaGraphDestroy(graph);
```

---

## 10. Events & Timing

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Record events
cudaEventRecord(start, stream);          // Record start on stream
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);           // Record stop

// Wait and measure
cudaEventSynchronize(stop);             // CPU waits for stop event

float ms = 0;
cudaEventElapsedTime(&ms, start, stop); // milliseconds
printf("Kernel time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);

// Blocking event (CPU waits immediately)
cudaEventCreateWithFlags(&event, cudaEventBlockingSync);

// Disable timing (lower overhead for sync-only events)
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
```

---

## 11. Atomic Operations

### Integer Atomics

```cpp
int atomicAdd(int* addr, int val);           // Returns old value
int atomicSub(int* addr, int val);
int atomicExch(int* addr, int val);          // Exchange
int atomicMin(int* addr, int val);
int atomicMax(int* addr, int val);
int atomicAnd(int* addr, int val);
int atomicOr(int* addr, int val);
int atomicXor(int* addr, int val);
int atomicCAS(int* addr, int compare, int val); // Compare-and-swap

// Also available for unsigned int, unsigned long long, float (atomicAdd only)
unsigned long long atomicAdd(unsigned long long*, unsigned long long);
float atomicAdd(float*, float);           // Native since CC 2.0
double atomicAdd(double*, double);        // Native since CC 6.0
```

### Histogram Example

```cpp
__global__ void histogram(const int* data, int* hist, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&hist[data[i]], 1);
    }
}
```

### Lock-Free Stack (CAS pattern)

```cpp
struct Node { int val; Node* next; };
__device__ Node* head = nullptr;

__device__ void push(Node* node) {
    Node* old;
    do {
        old = head;
        node->next = old;
    } while (atomicCAS((unsigned long long*)&head,
                       (unsigned long long)old,
                       (unsigned long long)node) != (unsigned long long)old);
}
```

---

## 12. Warp-Level Primitives

### Warp Vote Functions

```cpp
// All/any/ballot for 32 threads in warp
unsigned __ballot_sync(unsigned mask, int predicate);   // Bitmask of true threads
int __all_sync(unsigned mask, int predicate);            // 1 if ALL true
int __any_sync(unsigned mask, int predicate);            // 1 if ANY true

// Example: active thread mask
unsigned mask = __activemask();
```

### Warp Shuffle (direct register exchange without shared mem)

```cpp
// Broadcast: all threads get value from lane srcLane
T __shfl_sync(unsigned mask, T var, int srcLane, int width=32);

// Shift down: thread i gets value from thread i+delta
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=32);

// Shift up: thread i gets value from thread i-delta
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=32);

// XOR: thread i gets value from thread i XOR laneMask
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32);

// Warp reduction using shuffle
__device__ float warpReduceSum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;  // Lane 0 holds sum
}
```

### Block Reduction Pattern

```cpp
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // One per warp
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
```

---

## 13. Dynamic Parallelism

> Requires Compute Capability ≥ 3.5. Compile with `-rdc=true`.

```cpp
__global__ void childKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.0f;
}

__global__ void parentKernel(float* data, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int blocks = (n + 255) / 256;
        childKernel<<<blocks, 256>>>(data, n);
        cudaDeviceSynchronize();  // Device-side sync
    }
}
```

```bash
# Compile with relocatable device code
nvcc -arch=sm_86 -rdc=true -o program program.cu
```

---

## 14. Cooperative Groups

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(float* data, int N) {
    // Thread block group
    cg::thread_block block = cg::this_thread_block();
    block.sync();  // equivalent to __syncthreads()

    // Warp-level group
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    float val = warp.shfl_down(data[threadIdx.x], 1);

    // Arbitrary tile sizes (power of 2 up to 32)
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);

    // Grid-wide sync (requires cooperative kernel launch)
    cg::grid_group grid = cg::this_grid();
    grid.sync();
}

// Cooperative kernel launch (grid-wide sync requires this)
void* args[] = { &data, &N };
cudaLaunchCooperativeKernel((void*)kernel, grid, block, args, sharedMem, stream);
```

---

## 15. CUDA Libraries

### cuBLAS (Dense Linear Algebra)

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// SGEMM: C = alpha*A*B + beta*C
// Note: cuBLAS uses column-major order!
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
            M, N, K,                    // Dimensions
            &alpha,
            d_A, M,                     // A (M×K), lda=M
            d_B, K,                     // B (K×N), lda=K
            &beta,
            d_C, M);                    // C (M×N), lda=M

cublasDestroy(handle);
```

### cuSPARSE

```cpp
#include <cusparse.h>
cusparseHandle_t handle;
cusparseCreate(&handle);
// SpMV, SpMM, sparse formats (CSR, COO, BSR), etc.
cusparseDestroy(handle);
```

### cuFFT

```cpp
#include <cufft.h>
cufftHandle plan;
cufftPlan1d(&plan, N, CUFFT_C2C, 1);  // Complex-to-complex 1D FFT
cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD);
cufftExecC2C(plan, d_out, d_in, CUFFT_INVERSE);
cufftDestroy(plan);
```

### cuRAND

```cpp
#include <curand.h>
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
curandGenerateUniform(gen, d_rand, N);   // Uniform [0,1)
curandGenerateNormal(gen, d_rand, N, 0.0f, 1.0f);  // Normal
curandDestroyGenerator(gen);
```

### cuDNN (Deep Neural Networks)

```cpp
#include <cudnn.h>
cudnnHandle_t handle;
cudnnCreate(&handle);
// Convolutions, pooling, activations, batch norm, RNN, etc.
cudnnDestroy(handle);
```

### Thrust (STL-like GPU algorithms)

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

thrust::device_vector<float> d_vec(N, 1.0f);
thrust::sort(d_vec.begin(), d_vec.end());
float sum = thrust::reduce(d_vec.begin(), d_vec.end());
thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                  thrust::negate<float>());
```

### CUB (CUDA UnBound — block/warp primitives)

```cpp
#include <cub/cub.cuh>

// Block-level reduce
__global__ void kernel(float* input, float* output) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float val = input[threadIdx.x];
    float result = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0) *output = result;
}

// Device-wide sort
void* d_temp = nullptr; size_t tempBytes = 0;
cub::DeviceRadixSort::SortKeys(d_temp, tempBytes, d_keys_in, d_keys_out, N);
cudaMalloc(&d_temp, tempBytes);
cub::DeviceRadixSort::SortKeys(d_temp, tempBytes, d_keys_in, d_keys_out, N);
```

---

## 16. Error Handling

### Runtime API Errors

```cpp
// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

// Check async kernel errors
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());       // Catches launch config errors
CUDA_CHECK(cudaDeviceSynchronize());  // Catches runtime kernel errors
```

### Common Error Codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `cudaSuccess` | No error |
| 1 | `cudaErrorInvalidValue` | Invalid argument |
| 2 | `cudaErrorMemoryAllocation` | cudaMalloc failed (OOM) |
| 10 | `cudaErrorInvalidDevice` | Invalid device ordinal |
| 35 | `cudaErrorInsufficientDriver` | Driver version too old |
| 77 | `cudaErrorIllegalAddress` | Illegal memory access |
| 98 | `cudaErrorNoKernelImageForDevice` | PTX not compatible |
| 700 | `cudaErrorLaunchFailed` | Kernel launch failed |
| 719 | `cudaErrorLaunchTimeout` | Windows TDR killed kernel |

### Device Properties

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);  // Device 0

printf("Device: %s\n", prop.name);
printf("Compute: %d.%d\n", prop.major, prop.minor);
printf("SMs: %d\n", prop.multiProcessorCount);
printf("Global Mem: %.1f GB\n", prop.totalGlobalMem / 1e9);
printf("Shared Mem/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
printf("Warp Size: %d\n", prop.warpSize);
printf("L2 Cache: %d MB\n", prop.l2CacheSize / (1024*1024));
printf("Mem Bandwidth: %.1f GB/s\n",
       2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
printf("Peak TFLOPS: %.1f\n",
       2.0 * prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor
       * prop.clockRate / 1e9);
```

---

## 17. Profiling: Nsight & nvprof

### nvprof (Legacy, pre-Ampere)

```bash
# Basic profile
nvprof ./program

# CSV output
nvprof --csv --log-file profile.csv ./program

# Specific metrics
nvprof --metrics all ./program
nvprof --metrics gld_efficiency,gst_efficiency,sm_efficiency ./program

# Timeline trace
nvprof --output-profile timeline.nvvp ./program

# Track memory transfers
nvprof --print-gpu-trace ./program

# API trace
nvprof --print-api-trace ./program

# Key metrics to watch
nvprof --metrics achieved_occupancy,sm_efficiency,ipc,\
gld_efficiency,gst_efficiency,shared_efficiency,\
l1_cache_global_hit_rate,l2_l1_read_hit_rate ./program
```

### Nsight Systems (ncu for timeline, nsys for system)

```bash
# System-level profile (CPU + GPU timeline)
nsys profile --stats=true -o report ./program
nsys profile -t cuda,osrt,nvtx --stats=true -o report ./program

# Launch Nsight Systems GUI
nsys-ui report.nsys-rep

# Quick stats
nsys stats report.nsys-rep

# Capture specific range
nsys profile --capture-range=cudaProfilerApi ./program
```

### Nsight Compute (ncu — kernel-level profiling)

```bash
# Profile all kernels
ncu ./program

# Save to file
ncu -o profile ./program
ncu-ui profile.ncu-rep   # Open in GUI

# Full metrics collection
ncu --set full -o profile ./program

# Specific sections
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./program

# Target specific kernel by name
ncu --kernel-name myKernel ./program

# Replay mode (for accurate metrics without kernel re-runs)
ncu --replay-mode kernel ./program

# Output to CSV
ncu --csv ./program > profile.csv

# Key metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active ./program

# Roofline analysis
ncu --set roofline -o roofline ./program
```

### NVTX Annotations (mark ranges for profiler)

```cpp
#include <nvtx3/nvToolsExt.h>

// Push/pop named ranges
nvtxRangePush("Data Preprocessing");
preprocessData(data, N);
nvtxRangePop();

nvtxRangePush("Kernel Launch");
myKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();
nvtxRangePop();

// Named range with ID (C++ API)
nvtxRangeId_t id = nvtxRangeStart("Forward Pass");
forwardPass();
nvtxRangeEnd(id);

// Mark a point in time
nvtxMark("Checkpoint A");

// With color
nvtxEventAttributes_t attribs = {};
attribs.version = NVTX_VERSION;
attribs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
attribs.colorType = NVTX_COLOR_ARGB;
attribs.color = 0xFF00FF00;  // Green
attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
attribs.message.ascii = "My Green Range";
nvtxRangePushEx(&attribs);
// ...
nvtxRangePop();
```

### Programmatic Profiler Control

```cpp
#include <cuda_profiler_api.h>

// Start/stop profiler capture from inside program
cudaProfilerStart();
hotKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();
cudaProfilerStop();
```

### Key Profiling Metrics Explained

| Metric | Ideal | Meaning |
|--------|-------|---------|
| **SM Efficiency** | >80% | % of time at least one warp is active |
| **Achieved Occupancy** | depends | Active warps / max possible warps |
| **Global Load Efficiency** | 100% | Useful bytes / total bytes loaded |
| **Global Store Efficiency** | 100% | Useful bytes / total bytes stored |
| **Shared Memory Efficiency** | 100% | No bank conflicts |
| **Warp Execution Efficiency** | 100% | Active lanes / 32 per warp |
| **IPC** | max | Instructions per clock cycle |
| **L1/L2 Hit Rate** | high | Cache effectiveness |
| **DRAM Utilization** | high for BW-bound | % peak memory bandwidth used |

---

## 18. NVIDIA System Management (nvidia-smi)

### Basic Queries

```bash
# Show all GPUs
nvidia-smi

# Concise one-liner per GPU
nvidia-smi -L

# Detailed GPU info
nvidia-smi -q
nvidia-smi -q -d MEMORY      # Memory only
nvidia-smi -q -d UTILIZATION # Utilization only
nvidia-smi -q -d TEMPERATURE # Temperature only
nvidia-smi -q -d POWER       # Power only
nvidia-smi -q -d CLOCK       # Clocks only
nvidia-smi -q -d ECC         # ECC error counts

# Specific GPU (index 0)
nvidia-smi -i 0 -q
```

### Monitoring

```bash
# Continuous monitoring (1 second interval)
nvidia-smi dmon

# Custom loop with specific fields
nvidia-smi dmon -s pcvumt      # power, clock, volatile util, mem, temp
# s = power state, p = power, c = sm clock, v = volatile util
# u = mem util, m = fb mem usage, t = temp, e = ecc

# One-line loop every N seconds
nvidia-smi -l 2                          # Refresh every 2s
watch -n 1 nvidia-smi                    # Alternative

# CSV output for logging
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,\
pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,\
utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
--format=csv -l 1 > gpu_log.csv

# Running processes
nvidia-smi pmon -s u
nvidia-smi pmon -d 1            # 1s interval
```

### Useful Query Fields (--query-gpu)

```bash
nvidia-smi --query-gpu=\
  name,\
  index,\
  uuid,\
  driver_version,\
  cuda.version,\
  pstate,\
  temperature.gpu,\
  temperature.memory,\
  power.draw,\
  power.limit,\
  power.default_limit,\
  power.max_limit,\
  clocks.sm,\
  clocks.mem,\
  clocks.gr,\
  clocks.max.sm,\
  utilization.gpu,\
  utilization.memory,\
  memory.total,\
  memory.used,\
  memory.free,\
  compute_mode,\
  ecc.errors.corrected.volatile.total,\
  ecc.errors.uncorrected.volatile.total \
--format=csv,noheader
```

### Power & Clock Management

```bash
# Set power limit (Watts) — requires root
nvidia-smi -i 0 -pl 300

# Enable/disable persistence mode (keeps driver loaded)
nvidia-smi -pm 1    # Enable
nvidia-smi -pm 0    # Disable

# Set application clocks (SM clock, Memory clock)
nvidia-smi -i 0 --applications-clocks 1215,1410    # MHz
nvidia-smi -i 0 --reset-applications-clocks

# Lock GPU clock to specific frequency (for consistent benchmarks)
nvidia-smi -i 0 --lock-gpu-clocks=1400,1400
nvidia-smi -i 0 --reset-gpu-clocks

# Lock memory clock
nvidia-smi -i 0 --lock-memory-clocks=9501
nvidia-smi -i 0 --reset-memory-clocks
```

### ECC & Compute Mode

```bash
# Enable/disable ECC (requires reboot)
nvidia-smi -i 0 --ecc-config=1    # Enable
nvidia-smi -i 0 --ecc-config=0    # Disable

# Clear ECC error counts
nvidia-smi -i 0 --clear-volatile-retired-pages

# Set compute mode
nvidia-smi -i 0 -c 0   # Default (multiple processes)
nvidia-smi -i 0 -c 1   # Exclusive thread (one thread)
nvidia-smi -i 0 -c 2   # Prohibited
nvidia-smi -i 0 -c 3   # Exclusive process
```

### Multi-GPU & NVLink

```bash
# NVLink status
nvidia-smi nvlink --status -i 0
nvidia-smi nvlink --capabilities -i 0

# Topology (how GPUs are connected)
nvidia-smi topo -m

# P2P access matrix
nvidia-smi topo -p2p r    # read bandwidth
nvidia-smi topo -p2p w    # write bandwidth

# NVSwitch fabric info
nvidia-smi fabric -i 0
```

### Multi-Instance GPU (MIG) — A100/H100

```bash
# Enable MIG mode
nvidia-smi -i 0 -mig 1

# List MIG profiles
nvidia-smi mig -lgip                      # GPU instance profiles
nvidia-smi mig -lcip                      # Compute instance profiles

# Create GPU instance (profile 9 = 1g.5gb on A100)
nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C

# List instances
nvidia-smi mig -lgi     # GPU instances
nvidia-smi mig -lci     # Compute instances

# Destroy all instances
nvidia-smi mig -dci && nvidia-smi mig -dgi

# Disable MIG
nvidia-smi -i 0 -mig 0
```

### Process Management

```bash
# Kill GPU process by PID
nvidia-smi --id=0 --kill-processes-on-device

# List compute processes
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv

# Accounting mode (track process GPU usage)
nvidia-smi -am 1            # Enable
nvidia-smi -q -d ACCOUNTING # View
nvidia-smi --clear-accounted-pids
```

### Environment Variables

```bash
# Select GPUs visible to application
export CUDA_VISIBLE_DEVICES=0,1          # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0            # Only GPU 0
export CUDA_VISIBLE_DEVICES=""           # No GPUs
export CUDA_VISIBLE_DEVICES=MIG-xxx      # Use specific MIG device

# Force specific device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID     # Physical PCI order
export CUDA_DEVICE_ORDER=FASTEST_FIRST  # Default (by perf)

# Disable caching of PTX JIT compilation
export CUDA_CACHE_DISABLE=1

# Set JIT cache directory
export CUDA_CACHE_PATH=/tmp/cuda_cache

# Enable CUDA malloc statistics
export CUDA_LAUNCH_BLOCKING=1           # Serialize all kernel launches (debug only!)
```

---

## 19. Performance Optimization Checklist

### Memory

- [ ] Use **coalesced global memory accesses** (consecutive threads → consecutive addresses)
- [ ] Maximize **shared memory** use to reduce global memory traffic
- [ ] Avoid **shared memory bank conflicts** (stride access → serialized)
- [ ] Use **pinned (page-locked) host memory** for all H2D/D2H transfers
- [ ] Use `cudaMallocPitch` / `cudaMalloc3D` for 2D/3D arrays
- [ ] Prefetch Unified Memory with `cudaMemPrefetchAsync`
- [ ] Prefer **structure of arrays** (SoA) over array of structures (AoS) for coalescing

### Threads & Occupancy

- [ ] Choose block size as multiple of 32 (warp size); common sweet spots: 128, 256
- [ ] Target ≥50% occupancy (use CUDA Occupancy Calculator)
- [ ] Minimize register usage to increase occupancy (use `__launch_bounds__`)
- [ ] Avoid thread divergence within warps
- [ ] Use grid-stride loops when N >> total threads

### Latency Hiding

- [ ] Use **streams** to overlap compute and H2D/D2H transfers
- [ ] Overlap multiple independent kernel calls in different streams
- [ ] Use **CUDA Graphs** to reduce kernel launch overhead in repeated workloads

### Compute

- [ ] Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` to hint register allocator
- [ ] Enable `--use_fast_math` for non-critical floating-point code
- [ ] Use **half precision (FP16)** or **TF32** where precision allows
- [ ] Use **Tensor Cores** via cuBLAS/cuDNN or WMMA API for matrix ops
- [ ] Fuse kernels to reduce memory round-trips
- [ ] Unroll loops: `#pragma unroll` or `#pragma unroll N`

### Miscellaneous

- [ ] Profile first — identify actual bottleneck (memory-bound vs compute-bound)
- [ ] Use `-lineinfo` for profiler source correlation without debug overhead
- [ ] Annotate with NVTX for clear profiler timelines
- [ ] Compile with `-arch=native` or target arch to get best code generation
- [ ] Ensure no false dependencies (separate read/write arrays where possible)

---

## 20. Compute Capability Reference

| Architecture | CC | Example GPUs | Key Features |
|---|---|---|---|
| Kepler | 3.0–3.7 | K80, K40 | Dynamic Parallelism (3.5+), Hyper-Q |
| Maxwell | 5.0–5.3 | GTX 750 Ti, GTX 970 | Improved shared mem, unified L1 |
| Pascal | 6.0–6.2 | P100, GTX 1080 | NVLink, FP16, Unified Memory improvements |
| Volta | 7.0 | V100 | Tensor Cores, Independent Thread Scheduling |
| Turing | 7.5 | RTX 2080, T4 | RT Cores, INT8/INT4 Tensor Cores |
| Ampere | 8.0–8.6 | A100, RTX 3090 | 3rd-gen Tensor Cores, TF32, BF16, MIG |
| Ada Lovelace | 8.9 | RTX 4090, L40 | 4th-gen Tensor Cores, Ada SMs |
| Hopper | 9.0 | H100 | Transformer Engine, FP8, NVLink 4.0, MIG 2.0 |
| Blackwell | 10.0 | B100, B200 | 5th-gen Tensor Cores, FP4, GB200 NVLink |

### WMMA API (Tensor Core Matrix Multiply)

```cpp
#include <mma.h>
using namespace nvcuda;

// 16×16×16 FP16 matrix multiply
__global__ void wmmaKernel(half* A, half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>               c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
```

---

## Quick Reference Card

```
Kernel launch:  kernel<<<grid, block, sharedMem, stream>>>(args)
Thread global:  idx = blockIdx.x * blockDim.x + threadIdx.x
Grid-stride:    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
Block sync:     __syncthreads()
Warp sync:      __syncwarp()
Fence:          __threadfence() / __threadfence_block()
Shuffle sum:    val += __shfl_down_sync(0xffffffff, val, offset)
Atomic:         atomicAdd(ptr, val)

Malloc:         cudaMalloc(&d_ptr, bytes)
Free:           cudaFree(d_ptr)
H2D:            cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)
D2H:            cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost)
Async:          cudaMemcpyAsync(d, h, bytes, kind, stream)
Prefetch:       cudaMemPrefetchAsync(ptr, bytes, device, stream)

Event time:     cudaEventRecord(e, s) → cudaEventElapsedTime(&ms, e0, e1)
Stream:         cudaStreamCreate(&s) → use s → cudaStreamSynchronize(s)
Graph:          cudaStreamBeginCapture → ... → cudaStreamEndCapture → Instantiate → Launch

nvidia-smi:     nvidia-smi -l 2          (monitor every 2s)
                nvidia-smi -q -d MEMORY  (memory query)
                nvidia-smi topo -m       (topology)# CUDA Complete Cheatsheet

> **Covers:** CUDA C/C++ Programming, Memory Management, Synchronization, Streams, Profiling, NVIDIA System Tools, and Best Practices

---

## Table of Contents

1. [GPU Architecture Concepts](#1-gpu-architecture-concepts)
2. [Compilation & Setup](#2-compilation--setup)
3. [Kernel Basics](#3-kernel-basics)
4. [Thread Hierarchy & Indexing](#4-thread-hierarchy--indexing)
5. [Memory Hierarchy](#5-memory-hierarchy)
6. [Memory Management APIs](#6-memory-management-apis)
7. [Unified Memory](#7-unified-memory)
8. [Synchronization](#8-synchronization)
9. [Streams & Concurrency](#9-streams--concurrency)
10. [Events & Timing](#10-events--timing)
11. [Atomic Operations](#11-atomic-operations)
12. [Warp-Level Primitives](#12-warp-level-primitives)
13. [Dynamic Parallelism](#13-dynamic-parallelism)
14. [Cooperative Groups](#14-cooperative-groups)
15. [CUDA Libraries](#15-cuda-libraries)
16. [Error Handling](#16-error-handling)
17. [Profiling: Nsight & nvprof](#17-profiling-nsight--nvprof)
18. [NVIDIA System Management (nvidia-smi)](#18-nvidia-system-management-nvidia-smi)
19. [Performance Optimization Checklist](#19-performance-optimization-checklist)
20. [Compute Capability Reference](#20-compute-capability-reference)

---

## 1. GPU Architecture Concepts

| Term | Description |
|------|-------------|
| **SM (Streaming Multiprocessor)** | Core compute unit; each GPU has many SMs |
| **CUDA Core** | Scalar FP32/INT32 execution unit inside an SM |
| **Tensor Core** | Matrix multiply unit (FP16/BF16/INT8); present since Volta (CC 7.0) |
| **Warp** | Group of 32 threads that execute in lockstep (SIMT) |
| **Block** | Programmer-defined group of threads; assigned to one SM |
| **Grid** | Collection of all blocks for a kernel launch |
| **L1 Cache / Shared Mem** | Fast on-chip memory shared per SM (configurable split) |
| **L2 Cache** | Shared across all SMs |
| **Global Memory** | Main GPU DRAM (HBM2/GDDR6); largest, slowest |
| **Register File** | Per-thread storage; fastest but limited (65536 per SM) |
| **Occupancy** | Ratio of active warps / max warps per SM |
| **Bank Conflict** | Multiple threads accessing same shared memory bank → serialized |

### SM Execution Model

```
GPU
└── N × Streaming Multiprocessors (SM)
    ├── Warp Schedulers (2–4 per SM)
    ├── Register File (65536 × 32-bit registers)
    ├── Shared Memory / L1 Cache (unified pool)
    ├── CUDA Cores (FP32, INT32)
    ├── Tensor Cores (Volta+)
    └── Special Function Units (SFU)
```

---

## 2. Compilation & Setup

### nvcc Compiler

```bash
# Basic compilation
nvcc -o program program.cu

# Specify GPU architecture (always recommended!)
nvcc -arch=sm_86 -o program program.cu          # Ampere (RTX 30xx)
nvcc -arch=sm_89 -o program program.cu          # Ada (RTX 40xx)
nvcc -arch=sm_90 -o program program.cu          # Hopper (H100)

# Generate PTX for multiple architectures (fat binary)
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -o program program.cu

# Optimization flags
nvcc -O3 -arch=sm_86 -o program program.cu

# Debug build
nvcc -G -g -arch=sm_86 -o program program.cu   # -G enables device debug info

# Line info (profiler-friendly, minimal overhead)
nvcc -lineinfo -arch=sm_86 -o program program.cu

# C++17 standard
nvcc -std=c++17 -arch=sm_86 -o program program.cu

# Enable fast math
nvcc --use_fast_math -arch=sm_86 -o program program.cu

# Verbose PTX output
nvcc -ptx -arch=sm_86 program.cu                # outputs program.ptx

# Show register usage
nvcc --ptxas-options=-v -arch=sm_86 program.cu
```

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCudaProject CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")  # Multi-arch

add_executable(program main.cu kernel.cu)
target_compile_options(program PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
)
```

### Include / Link

```cpp
#include <cuda_runtime.h>       // Runtime API
#include <cuda.h>               // Driver API
#include <device_launch_parameters.h>
#include <cuda_fp16.h>          // Half precision
#include <cooperative_groups.h> // Cooperative groups
#include <cuda/atomic>          // C++20-style atomics (libcu++)
```

---

## 3. Kernel Basics

### Function Qualifiers

| Qualifier | Callable From | Executes On |
|-----------|--------------|-------------|
| `__global__` | Host (or device w/ Dynamic Parallelism) | Device |
| `__device__` | Device only | Device |
| `__host__` | Host only | Host |
| `__host__ __device__` | Both | Both |
| `__noinline__` | Device | Device (prevents inlining) |
| `__forceinline__` | Device | Device (forces inlining) |

### Kernel Launch Syntax

```cpp
// Basic launch
kernel<<<gridDim, blockDim>>>(args...);

// With shared memory and stream
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);

// Example
dim3 block(256);
dim3 grid((N + block.x - 1) / block.x);
myKernel<<<grid, block>>>(d_data, N);
```

### Simple Kernel Example

```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch
int N = 1 << 20;  // 1M elements
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

### Kernel Variable Specifiers

```cpp
__shared__ float tile[32][32];    // Shared memory (block scope)
__constant__ float coeff[256];    // Constant memory (read-only, cached)
__device__ int globalCounter;     // Device global variable
__managed__ int sharedVar;        // Unified/managed variable
```

---

## 4. Thread Hierarchy & Indexing

### Built-in Variables

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | `dim3` | Thread index within block (x, y, z) |
| `blockIdx` | `dim3` | Block index within grid (x, y, z) |
| `blockDim` | `dim3` | Dimensions of each block |
| `gridDim` | `dim3` | Dimensions of grid |
| `warpSize` | `int` | Always 32 on current hardware |

### 1D, 2D, 3D Index Patterns

```cpp
// 1D grid of 1D blocks
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid of 2D blocks (matrix indexing)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;

// 3D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;

// Grid-stride loop (handles N > total threads)
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
    process(i);
}
```

### Limits per Compute Capability

| Resource | Limit |
|----------|-------|
| Max threads per block | 1024 |
| Max block dimensions | 1024 × 1024 × 64 |
| Max grid dimensions X | 2³¹ - 1 |
| Max grid dimensions Y, Z | 65535 |
| Max warps per SM | 64 (Ampere) |
| Max blocks per SM | 32 (Ampere) |
| Shared memory per SM | 48–164 KB (configurable) |
| Registers per SM | 65536 |

---

## 5. Memory Hierarchy

| Type | Scope | Lifetime | Speed | Size |
|------|-------|----------|-------|------|
| **Register** | Thread | Kernel | ~1 cycle | ~256 KB/SM |
| **Shared Memory** | Block | Kernel | ~5 cycles | 48–164 KB/SM |
| **L1 Cache** | SM | Automatic | ~5 cycles | Part of shared mem pool |
| **L2 Cache** | GPU | Application | ~30 cycles | 2–80 MB |
| **Constant Memory** | Grid | Application | ~5 cycles (cached) | 64 KB |
| **Texture Memory** | Grid | Application | ~600 cycles (uncached) | Up to global mem |
| **Global Memory** | Grid | Application | ~600 cycles | GBs (DRAM) |
| **Local Memory** | Thread | Kernel | ~600 cycles | Part of global |
| **Unified Memory** | CPU+GPU | Application | Variable | System RAM + GPU VRAM |

### Shared Memory Usage

```cpp
// Static allocation
__global__ void kernel() {
    __shared__ float s_data[1024];
    // use s_data...
}

// Dynamic allocation (specify size at launch)
__global__ void kernel(int n) {
    extern __shared__ float s_data[];  // extern keyword required
    // ...
}
// Launch: kernel<<<grid, block, n * sizeof(float)>>>(n);

// 2D tile (matrix multiply pattern)
__global__ void matMul(float* A, float* B, float* C, int N) {
    const int TILE = 32;
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

### Constant Memory

```cpp
__constant__ float d_filter[256];

// Copy to constant memory (host side)
cudaMemcpyToSymbol(d_filter, h_filter, 256 * sizeof(float));

// Read in kernel (broadcast to all threads in warp = very fast)
__global__ void applyFilter(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] *= d_filter[i % 256];
}
```

### Texture Memory

```cpp
// 1D texture (legacy API — still useful for spatial locality)
texture<float, 1, cudaReadModeElementType> tex;

cudaBindTexture(0, tex, d_data, N * sizeof(float));

__global__ void kernel(int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = tex1Dfetch(tex, i);
}

cudaUnbindTexture(tex);

// Modern texture object API
cudaTextureObject_t texObj = 0;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.sizeInBytes = N * sizeof(float);
resDesc.res.linear.desc = cudaCreateChannelDesc<float>();

cudaTextureDesc texDesc = {};
texDesc.readMode = cudaReadModeElementType;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

// In kernel: float val = tex1Dfetch<float>(texObj, i);
cudaDestroyTextureObject(texObj);
```

---

## 6. Memory Management APIs

### Basic Allocation

```cpp
// Device memory
float* d_ptr;
cudaMalloc(&d_ptr, N * sizeof(float));
cudaFree(d_ptr);

// Pinned (page-locked) host memory — enables async transfers
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));
cudaFreeHost(h_pinned);

// Or via cudaHostAlloc with flags
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocDefault);
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocWriteCombined); // write-only from host
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocMapped);        // zero-copy
```

### Memory Copy

```cpp
// Synchronous copies
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);

// Asynchronous (requires pinned memory)
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);

// 2D pitched copy
cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);

// Memset
cudaMemset(d_ptr, 0, N * sizeof(float));
cudaMemsetAsync(d_ptr, 0, N * sizeof(float), stream);
```

### Pitched Memory (2D arrays)

```cpp
size_t pitch;
float* d_matrix;
cudaMallocPitch(&d_matrix, &pitch, width * sizeof(float), height);

// Access element [row][col] in kernel
float* row_ptr = (float*)((char*)d_matrix + row * pitch);
float val = row_ptr[col];

// Copy 2D host array → pitched device array
cudaMemcpy2D(d_matrix, pitch,
             h_matrix, width * sizeof(float),
             width * sizeof(float), height,
             cudaMemcpyHostToDevice);
```

### 3D Memory

```cpp
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr d_vol;
cudaMalloc3D(&d_vol, extent);

cudaMemcpy3DParms p = {};
p.srcPtr = make_cudaPitchedPtr(h_data, width*sizeof(float), width, height);
p.dstPtr = d_vol;
p.extent = extent;
p.kind = cudaMemcpyHostToDevice;
cudaMemcpy3D(&p);
```

---

## 7. Unified Memory

```cpp
// Allocate accessible from both CPU and GPU
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Use from host
for (int i = 0; i < N; i++) data[i] = i;

// Use from device
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Read back on host
printf("%f\n", data[0]);

cudaFree(data);

// Prefetch to GPU (avoids page faults during kernel)
int device;
cudaGetDevice(&device);
cudaMemPrefetchAsync(data, N * sizeof(float), device, stream);

// Prefetch back to CPU
cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId, stream);

// Memory advice hints
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device);     // Cache on GPU
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device);
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, device);
```

---

## 8. Synchronization

### Host–Device Synchronization

```cpp
cudaDeviceSynchronize();           // Wait for all GPU work to complete
cudaStreamSynchronize(stream);     // Wait for specific stream
cudaEventSynchronize(event);       // Wait for a specific event
```

### Thread Block Synchronization (Kernel)

```cpp
__syncthreads();          // Sync all threads in a block (barrier)
__syncwarp();             // Sync all threads in a warp (Volta+)
__syncwarp(mask);         // Sync subset of warp using 32-bit mask

// Synchronize and test predicate
int __syncthreads_count(int predicate);   // Returns number of threads with true predicate
int __syncthreads_and(int predicate);     // 1 if ALL threads true
int __syncthreads_or(int predicate);      // 1 if ANY thread true
```

### Thread Fences (Memory Ordering)

```cpp
__threadfence();          // Ensure memory writes visible to all threads on device
__threadfence_block();    // Ensure visible to threads in same block
__threadfence_system();   // Ensure visible to CPU and all GPU threads (Unified Memory)
```

---

## 9. Streams & Concurrency

### Stream Basics

```cpp
// Create / destroy streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamDestroy(stream1);

// Non-blocking stream (doesn't sync with default stream)
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

// Priority streams (lower number = higher priority)
int loPri, hiPri;
cudaDeviceGetStreamPriorityRange(&loPri, &hiPri);
cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, hiPri);

// Synchronize
cudaStreamSynchronize(stream1);

// Query without blocking
cudaError_t status = cudaStreamQuery(stream1);
// cudaSuccess = complete, cudaErrorNotReady = still running
```

### Overlapping Transfers and Kernels

```cpp
// Double buffering pattern (overlap H2D, compute, D2H)
for (int i = 0; i < nChunks; i++) {
    int curr = i & 1, next = 1 - curr;

    // Async copy current chunk
    cudaMemcpyAsync(d_buf[curr], h_buf + i * chunkSize,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice, streams[curr]);

    // Launch kernel on current
    kernel<<<grid, block, 0, streams[curr]>>>(d_buf[curr], chunkSize);

    // Copy result back
    cudaMemcpyAsync(h_out + i * chunkSize, d_buf[curr],
                    chunkSize * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[curr]);
}
cudaDeviceSynchronize();
```

### Stream Callbacks

```cpp
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    printf("Stream %p done\n", (void*)stream);
}

cudaStreamAddCallback(stream, myCallback, nullptr, 0);
```

### CUDA Graphs (capture & replay)

```cpp
// Capture a graph
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

for (int i = 0; i < 100; i++) {
    kernel<<<grid, block, 0, stream>>>(d_data, N);
}

cudaStreamEndCapture(stream, &graph);

// Instantiate and launch
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

for (int iter = 0; iter < 1000; iter++) {
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}

cudaGraphExecDestroy(graphExec);
cudaGraphDestroy(graph);
```

---

## 10. Events & Timing

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Record events
cudaEventRecord(start, stream);          // Record start on stream
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);           // Record stop

// Wait and measure
cudaEventSynchronize(stop);             // CPU waits for stop event

float ms = 0;
cudaEventElapsedTime(&ms, start, stop); // milliseconds
printf("Kernel time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);

// Blocking event (CPU waits immediately)
cudaEventCreateWithFlags(&event, cudaEventBlockingSync);

// Disable timing (lower overhead for sync-only events)
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
```

---

## 11. Atomic Operations

### Integer Atomics

```cpp
int atomicAdd(int* addr, int val);           // Returns old value
int atomicSub(int* addr, int val);
int atomicExch(int* addr, int val);          // Exchange
int atomicMin(int* addr, int val);
int atomicMax(int* addr, int val);
int atomicAnd(int* addr, int val);
int atomicOr(int* addr, int val);
int atomicXor(int* addr, int val);
int atomicCAS(int* addr, int compare, int val); // Compare-and-swap

// Also available for unsigned int, unsigned long long, float (atomicAdd only)
unsigned long long atomicAdd(unsigned long long*, unsigned long long);
float atomicAdd(float*, float);           // Native since CC 2.0
double atomicAdd(double*, double);        // Native since CC 6.0
```

### Histogram Example

```cpp
__global__ void histogram(const int* data, int* hist, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&hist[data[i]], 1);
    }
}
```

### Lock-Free Stack (CAS pattern)

```cpp
struct Node { int val; Node* next; };
__device__ Node* head = nullptr;

__device__ void push(Node* node) {
    Node* old;
    do {
        old = head;
        node->next = old;
    } while (atomicCAS((unsigned long long*)&head,
                       (unsigned long long)old,
                       (unsigned long long)node) != (unsigned long long)old);
}
```

---

## 12. Warp-Level Primitives

### Warp Vote Functions

```cpp
// All/any/ballot for 32 threads in warp
unsigned __ballot_sync(unsigned mask, int predicate);   // Bitmask of true threads
int __all_sync(unsigned mask, int predicate);            // 1 if ALL true
int __any_sync(unsigned mask, int predicate);            // 1 if ANY true

// Example: active thread mask
unsigned mask = __activemask();
```

### Warp Shuffle (direct register exchange without shared mem)

```cpp
// Broadcast: all threads get value from lane srcLane
T __shfl_sync(unsigned mask, T var, int srcLane, int width=32);

// Shift down: thread i gets value from thread i+delta
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=32);

// Shift up: thread i gets value from thread i-delta
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=32);

// XOR: thread i gets value from thread i XOR laneMask
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32);

// Warp reduction using shuffle
__device__ float warpReduceSum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;  // Lane 0 holds sum
}
```

### Block Reduction Pattern

```cpp
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // One per warp
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
```

---

## 13. Dynamic Parallelism

> Requires Compute Capability ≥ 3.5. Compile with `-rdc=true`.

```cpp
__global__ void childKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.0f;
}

__global__ void parentKernel(float* data, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int blocks = (n + 255) / 256;
        childKernel<<<blocks, 256>>>(data, n);
        cudaDeviceSynchronize();  // Device-side sync
    }
}
```

```bash
# Compile with relocatable device code
nvcc -arch=sm_86 -rdc=true -o program program.cu
```

---

## 14. Cooperative Groups

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(float* data, int N) {
    // Thread block group
    cg::thread_block block = cg::this_thread_block();
    block.sync();  // equivalent to __syncthreads()

    // Warp-level group
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    float val = warp.shfl_down(data[threadIdx.x], 1);

    // Arbitrary tile sizes (power of 2 up to 32)
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);

    // Grid-wide sync (requires cooperative kernel launch)
    cg::grid_group grid = cg::this_grid();
    grid.sync();
}

// Cooperative kernel launch (grid-wide sync requires this)
void* args[] = { &data, &N };
cudaLaunchCooperativeKernel((void*)kernel, grid, block, args, sharedMem, stream);
```

---

## 15. CUDA Libraries

### cuBLAS (Dense Linear Algebra)

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// SGEMM: C = alpha*A*B + beta*C
// Note: cuBLAS uses column-major order!
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
            M, N, K,                    // Dimensions
            &alpha,
            d_A, M,                     // A (M×K), lda=M
            d_B, K,                     // B (K×N), lda=K
            &beta,
            d_C, M);                    // C (M×N), lda=M

cublasDestroy(handle);
```

### cuSPARSE

```cpp
#include <cusparse.h>
cusparseHandle_t handle;
cusparseCreate(&handle);
// SpMV, SpMM, sparse formats (CSR, COO, BSR), etc.
cusparseDestroy(handle);
```

### cuFFT

```cpp
#include <cufft.h>
cufftHandle plan;
cufftPlan1d(&plan, N, CUFFT_C2C, 1);  // Complex-to-complex 1D FFT
cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD);
cufftExecC2C(plan, d_out, d_in, CUFFT_INVERSE);
cufftDestroy(plan);
```

### cuRAND

```cpp
#include <curand.h>
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
curandGenerateUniform(gen, d_rand, N);   // Uniform [0,1)
curandGenerateNormal(gen, d_rand, N, 0.0f, 1.0f);  // Normal
curandDestroyGenerator(gen);
```

### cuDNN (Deep Neural Networks)

```cpp
#include <cudnn.h>
cudnnHandle_t handle;
cudnnCreate(&handle);
// Convolutions, pooling, activations, batch norm, RNN, etc.
cudnnDestroy(handle);
```

### Thrust (STL-like GPU algorithms)

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

thrust::device_vector<float> d_vec(N, 1.0f);
thrust::sort(d_vec.begin(), d_vec.end());
float sum = thrust::reduce(d_vec.begin(), d_vec.end());
thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                  thrust::negate<float>());
```

### CUB (CUDA UnBound — block/warp primitives)

```cpp
#include <cub/cub.cuh>

// Block-level reduce
__global__ void kernel(float* input, float* output) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float val = input[threadIdx.x];
    float result = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0) *output = result;
}

// Device-wide sort
void* d_temp = nullptr; size_t tempBytes = 0;
cub::DeviceRadixSort::SortKeys(d_temp, tempBytes, d_keys_in, d_keys_out, N);
cudaMalloc(&d_temp, tempBytes);
cub::DeviceRadixSort::SortKeys(d_temp, tempBytes, d_keys_in, d_keys_out, N);
```

---

## 16. Error Handling

### Runtime API Errors

```cpp
// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

// Check async kernel errors
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());       // Catches launch config errors
CUDA_CHECK(cudaDeviceSynchronize());  // Catches runtime kernel errors
```

### Common Error Codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `cudaSuccess` | No error |
| 1 | `cudaErrorInvalidValue` | Invalid argument |
| 2 | `cudaErrorMemoryAllocation` | cudaMalloc failed (OOM) |
| 10 | `cudaErrorInvalidDevice` | Invalid device ordinal |
| 35 | `cudaErrorInsufficientDriver` | Driver version too old |
| 77 | `cudaErrorIllegalAddress` | Illegal memory access |
| 98 | `cudaErrorNoKernelImageForDevice` | PTX not compatible |
| 700 | `cudaErrorLaunchFailed` | Kernel launch failed |
| 719 | `cudaErrorLaunchTimeout` | Windows TDR killed kernel |

### Device Properties

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);  // Device 0

printf("Device: %s\n", prop.name);
printf("Compute: %d.%d\n", prop.major, prop.minor);
printf("SMs: %d\n", prop.multiProcessorCount);
printf("Global Mem: %.1f GB\n", prop.totalGlobalMem / 1e9);
printf("Shared Mem/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
printf("Warp Size: %d\n", prop.warpSize);
printf("L2 Cache: %d MB\n", prop.l2CacheSize / (1024*1024));
printf("Mem Bandwidth: %.1f GB/s\n",
       2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
printf("Peak TFLOPS: %.1f\n",
       2.0 * prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor
       * prop.clockRate / 1e9);
```

---

## 17. Profiling: Nsight & nvprof

### nvprof (Legacy, pre-Ampere)

```bash
# Basic profile
nvprof ./program

# CSV output
nvprof --csv --log-file profile.csv ./program

# Specific metrics
nvprof --metrics all ./program
nvprof --metrics gld_efficiency,gst_efficiency,sm_efficiency ./program

# Timeline trace
nvprof --output-profile timeline.nvvp ./program

# Track memory transfers
nvprof --print-gpu-trace ./program

# API trace
nvprof --print-api-trace ./program

# Key metrics to watch
nvprof --metrics achieved_occupancy,sm_efficiency,ipc,\
gld_efficiency,gst_efficiency,shared_efficiency,\
l1_cache_global_hit_rate,l2_l1_read_hit_rate ./program
```

### Nsight Systems (ncu for timeline, nsys for system)

```bash
# System-level profile (CPU + GPU timeline)
nsys profile --stats=true -o report ./program
nsys profile -t cuda,osrt,nvtx --stats=true -o report ./program

# Launch Nsight Systems GUI
nsys-ui report.nsys-rep

# Quick stats
nsys stats report.nsys-rep

# Capture specific range
nsys profile --capture-range=cudaProfilerApi ./program
```

### Nsight Compute (ncu — kernel-level profiling)

```bash
# Profile all kernels
ncu ./program

# Save to file
ncu -o profile ./program
ncu-ui profile.ncu-rep   # Open in GUI

# Full metrics collection
ncu --set full -o profile ./program

# Specific sections
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./program

# Target specific kernel by name
ncu --kernel-name myKernel ./program

# Replay mode (for accurate metrics without kernel re-runs)
ncu --replay-mode kernel ./program

# Output to CSV
ncu --csv ./program > profile.csv

# Key metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active ./program

# Roofline analysis
ncu --set roofline -o roofline ./program
```

### NVTX Annotations (mark ranges for profiler)

```cpp
#include <nvtx3/nvToolsExt.h>

// Push/pop named ranges
nvtxRangePush("Data Preprocessing");
preprocessData(data, N);
nvtxRangePop();

nvtxRangePush("Kernel Launch");
myKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();
nvtxRangePop();

// Named range with ID (C++ API)
nvtxRangeId_t id = nvtxRangeStart("Forward Pass");
forwardPass();
nvtxRangeEnd(id);

// Mark a point in time
nvtxMark("Checkpoint A");

// With color
nvtxEventAttributes_t attribs = {};
attribs.version = NVTX_VERSION;
attribs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
attribs.colorType = NVTX_COLOR_ARGB;
attribs.color = 0xFF00FF00;  // Green
attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
attribs.message.ascii = "My Green Range";
nvtxRangePushEx(&attribs);
// ...
nvtxRangePop();
```

### Programmatic Profiler Control

```cpp
#include <cuda_profiler_api.h>

// Start/stop profiler capture from inside program
cudaProfilerStart();
hotKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();
cudaProfilerStop();
```

### Key Profiling Metrics Explained

| Metric | Ideal | Meaning |
|--------|-------|---------|
| **SM Efficiency** | >80% | % of time at least one warp is active |
| **Achieved Occupancy** | depends | Active warps / max possible warps |
| **Global Load Efficiency** | 100% | Useful bytes / total bytes loaded |
| **Global Store Efficiency** | 100% | Useful bytes / total bytes stored |
| **Shared Memory Efficiency** | 100% | No bank conflicts |
| **Warp Execution Efficiency** | 100% | Active lanes / 32 per warp |
| **IPC** | max | Instructions per clock cycle |
| **L1/L2 Hit Rate** | high | Cache effectiveness |
| **DRAM Utilization** | high for BW-bound | % peak memory bandwidth used |

---

## 18. NVIDIA System Management (nvidia-smi)

### Basic Queries

```bash
# Show all GPUs
nvidia-smi

# Concise one-liner per GPU
nvidia-smi -L

# Detailed GPU info
nvidia-smi -q
nvidia-smi -q -d MEMORY      # Memory only
nvidia-smi -q -d UTILIZATION # Utilization only
nvidia-smi -q -d TEMPERATURE # Temperature only
nvidia-smi -q -d POWER       # Power only
nvidia-smi -q -d CLOCK       # Clocks only
nvidia-smi -q -d ECC         # ECC error counts

# Specific GPU (index 0)
nvidia-smi -i 0 -q
```

### Monitoring

```bash
# Continuous monitoring (1 second interval)
nvidia-smi dmon

# Custom loop with specific fields
nvidia-smi dmon -s pcvumt      # power, clock, volatile util, mem, temp
# s = power state, p = power, c = sm clock, v = volatile util
# u = mem util, m = fb mem usage, t = temp, e = ecc

# One-line loop every N seconds
nvidia-smi -l 2                          # Refresh every 2s
watch -n 1 nvidia-smi                    # Alternative

# CSV output for logging
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,\
pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,\
utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
--format=csv -l 1 > gpu_log.csv

# Running processes
nvidia-smi pmon -s u
nvidia-smi pmon -d 1            # 1s interval
```

### Useful Query Fields (--query-gpu)

```bash
nvidia-smi --query-gpu=\
  name,\
  index,\
  uuid,\
  driver_version,\
  cuda.version,\
  pstate,\
  temperature.gpu,\
  temperature.memory,\
  power.draw,\
  power.limit,\
  power.default_limit,\
  power.max_limit,\
  clocks.sm,\
  clocks.mem,\
  clocks.gr,\
  clocks.max.sm,\
  utilization.gpu,\
  utilization.memory,\
  memory.total,\
  memory.used,\
  memory.free,\
  compute_mode,\
  ecc.errors.corrected.volatile.total,\
  ecc.errors.uncorrected.volatile.total \
--format=csv,noheader
```

### Power & Clock Management

```bash
# Set power limit (Watts) — requires root
nvidia-smi -i 0 -pl 300

# Enable/disable persistence mode (keeps driver loaded)
nvidia-smi -pm 1    # Enable
nvidia-smi -pm 0    # Disable

# Set application clocks (SM clock, Memory clock)
nvidia-smi -i 0 --applications-clocks 1215,1410    # MHz
nvidia-smi -i 0 --reset-applications-clocks

# Lock GPU clock to specific frequency (for consistent benchmarks)
nvidia-smi -i 0 --lock-gpu-clocks=1400,1400
nvidia-smi -i 0 --reset-gpu-clocks

# Lock memory clock
nvidia-smi -i 0 --lock-memory-clocks=9501
nvidia-smi -i 0 --reset-memory-clocks
```

### ECC & Compute Mode

```bash
# Enable/disable ECC (requires reboot)
nvidia-smi -i 0 --ecc-config=1    # Enable
nvidia-smi -i 0 --ecc-config=0    # Disable

# Clear ECC error counts
nvidia-smi -i 0 --clear-volatile-retired-pages

# Set compute mode
nvidia-smi -i 0 -c 0   # Default (multiple processes)
nvidia-smi -i 0 -c 1   # Exclusive thread (one thread)
nvidia-smi -i 0 -c 2   # Prohibited
nvidia-smi -i 0 -c 3   # Exclusive process
```

### Multi-GPU & NVLink

```bash
# NVLink status
nvidia-smi nvlink --status -i 0
nvidia-smi nvlink --capabilities -i 0

# Topology (how GPUs are connected)
nvidia-smi topo -m

# P2P access matrix
nvidia-smi topo -p2p r    # read bandwidth
nvidia-smi topo -p2p w    # write bandwidth

# NVSwitch fabric info
nvidia-smi fabric -i 0
```

### Multi-Instance GPU (MIG) — A100/H100

```bash
# Enable MIG mode
nvidia-smi -i 0 -mig 1

# List MIG profiles
nvidia-smi mig -lgip                      # GPU instance profiles
nvidia-smi mig -lcip                      # Compute instance profiles

# Create GPU instance (profile 9 = 1g.5gb on A100)
nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C

# List instances
nvidia-smi mig -lgi     # GPU instances
nvidia-smi mig -lci     # Compute instances

# Destroy all instances
nvidia-smi mig -dci && nvidia-smi mig -dgi

# Disable MIG
nvidia-smi -i 0 -mig 0
```

### Process Management

```bash
# Kill GPU process by PID
nvidia-smi --id=0 --kill-processes-on-device

# List compute processes
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv

# Accounting mode (track process GPU usage)
nvidia-smi -am 1            # Enable
nvidia-smi -q -d ACCOUNTING # View
nvidia-smi --clear-accounted-pids
```

### Environment Variables

```bash
# Select GPUs visible to application
export CUDA_VISIBLE_DEVICES=0,1          # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0            # Only GPU 0
export CUDA_VISIBLE_DEVICES=""           # No GPUs
export CUDA_VISIBLE_DEVICES=MIG-xxx      # Use specific MIG device

# Force specific device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID     # Physical PCI order
export CUDA_DEVICE_ORDER=FASTEST_FIRST  # Default (by perf)

# Disable caching of PTX JIT compilation
export CUDA_CACHE_DISABLE=1

# Set JIT cache directory
export CUDA_CACHE_PATH=/tmp/cuda_cache

# Enable CUDA malloc statistics
export CUDA_LAUNCH_BLOCKING=1           # Serialize all kernel launches (debug only!)
```

---

## 19. Performance Optimization Checklist

### Memory

- [ ] Use **coalesced global memory accesses** (consecutive threads → consecutive addresses)
- [ ] Maximize **shared memory** use to reduce global memory traffic
- [ ] Avoid **shared memory bank conflicts** (stride access → serialized)
- [ ] Use **pinned (page-locked) host memory** for all H2D/D2H transfers
- [ ] Use `cudaMallocPitch` / `cudaMalloc3D` for 2D/3D arrays
- [ ] Prefetch Unified Memory with `cudaMemPrefetchAsync`
- [ ] Prefer **structure of arrays** (SoA) over array of structures (AoS) for coalescing

### Threads & Occupancy

- [ ] Choose block size as multiple of 32 (warp size); common sweet spots: 128, 256
- [ ] Target ≥50% occupancy (use CUDA Occupancy Calculator)
- [ ] Minimize register usage to increase occupancy (use `__launch_bounds__`)
- [ ] Avoid thread divergence within warps
- [ ] Use grid-stride loops when N >> total threads

### Latency Hiding

- [ ] Use **streams** to overlap compute and H2D/D2H transfers
- [ ] Overlap multiple independent kernel calls in different streams
- [ ] Use **CUDA Graphs** to reduce kernel launch overhead in repeated workloads

### Compute

- [ ] Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` to hint register allocator
- [ ] Enable `--use_fast_math` for non-critical floating-point code
- [ ] Use **half precision (FP16)** or **TF32** where precision allows
- [ ] Use **Tensor Cores** via cuBLAS/cuDNN or WMMA API for matrix ops
- [ ] Fuse kernels to reduce memory round-trips
- [ ] Unroll loops: `#pragma unroll` or `#pragma unroll N`

### Miscellaneous

- [ ] Profile first — identify actual bottleneck (memory-bound vs compute-bound)
- [ ] Use `-lineinfo` for profiler source correlation without debug overhead
- [ ] Annotate with NVTX for clear profiler timelines
- [ ] Compile with `-arch=native` or target arch to get best code generation
- [ ] Ensure no false dependencies (separate read/write arrays where possible)

---

## 20. Compute Capability Reference

| Architecture | CC | Example GPUs | Key Features |
|---|---|---|---|
| Kepler | 3.0–3.7 | K80, K40 | Dynamic Parallelism (3.5+), Hyper-Q |
| Maxwell | 5.0–5.3 | GTX 750 Ti, GTX 970 | Improved shared mem, unified L1 |
| Pascal | 6.0–6.2 | P100, GTX 1080 | NVLink, FP16, Unified Memory improvements |
| Volta | 7.0 | V100 | Tensor Cores, Independent Thread Scheduling |
| Turing | 7.5 | RTX 2080, T4 | RT Cores, INT8/INT4 Tensor Cores |
| Ampere | 8.0–8.6 | A100, RTX 3090 | 3rd-gen Tensor Cores, TF32, BF16, MIG |
| Ada Lovelace | 8.9 | RTX 4090, L40 | 4th-gen Tensor Cores, Ada SMs |
| Hopper | 9.0 | H100 | Transformer Engine, FP8, NVLink 4.0, MIG 2.0 |
| Blackwell | 10.0 | B100, B200 | 5th-gen Tensor Cores, FP4, GB200 NVLink |

### WMMA API (Tensor Core Matrix Multiply)

```cpp
#include <mma.h>
using namespace nvcuda;

// 16×16×16 FP16 matrix multiply
__global__ void wmmaKernel(half* A, half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>               c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
```

---

## Quick Reference Card

```
Kernel launch:  kernel<<<grid, block, sharedMem, stream>>>(args)
Thread global:  idx = blockIdx.x * blockDim.x + threadIdx.x
Grid-stride:    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
Block sync:     __syncthreads()
Warp sync:      __syncwarp()
Fence:          __threadfence() / __threadfence_block()
Shuffle sum:    val += __shfl_down_sync(0xffffffff, val, offset)
Atomic:         atomicAdd(ptr, val)

Malloc:         cudaMalloc(&d_ptr, bytes)
Free:           cudaFree(d_ptr)
H2D:            cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)
D2H:            cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost)
Async:          cudaMemcpyAsync(d, h, bytes, kind, stream)
Prefetch:       cudaMemPrefetchAsync(ptr, bytes, device, stream)

Event time:     cudaEventRecord(e, s) → cudaEventElapsedTime(&ms, e0, e1)
Stream:         cudaStreamCreate(&s) → use s → cudaStreamSynchronize(s)
Graph:          cudaStreamBeginCapture → ... → cudaStreamEndCapture → Instantiate → Launch

nvidia-smi:     nvidia-smi -l 2          (monitor every 2s)
                nvidia-smi -q -d MEMORY  (memory query)
                nvidia-smi topo -m       (topology)
ncu:            ncu --set full -o report ./program
nsys:           nsys profile --stats=true -o report ./program
```

---

*Generated for CUDA 12.x / Driver 550+. Always check the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) for the latest details.*
ncu:            ncu --set full -o report ./program
nsys:           nsys profile --stats=true -o report ./program
```

---

*Generated for CUDA 12.x / Driver 550+. Always check the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) for the latest details.*