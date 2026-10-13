# Kernel Coding Practice — ANSWERS
*Complete programs. CUDA ones run on Colab (`!nvcc file.cu -o out && ./out` in a GPU runtime). Plain C/C++ compile locally: `g++ -O2 -pthread file.cpp` / `gcc -O2 file.c`.*

---

## Part 1 — CUDA

### A1. Vector Add (complete program)
```cuda
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { hA[i] = i; hB[i] = 2.0f * i; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        if (hC[i] != hA[i] + hB[i]) { printf("FAIL at %d\n", i); return 1; }
    printf("PASS\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
```
Talking points: bounds check because grid overshoots; 256 threads = 8 warps, a solid default; ceiling division for blocks.

### A2. SAXPY
```cuda
__global__ void saxpy(float a, const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = a * x[i] + y[i];
}
```
Scalar `a` passes by value — it lands in constant/register space per thread, no memcpy needed.

### A3. ReLU
```cuda
__global__ void relu(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = fmaxf(0.0f, in[i]);
}
```
In-place: pass the same pointer for in/out — safe here because each thread reads and writes ONLY index i (no cross-thread dependency). `fmaxf` avoids a branch (no divergence).

### A4. Naive MatMul
```cuda
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
// launch:
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (N + 15) / 16);
matmulNaive<<<grid, block>>>(dA, dB, dC, N);
```
Weakness: zero data reuse — every thread streams 2N global elements; adjacent threads re-fetch overlapping rows/columns → badly memory-bound.

### A5. Tiled MatMul
```cuda
#define TILE 32
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                       // tiles fully loaded before use

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                       // done reading before next overwrite
    }
    if (row < N && col < N) C[row * N + col] = sum;
}
```
Each global element loaded once per tile, reused TILE times → global traffic cut ~TILE×.
Bank conflicts: in THIS access pattern `As[threadIdx.y][k]` broadcasts (same address per warp — fine) and `Bs[k][threadIdx.x]` is stride-1 (fine). A conflict appears when a warp reads a COLUMN, e.g. `tile[threadIdx.x][k]` — all 32 rows of one column live in one bank since bank = col % 32. Fix: pad to `[TILE][TILE+1]`.

### A5b. Register-Tiled MatMul
**Pseudocode (interviewer-friendly):**
```
register_tiled_matmul(A, B, C, M, N, K):
    // Each thread computes a TM×TN submatrix of C entirely in registers
    allocate regC[TM][TN] = 0    // lives in registers, no memory traffic

    for each tile along K (step BK):
        load BM×BK tile of A → shared memory As      // coalesced global read
        load BK×BN tile of B → shared memory Bs      // coalesced global read
        __syncthreads()

        for k = 0 to BK:                             // inner loop
            load TM values from As column k → regA   // one read per thread row
            load TN values from Bs row k   → regB    // one read per thread col
            for i in TM:
                for j in TN:
                    regC[i][j] += regA[i] * regB[j]  // pure register math
        __syncthreads()

    write regC[TM][TN] → global C                   // one write per output element
```

**CUDA sketch:**
```cuda
#define BM 64   // block tile rows
#define BN 64   // block tile cols
#define BK 16   // block tile depth
#define TM 4    // register tile rows per thread
#define TN 4    // register tile cols per thread

__global__ void matmulRegTiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float regC[TM][TN] = {};      // TM*TN registers per thread
    float regA[TM], regB[TN];

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);

    for (int bk = 0; bk < K; bk += BK) {
        // load tiles into shared memory (each thread loads multiple elements)
        // ... (index math omitted for clarity)
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int tm = 0; tm < TM; tm++)
                regA[tm] = As[threadRow * TM + tm][k];
            for (int tn = 0; tn < TN; tn++)
                regB[tn] = Bs[k][threadCol * TN + tn];
            for (int tm = 0; tm < TM; tm++)
                for (int tn = 0; tn < TN; tn++)
                    regC[tm][tn] += regA[tm] * regB[tn];   // all registers
        }
        __syncthreads();
    }
    // write regC to global C
}
```
**Why it's faster than shared-memory tiling alone:**
- Shared memory tiling (A5): eliminates redundant global reads — each element loaded once per tile, reused BK times. Still hits shared memory on every inner-loop access.
- Register tiling (A5b): the inner `k` loop never touches shared memory — `regA` and `regB` stay in registers. Shared memory is only accessed BK times per tile to load into registers, then pure register arithmetic for TM×TN×BK MACs.
- Result: shared mem bandwidth drops by TM×TN factor. This is why you get 5.65× on top of shared memory tiling — the inner loop is now fully register-resident, matching what cuBLAS does internally.
- Each thread computes TM×TN=16 outputs instead of 1 → amortizes index computation and load overhead across more useful work (instruction-level efficiency).

Key numbers to remember: shared tiling alone ~170 GFLOPS on sm_87, adding register tiling → ~968 GFLOPS (5.65×), Tensor Cores → ~1777 GFLOPS (10.4× total).

### A6. Transpose (coalesced both directions)
```cuda
#define TILE 32
__global__ void transpose(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE + 1];       // +1 kills bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // coalesced read
    __syncthreads();

    // swap block coordinates for output; threads still walk x for coalesced write
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // transposed read from tile
}
```
Naive version (`out[x*N+y] = in[y*N+x]`) coalesces the read but the write jumps stride-N per thread — uncoalesced. Trick: do the transpose INSIDE shared memory so both global accesses stay row-wise. The `tile[threadIdx.x][threadIdx.y]` read is a column read → would be a 32-way bank conflict → the `+1` padding shifts each row's bank mapping so it isn't.

### A7. Reduction (sum)
```cuda
__global__ void reduceSum(const float* in, float* out, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
```
(a) Sequential addressing keeps ACTIVE threads contiguous — warps 0..k fully active, rest fully idle → no intra-warp divergence. Interleaved (`tid % (2*s) == 0`) scatters active threads across every warp → all warps partially masked, all still scheduled.
(b) Combine partials: launch again on the per-block outputs until 1 remains, or `atomicAdd(out, sdata[0])` from thread 0.
(c) Last 32: warp shuffle — no shared memory or syncs needed within a warp:
```cuda
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

### A8. Dot Product
```cuda
__global__ void dot(const float* A, const float* B, float* result, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? A[i] * B[i] : 0.0f;   // fused multiply + load
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);   // result must be zeroed first!
}
```
Gotcha to mention: `cudaMemset(result, 0, sizeof(float))` before launch; float atomicAdd makes results non-deterministic in last bits (FP addition isn't associative) — fine for most uses, worth saying you know.

### A9. Grid-Stride Loop
```cuda
__global__ void vecAddGS(const float* A, const float* B, float* C, int N) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        C[i] = A[i] + B[i];
}
// launch with a FIXED reasonable grid, e.g. <<<256, 256>>>, for ANY N
```
Why preferred: decouples grid size from N (works when N exceeds max grid dims), one kernel handles any size, threads do multiple elements (amortizes index math), and you can size the grid to exactly fill the GPU (one block per SM slot) instead of oversubscribing.

### A10. Row-wise Softmax (stable)
```cuda
__global__ void softmaxRow(const float* in, float* out, int M, int N) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + row * N;
    float* y = out + row * N;

    // 1) row max
    float m = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) m = fmaxf(m, x[j]);
    sdata[tid] = m; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    m = sdata[0]; __syncthreads();

    // 2) sum of exp(x - max)
    float sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) sum += expf(x[j] - m);
    sdata[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0]; __syncthreads();

    // 3) write
    for (int j = tid; j < N; j += blockDim.x) y[j] = expf(x[j] - m) / sum;
}
// launch: softmaxRow<<<M, 256>>>(in, out, M, N);
```
Subtract max because `expf(large)` overflows to inf → nan. Shifting by the max makes the largest exponent exp(0)=1; mathematically identical (factor cancels in the ratio). This two-pass max+sum is also the seed idea of ONLINE softmax (fused attention) — one pass with running max/sum — good thing to volunteer given your Triton attention kernel.

### A11. Histogram
```cuda
// Version 1: global atomics — simple, contended
__global__ void hist(const unsigned char* data, int* bins, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(&bins[data[i]], 1);
}

// Version 2: shared-memory privatization — each block builds a private
// histogram, then merges once. ~1 global atomic per bin per block instead
// of per element.
__global__ void histPriv(const unsigned char* data, int* bins, int N) {
    __shared__ int local[256];
    for (int j = threadIdx.x; j < 256; j += blockDim.x) local[j] = 0;
    __syncthreads();

    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        atomicAdd(&local[data[i]], 1);          // shared-mem atomics: cheap
    __syncthreads();

    for (int j = threadIdx.x; j < 256; j += blockDim.x)
        atomicAdd(&bins[j], local[j]);          // one merge per bin per block
}
```
Race: many threads incrementing the same bin — read-modify-write must be atomic. Privatization moves the contention into fast shared memory and slashes global atomic traffic.

### A12. Bugs in `scale`
1. **No global index** — `int i = threadIdx.x` only; every block writes the same first-blockDim elements. Need `blockIdx.x * blockDim.x + threadIdx.x`.
2. **`<<<1, N>>>` breaks for N > 1024** — max threads per block. Need multi-block launch with ceiling division.
3. **No bounds check** — with a proper multi-block launch, the last block overshoots; need `if (i < N)`.
Fixed:
```cuda
__global__ void scale(float* data, int N, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] *= factor;
}
int t = 256; scale<<<(N + t - 1) / t, t>>>(d_data, N, 2.0f);
```

### A13. Bugs in `avgPool`
1. **`__syncthreads()` inside a divergent branch** (`if (threadIdx.x < 128)`) — threads 128–255 never reach the barrier → deadlock/UB. The barrier must be OUTSIDE, before the if.
2. **Unguarded global read** — `in[i]` with no `i < N` check reads out of bounds in the last block.
Fixed:
```cuda
__global__ void avgPool(const float* in, float* out, int N) {
    __shared__ float tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (i < N) ? in[i] : 0.0f;
    __syncthreads();                              // ALL threads hit this
    if (threadIdx.x < 128 && i < N)
        out[i] = (tile[threadIdx.x] + tile[threadIdx.x + 128]) * 0.5f;
}
```

---

## Part 2 — Plain C/C++

### A14. Loop-order MatMul with timing
```cpp
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

const int N = 512;

void mm_ijk(const float* A, const float* B, float* C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += A[i*N+k] * B[k*N+j];  // B: stride-N
            C[i*N+j] = s;
        }
}
void mm_ikj(const float* A, const float* B, float* C) {
    memset(C, 0, N*N*sizeof(float));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i*N+k];
            for (int j = 0; j < N; j++) C[i*N+j] += a * B[k*N+j];  // both stride-1
        }
}
void mm_jik(const float* A, const float* B, float* C) {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += A[i*N+k] * B[k*N+j];  // B: stride-N
            C[i*N+j] = s;
        }
}

template <typename F> double timeit(F f, const float* A, const float* B, float* C) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f(A, B, C);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    std::vector<float> A(N*N), B(N*N), C(N*N);
    for (int i = 0; i < N*N; i++) { A[i] = i % 7; B[i] = i % 5; }
    printf("ijk: %.1f ms\n", timeit(mm_ijk, A.data(), B.data(), C.data()));
    printf("ikj: %.1f ms\n", timeit(mm_ikj, A.data(), B.data(), C.data()));
    printf("jik: %.1f ms\n", timeit(mm_jik, A.data(), B.data(), C.data()));
}
```
Prediction: **ikj wins** — its inner loop reads B and writes C at stride-1 (full cache-line use, vectorizes). ijk/jik read B down a column: stride-N, a miss per element at large N. Identical FLOPs, several-fold time difference — pure memory behavior.

### A15. Blocked CPU MatMul
```cpp
const int BS = 64;
void mm_blocked(const float* A, const float* B, float* C) {
    memset(C, 0, N*N*sizeof(float));
    for (int ii = 0; ii < N; ii += BS)
    for (int kk = 0; kk < N; kk += BS)
    for (int jj = 0; jj < N; jj += BS)
        for (int i = ii; i < ii + BS && i < N; i++)
        for (int k = kk; k < kk + BS && k < N; k++) {
            float a = A[i*N+k];
            for (int j = jj; j < jj + BS && j < N; j++)
                C[i*N+j] += a * B[k*N+j];
        }
}
```
Why it helps even with ikj order: at large N a whole row of B/C no longer fits in L1/L2, so rows get evicted between reuses. Blocking keeps a BS×BS working set resident across its reuse window — same idea as GPU shared-memory tiling, one level up the hierarchy.

### A16. Transpose: naive vs blocked
```cpp
void transpose_naive(float* a, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            std::swap(a[i*n + j], a[j*n + i]);   // second access: stride-n
}
void transpose_blocked(float* a, int n) {
    const int B = 32;
    for (int ii = 0; ii < n; ii += B)
        for (int jj = ii; jj < n; jj += B)
            for (int i = ii; i < ii + B && i < n; i++)
                for (int j = (ii == jj ? i + 1 : jj); j < jj + B && j < n; j++)
                    std::swap(a[i*n + j], a[j*n + i]);
}
```
Naive: one side of the swap walks a column — a miss per element at n=4096 (row stride 16KB >> cache line). Blocked: both the row-side and column-side touches stay within a 32×32 tile pair that fits in L1, so each fetched line is fully used before eviction.

### A17. Pool Allocator
```c
#include <stddef.h>
#include <stdint.h>

#define POOL_BLOCKS 1024
#define BLOCK_SIZE  64        /* >= sizeof(void*) and cache-line friendly */

static _Alignas(64) unsigned char pool[POOL_BLOCKS * BLOCK_SIZE];
static void* free_list = NULL;
static int initialized = 0;

static void pool_init(void) {
    free_list = NULL;
    for (int i = POOL_BLOCKS - 1; i >= 0; i--) {
        void* block = pool + (size_t)i * BLOCK_SIZE;
        *(void**)block = free_list;     /* intrusive next pointer */
        free_list = block;
    }
    initialized = 1;
}
void* pool_alloc(void) {
    if (!initialized) pool_init();
    if (!free_list) return NULL;
    void* block = free_list;
    free_list = *(void**)block;
    return block;
}
void pool_free(void* block) {
    if (!block) return;
    *(void**)block = free_list;
    free_list = block;
}
```
Why embedded/RT systems use pools: O(1) deterministic alloc/free (no worst-case allocator walk), zero external fragmentation, no syscalls/locks, memory budget fixed at compile time. Alignment: blocks should be aligned to max(alignof(max_align_t), 64 for cache-line use) — here the pool itself is 64-aligned and BLOCK_SIZE is a multiple of it, so every block is 64-aligned.

### A18. Aligned alloc from malloc
```c
#include <stdlib.h>
#include <stdint.h>

void* aligned_alloc64(size_t n) {
    void* raw = malloc(n + 64 + sizeof(void*));
    if (!raw) return NULL;
    uintptr_t base = (uintptr_t)raw + sizeof(void*);
    uintptr_t aligned = (base + 63) & ~(uintptr_t)63;   /* round up to 64 */
    ((void**)aligned)[-1] = raw;                        /* stash original */
    return (void*)aligned;
}
void aligned_free64(void* p) {
    if (p) free(((void**)p)[-1]);
}
```
Why alignment matters: aligned SIMD loads (e.g. 32/64-byte vectors) are single instructions on one cache line; misaligned ones may split across two lines/pages. 64-byte alignment also guarantees a struct starts at a cache-line boundary — the false-sharing fix.

### A19. Reverse linked list
```c
struct Node { int val; struct Node* next; };

struct Node* reverse_iter(struct Node* head) {
    struct Node* prev = NULL;
    while (head) {
        struct Node* next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
struct Node* reverse_rec(struct Node* head) {
    if (!head || !head->next) return head;
    struct Node* new_head = reverse_rec(head->next);
    head->next->next = head;
    head->next = NULL;
    return new_head;
}
```
Recursive risk: one stack frame per node — a million-node list blows the (typically 1–8MB) stack. Not tail-recursive, so the compiler can't fix it. Iterative is O(1) space; always the production answer.

### A20. memcpy from scratch
```c
#include <stddef.h>
#include <stdint.h>

void* my_memcpy_naive(void* dst, const void* src, size_t n) {
    unsigned char* d = dst; const unsigned char* s = src;
    while (n--) *d++ = *s++;
    return dst;
}

void* my_memcpy_word(void* dst, const void* src, size_t n) {
    unsigned char* d = dst; const unsigned char* s = src;
    /* head: byte-copy until dst is word-aligned */
    while (n && ((uintptr_t)d & (sizeof(size_t) - 1))) { *d++ = *s++; n--; }
    /* body: word-at-a-time */
    size_t* dw = (size_t*)d; const size_t* sw = (const size_t*)s;
    while (n >= sizeof(size_t)) { *dw++ = *sw++; n -= sizeof(size_t); }
    /* tail: remaining bytes */
    d = (unsigned char*)dw; s = (const unsigned char*)sw;
    while (n--) *d++ = *s++;
    return dst;
}
```
Word version: 8 bytes per load/store instead of 1 → ~8x fewer instructions and memory ops; aligned word accesses use the cache optimally. (Real libc goes further: SIMD, rep movsb, size dispatch — worth mentioning you know yours is the teaching version.)

### A21. Popcount three ways
```c
int pc_naive(unsigned n)  { int c = 0; while (n) { c += n & 1; n >>= 1; } return c; }
int pc_kern(unsigned n)   { int c = 0; while (n) { n &= n - 1; c++; }    return c; }  /* one iter per SET bit */
static unsigned char T[256];  /* init once: T[i] = pc_naive(i) */
int pc_table(unsigned n) {
    return T[n & 0xff] + T[(n >> 8) & 0xff] + T[(n >> 16) & 0xff] + T[n >> 24];
}
```
Naive: 32 iterations always. Kernighan: iterations = popcount — wins on sparse values. Table: constant 4 lookups — wins in bulk loops (table stays hot in L1). Real answer in production: `__builtin_popcount` → a single hardware instruction; say you know that.

### A22. Struct layout math
```c
struct S { char a; double b; int c; char d; };
```
On 64-bit, 8-byte-aligned double: a@0 (1) → pad 7 → b@8 (8) → c@16 (4) → d@20 (1) → tail pad 3 (struct align = 8) = **24 bytes**.
Reordered largest-first:
```c
struct S2 { double b; int c; char a; char d; };  /* 8 + 4 + 1 + 1 = 14 → pad to 16 */
```
= **16 bytes** — a third smaller; in an array of millions, a third less cache traffic.

### A23. False sharing demo
```cpp
#include <thread>
#include <chrono>
#include <cstdio>

struct Padded { alignas(64) long v; };   // one cache line each
long shared_arr[2];                      // adjacent — same cache line
Padded padded_arr[2];

template <typename A> double run(A* arr) {
    auto work = [&](int idx) {
        for (long i = 0; i < 100'000'000; i++) arr[idx].v++;
    };
    auto t0 = std::chrono::high_resolution_clock::now();
    std::thread t1([&]{ work(0); }), t2([&]{ work(1); });
    t1.join(); t2.join();
    auto t1e = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1e - t0).count();
}
// (wrap shared_arr in a struct with .v, or duplicate the lambda, to reuse run)
```
Adjacent counters share one 64-byte line; each increment invalidates the other core's copy → the line ping-pongs on every write (several-fold slowdown is typical). `alignas(64)` gives each counter its own line — no logical sharing, now no physical sharing either.

### A24. Branchless max / abs
```c
int bmax(int a, int b) { int d = a - b; return b + (d & ~(d >> 31)); } /* caution: a-b can overflow */
int babs(int x)        { int m = x >> 31; return (x ^ m) - m; }        /* m = 0 or -1 */
```
`x >> 31` (arithmetic shift) smears the sign bit: 0 for positive, all-1s for negative. abs: XOR with all-1s = bitwise NOT, minus (-1) = +1 → two's-complement negate, only when negative.
When worth it: unpredictable data in a hot loop, where a mispredict (~15+ cycles) beats 2-3 ALU ops. When NOT: predictable branches (predictor is nearly free) — and modern compilers emit `cmov` for the ternary anyway. Honest answer: write the clear version, check the assembly, only hand-optimize when the profiler says so.

### A25. Ring buffer
```c
#include <stddef.h>
#define CAP 256                      /* power of two */
typedef struct {
    int data[CAP];
    size_t head, tail;               /* head = next write, tail = next read */
} Ring;

int ring_empty(const Ring* r) { return r->head == r->tail; }
int ring_full (const Ring* r) { return r->head - r->tail == CAP; }

int ring_push(Ring* r, int v) {
    if (ring_full(r)) return 0;
    r->data[r->head & (CAP - 1)] = v;   /* mask instead of modulo */
    r->head++;
    return 1;
}
int ring_pop(Ring* r, int* out) {
    if (ring_empty(r)) return 0;
    *out = r->data[r->tail & (CAP - 1)];
    r->tail++;
    return 1;
}
```
Power-of-two + mask: `x & (CAP-1)` is one AND; `%` on a non-constant divisor is a division — tens of cycles. Free-running head/tail (no wrap) also makes full/empty unambiguous without wasting a slot. This is the standard SPSC queue skeleton — the same structure appears in driver/DSP runtimes, relevant to their embedded angle.

---

## Part 3 — Rapid-fire answers

**R1.** `int i = blockIdx.x * blockDim.x + threadIdx.x;`
**R2.** `int blocks = (N + threads - 1) / threads;`
**R3.** `kernel<<<grid, block, sharedBytes, stream>>>(args);` (sharedBytes and stream optional, default 0)
**R4.** Static: `__shared__ float buf[48];` Dynamic: `extern __shared__ float buf[];` + pass size as the 3rd launch parameter.
**R5.** `cudaMemcpy(dst, src, bytes, kind)`; kinds: `cudaMemcpyHostToDevice`, `DeviceToHost`, `DeviceToDevice`, `HostToHost` — you use the first three.
**R6.** Blocks the host until all prior GPU work completes. Kernel launches are async — you need it (or an implicit sync like a blocking memcpy) before trusting results or timing on the host side.
**R7.** `cudaError_t err = cudaGetLastError();` right after launch (catches launch-config errors); `cudaDeviceSynchronize()`'s return catches execution errors.
**R8.** A 16-byte vector type (4 floats). Loading via `float4` issues one 128-bit transaction per thread — fewer instructions, wider transactions, better bandwidth utilization.
**R9.** `__global__`: kernel — runs on device, called from host (launch syntax). `__device__`: runs on device, callable only from device code. `__host__`: normal CPU function (default; combine `__host__ __device__` to compile for both).
**R10.** `index = row * N + col` (N = number of columns).

---

## Part 4 — Parallel C++, SIMD, and Compiler Awareness

### A26. Multithreaded Reduction
```cpp
#include <thread>
#include <vector>
#include <numeric>

double parallel_sum(const float* a, size_t n, int T) {
    std::vector<std::thread> threads;
    std::vector<double> partial(T, 0.0);          // one slot per thread — no sharing

    size_t chunk = (n + T - 1) / T;
    for (int t = 0; t < T; t++) {
        threads.emplace_back([=, &partial] {
            size_t lo = t * chunk, hi = std::min(n, lo + chunk);
            double s = 0.0;                        // LOCAL accumulator (register-resident)
            for (size_t i = lo; i < hi; i++) s += a[i];
            partial[t] = s;                        // one write at the end
        });
    }
    for (auto& th : threads) th.join();
    return std::accumulate(partial.begin(), partial.end(), 0.0);
}
```
Why not one shared accumulator + mutex: every add serializes through the lock — you've built a slower single-threaded program with overhead. Why not std::atomic: no lock, but every increment still bounces the cache line between cores (contention) — barely better. Partial sums: zero synchronization in the hot loop, one combine at the end. This is exactly the GPU reduction philosophy — private work, merge once. (Note: `partial[t]` slots are adjacent → mild false sharing on the FINAL write only; pad if you want to be thorough.)

### A27. Parallel MatMul
```cpp
void mm_parallel(const float* A, const float* B, float* C, int N, int T) {
    std::vector<std::thread> threads;
    int rows_per = (N + T - 1) / T;
    for (int t = 0; t < T; t++) {
        threads.emplace_back([=] {
            int lo = t * rows_per, hi = std::min(N, lo + rows_per);
            for (int i = lo; i < hi; i++) {
                for (int j = 0; j < N; j++) C[i*N + j] = 0;
                for (int k = 0; k < N; k++) {
                    float a = A[i*N + k];
                    for (int j = 0; j < N; j++) C[i*N + j] += a * B[k*N + j];
                }
            }
        });
    }
    for (auto& th : threads) th.join();
}
```
Rows of C = disjoint OUTPUT ownership: each thread writes its own rows, reads shared A/B (reads never race). Splitting k instead means two threads BOTH do `C[i*N+j] +=` on the same element — a read-modify-write race → wrong results (or you'd need atomics/locks in the innermost loop, killing performance). Principle (same as CUDA): partition by output, never by reduction dimension.

### A28. Parallelizable?
(a) **Yes** — each i independent. Embarrassingly parallel (this is a GPU map).
(b) **No as written** — loop-carried dependency: a[i] needs a[i-1] from THIS loop. This is a prefix sum → parallelizable with the scan algorithm (Q30), not by naive splitting.
(c) **Yes with care** — reduction: dependency is on `sum` only. Partial sums per thread + combine (A26), or `omp reduction`.
(d) **Yes with care** — different i may hit the same bucket → race on hist. Privatize: per-thread histograms, merge at the end (the CPU version of A11's shared-memory privatization).
Pattern names to say out loud: map / scan / reduction / histogram — knowing which pattern a loop is IS the parallelization skill.

### A29. CPU Stable Softmax
```cpp
#include <cmath>
#include <algorithm>

void softmax(const float* x, float* y, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) m = std::max(m, x[i]);    // pass 1: max

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {                          // pass 2: exp + sum
        y[i] = std::exp(x[i] - m);
        sum += y[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) y[i] *= inv;               // pass 3: normalize
}
```
Max subtraction: exp(large) overflows to inf → inf/inf = nan. Shifting makes the largest term exp(0)=1; the shift cancels in the ratio, so the math is identical. Two-pass fusion: pass 2 already stores exp values in y, so passes 2+3 only fuse if you accept re-reading y (that's what's written). The ONE-pass version is online softmax — maintain running max and rescale the running sum when max updates — the algorithm inside FlashAttention and my Triton attention kernel. Volunteer that connection.

### A30. Prefix Sum
```cpp
// sequential inclusive scan
void scan(const float* in, float* out, int n) {
    float acc = 0;
    for (int i = 0; i < n; i++) { acc += in[i]; out[i] = acc; }
}
```
Parallel two-pass: (1) split into T chunks; each thread scans its chunk locally and records its chunk TOTAL; (2) sequentially (or recursively) scan the T totals to get each chunk's OFFSET; (3) each thread adds its offset to its chunk. Harder than reduction because every output element depends on ALL prior inputs — reduction collapses to one value (order-free combine), scan must preserve per-position results, so it costs two sweeps over the data instead of one.

### A31. Dot Product Precision + ILP
```cpp
double dot(const float* a, const float* b, size_t n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;       // 4 independent accumulators
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += (double)a[i]   * b[i];
        s1 += (double)a[i+1] * b[i+1];
        s2 += (double)a[i+2] * b[i+2];
        s3 += (double)a[i+3] * b[i+3];
    }
    for (; i < n; i++) s0 += (double)a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}
```
Precision: float has ~7 digits; once the running sum is large, small products vanish (absorption). Fixes: accumulate in double (shown), or Kahan compensated summation, or pairwise/tree summation. Why 4 accumulators is faster even single-threaded: one accumulator makes every add depend on the previous (a serial chain gated by FP-add latency, ~4 cycles); independent accumulators fill the pipeline — instruction-level parallelism. Same latency-hiding idea as register tiling on GPU.

### A32. Aliasing
As written the compiler must assume c might overlap a or b — writing c[i] could change a[i+1] — so it can't freely batch loads ahead of stores; it emits runtime overlap checks or conservative code.
```cpp
void add(const float* __restrict__ a, const float* __restrict__ b,
         float* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}
```
`__restrict__` = "through this pointer is the ONLY way this memory is accessed here." Compiler can now vectorize cleanly, keep values in registers, reorder freely. If you lie (pass overlapping pointers): undefined behavior — silently wrong results, no diagnostic. Also note: making a and b `const` helps the human; `restrict` is what helps the optimizer.

### A33. Vectorization Blockers
(a) **Vectorizes** — pure elementwise map, unit stride. (Aliasing caveat from A32 may add a runtime check.)
(b) **Vectorizes at -O3 usually** — the if converts to a mask/blend (predication), sum via vector reduction. Say: "branches become masks."
(c) **Does not vectorize (or poorly)** — scatter with possible duplicate indices: two lanes hitting the same x[idx] is a dependence the compiler can't rule out. This is the histogram problem again.
(d) **Does not vectorize** — opaque external call: unknown side effects, not vectorizable unless f gets inlined or has a declared SIMD variant (`#pragma omp declare simd`).

### A34. Templated axpy
```cpp
template <typename T>
void axpy(T a, const T* __restrict__ x, T* __restrict__ y, int n) {
    for (int i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}
// axpy<float>(2.0f, xf, yf, n);  axpy<double>(2.0, xd, yd, n);
```
vs void* + element size (C's qsort style): templates stamp out a fully typed copy per T — direct arithmetic, inlinable, vectorizable. The void* version does opaque byte shuffling through function pointers — no inlining, no vectorization, runtime dispatch per element. Cost of templates: one compiled copy per instantiation (code size, compile time) and header-visible implementations. For kernels, always templates — this is exactly why BLAS libraries have s/d/c/z variants.

### A35. vector vs raw vs stack; move
- `std::vector`: default for owned, sized-at-runtime data — RAII (no leaks), bounds-checkable, contiguous (data() interops with any kernel API). Zero per-element overhead vs new[].
- `new float[n]`: only when interfacing with code that demands manual ownership; otherwise strictly worse than vector.
- Stack array (`float buf[256]`): small fixed scratch — fastest allocation (free), but limited size and scope.
- `std::move(v)`: transfers the internal pointer/size/capacity (3 words copied) and nulls the source — NO element copy. O(1) vs O(n). Use when handing a big buffer to another owner. Kernel-relevant framing: vector for buffers at the edges, raw pointers (from .data()) inside hot loops.

### A36. Const Correctness
```cpp
float dot(const float* a, const float* b, int n);
```
- `const float*` on both inputs: function can't modify the arrays — compiler enforces the contract, caller can pass const data.
- Optimization honesty: pointer-to-const mostly helps CORRECTNESS/readability; the compiler often can't assume the data doesn't change through other aliases anyway (that's restrict's job). Where const genuinely enables optimization: const local values (foldable), and combined with restrict.
- Say the pairing: "const documents intent, restrict unlocks the optimizer."

### A37. -O Levels & Reading Output
- **-O0**: no optimization; every variable lives in memory (debuggable, slow).
- **-O2**: the production default — inlining, loop-invariant code motion, common-subexpression elimination, dead-code elimination, strength reduction, auto-vectorization (gcc 12+; clang vectorizes at -O2 historically), unrolling where profitable.
- **-O3**: -O2 plus aggressive vectorization/unrolling, function cloning — usually faster, occasionally slower (code bloat, i-cache pressure).
- **-march=native**: emit for THIS CPU's ISA (AVX2/AVX-512) instead of a lowest-common-denominator baseline — often the biggest single flag for numeric kernels.
- Looking at output: `g++ -S -O2 file.cpp` (assembly to file.s), or Compiler Explorer (godbolt.org) interactively. Three -O2 examples to name: inlining, loop-invariant code motion, dead-code elimination.

### A38. Inlining & Unrolling
After inlining, the callee's body is visible IN the caller's context, which unlocks: constant propagation into the body (n known → loop fully unrollable), removing argument/return overhead, vectorizing across what was a call boundary, deleting branches that are dead for these arguments. Backfires when: big functions inlined into many sites → code bloat → instruction-cache misses; or unrolling past the point where the loop body exceeds the µop cache. Rule of thumb to say: "inlining is the gateway optimization — most others need it first; that's why opaque function calls kill vectorization (A33d)."

### A39. OpenMP
```cpp
// reduction
double sum = 0;
#pragma omp parallel for reduction(+:sum)
for (long i = 0; i < n; i++) sum += a[i];

// matmul (rows of C partitioned across threads automatically)
#pragma omp parallel for
for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++) {
        float x = A[i*N + k];
        for (int j = 0; j < N; j++) C[i*N + j] += x * B[k*N + j];
    }
```
`reduction(+:sum)`: each thread gets a PRIVATE zero-initialized copy of sum, loops on it locally, and OpenMP combines the copies at the barrier — literally A26's partial-sum pattern, generated for you. `schedule(static)`: iterations pre-split into equal contiguous chunks — zero runtime overhead, right answer when iterations cost the same (matmul rows). `schedule(dynamic[,chunk])`: threads grab chunks from a queue — overhead per grab, but wins when iteration cost varies (irregular work). Compile: `g++ -O2 -fopenmp`.

### A40. Sync Cost Ladder
Fastest → slowest: **(c) thread-local + combine** ≫ **(b) atomic** > **(a) mutex**.
- Mutex per increment: lock/unlock (possible syscall on contention) around a 1-cycle op — hundreds of cycles each, plus serialization.
- Atomic: no lock, but the cache line ping-pongs between cores on every increment — tens of cycles under contention.
- Thread-local: the hot loop touches only a register/private line — full speed; one combine at the end.
General principle (say this — it's the whole philosophy): **structure parallel work so synchronization happens O(threads) times, not O(work) times.** Private accumulate, merge once. It's A26, it's omp reduction, it's the GPU block reduction, it's histogram privatization — one idea wearing four costumes.

---

## Part 3b — Rapid-fire answers

**R11.** `g++ -S -O2 file.cpp` (writes file.s), or paste into Compiler Explorer (godbolt.org). (`objdump -d a.out` for a built binary.)
**R12.** `g++ -O2 -march=native -fopenmp -pthread file.cpp -o prog`
**R13.** Forces 64-byte alignment (cache-line). Uses: give each thread's data its own cache line (kill false sharing); align buffers for SIMD.
**R14.** `std::atomic<int>` — lock-free hardware RMW for a simple counter, no lock overhead or blocking. Mutex only when protecting MULTIPLE variables that must change together.
**R15.** gcc: `-fopt-info-vec` (or `-fopt-info-vec-missed` for why NOT — the more useful one). clang: `-Rpass=loop-vectorize` / `-Rpass-missed=loop-vectorize`.

---

## Part 5 — Quantization (Inference Kernels)

*Directly relevant for NPU/edge AI roles. All code is plain C++ unless noted.*

---

### A41. Symmetric Per-Tensor Quantization (float → int8)
```cpp
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

// Compute scale from max absolute value; zero_point = 0 (symmetric)
float compute_scale_symmetric(const float* x, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) max_abs = std::max(max_abs, std::abs(x[i]));
    return max_abs / 127.0f;   // INT8 range: [-127, 127] (leave -128 unused for symmetry)
}

void quantize_symmetric(const float* x, int8_t* q, int n, float scale) {
    for (int i = 0; i < n; i++) {
        float val = std::round(x[i] / scale);
        q[i] = (int8_t)std::max(-127.0f, std::min(127.0f, val));   // clamp
    }
}

void dequantize_symmetric(const int8_t* q, float* x, int n, float scale) {
    for (int i = 0; i < n; i++) x[i] = q[i] * scale;
}
```
Key points: scale = max_abs / 127 maps the float range onto [-127, 127]. -128 is intentionally avoided so that negation of any value stays in range (important for symmetric math). Rounding before clamping reduces quantization error vs truncating. Dequant is just a multiply — cheap at inference.

---

### A42. Asymmetric Per-Tensor Quantization (float → uint8)
```cpp
struct QuantParams { float scale; int32_t zero_point; };

QuantParams compute_params_asymmetric(const float* x, int n) {
    float x_min = x[0], x_max = x[0];
    for (int i = 1; i < n; i++) { x_min = std::min(x_min, x[i]); x_max = std::max(x_max, x[i]); }
    float scale = (x_max - x_min) / 255.0f;                          // uint8: [0, 255]
    int32_t zp = (int32_t)std::round(-x_min / scale);
    zp = std::max(0, std::min(255, zp));
    return {scale, zp};
}

void quantize_asymmetric(const float* x, uint8_t* q, int n, QuantParams p) {
    for (int i = 0; i < n; i++) {
        int32_t val = (int32_t)std::round(x[i] / p.scale) + p.zero_point;
        q[i] = (uint8_t)std::max(0, std::min(255, val));
    }
}

void dequantize_asymmetric(const uint8_t* q, float* x, int n, QuantParams p) {
    for (int i = 0; i < n; i++) x[i] = p.scale * (q[i] - p.zero_point);
}
```
Asymmetric vs symmetric: asymmetric shifts the zero point to cover the actual data range (critical for ReLU outputs which are all non-negative — symmetric wastes half the range). Cost: zero_point correction must be accounted for during GEMM (extra bias term per row). Symmetric is simpler for weights; asymmetric is better for activations.

---

### A43. Quantized Dot Product (INT8 → INT32 accumulation)
```cpp
// Symmetric INT8 inputs, accumulate in INT32 to avoid overflow, then scale to float
float qdot(const int8_t* a, const int8_t* b, int n,
           float scale_a, float scale_b) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++)
        acc += (int32_t)a[i] * (int32_t)b[i];   // INT8*INT8 → INT32: no overflow up to n~16M
    return acc * scale_a * scale_b;              // single dequant at the end
}
```
Why accumulate in INT32: INT8 * INT8 → up to 127*127 = 16129 per term; summing n=4096 terms → ~66M, which overflows INT16 (max 32767) but fits INT32 comfortably. The key insight: defer dequantization to AFTER the accumulation — if you dequant each product to float first, you lose the throughput benefit of integer arithmetic entirely. Scale product `scale_a * scale_b` can be precomputed once.

---

### A44. Quantized MatMul (INT8 weights × INT8 activations → INT32 → float)
```cpp
// C[M][N] = A[M][K] * B[K][N], all INT8, output float
// Symmetric quantization assumed (zero_point = 0)
void qmatmul(const int8_t* A, const int8_t* B, float* C,
             int M, int N, int K,
             float scale_a, float scale_b) {
    float output_scale = scale_a * scale_b;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];
            C[i*N + j] = acc * output_scale;
        }
    }
}
```
This is the skeleton of every quantized GEMM (TFLite, ONNX Runtime, QNN). On NPUs/DSPs the inner loop runs as INT8 MAC units which give 4× throughput over FP32 MACs at the same area. The accumulator stays INT32 until the final store — requantize to INT8 if feeding another quantized layer.

---

### A45. Requantization (INT32 → INT8, layer-to-layer)
```cpp
// After a quantized GEMM the accumulator is INT32.
// Requantize to INT8 to feed the next layer without going back to float.
// multiplier = (scale_input * scale_weight) / scale_output
void requantize(const int32_t* acc, int8_t* out, int n,
                float multiplier, int32_t output_zero_point = 0) {
    for (int i = 0; i < n; i++) {
        float val = acc[i] * multiplier + output_zero_point;
        val = std::round(val);
        out[i] = (int8_t)std::max(-127.0f, std::min(127.0f, val));
    }
}
```
Why requantize instead of dequant→float→quant: keeps the entire inference in integer domain — no float memory traffic between layers, critical for edge/NPU where FP units may not exist. In practice the multiplier is approximated as a fixed-point integer + shift (e.g., TFLite's `MultiplyByQuantizedMultiplier`) to avoid any floats at all.

---

### A46. Per-Channel Quantization (weights)
```cpp
// Each output channel gets its own scale — much better accuracy than per-tensor for weights
void quantize_per_channel(const float* W, int8_t* Wq,
                          float* scales, int out_ch, int in_ch) {
    for (int oc = 0; oc < out_ch; oc++) {
        const float* row = W + oc * in_ch;
        float max_abs = 0.0f;
        for (int ic = 0; ic < in_ch; ic++) max_abs = std::max(max_abs, std::abs(row[ic]));
        scales[oc] = max_abs / 127.0f;
        for (int ic = 0; ic < in_ch; ic++) {
            float v = std::round(row[ic] / scales[oc]);
            Wq[oc * in_ch + ic] = (int8_t)std::max(-127.0f, std::min(127.0f, v));
        }
    }
}
```
Why per-channel: different output channels of a weight matrix can have very different magnitude distributions. A single global scale undershoots the small channels (losing precision) while clamping the large ones (losing range). Per-channel gives each filter its own scale → much tighter quantization error. Cost: during GEMM, each output row is multiplied by a different `scales[oc] * scale_activation` — one extra multiply per output element (negligible).

---

### A47. Fused Quantized ReLU (clamp in integer domain)
```cpp
// ReLU after a quantized layer: just clamp to [0, 127] for symmetric INT8
// No dequant needed — integer clamp IS ReLU in the quantized domain
void qrelu_symmetric(int8_t* x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] < 0 ? 0 : x[i];
}

// For asymmetric uint8: ReLU maps to clamping at zero_point (represents 0.0f)
void qrelu_asymmetric(uint8_t* x, int n, uint8_t zero_point) {
    for (int i = 0; i < n; i++) if (x[i] < zero_point) x[i] = zero_point;
}
```
This is operator fusion in the quantized domain — the activation is folded into the requantization step with no extra pass. For symmetric INT8: 0.0f maps to integer 0, so ReLU = clamp negatives to 0. For asymmetric: 0.0f maps to zero_point, so ReLU = clamp values below zero_point. Both are a branchless max — hardware does this in a single cycle.

---

### A48. Weight-Only Quantization (INT4, for LLM inference)
```cpp
// Pack two INT4 weights into one byte — common in LLM weight-only quant (GPTQ, AWQ style)
void pack_int4(const int8_t* weights, uint8_t* packed, int n) {
    // n must be even; weights assumed in [-8, 7]
    for (int i = 0; i < n / 2; i++)
        packed[i] = ((weights[2*i] & 0x0F) | ((weights[2*i+1] & 0x0F) << 4));
}

void unpack_int4(const uint8_t* packed, int8_t* weights, int n) {
    for (int i = 0; i < n / 2; i++) {
        weights[2*i]   = (int8_t)(packed[i] & 0x0F);
        weights[2*i+1] = (int8_t)((packed[i] >> 4) & 0x0F);
        // sign-extend from 4 bits
        if (weights[2*i]   & 0x08) weights[2*i]   |= 0xF0;
        if (weights[2*i+1] & 0x08) weights[2*i+1] |= 0xF0;
    }
}
```
Why INT4 for LLMs: weights dominate memory bandwidth in autoregressive decoding (each token does a full weight load for one output). INT4 halves the weight size vs INT8 → halves memory bandwidth → near 2× token throughput on memory-bound inference. Activations typically stay FP16/INT8; only weights are INT4 (weight-only quant). Sign-extension from 4-bit: bit 3 is the sign bit; if set, OR with 0xF0 to extend to INT8.

---

### A49. Quantization Error Analysis (coding exercise style)
```cpp
// Measure SNR (signal-to-noise ratio) between float and quantized-dequantized
#include <cmath>
double quantization_snr(const float* original, const float* reconstructed, int n) {
    double signal = 0.0, noise = 0.0;
    for (int i = 0; i < n; i++) {
        signal += (double)original[i] * original[i];
        double err = original[i] - reconstructed[i];
        noise  += err * err;
    }
    return 10.0 * std::log10(signal / noise);   // dB; higher = better
}
// Expected: INT8 symmetric ~48 dB, INT4 ~24 dB, FP16 ~96 dB
```
Rule of thumb: each bit of quantization gives ~6 dB SNR. INT8 → ~48 dB is enough for most inference with <1% accuracy drop. INT4 → ~24 dB needs calibration techniques (GPTQ, AWQ, SmoothQuant) to recover accuracy. This metric is what you'd use when profiling a new quantization scheme on an NPU.

---

## Part 5b — Quantization Rapid-Fire

**Q1.** Why accumulate in INT32 during quantized GEMM?
INT8*INT8 products reach 16129; summing K=4096 of them → ~66M, overflows INT16, fits INT32.

**Q2.** Symmetric vs asymmetric — when to use each?
Symmetric for weights (near-zero-mean distributions fit well, simpler math). Asymmetric for activations (post-ReLU data is non-negative — symmetric wastes half the range).

**Q3.** What is a zero point?
The integer value that represents 0.0f in asymmetric quantization. Dequant: `x_float = scale * (x_int - zero_point)`.

**Q4.** Why per-channel quantization for weights?
Each output channel has its own magnitude distribution. One global scale forces a bad tradeoff — per-channel scale minimizes max quantization error per channel.

**Q5.** What is requantization?
Converting INT32 accumulator output back to INT8 to feed the next layer, using a precomputed fixed-point multiplier. Avoids float memory traffic between layers.

**Q6.** Why INT4 for LLM weights but not activations?
LLM decoding is memory-bandwidth-bound (weights loaded per token). INT4 halves weight size → halves bandwidth → ~2× throughput. Activations are computed fresh each forward pass and have wider dynamic range — INT4 loses too much precision there.

**Q7.** What is calibration in post-training quantization?
Running a representative dataset through the model to collect activation statistics (min/max or percentiles) to set scale/zero_point. Static quant needs this upfront; dynamic quant computes it per-inference (slower but more accurate for outlier inputs).

**Q8.** What is SmoothQuant / why do LLM activations hurt INT8?
LLMs (especially attention layers) have activation outliers — a few channels with 100× larger magnitude than the rest. Naively quantizing with one scale destroys the small channels. SmoothQuant migrates the difficulty from activations to weights (where per-channel quant handles it) by scaling: `Y = (X / s) * (s * W)` where s absorbs the outlier magnitude.