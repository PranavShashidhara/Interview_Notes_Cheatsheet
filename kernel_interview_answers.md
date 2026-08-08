# Kernel Interview Practice — ANSWERS
*Check yourself against these after attempting each question. Phrasing here is tuned to be said out loud.*

---

## Section A — C/C++ Fundamentals

**A1. Pointer vs reference.** A pointer is a variable holding an address — it can be reassigned, can be null, and supports arithmetic. A reference is an alias for an existing object — must be bound at initialization, can't be null, can't be reseated. Use references for function parameters when the argument must exist; use pointers when null is meaningful, when reseating is needed, or for arrays/dynamic memory.

**A2. Const positions.** `const int* p` — pointer to const int: the *data* can't be modified through p, but p can point elsewhere. `int* const p` — const pointer: p can't be reseated, but data is mutable. `const int* const p` — neither. Trick: read right-to-left.

**A3. Stack vs heap.** Stack: automatic, fast (pointer bump), scoped lifetime, limited size (~MBs). Heap: manual/dynamic, slower (allocator bookkeeping, possible syscalls), arbitrary lifetime, fragmentation risk. Perf: stack data is hot in cache and allocation is nearly free; heavy heap churn in hot loops is a classic slowdown.

**A4. Memory leak.** Allocated memory that's never freed and becomes unreachable.
```c
void leak() {
    int* p = malloc(100 * sizeof(int));
    p = NULL;            // original allocation now unreachable — leak
}
// Fix: free(p) before reassigning, or don't reassign.
```

**A5. malloc/free vs new/delete.** `malloc` allocates raw bytes, returns void*, no constructors. `new` allocates AND calls constructors, is type-safe, throws on failure. `delete` calls destructors; `free` doesn't. Mixing them (malloc + delete, or new + free) is undefined behavior — destructors/bookkeeping mismatch.

**A6. Pointer arithmetic.** Advances by *element size*, not bytes. `int* p` at 0x1000 → `p + 3` is 0x1000 + 3 × sizeof(int) = 0x100C.

**A7. Dangling pointer.** A pointer to memory that's been freed or gone out of scope.
```c
int* p = malloc(sizeof(int));
free(p);
*p = 5;   // dangling — UB
```
Also: returning the address of a local variable.

**A8. restrict.** A promise to the compiler that this pointer is the *only* way that memory is accessed in this scope — no aliasing. Lets the compiler keep values in registers, reorder loads/stores, and vectorize, instead of conservatively reloading after every write. Big deal in kernel-style loops taking multiple array pointers.

**A9. Struct padding.** Compilers insert padding so each member sits at its alignment boundary (int on 4, double on 8, etc.). `{char; double; char}` can be 24 bytes while `{double; char; char}` is 16. Order members largest-to-smallest to minimize size — matters for cache footprint and arrays of structs.

**A10. Pass by value/pointer/reference.** Value copies (fine for scalars, terrible for big objects). Pointer and reference both avoid the copy. For a large matrix: `const Matrix&` (C++) — no copy, can't be null, signals read-only. Pointer if the API is C or null is a valid input.

**A11. inline.** Removes call overhead and enables cross-boundary optimization (constant folding into the body). Helps for tiny hot functions. Hurts when over-applied: code bloat → instruction-cache pressure. Modern compilers mostly decide themselves; `inline` is mainly a linkage hint.

**A12. Undefined behavior — examples.** (1) Out-of-bounds array access. (2) Use-after-free / dangling pointer dereference. (3) Signed integer overflow. (Also: data races, null deref, misaligned access.)

**A13. static.** Inside a function: the variable persists across calls, initialized once. At file scope: restricts the symbol to internal linkage — invisible to other translation units.

**A14. RAII.** Resource Acquisition Is Initialization — tie a resource's lifetime to an object's scope: acquire in constructor, release in destructor. Guarantees cleanup on every exit path, including exceptions. `std::vector`, `unique_ptr`, lock guards are all RAII.

**A15. Row-major vs column-major.** C/C++ is row-major: `a[i][j]` and `a[i][j+1]` are adjacent in memory; `a[i][j]` and `a[i+1][j]` are a full row apart. So the inner loop should walk `j` (columns within a row) for sequential access. Fortran/MATLAB are column-major — the transpose of this rule.

---

## Section B — Performance & Systems Thinking

**B1. Row-by-row is faster** because C is row-major: row iteration touches consecutive addresses, so each 64-byte cache line loaded serves ~16 ints before the next miss. Column iteration jumps a full row stride each step — potentially a cache miss *per element*, and for large strides it also defeats the hardware prefetcher.

**B2. Cache line.** The unit of transfer between memory and cache — typically 64 bytes on CPU. Touching 1 byte pulls in all 64. Cache-friendly code: sequential access, small working sets, data structured so everything on a fetched line gets used (SoA vs AoS decisions).

**B3. False sharing.** Two threads write *different* variables that happen to sit on the *same* cache line. Coherence protocol ping-pongs the line between cores even though there's no logical sharing. Fix: pad/align per-thread data to separate cache lines.

**B4. Branch mispredictions** flush the pipeline (~15-20 cycles). In hot loops: sort data so the branch becomes predictable, replace branches with arithmetic/conditional moves (`x = cond * a + (1-cond) * b`), or hoist invariant conditions out of the loop.

**B5. Loop unrolling** processes multiple elements per iteration — fewer branch/counter instructions per element, more independent operations in flight for ILP, and enables wider vectorization. Cost: code size; compilers usually do it for you at -O2/-O3.

**B6. SIMD** — one instruction operating on multiple data lanes in a single core (AVX: 8 floats at once). Multithreading is parallelism *across* cores. They stack: N cores × 8-wide SIMD. GPUs generalize this idea massively (SIMT: warps of 32 threads executing in lockstep).

**B7. Arithmetic intensity** = FLOPs per byte moved from memory. Hardware has a fixed compute peak and a fixed bandwidth peak; AI decides which one you hit first. Low AI (e.g., vector add: 1 FLOP per 12 bytes) → memory-bound no matter what. High AI (large matmul: O(N) reuse per element) → compute-bound. This is the x-axis of the roofline chart.

**B8. Race condition** — outcome depends on unsynchronized timing of concurrent accesses, at least one being a write. Prevent with mutexes/locks, atomics, or by design — partition data so threads never touch the same locations (the GPU-friendly answer).

---

## Section C — CUDA / GPU Concepts

**C1. Thread hierarchy.** Threads group into warps of 32 (execute in lockstep — the real hardware unit). Warps group into blocks (up to 1024 threads; a block lives on one SM and shares shared memory + can sync). Blocks form the grid. Typical block sizes: 128–256 threads — multiples of 32, enough warps to hide latency without exhausting registers/shared memory.

**C2. Global index.** 1D: `int i = blockIdx.x * blockDim.x + threadIdx.x;` 2D: `int col = blockIdx.x * blockDim.x + threadIdx.x; int row = blockIdx.y * blockDim.y + threadIdx.y;` Always bounds-check `if (i < N)` since the grid overshoots.

**C3. Coalescing.** Global memory is served in 32–128 byte transactions. When a warp's 32 threads hit consecutive addresses, the hardware merges them into one or two transactions. Random addresses land on 32 different lines → up to 32 separate transactions for the same amount of useful data. That transaction blow-up plus wasted bytes per line is the ~25x collapse (65 → 12 GB/s) in my benchmark. Strided stayed near sequential because modest strides still fall within/adjacent transactions.

**C4. Memory spaces.** Registers: per-thread, fastest, ~KB per thread budget. Shared memory: per-block scratchpad on the SM, ~100x faster than global, programmer-managed, where tiles live. L1/L2: hardware caches, not directly controlled. Global: GB-scale device memory, hundreds of cycles away. Optimization is largely the art of keeping traffic as high in this hierarchy as possible.

**C5. Bank conflicts.** Shared memory is split into 32 banks; bank = (word address) % 32. In `tile[32][32]`, element `[row][col]` sits at flat index `row*32 + col`, and `(row*32 + col) % 32 = col` — the row term vanishes. So an entire column is in ONE bank; a warp reading a column serializes 32-ways. Padding to `[32][33]` makes the stride 33, so `row*33 % 32` differs per row — column reads now spread across banks. One wasted column of storage buys back a 32x serialization.

**C6. Occupancy** — ratio of active warps per SM to the hardware max. Limited by: registers per thread × threads, shared memory per block, and block size granularity. Higher occupancy = more warps to swap in while others wait on memory. But it's a means, not an end — enough ILP per thread can hide latency at low occupancy (the register-tiling story).

**C7. Register tiling.** Each thread computes a small output tile (e.g., 4x4) held in registers, reusing each loaded value across many MACs. Raises arithmetic intensity and cuts shared-memory traffic. Cost: more registers per thread → *lower* occupancy. Worth it because each thread now has abundant independent work (ILP) to hide latency — you've replaced warp-level latency hiding with instruction-level. This took my kernel 171 → 968 GFLOPS.

**C8. Warp divergence.** Threads in a warp execute in lockstep; if a branch splits them, both paths execute serially with threads masked off.
```cuda
if (threadIdx.x % 2 == 0) do_A(); else do_B();  // both paths run — ~half throughput
```
Fix: branch on warp-aligned conditions (`threadIdx.x / 32`), restructure data so warps are uniform.

**C9. Tensor Cores.** Dedicated matrix-multiply-accumulate units doing a small matrix op (e.g., 16x16x16) per instruction, versus one FMA per instruction on CUDA cores. Accessed via WMMA intrinsics with fragments. They're a separate, wider data path — that's why they stacked another ~2x (968 → 1,777 GFLOPS) on top of register tiling, which had already maxed the SIMT path.

**C10. __syncthreads().** Block-wide barrier: every thread waits until all arrive, and it fences shared-memory visibility. Needed between writing a tile to shared memory and reading it. Inside a divergent branch (where some threads skip it) → deadlock/UB, because the barrier waits for threads that never come.

**C11. Nsight Compute vs Systems.** Compute = per-kernel internals: occupancy, cache hit rates, warp stalls, bank conflicts, roofline. Systems = whole-app timeline: CPU-GPU interaction, launch overhead, transfer timing, GPU idle gaps. One sentence: Systems tells you IF the GPU is busy and when; Compute tells you WHY a specific kernel is slow. I used Compute for the roofline/L1/occupancy analysis.

**C12. Roofline.** X: arithmetic intensity, Y: achieved FLOPs, both log. Sloped line = bandwidth ceiling (perf grows with AI); flat line = compute ceiling; they meet at the ridge point. Your kernel's dot vs. the ceiling above it = your bound and your headroom. My fp32/fp16 kernels sat under the sloped (memory) roof; cuBLAS/CUTLASS sat at the flat (compute) roof.

**C13. Why memory-bound vs cuBLAS.** My L1 hit rate topped out ~50%, so a large share of accesses fell through to L2/global, keeping effective AI low; occupancy was also modest. cuBLAS/CUTLASS use deeper tiling, double-buffering, and tuned layouts so nearly all reuse is captured in registers/shared memory — data movement stops being the limit, so they ride the compute roof.

**C14. num_stages.** Software pipelining depth for the K-loop: prefetch N tiles ahead so loads for tile i+1..i+N-1 overlap compute on tile i. More stages = more latency hiding but more shared memory/registers per stage → can cut occupancy. Diminishing returns past ~3 in my sweep (218/216/209 TFLOPS for 3/4/5 at N=4096) — the hiding saturates while the resource cost keeps climbing.

**C15. Jetson unified memory.** Orin Nano is one SoC: ARM CPU + Ampere GPU sharing one LPDDR5 pool — no VRAM, no PCIe hop, no H2D/D2H copies needed. Tradeoffs vs discrete: you skip transfer overhead entirely, but CPU and GPU contend for the same (much lower) bandwidth. Relevant here: it's the same class of edge/embedded constraint MulticoreWare's chip targets.

**C16. Launch overhead.** Fixed cost (~µs) to set up and dispatch a kernel. For N=512 matmul the compute finishes so fast the overhead is a big fraction of runtime, and there isn't enough work to fill the SMs — my whole config sweep was within ~4 TFLOPS at that size. Big-N amortizes it away.

**C17. Transfer to NPU/DSP.** The ISA changes; the discipline doesn't. Roofline reasoning, arithmetic intensity, exploiting the memory hierarchy, profiling-driven bottleneck hunting, tiling for data reuse, understanding compute vs bandwidth limits — those are hardware-universal. My depth is CUDA/SIMT, and I'd need to ramp on the specific architecture — but I've already self-taught one accelerator stack from scratch, which is evidence I can do it again, and my ECE background means reading an architecture manual isn't foreign territory.

---

## Section D — Code Answers

**D1. Vector add.**
```cuda
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
// Launch:
int threads = 256;
int blocks = (N + threads - 1) / threads;   // ceiling division
vecAdd<<<blocks, threads>>>(A, B, C, N);
```

**D2. Naive matmul.**
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
```
Weakness to state: every thread streams a full row of A and column of B from global memory — no reuse, so it's badly memory-bound.

**D3. Tiled matmul.**
```cuda
#define TILE 32
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];   // pad to [TILE][TILE+1] if reading columns
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = (row < N && t*TILE + threadIdx.x < N)
            ? A[row * N + t*TILE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t*TILE + threadIdx.y < N && col < N)
            ? B[(t*TILE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}
```
Why faster: each global element is loaded once per tile and reused TILE times from shared memory — global traffic drops by ~TILE×, raising arithmetic intensity. The two `__syncthreads()` bracket the load and the consume phases.

**D4. Parallel reduction.**
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
    if (tid == 0) out[blockIdx.x] = sdata[0];   // then reduce block results
}
```
Tree approach: halve active threads each step — O(log n) steps per block. The `tid < s` form keeps active threads contiguous (low divergence) vs. the naive interleaved version. Follow-ups worth knowing: warp-shuffle (`__shfl_down_sync`) for the last 32, or atomicAdd for the final combine.

**D5. In-place transpose.**
```cpp
void transpose(float* a, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            std::swap(a[i * n + j], a[j * n + i]);
}
```
Cache problem: `a[i*n+j]` walks a row (sequential — fine), but its partner `a[j*n+i]` walks a *column* — stride-n jumps, a cache miss per element once n is large. Fix: blocked/tiled transpose — process B×B sub-tiles so both sides of the swap stay resident in cache. (Same idea as the shared-memory transpose on GPU.)

**D6. Fixed-size pool allocator.**
```c
#define POOL_BLOCKS 1024
#define BLOCK_SIZE  64
static unsigned char pool[POOL_BLOCKS * BLOCK_SIZE];
static void* free_list = NULL;
static int initialized = 0;

static void pool_init(void) {
    for (int i = 0; i < POOL_BLOCKS; i++) {
        void* block = pool + i * BLOCK_SIZE;
        *(void**)block = free_list;    // each free block stores next ptr
        free_list = block;
    }
    initialized = 1;
}
void* pool_alloc(void) {
    if (!initialized) pool_init();
    if (!free_list) return NULL;       // exhausted
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
Talking points: O(1) alloc/free, zero fragmentation, no syscalls — why embedded/real-time systems (and GPU-adjacent runtimes) use pools over malloc.

**D7. Loop order for matmul.** `ikj` is best in row-major C:
```cpp
for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
        float a = A[i*n + k];                 // scalar, hoisted
        for (int j = 0; j < n; j++)
            C[i*n + j] += a * B[k*n + j];     // C and B both walk rows — sequential
    }
```
In `ijk`, the inner loop reads `B[k*n + j]` down a column — stride-n misses. In `ikj`, both inner-loop accesses are unit-stride and it vectorizes cleanly. Classic demonstration that identical FLOPs can differ several-fold on memory behavior alone.

**D8. Reverse linked list.**
```c
struct Node { int val; struct Node* next; };
struct Node* reverse(struct Node* head) {
    struct Node* prev = NULL;
    while (head) {
        struct Node* next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```

**D9. Branch-light NaN detection.** Key fact: NaN != NaN.
```c
int has_nan(const float* a, int n) {
    int found = 0;
    for (int i = 0; i < n; i++)
        found |= (a[i] != a[i]);   // 1 only for NaN; no data-dependent branch
    return found;
}
```
The comparison result is used arithmetically, so the loop body is branchless and vectorizable. (Early-exit version branches but can be faster if NaNs are common — worth saying you see the tradeoff.)

**D10. Spot the bugs.** Three:
1. **Returning a stack array** — `arr` dies when the function returns; caller gets a dangling pointer. Fix: `malloc` (caller frees) or have the caller pass a buffer.
2. **Off-by-one** — `i <= n` writes one past the end. Should be `i < n`.
3. (Minor) `i * 2.0` is a double expression narrowed into a float — harmless here but worth noticing; write `2.0f`.

---

## Section E — Behavioral Answers (bullets to riff from)

**E1. Intro** — use the locked version: 4 years (Oracle pipelines + UMD RAG platform) → ECE undergrad with 8051 assembly → self-taught CUDA/Triton because the MS didn't cover systems → Jetson benchmark suite: memory hierarchy characterization, register tiling + WMMA for 10x, Nsight Compute roofline vs cuBLAS/CUTLASS → "this is the layer I want to work at."

**E2. Experience gap.** "I want to be upfront — my professional experience is closer to 4 years, and the kernel work specifically is self-driven rather than on the job. What I'd point to is the depth: I didn't just run tutorials — I characterized a memory hierarchy, diagnosed why my kernels were memory-bound from L1 hit rates, and closed a 10x gap through profiling. If there's a specific bar the 5 years is standing in for, I'm happy to be tested against it directly."

**E3. Why kernels.** "The pattern across my career is drilling into *why* something is slow — at Oracle that was query plans and partitioning; at UMD it was retrieval latency. GPU kernels are that same instinct at the lowest level, where the wins are biggest and most measurable. Once I started the benchmark work I found it's the most engaged I've been with any technical problem."

**E4. Non-CUDA hardware.** "My depth is CUDA and the SIMT model, honestly. But the transferable core is the method — roofline thinking, arithmetic intensity, tiling for reuse, profiling before optimizing. Those are hardware-universal. And I've already proven I can self-teach one accelerator stack from zero; ramping on your architecture during the training period is exactly the kind of thing I've done before, with an ECE background that makes architecture docs readable territory."

**E5. 8051 work.** Have 2-3 concrete specifics ready in YOUR words — what you built, which peripherals (timers? UART? interrupts?), what the assembly actually did. Don't let this be a one-liner you can't back up; a single concrete detail ("I wrote the interrupt service routine for X") beats a vague claim.

**E6. One optimization, with numbers.** "Naive matmul on the Jetson: 171 GFLOPS. Nsight Compute showed it memory-bound — poor reuse, every element re-fetched. Register tiling — each thread computing a small output tile from registers — took it to 968, a 5.65x jump, by trading occupancy for arithmetic intensity. Then Tensor Core WMMA, a separate hardware path, stacked to 1,777 — above cuBLAS's 1,188 fp32 SIMT on that board. The roofline showed my fp32 was still memory-bound at ~50% L1 hit rate, so I can tell you exactly where the next gains are — that diagnosis-driven loop is the job, as I understand it."