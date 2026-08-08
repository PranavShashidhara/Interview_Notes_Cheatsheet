# Kernel Coding Practice — QUESTIONS
*Write each program by hand (doc or paper) BEFORE checking the answer file. CUDA ones can be verified on Colab; plain C/C++ ones compile locally with `g++ -O2 file.cpp` or `gcc -O2 file.c`.*

---

## Part 1 — CUDA Kernels (write cold; verify on Colab if you want)

**Q1. Vector Add.** Write a complete CUDA program: kernel `C[i] = A[i] + B[i]` for N floats, plus host code (allocate, copy, launch, copy back, verify). Include correct launch config for any N.

**Q2. SAXPY.** Write the kernel for `y[i] = a * x[i] + y[i]` (scalar a). One-liner kernel — focus on getting the signature and indexing perfect without thinking.

**Q3. Elementwise ReLU.** Kernel: `out[i] = max(0, in[i])`. Then: how would you make it in-place, and does anything change?

**Q4. Naive MatMul.** Kernel for `C = A × B`, N×N, row-major, one thread per output element. 2D grid/block indexing. State its memory-traffic weakness in one sentence.

**Q5. Tiled MatMul (shared memory).** The big one — write it fully: shared memory tiles, boundary guards for N not divisible by TILE, both `__syncthreads()` in the right places. Then answer: where would a bank conflict appear if this were a transpose-style access, and what's the fix?

**Q6. Matrix Transpose (shared memory).** Kernel: `out[j][i] = in[i][j]` using a shared-memory tile so both global read AND global write are coalesced. Why does the naive version have uncoalesced writes? Why must the tile be padded `[TILE][TILE+1]`?

**Q7. Parallel Reduction (sum).** Full kernel: block-level tree reduction in shared memory, sequential-addressing version (`tid < s`). Then: (a) why is sequential addressing better than interleaved (`tid % (2*s) == 0`)? (b) how do you combine per-block partial sums? (c) what would you use for the final 32 threads instead?

**Q8. Dot Product.** Combine Q1 + Q7: kernel computing `sum(A[i] * B[i])` — elementwise multiply + block reduction + atomicAdd for the final combine. Write it fully.

**Q9. Grid-Stride Loop.** Rewrite Q1's vector add as a grid-stride loop. Why is this pattern preferred for arbitrary N / limited grid sizes?

**Q10. Softmax (row-wise, numerically stable).** Kernel: one block per row of an M×N matrix; compute `exp(x - max) / sum(exp(x - max))`. Requires TWO reductions (max, then sum). Why subtract the max?

**Q11. Histogram.** Kernel: count occurrences of values 0–255 from a byte array into `bins[256]`. What's the race condition, and what are two ways to handle it (global atomics vs. shared-memory privatization)?

**Q12. Find the bugs.** Fix everything wrong:
```cuda
__global__ void scale(float* data, int N, float factor) {
    int i = threadIdx.x;
    data[i] = data[i] * factor;
}
// host:
scale<<<1, N>>>(d_data, N, 2.0f);
```
(There are at least 3 issues for large N.)

**Q13. Find the bugs #2.**
```cuda
__global__ void avgPool(float* in, float* out, int N) {
    __shared__ float tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = in[i];
    if (threadIdx.x < 128) {
        __syncthreads();
        out[i] = (tile[threadIdx.x] + tile[threadIdx.x + 128]) / 2.0f;
    }
}
```
(Two serious bugs.)

---

## Part 2 — Plain C/C++ (compile and run locally)

**Q14. Cache-friendly MatMul (CPU).** Write matmul three ways — `ijk`, `ikj`, `jik` loop orders — with a timing harness (`std::chrono`). Predict which is fastest before running, then verify. Explain the result in terms of stride and cache lines.

**Q15. Blocked (tiled) CPU MatMul.** Add cache blocking to Q14's best version (tile size ~64). Why does blocking help on CPU for large N even with the right loop order?

**Q16. In-place Transpose + Blocked Transpose.** Write both. Time them at N=4096. Why does the blocked version win?

**Q17. Fixed-size Pool Allocator.** Full implementation: `pool_init`, `pool_alloc`, `pool_free` over a static buffer using an intrusive free list. Then: why do embedded/real-time systems prefer this over malloc? What alignment guarantee should alloc provide?

**Q18. Aligned Allocation.** Write `void* aligned_alloc64(size_t n)` returning 64-byte-aligned memory using only `malloc` (store the original pointer for free). Why does alignment matter for SIMD and cache lines?

**Q19. Reverse a Linked List.** Iterative, in C. Then the recursive version. What's the stack-depth risk of the recursive one?

**Q20. memcpy from scratch.** Write a naive byte-copy `my_memcpy`, then a word-at-a-time (size_t) version handling unaligned head/tail. Why is the word version faster?

**Q21. Count Set Bits.** Three versions: naive loop, Kernighan's `n & (n-1)` trick, and lookup table. When does each win?

**Q22. Struct Layout.** Given `struct S { char a; double b; int c; char d; };` — compute sizeof on a 64-bit system with 8-byte alignment, then reorder to minimize it. Show the math.

**Q23. False Sharing Demo.** Write a two-thread program (std::thread) where each thread increments its own counter — first with counters adjacent in one array, then padded to separate cache lines. Explain the timing difference.

**Q24. Branchless Max/Abs.** Write branchless `max(a,b)` for ints and branchless `abs(x)`. When is branchless actually worth it vs. letting the compiler emit cmov?

**Q25. Ring Buffer.** Fixed-capacity circular queue in C: `push`, `pop`, `is_full`, `is_empty`, power-of-two capacity with mask indexing. Why is power-of-two + mask faster than modulo?

---

## Part 3 — Rapid-fire syntax checks (say the answer instantly)

**R1.** Exact formula for global thread index, 1D.
**R2.** Ceiling-division formula for number of blocks.
**R3.** Kernel launch syntax with shared memory size and stream.
**R4.** How do you declare 48 floats of static shared memory? Dynamic shared memory?
**R5.** cudaMemcpy signature and the 4 direction enums — which do you actually use?
**R6.** What does `cudaDeviceSynchronize()` do and when do you need it after a kernel launch?
**R7.** How do you check for a kernel launch error?
**R8.** `float4` — what is it and why load with it?
**R9.** Difference between `__device__`, `__global__`, `__host__` function qualifiers.
**R10.** Row-major index of element (row, col) in an N-column matrix.

---

## Part 4 — Parallel C++, SIMD, and Compiler Awareness (all local, `g++ -O2 -pthread`, add `-fopenmp` where noted)

**Q26. Multithreaded Reduction.** Sum a large float array using std::thread with T threads: partition the array, each thread computes a partial sum, main thread combines. Why partial sums instead of one shared accumulator with a mutex? Instead of a std::atomic?

**Q27. Parallel MatMul.** Parallelize your best matmul (ikj or blocked) across std::threads by partitioning output rows. Why partition ROWS of C rather than the k-dimension? What goes wrong if two threads share a k-split of the same output element?

**Q28. Can this loop be parallelized?** For each, say parallelizable or not, and why:
```cpp
// (a)
for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
// (b)
for (int i = 1; i < n; i++) a[i] = a[i-1] + b[i];
// (c)
for (int i = 0; i < n; i++) sum += a[i];
// (d)
for (int i = 0; i < n; i++) hist[bucket(a[i])]++;
```
For the non-trivial ones, what technique makes them parallel anyway?

**Q29. CPU Stable Softmax.** Plain C++ function: softmax over a float array (max pass, exp-sum pass, normalize pass). Why three passes — and how would you fuse to two? Why subtract the max?

**Q30. Prefix Sum (Scan).** Write the sequential inclusive scan. Then explain (or code) the two-pass parallel version: chunk-local scans, scan of chunk totals, add offsets. Why is scan harder to parallelize than reduction?

**Q31. Dot Product with Accumulation Care.** CPU dot product of two float arrays. Why can float accumulation lose precision at large N, and what are two fixes? Why do 4 independent accumulators run faster than 1 even single-threaded?

**Q32. Aliasing / restrict.** 
```cpp
void add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}
```
Why can't the compiler fully vectorize this as written? What does `__restrict__` promise, and what happens if you lie? Rewrite the signature.

**Q33. Spot the Vectorization Blockers.** For each loop, will it auto-vectorize at -O2/-O3? If not, why:
```cpp
// (a)
for (int i = 0; i < n; i++) x[i] = y[i] * 2.0f;
// (b)
for (int i = 0; i < n; i++) { if (x[i] > 0) s += x[i]; }
// (c)
for (int i = 0; i < n; i++) x[idx[i]] += y[i];
// (d)
for (int i = 0; i < n; i++) x[i] = f(y[i]);   // f is an extern function
```

**Q34. Generic Kernel via Templates.** Write a templated `axpy<T>(T a, const T* x, T* y, int n)` working for float and double. Why templates instead of void* + element size (the C way)? What's the cost?

**Q35. std::vector vs Raw Arrays; Move Semantics.** When is `std::vector<float>` the right call vs `new float[n]` vs a stack array in kernel-style code? What does `std::move` actually do to a vector — what gets copied?

**Q36. Const Correctness.** Fix this API so it can't be misused, and explain each const:
```cpp
float dot(float* a, float* b, int n);
```
Where can const placement affect optimization vs. just safety?

**Q37. What Do -O Levels Do?** Explain what changes between -O0, -O2, -O3 (and -march=native). How do you actually LOOK at what the compiler emitted (two ways)? Name three specific optimizations -O2 performs.

**Q38. Inlining & Unrolling in Practice.** What does the compiler do when it inlines a function into a hot loop — what NEW optimizations become possible after inlining? When does aggressive inlining/unrolling backfire?

**Q39. OpenMP Version.** (compile with `-fopenmp`) Rewrite the reduction and the matmul with `#pragma omp parallel for`. What does the `reduction(+:sum)` clause do under the hood? What's `schedule(static)` vs `schedule(dynamic)` and when does each win?

**Q40. Synchronization Cost Ladder.** Rank these from fastest to slowest for N threads incrementing a counter 10M times each, and explain why: (a) mutex per increment, (b) std::atomic per increment, (c) thread-local counter + combine at the end. What's the general principle for parallel kernel design?

---

## Part 3b — Rapid-fire additions (C++/compiler)

**R11.** How do you tell g++ to show you the assembly for a file? Two ways.
**R12.** Compile command for: optimized, native-arch, OpenMP, pthread program.
**R13.** What does `alignas(64)` on a struct member/array do, and one reason to use it?
**R14.** `std::atomic<int>` vs `int` protected by mutex — which for a simple counter and why?
**R15.** What flag asks gcc/clang to REPORT which loops vectorized?