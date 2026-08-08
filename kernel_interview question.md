# Kernel Interview Practice — QUESTIONS
*Practice sheet for MulticoreWare AI Kernel Engineer prep. Answer each out loud or on paper before checking the answer sheet.*

---

## Section A — C/C++ Fundamentals

**A1.** What's the difference between a pointer and a reference in C++? When would you use each?

**A2.** What does `const int* p`, `int* const p`, and `const int* const p` each mean?

**A3.** What's the difference between stack and heap allocation? What are the performance implications of each?

**A4.** What is a memory leak? Write a small code example that leaks, and the fix.

**A5.** What's the difference between `malloc`/`free` and `new`/`delete`? What happens if you mix them?

**A6.** What is pointer arithmetic? If `int* p` points to address 0x1000, what address does `p + 3` point to?

**A7.** What is a dangling pointer? Give an example of how one gets created.

**A8.** What does the `restrict` keyword do in C, and why does it matter for performance?

**A9.** What is struct padding/alignment? Why does member ordering in a struct affect its size?

**A10.** What's the difference between pass-by-value, pass-by-pointer, and pass-by-reference? Which would you use for a large matrix and why?

**A11.** What is `inline` and when does it help performance? When can it hurt?

**A12.** What is undefined behavior? Give three examples.

**A13.** What are `static` variables inside a function vs. at file scope?

**A14.** What is RAII in C++? Why does it matter for resource management?

**A15.** Explain row-major vs. column-major memory layout. Which does C/C++ use, and why does it matter for loop ordering?

---

## Section B — Performance & Systems Thinking (CPU-side, no GPU needed)

**B1.** Why is iterating a 2D array row-by-row faster than column-by-column in C?

**B2.** What is a cache line? What is cache-friendly code?

**B3.** What is false sharing in multithreaded code?

**B4.** Why can branch mispredictions hurt performance? How would you restructure a hot loop to reduce branching?

**B5.** What is loop unrolling and why does it help?

**B6.** What is SIMD? How does it differ from multithreading?

**B7.** Explain arithmetic intensity. Why does it determine whether code is compute-bound or memory-bound?

**B8.** What is a race condition? How do you prevent one?

---

## Section C — CUDA / GPU Concepts

**C1.** Explain the CUDA thread hierarchy: thread, warp, block, grid. What sizes are typical and why?

**C2.** How do you compute a global thread index in a 1D kernel? In a 2D kernel?

**C3.** What is memory coalescing? Why is random access ~25x slower than sequential in your benchmark?

**C4.** What is shared memory? How does it differ from global memory, L1, and registers?

**C5.** What are shared memory bank conflicts? Why does reading a column of a 32x32 tile cause a 32-way conflict, and how does padding to [32][33] fix it?

**C6.** What is occupancy? What limits it?

**C7.** What is register tiling? What does it trade off, and why is the trade worth it?

**C8.** What is warp divergence? Give a code example that causes it.

**C9.** What are Tensor Cores? Why did WMMA give you ~2x on top of register tiling?

**C10.** What is `__syncthreads()` and when do you need it? What happens if you put it inside a divergent branch?

**C11.** Explain the difference between Nsight Compute and Nsight Systems. Which did you use and for what?

**C12.** What is roofline analysis? Where did your kernels land and why?

**C13.** Why were your hand-written fp32 kernels memory-bound while cuBLAS/CUTLASS were compute-bound?

**C14.** What is `num_stages` (software pipelining) in Triton? What does increasing it trade off?

**C15.** What is unified memory on Jetson? How does it differ from a discrete GPU's memory model?

**C16.** What does kernel launch overhead mean, and why do small matrices (N=512) show poor GPU utilization?

**C17.** How would your CUDA/GPU optimization skills transfer to a non-GPU accelerator (NPU/DSP) like MulticoreWare's target hardware?

---

## Section D — Write the Code (paper/doc practice, no compilation needed)

**D1.** Write a CUDA kernel for vector addition: `C[i] = A[i] + B[i]` for N elements. Include the launch configuration.

**D2.** Write a naive CUDA matmul kernel: `C = A * B` for N x N matrices.

**D3.** Write a tiled matmul kernel using shared memory. Explain why it's faster than the naive version.

**D4.** Write a parallel reduction (sum of N floats) in CUDA. Explain the tree-based approach.

**D5.** In plain C++, write a function that transposes an N x N matrix in-place. Then explain the cache behavior problem with large N.

**D6.** In plain C, implement a simple fixed-size memory pool allocator (alloc + free from a pre-allocated buffer).

**D7.** In plain C++, write a cache-friendly loop to multiply two matrices (loop reordering: which of ijk, ikj, jik is best and why?).

**D8.** Write plain C code to reverse a linked list. (Classic pointer-manipulation warm-up.)

**D9.** In plain C, write a function that detects if a float array of size N contains any NaN, without branching per element if possible.

**D10.** Spot the bugs: 
```c
float* make_array(int n) {
    float arr[n];
    for (int i = 0; i <= n; i++) arr[i] = i * 2.0;
    return arr;
}
```

---

## Section E — Behavioral / Fit (quick verbal reps)

**E1.** Give your 30-45 second intro.

**E2.** "The role asks for 5+ years — tell me about that gap."

**E3.** "Why kernel/systems work, when most of your background is data and ML applications?"

**E4.** "Your experience is CUDA/NVIDIA — our chip is a custom NPU. How comfortable are you outside the CUDA ecosystem?"

**E5.** "Tell me about the 8051/assembly work in your undergrad."

**E6.** "Walk me through one specific optimization you made and the numbers before/after."