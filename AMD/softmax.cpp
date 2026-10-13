// Row-wise softmax on AMD GPUs (HIP). One block per row.
//   softmax_3pass : reads the row from global memory three times (max, sum, write)
//   softmax_reg   : reads the row once into registers, then reduces and writes once
//
// Build:  hipcc --offload-arch=gfx942 -O3 softmax.cpp -o softmax
// Run:    ./softmax

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) do {                                            \
    hipError_t e = (cmd);                                              \
    if (e != hipSuccess) {                                             \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                    \
                hipGetErrorString(e), __FILE__, __LINE__);             \
        exit(1);                                                       \
    }                                                                  \
} while (0)

constexpr int ROWS    = 16384;
constexpr int COLS    = 4096;            // elements per row
constexpr int THREADS = 256;             // multiple of 64 (wavefront size on MI300X)
constexpr int ELEMS   = COLS / THREADS;  // values each thread keeps in registers

// ---- reductions: wavefront first (shuffles), then across wavefronts (LDS) ----
__device__ __forceinline__ float wave_max(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down(v, off));
    return v;
}
__device__ __forceinline__ float wave_sum(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}

template <bool IS_MAX>
__device__ float block_reduce(float v, float* smem) {
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int nwaves = blockDim.x / warpSize;

    v = IS_MAX ? wave_max(v) : wave_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (threadIdx.x < nwaves) ? smem[threadIdx.x] : (IS_MAX ? -INFINITY : 0.f);
    if (wid == 0) {
        v = IS_MAX ? wave_max(v) : wave_sum(v);
        if (lane == 0) smem[32] = v;          // broadcast slot
    }
    __syncthreads();
    v = smem[32];
    __syncthreads();                          // smem is safe to reuse after this
    return v;
}

// ---- kernel 1: three passes over global memory ----
__global__ void softmax_3pass(const float* in, float* out, int cols) {
    __shared__ float smem[33];
    const float* x = in  + (size_t)blockIdx.x * cols;whenever x[c]
    float*       y = out + (size_t)blockIdx.x * cols;

    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, x[c]);
    m = block_reduce<true>(m, smem);

    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += expf(x[c] - m);
    s = block_reduce<false>(s, smem);

    const float inv = 1.f / s;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = expf(x[c] - m) * inv;
}

// ---- kernel 2: one read, one write (row cached in registers) ----
template <int E>
__global__ void softmax_reg(const float* in, float* out, int cols) {
    __shared__ float smem[33];
    const float* x = in  + (size_t)blockIdx.x * cols;
    float*       y = out + (size_t)blockIdx.x * cols;

    float vals[E];
    float m = -INFINITY;
    #pragma unroll
    for (int i = 0; i < E; i++) {
        int c = threadIdx.x + i * blockDim.x;   // consecutive threads -> consecutive addresses
        vals[i] = (c < cols) ? x[c] : -INFINITY;
        m = fmaxf(m, vals[i]);
    }
    m = block_reduce<true>(m, smem);

    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < E; i++) {
        vals[i] = expf(vals[i] - m);            // exp(-inf) = 0 for padding
        s += vals[i];
    }
    s = block_reduce<false>(s, smem);

    const float inv = 1.f / s;
    #pragma unroll
    for (int i = 0; i < E; i++) {
        int c = threadIdx.x + i * blockDim.x;
        if (c < cols) y[c] = vals[i] * inv;
    }
}

// ---- timing: warmup, then average over many launches ----
template <typename F>
float time_ms(F launch, int iters = 20) {
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(s));
    for (int i = 0; i < iters; i++) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    return ms / iters;
}

int main() {
    const size_t n = (size_t)ROWS * COLS;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_in(n), h_out(n), ref(n);
    srand(1);
    for (size_t i = 0; i < n; i++) h_in[i] = (rand() / (float)RAND_MAX) * 20.f - 10.f;
    for (int c = 0; c < COLS; c++) h_in[c] += 1000.f;   // row 0 has huge values: tests numerical stability

    // CPU reference in double precision
    for (int r = 0; r < ROWS; r++) {
        const float* x = &h_in[(size_t)r * COLS];
        double m = -1e300;
        for (int c = 0; c < COLS; c++) m = fmax(m, (double)x[c]);
        double s = 0;
        for (int c = 0; c < COLS; c++) s += exp((double)x[c] - m);
        for (int c = 0; c < COLS; c++) ref[(size_t)r * COLS + c] = (float)(exp((double)x[c] - m) / s);
    }

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    struct Kernel { const char* name; void (*run)(const float*, float*); };
    Kernel kernels[] = {
        {"3pass", [](const float* i, float* o) { softmax_3pass<<<ROWS, THREADS>>>(i, o, COLS); }},
        {"reg",   [](const float* i, float* o) { softmax_reg<ELEMS><<<ROWS, THREADS>>>(i, o, COLS); }},
    };

    // "Ideal" traffic: read the input once and write the output once.
    const double ideal_bytes = 2.0 * (double)bytes;

    for (auto& k : kernels) {
        HIP_CHECK(hipMemset(d_out, 0, bytes));
        k.run(d_in, d_out);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));

        double maxdiff = 0;
        bool bad = false;
        for (size_t i = 0; i < n; i++) {
            double d = fabs((double)h_out[i] - (double)ref[i]);
            if (std::isnan(h_out[i]) || std::isinf(h_out[i])) bad = true;
            if (d > maxdiff) maxdiff = d;
        }

        float ms = time_ms([&] { k.run(d_in, d_out); });
        double gbps = ideal_bytes / (ms * 1e-3) / 1e9;
        bool pass = !bad && maxdiff < 1e-6;
        printf("%-6s | %.3f ms | %.1f GB/s (effective) | max diff %.2e | %s\n",
               k.name, ms, gbps, maxdiff, pass ? "PASS" : "FAIL");
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}