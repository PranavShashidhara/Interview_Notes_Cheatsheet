# Likely Kernel Interview Questions

---

## 1. SAXPY (Warmup)

**C++:**
```cpp
void saxpy(int n, float a, float* x, float* y) {
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}
```

**CUDA:**
```cpp
__global__ void saxpy(int n, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
// launch: saxpy<<<(n+255)/256, 256>>>(n, a, x, y)
```
- Y = a*X + Y, simple elementwise
- Each CUDA thread handles one element

---

## 2. ReLU

**C++:**
```cpp
void relu(float* in, float* out, int N) {
    for (int i = 0; i < N; i++)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}
```

**CUDA:**
```cpp
__global__ void relu(float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = fmaxf(0.0f, in[i]);  // fmaxf = single GPU instruction
}
```

---

## 3. Matrix Transpose

**C++ naive:**
```cpp
void transpose(const float* in, float* out, int N) {
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            out[x * N + y] = in[y * N + x];  // write is strided — slow
}
```

**C++ blocked (cache friendly):**
```cpp
#define TILE 32
void transposeBlocked(const float* in, float* out, int N) {
    for (int i = 0; i < N; i += TILE)
        for (int j = 0; j < N; j += TILE)
            for (int ii = i; ii < i + TILE && ii < N; ii++)
                for (int jj = j; jj < j + TILE && jj < N; jj++)
                    out[jj * N + ii] = in[ii * N + jj];  // tile fits in cache
}
```

**CUDA (shared memory):**
```cpp
#define TILE 32
__global__ void transpose(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 avoids bank conflicts
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // coalesced read
    __syncthreads();
    int ox = blockIdx.y * TILE + threadIdx.x;  // swap block coords
    int oy = blockIdx.x * TILE + threadIdx.y;
    if (ox < N && oy < N)
        out[oy * N + ox] = tile[threadIdx.x][threadIdx.y]; // coalesced write
}
```
- Shared memory stages the transpose — both global read and write stay coalesced
- `+1` padding: shifts each row by 1 so column reads hit different banks

---

## 4. Matmul — Naive

**C++:**
```cpp
void matmulNaive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];  // B strided — slow
            C[i * N + j] = sum;
        }
}
```

**CUDA:**
```cpp
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];  // B strided — slow
        C[row * N + col] = sum;
    }
}
// launch: dim3 block(16,16), grid((N+15)/16, (N+15)/16)
```

---

## 5. Matmul — Cache Friendly (ikj)

**C++ only** (CUDA equivalent is tiled below):
```cpp
void matmulFast(const float* A, const float* B, float* C, int N) {
    memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];         // hoist — loaded once, reused N times
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];  // B and C both stride-1
        }
}
```
- ikj: innermost j makes B and C sequential
- scalar `a` × entire row of B accumulated into row of C

---

## 6. Matmul — Tiled

**C++:**
```cpp
#define TILE 32
void matmulTiled(const float* A, const float* B, float* C, int N) {
    memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i += TILE)
        for (int k = 0; k < N; k += TILE)
            for (int j = 0; j < N; j += TILE)
                for (int ii = i; ii < i + TILE && ii < N; ii++)
                    for (int kk = k; kk < k + TILE && kk < N; kk++) {
                        float a = A[ii * N + kk];
                        for (int jj = j; jj < j + TILE && jj < N; jj++)
                            C[ii * N + jj] += a * B[kk * N + jj];
                    }
}
```

**CUDA (shared memory tiling):**
```cpp
#define TILE 16
__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();                        // wait for tile to be loaded
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                        // wait before loading next tile
    }
    if (row < N && col < N) C[row * N + col] = sum;
}
```
- Threads cooperatively load tiles into shared memory
- Two syncthreads: one after load, one after compute

---

## 7. Softmax (Numerically Stable)

**C++:**
```cpp
void softmax(float* in, float* out, int N) {
    float maxVal = in[0];
    for (int i = 0; i < N; i++)
        if (in[i] > maxVal) maxVal = in[i];     // pass 1: find max

    float sum = 0;
    for (int i = 0; i < N; i++) {
        out[i] = expf(in[i] - maxVal);           // pass 2: exp(x - max)
        sum += out[i];
    }

    for (int i = 0; i < N; i++)
        out[i] /= sum;                           // pass 3: normalize
}
```

**CUDA (naive — one thread does entire softmax):**
```cpp
__global__ void softmax(float* in, float* out, int N) {
    float maxVal = in[0];
    for (int i = 0; i < N; i++) maxVal = fmaxf(maxVal, in[i]);

    float sum = 0;
    for (int i = 0; i < N; i++) sum += expf(in[i] - maxVal);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = expf(in[i] - maxVal) / sum;
}
```
- Subtract max before exp — e^1000 = inf, e^(1000-1000) = 1.0
- Three passes: max → exp+sum → normalize

---

## 8. Parallel Reduction (Sum)

**C++:**
```cpp
float reduceSum(float* in, int N) {
    float sum = 0;
    for (int i = 0; i < N; i++) sum += in[i];
    return sum;
}
```

**CUDA (tree reduction):**
```cpp
__global__ void reduce(float* in, float* out, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {  // s >>= 1 is s = s/2
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];  // one result per block
}
```
Tree reduction per block:
```
[1, 2, 3, 4, 5, 6, 7, 8]
[3,    7,    11,   15  ]   s=4
[10,         26        ]   s=2
[36                    ]   s=1  → log2(N) steps
```

---

## 9. Linear Layer (Y = XW^T + b)

**C++:**
```cpp
// X: [M x K], W: [N x K], b: [N], Y: [M x N]
void linear(const float* X, const float* W, const float* b,
            float* Y, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = b[n];
            for (int k = 0; k < K; k++)
                sum += X[m * K + k] * W[n * K + k];  // W transposed logically
            Y[m * N + n] = sum;
        }
}
```

**CUDA (one thread per output element):**
```cpp
// X: [M x K], W: [N x K], b: [N], Y: [M x N]
__global__ void linear(const float* X, const float* W, const float* b,
                       float* Y, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron (n)
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch element (m)
    if (row < M && col < N) {
        float sum = b[col];
        for (int k = 0; k < K; k++)
            sum += X[row * K + k] * W[col * K + k];
        Y[row * N + col] = sum;
    }
}
// launch: dim3 block(16,16), grid((N+15)/16, (M+15)/16)
```

**CUDA (tiled shared memory — production style):**
```cpp
#define TILE 16
// Computes Y = X * W^T + b  (X:[M,K], W:[N,K], Y:[M,N])
__global__ void linearTiled(const float* X, const float* W, const float* b,
                            float* Y, int M, int N, int K) {
    __shared__ float Xs[TILE][TILE];
    __shared__ float Ws[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;  // m index
    int col = blockIdx.x * TILE + threadIdx.x;  // n index
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int kX = t * TILE + threadIdx.x;
        int kW = t * TILE + threadIdx.x;
        Xs[threadIdx.y][threadIdx.x] = (row < M && kX < K) ? X[row * K + kX] : 0.0f;
        Ws[threadIdx.y][threadIdx.x] = (col < N && kW < K) ? W[col * K + kW] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += Xs[threadIdx.y][k] * Ws[threadIdx.x][k];
        __syncthreads();
    }
    if (row < M && col < N)
        Y[row * N + col] = sum + b[col];  // add bias after accumulation
}
```
- Equivalent to matmul of X with W^T, plus bias broadcast over batch dim
- Bias added after the dot product — can also fuse into first tile iteration
- In practice: cuBLAS `gemm` handles this; writing it tests your tiling knowledge

---

## 10. Conv2D (2D Convolution)

**C++ naive:**
```cpp
// input:  [H x W], kernel: [KH x KW], output: [OH x OW]
// OH = H - KH + 1, OW = W - KW + 1  (no padding, stride=1)
void conv2d(const float* in, const float* kernel, float* out,
            int H, int W, int KH, int KW) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            float sum = 0;
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    sum += in[(oh + kh) * W + (ow + kw)] * kernel[kh * KW + kw];
            out[oh * OW + ow] = sum;
        }
}
```
- Output index `(oh, ow)`, kernel offset `(kh, kw)` → input index `(oh+kh, ow+kw)`
- OH = H - KH + 1, OW = W - KW + 1 (valid convolution, no padding)

**C++ with batch and channels (realistic):**
```cpp
// input:  [N x C x H x W], kernel: [F x C x KH x KW], output: [N x F x OH x OW]
void conv2dNCHW(const float* in, const float* kernel, float* out,
                int N, int C, int H, int W,
                int F, int KH, int KW) {
    int OH = H - KH + 1, OW = W - KW + 1;
    for (int n = 0; n < N; n++)
        for (int f = 0; f < F; f++)         // output filter
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                    float sum = 0;
                    for (int c = 0; c < C; c++)     // input channel
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                sum += in[((n * C + c) * H + oh + kh) * W + ow + kw]
                                     * kernel[((f * C + c) * KH + kh) * KW + kw];
                    out[((n * F + f) * OH + oh) * OW + ow] = sum;
                }
}
```

**CUDA (one thread per output element, single channel):**
```cpp
__global__ void conv2d(const float* in, const float* kernel, float* out,
                       int H, int W, int KH, int KW, int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;  // output col
    int oh = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    if (oh < OH && ow < OW) {
        float sum = 0;
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                sum += in[(oh + kh) * W + (ow + kw)] * kernel[kh * KW + kw];
        out[oh * OW + ow] = sum;
    }
}

// launch: dim3 block(16,16), grid((OW+15)/16, (OH+15)/16)
```
- Each thread computes one output pixel by sliding the kernel over the input patch
- No shared memory here — kernel is small enough to stay in L1/registers for small KH/KW

**Key formulas to remember:**
```
OH = (H + 2*pad - KH) / stride + 1
OW = (W + 2*pad - KW) / stride + 1
Input patch for output (oh, ow): rows [oh*stride .. oh*stride+KH], cols [ow*stride .. ow*stride+KW]
```

---

## Key Concepts to Always Mention

| Concept | C++ | CUDA |
|---|---|---|
| Memory layout | `A[row*N+col]` flat row-major | same |
| Cache line | 64 bytes = 16 floats, use stride-1 | coalesced warp access |
| Tiling | tiles fit in L1/L2 cache | tiles fit in shared memory |
| Loop order | ikj beats ijk — B sequential | thread indexing determines access pattern |
| Numerical stability | subtract max before softmax | same |
| Synchronization | single threaded, no sync needed | `__syncthreads()` before/after shared mem |
| Bank conflicts | N/A | use `TILE+1` padding in shared memory |
