#include <iostream>
#include <string>
#define TILE 32

void matmul(const float* A, const float* B, float* C, int N){

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
            float sum = 0; 
            for(int k = 0; k < N; k++)
                sum+=A[i*N+k] * B[k*N + j];
            C[i*N+j] = sum;
        }
}

void matmulFast(const float* A, const float* B, float* C, int N){
    for(int i = 0; i < N; i++)
        for(int k = 0; k < N; k++)
        {
            float a = A[i*N+k];
            for(int j = 0; j < N; j++){
                C[i*N+j] += a * B[k*N+j];
            }
        }
}

void matmultiled(const float* A, const float* B, float *C, int N){
    memset(C, 0, N * N * sizeof(float));
    for(int i = 0; i < N; i+=TILE)
        for(int k = 0; k < N; k+= TILE)
            for(int j = 0; j < N; j+=TILE)
                for(int ii = i; ii < i + TILE && ii < N; ii++)
                    for(int kk = k; kk < k + TILE && kk < N; kk++){
                        float a = A[ii*N + kk];
                        for(int jj = j; jj < j + TILE && jj < N; jj++)
                        C[ii*N+jj] += a * B[kk*N+jj];
                    }
}

void conv2D(const float* in, const float* kernel, float* C, int N, int KH, int KH, int H, int W){
    int OH = H - KH + 1; 
    int OW = W - KW + 1; 
    for(int oh = 0; oh < OH; oh++)
        for(int ow = 0; ow < OW; ow++){
            float sum = 0;
            for(int kh = 0; kh < KH; kh++)
                for(int kw = 0; kw < KW; kw++)
                    sum+=in[(oh + kh) * W + (ow+kw)] * kernel[kh * KW + kw];
}

float reduceSum(const float* in, int N){ 
    int sum = 0; 
    for(int i = 0; i < N; i++) sum+= in[i];
    return sum;
}

void softmax(float* in, float* out, int N){
    float maxVal = in[0]; 
    float sum = 0; 
    for(int i = 0; i < N; i++)
        if(maxVal < in[i]){
            maxVal = in[i];
        }
    for(int i = 0; i < N; i++){
        out[i] = expf(in[i] - maxVal); 
        sum+=out[i];
    }
    for(int i = 0; i < N; i++){
        out[i] = out[i] / sum;
    }
}

void transpose(const float* in, float* out, int N){
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            out[i*N+j] = in[j*N+i]; 
}

void transposemod(const float* in, float* out, int N){
    for(int i = 0; i < N; i+=TILE)
        for(int j = 0; j < N; j+=TILE)
            for(int ii = i; ii < i + TILE && ii < N; ii++)
                for(int jj = j; jj < j + TILE && jj < N; jj++){
                    out[jj * N + ii] = in[ii * N + jj];
                }
}

void relu(const float* in, float* out, int N){
    for(int i = 0; i < N; i++)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f; 
}


# CUDA stuff

void relu(const float* in, float* out, int N){
int x = blockIdx.x * blockDim.x + threadIdx.x; 
if(i < N) out[i] = fmaxf(0.0f, in[x]);
}

__global__ void saxpy(int n, float a, float* x, float* y) {
int x = blockIdx.x * blockDim.x + threadIdx.x; 
if(i < n) y[i] = a * x[i] + y[i]; 
}
#define TILE 32
__global__ void transpose(const float* in, float* out, int N) {
    __shared__ tile[TILE][TILE];
    int x = TILE  * blockIdx.x + threadIdx.x; 
    int y = blockIdx.y * TILE + threadIdx.y; 

    if(x < N && y < N){
        tile[threadIdx.y][threadIdx.x] = in[y * N + x]; 
    }
    __syncthreads(); 
    int ox = blockIdx.y * TILE + threadIdx.x; 
    int oy = blockIdx.x * TILE + threadIdx.y; 
    if(ox < N && oy < N)
        out[ox * N + oy] = tile[threadIdx.x][threadIdx.y];
}

__global__ void matmul(const float* A, const float* B, float* C, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < N && col < N){ 
        float sum = 0; 
        for(int k = 0; k < N; k++){
            
        }
    }
}
