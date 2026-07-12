# GPU Roadmap — Weekly Action Plan

## Resume grouping (3 projects total)

- **Project 1: CUDA/Tensor Core Kernel Suite** — Weeks 1, 3, 4, 5, part of 8
- **Project 2: Compiler & Runtime Integration** — Weeks 6, 7, part of 8
- **Project 3: Distributed Training Communication Analysis** — Week 2 (folds into existing `parallelism-ladder`)

---

## PROJECT 1: CUDA/Tensor Core Kernel Suite

### Week 1 — INT8 Quantized GEMM

**Quantization scheme:**
```
scale[j] = max(abs(W[:, j])) / 127
W_int8[i,j] = round(W[i,j] / scale[j])
output_fp16 = int32_accumulator * scale_row * scale_col   # fused in-kernel
```

**Core intrinsic:** `__dp4a(int, int, int)` — 4-way INT8 dot product, INT32 accumulate (sm_61+, works on Jetson).

**Steps:**
1. Build correctness harness first: max abs error, mean relative error, cosine similarity vs fp16 reference
2. Naive INT8 kernel — correctness oracle
3. `dp4a` tiled INT8 kernel (reuse existing tiling skeleton)
4. Fused dequant epilogue in-kernel
5. Benchmark sweep 512→4096 vs existing fp16 WMMA / fp32 tiled kernels
6. Chart: throughput gain vs accuracy loss
7. Profile: `ncu --set full -o int8_gemm_report ./int8_gemm_bench`

**Output:** README section + chart + extended `plot_results.py` with `int8_gemm` variant.

---

### Week 3 — Fused Attention: Correctness + Online Softmax

**Steps:**
1. Naive 3-kernel baseline: S=QKᵀ → softmax(S) → S·V, fp32, global memory — reference oracle
2. Fused two-pass version: tile K/V through shared memory, two passes over K
3. Validate vs `torch.nn.functional.scaled_dot_product_attention`, `atol=1e-3`
4. Implement online softmax (single pass):
```
for each K/V tile j:
    S_ij = Q_i @ K_j^T * scale
    m_new = max(m_i, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
    acc = exp(m_i - m_new) * acc + P_ij @ V_j
    m_i, l_i = m_new, l_new
output = acc / l_i
```
5. Write the derivation into README as you implement it

**Output:** working single-pass fused attention kernel, fp32, validated.

---

### Week 4 — Fused Attention: fp16, Causal, Benchmarks, Custom Op

**Steps:**
1. Convert to fp16 compute / fp32 accumulate
2. Add causal masking — skip fully-masked K/V tiles entirely
3. Tile-size sweep (Br×Bc), document shared-memory-capacity vs occupancy tradeoff
4. Benchmark grid: seq lengths 512/1K/2K/4K/8K, head dims 64/128, vs naive and SDPA
5. Charts: latency vs seq length, peak memory vs seq length (O(N²) vs O(N)); "naive OOMs at 8K, mine doesn't" table
6. Nsight Compute pass: quantify memory traffic reduction vs naive
7. Wrap as PyTorch custom op:
```python
from torch.library import Library, impl
mylib = Library("mylib", "DEF")
mylib.define("fused_attention(Tensor q, Tensor k, Tensor v, bool causal) -> Tensor")

@impl(mylib, "fused_attention", "CUDA")
def fused_attention_cuda(q, k, v, causal):
    return your_cuda_extension.fused_attention(q, k, v, causal)

@impl(mylib, "fused_attention", "Meta")
def fused_attention_meta(q, k, v, causal):
    return torch.empty_like(q)
```

**Output:** complete fused attention kernel with benchmarks, charts, custom op wrapper.

---

### Week 5 — WMMA Multistage Pipeline

**Goal:** close gap between current WMMA kernel (~25% of CUTLASS fp16) and target 50%+.

**Key primitives:** `nvcuda::wmma::fragment<...>`, `cuda::memcpy_async` / `cp.async.cg.shared.global`, `__pipeline_wait_prior<N>()`.

**Steps:**
1. Read CUTLASS's `include/cutlass/gemm/threadblock/mma_multistage.h` for the pattern
2. Implement staged loading:
```cuda
constexpr int STAGES = 4;
for (int s = 0; s < STAGES - 1; s++) {
    cp_async_load(shared_buf[s], global_ptr, tile_offset(s));
    __pipeline_commit();
}
for (int tile = 0; tile < num_tiles; tile++) {
    __pipeline_wait_prior<STAGES - 2>();
    __syncthreads();
    if (tile + STAGES - 1 < num_tiles)
        cp_async_load(shared_buf[(tile + STAGES - 1) % STAGES], global_ptr, tile_offset(tile + STAGES - 1));
    __pipeline_commit();
    wmma::load_matrix_sync(a_frag, shared_buf[tile % STAGES].a, ...);
    wmma::load_matrix_sync(b_frag, shared_buf[tile % STAGES].b, ...);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __syncthreads();
}
```
3. Debug/tune — watch register pressure vs occupancy (budget slack, hardest week)
4. Benchmark vs CUTLASS fp16 TC and original single-stage WMMA
5. Profile: `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed`
6. Document occupancy-vs-throughput tradeoff

**Fallback:** if 50% not hit, document gap analysis honestly as "future work."

---

### Week 8 (part 1) — A100 Cross-Architecture Validation

**Steps:**
1. Rent A100 (Lambda/RunPod), confirm setup:
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.get_device_capability())"  # expect (8,0)
```
2. Re-run full kernel suite on sm_80: original ladder + INT8 + attention + pipelined WMMA
3. Fill cross-architecture table (Jetson sm_87 vs A100 sm_80 GFLOPS + regime per kernel)
4. Write cross-architecture findings — where do roofline regimes flip?

**Project 1 resume bullet:** *Built and optimized a CUDA kernel suite spanning register-tiled/Tensor Core GEMM, INT8 quantization with fused dequant, and Flash-Attention-style fused attention; validated across Jetson (sm_87) and A100 (sm_80), closing the Tensor Core gap from 25% to X% of CUTLASS via multistage pipelining*

---

## PROJECT 2: Compiler & Runtime Integration

### Week 6 — Triton Ports + PTX Extraction

**Steps:**
1. `pip install triton --break-system-packages`
2. Work through Triton matmul + fused-softmax tutorials
3. Port GEMM to Triton with autotuning:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...): ...
```
4. Benchmark: your CUDA vs your Triton vs cuBLAS, plus lines-of-code comparison
5. Port fused attention to Triton — compare against `triton-lang/triton/python/tutorials/06-fused-attention.py`
6. Start watching vLLM issue queue (`good first issue` + kernel/quant/benchmark-adjacent)

**Output:** Triton GEMM + attention repos, benchmarked.

---

### Week 7 — PTX Analysis + vLLM PR

**Steps:**
1. Extract Triton PTX:
```python
kernel = matmul_kernel.warmup(*args, grid=grid)
print(kernel.asm['ptx'])
```
2. Extract your CUDA PTX/SASS:
```bash
nvcc -ptx your_kernel.cu -o your_kernel.ptx
cuobjdump --dump-sass your_kernel.o > your_kernel.sass
```
3. Write up 3-4 concrete findings: pipelining depth, register allocation, where Triton's autotuner wins/loses vs hand-scheduling
4. vLLM PR:
```bash
git clone https://github.com/vllm-project/vllm && cd vllm
pip install -e . --break-system-packages
```
   - Comment your approach on a `good first issue` before writing code
   - Keep scope small: shape edge-case, test gap, benchmark script, docs fix
   - Run relevant tests: `pytest tests/kernels/ -k your_area`
   - Install `pre-commit` to match CI lint
   - Let review ride into Week 8 if needed

**Output:** PTX comparison writeup, vLLM PR opened.

---

### Week 8 (part 2) — Custom Ops + torch.compile

**Steps:**
1. Package best kernels as PyTorch custom ops:
```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    ext_modules=[CUDAExtension('my_kernels', ['bindings.cpp', 'gemm_wmma.cu', 'attention.cu'])],
    cmdclass={'build_ext': BuildExtension}
)
```
2. Test under torch.compile, check for graph breaks:
```python
@torch.compile
def model_forward(x, w):
    return torch.ops.mylib.wmma_gemm(x, w)
torch._dynamo.explain(model_forward)(x, w)
```
3. Benchmark vs Inductor-generated code (`torch.compile(..., mode="max-autotune")`)
4. Consolidate READMEs; draft 90-second spoken narrative per project

**Project 2 resume bullet:** *Ported hand-tuned CUDA kernels to Triton and PyTorch custom operators, analyzing generated PTX against hand-scheduled code and benchmarking under torch.compile; contributed a merged fix to vLLM*

---

## PROJECT 3: Distributed Training Communication Analysis

### Week 2 — NCCL / Comms Characterization

**Steps:**
1. Add NCCL profiling hooks to `parallelism-ladder` using `torch.profiler`:
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
prof.export_chrome_trace("nccl_trace.json")
```
2. Sweep message sizes 1KB→1GB, time with `torch.cuda.Event`, plot effective bandwidth vs message size
3. Nsight Systems trace of an FSDP/TP training step:
```bash
nsys profile -o fsdp_trace --trace=cuda,nvtx,osrt python train_step.py
```
4. Add `torch.cuda.nvtx.range_push/pop` around compute sections to separate compute vs comms on timeline
5. Compute "% exposed comms" per parallelism strategy
6. Build cross-strategy table: DDP/FSDP/PP/TP — dominant collective, exposed comms %, bottleneck

**Output:** new "Communication Characterization" section in `parallelism-ladder` README.

**Project 3 resume bullet:** *Profiled NCCL collective performance and compute/communication overlap across DDP/FSDP/TP configurations, identifying communication-bound regimes via Nsight Systems*

---

## Standing rules (every week)

- DS&A: 4-5 hrs/week
- Any interview preempts this schedule
- Something public ships every Friday
- Repos stay separate: `cuda-memory-hierarchy-benchmarks` (Project 1), `triton-kernel-comparison` (Project 2), `parallelism-ladder` (Project 3)