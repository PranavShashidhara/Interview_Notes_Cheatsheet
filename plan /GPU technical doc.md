# GPU Roadmap — Weekly Action Plan

## Resume grouping (3 projects total)

- **Project 1: CUDA/Tensor Core Kernel Suite** — Weeks 1, 3, 4, 5, part of 8
- **Project 2: Compiler & Runtime Integration** — Weeks 6, 7, part of 8
- **Project 3: Distributed Training & Inference Communication Analysis** — Week 2 + Week 2b (folds into existing `parallelism-ladder`)

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

### Week 8 (part 1b) — End-to-End Tie-In (self-contained upgrade, no external dependency)

**Goal:** convert "I built fast kernels" into "I built kernels that speed up a real model," fully within your own control.

**Steps:**
1. Take one real transformer layer (small Qwen model, ties to existing `parallelism-ladder` work)
2. Run its forward pass using: (a) stock PyTorch ops, (b) your quantized GEMM + fused attention kernels substituted in
3. Measure one headline number: end-to-end forward-pass speedup (your kernels vs stock)
4. Document any accuracy delta introduced by the INT8 path in this real-model context (not just synthetic benchmark)
5. Add as final section of the kernel suite README — this is the capstone result of Project 1

**Output:** one clear "Xx faster forward pass using my kernels vs stock PyTorch" result, backed by a real model layer, not synthetic-only benchmarks.

**Project 1 resume bullet:** *Built and optimized a CUDA kernel suite spanning register-tiled/Tensor Core GEMM, INT8 quantization with fused dequant, and Flash-Attention-style fused attention; validated across Jetson (sm_87) and A100 (sm_80), closing the Tensor Core gap from 25% to X% of CUTLASS via multistage pipelining, and demonstrated Xx end-to-end speedup on a real transformer layer forward pass*

**If metrics come in strong (WMMA pipeline hits 60%+ of CUTLASS, or end-to-end speedup is large) — stretch additions to push toward 9.5-10:**
- Sweep the pipeline depth (STAGES = 2, 3, 4, 5, 6) and plot occupancy/throughput as a function of stage count — turns one good number into a full characterization curve
- Extend the end-to-end tie-in to a full multi-layer forward pass (not just one layer) and report both latency and peak memory, since memory savings from the attention kernel compound across layers
- Try INT4 in addition to INT8 if the accuracy holds — pushing precision further is a natural extension once INT8 works and reads as "explored the full precision spectrum," not just landed on one point

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
6. Start watching vLLM issue queue (`good first issue` + kernel/quant/benchmark-adjacent) — treat as opportunistic bonus, not a scored deliverable

**Output:** Triton GEMM + attention repos, benchmarked.

---

### Week 7 — PTX Analysis + Autotuner Sweep (self-contained centerpiece — no external dependency)

**Goal:** make the compiler-vs-hand-tuned analysis itself the artifact, rigorous enough to stand alone regardless of any OSS PR outcome.

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
3. **Systematic autotuner sweep (the core upgrade):**
   - Vary `num_stages` (2, 3, 4, 5) and `num_warps` (2, 4, 8) in your Triton config across a matrix size range (512 → 4096)
   - Plot the resulting performance surface (size × config → GFLOPS)
   - At each size, compare the autotuner's *chosen* config against your hand-built pipeline depth from Week 5
   - Produce a quantified finding, e.g.: "Triton's autotuner converges to a 3-stage pipeline at N=2048, matching my hand-tuned choice; at N=512 it picks 2 stages and underperforms my hand-tuned 3-stage config by X%"
4. Package this as a standalone mini-report: dedicated README section with the performance-surface chart + 3-4 written findings on register allocation, pipelining depth, and where the compiler wins/loses vs manual scheduling
5. **Optional bonus (not required for project completeness):** open a vLLM PR informed by this characterization data — a `good first issue` fix backed by real profiling is a stronger submission than a guess, but the project's score should not depend on it merging
```bash
git clone https://github.com/vllm-project/vllm && cd vllm
pip install -e . --break-system-packages
```
   - Comment your approach on the issue before writing code; keep scope small; run relevant tests (`pytest tests/kernels/ -k your_area`); install `pre-commit`

**Output:** standalone PTX/autotuner-sweep report with performance-surface chart — complete and citable independent of any external review outcome. vLLM PR attempted as bonus, tracked separately.

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

**Project 2 resume bullet:** *Ported hand-tuned CUDA kernels to Triton and PyTorch custom operators; characterized Triton's autotuner against hand-scheduled pipelining across a size/config sweep, analyzing generated PTX and benchmarking under torch.compile* (add "; contributed to vLLM" only if a PR merges — treat as optional addendum, not core claim)

**If metrics come in strong (autotuner sweep reveals a clear, surprising pattern, or vLLM PR merges quickly) — stretch additions to push toward 9.5-10:**
- If a pattern in the autotuner sweep suggests a genuine gap (e.g., Triton consistently underperforms at a specific size regime) — file it as a real GitHub issue on the Triton repo with your data attached, even if you don't fix it yourself; a well-characterized issue with reproducible data is itself a contribution
- If the first vLLM PR merges cleanly, attempt a second, more substantive one (ideally kernel-adjacent) while momentum/reviewer trust is fresh — "contributor" with 2 merged PRs reads meaningfully stronger than 1
- Extend the custom-op benchmarking to include a second, harder-to-fuse op (e.g., a full attention+GEMM fused block) under torch.compile, to show the technique generalizes beyond one kernel

---

## PROJECT 3: Distributed Training & Inference Communication Analysis

### Week 2 — NCCL / Comms Characterization (Training)

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

**Multi-GPU rental (self-contained upgrade — non-negotiable for this project to score above 8):**
7. Rent 2-4x A100/H100 (few hours, ~$15-20) — single-GPU NCCL results are a methodology demo, not a real distributed-comms characterization; this is the cheapest single point-gain on the entire roadmap
8. Re-run the message-size sweep and DDP/FSDP/PP/TP comparison on real multi-GPU hardware
9. If the rental platform offers a choice, compare NVLink vs PCIe topology effects on collective bandwidth — a specific, rare data point

**Diagnose-and-fix upgrade (converts "measured" into "measured and resolved"):**
10. From the multi-GPU sweep, identify one clearly comms-bound configuration
11. Implement a concrete fix — gradient bucketing/coalescing adjustment, overlap reordering, or communication-computation reordering
12. Measure and report before/after numbers for the fix (exposed comms % reduction, throughput improvement)

**Output:** new "Communication Characterization" section in `parallelism-ladder` README, backed by real multi-GPU data, including one diagnosed-and-resolved comms bottleneck with before/after numbers.

---

### Week 2b — Distributed Inference Comms Characterization (extends the same multi-GPU rental)

**Goal:** apply the same comms-profiling methodology to serving, not just training — this is what Google/Sarvam/Anthropic JDs actually mean by "distributed systems" in an inference context, and it's a distinct skill from training-side parallelism.

**Steps:**
1. Set up tensor-parallel inference serving using vLLM (`tensor_parallel_size > 1`) on your rented multi-GPU instance
2. Profile the same way as training: NCCL collective activity during a serving forward pass (all-reduce across TP ranks per layer), using `torch.profiler` + Nsight Systems
3. Measure request latency and throughput vs TP degree (TP=1, 2, 4) — characterize the tradeoff: more GPUs reduces per-GPU memory/compute but adds comms overhead per token generated
4. Compare comms pattern shape: training comms (large, infrequent — one all-reduce per gradient sync) vs inference comms (small, frequent — one all-reduce per layer per token) — this contrast is itself a strong technical finding
5. Identify whether serving is comms-bound at small batch sizes (common in low-latency serving) — quantify the crossover point where TP overhead exceeds its throughput benefit
6. If time allows: compare tensor parallelism vs pipeline parallelism for inference serving specifically — different tradeoffs than training (TP adds per-token latency, PP adds pipeline bubble at low batch)

**Output:** new "Distributed Inference Communication Characterization" section — latency/throughput vs TP degree chart, training-vs-inference comms pattern comparison, comms-bound crossover analysis.

**Project 3 resume bullet:** *Profiled NCCL collective performance and compute/communication overlap across DDP/FSDP/TP training configurations and tensor-parallel inference serving on multi-GPU hardware; diagnosed a communication-bound configuration and implemented a fix reducing exposed comms time by X%; characterized latency/throughput tradeoffs of TP degree in serving contexts*

**If metrics come in strong (the comms fix yields a large improvement, or the training-vs-inference contrast is unusually clean) — stretch additions to push toward 9.5-10:**
- If the fix generalizes, apply the same technique across all four parallelism strategies (DDP/FSDP/PP/TP) rather than just the one where it was diagnosed, and report the fix's impact across all of them — turns one point-fix into a systematic optimization pass
- Add a batch-size sweep to the inference TP analysis — the comms-bound crossover point moves with batch size, and mapping that curve (not just one crossover point) is a stronger, more complete result
- If NVLink vs PCIe comparison data came out clean, extend it into a short standalone writeup on interconnect topology's effect on both training and inference comms — this is genuinely rare public data and could be shared/circulated on its own (GPU-mode Discord, a short blog post), which raises visibility beyond just the resume line

---

## Standing rules (every week)

- DS&A: 4-5 hrs/week
- Any interview preempts this schedule
- Something public ships every Friday
- Repos stay separate: `cuda-memory-hierarchy-benchmarks` (Project 1), `triton-kernel-comparison` (Project 2), `parallelism-ladder` (Project 3)