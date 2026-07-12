# GPU Performance / Systems Engineering Roadmap — Full Context Document

**Purpose of this document:** consolidated roadmap + technical implementation reference for an 8-week GPU/systems engineering portfolio build, intended to be fed as context to an LLM for future planning, drafting, or troubleshooting sessions.

**Person:** Pranav Shashidhara — MS Data Science (UMD, May 2026, GPA 3.97), ~4 yrs experience (3 yrs Technical Analyst at Oracle Bangalore — ETL/PL-SQL/APEX; ~1 yr ML Engineer at UMD MTech Ventures — agentic AI systems). On F-1 OPT, actively job searching, self-imposed December 2026 checkpoint (return to India if no US offer by then).

**Goal:** build a credible, verifiable GPU performance / ML systems portfolio to target feeder-tier roles now (Google GPU Perf, Sarvam AI training infra) with a realistic 2027–2028 path to staff-tier labs (Anthropic, OpenAI, Baseten).

---

## 1. Existing assets (before this plan starts)

**`cuda-memory-hierarchy-benchmarks`** (public repo, Jetson Orin Nano sm_87):
- Full kernel ladder: naive → tiled v1 (32×32 shared mem) → tiled v2 (WPT=4, 8×8 register tile) → tiled v3 (cp.async + WPT=4) → TC WMMA (fp16) → cuBLAS/CUTLASS ceilings
- Key results (2048×2048): naive 171 GFLOPS → tiled v2 968 GFLOPS (5.65× — register blocking, biggest fp32 win) → WMMA 1,777 GFLOPS (10.4×, still memory-bound) vs CUTLASS fp16 TC 7,023 GFLOPS (WMMA sits at ~25% of this ceiling — the key gap to close)
- Roofline analysis via Nsight Compute: hand-written kernels are memory-bound; library fp32 (cuBLAS/CUTLASS) crosses the ridge point into compute-bound; fp16 TC kernels return to memory-bound because sm_87's ~40 TOPS roofline outruns available bandwidth
- Memory bandwidth microbenchmarks: sequential vs strided vs random access patterns, cp.async pipelining

**`parallelism-ladder`** (public repo):
- Distributed fine-tuning progression: DDP → FSDP → Pipeline Parallel → Tensor Parallel on Qwen2.5-3B
- NCCL instrumentation already present

**LLM Orchestration Platform (LOP):**
- Llama-3.1 8B fine-tuned via QLoRA (4-bit), vLLM inference with adapter merging, KV-cache optimization, NVFP4 quantization exposure

**News Research Agent, MediAssist AI, Toxicity Classification:** separate agentic/applied-AI portfolio track — not part of this GPU roadmap, no overlap, kept on a different resume variant.

**Diagnostic note on communication style:** technical depth is real; occasional gap is *vocabulary mapping* — e.g., not initially recognizing "register blocking" as the canonical name for already-implemented WPT=4 tiling. Fix: explicitly map every implemented technique to its textbook name + one-line result before interviews.

---

## 2. Target roles and honest calibration

| Target | Tier | Comp (where known) | Calibration |
|---|---|---|---|
| Google, SWE GPU Performance (Sunnyvale) | Apply now | $147K–211K + 15% bonus + equity | Meets all minimum quals today; 3/4 preferred quals met; referral path via Nanda (existing, slow-moving — needs specific req to unstick) |
| Sarvam AI, ML Engineer Training Infra (Bengaluru) | Apply now (after Weeks 1–7 artifacts) | Unverified — treat ₹84 LPA anecdote as unreliable; realistic early-career estimate ~₹25–45 LPA + equity, confirm directly with recruiter | Explicit "exceptional early-career" exception path stated in JD |
| Baseten, GPU Kernel Engineer (SF) | Long-horizon (2027–28) | $180K–360K | Specialist role, 4-month-old req w/ low applicant count (scarcity signal), but expects near-veteran kernel depth |
| Anthropic, GPU Performance Engineer | Long-horizon (2027–28) | $280K–850K | Staff-level bar ("thousands of GPUs," "production ML systems at scale"); portfolio work cannot close the production-scale gap |
| OpenAI, Inference — AMD GPU Enablement (SF) | Long-horizon (2027–28) | $295K–555K | Role *type* (systems integration, bring-up/debug) fits well; AMD/HIP arbitrage noted but deliberately skipped in current plan; scale gap unclosed |

**Core strategic finding:** projects substitute for experience most effectively when (a) objectively verifiable by a stranger in minutes, and (b) the skill is scarce enough that companies can't insist on veteran experience. GPU kernel work satisfies both better than almost any other engineering specialty — a GEMM at X% of cuBLAS is a fact, not a claim. This is why the project-heavy strategy is viable here specifically, more than it would be for e.g. generic "AI engineer" roles.

**Sequencing logic:** feeder tier now → production-scale experience 2027–2028 → staff-tier labs become realistic, likely via a warm path from sustained public OSS/kernel presence rather than a cold application.

---

## 3. Standing rules (run throughout all 8 weeks)

- Apply to live reqs immediately, not after prep — prep runs in parallel with active applications, not before them
- Google application + Nanda referral ping (with specific req link) — highest-priority open action, not yet completed as of this document
- DS&A: 4–5 hrs/week throughout, medium difficulty, arrays/graphs/DP rotation — Google's loop tests this regardless of GPU depth
- Any real interview/phone screen preempts the build schedule entirely
- Something public ships (commit, README update, chart) every Friday
- Repos stay separate and individually legible: `cuda-memory-hierarchy-benchmarks`, `triton-kernel-comparison` (new), `parallelism-ladder` — not consolidated into one monorepo
- HIP/AMD thread deliberately excluded from this version of the plan (was scoped, then dropped by request — CUDA-only focus retained)

---

## 4. Week-by-week plan with technical implementation

### Week 1 — Low-Precision: INT8 Quantized GEMM

**Goal:** per-channel quantized INT8 GEMM with fused dequant epilogue, benchmarked against existing fp16/fp32 kernels.

**Quantization scheme:**
```
scale[j] = max(abs(W[:, j])) / 127          # per-output-channel, symmetric
W_int8[i,j] = round(W[i,j] / scale[j])       # clamp to [-127, 127]
output_fp16 = int32_accumulator * scale_row * scale_col   # fused in-kernel
```

**Core intrinsic:** `__dp4a(int, int, int)` — 4-way INT8 dot product, INT32 accumulate, native on sm_61+ (works on both Jetson sm_87 and A100 sm_80). Stretch: INT8 tensor core path via `wmma::experimental::precision::s8` fragments.

**Validation:** correctness harness before kernel work — max abs error, mean relative error, cosine similarity vs fp16 reference. Build this first, use it to validate every subsequent kernel this week.

**Profiling:**
```bash
ncu --set full -o int8_gemm_report ./int8_gemm_bench
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./int8_gemm_bench
```

**Deliverable:** benchmark sweep 512→4096, tradeoff chart (throughput gain vs accuracy loss), extend `plot_results.py` with `int8_gemm` as recognized variant, README section linking to NVFP4/vLLM serving work in LOP.

**Resume bullet:** *Built INT8 GEMM with per-channel scaling and fused dequantization epilogue (dp4a); Xx throughput over fp16 at <Y% accuracy delta*

---

### Week 2 — Distributed Systems: NCCL / Comms Characterization

**Goal:** extract and deepen a communication-characterization section inside the existing `parallelism-ladder` repo.

**Stack:** `torch.distributed` (backend=`nccl`), `NCCL_DEBUG=INFO`, `torch.profiler`.

**Instrumentation:**
```python
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
prof.export_chrome_trace("nccl_trace.json")
```

**Message-size sweep:** loop tensor sizes 1KB→1GB (powers of 2), time with `torch.cuda.Event`, plot effective bandwidth vs message size — reveals latency-bound → bandwidth-bound crossover point.

**Overlap analysis via Nsight Systems:**
```bash
nsys profile -o fsdp_trace --trace=cuda,nvtx,osrt python train_step.py
nsys stats fsdp_trace.nsys-rep
```
Use `torch.cuda.nvtx.range_push/pop` around compute sections to visually separate compute vs comms on the timeline. Compute "% exposed comms" = comms time not overlapping compute / total comms time.

**Deliverable:** cross-strategy table (DDP/FSDP/PP/TP — dominant collective type, exposed comms %, bottleneck location), new README section "Communication Characterization."

**Resume bullet:** *Characterized NCCL collective performance and compute/communication overlap across DDP/FSDP/TP configurations; identified communication-bound regimes via Nsight Systems timeline analysis*

---

### Weeks 3–4 — Fused Attention Kernel (Flash-Attention-style)

**Goal:** online-softmax fused attention kernel, Q/K/V tiled, causal masking, benchmarked vs naive and PyTorch SDPA.

**Reference/correctness oracle:**
```python
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Kernel structure:** grid = one block per (batch, head, query-tile); shared memory holds K/V tile (`Bc × head_dim`); registers hold running max `m_i`, running sum `l_i`, output accumulator.

**Online softmax algorithm (memorize this — standard whiteboard interview question):**
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

**Causal optimization:** skip any K/V tile where `tile_start > query_row` entirely — don't load it, free speedup, quantify separately.

**Validation:** `torch.testing.assert_close(output, ref, atol=1e-3, rtol=1e-3)`.

**Benchmark matrix:** seq lengths 512/1K/2K/4K/8K, head dims 64/128, vs naive 3-kernel baseline and PyTorch SDPA. Headline charts: latency vs seq length, peak memory vs seq length (O(N²) vs O(N), "naive OOMs at 8K, mine doesn't" table).

**PyTorch custom op wrapper (build once, reuse for all future kernels):**
```python
from torch.library import Library, impl

mylib = Library("mylib", "DEF")
mylib.define("fused_attention(Tensor q, Tensor k, Tensor v, bool causal) -> Tensor")

@impl(mylib, "fused_attention", "CUDA")
def fused_attention_cuda(q, k, v, causal):
    return your_cuda_extension.fused_attention(q, k, v, causal)

@impl(mylib, "fused_attention", "Meta")
def fused_attention_meta(q, k, v, causal):
    return torch.empty_like(q)   # shape-only, enables tracing/compile
```

**Resume bullet:** *Implemented Flash-Attention-style fused kernel with online softmax and tiled Q/K/V: Xx speedup over naive at 4K context, O(N) memory scaling validated to 8K where naive OOMs*

---

### Week 5 — WMMA Multistage Pipeline (close CUTLASS gap: current ~25% → target 50%+)

**Goal:** software-pipeline the existing WMMA GEMM kernel to close the gap against CUTLASS fp16 TC (1,777 vs 7,023 GFLOPS currently).

**Key primitives:**
- `nvcuda::wmma::fragment<...>` — tensor core fragment types
- `cuda::memcpy_async` / `cp.async.cg.shared.global` (PTX) — async global→shared copy
- `__pipeline_wait_prior<N>()` — wait for all but N most recent async copies

**Pipeline skeleton:**
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

**Profiling focus:** occupancy vs single-stage version —
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed ./wmma_pipelined_bench
```
Expect occupancy to drop while throughput rises — this divergence is the key finding (more registers/thread for pipeline state → fewer resident warps → but each thread has more ILP to hide latency).

**Reference (read, don't copy):** CUTLASS `include/cutlass/gemm/threadblock/mma_multistage.h`.

**Fallback:** if target not hit, document the gap analysis honestly as "future work" in README — still a credible artifact.

**Resume bullet:** *Optimized fp16 Tensor Core GEMM from 25% to X% of CUTLASS via multistage software pipelining (double-buffered cp.async, staged fragment loads)*

---

### Weeks 6–7 — Triton Ports + PTX Analysis + OSS Contribution (vLLM)

**Setup:** `pip install triton --break-system-packages`

**Triton GEMM (autotuned):**
```python
import triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```
`num_stages` is directly comparable to the hand-written pipeline depth from Week 5 — this is the bridge for the compiler-vs-hand-tuned comparison writeup.

**PTX extraction — Triton side:**
```python
kernel = matmul_kernel.warmup(*args, grid=grid)
print(kernel.asm['ptx'])   # or kernel.asm['ttgir'] for Triton's own IR
```

**PTX/SASS extraction — CUDA side:**
```bash
nvcc -ptx your_kernel.cu -o your_kernel.ptx
cuobjdump --dump-sass your_kernel.o > your_kernel.sass
```

**Fused attention in Triton:** adapt from `triton-lang/triton/python/tutorials/06-fused-attention.py` — understand it, reimplement independently, then compare (don't copy verbatim).

**Analysis deliverable:** 3–4 concrete findings on pipelining depth, register allocation, and where the compiler's autotuner wins/loses vs manual scheduling.

**vLLM OSS contribution workflow:**
```bash
git clone https://github.com/vllm-project/vllm && cd vllm
pip install -e . --break-system-packages
```
- Filter issues: `good first issue` label + kernel/quantization/benchmark-adjacent
- Comment proposed approach on the issue before writing code
- Keep first PR scope small: shape edge-case, test gap, benchmark script, docs fix — not a new kernel
- Run relevant tests locally before opening PR: `pytest tests/kernels/ -k your_area`
- Install `pre-commit` to match their CI lint requirements
- Review cycles run 1–3 weeks; let it ride into Week 8 if needed

**Resume bullets:**
- *Ported hand-pipelined CUDA kernels to Triton; analyzed generated PTX to characterize compiler autotuning vs manual scheduling tradeoffs*
- *Contributor to vLLM — [PR one-liner]* (once merged; keep "open PR" off resume, fair game in conversation)

---

### Week 8 — A100 Validation + PyTorch Custom Ops + torch.compile

**Rental setup:**
```bash
nvidia-smi                          # confirm A100
nvcc --version                      # confirm CUDA toolkit
python -c "import torch; print(torch.cuda.get_device_capability())"  # expect (8,0)
```

**Re-run scope:** full ladder + Week 1 INT8 + Weeks 3–4 attention + Week 5 pipelined WMMA, all on sm_80.

**Cross-architecture table (fill in during Week 8):**
| Kernel | Jetson sm_87 GFLOPS | A100 sm_80 GFLOPS | Regime (Jetson) | Regime (A100) |
|---|---|---|---|---|
| Tiled v2 (reg tile) | 968 | ? | Memory-bound | ? |
| WMMA (single-stage) | 1,777 | ? | Memory-bound | ? |
| WMMA (pipelined) | ? | ? | ? | ? |

**Custom op build:**
```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    ext_modules=[CUDAExtension('my_kernels', ['bindings.cpp', 'gemm_wmma.cu', 'attention.cu'])],
    cmdclass={'build_ext': BuildExtension}
)
```

**torch.compile integration:**
```python
@torch.compile
def model_forward(x, w):
    return torch.ops.mylib.wmma_gemm(x, w)

torch._dynamo.explain(model_forward)(x, w)   # check for graph breaks
```

**Compare vs Inductor:**
```python
compiled_baseline = torch.compile(lambda a, b: a @ b, mode="max-autotune")
# benchmark compiled_baseline vs torch.ops.mylib.wmma_gemm on identical inputs
```

**Final deliverables:** consolidated README across all repos, 90-second spoken narrative per project (interview prep), cross-architecture writeup.

**Resume bullet:** *Packaged custom CUDA kernels as PyTorch operators (torch.library, autograd, meta kernels); benchmarked against Inductor-generated code under torch.compile*

---

## 5. JD coverage matrix (Google / Anthropic / OpenAI / Sarvam composite)

| Skill area | Covered by | Status |
|---|---|---|
| CUDA kernels, tensor cores | Existing suite + Week 5 | Strong |
| Flash Attention / attention optimization | Weeks 3–4 | New |
| Triton | Weeks 6–7 | New |
| PTX / compiler / codegen exposure | Week 7 | New |
| Nsight profiling (Compute + Systems) | Existing suite + Week 2 | Strong |
| Kernel fusion / memory bandwidth optimization | Existing suite (reframed) | Strong |
| INT8/FP8, mixed precision | Week 1 + existing NVFP4 serving (LOP) | New + existing |
| NCCL, collectives, model parallelism | Week 2 + parallelism-ladder | Existing, deepened |
| PyTorch internals, custom operators | Week 8 | New |
| OSS contribution (training-infra ecosystem) | Weeks 6–7 (vLLM) | New |
| torch.compile / XLA | Week 8 (partial — Inductor comparison only) | Partial |
| HIP/AMD, RCCL | — | Deliberately excluded from this plan |
| Production / large-cluster scale | — | Not coverable by solo projects — feeder-role gap |

---

## 6. Reference material (read during relevant weeks, don't copy)

- CUTLASS docs — `docs/efficient_gemm.md`, `include/cutlass/gemm/threadblock/mma_multistage.h`
- Triton official tutorials — `triton-lang/triton/python/tutorials/` (matmul, fused-softmax, fused-attention)
- "Flash Attention" and "Flash Attention 2" papers — online softmax derivation, IO-complexity analysis
- NVIDIA "CUDA C++ Best Practices Guide" and Tensor Core programming docs
- vLLM `CONTRIBUTING.md` and kernels directory structure

---

## 7. Open threads / unresolved as of last discussion

- Nanda referral ping + Google application: **not yet sent** — top-priority open action, independent of project timeline
- Sarvam compensation: unverified from public sources; do not anchor expectations, confirm directly with recruiter if/when in process
- Whether to formally add Sarvam to active warm-path outreach now vs. wait for Week 1–7 artifacts to exist first: undecided
- WMMA multistage pipeline (Week 5) is higher-risk/effort than other weeks — budget slack, has an explicit "document as future work" fallback
- Compression/cut order if an interview lands mid-plan: Week 8 comms analysis first (parallelism-ladder already has partial evidence), then Week 5 WMMA pipeline second; Weeks 1–4 and 6–7 are considered non-negotiable core