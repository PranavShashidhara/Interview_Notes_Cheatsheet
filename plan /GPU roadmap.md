# GPU Performance / Systems Engineering Roadmap

**Target checkpoint:** December 2026
**Primary near-term targets:** Google (Software Engineer, GPU Performance — Sunnyvale), Sarvam AI (ML Engineer, Training Infra — Bengaluru)
**Stretch / long-horizon targets (2027–2028):** Anthropic (GPU Performance Engineer), OpenAI (Inference, AMD GPU Enablement), Baseten (GPU Kernel Engineer)

---

## Standing rules (run throughout, never deprioritized)

- Apply to live reqs **now**, not after prep — prep runs in parallel, not in series
- Google application + Nanda referral ping — **send this week, before Week 1 starts**
- DS&A: 4–5 hrs/week throughout, medium difficulty, arrays/graphs/DP rotation
- Any real interview/phone screen **preempts** the build schedule
- Something public ships (commit, README update, chart) every Friday
- Repos stay separate and individually legible — not one monorepo

---

## Repo structure

1. **`cuda-memory-hierarchy-benchmarks`** (existing, flagship) — extend with INT8 kernel, fused attention, WMMA pipeline, A100 re-run, custom ops
2. **`triton-kernel-comparison`** (new) — Triton ports, PTX/compiler analysis, vLLM PR reference
3. **`parallelism-ladder`** (existing) — extend with NCCL/comms characterization section

---

## Week-by-week plan

### Week 1 — Low-Precision (INT8 Quantized GEMM)
- Days 1–2: Per-channel absmax quantization math + accuracy harness (max abs error, cosine sim vs fp16)
- Days 3–4: Naive INT8 reference kernel for correctness oracle
- Days 3–5: `dp4a` INT8 GEMM kernel (sm_87-native, reuse existing tiling skeleton)
- Day 5: Fused dequantization epilogue (INT32 acc → scale → fp16 output, in-kernel)
- Days 6–7: Benchmark sweep (512→4096) vs fp16 WMMA/fp32 tiled; tradeoff chart (throughput gain vs accuracy loss); connect to NVFP4/vLLM serving work in README

**Bullet:** *Built INT8 GEMM with per-channel scaling and fused dequantization epilogue (dp4a); Xx throughput over fp16 at <Y% accuracy delta*

### Week 2 — Distributed Systems (NCCL / Comms Characterization)
- Days 1–2: Instrument parallelism-ladder with NCCL profiling hooks (CUDA events or `torch.profiler`)
- Day 3: Collective latency/bandwidth sweep (1KB→1GB), plot latency-bound → bandwidth-bound crossover
- Days 4–5: Nsight Systems trace of FSDP/TP training step; quantify % exposed (non-overlapped) communication
- Days 6–7: Cross-strategy table (DDP/FSDP/PP/TP — dominant collective, exposed comms %, bottleneck); new README section "Communication Characterization"

**Bullet:** *Characterized NCCL collective performance and compute/communication overlap across DDP/FSDP/TP; identified communication-bound regimes via Nsight Systems*

### Weeks 3–4 — Fused Attention Kernel
- Week 3, Days 1–2: Naive 3-kernel baseline (QKᵀ → softmax → ·V), fp32, global memory — correctness oracle
- Days 3–5: Fused two-pass version (shared-mem tiled K/V, two passes over K)
- Days 6–7: Online softmax — single pass, running max/sum, rescaling derivation written into README
- Week 4, Days 1–2: fp16 compute/fp32 accumulate, causal masking (skip fully-masked tiles)
- Days 3–4: Tile-size sweep (Br×Bc), shared-memory-capacity vs occupancy tradeoff
- Days 5–6: Benchmark grid (seq 512→8K, head dim 64/128) vs naive + PyTorch SDPA; latency + O(N²)→O(N) memory charts
- Day 7: Nsight Compute — quantify memory traffic reduction directly

**Bullet:** *Implemented Flash-Attention-style fused kernel with online softmax: Xx over naive at 4K context, O(N) memory validated to 8K*

### Week 5 — WMMA Multistage Pipeline (close CUTLASS gap: 25% → 50%+)
- Days 1–2: Study CUTLASS multistage pipeline pattern (cp.async staged loads overlapping MMA issue)
- Days 3–5: Implement staged loading, software-pipeline 3–4 stages, double/triple shared-mem buffering
- Day 6: Debug/tune — budget slack, watch register pressure vs occupancy
- Day 7: Benchmark vs CUTLASS fp16 TC and original single-stage WMMA; document occupancy-vs-throughput tradeoff
- *(If target missed: document gap analysis honestly as "future work" — still a strong artifact)*

**Bullet:** *Optimized fp16 Tensor Core GEMM from 25% to X% of CUTLASS via multistage software pipelining*

### Weeks 6–7 — Triton Ports + PTX Analysis + OSS Contribution
- Week 6, Days 1–2: Triton tutorials (matmul, fused softmax)
- Days 3–5: Port GEMM to Triton, autotune, 3-way benchmark + LOC comparison
- Days 6–7: Port fused attention to Triton, compare vs Triton's own tutorial implementation
- Week 7, Days 1–2: PTX analysis — Triton-generated vs hand-written (pipelining, register allocation); 3–4 concrete findings
- Days 3–7: vLLM PR — watch issue queue from Week 4 onward, filter `good first issue` (kernel/quant/benchmark-adjacent), comment approach before coding, small scope, follow contribution guide exactly. Review may run past Week 7 — fine, let it ride.

**Bullets:**
- *Ported hand-pipelined CUDA kernels to Triton; analyzed generated PTX to characterize compiler autotuning vs manual scheduling*
- *Contributor to vLLM — [PR one-liner]* (once merged)

### Week 8 — A100 Validation + PyTorch Custom Ops
- Days 1–2: Rent A100, re-run full suite (GEMM ladder + INT8 + attention + pipelined WMMA) on sm_80
- Day 3: Cross-architecture writeup — where do roofline regimes flip vs Jetson sm_87?
- Days 4–5: Wrap 2–3 best kernels as PyTorch custom ops (`torch.library` + `cpp_extension`, autograd stubs, meta/fake kernel registration)
- Days 6–7: Run under `torch.compile`, confirm no graph breaks, benchmark vs Inductor-generated code. Consolidated README across all repos. Draft 90-second spoken narrative per project.

**Bullet:** *Packaged custom CUDA kernels as PyTorch operators (torch.library, autograd, meta kernels); benchmarked against Inductor-generated code under torch.compile*

---

## JD coverage matrix

| Skill area | Covered by | Status |
|---|---|---|
| CUDA kernels, tensor cores | Existing suite + Week 5 | Strong |
| Flash Attention / attention optimization | Weeks 3–4 | New |
| Triton | Weeks 6–7 | New |
| PTX / compiler exposure | Week 7 | New |
| Nsight profiling | Existing suite | Strong |
| Kernel fusion / bandwidth optimization | Existing suite (reframed) | Strong |
| INT8/FP8, mixed precision | Week 1 + existing NVFP4 serving work | New + existing |
| NCCL, collectives, model parallelism | Week 2 + parallelism-ladder | Existing, deepened |
| PyTorch internals, custom operators | Week 8 | New |
| OSS contribution | Weeks 6–7 (vLLM) | New |
| torch.compile / XLA | Week 8 (partial) | Partial |
| Production / cluster scale | — | **Not coverable by projects — feeder-role gap** |

---

## Target tiering (honest calibration)

**Apply now / in-progress:**
- Google, Software Engineer GPU Performance (Sunnyvale) — meets minimum quals today; Nanda referral path live
- Sarvam AI, ML Engineer (Training Infra) — explicit early-career exception path; strong fit once Weeks 1–7 artifacts exist

**Apply opportunistically (rolling, low cost, low odds):**
- Anthropic, GPU Performance Engineer — staff-level bar, real target for 2027–2028 via feeder role
- OpenAI, Inference (AMD GPU Enablement) — role-type fits well (systems integration), but scale gap unclosed; AMD/HIP thread deliberately skipped in this plan
- Baseten, GPU Kernel Engineer — same tier as Anthropic/OpenAI

**Sequencing logic:** feeder tier (Google, Sarvam) now → production-scale experience 2027–2028 → staff-tier labs (Anthropic, OpenAI, Baseten) become realistic, likely via a warm path from sustained public OSS/kernel work rather than a cold application.

---

## Open threads / decisions pending
- Sarvam compensation: unverified from public sources (multiple low-reliability results found); treat as "find out when it's real," not a modeled expectation
- Whether to fold Sarvam into active warm-path outreach now vs. wait for Week 1–7 artifacts
- HIP/AMD thread deliberately dropped from this version of the plan (see conversation history if revisiting)