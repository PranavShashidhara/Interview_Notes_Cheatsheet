# Distributed Training and Inference

## Why Distributed Training?

Models too large for single GPU memory. Training too slow on single GPU. Scale to improve quality (scaling laws: loss ~ params^(-alpha), compute^(-beta)).

**Memory breakdown for a 7B model in FP32**:
- Parameters: 7B * 4 bytes = 28 GB
- Gradients: 28 GB
- Optimizer states (Adam: 2 copies): 56 GB
- Activations (batch-dependent): varies
- Total: 100+ GB → requires multiple A100 80GB GPUs even with mixed precision

---

## Types of Parallelism

### Data Parallelism (DP)
Replicate full model on each GPU. Split batch across GPUs. Each GPU computes gradients on its mini-batch. Aggregate gradients (AllReduce). Update all replicas identically.

**AllReduce**: sum gradients across all GPUs; broadcast result back. Ring-AllReduce: each GPU sends/receives from neighbors in ring; bandwidth-efficient O(2*(N-1)/N * size).

**When to use**: model fits on single GPU; want to scale throughput with more data.

**PyTorch DDP (DistributedDataParallel)**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")  # NCCL for GPU
model = DDP(model.to(local_rank), device_ids=[local_rank])

# Training loop: identical to single GPU
# DDP handles gradient sync automatically via hooks
```

**Launch**:
```bash
torchrun --nproc_per_node=8 train.py
```

### Model Parallelism (MP)

#### Tensor Parallelism (TP)
Split individual layers across GPUs. For matrix multiply Y = X * W:
- Column-parallel: split W by columns; each GPU computes partial output; AllGather for full output
- Row-parallel: split W by rows; each GPU computes partial Y from its X slice; AllReduce

Used for: attention heads (split by head), FFN layers.
Communication: AllReduce/AllGather per layer. High communication overhead — works best within node (NVLink).

**Used in Megatron-LM** for training GPT-3 scale models.

#### Pipeline Parallelism (PP)
Split layers across GPUs. GPU 0 has layers 0-8, GPU 1 has layers 9-16, etc.

**Naive pipeline**: GPU 0 runs forward → send to GPU 1 → ... → backward passes in reverse. GPUs mostly idle (bubble).

**Micro-batch pipelining (GPipe, PipeDream)**: split batch into micro-batches; overlap forward of micro-batch m+1 with backward of micro-batch m. Reduces bubble from O(stages) to O(stages/micro-batches).

Communication: only between adjacent pipeline stages. Works well across nodes.

### 3D Parallelism (Megatron-LM + DeepSpeed)
Combine DP + TP + PP for largest models.
- TP within a node (fast NVLink)
- PP across nodes (slower InfiniBand, fewer transfers)
- DP across groups of nodes

Used to train GPT-3, LLaMA, Megatron-Turing NLG (530B).

---

## ZeRO (Zero Redundancy Optimizer) — DeepSpeed

Standard DP replicates optimizer states, gradients, and parameters across all GPUs — redundant.

### ZeRO Stages
**Stage 1**: Partition optimizer states across GPUs. Each GPU maintains optimizer state for its parameter shard. AllGather for parameter update. 4x memory reduction for optimizer states.

**Stage 2**: Partition optimizer states + gradients. Each GPU only stores gradients for its parameter shard. ReduceScatter instead of AllReduce. 8x memory reduction.

**Stage 3**: Partition optimizer states + gradients + parameters. Parameters gathered on-demand during forward/backward. Full model can exceed total GPU memory. Used in your context: QLoRA effectively does parameter-efficient version of this.

**ZeRO-Offload**: offload optimizer states (and gradients) to CPU RAM. Train large models on fewer GPUs.

**ZeRO-Infinity**: offload to NVMe SSD. Extreme memory savings.

```python
# DeepSpeed config (ds_config.json)
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "bf16": {"enabled": true}
}
```

---

## Gradient Accumulation

Simulate larger effective batch size with limited GPU memory:
```python
accumulation_steps = 8
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()  # accumulates gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch = per_gpu_batch * accumulation_steps * n_gpus.

---

## Mixed Precision Training (Your Llama Fine-tuning)

### FP16 Training
- Forward and backward in FP16 (half precision)
- Master copy of weights in FP32
- Loss scaling to prevent FP16 underflow (gradients < 1e-8 round to 0)
- 2x memory savings; 2-8x faster on GPUs with Tensor Cores (A100, V100, H100)

### BF16 Training (Brain Float 16)
- Same exponent range as FP32 (no loss scaling needed)
- Lower precision mantissa than FP16
- More numerically stable; preferred for LLM training
- Requires Ampere+ GPUs (A100, RTX 3090+)

### Training Precision in Your QLoRA Project
- Base model: 4-bit NF4 quantization
- LoRA adapters: BF16
- Optimizer: full BF16 (or paged AdamW in QLoRA)
- Gradient checkpointing: recompute activations on backward instead of storing

---

## Gradient Checkpointing

Trade compute for memory: don't store intermediate activations during forward pass. Recompute them during backward when needed.

Memory: O(sqrt(n)) instead of O(n) for n layers.
Compute overhead: ~33% extra forward pass work.

```python
model.gradient_checkpointing_enable()
# Or in PyTorch:
from torch.utils.checkpoint import checkpoint
output = checkpoint(layer, input)
```

Essential for fine-tuning large models or long sequences.

---

## Distributed Training Libraries

### PyTorch DDP
Best for data parallelism. Tight integration with PyTorch. Gradient sync via NCCL ring-AllReduce. Each process = one GPU.

### DeepSpeed
Microsoft. ZeRO optimizer. Works with DDP. Adds ZeRO-1/2/3, offloading, pipeline parallelism. Integration with Hugging Face Trainer.

### Megatron-LM
NVIDIA. Tensor + pipeline parallelism for largest LLMs. Used internally for training at scale.

### FSDP (Fully Sharded Data Parallel — PyTorch native)
PyTorch's native ZeRO Stage 3 equivalent. Shards parameters, gradients, optimizer states across GPUs. More integrated with PyTorch than DeepSpeed.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)
```

### Accelerate (HuggingFace)
Abstraction layer over DDP, DeepSpeed, FSDP, MPS (Apple Silicon). Same training code runs on 1 GPU, multi-GPU, TPU.

```python
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
# Training loop unchanged; Accelerate handles distribution
```

Used implicitly in Unsloth/Trainer for your LLaMA fine-tuning.

### Unsloth (Your Project)
- 2x speedup over standard LoRA via custom triton kernels
- 70% memory reduction
- Fused RoPE, RMSNorm, cross-entropy kernels
- Supports QLoRA on single GPU; effectively ZeRO via quantization
- Compatible with Hugging Face ecosystem

---

## Distributed Inference

### Challenges
- Single GPU too slow for large models (high latency)
- Large models don't fit on single GPU
- High throughput needed for many concurrent requests

### Tensor Parallelism for Inference
Same as training: split model across GPUs. Each request uses all GPUs simultaneously. Low latency at cost of high GPU count per request.

### Model Sharding with Pipeline Parallelism
Different layers on different GPUs. Better throughput when many requests can be batched across stages.

### KV Cache Management
During autoregressive generation, KV cache grows with sequence length. Multiple concurrent requests share GPU memory.

**Paged Attention (vLLM)**: manages KV cache like virtual memory. Pages of KV cache allocated/freed dynamically. Allows high batch sizes without OOM. 10-24x throughput improvement over naive implementation.

### vLLM
Production LLM inference engine. Features:
- Paged Attention for efficient KV cache
- Continuous batching: new requests added to running batch at any step (no waiting for current batch to finish)
- Tensor parallelism across multiple GPUs
- AWQ/GPTQ quantization support
- OpenAI-compatible API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2)
outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=512))
```

### llama.cpp (Your MediAssist Offline Stack)
CPU/GPU inference for GGUF models. Key features:
- GGUF format: efficient quantized storage
- Q4_K_M, Q5_K_M, Q8_0: different quality/speed tradeoffs
- CPU inference with BLAS optimization
- GPU offload: partial layers to GPU, rest on CPU
- Metal support (Apple Silicon)
- Used for your BioGPT GGUF offline mode

### TensorRT (NVIDIA)
Compiles model to optimized GPU kernel. Supports quantization (INT8, FP16). Layer fusion, kernel auto-tuning. Significant speedup (2-10x) over PyTorch for fixed-shape inference.

### ONNX Runtime (Your MLOps Project)
Cross-platform inference. Supports multiple execution providers: CUDA, TensorRT, OpenVINO, CPU. Graph optimizations: operator fusion, constant folding.

### Speculative Decoding
Use small draft model to propose K tokens; large model verifies all in one forward pass.
- Accept tokens where large model agrees; reject and resample on first disagreement
- Lossless (same distribution as target model); 2-3x speedup in practice
- Requires aligned draft+target model pair
- Particularly useful for latency-sensitive applications

---

## Communication Backends

| Backend | Use Case | Notes |
|---|---|---|
| NCCL | GPU-to-GPU | NVIDIA Collective Communications Library; fastest for GPUs |
| Gloo | CPU fallback | Works on CPU; slower |
| MPI | HPC clusters | Traditional HPC communication |
| InfiniBand | Cross-node GPU | High bandwidth, low latency interconnect |
| NVLink | Intra-node GPU | Faster than PCIe; connects GPUs within one server |

---

## NCCL (NVIDIA Collective Communications Library) — Deep Dive

### What NCCL Does
NCCL provides optimized implementations of collective operations (AllReduce, AllGather, ReduceScatter, Broadcast, Reduce) for multi-GPU and multi-node GPU communication. It is the default backend for `torch.distributed` on GPUs.

### Key Collective Operations
- **AllReduce**: every GPU ends up with the sum (or other reduction) of all GPUs' data. Used in DDP gradient sync.
- **ReduceScatter**: reduce data across GPUs, scatter result so each GPU gets 1/N of the reduced output. Used in ZeRO Stage 2, FSDP.
- **AllGather**: each GPU broadcasts its shard; every GPU ends up with full data. Used in ZeRO Stage 3 / FSDP for parameter gathering before forward.
- **All-to-All**: each GPU sends distinct data to every other GPU. Used in MoE expert routing.

### How NCCL Chooses Algorithms
NCCL auto-selects the best algorithm based on message size, topology, and GPU count:

| Algorithm | Best For | How It Works |
|---|---|---|
| **Ring** | Medium-large messages (>256KB) | GPUs arranged in logical ring; each sends/receives to/from neighbors; bandwidth-optimal O(2*(N-1)/N * data) |
| **Tree** | Small-medium messages | Binary tree reduction then broadcast; latency-optimal O(2*log(N)) steps |
| **Direct** (NVLS) | Small messages, NVSwitch | Single-hop via NVSwitch; lowest latency |

### NCCL Topology Detection
NCCL probes the hardware topology at `init_process_group()`:
- Discovers NVLink/NVSwitch connections within a node
- Detects PCIe topology (which GPUs share a PCIe switch)
- Identifies InfiniBand/RoCE NICs and their GPU affinity (NUMA locality)
- Builds optimal ring/tree paths to maximize bandwidth and minimize PCIe hops

**NCCL_TOPO_DUMP_FILE**: set this env var to dump the detected topology for debugging.

### NCCL Environment Variables (Tuning & Debugging)
```bash
# Debugging
NCCL_DEBUG=INFO              # Print topology, ring order, algorithm selection
NCCL_DEBUG_SUBSYS=ALL        # Detailed subsystem logging (INIT, COLL, NET, etc.)

# Performance tuning
NCCL_ALGO=Ring               # Force algorithm (Ring, Tree, CollNet)
NCCL_PROTO=Simple            # Protocol (Simple, LL, LL128) — LL128 best for small msgs
NCCL_MIN_NCHANNELS=4         # Minimum parallel channels (more = more bandwidth, more SM usage)
NCCL_MAX_NCHANNELS=16        # Maximum parallel channels
NCCL_BUFFSIZE=4194304        # Per-channel buffer size (default 4MB)
NCCL_SOCKET_IFNAME=ib0       # Force network interface for multi-NIC nodes
NCCL_IB_HCA=mlx5_0           # Force specific InfiniBand HCA
NCCL_NET_GDR_LEVEL=5         # GPUDirect RDMA level (0=disabled, 5=max)
NCCL_P2P_DISABLE=0           # Disable peer-to-peer (useful for debugging)
NCCL_SHM_DISABLE=0           # Disable shared memory transport
NCCL_CROSS_NIC=1             # Allow cross-NIC communication (multi-rail)
```

### Common NCCL Failure Modes
- **Hang at init**: mismatched `world_size` or firewall blocking NCCL bootstrap port
- **Hang during collective**: one rank crashed or diverged; NCCL has no timeout by default → set `NCCL_TIMEOUT` or `TORCH_NCCL_BLOCKING_WAIT=1`
- **NCCL error "unhandled system error"**: GPU memory OOM during NCCL buffer allocation; reduce `NCCL_BUFFSIZE` or free memory
- **Slow AllReduce**: NCCL falling back to PCIe instead of NVLink/IB; check `NCCL_DEBUG=INFO` for topology

---

## RDMA and Network Transports

### The Network Stack Problem
Traditional TCP/IP networking involves:
1. Application copies data to kernel buffer (CPU involved)
2. Kernel processes TCP/IP headers
3. Data sent to NIC via DMA
4. Reverse on receiving side
5. **Latency: ~50-100μs per message; CPU overhead: significant**

For distributed training where AllReduce happens every backward step, this overhead is devastating.

### RDMA (Remote Direct Memory Access)
RDMA allows one machine to directly read/write another machine's memory **without involving the remote CPU or OS kernel**.

**Key properties**:
- **Zero-copy**: data moves directly between application buffers and NIC (no kernel buffer)
- **Kernel bypass**: no OS involvement in data path after initial setup
- **CPU offload**: NIC handles protocol processing; CPU free for compute
- **Latency: ~1-2μs** (vs 50-100μs TCP); **Throughput: line-rate** (200-400 Gb/s)

### RDMA Implementations

| Transport | Full Name | Network | Notes |
|---|---|---|---|
| **InfiniBand** | Native IB | InfiniBand fabric | Gold standard for HPC/AI. Dedicated switches. Lossless by design. |
| **RoCE v2** | RDMA over Converged Ethernet | Standard Ethernet + PFC | RDMA semantics over Ethernet. Cheaper switches. Needs lossless config (PFC/ECN). |
| **iWARP** | Internet Wide Area RDMA Protocol | Standard Ethernet | RDMA over TCP. Works over lossy networks. Higher latency than RoCE. Rare in AI clusters. |

### InfiniBand Speeds

| Generation | Per-lane | 4x (typical) | Used In |
|---|---|---|---|
| EDR | 25 Gb/s | 100 Gb/s | Older clusters |
| HDR | 50 Gb/s | 200 Gb/s | A100 DGX systems |
| NDR | 100 Gb/s | 400 Gb/s | H100 DGX/HGX |
| XDR | 200 Gb/s | 800 Gb/s | GB200 NVL72 |

### GPUDirect RDMA (GDR)
Without GDR: GPU → CPU memory (PCIe) → NIC (PCIe) → network → NIC → CPU → GPU
With GDR: GPU → NIC (direct PCIe peer) → network → NIC → GPU

**Eliminates the CPU bounce buffer entirely**. Requires:
- NVIDIA GPU + Mellanox/NVIDIA NIC on same PCIe root complex
- `nvidia-peermem` kernel module loaded
- NCCL: `NCCL_NET_GDR_LEVEL=5`

### GPUDirect RDMA + NVLink Topology (DGX-style)
```
Node 1                              Node 2
┌─────────────────────┐             ┌─────────────────────┐
│ GPU0 ←NVLink→ GPU1  │             │ GPU0 ←NVLink→ GPU1  │
│  ↕           ↕      │             │  ↕           ↕      │
│ GPU2 ←NVLink→ GPU3  │             │ GPU2 ←NVLink→ GPU3  │
│  ↕           ↕      │             │  ↕           ↕      │
│ GPU4 ←NVLink→ GPU5  │             │ GPU4 ←NVLink→ GPU5  │
│  ↕           ↕      │             │  ↕           ↕      │
│ GPU6 ←NVLink→ GPU7  │             │ GPU6 ←NVLink→ GPU7  │
│  ↕↕          ↕↕     │             │  ↕↕          ↕↕     │
│ NIC0        NIC1    │── IB/RoCE ──│ NIC0        NIC1    │
└─────────────────────┘             └─────────────────────┘
```

### Multi-Rail Networking
Modern GPU nodes have **multiple NICs** (4-8 NICs for 8 GPUs). Each GPU is NUMA-affinitized to a specific NIC:
- GPU 0-1 → NIC 0, GPU 2-3 → NIC 1, GPU 4-5 → NIC 2, GPU 6-7 → NIC 3
- NCCL uses all NICs in parallel (**multi-rail**) for aggregate bandwidth
- DGX H100: 8× ConnectX-7 NICs = 8 × 400 Gb/s = 3.2 Tb/s aggregate

---

## The Multi-Node Scaling Cliff

### Why Crossing Node Boundaries is Hard

**Intra-node** (NVLink/NVSwitch):
- Bandwidth: 600-900 GB/s (NVSwitch on DGX A100/H100)
- Latency: ~1-5μs
- All GPUs connected to all GPUs (full-bisection via NVSwitch)

**Inter-node** (InfiniBand/RoCE):
- Bandwidth: 25-100 GB/s per NIC (200-800 Gb/s)
- Latency: ~5-10μs (RDMA) or ~50-100μs (TCP)
- **10-30× less bandwidth** than intra-node NVLink

This cliff means collective operations that were "free" within a node suddenly dominate training time across nodes.

### Quantifying the Impact

**AllReduce for a 7B model gradient sync (14 GB in FP16)**:

| Scenario | Bandwidth | Time | Notes |
|---|---|---|---|
| 8 GPUs, 1 node, NVSwitch | 600 GB/s effective | ~47 ms | Ring-AllReduce, NVLink |
| 16 GPUs, 2 nodes, 1× HDR IB | 25 GB/s cross-node | ~1.1 s | Cross-node link is bottleneck |
| 16 GPUs, 2 nodes, 8× HDR IB | 200 GB/s cross-node | ~140 ms | Multi-rail helps significantly |
| 64 GPUs, 8 nodes, 8× NDR IB | 400 GB/s cross-node | ~70 ms | But more nodes = more hops |

### Scaling Problems by GPU Count

**1-8 GPUs (single node)**: Typically "just works." NVLink/NVSwitch handles everything. DDP or FSDP, minimal tuning.

**9-16 GPUs (2 nodes)**: **The first scaling cliff.**
- First exposure to inter-node communication
- AllReduce suddenly bottlenecked by slowest link (IB/RoCE)
- Common issues:
  - GDR not enabled → data bouncing through CPU
  - Wrong NIC affinity → PCIe contention
  - TCP fallback instead of RDMA → 10× slower
  - Firewall/security group blocking NCCL ports
  - Clock skew causing NCCL timeouts

**17-64 GPUs (3-8 nodes)**:
- Network congestion becomes real (shared IB fabric)
- Fat-tree topology: oversubscription at spine switches
- Stragglers: one slow node delays AllReduce for all
- Need to start thinking about communication-computation overlap

**64-512 GPUs (8-64 nodes)**:
- Must use 3D parallelism (DP + TP + PP)
- TP within node (NVLink), PP across nodes (IB), DP across groups
- Network topology awareness critical for placement
- Checkpointing overhead becomes significant

**512+ GPUs (64+ nodes)**:
- Failure rate: with 1000 GPUs, expect ~1 GPU failure per day
- Need elastic training / automatic recovery
- Network partitions, NIC flaps, switch failures
- Congestion control tuning (ECN, PFC for RoCE)
- Custom network topologies (rail-optimized, dragonfly)

### The Overlap Strategy (Hiding Communication Behind Compute)
Key optimization: overlap AllReduce with backward pass computation.

DDP already does this — hooks on each parameter trigger AllReduce as soon as that parameter's gradient is ready (not waiting for full backward):
```
Backward:  [layer N grad] [layer N-1 grad] [layer N-2 grad] ...
AllReduce: .............. [layer N AR     ] [layer N-1 AR   ] ...
```
This is why DDP uses **bucketed AllReduce** — groups small parameter gradients into ~25MB buckets for efficiency.

**When overlap breaks down**:
- If communication > computation per layer, you can't hide it
- At multi-node scale, cross-node AllReduce takes longer → harder to overlap
- Solution: reduce communication (gradient compression, local SGD) or increase compute per step (larger batch)

### Gradient Compression Techniques
Used when network bandwidth is the bottleneck:

**Top-K sparsification**: only communicate top K% of gradient values. Rest accumulated locally ("error feedback"). Can compress 100-1000×. Quality impact depends on K.

**PowerSGD**: low-rank approximation of gradient matrices. Communicate rank-R approximation instead of full gradient. Compression ratio = d/(2R) for d-dimensional gradient. Minimal quality impact at R=4.

**Local SGD / Post-Local SGD**: each node trains independently for H steps, then averages models. Reduces communication frequency by H×. Works well with large batch sizes. Used in Meta's LLaMA training.

---

## Network Debugging and Diagnostics

### Diagnosing Multi-Node Issues

**Step 1: Verify RDMA is working**
```bash
# Check IB device status
ibstat                    # Shows HCA state (should be "Active")
ibstatus                  # Shows link speed, width, state

# Test RDMA connectivity
ib_write_bw -d mlx5_0    # Server side
ib_write_bw -d mlx5_0 <server_ip>  # Client side → shows bandwidth

# Expected: ~24 GB/s for HDR, ~48 GB/s for NDR per port
```

**Step 2: Verify GPUDirect RDMA**
```bash
# Check nvidia-peermem module
lsmod | grep nvidia_peermem   # Should show nvidia_peermem

# If missing:
modprobe nvidia-peermem
```

**Step 3: NCCL diagnostics**
```bash
# Run NCCL test
# Build: git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make MPI=1
mpirun -np 16 --hostfile hosts ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1

# Output shows bus bandwidth per message size
# Expected: ~300 GB/s intra-node (NVSwitch), ~20-40 GB/s inter-node (IB)
```

**Step 4: Check NIC-GPU affinity**
```bash
nvidia-smi topo -m
# Shows NVLink, PCIe, SYS connections between GPUs and NICs
# "SYS" means crossing NUMA boundary → performance penalty
# GPU should show "NODE" or "PIX" connection to its IB NIC
```

### Common Multi-Node Failure Patterns

| Symptom | Likely Cause | Fix |
|---|---|---|
| Training hangs at first AllReduce | NCCL can't establish connections | Check firewall, security groups; set `NCCL_SOCKET_IFNAME` |
| Training hangs randomly after hours | Straggler or OOM on one rank; NCCL timeout | Set `TORCH_NCCL_BLOCKING_WAIT=1`, `NCCL_TIMEOUT=1800` |
| 10× slower than expected inter-node | TCP fallback, no GDR | Check `NCCL_DEBUG=INFO` for transport; load `nvidia-peermem` |
| Good single-node, 50% scaling to 2 nodes | AllReduce bottlenecked on single NIC | Enable multi-rail: `NCCL_CROSS_NIC=1`, verify all NICs active |
| Intermittent NaN/divergence at scale | Bit-flip from faulty NIC/cable; non-deterministic reduction order | ECC check; use `NCCL_ALGO=Ring` for determinism |
| RoCE performance collapse under load | PFC storm / deadlock | Tune ECN thresholds; check switch PFC config |

---

## Network Topology for Large Clusters

### Fat-Tree (Clos) Topology
Standard data center topology. Leaf switches → spine switches. Full bisection bandwidth (non-blocking) if 1:1 oversubscription. **Most common for AI clusters.**

### Rail-Optimized Topology
Each GPU index connects to its own dedicated network "rail":
- All GPU₀s across all nodes share Rail₀ (NIC₀ → Switch₀)
- All GPU₁s across all nodes share Rail₁ (NIC₁ → Switch₁)
- Reduces cross-rail traffic; matches NCCL's ring pattern
- Used in Meta's Grand Teton (H100) clusters

### NVIDIA DGX SuperPOD
- 32 DGX H100 nodes = 256 GPUs
- Leaf-spine InfiniBand fabric (NDR 400 Gb/s)
- Non-blocking full bisection bandwidth
- SHARP (Scalable Hierarchical Aggregation and Reduction Protocol): offloads AllReduce to switch ASICs

### SHARP (In-Network Computing)
Instead of ring-AllReduce across GPUs, SHARP performs reduction **inside the InfiniBand switches**:
- GPUs send data to leaf switch → switch reduces → sends to spine → spine reduces → result propagated back
- Cuts AllReduce latency in half for large clusters
- Requires Mellanox Quantum switches + NCCL with SHARP support
- `NCCL_COLLNET_ENABLE=1` to activate

---

## Interview Key Points — Networking & Scaling

- **Why does crossing 2 nodes hurt so much?** Intra-node NVLink: 600-900 GB/s. Inter-node IB: 25-100 GB/s per NIC. 10-30× bandwidth gap. AllReduce time dominated by slowest link. Mitigation: multi-rail NICs, overlap communication with compute, use TP within node and PP across nodes.
- **What is RDMA and why does it matter for training?** RDMA = direct memory-to-memory transfer across machines without CPU/kernel involvement. 1-2μs latency vs 50-100μs TCP. Critical because distributed training does thousands of collectives per minute — even 50μs overhead per operation is devastating at scale.
- **What is GPUDirect RDMA?** Extends RDMA to GPU memory — NIC reads/writes GPU HBM directly via PCIe peer access, skipping CPU memory entirely. Without it, data takes GPU→CPU→NIC→network→NIC→CPU→GPU path. With it: GPU→NIC→network→NIC→GPU.
- **How does NCCL handle multi-node communication?** Auto-detects topology (NVLink, PCIe, IB), builds optimal ring/tree. Uses GPUDirect RDMA for cross-node. Selects algorithm (ring for large messages, tree for small). Overlaps communication with backward pass via gradient bucketing.
- **RoCE vs InfiniBand?** IB: purpose-built lossless fabric, dedicated switches, guaranteed delivery, higher cost. RoCE: RDMA over standard Ethernet, cheaper, but needs PFC (Priority Flow Control) for losslessness — PFC storms and deadlocks are common failure modes. IB is preferred for training; RoCE common in cloud (cheaper).
- **What is a PFC storm and why does RoCE suffer from it?** PFC pauses traffic to prevent packet loss (required for RDMA). If one slow receiver triggers PFC, it propagates upstream — pause frames cascade across the fabric ("PFC storm"), collapsing throughput for the entire cluster. IB handles congestion natively without this issue.
- **How do you scale beyond 1000 GPUs?** 3D parallelism (TP + PP + DP). Elastic training for fault tolerance. Rail-optimized or fat-tree topology. SHARP for in-network reduction. Local SGD to reduce communication frequency. Checkpoint every N minutes; expect daily hardware failures.

---

## Scaling Laws (Chinchilla)

Kaplan et al. 2020 / Chinchilla 2022: optimal compute allocation.

Chinchilla finding: for a given compute budget C (FLOPs):
- Optimal model size: N ~ sqrt(C)
- Optimal training tokens: D ~ sqrt(C)
- N and D should scale roughly equally

Earlier models (GPT-3) were over-parameterized and under-trained. LLaMA trains smaller models on much more data → competitive quality at smaller model size (important for inference efficiency).

---

## HPC (Your Resume: HPC Skills)

### SLURM Job Submission
```bash
#!/bin/bash
#SBATCH --job-name=llm_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

module load cuda/12.1
srun torchrun --nnodes=4 --nproc_per_node=8 train.py
```

### Key HPC Concepts
- **Node**: single physical machine with multiple GPUs
- **Task/Rank**: single process; typically one per GPU
- **InfiniBand**: high-speed inter-node interconnect (200 Gb/s HDR)
- **Shared filesystem (Lustre)**: parallel filesystem for training data and checkpoints
- **Checkpoint**: save model state periodically; resume after node failure

---

## FlashAttention

### Why Attention is Memory-Bound
- Standard attention materializes full N×N matrix in HBM → O(N²) memory
- HBM bandwidth (2 TB/s on A100) is the bottleneck, not compute (312 TFLOPS)
- Attention is memory-bound because arithmetic intensity is low

### FlashAttention Algorithm
- **Tiling**: split Q, K, V into blocks that fit in SRAM (~20MB on A100)
- Compute attention block-by-block using **online softmax** (log-sum-exp accumulation)
- Never materialize full N×N matrix → **O(N) memory** instead of O(N²)
- **Speed**: 2-4× faster due to reduced HBM reads (SRAM is 10-100× faster)

### FlashAttention v2
- Better work partitioning across warps/thread blocks
- Reduces non-matmul operations
- ~2× faster than v1

---

## Prefill vs Decode Phase

### Prefill (Prompt Processing)
- Process all input tokens in parallel
- **Compute-bound**: matrix multiplications on full sequence
- Determines **time-to-first-token (TTFT)**

### Decode (Token Generation)
- Generate one token at a time, autoregressively
- **Memory-bandwidth-bound**: each step reads full KV cache from HBM for a single token
- Determines **tokens-per-second (TPS)**

### Optimization Implications
| Phase | Bottleneck | Optimization |
|---|---|---|
| Prefill | Compute | Tensor parallelism, larger batch |
| Decode | Memory bandwidth | KV cache quantization, GQA/MQA, speculative decoding |

### Chunked Prefill
- Split long prompts into chunks, interleave with decode requests
- Prevents long prompts from blocking decode for other requests
- Used in vLLM's continuous batching

---

## Quantization Comparison

| Method | Approach | Quality | Speed | Best For |
|---|---|---|---|---|
| **GPTQ** | Layer-by-layer, gradient-based | Good at INT4 | Fast inference | GPU deployment |
| **AWQ** | Activation-aware, preserves salient channels | Better than GPTQ | Fast inference | GPU deployment |
| **NF4** | Normalized float, information-theoretically optimal for normal weights | Good | Used with QLoRA | Fine-tuning |
| **GGUF** | Block-wise quantization, CPU-friendly | Varies (Q4-Q8) | CPU inference | Local/edge deployment |
| **SmoothQuant** | Smooth activation outliers into weights | Good at INT8 | W8A8 | Weight+activation quant |

### When to Use What
- **INT8 (W8A8)**: minimal quality loss, 2× speedup, good default
- **INT4 (W4A16)**: noticeable quality loss on small models, fine for 7B+, 3-4× memory savings
- **Always benchmark**: run eval suite on quantized model before deploying

---

## MoE (Mixture of Experts) Parallelism

### How MoE Works
- Replace dense FFN with multiple **expert** FFNs (e.g., 8 experts)
- **Router** selects top-k experts per token (typically k=2)
- Only k/N of FFN parameters activated per token → faster inference

### Expert Parallelism (EP)
- Distribute experts across GPUs: GPU₁ has experts 1-4, GPU₂ has experts 5-8
- Tokens routed to the GPU holding their selected expert → **all-to-all communication**
- Combine EP with TP for the attention layers

### Load Balancing
- Auxiliary loss to encourage uniform expert utilization
- Without it: some experts get all tokens ("winner-take-all"), others unused
- Mixtral uses top-2 routing with load balancing loss

---

## Back-of-Envelope GPU Math

### Memory Estimation
- Model parameters: 1B params × 2 bytes (FP16) = 2 GB
- KV cache per token: 2 × n_layers × d_model × 2 bytes
- For 70B model: ~140 GB in FP16 → needs 2× A100 80GB or 1× H100

### Throughput Estimation
- A100: 312 TFLOPS (FP16), 2 TB/s HBM bandwidth
- Decode throughput ≈ HBM_bandwidth / (2 × model_size_bytes)
- 70B model: 2TB/s ÷ (2 × 140GB) ≈ 7 tokens/s per GPU

### Scaling for QPS
- Target: 100 req/s, avg 200 output tokens, 70B model
- Tokens/s needed: 100 × 200 = 20,000 tokens/s
- Per GPU: ~7 tokens/s → need ~2,857 GPUs? No — use continuous batching
- With batching (batch=64): ~200 tokens/s per GPU → ~100 GPUs
- With INT4 quantization: 2× throughput → ~50 GPUs

---

## Interview Key Points

- **Data vs model parallelism trade-off?** Data parallelism: full model replicated; simple; limited by single GPU memory. Model parallelism: larger models; more communication overhead; complex.
- **What is AllReduce?** Collective operation: sum (or other reduction) across all processes; broadcast result to all. Ring-AllReduce is bandwidth-optimal: O(2*(N-1)/N * data) total communication.
- **How does ZeRO reduce memory?** Partition redundant optimizer states, gradients, parameters across GPUs. Each GPU only stores 1/N of each. Parameters gathered on-demand. Eliminates replication without adding communication vs DDP.
- **Design: serve a 70B model at 100 req/s?** INT4 quantization + tensor parallelism across 2 GPUs per replica + continuous batching (vLLM) + ~25 replicas behind load balancer.
- **Prefill vs decode bottleneck?** Prefill is compute-bound (parallelize with TP). Decode is memory-bandwidth-bound (optimize with GQA, KV cache quantization, speculative decoding).
- **FlashAttention: same O(n²) FLOPs but faster — why?** Reduces HBM reads by tiling to SRAM. Bottleneck is memory bandwidth, not compute.
- **Why is pipeline parallelism useful for cross-node training?** TP requires frequent all-reduce (needs fast NVLink); PP only communicates activations at stage boundaries (less frequent; tolerates slower InfiniBand).
- **What is continuous batching?** Traditional batching: wait for full batch; GPU idle when some sequences finish early. Continuous batching: insert new sequences into ongoing batch when slots free. Higher GPU utilization.
- **Trade-off between quantization quality levels?** Q4_K_M: 4-bit; smallest; fastest; minor quality loss. Q8_0: 8-bit; near-lossless; 2x larger. F16: full half-precision; best quality; largest. Choose based on memory constraints and quality requirements.
