# LLMs and Generative AI

## Language Model Fundamentals

### What is a Language Model?
A probability distribution over sequences of tokens:
P(x_1, x_2, ..., x_n) = product of P(x_t | x_1, ..., x_{t-1})

Autoregressive generation: predict next token given all previous tokens.

### Tokenization
- **BPE (Byte-Pair Encoding)**: iteratively merge most frequent adjacent byte pairs; used in GPT series
- **WordPiece**: similar to BPE but uses likelihood maximization; used in BERT
- **SentencePiece**: language-agnostic, operates on raw Unicode; used in LLaMA, T5
- Typical vocabulary: 32K-128K tokens

---

## Transformer Architecture (Foundation)

### Self-Attention
Q = X * W_Q, K = X * W_K, V = X * W_V

Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V

- Q*K^T: compatibility between queries and keys
- sqrt(d_k): scaling to prevent softmax saturation in high dimensions
- Result: weighted sum of values based on attention scores

### Multi-Head Attention
Split into h heads, each with dimension d_k = d_model / h. Compute attention independently per head, concatenate, project.

Allows attending to different aspects/positions simultaneously.

### Feed-Forward Network
FFN(x) = max(0, x*W_1 + b_1)*W_2 + b_2

Applied position-wise (independently to each token). Typically 4x expansion.

### Positional Encoding
Sinusoidal (original): PE(pos, 2i) = sin(pos/10000^(2i/d_model))
RoPE (Rotary PE): rotates Q and K by position-dependent angle; used in LLaMA, GPT-NeoX.
ALiBi: adds position bias directly to attention scores; generalizes to longer sequences.

---

## Pre-training Objectives

### Causal Language Modeling (CLM)
Predict next token; left-to-right. Used in: GPT, LLaMA, Claude.
Loss = -sum log P(x_t | x_{<t})

### Masked Language Modeling (MLM)
Mask 15% of tokens; predict them. Bidirectional context. Used in: BERT.

### Prefix LM
Bidirectional attention on prefix, causal on suffix. Used in: T5, UL2.

---

## Instruction Tuning and RLHF

### Supervised Fine-Tuning (SFT)
Fine-tune pretrained LLM on curated (instruction, response) pairs. Teaches model to follow instructions.

### RLHF (Reinforcement Learning from Human Feedback)
1. **SFT**: supervised fine-tuning on demonstrations
2. **Reward Model Training**: human annotators rank model outputs; train reward model (RM) on preference pairs
3. **PPO (Proximal Policy Optimization)**: optimize policy (LLM) to maximize RM score while staying close to SFT model:
   - Objective: E[r(x,y)] - beta * KL(pi_RL || pi_SFT)
   - KL penalty prevents reward hacking / over-optimization

### DPO (Direct Preference Optimization)
Eliminates need for separate RM and RL loop. Directly optimizes on preference pairs using a closed-form objective derived from RLHF. Simpler, more stable than PPO-based RLHF.

---

## Fine-Tuning Techniques

### Full Fine-Tuning
Update all model parameters. Expensive; catastrophic forgetting risk.

### LoRA (Low-Rank Adaptation)
Freeze original weights; add low-rank decomposition: W' = W + A*B where A is d x r, B is r x k, r << min(d,k).
Only A and B trained. Typical r=4-64. Reduces trainable params by ~100x.

### QLoRA (Quantized LoRA — Your Llama 3.1 Project)
Base model in 4-bit NF4 quantization (via bitsandbytes). LoRA adapters in BF16. Enables fine-tuning 70B model on single GPU.
- **NF4 (NormalFloat4)**: quantization format optimized for normally distributed weights
- **Double quantization**: quantizes the quantization constants themselves to save further memory
- **Paged optimizers**: offload optimizer states to CPU when GPU memory full

**Your project**: Fine-tuned Llama 3.1 8B-Instruct using Unsloth + Liger Kernels with 4-bit QLoRA.

### Unsloth
Optimized LoRA implementation with custom CUDA kernels. 2x speedup, 70% memory reduction vs naive LoRA. Fuses operations, avoids materializing large intermediate tensors.

### Liger Kernels
Fused triton kernels for transformer operations (RoPE, RMSNorm, SiLU, cross-entropy). Reduces memory by chunked cross-entropy computation.

### PEFT (Parameter-Efficient Fine-Tuning) Methods
- **LoRA / QLoRA**: low-rank weight updates (most popular)
- **Prefix Tuning**: prepend trainable prefix tokens to input
- **Prompt Tuning**: learn soft prompt embeddings only
- **Adapter**: small bottleneck layers inserted into transformer blocks

---

## Quantization

### Why Quantize?
Reduce model size and inference latency. INT8/INT4 vs FP32.

### GGUF Format (Your Project: Q4_K_M)
Format for llama.cpp CPU/GPU inference.
- **Q4_K_M**: 4-bit quantization, medium quality variant using K-quants
- K-quants: quantize in groups with separate scales; higher quality than naive INT4
- Enables running large models on CPU

### Quantization Types
- **Post-Training Quantization (PTQ)**: quantize after training; fast but may lose accuracy
- **Quantization-Aware Training (QAT)**: simulate quantization during training; better accuracy
- **GPTQ**: layer-wise quantization using second-order information; good quality
- **AWQ**: activation-aware weight quantization; preserves salient weights

---

## Decoding Strategies

### Greedy
Always pick argmax token. Fast; repetitive; suboptimal.

### Beam Search
Maintain top-k sequences at each step. Score = sum of log probabilities. Better than greedy for translation.

### Sampling
Sample from probability distribution: temperature controls sharpness.

**Temperature scaling**: p_i = exp(logit_i / T) / sum(exp(logit_j / T))
- T < 1: sharper (more deterministic)
- T > 1: flatter (more random)
- T = 0: equivalent to greedy

### Top-k Sampling
Sample from top k tokens only. Cuts off long tail; k typically 50.

### Top-p (Nucleus) Sampling
Sample from smallest set of tokens whose cumulative probability >= p (typically 0.9). Adapts dynamically to distribution shape. Often better than top-k.

### Repetition Penalty
Penalize previously generated tokens to reduce repetition.

---

## Agentic LLM Systems (Your Resume)

### ReAct Pattern
Reasoning + Acting: LLM alternates between:
- **Thought**: internal reasoning
- **Action**: tool call (search, code execution, DB query)
- **Observation**: tool result
Continues until task complete or max steps.

### Tool Use / Function Calling
Structured JSON schema defines available tools. LLM generates structured function call; system executes; result returned to LLM.

### Memory Types
- **In-context**: conversation history in context window
- **External (Vector DB)**: semantic search over past interactions
- **Episodic**: specific past events
- **Semantic**: general knowledge

### Multi-Agent Systems (AutoGen — Your Resume)
Multiple agents with different roles: planner, executor, critic, etc.
- **Orchestrator**: coordinates agents
- **Tool-use agents**: execute specific tasks
- **Conversation patterns**: two-agent, group chat, nested chat

Used in your MTech Ventures work: hybrid retrieval over 50K+ docs, serving 17 analysts with 100-300 queries/week.

### Prompt Engineering Techniques
- **Zero-shot**: no examples; relies on instruction following
- **Few-shot**: provide examples in prompt; in-context learning
- **Chain-of-thought (CoT)**: "Let's think step by step"; improves multi-step reasoning
- **Tree of Thought (ToT)**: explore multiple reasoning branches
- **System prompt**: sets persona, constraints, output format

---

## Email Generator (Your Resume: 350+ emails, 10% reply rate)

### Agentic Email Pipeline
1. Retrieve prospect context from CRM/DB
2. Generate personalized email via LLM with structured prompt
3. Output structured email (JSON with subject, body, CTA)
4. Human review or auto-send

Key prompt engineering: explicit persona, output format (XML/JSON), length constraint, tone, clear CTA instruction.

---

## Evaluation Metrics for LLMs

### Intrinsic Metrics
- **Perplexity**: PP = exp(H(p,q)); lower = better language model; standard train/eval metric
- **Token accuracy**: exact token prediction rate; mainly diagnostic

### Extrinsic / Task Metrics
- **BLEU**: n-gram precision; machine translation
- **ROUGE-L**: LCS-based recall; summarization
- **BERTScore**: contextual semantic similarity
- **EM (Exact Match)**: for QA tasks; binary
- **F1 over tokens**: partial credit for QA

### Alignment / Safety Metrics
- **Helpfulness rate**: human or LLM-judge score on response quality
- **Refusal rate**: rate of inappropriate refusals
- **Toxicity score**: classifier-based; PerspectiveAPI, Detoxify
- **Factual consistency**: entailment-based; NLI model scores
- **Reply rate** (your email project: 10%): real-world engagement metric

### LLM-as-Judge
Use a capable LLM (e.g., GPT-4, Claude) to evaluate outputs on criteria like helpfulness, accuracy, conciseness, coherence. Correlates well with human evaluation. Cost-effective.

### MMLU (Massive Multitask Language Understanding)
5-shot multiple choice across 57 subjects. Standard benchmark for knowledge and reasoning.

### MT-Bench
Two-turn conversations judged by GPT-4. Tests instruction following and multi-turn coherence.

---

## AWS Bedrock (Your MediAssist / Work Projects)

### What It Is
Fully managed API service for foundation models (Claude, LLaMA, Titan, Stable Diffusion, etc.) on AWS.

### Key Features
- No need to manage model infrastructure
- Fine-tuning support via continued pre-training
- Knowledge Bases: managed RAG with S3 + vector store
- Guardrails: content filtering, PII redaction, grounding checks

### Integration Pattern (Your Projects)
Route online requests → AWS Bedrock (Claude 3.5) → response
Route offline/air-gapped requests → local BioGPT GGUF via llama.cpp → response
Fallback logic based on connectivity detection.

---

## Interview Key Points

- **Why does temperature affect creativity?** Higher T flattens softmax → equal probability tokens → more diverse sampling
- **Why use KL penalty in RLHF?** Prevents reward hacking; model shouldn't drift far from SFT behavior
- **LoRA vs full fine-tuning trade-off?** LoRA: parameter efficient, less catastrophic forgetting, faster; full fine-tuning: higher ceiling, needs more compute
- **What is context window?** Max tokens the model can attend to at once; LLaMA3: 8K, Claude 3.5: 200K; limited by quadratic attention cost
- **How does instruction tuning differ from pre-training?** Pre-training: predict next token on massive corpus; instruction tuning: supervised on (instruction, good response) pairs to align behavior
- **What is flash attention?** Fused attention kernel that avoids materializing full NxN attention matrix; O(N) memory instead of O(N^2); enables longer context
