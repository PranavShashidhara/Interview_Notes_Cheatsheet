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

> **See also:** [13_Transformers_and_Architectures.md](13_Transformers_and_Architectures.md) for detailed coverage of transformer architecture, self-attention, positional encoding (RoPE, ALiBi, sinusoidal), and model variants.

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

> **See also:** [07_RAG_and_Embeddings.md](07_RAG_and_Embeddings.md) for comprehensive coverage of RAG pipelines, chunking strategies, embedding models, vector databases, hybrid retrieval, re-ranking, embeddings (Word2Vec, GloVe, SBERT, CLIP, contrastive learning), and advanced RAG techniques.

---

## LLM Architecture Variants

### Encoder-Only (BERT-style)
Bidirectional attention; [CLS] token for classification; MLM pre-training.
Best for: classification, NER, semantic similarity, extractive QA.

### Decoder-Only (GPT-style)
Causal (left-to-right) attention; CLM pre-training.
Best for: text generation, instruction following, agentic tasks.
Examples: GPT-4, LLaMA, Claude, Mistral.

### Encoder-Decoder (T5/BART-style)
Encoder: bidirectional; Decoder: causal with cross-attention to encoder.
Best for: translation, summarization, seq2seq tasks.
Examples: T5, BART, mT5.

### Mixture of Experts (MoE)
Replace dense FFN with N expert FFNs; router selects top-k experts per token.
- Only k of N experts activated per token → sparse computation
- Same parameter count as dense model but lower FLOPS per forward pass
- **Load balancing loss**: auxiliary loss to prevent all tokens routing to same expert
- Examples: Mixtral 8x7B, GPT-4 (rumored), Gemini 1.5
- Challenge: all experts must fit in memory; load imbalance

### State Space Models (SSMs) / Mamba
Alternative to attention for sequence modeling.
- **S4/Mamba**: structured state space; O(N) inference vs O(N^2) attention
- Mamba: selective SSM; input-dependent state update (unlike RNN with fixed dynamics)
- Better than transformers on very long sequences; no explicit attention matrix
- Hybrid: Mamba + attention layers (Jamba by AI21)

### GQA (Grouped Query Attention) and MQA
- **MQA (Multi-Query Attention)**: all heads share single K,V; faster inference, less memory for KV cache
- **GQA (Grouped Query Attention)**: groups of heads share K,V; balance between MHA and MQA
- Used in: LLaMA3, Mistral, Gemma

---

---

> **See also:** [14_Distributed_Training_and_Inference.md](14_Distributed_Training_and_Inference.md) for KV cache, Flash Attention, speculative decoding, continuous batching, PagedAttention (vLLM), and distributed inference optimization.

---

## Hallucination and Grounding

### Types of Hallucinations
- **Factual hallucination**: states incorrect facts with confidence (names, dates, citations)
- **Faithfulness hallucination**: summary/answer contradicts source document
- **Intrinsic**: contradicts source; **Extrinsic**: adds info not in source

### Why Hallucinations Happen
- LLM learns to produce plausible-sounding text, not verified facts
- Knowledge encoded in weights may be outdated or incorrect
- Model may interpolate/extrapolate incorrectly from training patterns
- Decoding at high temperature increases hallucination rate

### Mitigation Strategies
- **RAG**: ground generation in retrieved documents; ask model to cite sources
- **Chain-of-thought**: explicit reasoning steps expose errors
- **Self-consistency**: sample multiple outputs, take majority vote
- **Fact-checking prompts**: "Only state facts you are certain of"
- **Guardrails**: NLI-based entailment check (does output follow from context?)
- **Low temperature**: reduces randomness but doesn't eliminate hallucination

### Grounding Evaluation
- **ROUGE-L / BERTScore** against source docs
- **NLI entailment score**: classify (premise=context, hypothesis=generated claim) as entail/neutral/contradict
- **FActScoring**: decompose into atomic facts, verify each independently

---

## Constitutional AI and RLAIF

### Constitutional AI (Anthropic)
Reduce reliance on human labelers for harmful content detection.
1. **SL-CAI**: model critiques and revises its own responses using a "constitution" (list of principles)
2. **RL-CAI**: train reward model using AI-generated preference data (not human), then PPO

### RLAIF (Reinforcement Learning from AI Feedback)
Replace human raters with LLM-generated preference labels.
- LLM-judge rates output pairs → preference dataset → reward model → PPO
- Scales better than RLHF; quality depends on judge model capability
- Risk: reward model inherits judge's biases

### Constitutional Principles Examples
- "Choose the response that is least likely to contain harmful content"
- "Choose the response that is most helpful and honest"
Applied at critique + revision stage and at preference labeling stage.

---

## Multimodal LLMs

### Vision-Language Models (VLMs)
Combine vision encoder + language model:
- **CLIP-based**: encode image with CLIP ViT, project to LLM token space (LLaVA, InstructBLIP)
- **Native multimodal**: interleaved image+text tokens from scratch (Flamingo, Gemini)

### LLaVA Architecture
1. ViT (Vision Transformer) encodes image → patch embeddings
2. MLP projection maps visual embeddings to LLM input space
3. LLM (LLaMA/Mistral) processes interleaved text+image tokens

### CLIP (Contrastive Language-Image Pretraining)
- Dual encoder: image encoder (ViT) + text encoder (Transformer)
- Trained on 400M (image, text) pairs with InfoNCE contrastive loss
- Zero-shot classification: embed image, embed class names, find nearest class

### Audio / Speech Models
- **Whisper**: encoder-decoder transformer for ASR; trained on 680K hours
- **VALL-E**: TTS via audio codec tokens; voice cloning from 3-second sample
- **GPT-4o**: native audio input/output; end-to-end without intermediate ASR

### Key Challenges in Multimodal
- Modality alignment: map visual/audio features to text token space
- Hallucination worsens with image input (model invents image details)
- Long video: thousands of frames exceed context window

---

> **See also:** [14_Distributed_Training_and_Inference.md](14_Distributed_Training_and_Inference.md) for data/tensor/pipeline parallelism, ZeRO optimization, mixed precision training, gradient checkpointing, and distributed training libraries.

---

## Context Length Handling

### Positional Encoding for Long Context
- **RoPE + YaRN**: extend RoPE to longer contexts by adjusting rotation frequencies; LLaMA uses this
- **ALiBi**: linear position bias; generalizes to longer sequences than seen in training without fine-tuning
- **NTK-aware scaling**: scale base of RoPE to interpolate positions

### Sliding Window Attention (Mistral)
Each token attends to only last W tokens (window size). Rolling buffer KV cache.
- Efficient O(N·W) attention; may miss very long-range dependencies
- Combined with full attention on some layers for global context

### Context Compression
- **LLMLingua**: remove tokens the LLM can predict easily; compress prompt by 3-20x
- **RAG instead of long-context**: retrieve relevant chunks rather than loading entire document
- **Summary memory**: summarize old context, append summary + recent turns

### Lost in the Middle Problem
LLMs perform worse on information in the middle of long contexts than at the start or end. Mitigation: put important info at beginning or end; re-rank retrieved chunks accordingly.

---

> **See also:** [14_Distributed_Training_and_Inference.md](14_Distributed_Training_and_Inference.md) for Flash Attention and the quadratic bottleneck in attention computation.

---

## Interview Key Points

- **Why does temperature affect creativity?** Higher T flattens softmax → equal probability tokens → more diverse sampling
- **Why use KL penalty in RLHF?** Prevents reward hacking; model shouldn't drift far from SFT behavior
- **LoRA vs full fine-tuning trade-off?** LoRA: parameter efficient, less catastrophic forgetting, faster; full fine-tuning: higher ceiling, needs more compute
- **What is context window?** Max tokens the model can attend to at once; LLaMA3: 8K, Claude 3.5: 200K; limited by quadratic attention cost
- **How does instruction tuning differ from pre-training?** Pre-training: predict next token on massive corpus; instruction tuning: supervised on (instruction, good response) pairs to align behavior
- **What is flash attention?** Fused attention kernel that avoids materializing full NxN attention matrix; O(N) memory instead of O(N^2); enables longer context
- **Why use hybrid retrieval over dense-only?** BM25 handles exact keyword matches better; dense handles semantic similarity; together they cover more failure modes
- **What's the difference between bi-encoder and cross-encoder?** Bi-encoder: encode query and doc independently (fast, scalable); cross-encoder: concatenate and score jointly (accurate, slow); use bi-encoder to retrieve, cross-encoder to rerank
- **Why does MoE use less compute despite more parameters?** Only top-k experts activate per token; rest are skipped; total FLOPs = k/N × dense equivalent
- **What is the KV cache and why does it matter?** Stores K,V tensors for past tokens so they don't need recomputation each autoregressive step; crucial for inference efficiency; memory grows linearly with sequence length
- **Hallucination vs factual error?** Hallucination specifically means model generates plausible-sounding but false info; factual errors are a subset; hallucination also includes faithfulness failures (contradicts source)
- **What is speculative decoding?** Draft model proposes K tokens; target model verifies in one pass; lossless 2-3x speedup; requires draft/target model alignment
- **RAG vs fine-tuning: when to use each?** RAG: knowledge is external/dynamic/large; need citations; fine-tuning: task format/style/behavior change; domain-specific phrasing; not for injecting new facts
- **What is GQA and why does it help?** Groups of query heads share K,V heads; reduces KV cache memory proportionally; speeds inference without large quality loss vs full MHA
