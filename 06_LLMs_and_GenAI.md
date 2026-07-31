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

## Scaling Laws

### Chinchilla Scaling Laws (Hoffmann et al., 2022)
- **Key finding**: compute-optimal training requires tokens ≈ 20 × parameters
- A 10B param model should be trained on ~200B tokens for compute-optimal performance
- **Power-law relationship**: `L(N, D) ≈ A/N^α + B/D^β + E` where N = params, D = data, L = loss
- LLaMA 1 intentionally "over-trained" (more tokens than Chinchilla-optimal) to get better inference-time performance at smaller sizes

### Emergent Capabilities
- Abilities that appear abruptly at certain model scales (e.g., few-shot reasoning, chain-of-thought)
- Debate: some argue emergence is a mirage of metric choice (Wei et al. vs Schaeffer et al.)
- Practical implication: can't always predict which capabilities a larger model will gain

### Scaling Implications
- Training compute scales as ~6ND FLOPs (N params, D tokens)
- Doubling model quality roughly requires 10× compute
- Inference cost scales linearly with N — training is a one-time cost, inference is ongoing

---

## Pre-training Data

### Data Quality > Quantity
- **FineWeb** (HuggingFace): carefully filtered Common Crawl, outperforms larger unfiltered datasets
- **DCLM**: data-centric approach to pre-training curation
- Quality filters: perplexity filtering, deduplication, toxicity removal, language detection

### Deduplication
- **Exact dedup**: hash-based, removes identical documents
- **Near-dedup (MinHash + LSH)**: detects near-identical content, critical for avoiding memorization
- Impact: 3-5% of training data is typically exact duplicates; removing them improves generalization

### Data Mixing
- Ratio of code vs web vs books vs scientific text affects downstream capabilities
- More code → better reasoning. More multilingual → better cross-lingual transfer.
- Tokenization efficiency varies: English ≈ 1.3 tokens/word; CJK languages ≈ 2-3 tokens/word

---

## Reasoning Models and Test-Time Compute

### Chain-of-Thought at Inference
- Models generate explicit reasoning steps before answering
- **Test-time compute scaling**: spending more compute at inference (longer chains, multiple samples) improves accuracy
- Best-of-N sampling: generate N responses, pick best via reward model or self-consistency

### o1/o3-Style Reasoning
- Long internal thinking chains (may be hidden from user)
- Uses search during generation (MCTS-like exploration of reasoning paths)
- Trained to "think before answering" via RL on verifiable tasks (math, code)

### DeepSeek R1 / GRPO
- **GRPO (Group Relative Policy Optimization)**: RL without a separate reward model
  - Sample multiple outputs for each prompt
  - Score them (e.g., correctness for math)
  - Compute advantage relative to group mean
  - Update policy to increase probability of above-average outputs
- **Cold-start RL**: train reasoning from base model without SFT stage
- Key insight: RL alone can induce chain-of-thought reasoning behavior

### Other RLHF Alternatives
- **DPO (Direct Preference Optimization)**: implicit reward model from preference pairs. Simpler than PPO.
- **ORPO**: combines SFT and alignment in a single stage via odds ratio penalty
- **SimPO**: reference-free DPO variant using average log probability as reward
- **Process Reward Models (PRMs)**: reward each reasoning step, not just the final answer. Better for math/code.

---

## LLM Security and Adversarial Attacks

### Prompt Injection
- **Direct**: user crafts input that overrides system instructions
- **Indirect**: injected via retrieved documents (RAG poisoning) or tool outputs
- **Mitigations**: input sanitization, output filtering, separate system/user contexts, instruction hierarchy

### Jailbreaking Patterns
- Role-playing ("pretend you're an unrestricted AI")
- Encoding tricks (Base64, Caesar cipher)
- Many-shot jailbreaking (providing many examples of desired behavior)
- Gradient-based attacks (GCG: adversarial suffixes found via optimization)

### Defenses
- Constitutional AI: train model to self-critique and revise
- Llama Guard / NeMo Guardrails: input/output classifiers
- System prompt protection: instruction hierarchy, delimiters

---

## LLM Guardrails — Building and Using

### What Are Guardrails?
Programmatic checks applied **before** (input guardrails) and **after** (output guardrails) LLM calls to enforce safety, quality, and compliance policies. They sit as middleware in the LLM pipeline.

### Architecture Pattern
```
User Input → Input Guardrails → LLM → Output Guardrails → Response
                 ↓ (block/rewrite)           ↓ (block/rewrite/fallback)
```

### Input Guardrails
| Guardrail | How It Works | When to Use |
|-----------|-------------|-------------|
| **Topic restriction** | Classifier (fine-tuned BERT or LLM-as-judge) rejects off-topic prompts | Customer support bots, domain-specific assistants |
| **PII detection/redaction** | Regex + NER (spaCy, Presidio) to detect names, SSNs, emails; mask before sending to LLM | Healthcare, finance, any regulated domain |
| **Prompt injection detection** | Classifier trained on injection examples; heuristic checks for instruction overrides | Any user-facing LLM application |
| **Token budget check** | Count tokens (tiktoken / tokenizer); reject or truncate if over budget | Cost control, latency control |
| **Rate limiting / abuse detection** | Track request frequency, repeated jailbreak attempts per user | Production deployments |

### Output Guardrails
| Guardrail | How It Works | When to Use |
|-----------|-------------|-------------|
| **Hallucination / grounding check** | NLI model (entailment score) verifies output is supported by context | RAG systems, factual QA |
| **Content filtering** | Toxicity classifier (Perspective API, custom model); block or flag harmful output | All user-facing apps |
| **Structured output validation** | JSON schema validation, regex matching, Pydantic parsing | Tool calling, API responses |
| **Citation verification** | Check that cited sources actually exist and support the claim | Research, medical, legal |
| **PII leakage check** | Re-scan output for PII that may have been generated (not just echoed) | Regulated domains |

### Guardrail Frameworks
- **NVIDIA NeMo Guardrails**: define rails in Colang (domain-specific language); supports topical, safety, and dialog rails; programmable flows
- **Guardrails AI (guardrails-ai)**: Python validators wrapping LLM calls; built-in validators for toxicity, PII, JSON, hallucination; composable via `Guard` object
- **AWS Bedrock Guardrails**: managed service; content filters, denied topics, word/phrase filters, PII redaction, grounding checks with configurable thresholds
- **Llama Guard**: fine-tuned LLaMA model that classifies input/output into safety categories; works as an input/output classifier in pipeline

### Building Custom Guardrails — Key Design Decisions
1. **Classifier vs LLM-as-judge**: classifier (BERT-based) is faster/cheaper (~5ms); LLM-as-judge is more flexible but adds latency (~500ms+) and cost
2. **Blocking vs flagging**: hard block (reject response) vs soft flag (log + return with warning); depends on risk tolerance
3. **Synchronous vs async**: sync guardrails add latency to every call; async guardrails run in background and flag retroactively
4. **Cascading**: chain multiple guardrails; fail-fast on first violation vs run all and aggregate
5. **Fallback responses**: predefined safe responses when guardrail triggers ("I can't help with that" vs redirect to human)

### Guardrails in Production — Gotchas
- Guardrail latency adds up: 3 guardrails × 50ms each = 150ms overhead per request
- False positives erode user trust; tune thresholds on real traffic, not synthetic data
- Adversarial users iterate against your guardrails — they need updating over time
- Guardrails are defense-in-depth, NOT a replacement for alignment training (RLHF/CAI)

### Interview Questions
- **Why not just rely on RLHF for safety?** RLHF shapes model tendencies but can't guarantee behavior; guardrails provide deterministic enforcement at inference time
- **Guardrails AI vs NeMo Guardrails?** Guardrails AI: Python-native validators, composable, good for structured output; NeMo: Colang-based dialog flows, better for conversational rail design
- **How do you handle guardrail latency?** Run input/output guardrails in parallel where independent; use fast classifiers over LLM-as-judge; cache frequent decisions; async for non-critical checks

---

## Prompt Token Optimization — Keeping Usage Small

### Why It Matters
- **Cost**: API pricing is per-token (input tokens often cheaper than output, but still add up at scale)
- **Latency**: time-to-first-token scales with prompt length (prefill phase); longer prompts = slower responses
- **Context window**: finite budget shared between prompt and generation; bloated prompts leave less room for output
- **Lost-in-the-middle**: models attend worse to information in the middle of long contexts

### Techniques to Reduce Prompt Tokens

#### 1. Prompt Engineering for Conciseness
- Remove filler words, redundant instructions, and verbose examples
- Use structured formats (tables, key:value) instead of prose
- Consolidate repeated instructions into a single system prompt
- "Show, don't tell": one clear example often replaces paragraphs of instructions
- **Measured impact**: well-edited prompts are often 30-50% shorter with same performance

#### 2. Few-Shot → Zero-Shot (When Possible)
- Each few-shot example costs tokens; evaluate if zero-shot + clear instructions performs comparably
- If few-shot is needed: use the minimum number of examples; pick short, representative ones
- **Dynamic few-shot**: retrieve relevant examples per query (not fixed set) using embedding similarity

#### 3. Prompt Caching (Provider-Level)
- **Anthropic**: cache_control breakpoints; cached prefill tokens cost 90% less; cache has 5-min TTL
- **OpenAI**: automatic prefix caching; 50% discount on cached input tokens
- **Design for caching**: put static content (system prompt, instructions, schema) first; dynamic content (user query) last; cache hits require exact prefix match
- **Key insight**: prompt structure matters — reorder so cacheable content is at the front

#### 4. Context Compression
- **LLMLingua / LongLLMLingua**: use small LM to score token importance; drop low-information tokens; 3-20x compression with <5% quality loss
- **Summarize-then-query**: summarize long documents before including in prompt; trade one-time summarization cost for repeated savings
- **Summary memory for multi-turn**: compress old conversation turns into a summary; keep only recent turns verbatim

#### 5. RAG Instead of Stuffing
- Retrieve only relevant chunks (top-k) instead of loading entire documents into context
- Smaller chunk sizes (256-512 tokens) with good retrieval > large chunks with noise
- Re-ranking (cross-encoder) after retrieval filters out irrelevant chunks before they hit the prompt
- **Token savings**: 100K document → 3 relevant chunks × 512 tokens = 1.5K tokens

#### 6. Structured Prompt Design
- **System/user separation**: put reusable instructions in system prompt (cached); per-request data in user prompt
- **Schema-driven prompts**: define output schema concisely (JSON schema) instead of describing format in prose
- **Tool/function calling**: let the API handle tool descriptions natively rather than embedding them in prompt text
- **Avoid restating the obvious**: don't tell the model what it already knows from context

#### 7. Token-Aware Engineering
- Use shorter variable/field names in structured prompts (within reason)
- Prefer common words (fewer tokens) over rare synonyms: "use" (1 token) vs "utilize" (1-2 tokens)
- Be aware: numbers, code, and non-English text tokenize less efficiently
- Whitespace and formatting tokens add up: `\n\n\n` costs tokens; minimize unnecessary blank lines

### Cost Optimization Beyond Prompt Shrinking
- **Model routing**: use cheaper/smaller models for simple queries; route complex ones to larger models
- **Batch API**: 50% discount for non-real-time workloads (Anthropic, OpenAI)
- **Output token limits**: set `max_tokens` appropriately; don't request 4096 when you need 200
- **Streaming + early stopping**: stop generation when you detect the answer is complete

### Interview Questions
- **How would you reduce LLM API costs by 10x?** Prompt caching (structure for cache hits), model routing (cheap model for easy queries), RAG (retrieve not stuff), batch API for offline, compress/summarize long context
- **Prompt caching vs semantic caching?** Prompt caching: exact prefix match at provider level, automatic discount; semantic caching: application-level, cache responses for semantically similar queries, requires embedding + similarity threshold, risk of stale/wrong cached answers
- **When does context compression hurt?** When the compressed prompt drops tokens critical to the task; compression works best on redundant/verbose content; always evaluate on your specific task after compression

---

## Interview Key Points

- **Why does temperature affect creativity?** Higher T flattens softmax → equal probability tokens → more diverse sampling
- **Why use KL penalty in RLHF?** Prevents reward hacking; model shouldn't drift far from SFT behavior
- **LoRA vs full fine-tuning trade-off?** LoRA: parameter efficient, less catastrophic forgetting, faster; full fine-tuning: higher ceiling, needs more compute
- **What are Chinchilla scaling laws?** Compute-optimal training requires ~20 tokens per parameter. Training a small model on more data can beat a larger under-trained model.
- **What is GRPO and how does it differ from PPO?** GRPO doesn't need a separate reward model — it uses group-relative advantages from sampled outputs. Simpler pipeline.
- **When does RAG fail?** When retrieved chunks are irrelevant, context window overflows, embedding model mismatches the domain, or the reader ignores retrieved content.
- **Process vs outcome reward models?** PRMs reward each step (better for multi-step reasoning); ORMs only reward the final answer (simpler but less signal).
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
