# Transformers and Famous Architectures

## Original Transformer (Vaswani et al., 2017 — "Attention Is All You Need")

### Architecture Overview
Encoder-Decoder architecture for sequence-to-sequence tasks (machine translation).

**Encoder**: 6 identical layers, each with:
1. Multi-Head Self-Attention
2. Feed-Forward Network (FFN)
(+ residual connections and Layer Normalization after each sub-layer)

**Decoder**: 6 identical layers, each with:
1. Masked Multi-Head Self-Attention (causal; prevents attending to future tokens)
2. Cross-Attention (attends to encoder outputs)
3. Feed-Forward Network

### Self-Attention Mechanism
Q = X * W_Q, K = X * W_K, V = X * W_V

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

- **Q*K^T**: compatibility matrix (n x n); each token scores against all others
- **sqrt(d_k)**: prevents softmax saturation when d_k is large
- **Softmax**: converts scores to probability distribution
- **Weighted sum of V**: output is weighted blend of values

**Complexity**: O(n^2 * d) in time and space — quadratic in sequence length.

### Multi-Head Attention
h independent attention heads, each with dimension d_k = d_model / h.
Concatenate outputs; project with W_O.

MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O

Each head can attend to different representation subspaces / positions.

### Positional Encoding (Sinusoidal)
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Added to embeddings. Allows model to learn position-relative patterns.

### Feed-Forward Network
FFN(x) = ReLU(x * W_1 + b_1) * W_2 + b_2

Applied identically to each position (position-wise). Dimension expansion: d_model → 4*d_model → d_model.

### Layer Normalization
Applied after each sub-layer (Post-LN: add → norm; Pre-LN: norm → sub-layer → add). Pre-LN more stable for deep transformers.

---

## BERT (Bidirectional Encoder Representations from Transformers, 2018)

### Architecture
Encoder-only transformer. Bidirectional: attends to full context (both left and right).

### Pre-training Objectives
1. **Masked Language Modeling (MLM)**: randomly mask 15% tokens (80% [MASK], 10% random word, 10% unchanged); predict them. Bidirectional context.
2. **Next Sentence Prediction (NSP)**: classify whether sentence B follows sentence A. (Later found less useful; removed in RoBERTa.)

### Input Format
[CLS] sentence A [SEP] sentence B [SEP]
- [CLS] token: pooled representation used for classification
- [SEP]: separator token

### Variants and Fine-tuning
- **BERT-base**: 12 layers, 768 hidden, 12 heads, 110M params
- **BERT-large**: 24 layers, 1024 hidden, 16 heads, 340M params
- Fine-tune for: classification (linear on [CLS]), NER (linear on each token), QA (span prediction)

### RoBERTa
Improved BERT: more data (160GB vs 16GB), longer training, larger batches, dynamic masking, removed NSP. Significantly stronger than BERT.

### XLM-RoBERTa (Your Toxicity Project)
Multilingual RoBERTa. 100 languages, 2.5TB CommonCrawl data, 250K sentencepiece vocabulary. 560M params (large variant). Used in your multilingual toxicity classification.

---

## GPT Series (Generative Pre-trained Transformers)

### GPT-1 (2018)
Decoder-only transformer. Causal (left-to-right) self-attention. Pretrained with CLM (next token prediction) on BooksCorpus. Fine-tuned for downstream tasks.

### GPT-2 (2019)
Scale: 1.5B params. Same architecture; no fine-tuning; zero-shot task transfer via prompting. Demonstrated emergent capabilities.

### GPT-3 (2020)
175B params. In-context learning via few-shot prompting — no gradient updates at all. Strong across diverse tasks with just examples in context.

### GPT-3.5 / InstructGPT
RLHF alignment: SFT on demonstrations + reward model trained on human preferences + PPO optimization. Dramatically improves instruction following.

### GPT-4 (2023)
Multimodal (text + image). Mixture-of-Experts architecture (rumored). Strongest reasoning and instruction following to date.

### Architecture Details (GPT-series)
- Decoder-only (causal attention mask; each token attends only to preceding tokens)
- Pre-LayerNorm for training stability
- Learned positional embeddings (not sinusoidal)
- No encoder; no cross-attention
- Vocabulary: 50K BPE tokens (GPT-2); 100K tiktoken (GPT-4)

---

## LLaMA Series (Meta)

### LLaMA 1 (2023)
Open-weight models: 7B, 13B, 30B, 65B params. Trained on public data (CommonCrawl, GitHub, Wikipedia, Books, ArXiv, StackExchange). Outperforms GPT-3 despite smaller scale due to longer training.

### LLaMA 2 (2023)
7B, 13B, 34B, 70B. RLHF-aligned (LLaMA 2-Chat). Grouped Query Attention (GQA) in 34B/70B. 4096 context window.

### LLaMA 3.1 (Your Fine-tuning Project)
8B, 70B, 405B. 128K context window. Tiktoken tokenizer with 128K vocabulary. Multilingual. Instruction-tuned variant: Llama-3.1-8B-Instruct.

**Your project**: Fine-tuned Llama 3.1 8B-Instruct using Unsloth + Liger Kernels (4-bit QLoRA), exported Q4_K_M GGUF via llama.cpp, pushed to HuggingFace.

### Architectural Innovations in LLaMA vs GPT

#### RoPE (Rotary Positional Embedding)
Instead of adding position embedding to Q and K, rotate Q and K by position-dependent angle in complex space.

RoPE(q, m) = q * e^(im*theta)  (complex rotation by position m)

Benefits:
- Relative positions encoded implicitly in dot product
- Extrapolates to unseen sequence lengths better than learned absolute PE
- No additional parameters

Used in: LLaMA, GPT-NeoX, Mistral, Falcon, Qwen.

#### RMSNorm (Root Mean Square Normalization)
Simpler than LayerNorm: normalize by RMS, no mean centering, no bias.
RMSNorm(x) = x / RMS(x) * gamma

Faster than LayerNorm; empirically similar quality.

#### SwiGLU Activation (FFN variant)
FFN(x) = (SiLU(x * W_1) ⊙ (x * W_2)) * W_3

Gated linear unit with SiLU. 8/3 expansion factor (vs 4 in original FFN). Stronger than ReLU FFN.

Used in: LLaMA, PaLM, GPT-4 (rumored).

#### Grouped Query Attention (GQA)
- **Multi-Head Attention (MHA)**: h Q heads, h K heads, h V heads
- **Multi-Query Attention (MQA)**: h Q heads, 1 K head, 1 V head; less memory; less quality
- **Grouped Query Attention (GQA)**: h Q heads, g K/V heads (g < h); balance between MHA and MQA; used in LLaMA 2 70B, LLaMA 3

GQA reduces KV cache memory during autoregressive decoding. Critical for long sequences.

#### KV Cache
During autoregressive generation, cache K and V tensors for all past tokens to avoid recomputation.
Memory: 2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per_element
At 100K tokens, KV cache can be GBs. GQA/MQA reduces this.

---

## T5 (Text-To-Text Transfer Transformer, 2020)

### Architecture
Encoder-decoder (full Transformer). Unified framework: every NLP task is text-to-text.

- Translation: "translate English to German: The house is wonderful." → "Das Haus ist wunderbar."
- Summarization: "summarize: [article]" → "[summary]"
- Classification: "sst2 sentence: This movie is great." → "positive"

### Pre-training
Span corruption: mask random spans; predict concatenated masked spans. More efficient than token-level masking.

### Variants
- T5-small (60M) to T5-11B
- FLAN-T5: instruction-tuned; strong few-shot performance

---

## Claude Architecture (Anthropic)

### Constitutional AI
1. SFT on helpful demonstrations
2. Critique + revision via constitutional principles (RL from AI Feedback — RLAIF)
3. PPO with AI-generated preference labels instead of purely human labels
Scales better than pure RLHF; reduces human labeling bottleneck.

### Context Window
Claude 3.5: 200K tokens. Uses sliding window attention + memory mechanisms. Flash Attention for efficiency.

---

## Mistral and Mixtral

### Mistral 7B
Sliding Window Attention (SWA): each token attends to window of W previous tokens (W=4096); efficient for long sequences.
GQA: faster decoding. Strong 7B model; outperforms LLaMA 2 13B.

### Mixtral 8x7B (Mixture of Experts)
8 expert FFN networks per layer; router selects top-2 experts per token.
Effective params: 47B total; active: ~13B per token. Similar throughput to 13B dense but quality closer to 70B.

---

## Architecture Comparison Table

| Model | Type | Params | Key Innovation | Use Case |
|---|---|---|---|---|
| BERT | Encoder | 110M-340M | MLM bidirectional | Understanding, classification |
| XLM-R | Encoder | 125M-560M | Multilingual MLM | Cross-lingual tasks |
| GPT-2/3 | Decoder | 117M-175B | CLM at scale | Generation |
| T5 | Enc-Dec | 60M-11B | Unified text-to-text | All NLP tasks |
| LLaMA 3.1 | Decoder | 8B-405B | RoPE, GQA, SwiGLU | Open-source generation |
| Mistral | Decoder | 7B | SWA, GQA | Efficient long context |
| Mixtral | Decoder (MoE) | 8x7B | Mixture of Experts | High quality, efficient |
| Claude 3.5 | Decoder | Unknown | Constitutional AI | Safe, helpful assistant |

---

## Efficient Attention Variants

### Multi-Query Attention (MQA)
All query heads share single K/V. Faster decoding; smaller KV cache. Minor quality degradation.

### ALiBi (Attention with Linear Biases)
Add a linear bias to attention scores based on relative position: score = Q*K^T - m * |i-j|.
No positional embeddings needed. Extrapolates to longer sequences than seen in training.

---

> **See also:** [14_Distributed_Training_and_Inference.md](14_Distributed_Training_and_Inference.md) for Flash Attention (v1/v2), which provides O(N) memory and 2-4x speedup for attention computation.

---

## Interview Key Points

- **Why decoder-only for generation?** Causal masking allows autoregressive generation; encoder-decoder adds complexity without benefit for pure generation; one model handles both encoding and generation.
- **Why is RoPE better than learned PE?** Encodes relative positions; extrapolates to longer sequences; no extra parameters; applied directly to Q and K.
- **What is KV cache?** Cached K, V tensors for all past tokens during autoregressive generation. Avoids O(n^2) recomputation. GQA/MQA reduces cache memory.
- **Encoder vs Decoder attention difference?** Encoder: bidirectional (attend to all tokens); Decoder: causal (attend only to past tokens via mask).
- **Why is MoE efficient?** Sparse activation: only 2 of 8 experts active per token. Total params large, active params small. Better quality per FLOP.
- **How does BERT differ from GPT?** BERT: bidirectional encoder; MLM; better for understanding. GPT: causal decoder; CLM; better for generation. BERT cannot do autoregressive generation.
- **What is temperature in decoding?** Scales logits before softmax: logit_scaled = logit / T. Low T → peaked distribution → deterministic. High T → flat → diverse/creative.
