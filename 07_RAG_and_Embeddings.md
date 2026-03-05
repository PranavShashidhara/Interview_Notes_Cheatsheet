# RAG and Embeddings

## What is RAG?

Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative model:
1. Retrieve relevant documents from a knowledge base given the query
2. Augment the LLM prompt with retrieved context
3. Generate a grounded, factual response

**Why RAG?** Avoids hallucination, reduces need for fine-tuning, keeps knowledge up-to-date, provides provenance.

---

## RAG Pipeline Architecture

```
Query
  |
  v
Query Encoder (embedding model)
  |
  v
Vector Store (ANN search) --> Retrieved chunks
  |
  v
Prompt Construction (query + context)
  |
  v
LLM (generation)
  |
  v
Response
```

---

## Embeddings

### What are Embeddings?
Dense vector representations of text in a continuous semantic space. Semantically similar texts have vectors with high cosine similarity.

### How They Are Trained
- **Contrastive learning**: pull positive pairs (semantically similar) together, push negative pairs apart
- **Objective**: InfoNCE loss or triplet loss
- **SBERT (Sentence-BERT)**: Siamese BERT network; encode sentence pairs, compute cosine similarity; trained on NLI and STS datasets

### all-MiniLM-L6-v2 (Your Projects: MediAssist + RAG Pipeline)
- Architecture: MiniLM (distilled from larger BERT)
- 6 layers, 22M parameters, 384-dimensional embeddings
- Very fast; good quality on semantic similarity
- Used to embed PubMedQA + MedQuad into Pinecone
- Pairing: encode query → cosine search in Pinecone → return top-k chunks

### Embedding Model Selection
| Model | Dims | Notes |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | Fast, general purpose |
| all-mpnet-base-v2 | 768 | Better quality, slower |
| text-embedding-ada-002 | 1536 | OpenAI; strong baseline |
| E5-large | 1024 | Strong retrieval-focused |
| BGE-M3 | 1024 | Multilingual, multi-granularity |
| BioSentBERT | 768 | Biomedical domain (for MedQA) |

---

## Vector Databases

### Pinecone (Your MediAssist Project)
Fully managed vector database. Key concepts:
- **Index**: collection of vectors with same dimension
- **Namespace**: partitioning within index
- **Metadata**: attach JSON to each vector for filtering
- **Upsert**: insert or update vector by ID
- **Query**: return top-k nearest neighbors with scores

```python
# Typical Pinecone workflow
index.upsert(vectors=[(id, embedding, metadata)])
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
```

### Other Vector Stores
| Store | Type | Notes |
|---|---|---|
| Pinecone | Managed SaaS | Zero ops, scales easily |
| Weaviate | Self-hosted / Cloud | Built-in hybrid search |
| Chroma | Lightweight | Dev/local use |
| FAISS | Library (Facebook) | In-memory; very fast; no persistence |
| pgvector | Postgres extension | SQL + vectors in same DB |
| Qdrant | Self-hosted | Filtering + payload |

---

## Chunking Strategies

### Fixed-size Chunking
Split by token/character count (e.g., 512 tokens). Simple. Can cut mid-sentence.

### Sentence-based Chunking
Split at sentence boundaries. Preserves semantic units. Variable chunk size.

### Recursive Character Splitting (LangChain default)
Try splitting on paragraphs → sentences → words → characters. Respects document structure.

### Semantic Chunking
Use embedding similarity to detect topic shifts; split at semantic boundaries. Higher quality, more expensive.

### Overlap
Include 10-20% overlap between chunks to prevent losing context at boundaries.

---

## Retrieval Methods

### Dense Retrieval
Encode query and documents with same encoder; cosine similarity search in embedding space. Good for semantic similarity. Requires vector store.

### Sparse Retrieval (BM25 / TF-IDF)
**TF-IDF**: score = TF(term, doc) * IDF(term) where IDF = log(N/df)
**BM25**: improved TF-IDF with document length normalization and term saturation:
score(D,Q) = sum over terms [IDF(t) * f(t,D)*(k1+1) / (f(t,D) + k1*(1-b+b*|D|/avgdl))]

Good for keyword matching; does not require GPU; handles rare/exact terms well.

### Hybrid Retrieval (Your MTech Ventures Work: 50K+ docs)
Combines dense (semantic) + sparse (keyword) retrieval. Merge scores via Reciprocal Rank Fusion (RRF) or linear combination.

RRF score(d) = sum_r 1 / (k + rank_r(d)), k=60 typically

Best of both worlds: semantic generalization + exact keyword matching.

### Reranking
After initial retrieval (top-k, k=50-100), use a cross-encoder to rerank and return top-n (n=5-10).

**Cross-encoder**: takes (query, document) pair as input; produces single relevance score. More accurate than bi-encoder but slower (no precomputed embeddings).

Models: ms-marco-MiniLM-L-6-v2, Cohere Rerank, Jina Reranker.

---

## Advanced RAG Patterns

### Naive RAG
Chunk → embed → store → retrieve → generate. Simple baseline.

### Query Expansion / Rewriting
Rephrase query using LLM before retrieval. Handles vague or short queries.
- HyDE (Hypothetical Document Embeddings): generate hypothetical answer, embed it, retrieve by its embedding

### Multi-query Retrieval
Generate multiple query variants; retrieve for each; deduplicate results. Reduces sensitivity to query phrasing.

### Parent-Child Chunking
Store fine-grained child chunks for retrieval; pass parent (larger) chunk to LLM for more context.

### Self-RAG
LLM decides when to retrieve, what to retrieve, and whether retrieved docs are useful. Generates special tokens (RETRIEVE, ISREL, ISSUP).

### Iterative RAG / FLARE
Generate response; identify uncertain spans; retrieve for those specific spans; regenerate.

### Contextual Compression
After retrieval, use LLM to extract only the relevant portions of each chunk before adding to context.

### Graph RAG
Build knowledge graph from documents; retrieve subgraphs; enables multi-hop reasoning across entities.

---

## Approximate Nearest Neighbor (ANN) Search

### Why ANN?
Exact kNN is O(n*d) per query. ANN trades small accuracy loss for massive speedup.

### HNSW (Hierarchical Navigable Small World)
- Build a layered graph; top layers coarse, bottom layer dense
- Navigate from coarse to fine during search
- O(log N) search complexity
- Used by Pinecone, Qdrant, Weaviate by default

### IVF (Inverted File Index)
- Cluster vectors into Voronoi cells (k-means)
- At query time: find nprobe nearest centroids; search only those cells
- Used in FAISS

### FAISS
Facebook AI Similarity Search. Library for efficient vector search:
- IndexFlatL2: exact L2 search
- IndexIVFFlat: ANN via IVF
- IndexIVFPQ: IVF + Product Quantization (compressed vectors)
- IndexHNSWFlat: HNSW-based

### Similarity Metrics
- **Cosine similarity**: dot product of unit vectors; angle between vectors; range [-1,1]; most common for text
- **Dot product**: magnitude + direction; used when embeddings are normalized
- **Euclidean (L2)**: straight-line distance; sensitive to magnitude
- **Manhattan (L1)**: sum of absolute differences

---

## RAG for Domain-Specific QA (Your MediAssist)

### Datasets
- **PubMedQA**: biomedical QA from PubMed abstracts; pqa_labeled (1K) + pqa_unlabeled (61K)
- **MedQuad**: medical QA from NIH websites; ~47K pairs

### Domain Adaptation Strategies
1. Use domain-specific embedding model (BioSentBERT, PubMedBERT)
2. Fine-tune embeddings on domain corpus
3. Use domain-specific LLM (BioGPT) for offline/air-gapped mode
4. Add metadata filters (e.g., filter by specialty, condition)

### Offline/Air-Gapped RAG (Your MediAssist)
Local pipeline:
- BioGPT GGUF via llama.cpp: local LLM inference
- faster-whisper: local STT
- Glow-TTS: local TTS
- easyOCR: local OCR
- Fully functional without internet access

---

## RAG Evaluation Metrics

### RAGAS Framework
End-to-end RAG evaluation:

| Metric | Definition | Range |
|---|---|---|
| Faithfulness | Fraction of answer claims supported by context | [0,1] |
| Answer Relevance | Similarity of answer to question | [0,1] |
| Context Precision | Fraction of retrieved chunks that are relevant | [0,1] |
| Context Recall | Fraction of reference answer info found in context | [0,1] |
| Context Entity Recall | Named entities from reference found in context | [0,1] |

### Retrieval Metrics
- **MRR (Mean Reciprocal Rank)**: 1/rank of first relevant doc; averaged over queries
- **MAP@k**: mean average precision at k
- **Recall@k**: fraction of relevant docs in top-k
- **NDCG@k**: normalized discounted cumulative gain at k

### Generation Metrics
- **EM (Exact Match)**: binary; strict
- **Token F1**: character/token overlap with reference
- **BERTScore**: semantic similarity to reference
- **LLM-as-judge**: GPT-4 or Claude scores on custom rubric

---

## LangChain RAG Implementation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

# Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="medassist")

# Retrieve and generate
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)
```

---

## Interview Key Points

- **Why is chunking important?** Chunks too large: noise dilutes relevant info; too small: missing context. Overlap prevents boundary information loss.
- **When to use BM25 vs dense retrieval?** BM25 for exact keyword/entity match (rare terms, codes, product names); dense for paraphrase/semantic similarity. Hybrid typically best.
- **What is the difference between bi-encoder and cross-encoder?** Bi-encoder encodes query and doc separately (precomputable); fast but less accurate. Cross-encoder encodes jointly; more accurate; used for reranking.
- **How do you handle long documents that exceed chunk size?** Parent-child chunking, summarize long docs, sliding window with overlap.
- **What is HyDE?** Generate a hypothetical answer to the query, embed that answer, use its embedding for retrieval. Bridges query-document vocabulary gap.
- **Why does RAG reduce hallucination?** LLM is constrained to generate from provided context; faithfulness can be checked against retrieved text; uncertainty reduces when answer is explicit in context.
