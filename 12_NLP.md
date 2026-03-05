# NLP (Natural Language Processing)

## Text Preprocessing

### Pipeline
```
Raw Text → Cleaning → Tokenization → Normalization → Feature Extraction → Model
```

### Cleaning
- Remove HTML tags, URLs, special characters
- Lowercase (task-dependent; named entities may need case)
- Handle contractions: "don't" → "do not"
- Remove or replace emojis (or use as features for sentiment)

### Tokenization
- **Word tokenization**: split on whitespace and punctuation
- **Sentence tokenization**: split on sentence boundaries (SpaCy, NLTK)
- **Subword tokenization (BPE, WordPiece)**: handles unknown words; used in transformers

### Normalization
- **Stemming**: rule-based suffix stripping; fast but crude; "running" → "run", "studies" → "studi" (Porter Stemmer)
- **Lemmatization**: dictionary-based; returns valid base form; "studies" → "study", "better" → "good" (slower but accurate)
- **Stop word removal**: remove high-frequency low-information words (the, is, a); use task-appropriately (removes "not" in sentiment = bad)

---

## Text Representation

### Bag of Words (BoW)
Document represented as vector of word counts. Ignores order and grammar.

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)  # sparse matrix n_docs x vocab_size
```

### TF-IDF
Term Frequency — Inverse Document Frequency.

TF(t,d) = count of t in d / total terms in d
IDF(t) = log(N / df(t))  where df = number of docs containing t
TF-IDF(t,d) = TF(t,d) * IDF(t)

High TF-IDF: term is frequent in this document but rare overall → distinctive.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
X = tfidf.fit_transform(corpus)
```

### N-grams
Contiguous sequence of n tokens. Captures local context.
- Unigram: "the", "cat"
- Bigram: "the cat", "cat sat"
- Trigram: "the cat sat"
Trade-off: higher n → more context, sparser representation.

### Word Embeddings

#### Word2Vec
Predicts word from context (CBOW) or context from word (Skip-gram).
- CBOW: predict center word from surrounding context words
- Skip-gram: predict surrounding words given center word
- Trained with negative sampling (NCE)
- Result: word vectors where similar words have high cosine similarity
- Captures analogies: king - man + woman ≈ queen

#### GloVe (Global Vectors)
Matrix factorization on global word co-occurrence matrix. Combines local (Word2Vec) and global (count-based) statistics.

#### FastText
Extension of Word2Vec; represents words as sum of character n-gram embeddings. Handles rare words and morphological variations. Good for multilingual NLP.

#### Contextual Embeddings (BERT, RoBERTa, XLM-RoBERTa)
Unlike static embeddings, word vector depends on context. "bank" near "river" vs "bank" near "money" have different embeddings.

---

## Classic NLP Tasks

### Text Classification
Assign label to text. Approaches: TF-IDF + LogReg/SVM (strong baseline), fine-tuned BERT.

Your Toxicity Classification: XLM-RoBERTa-Large (560M), multi-task with toxicity + intent heads.

### Named Entity Recognition (NER)
Tag tokens with entity types (PER, ORG, LOC, DATE).
Approaches: CRF, BiLSTM-CRF, BERT fine-tuned for token classification.

### Part-of-Speech (POS) Tagging
Assign grammatical role: noun, verb, adjective, etc. Used in parsing, lemmatization, feature engineering.

### Dependency Parsing
Identify grammatical relations between words. SpaCy provides efficient dependency parser.

### Coreference Resolution
Identify all mentions referring to same entity. "Alice said she was tired" → she = Alice.

### Machine Translation
seq2seq with attention → Transformer; dominant approach now is multilingual LLMs.

---

## Sentiment Analysis

### Approaches
- **Lexicon-based**: VADER, AFINN; rule-based; fast; no training needed
- **ML-based**: TF-IDF + classifier; supervised; needs labeled data
- **Deep learning**: BERT fine-tuned; state-of-the-art; handles context and negation

### Aspect-Based Sentiment Analysis (ABSA)
Not just overall sentiment, but sentiment per aspect: "food was great but service was slow" → food: positive, service: negative.

### Your Oracle Work
Sentiment analysis of 30+ customer discovery calls. Likely approaches: audio → transcript (Whisper) → BERT-based sentiment scoring → aggregate by theme/topic.

---

## Multi-task Learning (Your Toxicity Project)

### Architecture
Shared backbone + multiple task-specific heads.

```python
class MultitaskClassifier(nn.Module):
    def __init__(self, backbone_name, n_intents):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.toxicity_head = nn.Linear(hidden, 2)    # binary
        self.intent_head = nn.Linear(hidden, n_intents)  # multi-label

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]     # [CLS] token
        return self.toxicity_head(cls), self.intent_head(cls)
```

### Intent Masking (Your Innovation)
For multilingual data without all intent labels, mask loss for undefined labels:
```python
intent_loss = F.binary_cross_entropy_with_logits(
    intent_logits, intent_labels, weight=intent_mask  # 0 where label absent
)
```

### Loss Combination
```python
total_loss = alpha * toxicity_loss + beta * intent_loss
```
Weight by task importance or use uncertainty weighting.

---

## Multilingual NLP

### XLM-RoBERTa (Your Project: 15+ languages)
Pretrained on 2.5TB of CommonCrawl data in 100 languages using MLM objective. Shared multilingual vocabulary (250K sentencepiece tokens). Strong zero-shot cross-lingual transfer.

Used in your project for 15+ languages: English, Arabic, Russian, Chinese, Spanish, German, French, Italian, Portuguese, Hindi, Ukrainian, Turkish, Tatar, Kazakh.

### Cross-lingual Transfer Learning
Train on high-resource language → evaluate on low-resource language. Works because XLM-RoBERTa learns language-agnostic representations.

### Multilingual Evaluation
Per-language F1 and ROC-AUC per intent class. Critical for fairness: model should not systematically underperform on specific languages.

### Jigsaw Dataset (Used in Your Project)
Toxic comment classification; English; 6 labels (toxic, severe toxic, obscene, threat, insult, identity hate). Used as primary English training source.

### TextDetox Dataset (Used in Your Project)
Multilingual toxicity; 14+ non-English languages. Combined with Jigsaw for comprehensive multilingual coverage.

---

## Text Generation

### Sequence-to-Sequence (Seq2Seq)
Encoder reads input sequence → context vector → Decoder generates output sequence.
Applications: translation, summarization, question answering.

### Attention Mechanism
Instead of fixed context vector, decoder attends to all encoder hidden states:
e_ij = score(s_{i-1}, h_j)      (alignment score)
alpha_ij = softmax(e_ij)
c_i = sum_j alpha_ij * h_j       (context vector)

### Extractive vs Abstractive Summarization
- **Extractive**: select sentences from source; faithful but restricted to source vocabulary
- **Abstractive**: generate new text; more flexible; can hallucinate

---

## Text Cleaning for LLM Pipelines

### Preprocessing for RAG
- Remove boilerplate (headers, footers, nav menus)
- Deduplicate near-duplicate paragraphs (MinHash, SimHash)
- Detect and handle tables, lists, code blocks separately
- Language detection and filtering (langdetect — used in your MediAssist)

### OCR Post-processing (Your MediAssist: Textract, easyOCR)
- Fix character confusion: 0 vs O, l vs 1
- Reconstruct sentence boundaries lost in OCR
- Handle multi-column layouts

---

## Evaluation

### Classification Tasks
Precision, Recall, F1 (macro, micro, weighted), ROC-AUC per class. See Metrics cheatsheet.

### Generation Tasks
BLEU, ROUGE, BERTScore, METEOR, human evaluation.

### NER
- Entity-level F1: exact span match (start, end, type all correct)
- Partial match: more lenient

### Common Baselines to Beat
- Majority class classifier (accuracy baseline for classification)
- TF-IDF + Logistic Regression (strong baseline for text classification; often within 3-5% of BERT on clean data)
- BM25 (strong baseline for retrieval)

---

## NLP Libraries

### HuggingFace Transformers (Your Projects)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2)

inputs = tokenizer("This is a test.", return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
logits = outputs.logits
probs = torch.softmax(logits, dim=-1)
```

### SpaCy
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a UK startup.")
for ent in doc.ents:
    print(ent.text, ent.label_)  # Apple ORG, UK GPE
```

### NLTK
Classic NLP toolkit. Tokenization, POS tagging, stemming, stopwords. Less performant than SpaCy but comprehensive and educational.

### Langdetect (Your MediAssist)
```python
from langdetect import detect
lang = detect("Bonjour, comment allez-vous?")  # 'fr'
```

---

## Interview Key Points

- **TF-IDF vs Word2Vec**: TF-IDF is sparse, interpretable, no training needed; Word2Vec is dense, captures semantic similarity but needs training corpus.
- **Why subword tokenization?** Handles unknown words and rare words; vocabulary size manageable; works across languages with shared morphology.
- **Stemming vs Lemmatization**: stemming is fast rule-based (may produce non-words); lemmatization is slower dictionary-based (always valid forms); use lemmatization when word meaning matters.
- **Why XLM-RoBERTa over mBERT?** Larger training corpus (2.5TB vs Wikipedia), larger vocabulary (250K vs 119K), stronger multilingual performance especially on low-resource languages.
- **How do you handle class imbalance in NLP?** Weighted loss, oversampling (with care for text), augmentation (back-translation, EDA), threshold tuning, evaluate with macro F1 / PR-AUC.
- **What is catastrophic forgetting?** When fine-tuning on new task, model forgets pretrained knowledge. Mitigation: low learning rate, LoRA (only train adapters), elastic weight consolidation.
