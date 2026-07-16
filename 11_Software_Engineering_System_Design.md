# Software Engineering & System Design

## Software Engineering Fundamentals

### SOLID Principles
- **S — Single Responsibility**: a class/module does one thing; one reason to change
- **O — Open/Closed**: open for extension, closed for modification (add new code, don't change existing)
- **L — Liskov Substitution**: subclasses replaceable for base class without breaking code
- **I — Interface Segregation**: prefer small specific interfaces over large general ones
- **D — Dependency Inversion**: depend on abstractions, not concretions

### DRY, KISS, YAGNI
- **DRY (Don't Repeat Yourself)**: every piece of knowledge has a single authoritative representation
- **KISS (Keep It Simple Stupid)**: simplest solution that works; avoid unnecessary complexity
- **YAGNI (You Aren't Gonna Need It)**: don't build features until they're needed

### Design Patterns

#### Creational
- **Singleton**: only one instance; common for DB connections, config
- **Factory Method**: create objects without specifying exact class
- **Builder**: construct complex objects step by step

#### Structural
- **Adapter**: interface compatibility between incompatible classes
- **Decorator**: add behavior dynamically without modifying class
- **Facade**: simplified interface to complex subsystem

#### Behavioral
- **Strategy**: define family of algorithms; swap at runtime (e.g., different retrieval strategies in RAG)
- **Observer**: event-driven; subscribers notified of state changes
- **Chain of Responsibility**: pass request along chain of handlers (LangChain's chain pattern)

---

## REST API Design

### Principles
- **Stateless**: each request contains all info needed; server stores no client state
- **Resource-based**: URLs represent resources (nouns, not verbs)
- **HTTP methods**: GET (read), POST (create), PUT (replace), PATCH (partial update), DELETE (remove)
- **Status codes**: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 422 Validation Error, 500 Internal Server Error

### REST API Conventions
```
GET    /users              List all users
GET    /users/{id}         Get specific user
POST   /users              Create user
PUT    /users/{id}         Replace user
PATCH  /users/{id}         Partial update
DELETE /users/{id}         Delete user
GET    /users/{id}/orders  Nested resource
```

### FastAPI (Your MLOps Project)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI()

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: float

session = ort.InferenceSession("model.onnx")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        input_array = np.array([request.features], dtype=np.float32)
        output = session.run(None, {"input": input_array})[0]
        return PredictResponse(prediction=int(output[0]), probability=float(max(output[0])))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## System Design Framework

### Approach (Use in Interviews)
1. **Clarify requirements**: functional (what it does), non-functional (scale, latency, availability)
2. **Estimate scale**: DAU, QPS, storage, bandwidth
3. **High-level design**: major components and data flow
4. **Deep dive**: critical components; trade-offs
5. **Scale and reliability**: caching, sharding, replication, monitoring

---

## ML System Design

### ML Pipeline Architecture

```
Data Sources → Ingestion → Feature Store → Training → Model Registry
                                                           |
                                                    Model Serving → Monitoring
                                                           |
                                                      Client Apps
```

### Feature Store
Centralized repository for features:
- **Online**: low-latency serving (Redis, DynamoDB); real-time inference
- **Offline**: historical features (Parquet, Hive); training data
- Point-in-time correctness: avoid training-serving skew by using same feature computation

### Model Serving Patterns
- **Online (synchronous)**: REST API; FastAPI + ONNX (your project); low latency
- **Batch**: scheduled scoring on large datasets; Spark, Airflow
- **Streaming**: real-time scoring on event streams; Kafka + model consumer
- **Edge**: model deployed on device; low-latency, no network dependency

### Training-Serving Skew
When features computed differently at training vs inference. Causes silent degradation. Solution: share feature computation code via feature store.

### Model Versioning and A/B Testing
- Shadow mode: new model runs in parallel; no impact; compare outputs
- Canary deployment: route 5% traffic to new model; monitor; increase if good
- A/B test: random split; statistical significance test on business metric

---

## Databases and Storage

### ACID Properties (Relational DB)
- **Atomicity**: transaction is all-or-nothing
- **Consistency**: DB moves from valid to valid state
- **Isolation**: concurrent transactions behave as if serial
- **Durability**: committed data persists despite failures

### CAP Theorem
Distributed system can guarantee only 2 of:
- **Consistency**: all nodes see same data
- **Availability**: every request gets a response
- **Partition Tolerance**: system works despite network partition

Real systems choose CP (Postgres, Zookeeper, HBase) or AP (Cassandra, DynamoDB, CouchDB). Network partitions inevitable, so practically choose C or A.

### SQL vs NoSQL
| Feature | SQL (Postgres) | NoSQL (MongoDB, DynamoDB) |
|---|---|---|
| Schema | Fixed | Flexible |
| Query | Rich SQL | Limited / document scan |
| ACID | Full | Often eventual consistency |
| Scaling | Vertical + read replicas | Horizontal sharding |
| Relations | Foreign keys | Embed or application-level |
| Use case | Complex queries, transactions | High write throughput, variable schema |

### Indexing Strategies
- B-tree: default; equality + range queries
- Hash: equality only; faster for exact match
- GIN: for full-text search, arrays, JSONB
- Partial index: index subset of rows where condition is true; smaller and faster
- Composite index: multiple columns; column order matters; matches left-prefix

### Database Scaling
- **Read replicas**: replicate to secondary nodes; route reads there
- **Connection pooling**: PgBouncer, RDS Proxy; reuse DB connections
- **Partitioning**: horizontal (sharding by key), vertical (split tables by columns), range, hash, list
- **Caching**: Redis/Memcached in front of DB; cache aside, write-through, write-behind patterns

---

## Caching

### Cache Strategies
- **Cache-aside (lazy loading)**: app checks cache → miss → load from DB → write to cache
- **Write-through**: write to cache and DB simultaneously
- **Write-behind**: write to cache; async flush to DB; risk of data loss
- **Read-through**: cache layer handles DB reads transparently

### Cache Eviction Policies
- **LRU (Least Recently Used)**: evict least recently accessed; good for temporal locality
- **LFU (Least Frequently Used)**: evict least accessed overall
- **TTL (Time-To-Live)**: expire after fixed duration; prevents stale data

### Redis Use Cases
- Session storage, rate limiting, pub/sub messaging, sorted sets for leaderboards, distributed locks, caching LLM responses

---

## Messaging and Queues

### Why Message Queues?
Decouple producers and consumers; async processing; buffer traffic spikes; retry on failure.

### Kafka (Relevant to Your Data Engineering)
- **Topic**: category/feed; partitioned for parallelism
- **Partition**: ordered immutable sequence; each has leader + replicas
- **Consumer Group**: consumers share partitions; each partition consumed by one consumer per group
- **Offset**: position in partition; committed by consumer
- **Retention**: messages retained for configured period (not deleted on consume)

Kafka guarantees: at-least-once, at-most-once, or exactly-once (with transactions).

Use cases: event streaming, CDC (Debezium), log aggregation, ML feature pipelines.

---

## Microservices vs Monolith

### Monolith
Single deployable unit. Simple to start. Hard to scale independently. Tight coupling.

### Microservices
Independent services per business domain. Independent deployment, scaling, tech stack. Complex: distributed tracing, service discovery, eventual consistency.

### When to Use Each
Monolith first; extract services when scaling pain identified. Premature microservices add operational overhead without benefit.

### Service Communication
- **Synchronous REST/gRPC**: simple; tight coupling; cascading failures
- **Async messaging (Kafka/RabbitMQ)**: loose coupling; resilient; harder to reason about

---

## Reliability Engineering

### SLI / SLO / SLA
- **SLI (Service Level Indicator)**: measured metric (request success rate, p99 latency)
- **SLO (Service Level Objective)**: target for SLI (99.9% success rate, p99 < 200ms)
- **SLA (Service Level Agreement)**: contractual SLO with consequences for breach

### Error Budget
Error budget = 1 - SLO. If 99.9% SLO, 0.1% budget = ~43 min/month downtime allowed. When exhausted, freeze new features; focus on reliability.

### Common Reliability Patterns
- **Circuit Breaker**: stop calling failing service; fast-fail; retry after cooldown
- **Retry with Exponential Backoff**: retry transient failures; avoid thundering herd
- **Rate Limiting**: protect service from overload; token bucket, leaky bucket algorithms
- **Bulkhead**: isolate failures; thread pool per dependency; one slow service can't block all others
- **Health Checks**: liveness (is service alive?), readiness (can it serve traffic?)

---

## Distributed Systems Concepts

### Consensus
- **Raft**: leader election + log replication; used in etcd (Kubernetes), CockroachDB
- **Paxos**: theoretical foundation; complex to implement

### Distributed Transactions
- **Two-Phase Commit (2PC)**: coordinator prepares → participants vote → commit/rollback. Blocking if coordinator fails.
- **Saga**: chain of local transactions; compensating transactions for rollback. Either choreography (events) or orchestration (coordinator).

### Consistency Models
- **Strong**: every read sees most recent write
- **Eventual**: all replicas converge; reads may see stale data temporarily
- **Read-your-writes**: user always sees their own writes
- **Causal**: causally related operations seen in order

---

## ML System Design Walkthroughs

### Design: Real-Time Recommendation System (YouTube/Netflix Scale)

**Requirements**: personalized recommendations for 100M+ users, <200ms latency, handle 100K QPS

**Architecture (4 stages)**:
1. **Candidate Generation** (~1000 candidates, <50ms)
   - Two-tower model: user embedding + item embedding → ANN retrieval (FAISS/ScaNN)
   - Multiple sources: collaborative filtering, content-based, trending, user history
2. **Scoring/Ranking** (~100 candidates, <50ms)
   - Deep ranking model (DLRM/DCN-v2): dense + sparse features, feature interactions
   - Features: user profile, item attributes, context (time, device), interaction history
3. **Re-ranking** (<20ms)
   - Business rules: diversity, freshness, content policy
   - Explore/exploit: epsilon-greedy or Thompson Sampling for new items
4. **Policy Layer**
   - Deduplication, frequency capping, A/B test bucketing

**Data Pipeline**: user events → Kafka → Flink (real-time features) + Spark (batch features) → Feature Store (online: Redis, offline: Hive)

**Capacity Math**: 100M users × 10 req/day = 1B req/day ≈ 12K QPS average, 100K peak. Each request: feature fetch (5ms) + model inference (20ms) + ANN lookup (10ms) + re-rank (5ms) = ~40ms.

### Design: LLM-Powered Search System

**Architecture**:
1. **Query Understanding**: intent classification, query expansion, entity extraction
2. **Retrieval**: hybrid search (BM25 + dense embeddings via pgvector/FAISS)
3. **Reranking**: cross-encoder reranker on top-50 candidates
4. **Generation**: LLM synthesizes answer from top-5 passages with citations
5. **Guardrails**: input/output filtering, hallucination detection

**Latency Budget (total: 2s)**:
- Query processing: 50ms
- Retrieval: 100ms
- Reranking: 200ms
- LLM generation: 1500ms (streaming)

---

## Monitoring and Observability

### Prometheus + Grafana Stack
- **Prometheus**: pull-based metrics collection. Instrument your serving code with counters, histograms, gauges.
- **Grafana**: visualization dashboards. Track: request latency (p50/p95/p99), error rate, model prediction distribution, feature drift.
- **Alertmanager**: alert on SLO violations (e.g., p99 > 200ms for 5 min).

### ML-Specific Monitoring
- **Data drift**: PSI (Population Stability Index) or KL divergence on input features
- **Prediction drift**: distribution of model outputs shifts over time
- **Model quality**: track online metrics (CTR, conversion) vs offline (AUC, NDCG)
- **Concept drift**: relationship between features and target changes → model performance degrades

### Drift Detection Formula (PSI)
```
PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
```
PSI < 0.1: no shift. 0.1-0.25: moderate. > 0.25: significant → investigate or retrain.

---

## gRPC for ML Serving

### Why gRPC over REST
| Aspect | REST | gRPC |
|---|---|---|
| Serialization | JSON (text) | Protobuf (binary, 3-10× smaller) |
| Latency | Higher | Lower (binary, HTTP/2) |
| Streaming | Limited | Bidirectional streaming |
| Schema | OpenAPI (optional) | Required (.proto files) |
| Use case | Public APIs | Internal microservices, model serving |

### Protobuf Definition
```protobuf
service ModelService {
  rpc Predict (PredictRequest) returns (PredictResponse);
  rpc StreamPredict (PredictRequest) returns (stream PredictResponse);  // streaming
}

message PredictRequest {
  repeated float features = 1;
  string model_version = 2;
}

message PredictResponse {
  repeated float predictions = 1;
  float latency_ms = 2;
}
```

---

## Model Drift and Retraining

### Automated Retraining Pipeline
1. **Monitor**: track PSI/KL on features + model performance metrics
2. **Trigger**: when drift exceeds threshold OR scheduled (weekly/monthly)
3. **Retrain**: on fresh data, validate on holdout
4. **Shadow deploy**: run new model in shadow mode, compare predictions
5. **Promote**: if shadow model meets quality bar, swap via blue-green deployment
6. **Rollback**: if online metrics degrade within first hour, auto-rollback

### Latency Budget Decomposition
```
Total P99 budget: 200ms
├── Network (client → LB → service): 10ms
├── Feature retrieval (feature store): 30ms
├── Preprocessing: 10ms
├── Model inference: 50ms
├── Postprocessing: 10ms
├── Response serialization: 5ms
└── Buffer: 85ms
```
Identify bottleneck → optimize that component first (model inference → quantize/distill; feature retrieval → cache hot features).

---

## Interview Key Points

- **How would you design a real-time fraud detection system?** Event stream (Kafka) → feature computation → online model serving → rule engine → alert service. Feature store provides real-time + historical features.
- **How would you design an LLM-powered search system (like your MTech work)?** Query → embedding → hybrid retrieval (dense + BM25) → reranker → LLM augmentation → response with citations. Postgres + pgvector for structured + vector search.
- **How would you scale a model serving endpoint?** Horizontal scaling with K8s HPA, load balancer in front, model caching (avoid re-loading), batch inference for throughput, ONNX/TensorRT for optimization, async endpoints.
- **Your model's P99 latency is 500ms, SLA is 200ms — how do you debug?** Decompose latency budget, profile each component, check if it's model inference (quantize), feature retrieval (cache), or network (CDN/edge).
- **How do you detect model drift?** PSI on input features, KL divergence on predictions, track online metrics. Trigger retraining when PSI > 0.25.
- **Trade-off between consistency and availability in your data systems?** For analytics and ETL, eventual consistency acceptable (use AP systems like Cassandra for high-write). For financial/reporting data (Oracle ETL work), strong consistency required.
