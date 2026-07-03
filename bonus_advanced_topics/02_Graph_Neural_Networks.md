# Graph Neural Networks (GNNs)

## Overview

**Graphs**: nodes (entities) + edges (relationships). Examples: social networks, citations, molecules, knowledge graphs.

**GNNs**: neural networks operating on graph structure; learn node/edge/graph representations.

**Key insight**: leverage both node features AND graph structure.

---

## Graph Basics

### Graph Representation
- **Adjacency matrix A**: A[i,j] = 1 if edge exists, 0 otherwise
- **Node features X**: d-dimensional features per node
- **Edge attributes**: optional weights/types

```python
import torch
from torch_geometric.data import Data

# Create a simple graph
x = torch.randn(num_nodes, node_feature_dim)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # edges
data = Data(x=x, edge_index=edge_index)
```

### Graph Tasks
- **Node classification**: predict node labels (e.g., classify users)
- **Link prediction**: predict edges (e.g., friend suggestions)
- **Graph classification**: classify entire graphs (e.g., molecular property)
- **Node embedding**: learn representations for downstream tasks

---

## Graph Convolutional Networks (GCN)

**Core idea**: aggregate features from neighboring nodes.

### Message Passing Framework
Each node receives messages from neighbors; aggregates and updates.

```
h_v^(k+1) = σ(W · AGGREGATE(h_u^(k) for u in neighbors(v)))
```

### Spectral Convolution
Convolution in spectral (frequency) domain using graph Laplacian.

**Graph Laplacian**: L = D - A (D = degree matrix)

Spectral convolution: filter operates on eigenvalues of L.

**Problem**: expensive eigendecomposition; not scalable

### Spatial Convolution (ChebNet / GCN)

Approximate spectral convolution via Chebyshev polynomials; much faster.

**GCN aggregation rule**:
```
h_v^(k+1) = ReLU(W · mean(h_v^(k), h_u^(k) for u in neighbors(v)))
```

Or with normalization:
```
h_v^(k+1) = ReLU(W · Σ_u A_vu / sqrt(d_v · d_u) · h_u^(k))
```

where d_v = degree of node v (normalizes by neighbor count)

```python
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

---

## GraphSAGE (Graph SAmple and aggreGatE)

**Problem with GCN**: requires full graph at inference (not scalable for large graphs)

**Solution**: sample neighbors during training; inductive learning (generalize to unseen nodes).

```
1. Sample fixed-size neighborhood for each node
2. Aggregate sampled neighbors' features
3. Update node representation
```

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

**Advantages**: inductive (handles new nodes); efficient sampling; scalable

---

## Graph Attention Networks (GAT)

Learn different weights for different neighbors (attention).

```
h_v^(k+1) = σ(Σ_u α_vu · W · h_u^(k))
```

where α_vu = attention weight (learned) from u to v

**Attention computation**:
```
e_vu = ReLU(a^T · concat(W·h_v, W·h_u))
α_vu = exp(e_vu) / Σ_w exp(e_vw)  # softmax over neighbors
```

**Multi-head attention**: multiple attention heads; concatenate results

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

**Advantages**: interpretable (attention weights); flexible aggregation

---

## Graph Isomorphism Network (GIN)

Theoretically more expressive than GCN; injective aggregation function.

```
h_v^(k+1) = MLP((1 + ε) · h_v^(k) + Σ_u h_u^(k))
```

ε: learnable parameter controlling self-loop weight

**Expressiveness**: can distinguish more graph structures than GCN

---

## Pooling / Readout

### Global Pooling
Aggregate all node features to single graph representation.

```python
# Mean pooling
graph_embedding = h.mean(dim=0)

# Max pooling
graph_embedding = h.max(dim=0)[0]

# Attention pooling: learned weights
weights = softmax(MLP(h))
graph_embedding = (h * weights).sum(dim=0)
```

### Hierarchical Pooling
Gradually reduce nodes (like image pyramids).

```python
# DiffPool: learn soft clustering
cluster_assignment = softmax(MLP(h))  # soft node-to-cluster
h_pooled = cluster_assignment.T @ h  # aggregate clusters
```

---

## Knowledge Graphs & Embedding

**Knowledge Graph**: entities (nodes) + relations (edge types)

Example: (Albert_Einstein, WorkedAt, Princeton)

### Embedding Methods

**TransE**: relation vector r ≈ tail - head
```
score(h, r, t) = ||h + r - t||
```
Learns: h_embed, r_embed, t_embed

**DistMult**: tensor factorization
```
score(h, r, t) = <h, r, t>  (trilinear product)
```

**ComplEx**: complex-valued embeddings (handles asymmetry better)

```python
import torch
from torch_geometric.nn import TransE

model = TransE(num_entities, num_relations, embedding_dim=50)
score = model.score(head_idx, relation_idx, tail_idx)
```

**Task**: link prediction (find missing edges), triple classification

---

## Training & Loss

### Node Classification
Supervised: use labeled nodes as training set; predict unlabeled nodes.

```python
loss = F.cross_entropy(model(x, edge_index)[train_mask], y[train_mask])
```

### Link Prediction
Predict missing edges. Supervised: positive (real edges) vs negative (non-edges).

```python
# For each edge (u, v): score should be high
# For non-edge pairs: score should be low
pos_edges = ...
neg_edges = sample_negative_edges(num_negs)

loss = F.binary_cross_entropy_with_logits(
    model.predict_edge(pos_edges), 
    torch.ones(pos_edges.size(0))
) + F.binary_cross_entropy_with_logits(
    model.predict_edge(neg_edges),
    torch.zeros(neg_edges.size(0))
)
```

### Graph Classification
Treat entire graphs as samples.

```python
# DataLoader yields multiple graphs
for batch in DataLoader(graph_dataset, batch_size=32):
    embeddings = model(batch.x, batch.edge_index, batch.batch)  # batch groups nodes by graph
    logits = classifier(embeddings)
    loss = F.cross_entropy(logits, batch.y)
```

---

## Scalability & Challenges

### Over-smoothing
Stacking many layers → all nodes become similar (representations converge).

**Solution**: residual connections, normalization, intermediate supervision

### Heterogeneous Graphs
Nodes/edges of different types. Requires separate embeddings per type.

**Example**: social network (users, posts, tags)

### Dynamic Graphs
Nodes/edges appear/disappear over time.

**Solution**: temporal GNNs (store history, update incrementally)

---

## Applications

- **Social Networks**: friend recommendation (link prediction), community detection
- **Chemistry**: molecular property prediction, drug discovery (graph = molecule atoms)
- **Citation Networks**: paper classification (node), finding similar papers
- **Recommendation**: heterogeneous graph (users, items, properties)
- **Knowledge Graphs**: entity linking, question answering

---

## Interview Key Points

- **GCN aggregation: why normalize by degree?** Prevents high-degree nodes from dominating; symmetric normalization balances contribution.
- **GCN vs GraphSAGE?** GCN: transductive (needs full graph). GraphSAGE: inductive (handles new nodes via neighbor sampling).
- **GAT vs GCN?** GAT: learns importance per neighbor (interpretable). GCN: uniform neighbors (faster).
- **How to handle heterogeneous graphs?** Separate embeddings per node type; type-specific aggregation functions.
- **Pooling vs readout?** Pooling: aggregate into fewer nodes (hierarchical). Readout: single graph embedding (classification).
- **Why over-smoothing?** Many layers → averaging out differences → all nodes similar embeddings.
- **Link prediction: why negative sampling?** Full softmax over all non-edges expensive; sample subset.
- **GNN complexity?** Message passing per layer; aggregation O(E) (edges); L layers → O(L·E). Sampling (GraphSAGE) reduces to O(L·S) where S = sample size.
