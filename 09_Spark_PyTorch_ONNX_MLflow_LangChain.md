# Spark, PyTorch, ONNX, MLflow, LangChain, and LangGraph

## Apache Spark (PySpark)

### Architecture
- **Driver**: main program; creates SparkContext; schedules jobs
- **Cluster Manager**: allocates resources (YARN, Kubernetes, Standalone, Mesos)
- **Worker Nodes**: execute tasks; hold partitions in memory
- **Executor**: JVM process on worker; runs tasks; caches data
- **Task**: unit of work on one partition

### Core Abstractions

#### RDD (Resilient Distributed Dataset)
Immutable distributed collection. Fault-tolerant via lineage (can recompute lost partitions).

```python
rdd = sc.parallelize([1, 2, 3, 4, 5], numSlices=3)
rdd.map(lambda x: x*2).filter(lambda x: x > 4).collect()
```

**Transformations** (lazy): map, filter, flatMap, groupByKey, reduceByKey, join, union
**Actions** (trigger execution): collect, count, first, take, reduce, saveAsTextFile

#### DataFrame / Dataset API (Preferred)
Structured API with schema. Optimized by Catalyst optimizer and Tungsten execution engine.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when

spark = SparkSession.builder.appName("ETL").getOrCreate()
df = spark.read.parquet("s3://bucket/data/")

# Transformations
result = (df
    .filter(col("year") == 2023)
    .groupBy("category")
    .agg(avg("amount").alias("avg_amount"), count("*").alias("n"))
    .orderBy(col("avg_amount").desc())
)
result.write.parquet("s3://bucket/output/", mode="overwrite")
```

### Execution Model
1. **Job**: triggered by action
2. **Stage**: sequence of narrow transformations (no shuffle); broken by wide transformations (groupBy, join, sort)
3. **Task**: one stage processed on one partition

**Narrow transformation**: each input partition maps to one output partition (map, filter). Pipelined in same stage.
**Wide transformation**: requires shuffling data across partitions (groupBy, join, sort). Causes stage boundary.

### Your Netflix Project
- Processed 100M records; modeled on 10M-record subset
- Stored data as Parquet on Amazon S3
- Built lookup dictionaries (user2movie, movie2user, usermovie2rating) to reduce time complexity
- Used Databricks for distributed processing

### Spark Optimization
- **Caching**: `df.cache()` or `df.persist(StorageLevel.MEMORY_AND_DISK)` for reused DataFrames
- **Broadcast joins**: small table replicated to all executors; avoids shuffle: `spark.sql.autoBroadcastJoinThreshold`
- **Partition tuning**: `spark.sql.shuffle.partitions` (default 200; adjust to data size)
- **Predicate pushdown**: filter early before joins and aggregations
- **Avoid UDFs when possible**: use built-in SQL functions (Catalyst can optimize); Python UDFs are slow (serialization overhead)
- **Coalesce vs Repartition**: coalesce reduces partitions without full shuffle; repartition fully reshuffles

### Parquet Format (Used in Your Projects)
- Columnar; predicate pushdown; efficient compression per column
- Supports schema evolution; partition discovery
- Read only required columns: reduces I/O dramatically
- Write partitioned by date/category for efficient time-range queries

---

## PyTorch

### Core Concepts

#### Tensor Operations
```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()       # compute gradients
print(x.grad)      # tensor([[2., 4.], [6., 8.]])

# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

#### Autograd (Automatic Differentiation)
PyTorch builds a dynamic computation graph as operations execute. `backward()` traverses graph in reverse, accumulating gradients in `.grad` attributes.

`requires_grad=True` marks leaf tensors for gradient tracking.
`torch.no_grad()`: context manager to disable gradient computation (inference, evaluation).
`detach()`: removes tensor from graph.

#### Custom Module
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
```

#### Training Loop
```python
model = MLP(784, 256, 10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
```

#### DataLoader
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

loader = DataLoader(CustomDataset(X, y), batch_size=32, shuffle=True, num_workers=4)
```

### Mixed Precision (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Saving and Loading
```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

---

## ONNX (Open Neural Network Exchange)

### What is ONNX?
Open format for representing ML models. Framework-agnostic intermediate representation. Enables deployment across different runtimes and hardware.

### PyTorch → ONNX Export (Your MLOps Project)
```python
dummy_input = torch.randn(1, input_dim)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17
)
```

### ONNX Runtime Inference
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: numpy_input})[0]
```

### Why ONNX?
- Framework-agnostic deployment (PyTorch → TF Serving, ONNX Runtime, TensorRT)
- Faster inference via graph optimizations (operator fusion, constant folding)
- Hardware portability: CPU, GPU, NPU, edge devices
- Model size can be reduced via quantization post-export

### ONNX in Your Pipeline (Titanic Project)
PyTorch training → ONNX export → FastAPI inference endpoint → Docker container → pytest CI suite.

---

## MLflow

### What is MLflow?
Open-source platform for managing the ML lifecycle: tracking, model registry, projects, deployment.

### Components

#### Tracking
```python
import mlflow

mlflow.set_experiment("debt-default-prediction")

with mlflow.start_run():
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("accuracy", 0.99)
    mlflow.log_metric("macro_f1", 0.97)
    mlflow.sklearn.log_model(model, "lgbm_model")
    mlflow.log_artifact("feature_importance.png")
```

#### Model Registry
- **Staging**: candidate model; under testing
- **Production**: deployed model
- **Archived**: retired version

```python
mlflow.register_model("runs:/run_id/lgbm_model", "DebtDefaultClassifier")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage("DebtDefaultClassifier", version=1, stage="Production")
```

#### MLflow Projects
Packaging format with conda.yaml or requirements.txt. `mlflow run .` to execute.

#### Serving
```bash
mlflow models serve -m "models:/DebtDefaultClassifier/Production" --port 5000
```

### Auto-logging
```python
mlflow.sklearn.autolog()    # automatically logs params, metrics, artifacts
mlflow.pytorch.autolog()
mlflow.xgboost.autolog()
```

### Comparison with W&B (Your Animal Classification Project)
- **Weights & Biases (W&B)**: richer UI; real-time charts; system metrics; collaborative; hosted
- **MLflow**: open-source; self-hostable; more ML lifecycle management (registry, deployment)
- Both used in your projects: MLflow for structured MLOps pipelines; W&B for experiment visualization

---

## LangChain

### What is LangChain?
Framework for building LLM-powered applications. Provides abstractions for chains, prompts, retrievers, tools, memory, and agents.

### Core Primitives

#### LLM and Chat Models
```python
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
response = llm([SystemMessage(content="You are a medical assistant."),
                HumanMessage(content="What is hypertension?")])
```

#### Prompt Templates
```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in {domain}."),
    ("human", "{question}")
])
prompt = template.format_messages(domain="medicine", question="What is diabetes?")
```

#### Chains (LCEL — LangChain Expression Language)
```python
from langchain_core.output_parsers import StrOutputParser

chain = template | llm | StrOutputParser()
result = chain.invoke({"domain": "medicine", "question": "What is hypertension?"})
```

#### Retrieval Chain
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type="stuff"  # or "map_reduce", "refine", "map_rerank"
)
```

#### Memory
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5, return_messages=True)
```

#### Agents
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun

tools = [DuckDuckGoSearchRun()]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "What are the latest treatments for Type 2 diabetes?"})
```

### Chain Types for RAG
- **stuff**: concatenate all docs into context; simple; limited by context window
- **map_reduce**: process each doc independently; reduce results; handles many docs
- **refine**: iteratively refine answer with each doc; high quality; slow
- **map_rerank**: score each doc independently; return highest-scored answer

---

## LangGraph

### What is LangGraph?
Extension of LangChain for building stateful, multi-actor, graph-based agent workflows. Built on a state machine abstraction.

### Core Concepts

#### State Graph
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    answer: str
    iterations: int

graph = StateGraph(AgentState)
```

#### Nodes and Edges
```python
def retrieve(state: AgentState) -> AgentState:
    docs = retriever.get_relevant_documents(state["query"])
    return {**state, "retrieved_docs": docs}

def generate(state: AgentState) -> AgentState:
    answer = llm.invoke(build_prompt(state))
    return {**state, "answer": answer}

def should_continue(state: AgentState) -> str:
    if state["iterations"] >= 3 or answer_is_good(state["answer"]):
        return "end"
    return "retrieve"

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge("retrieve", "generate")
graph.add_conditional_edges("generate", should_continue, {"end": END, "retrieve": "retrieve"})
graph.set_entry_point("retrieve")

app = graph.compile()
result = app.invoke({"query": "What is hypertension?", "iterations": 0})
```

### LangGraph vs LangChain Agents
- **LangChain agents**: linear ReAct loop; less control over flow
- **LangGraph**: explicit graph; conditional branching; loops; state persistence; better for complex multi-step workflows
- Use LangGraph for: multi-agent orchestration, human-in-the-loop, complex state machines, retry logic

### Key Patterns
- **Supervisor pattern**: one agent routes tasks to specialized sub-agents
- **Parallel execution**: fan out to multiple agents; join results
- **Human-in-the-loop**: interrupt graph, wait for human input, resume
- **Memory persistence**: checkpointing state to DB between graph invocations

### Relationship to Your AutoGen Work
AutoGen provides similar multi-agent coordination. LangGraph is more explicit graph-based; AutoGen uses conversational pattern with agents. Both enable orchestration of complex multi-step LLM workflows.

---

## Interview Key Points

**Spark**
- Why is Spark faster than MapReduce? In-memory processing; RDD lineage avoids disk writes; DAG optimization; pipelining transformations.
- What causes a shuffle? Wide transformations: groupBy, join, sort, repartition. Shuffle = disk I/O + network transfer; expensive.
- Difference between cache() and persist()? cache() = persist(MEMORY_ONLY); persist() allows specifying storage level.

**PyTorch**
- Dynamic vs static computation graph? PyTorch: dynamic (define-by-run); TensorFlow 1.x: static. Dynamic allows Python control flow naturally.
- What is gradient accumulation? Sum gradients over multiple mini-batches before calling optimizer.step(). Simulates larger batch on limited memory.

**ONNX**
- What optimizations does ONNX Runtime apply? Operator fusion (conv+bn+relu → single op), constant folding, memory planning, quantization.

**MLflow**
- How do you compare experiments? Use MLflow UI or `mlflow.search_runs()` to query runs by metrics/params. Tag runs for filtering.

**LangChain/LangGraph**
- When to use LangGraph over simple chains? When you need loops, conditional branching, state persistence, human-in-the-loop, or multiple parallel agents.
