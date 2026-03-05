# AWS & Cloud ML (Bonus Cheatsheet)

## Core AWS Services (Resume-Relevant)

### Compute
| Service | Purpose | Your Usage |
|---|---|---|
| EC2 | Virtual machines | Model training, custom servers |
| Lambda | Serverless functions (max 15 min) | ETL triggers, lightweight inference |
| SageMaker | Managed ML platform | Training, hyperparameter tuning, endpoints |
| Fargate | Serverless containers (ECS/EKS) | ML API deployment |
| Batch | Managed batch computing | Large-scale data processing |

### Storage
| Service | Purpose | Notes |
|---|---|---|
| S3 | Object storage | Parquet files, model artifacts, datasets |
| EBS | Block storage attached to EC2 | Fast local disk |
| EFS | Shared file system (NFS) | Multi-instance access; ML cluster shared storage |

### AI/ML Services (Your Projects)
| Service | Purpose | Your Usage |
|---|---|---|
| Bedrock | Managed LLM API (Claude, LLaMA, Titan) | MediAssist online mode; Claude 3.5 |
| SageMaker | Full ML lifecycle | Training pipelines, model registry |
| Textract | OCR — extract text from images/PDFs | MediAssist prescription OCR |
| Polly | Text-to-speech (TTS) | MediAssist voice output; Hindi + English |
| Translate | Machine translation | MediAssist multilingual support |
| Transcribe / (Whisper) | Speech-to-text | MediAssist voice input |
| Rekognition | Image/video analysis | CV tasks |
| Comprehend | NLP (sentiment, entities, PII) | Text analytics |

---

## AWS Bedrock Deep Dive

### What It Provides
- API access to foundation models: Claude (Anthropic), LLaMA (Meta), Titan (Amazon), Mistral, Stable Diffusion (Stability AI)
- No model management; pay per token/image
- Data stays within your AWS account (privacy)
- Guardrails: built-in content filtering, PII detection, grounding checks

### Integration Pattern (Your MediAssist)
```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What are the symptoms of hypertension?"}
        ]
    })
)
result = json.loads(response["body"].read())
text = result["content"][0]["text"]
```

### Bedrock Knowledge Bases
Managed RAG: connect S3 data sources → Bedrock indexes and creates embeddings → query via API. Handles chunking, embedding, vector storage (OpenSearch Serverless), retrieval, and augmentation.

---

## AWS SageMaker

### Key Components
- **Training Jobs**: managed distributed training; auto-provisioning
- **Processing Jobs**: data preprocessing at scale
- **Hyperparameter Tuning**: Bayesian optimization across training jobs
- **Model Registry**: versioned model artifacts with approval workflow
- **Endpoints**: real-time inference; auto-scaling; A/B testing
- **Pipelines**: orchestrate end-to-end ML workflows (like Airflow but ML-native)
- **Feature Store**: managed online + offline feature storage

### Training Job
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role=sagemaker_role,
    instance_type="ml.p3.16xlarge",  # 8x V100 GPUs
    instance_count=2,
    framework_version="2.1.0",
    py_version="py310",
    hyperparameters={"lr": 0.001, "epochs": 10},
    distribution={"torch_distributed": {"enabled": True}}  # DDP
)
estimator.fit({"train": "s3://bucket/train/", "val": "s3://bucket/val/"})
```

### Real-time Endpoint
```python
predictor = estimator.deploy(
    initial_instance_count=2,
    instance_type="ml.g4dn.xlarge",
    endpoint_name="my-model-endpoint"
)
result = predictor.predict({"features": [1.0, 2.0, 3.0]})
```

---

## AWS Lambda for ML Pipelines

### Serverless ETL Pattern
```python
import boto3
import json

def lambda_handler(event, context):
    # Triggered by S3 put event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    # Process new data file
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read().decode()

    # Invoke SageMaker endpoint
    sm = boto3.client("sagemaker-runtime")
    response = sm.invoke_endpoint(
        EndpointName="my-model",
        ContentType="application/json",
        Body=json.dumps({"data": data})
    )
    prediction = json.loads(response["Body"].read())
    return {"statusCode": 200, "prediction": prediction}
```

**Limits**: 15 min timeout, 10 GB memory, 512 MB /tmp storage. Not suitable for GPU workloads — use SageMaker or EC2.

---

## AWS Solutions Architect Concepts (Your Certification)

### Well-Architected Framework Pillars
1. **Operational Excellence**: automate, iterate, monitor
2. **Security**: IAM least privilege, encryption at rest/transit, VPC
3. **Reliability**: multi-AZ, auto-scaling, health checks, backups
4. **Performance Efficiency**: right instance type, caching, CDN
5. **Cost Optimization**: reserved/spot instances, S3 lifecycle, right-sizing
6. **Sustainability**: minimize environmental impact

### IAM (Identity and Access Management)
- **User**: human identity
- **Role**: assumed by services (EC2, Lambda, SageMaker); temporary credentials
- **Policy**: JSON document defining allowed/denied actions on resources
- **Least privilege**: grant only permissions needed

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject"],
    "Resource": "arn:aws:s3:::my-ml-bucket/*"
  }]
}
```

### VPC (Virtual Private Cloud)
Isolated network. Public subnets (internet gateway access), private subnets (no direct internet). NAT gateway for outbound internet from private subnet. Security groups: stateful firewall at instance level. NACLs: stateless firewall at subnet level.

### Auto Scaling
- **Target tracking**: maintain metric at target (CPU 70%)
- **Step scaling**: scale by amount based on alarm threshold
- **Scheduled**: scale at known times
- **Predictive**: ML-based demand forecasting

---

## Multi-Modal Pipeline (Your MediAssist)

### Voice-to-Voice Architecture
```
User Audio
    |
    v
faster-whisper (STT) -- offline: local model
    OR
Amazon Transcribe -- online: managed API
    |
    v
Language Detection (langdetect)
    |
    v
Amazon Translate (if non-English)
    |
    v
RAG Pipeline (Pinecone + all-MiniLM-L6-v2)
    |
    v
LLM: Claude 3.5 via Bedrock (online)
    OR
BioGPT GGUF via llama.cpp (offline)
    |
    v
Amazon Translate (response to user language)
    |
    v
Amazon Polly TTS (online)
    OR
Glow-TTS (offline)
    |
    v
Audio Response
```

### AWS Textract (OCR for Medical Documents)
- Extracts text, forms, tables from PDFs and images
- Medical forms: prescription parsing, lab reports
- Returns structured JSON with detected text + bounding boxes
- Handles handwritten text (limited)
- Offline fallback: easyOCR

---

## Cost Optimization for ML on AWS

### Spot Instances
Up to 90% cheaper than on-demand. Can be interrupted. Use for fault-tolerant training with checkpointing.

```python
# SageMaker managed spot training
estimator = PyTorch(
    use_spot_instances=True,
    max_wait=7200,  # seconds to wait for spot
    max_run=3600,
    checkpoint_s3_uri="s3://bucket/checkpoints/"
)
```

### Reserved Instances / Savings Plans
1 or 3 year commitment. Up to 72% savings. Use for production inference endpoints.

### S3 Storage Classes
- **S3 Standard**: frequent access
- **S3 Intelligent-Tiering**: auto-moves between tiers
- **S3 Glacier**: archival; minutes-to-hours retrieval; much cheaper
- Set lifecycle policies to auto-archive old training data

### Right-Sizing
- Use CloudWatch metrics to identify underutilized instances
- Start with p3.2xlarge (1x V100) before going to p3.16xlarge (8x V100)
- Graviton instances (ARM) for CPU inference: 40% cheaper than x86

---

## AWS Certification Relevant Topics (Solutions Architect Associate)

### High Availability Patterns
- Multi-AZ RDS: synchronous standby replica; automatic failover
- Multi-region S3 replication: DR, latency reduction
- Route 53 health checks + failover routing

### Serverless Architecture
- API Gateway + Lambda + DynamoDB: fully serverless; scales to zero
- Step Functions for workflow orchestration with Lambda
- EventBridge for event-driven architectures

### Data Pipeline on AWS
- S3 (raw) → Glue Crawler (catalog) → Glue ETL (transform) → S3 (processed) → Athena (query) / Redshift (warehouse) / SageMaker (train)

### Key Services to Know
- **CloudWatch**: metrics, logs, alarms, dashboards
- **CloudTrail**: API audit logging; who did what when
- **CloudFormation / CDK**: infrastructure as code
- **ECR**: container image registry (for Docker images)
- **EKS**: managed Kubernetes
- **RDS**: managed relational DB (Postgres, MySQL, Aurora)

---

## Interview Key Points

- **When to use Lambda vs EC2 vs SageMaker for ML?** Lambda: lightweight preprocessing, webhook handlers, <15min, no GPU. EC2: full control, custom GPU setup, long-running. SageMaker: managed training + deployment, built-in distributed training, model registry, A/B testing.
- **How does Bedrock differ from directly calling Claude API?** Bedrock: stays in AWS VPC, no data sent to Anthropic, IAM-controlled access, same region as data. Direct API: simpler, outside AWS ecosystem.
- **How would you make a Lambda function more memory-efficient?** Import only needed modules, use lazy loading, reuse connections outside handler, use Layers for large dependencies, consider EFS for large model files.
- **What is the difference between SageMaker endpoint and Lambda for inference?** SageMaker: persistent GPU-backed compute, auto-scaling, A/B deployments, monitoring. Lambda: serverless CPU only, cold start, pay-per-invocation, simpler.
