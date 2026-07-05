# Data Engineering & ETL

## Core ETL Concepts

### ETL vs ELT
**ETL (Extract, Transform, Load)**
- Extract from source → Transform in staging → Load clean data to target
- Traditional; transform before loading; good for smaller data volumes

**ELT (Extract, Load, Transform)**
- Extract → Load raw to data warehouse → Transform in place (SQL/dbt)
- Modern cloud warehouses (Snowflake, BigQuery, Redshift); push compute to warehouse

### Your Resume Context
- Designed ETL workflows processing 200GB+ data (65M+ records) at Oracle
- Optimized queries reducing execution time by 50%
- Automated PL/SQL workflows saving 25 hours/week

---

## Extraction Patterns

### Batch Extraction
Pull data in scheduled batches (hourly, daily). Common with operational databases and file systems.

### Incremental Extraction
Only extract changed/new records since last run:
```sql
SELECT * FROM source WHERE modified_dt > last_extract_timestamp
```

Methods:
- **Timestamp-based**: requires reliable updated_at column
- **CDC (Change Data Capture)**: read database transaction log (binlog, WAL); captures inserts, updates, deletes in real-time
- **Checksum**: hash row data; compare to detect changes

### Real-time / Streaming Extraction
Kafka, Kinesis, Pub/Sub for event streams. Sub-second latency.

### API Extraction
REST / GraphQL calls; handle pagination, rate limits, auth. Salesforce API used in your work.

---

## Transformation Patterns

### Data Cleaning
- Handle nulls: impute (mean/median/mode/forward-fill), drop, or flag
- Remove duplicates: ROW_NUMBER() OVER (PARTITION BY key ORDER BY ts DESC) = 1
- Standardize formats: date formats, case normalization, trimming whitespace
- Validate constraints: referential integrity, range checks, regex patterns

### Data Type Conversions
- Cast strings to dates, numbers
- Handle locale differences (decimal separators, date formats)
- Normalize encodings (UTF-8)

### Business Logic Transformations
- Aggregations: SUM, AVG, COUNT GROUP BY
- Joins: combine tables
- Derived columns: computed fields
- Lookups: map codes to descriptions

### Deduplication (Resume Context)
```sql
WITH deduped AS (
    SELECT *,
    ROW_NUMBER() OVER (PARTITION BY business_key ORDER BY modified_dt DESC) rn
    FROM staging
)
SELECT * FROM deduped WHERE rn = 1;
```

---

## Loading Patterns

### Full Load
Truncate target and reload. Simple; expensive for large tables.

### Incremental Load
Append or upsert only changed records.

### MERGE / Upsert
```sql
MERGE INTO target t
USING source s ON t.id = s.id
WHEN MATCHED THEN UPDATE SET t.value = s.value
WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.value);
```

### Slowly Changing Dimensions (SCD)
- **Type 1**: Overwrite — no history retained
- **Type 2**: Add new row with effective dates; is_current flag — full history
- **Type 3**: Add previous value column — limited history

SCD Type 2 example:
```sql
-- Expire old record
UPDATE dim_customer SET end_date = SYSDATE, is_current = 'N'
WHERE customer_id = :id AND is_current = 'Y';

-- Insert new version
INSERT INTO dim_customer (customer_id, ..., start_date, end_date, is_current)
VALUES (:id, ..., SYSDATE, NULL, 'Y');
```

---

## Data Warehouse Concepts

### Dimensional Modeling
**Star Schema**: fact table at center, dimension tables around it
- **Fact table**: transactional/event data; numeric measures; foreign keys to dims
- **Dimension table**: descriptive attributes; slowly changing; customer, product, date

**Snowflake Schema**: normalized dimension tables. Less redundancy; more joins.

### Fact Types
- **Transactional fact**: one row per event (sales, clicks)
- **Periodic snapshot**: one row per period (daily balance)
- **Accumulating snapshot**: one row per pipeline instance updated over time

### Grain
The level of detail in a fact table. Define before design: "one row per order line item".

---

## Data Quality

### Dimensions of Quality
- **Completeness**: no missing required fields
- **Consistency**: same values across systems
- **Accuracy**: values match reality
- **Timeliness**: data is current
- **Uniqueness**: no unexpected duplicates
- **Validity**: values conform to domain rules

### Validation Approaches
- Schema validation: data types, nullability, constraints
- Range checks: salary > 0, date within expected range
- Referential integrity: FK exists in dim table
- Statistical checks: column mean/std within expected range (detect upstream changes)
- Great Expectations: Python framework for data quality assertions

---

## Pipeline Orchestration

### Apache Airflow
DAG (Directed Acyclic Graph) based workflow orchestration.
- **DAG**: defines tasks and dependencies
- **Operator**: unit of work (PythonOperator, BashOperator, SQLOperator)
- **Task instance**: one execution of an operator
- **Scheduler**: triggers DAGs based on schedule or external events
- **XCom**: pass small data between tasks

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('etl_pipeline', schedule_interval='@daily') as dag:
    extract = PythonOperator(task_id='extract', python_callable=extract_fn)
    transform = PythonOperator(task_id='transform', python_callable=transform_fn)
    load = PythonOperator(task_id='load', python_callable=load_fn)
    extract >> transform >> load
```

### Other Orchestrators
- **Prefect**: more Pythonic; easy local dev; dynamic workflows
- **Dagster**: asset-based; data lineage built-in
- **dbt**: SQL transformation layer; version control for SQL; automated testing

---

## Data Formats

### Row-oriented (OLTP)
- **CSV**: universal; no schema; slow for analytics
- **JSON/JSONL**: flexible schema; common for APIs; verbose
- **Avro**: row-based binary; schema evolution; good for streaming (Kafka)

### Columnar (OLAP)
- **Parquet**: columnar; compressed; predicate pushdown; used in your Netflix project (S3)
- **ORC**: columnar; optimized for Hive/Spark
- **Delta Lake**: Parquet + transaction log; ACID on data lakes

**Why columnar is faster for analytics**: query only relevant columns; compression is better (same-type values); vectorized reads.

---

## Postgres Integration (Your MTech Ventures Work)

### Salesforce → Postgres Pipeline
1. Extract Salesforce submissions via SOQL/REST API
2. Transform: normalize, flatten nested JSON, deduplicate
3. Load to Postgres via COPY or psycopg2 batch insert
4. Index on frequently queried columns
5. pgvector for embedding storage + ANN retrieval

### Postgres Performance
```sql
-- Batch inserts are 100x faster than row-by-row
COPY table_name FROM '/path/to/file.csv' CSV HEADER;

-- Use EXPLAIN ANALYZE to profile
EXPLAIN ANALYZE SELECT * FROM submissions WHERE analyst_id = 5;

-- Index for hybrid retrieval
CREATE INDEX idx_gin ON documents USING GIN(to_tsvector('english', content));
CREATE INDEX idx_embedding ON documents USING ivfflat(embedding vector_cosine_ops);
```

---

## AWS Data Engineering Services (Your Resume)

| Service | Purpose |
|---|---|
| S3 | Object storage; data lake; Parquet files |
| Glue | Serverless ETL; PySpark jobs; Catalog |
| Lambda | Event-driven transformation; lightweight ETL |
| Kinesis | Real-time data streaming |
| Redshift | Cloud data warehouse; columnar; SQL analytics |
| RDS | Managed relational DB (Postgres, MySQL) |
| DMS | Database Migration Service |
| Step Functions | Workflow orchestration for AWS services |
| Athena | Serverless SQL on S3 (Presto-based) |

---

## Salesforce Data Engineering (Your Resume)

### SOQL (Salesforce Object Query Language)
```sql
SELECT Id, Name, Amount, CloseDate
FROM Opportunity
WHERE CloseDate = THIS_QUARTER
AND StageName = 'Closed Won'
ORDER BY Amount DESC
LIMIT 100
```

### Salesforce to Postgres Pattern
- Use simple_salesforce Python library or Salesforce REST API
- Bulk API 2.0 for large data exports (>100K records)
- Handle Salesforce's SOQL limits: max 50K records per batch

### Dashboards (Your 10+ Salesforce Dashboards)
- Report types: tabular, summary, matrix
- Use formula fields for derived metrics
- Scheduled reports via workflow rules / Flow
- Einstein Analytics for advanced visualization

---

## Web Scraping Pipeline (Your 8K+ Records)

### Tools
- **BeautifulSoup**: HTML parsing; static content
- **Scrapy**: full framework; async; built-in pipelines
- **Playwright / Selenium**: headless browser; JavaScript-rendered pages
- **requests + lxml**: fast for known page structures

### Pipeline Design
1. URL queue management (avoid duplicates; prioritize)
2. Polite scraping: respect robots.txt, rate limit requests
3. Retry logic: exponential backoff on 429/5xx errors
4. Data validation: check expected fields present
5. Storage: raw → structured → database
6. Incremental: only scrape new/changed pages

---

---

## AWS Modern Data Architecture

### Data Warehouse & Lakehouse on AWS

#### Amazon Redshift
**Purpose**: Cloud-native data warehouse; columnar; optimized for OLAP queries

**Key Concepts**:
- **Distribution Keys**: how rows are distributed across nodes
  - EVEN: round-robin; use when no natural join key
  - KEY: distribute by fact table FK to co-locate joins
  - ALL: replicate small dims to all nodes
- **Sort Keys**: physical order on disk; speeds range queries and window functions
  - Compound: multiple columns (order matters)
  - Interleaved: equal weight for all columns (rare)
- **Workload Management (WLM)**: prioritize queries; concurrency slots; QOS
- **Compression**: automatic; ENCODE directive for specific columns
- **Vacuum**: reclaim space after deletes; consolidate rows
- **Analyze**: update statistics for query optimizer

**Performance Tuning**:
```sql
-- Check table distribution skew
SELECT
  schemaname, tablename, unsorted, redshift_state_reason
FROM pg_class_info
WHERE schemaname = 'analytics'
ORDER BY unsorted DESC;

-- Set sort key on fact table
ALTER TABLE fact_orders SORTKEY (order_date, customer_id);

-- Monitor slow queries
SELECT * FROM stl_query WHERE query_duration > 60000 ORDER BY query_duration DESC;

-- Explain plan
EXPLAIN SELECT * FROM fact_orders WHERE order_date > '2024-01-01';
```

**Best Practices**:
- Use DISTKEY on fact table FK; SORTKEY on fact date + dimensions
- Vacuum regularly on large tables (post-ETL)
- Compress textual columns (ENCODE ZSTD)
- Leverage Redshift Spectrum for S3 data (external tables)
- Use UNLOAD to S3 for backup and data sharing

#### AWS Glue
**Purpose**: Serverless ETL service; runs Spark jobs; integrated Catalog

**Components**:
- **Glue Jobs**: Spark (PySpark) or Scala ETL scripts; auto-scaling
- **Glue Catalog**: centralized metadata; integrates with Athena, Redshift, EMR
- **Glue Crawlers**: auto-discover schema from data sources (S3, JDBC, etc.)
- **Data Quality**: DPU-based monitoring; anomaly detection

**Common Pattern**:
```python
import sys
from awsglue.transforms import *
from awsglue.job import Job
from awsglue.context import GlueContext
from pyspark.sql.functions import *

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Read from Catalog
input_dyf = glueContext.create_dynamic_frame.from_catalog(
    database="raw",
    table_name="source_table"
)

# Transform
transformed = input_dyf.toDF() \
    .filter(col('is_active') == True) \
    .groupBy('customer_id').agg(sum('amount').alias('total'))

# Write to Redshift
glueContext.write_dynamic_frame.from_options(
    frame=transformed.toDF(),
    connection_type="redshift",
    connection_options={
        "url": "jdbc:redshift://cluster:5439/db",
        "user": "admin",
        "password": "...",
        "dbtable": "analytics.customer_summary",
        "tempdir": "s3://bucket/glue-temp/"
    }
)

job.commit()
```

#### Amazon Athena
**Purpose**: Serverless SQL on S3; Presto-based; query data lake directly

**Use Cases**:
- Ad-hoc analytics on S3 Parquet/CSV without loading to warehouse
- Partner with Glue Catalog for schema management
- Partition pruning: query only relevant S3 prefixes

```sql
-- Query S3 directly with partition pruning
SELECT customer_id, SUM(amount) AS revenue
FROM s3_data.events_parquet
WHERE year = 2024 AND month = 7
GROUP BY customer_id;
```

#### AWS Lambda
**Purpose**: Serverless compute; event-driven transformation; lightweight ETL

**Data Engineering Use**:
- Trigger on S3 PUT (new file) → transform → load
- Orchestrate Glue jobs via EventBridge
- Batch API calls with time/concurrency limits

```python
import json, boto3, pandas as pd

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    
    # Parse S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Download, transform, upload
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    df['processed'] = df['raw_value'] * 2
    
    s3.put_object(
        Bucket=bucket,
        Key=f"processed/{key}",
        Body=df.to_csv(index=False)
    )
    
    return {'statusCode': 200}
```

#### AWS Step Functions
**Purpose**: Serverless workflow orchestration; visual workflows; exception handling

**vs. Airflow**: simpler; AWS-native; no infra management; slower for high-concurrency pipelines

```json
{
  "StartAt": "ExtractTask",
  "States": {
    "ExtractTask": {
      "Type": "Task",
      "Resource": "arn:aws:states:::glue:startJobRun",
      "Parameters": {"JobName": "extract-job"},
      "Next": "TransformTask"
    },
    "TransformTask": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Next": "LoadTask"
    },
    "LoadTask": {
      "Type": "Task",
      "Resource": "arn:aws:states:::redshift:statement",
      "End": true
    }
  }
}
```

#### Other AWS Services
- **EMR (Elastic MapReduce)**: Spark/Hadoop clusters; on-demand or spot instances; cost-effective for batch processing
- **Kinesis**: real-time streaming ingestion; Kinesis Data Streams (Kafka-like), Kinesis Firehose (deliver to S3/Redshift)
- **EventBridge**: event routing; trigger Lambda/Step Functions on data events
- **IAM Roles & Policies**: fine-grained access control; cross-account access

---

## Reliable Data Ingestion from Diverse Sources

### Relational Databases (Oracle, PostgreSQL, MySQL)

#### JDBC Connection
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("JdbcIngest").getOrCreate()

df = spark.read.format("jdbc").options(
    url="jdbc:oracle:thin:@hostname:1521:DBNAME",
    dbtable="source_schema.source_table",
    user="username",
    password="password",
    numPartitions=10,  # parallel read
    partitionColumn="id",
    lowerBound=1,
    upperBound=1000000
).load()
```

#### Change Data Capture (CDC)
**Without CDC** (timestamp-based):
```sql
SELECT * FROM source WHERE modified_dt > :last_run_ts
```
Problems: deletes not captured; requires reliable updated_at; no guarantees if clocks skew

**With CDC** (database transaction log):
- **Oracle**: LogMiner, Oracle GoldenGate
- **Postgres**: WAL (Write-Ahead Logging), logical replication slots
- **MySQL**: binlog, Debezium
- Captures all changes (inserts, updates, deletes) in order; near real-time

**Debezium Pattern** (open-source CDC):
```
Postgres WAL → Kafka → Kafka Connect Sink → S3/Warehouse
```

#### Handling Non-Clean Primary Keys
- Tables without PK: use MD5(all columns) as surrogate key
- Composite keys: concatenate with separator
- Duplicate detection: ROW_NUMBER() OVER (PARTITION BY key ORDER BY extracted_at DESC)

### SaaS Platforms (Salesforce, etc.)

#### REST API Extraction
```python
import requests, time
from typing import List, Dict

class SalesforceExtractor:
    def __init__(self, instance_url, client_id, client_secret):
        self.instance_url = instance_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = self._get_token()
    
    def _get_token(self):
        """OAuth 2.0 password flow"""
        resp = requests.post(
            f"{self.instance_url}/services/oauth2/token",
            data={
                'grant_type': 'password',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'username': 'user@example.com',
                'password': 'password+security_token'
            }
        )
        return resp.json()['access_token']
    
    def query_soql(self, soql: str) -> List[Dict]:
        """Execute SOQL with pagination and rate limit handling"""
        records = []
        url = f"{self.instance_url}/services/data/v59.0/query"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        params = {'q': soql}
        
        while url:
            resp = requests.get(url, headers=headers, params=params)
            
            if resp.status_code == 429:  # Rate limit
                time.sleep(int(resp.headers.get('Retry-After', 60)))
                continue
            
            resp.raise_for_status()
            data = resp.json()
            records.extend(data['records'])
            
            # Handle pagination
            url = data.get('nextRecordsUrl')
            params = {}  # pagination uses URL directly
        
        return records
```

#### Schema Evolution Handling
- Store ingestion timestamp
- Track schema changes (add columns as NULL)
- Use schemas with optional fields (Avro evolution rules)

### Webhooks

#### Idempotent Webhook Processing
```python
from datetime import datetime
import hashlib

def process_webhook(payload):
    # Generate idempotency key
    idempotency_key = hashlib.md5(
        (payload['event_id'] + str(payload['timestamp'])).encode()
    ).hexdigest()
    
    # Check if already processed
    if redis.exists(f"webhook:{idempotency_key}"):
        return {"status": "already_processed"}
    
    # Process
    # ... insert/update logic ...
    
    # Mark as processed
    redis.setex(f"webhook:{idempotency_key}", 86400, "1")  # 24hr expiry
    
    return {"status": "processed"}
```

---

## Advanced Python for Production Data Work

### Pandas & Data Manipulation
```python
import pandas as pd
import numpy as np

# Efficient reading (dtypes, usecols)
df = pd.read_csv('large.csv', dtype={'id': 'int64', 'amount': 'float32'}, 
                  usecols=['id', 'amount', 'date'])

# Vectorized operations (faster than apply)
df['amount_usd'] = df['amount'] * df['exchange_rate']  # element-wise

# Group and aggregate efficiently
summary = df.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'count'],
    'date': 'max'
}).reset_index()

# Deduplication
df_unique = df.sort_values('timestamp').drop_duplicates(subset=['key'], keep='last')
```

### boto3 & AWS Integration
```python
import boto3
from concurrent.futures import ThreadPoolExecutor

s3 = boto3.client('s3')

# Parallel S3 uploads
def upload_file(bucket, key, data):
    s3.put_object(Bucket=bucket, Key=key, Body=data)

with ThreadPoolExecutor(max_workers=10) as executor:
    for file in files:
        executor.submit(upload_file, bucket, key, data)

# S3 batch operations
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket, Prefix='data/'):
    for obj in page['Contents']:
        print(obj['Key'])
```

### SQLAlchemy & ORM
```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine('postgresql://user:pwd@localhost/db')
Session = sessionmaker(bind=engine)
session = Session()

# Bulk insert (faster than row-by-row)
customers = [Customer(name=f'Customer {i}') for i in range(1000)]
session.bulk_insert_mappings(Customer, customers)
session.commit()

# Query with relationship
customer = session.query(Customer).filter(Customer.id == 1).first()
```

### asyncio & Concurrent API Calls
```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as resp:
        return await resp.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Run
results = asyncio.run(fetch_all(['url1', 'url2', 'url3']))
```

### Testing Data Pipelines
```python
import pytest
from unittest.mock import patch, MagicMock

def test_extract_salesforce():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            'records': [{'Id': '1', 'Name': 'Test'}],
            'nextRecordsUrl': None
        }
        
        records = extract_salesforce()
        assert len(records) == 1
        assert records[0]['Id'] == '1'

def test_transform_deduplication():
    input_df = pd.DataFrame({
        'id': [1, 1, 2],
        'value': [10, 20, 30]
    })
    
    result = deduplicate(input_df, 'id')
    assert len(result) == 2
```

---

## Advanced SQL & Query Performance

### Window Functions
```sql
-- Ranking within groups
SELECT
  customer_id, order_date, amount,
  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS order_num,
  RANK() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS amount_rank
FROM orders;

-- Running totals
SELECT
  customer_id, order_date, amount,
  SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_total
FROM orders;

-- Lead/Lag for trend detection
SELECT
  customer_id, month, revenue,
  LAG(revenue) OVER (PARTITION BY customer_id ORDER BY month) AS prev_month_revenue,
  ROUND((revenue - LAG(revenue) OVER (...)) / LAG(revenue) OVER (...) * 100, 2) AS mom_growth_pct
FROM monthly_revenue;
```

### CTEs & Complex Queries
```sql
-- Multi-step logic with CTEs
WITH customer_metrics AS (
    SELECT
      customer_id,
      COUNT(*) AS order_count,
      SUM(amount) AS lifetime_value,
      MAX(order_date) AS last_order_date
    FROM orders
    GROUP BY customer_id
),
churned_customers AS (
    SELECT * FROM customer_metrics
    WHERE last_order_date < DATE_SUB(NOW(), INTERVAL 90 DAY)
)
SELECT * FROM churned_customers WHERE lifetime_value > 1000;
```

### Performance Optimization
```sql
-- Index foreign keys and sort/group columns
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);

-- Partition large fact tables (e.g., by date)
-- In Redshift:
DISTKEY (customer_id) SORTKEY (order_date)

-- Use EXPLAIN ANALYZE to identify bottlenecks
EXPLAIN ANALYZE
SELECT * FROM orders WHERE order_date > '2024-01-01';

-- Avoid full table scans; leverage partition pruning
SELECT * FROM orders WHERE year = 2024 AND month = 7;  -- partition columns
```

---

## dbt for Transformation & Documentation

### Core Concepts
- **Models**: SQL files that define tables/views; version-controlled; auto-tested
- **Tests**: assertions on data (not null, unique, relationships, custom SQL)
- **Sources**: external data (raw warehouse tables); freshness checks
- **Snapshots**: SCD Type 2 tables for dimension history

### Example dbt Project
```yaml
# dbt_project.yml
name: 'analytics'
version: '1.0.0'
profile: 'redshift'

models:
  analytics:
    materialized: table
    staging:
      materialized: view
```

```sql
-- models/staging/stg_orders.sql
{{ config(
    materialized='view'
) }}

SELECT
  id,
  customer_id,
  order_date,
  amount,
  CAST(order_date AS DATE) AS order_day
FROM {{ source('raw', 'orders') }}
WHERE order_date >= '2020-01-01'
```

```sql
-- models/marts/fct_orders.sql
{{ config(
    materialized='table',
    sort=['order_date'],
    dist=['customer_id']
) }}

SELECT
  stg.id,
  stg.customer_id,
  stg.order_date,
  stg.amount,
  dim_customer.segment
FROM {{ ref('stg_orders') }} stg
LEFT JOIN {{ ref('dim_customer') }} dim_customer
  ON stg.customer_id = dim_customer.customer_id
```

```yaml
# models/schema.yml
version: 2
models:
  - name: fct_orders
    columns:
      - name: id
        tests:
          - not_null
          - unique
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customer')
              field: customer_id

sources:
  - name: raw
    tables:
      - name: orders
        freshness:
          warn_after: {count: 24, period: hour}
        loaded_at_field: created_at
```

### dbt Testing & Documentation
```bash
# Run tests
dbt test

# Generate docs
dbt docs generate
dbt docs serve

# CI/CD: only test changed models
dbt test --select state:modified+
```

---

## Dimensional & Lakehouse Modeling

### Dimensional Model Design
1. **Choose grain**: one row per order line item, customer per day, etc.
2. **Identify facts**: numeric measures (amount, quantity)
3. **Identify dimensions**: customer, product, date, location
4. **Handle slowly changing dimensions**: Type 2 for customer address, Type 1 for product name

```sql
-- Star schema: orders fact table + dimensions
CREATE TABLE fct_orders (
    order_id INT,
    customer_id INT,
    product_id INT,
    date_id INT,
    amount DECIMAL,
    quantity INT,
    FOREIGN KEY (customer_id) REFERENCES dim_customer,
    FOREIGN KEY (product_id) REFERENCES dim_product,
    FOREIGN KEY (date_id) REFERENCES dim_date
);

-- Dimension with SCD Type 2
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id INT,
    name VARCHAR,
    segment VARCHAR,
    effective_date DATE,
    end_date DATE,
    is_current BOOLEAN
);
```

### Lakehouse Architecture
**Medallion Architecture** (Bronze → Silver → Gold):
- **Bronze**: raw data; minimal transformation; no PII redaction; versioned
- **Silver**: cleaned, deduplicated, business-ready; conformed schemas
- **Gold**: aggregated, business logic applied; optimized for consumption

```
S3 Bronze Layer (raw Parquet)
  ↓ (dbt + Glue)
S3 Silver Layer (cleaned Parquet with partitions)
  ↓ (dbt)
Redshift Gold Layer (marts, ready for BI)
```

---

## REST & GraphQL API Integration

### REST API Patterns
```python
import requests
from datetime import datetime, timedelta
from typing import Iterator, Dict

class APIExtractor:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
        self.rate_limit_remaining = None
    
    def extract_with_pagination(self, endpoint: str) -> Iterator[Dict]:
        """Handle pagination, rate limits, retries"""
        page = 1
        while True:
            try:
                resp = requests.get(
                    f"{self.base_url}/{endpoint}",
                    headers=self.headers,
                    params={'page': page, 'per_page': 100},
                    timeout=30
                )
                
                # Rate limit handling
                if 'X-RateLimit-Remaining' in resp.headers:
                    self.rate_limit_remaining = int(resp.headers['X-RateLimit-Remaining'])
                    if self.rate_limit_remaining < 10:
                        time.sleep(60)  # Back off
                
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    break
                
                for record in data:
                    yield record
                
                page += 1
                
            except requests.exceptions.RequestException as e:
                if resp.status_code == 429:
                    time.sleep(int(resp.headers.get('Retry-After', 60)))
                else:
                    raise
```

### GraphQL Extraction
```python
import requests
import json

class GraphQLExtractor:
    def __init__(self, graphql_url, api_key):
        self.url = graphql_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
    
    def query(self, query: str, variables: dict = None) -> dict:
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        resp = requests.post(self.url, json=payload, headers=self.headers)
        resp.raise_for_status()
        
        result = resp.json()
        if 'errors' in result:
            raise Exception(f"GraphQL error: {result['errors']}")
        
        return result['data']
    
    def extract_all_with_cursor(self, query_template: str):
        """Pagination using cursor"""
        cursor = None
        all_data = []
        
        while True:
            query = query_template.replace('$cursor', f'"{cursor}"' if cursor else 'null')
            result = self.query(query)
            
            all_data.extend(result['nodes'])
            
            if result.get('pageInfo', {}).get('hasNextPage'):
                cursor = result['pageInfo']['endCursor']
            else:
                break
        
        return all_data
```

### Schema Evolution Handling
```python
def handle_schema_drift(df, schema_registry):
    """Track and adapt to schema changes"""
    current_schema = df.schema
    registered_schema = schema_registry.get_latest('topic')
    
    # Add missing fields with default values
    for field in registered_schema:
        if field not in current_schema:
            df = df.withColumn(field, lit(None))
    
    # Remove extra fields
    df = df.select([f for f in registered_schema if f in df.columns])
    
    return df
```

---

## Orchestration Deep Dive

### Apache Airflow Advanced Patterns
```python
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'data-engineering',
    'start_date': days_ago(1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=2)  # fail if not done in 2h
}

@dag(default_args=default_args, schedule_interval='@daily', tags=['etl'])
def data_pipeline():
    
    @task
    def extract():
        # Custom Python logic
        return {'records': 1000}
    
    @task
    def transform(extracted_data):
        return {'transformed': True}
    
    glue_task = GlueJobOperator(
        task_id='run_glue_job',
        job_name='my-glue-job'
    )
    
    # Set dependencies
    extract() >> transform() >> glue_task

# Instantiate DAG
etl_dag = data_pipeline()
```

### Prefect for Dynamic Workflows
```python
from prefect import flow, task
from prefect.tasks.shell import shell_run_command

@task(retries=2, retry_delay_seconds=60)
def extract_data():
    # More Pythonic; less boilerplate than Airflow
    return {"records": 1000}

@task
def transform_data(data):
    return {"transformed": True}

@flow(name="dynamic_pipeline")
def my_flow():
    data = extract_data()
    result = transform_data(data)
    return result

# Run
if __name__ == "__main__":
    my_flow()
```

---

## Data Quality & Governance

### Great Expectations
```python
from great_expectations.dataset import PandasDataset

def validate_data_quality(df):
    dataset = PandasDataset(df)
    
    # Null check
    assert dataset.expect_column_values_to_not_be_null('customer_id')['result']['element_count'] == len(df)
    
    # Unique check
    assert dataset.expect_column_values_to_be_unique('email')['result']['success']
    
    # Range check
    assert dataset.expect_column_values_to_be_between('age', min_value=0, max_value=150)['result']['success']
    
    # Custom SQL assertion
    assert dataset.expect_queried_table_row_count_to_equal_other_table_row_count('orders', 'expected_orders')
    
    return True
```

### Data Lineage Tracking
```python
# OpenMetadata / DataHub integration
def log_lineage(source_table, target_table, transformation_logic):
    """Track data lineage for governance"""
    lineage_record = {
        'source': source_table,
        'target': target_table,
        'logic': transformation_logic,
        'timestamp': datetime.utcnow(),
        'user': os.environ['USER']
    }
    
    # Send to lineage system
    lineage_service.record(lineage_record)
```

### Access Control & SOC 2 Compliance
- **Row-Level Security (RLS)**: filter by user/tenant in Redshift
- **Column-Level Security**: encrypt sensitive columns; decrypt on read
- **Data Masking**: PII redaction in Silver layer
- **Audit Logging**: track who accessed what, when

```sql
-- Row-level security in Redshift
CREATE RLS POLICY tenant_policy ON fact_orders
FOR SELECT TO role_analyst
WITH (tenant_id = current_user_id());

-- Column masking for PII
ALTER TABLE dim_customer
ALTER COLUMN ssn
SET DEFAULT (CASE WHEN current_user = 'admin' THEN ssn ELSE '***-**-****' END);
```

### Anomaly Detection
```python
def detect_anomalies(df, metric_column, lookback_days=30):
    """Statistical anomaly detection"""
    historical = df[df['date'] < datetime.utcnow() - timedelta(days=1)]
    historical_mean = historical[metric_column].mean()
    historical_std = historical[metric_column].std()
    
    # Flag values > 3 standard deviations
    df['is_anomaly'] = np.abs(df[metric_column] - historical_mean) > 3 * historical_std
    
    return df[df['is_anomaly']]
```

---

## Infrastructure as Code (Terraform)

### Redshift Cluster with IaC
```hcl
resource "aws_redshift_cluster" "analytics" {
  cluster_identifier        = "analytics-prod"
  engine_version            = "1.0.47980"
  node_type                 = "ra3.xlplus"
  number_of_nodes           = 3
  master_username           = "admin"
  master_password           = random_password.redshift_password.result
  database_name             = "analytics"
  publicly_accessible       = false
  
  subnet_group_name         = aws_redshift_subnet_group.main.name
  vpc_security_group_ids    = [aws_security_group.redshift.id]
  
  logging {
    enable = true
    log_destination_type = "cloudwatch"
  }
}

resource "aws_redshift_parameter_group" "main" {
  family = "redshift-1.0"
  
  parameter {
    name  = "wlm_json_configuration"
    value = jsonencode([
      {
        priority = 100
        queue_type = "default_queue"
      }
    ])
  }
}
```

### S3 Data Lake with Lifecycle Policies
```hcl
resource "aws_s3_bucket" "data_lake" {
  bucket = "company-data-lake-prod"
}

resource "aws_s3_bucket_lifecycle_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  rule {
    id = "bronze_to_glacier"
    filter {
      prefix = "bronze/"
    }
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}
```

---

## Monitoring, Alerting & Observability

### Key Metrics to Monitor
- **Pipeline health**: run duration, success rate, late arrivals
- **Data quality**: row counts, null %, distinctness, freshness
- **Warehouse health**: query latency, concurrency, WLM queue times
- **Cost**: compute cost per GB, per query

### CloudWatch Dashboards & Alarms
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Custom metric
cloudwatch.put_metric_data(
    Namespace='DataPipelines',
    MetricData=[
        {
            'MetricName': 'RowsExtracted',
            'Value': 1000000,
            'Unit': 'Count'
        }
    ]
)

# Alarm on metric
cloudwatch.put_metric_alarm(
    AlarmName='glue_job_failure',
    MetricName='glue_job_duration',
    Namespace='AWS/Glue',
    Threshold=3600,  # seconds
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=['arn:aws:sns:us-east-1:123456789:alert-topic']
)
```

---

## Key Interview Points

- **ETL vs ELT trade-off**: ETL: more control over transformation, no raw PII in warehouse; ELT: simpler pipelines, leverage warehouse compute, raw always available for reprocessing
- **How do you handle late-arriving data?** Watermarks in streaming; reprocessing windows; SCD Type 2 date correction
- **How to optimize a slow ETL query?** EXPLAIN PLAN; add indexes; partition pruning; reduce joins; use bulk/batch operations; columnar storage
- **What is CDC and why is it better than timestamp?** Captures all changes (including deletes) from transaction log; no dependency on application updating timestamps; near real-time
- **How do you ensure data quality in pipelines?** Schema validation at ingestion; business rule assertions (Great Expectations); anomaly detection on stats; alerting; idempotent reprocessing
- **What makes a pipeline idempotent?** Running it multiple times produces same result. Use MERGE/upsert instead of INSERT; clear/reload partitions atomically.

---

## Advanced Interview Questions

### Architecture & Design
- **Design a data pipeline for a SaaS company with 50M daily events**: Kinesis → Lambda → S3 (Bronze) → Glue (dedupe) → S3 (Silver) → Redshift (Gold); scale Lambda concurrency for burst load
- **How would you handle CDC from Oracle into Redshift?** LogMiner or GoldenGate → Kafka → Glue CDC consumer → S3 (Bronze) → Redshift
- **Explain a lakehouse architecture you've built**: Bronze (raw Parquet, daily partitions), Silver (cleaned, deduplicated, conformed schemas), Gold (marts optimized for BI)

### Data Quality & Reliability
- **Walk us through your data quality framework**: Great Expectations at ingestion; anomaly detection on metrics; lineage tracking; alerting on SLA misses
- **A critical dimension loads with 50% nulls. What's your process?** Check source; validate extraction logic; pause dependent pipelines; run impact analysis; communicate to stakeholders
- **How do you make pipelines resilient to upstream schema changes?** Whitelist expected columns; validate schema at ingestion; implement optional field handling; version schemas

### Performance Tuning
- **Redshift query is slow. Walk me through diagnosis**: EXPLAIN ANALYZE; check distribution skew (SELECT unsorted FROM pg_class_info); rebuild sort keys; check WLM; consider Spectrum for large scans
- **How do you optimize large Glue jobs?** Increase DPU; use AWS Glue job bookmarks for incremental; repartition data; cache intermediate DataFrames
- **Optimize a 2TB S3 data lake scan in Athena**: Use partition pruning; convert to Parquet; set optimal partition sizes (100-500MB); use Glue Catalog for schema inference

### Compliance & Multi-Tenancy
- **Design a secure data delivery system for external clients**: Curate datasets per client; use IAM roles + S3 bucket policies for access; encrypt at rest & in transit; audit logging; data contracts for schema stability
- **How would you handle GDPR right-to-be-forgotten in a data lake?** Mark deleted records in Silver; regenerate Gold datasets; maintain version history for audit; automate in Airflow
- **Implement row-level security for a shared analytics warehouse**: RLS policy per tenant in Redshift; filter in dbt models; test access matrix; audit who accessed what

### Incident & Troubleshooting
- **Walk through your process when a pipeline fails**: Check logs; retry with backoff; check upstream dependencies; validate data quality of last successful run; run compensating transaction if needed
- **A data warehouse query is blocking other queries. Diagnose**: Check PG_STAT_ACTIVITY; identify long-running query; cancel if safe; check workload management rules; add indexes for frequent queries
- **A dbt snapshot incorrectly captured history. How do you recover?** Restore from S3 backup; backfill snapshot with correct logic; regenerate dependent models; validate row counts match expected

### Cost Optimization
- **How would you reduce data warehouse costs by 30%?** Right-size cluster (RA3 cheaper than DC2 for mixed workload); compress columns aggressively; use Spectrum for cold data; optimize query patterns; consolidate redundant datasets
- **Lambda cost is exploding for API polling. How to fix?** Use Kinesis/EventBridge push instead of Lambda poll; batch API calls; increase batch size to reduce invocations; consider switching to managed connector (Fivetran)

---

## Real-World Scenarios

### Scenario 1: Building a Data Platform from Scratch
**Given**: Company with 10 SaaS sources, 100M records/day, 50 analytics users, 6-month runway

**Answer**: 
1. Start with managed connectors (Fivetran) for standard SaaS sources; custom Glue jobs for internal APIs
2. S3 Bronze layer (Parquet, daily partitions) for raw ingestion
3. dbt in S3 Silver for deduplication, schema normalization
4. Redshift Gold layer with dimensional model; optimize with DISTKEY/SORTKEY
5. Airflow for orchestration; test all dbt models; alerts on SLA misses
6. QuickSight for BI; row-level security for multi-tenant
7. Cost: ~$20K/month for Redshift RA3 (3 nodes) + Fivetran credits + Lambda/Glue

### Scenario 2: Debugging a Production Issue
**Given**: Customer revenue metric down 40% overnight; it was correct yesterday

**Process**:
1. Check Redshift queries: are joins still working? Dimension values changed?
2. Verify dbt test results: run `dbt test` on Gold layer
3. Check freshness: when did raw data last load? Glue job failures?
4. Inspect Bronze data: did source system change? API schema drift?
5. If found: hotfix in dbt, run full refresh, backfill metrics, notify stakeholders
6. Post-incident: add dbt test for revenue > previous day (flag anomalies)

### Scenario 3: Adding a New Data Source (Salesforce)
**Approach**:
1. Evaluate: Fivetran vs. custom Glue job (volume, frequency, complexity, cost)
2. Extract: Bulk API 2.0 for Opportunities & Accounts; handle 50K record batches
3. Land: S3 Bronze as JSONL or Parquet with ingestion_timestamp
4. Deduplicate: In dbt; use Salesforce Id as PK; handle soft deletes
5. Transform: Flatten nested objects; join to existing dims (customer); test
6. Load: UPSERT to Redshift Gold; update metrics
7. Monitor: Freshness check; row count validation; lineage in metadata store

---

## Commonly Used Tools & Technologies

| Category | Tools |
|---|---|
| **Extraction** | Fivetran, Airbyte, Kafka, Kinesis, APIs, CDC (Debezium) |
| **Transformation** | dbt, Spark (Glue/EMR), SQL, Python |
| **Storage** | S3, Redshift, Snowflake, BigQuery, Delta Lake |
| **Orchestration** | Airflow, Prefect, Step Functions, Dagster |
| **Quality** | Great Expectations, dbt tests, custom SQL assertions |
| **Monitoring** | CloudWatch, DataDog, custom metrics |
| **IaC** | Terraform, CloudFormation |
| **VCS** | Git (dbt, Airflow DAGs, Glue scripts) |
