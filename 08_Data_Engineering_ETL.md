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

## Key Interview Points

- **ETL vs ELT trade-off**: ETL: more control over transformation, no raw PII in warehouse; ELT: simpler pipelines, leverage warehouse compute, raw always available for reprocessing
- **How do you handle late-arriving data?** Watermarks in streaming; reprocessing windows; SCD Type 2 date correction
- **How to optimize a slow ETL query?** EXPLAIN PLAN; add indexes; partition pruning; reduce joins; use bulk/batch operations; columnar storage
- **What is CDC and why is it better than timestamp?** Captures all changes (including deletes) from transaction log; no dependency on application updating timestamps; near real-time
- **How do you ensure data quality in pipelines?** Schema validation at ingestion; business rule assertions (Great Expectations); anomaly detection on stats; alerting; idempotent reprocessing
- **What makes a pipeline idempotent?** Running it multiple times produces same result. Use MERGE/upsert instead of INSERT; clear/reload partitions atomically.
