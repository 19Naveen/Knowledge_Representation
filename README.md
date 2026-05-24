# 🧠 Knowledge Representation Platform (Revised Architecture)

An **AI-powered lakehouse data platform** that enables users to ingest, transform, query, and model data at scale—without managing infrastructure or risking data integrity.

---

# 🚀 Core Philosophy

> **Separate Storage, Compute, and Control Planes**

This platform is built on three strict principles:

1. **Object Storage is the source of truth (Parquet)**
2. **Compute is stateless (DuckDB / Spark)**
3. **Relational DB stores only metadata, not datasets**

---

# 🏗️ System Architecture

```text
                    ┌────────────────────┐
                    │     Frontend       │
                    │ Dashboards / UI    │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │      FastAPI       │
                    │  Control Plane     │
                    └─────────┬──────────┘
                              ↓
        ┌──────────────┬──────────────┬──────────────┐
        ↓              ↓              ↓

Object Storage     Metadata DB       Compute Layer
(Parquet Files)    (PostgreSQL)      (DuckDB/Spark)
```

---

# 🛠️ Core Modules

## 1. Data Ingestion (`Data Import`)

### Supported Sources

* CSV / Excel uploads
* Parquet files
* Database connections (PostgreSQL, Snowflake, etc.)

### Behavior

* All data is ingested as **read-only**
* Immediately stored in **object storage (Parquet format)**

### Storage Structure

```text
/user_id/dataset_name/
    raw/
    processed/
    features/
```

### Large Database Handling

* Read Replica (preferred)
* CDC pipelines (Airbyte/Debezium)
* No direct querying on production DB

---

## 2. Storage Layer (Lakehouse)

### Format

* **Parquet (columnar, compressed, partitioned)**

### Zones

```text
raw/        → immutable source data
processed/  → cleaned datasets (versioned)
features/   → ML-ready datasets
```

### Key Properties

* Immutable raw layer
* Versioned transformations
* Partitioned for performance

---

## 3. Compute Layer (`Data Transform`)

### Engines (based on scale)

| Data Size | Engine |
| --------- | ------ |
| <1GB      | Pandas |
| 1–10GB    | DuckDB |
| 10GB+     | Spark  |

### Execution Pattern

```text
Read (Parquet) → Transform → Write (Parquet)
```

### Guarantees

* No mutation of raw data
* All transformations produce new versions

---

## 4. Query Layer (`Query Studio`)

### Capabilities

* SQL-based querying on Parquet
* Natural Language → SQL generation
* Schema-aware execution

### Execution Engine

* DuckDB (default)
* Trino (future scaling)

### Example Flow

```text
User Query → SQL → DuckDB → Parquet → Result → JSON → UI
```

---

## 5. Dashboard & Serving Layer

### Strategy

#### Direct Query Mode

* Query Parquet using DuckDB
* Suitable for ad-hoc analytics

#### Optimized Mode

* Pre-aggregate datasets
* Store in OLAP DB (optional)

```text
Raw (50GB) → Aggregate → 50MB → Fast dashboards
```

### Caching

* Redis for query caching
* Avoid repeated scans

---

## 6. Machine Learning Layer

### Components

* AutoML pipeline
* Manual training (XGBoost, sklearn)
* Feature dataset generation

### Data Source

* Processed Parquet datasets
* NOT in-memory CSVs

### Execution Modes

* Small → local training
* Large → Spark ML

---

## 7. Strategy Simulator (`Deploy Sim`)

* Scenario testing using model outputs
* No direct dependency on raw datasets
* Uses processed/aggregated data

---

# 🧾 Metadata & Control Plane

## Stored in PostgreSQL

### Entities

```text
datasets
- id
- user_id
- blob_path
- schema
- version

transformations
- dataset_id
- steps
- version

jobs
- status
- logs
- execution_time

dashboards
- config
- queries
```

---

# 🔁 Data Flow (End-to-End)

```text
1. Ingest → Blob (raw)
2. Convert → Parquet
3. Transform → processed/
4. Query → DuckDB
5. Aggregate → optional OLAP
6. Visualize → Dashboard
7. Train → ML models
```

---

# ⚠️ Critical Design Rules

## MUST FOLLOW

* Raw data is **never stored in a database**
* All datasets are stored as **Parquet in object storage**
* DB connections are **read-only**
* Transformations are **immutable + versioned**

---

## NEVER DO

* Load 50GB datasets into PostgreSQL
* Query user production DB directly
* Overwrite raw datasets
* Depend on Pandas for large data

---

# 📊 Decision Matrix

| Scenario       | Approach       |
| -------------- | -------------- |
| Small CSV      | Pandas + Blob  |
| Medium dataset | DuckDB         |
| Large dataset  | Spark          |
| DB connection  | CDC / Replica  |
| Dashboard      | DuckDB + Cache |
| Fast BI        | OLAP DB        |

---

# 🎯 Platform Identity

> **A self-serve AI-powered lakehouse platform that transforms raw data into queryable, versioned datasets and enables analytics, ML, and simulation at scale.**

---

# 🔥 What This Enables

* No data duplication
* Infinite scalability (object storage)
* Safe data handling (read-only ingestion)
* High-performance querying (columnar execution)
* Unified analytics + ML workflow

---

# ⚙️ Getting Started

### Prerequisites
- Python 3.9+ | Node.js 18+ | Google Gemini API Key

### Installation

1.  **System Setup**
    ```bash
    git clone https://github.com/19Naveen/Knowledge_Representation.git
    ```

2.  **Backend Initialization**
    ```bash
    cd backend && python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Frontend Launch**
    ```bash
    cd frontend && npm install && npm run dev
    ```

---

# 🚧 Future Enhancements

* Dataset versioning (Git-like)
* Data lineage tracking
* Cost-based query optimization
* Multi-tenant query isolation
* Distributed query engine (Trino)

---

# 🧠 Final Note

> This platform is not a database.
>
> It is a **data processing and intelligence system built on a lakehouse architecture**.

---

## 🤝 About the Team
Developed with ❤️ by **Bit Bandits**.

