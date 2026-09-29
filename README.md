# 🎓 Vir — Agentic Campus AI Assistant & Indoor Navigation

**Vir** is an autonomous hybrid AI campus assistant and indoor navigation system built for **P.T. Lee Chengalvaraya Naicker College of Engineering and Technology (PTLCNCET)**.

The system combines **Agentic Reasoning & Tool Calling**, **Semantic Document Retrieval (RAG)**, **Hardened SQL Analytics**, and an **Indoor Topological Campus Navigation Engine**.

---

## 🌟 Core Modules

### 1. 📄 Agentic Document & Tabular Assistant (`App/`)
- **Autonomous Agent Reasoning Loop**: Uses Groq LLM tool calling to dynamically plan, call tools in sequence, and synthesize answers:
  - `vector_search`: Semantic search over academic regulations (R2021, R2025), admissions, prospectus, and transport schedules via **Qdrant** & **Jina AI Embeddings v3**.
  - `sql_query`: Natural-language-to-SQL generation over structured student, marks, faculty, and attendance datasets in **SQLite** with a 3-layer security lockdown.
  - `find_path`, `list_rooms`, `get_room_info`: Direct integration with indoor campus navigation.
- **Resilience & Production Hardening**:
  - `tenacity` retry with exponential backoff on Groq and Jina APIs (handles 429 rate limits and 5xx errors).
  - 100-chunk batching for Jina embeddings and 50-point batching for Qdrant upserts.
  - Transparent **Source Citations** extracted directly from vector document metadata.
  - Persistent **Multi-Turn Session Memory** stored in SQLite.
- **Interactive UI**: Streamlit web chat with simulated token streaming (`st.write_stream`), debug badges, and dynamic follow-up suggestions.

### 2. 🗺️ Indoor Campus Navigation Engine (`APP/Map/`)
- **Topological Campus Graph**: Multi-floor representation of classrooms, labs, faculty rooms, stairs, corridors, and amenities.
- **Shortest Path Routing**: Dijkstra-powered pathfinding returning step-by-step walking instructions and distance estimates.

---

## 🏗️ System Architecture

```
                                  User Prompt / Chat Query
                                             │
                                             ▼
                                   FastAPI (POST /chat)
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │   fast_path.py     │
                                  │ (Regex Classifier) │
                                  └──────────┬─────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼ (Bare RegNo / Direct Nav)                     ▼ (Complex / General)
           ┌───────────────────┐                           ┌───────────────────┐
           │ Fast-Path Direct  │                           │   Agentic Loop    │
           │ (SQL / Nav Route) │                           │(services/agent.py)│
           └─────────┬─────────┘                           └─────────┬─────────┘
                     │                                               │
                     │               ┌───────────────────────────────┴───────────────────────────────┐
                     │               ▼                               ▼                               ▼
                     │      ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
                     │      │  vector_search  │             │    sql_query    │             │ Navigation Tools│
                     │      │(Qdrant + Jina v3│             │ (SQLite app.db) │             │ (Campus Graph)  │
                     │      └────────┬────────┘             └────────┬────────┘             └────────┬────────┘
                     │               │                               │                               │
                     │               └───────────────────────┬───────┴───────────────────────────────┘
                     │                                       │ (Tool Outputs / Multi-Round)
                     │                                       ▼
                     │                            ┌─────────────────────┐
                     │                            │   Final Synthesis   │
                     │                            │      (Groq LLM)     │
                     │                            └──────────┬──────────┘
                     │                                       │
                     └───────────────────────┬───────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Response with Citations,    │
                              │ Follow-ups & Session Memory │
                              └─────────────────────────────┘
```

---

## 📂 Repository Layout

```
Vir_RAG_assistant/
├── App/                         # Core Agentic RAG Application
│   ├── data/                    # Consolidated databases & local vector fallback
│   │   ├── app.db               # Normalized SQLite database (students, marks, faculty)
│   │   └── sessions.db          # Persistent conversation history per session
│   ├── routes/                  # FastAPI endpoints (/chat, /upload, /suggestions)
│   ├── services/                # Backend intelligence services:
│   │   ├── agent.py             # Agent reasoning loop & multi-round dispatcher
│   │   ├── agent_tools.py       # Tool definitions & execution wrapper with citations
│   │   ├── fast_path.py         # Sub-millisecond regex classifier for instant lookups
│   │   ├── session_store.py     # SQLite session store for multi-turn conversational context
│   │   ├── sql_engine.py        # 3-layer hardened SQL generator & executor
│   │   ├── retriever.py         # Dense vector retrieval & context budget assembly
│   │   ├── embeddings.py        # Jina AI Embeddings v3 with batching & retries
│   │   ├── vectordb.py          # Qdrant client & batched point upserts
│   │   ├── llm.py               # Retried Groq LLM API wrapper
│   │   └── map_tools.py         # Navigation tool wrappers for the agent
│   ├── tests/                   # Test suites
│   │   └── test_agent.py        # Pytest unit tests & accuracy verification suite
│   ├── ingest_sqlite.py         # SQLite ETL builder with automated attendance calculations
│   ├── ingest_pdfs.py           # Batch PDF extractor & vector indexer
│   ├── app.py                   # Streamlit conversational web UI
│   ├── main.py                  # FastAPI REST backend server
│   ├── config.py                # Environment configuration loader
│   ├── requirements.txt         # Python dependencies
│   └── README.md                # Detailed App module documentation
│
├── APP/Map/                     # Indoor Campus Navigation System
│   ├── college_graph.json       # Multi-floor campus topological graph
│   └── navigate.py              # Dijkstra shortest path navigation engine
│
├── DATA/                        # Source campus datasets (Academic Excel sheets & documents)
└── README.md                    # Root project documentation
```

---

## ⚡ Quick Start

### 1. Set Up Environment & Dependencies

```bash
cd App
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`App/.env`)

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
JINA_API_KEY=your_jina_api_key
QDRANT_URL=https://your-cluster.qdrant.io   # Optional: local fallback if omitted
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Ingest Campus Data

```bash
# Ingest Excel sheets into normalized SQLite database:
python ingest_sqlite.py

# Ingest all regulation PDFs & documents into Qdrant vector database:
python ingest_pdfs.py
```

### 4. Run Verification Tests

```bash
# Run unit tests (Fast-path + 3-layer SQL Security):
python -m pytest tests/test_agent.py::TestFastPath tests/test_agent.py::TestSQLSecurity -v

# Run full live conversational accuracy test suite:
python tests/test_agent.py
```

### 5. Start Backend and Web UI

```bash
# Terminal 1 — Start FastAPI Backend:
uvicorn main:app --reload --port 8000

# Terminal 2 — Start Streamlit Web UI:
streamlit run app.py
```

- **Web Chat UI**: `http://localhost:8501`
- **FastAPI Interactive Docs**: `http://127.0.0.1:8000/docs`

---

## 🗺️ CLI Campus Navigation (`APP/Map/`)

You can also run standalone turn-by-turn pathfinding from the terminal:

```bash
cd APP/Map
python navigate.py "G09" "F17"
# Or search by landmark / department:
python navigate.py "Canteen" "ECE - IV Year"
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Agent Reasoning** | Groq API (`openai/gpt-oss-20b` / `llama-3.3-70b-versatile`) | Autonomous tool calling, planning, and natural synthesis |
| **Embeddings** | Jina AI (`jina-embeddings-v3`) | 1024-d dense vector embeddings with batching & retries |
| **Vector Store** | Qdrant Cloud / Local | Scalable cosine similarity vector search |
| **Relational DB** | SQLite (`data/app.db`) | Structured student records, marks, attendance, and analytics views |
| **Session Memory** | SQLite (`data/sessions.db`) | Persistent multi-turn conversation memory per session UUID |
| **Reliability** | `tenacity` | Exponential backoff retry on API rate limits and network errors |
| **REST API** | FastAPI + Uvicorn | High-performance asynchronous backend |
| **Web UI** | Streamlit | Chat interface with simulated token streaming |
| **Pathfinding** | Dijkstra Algorithm | Multi-floor indoor campus shortest path navigation |


---
## 📄 License

Developed for educational and institutional assistance at **P.T. Lee Chengalvaraya Naicker College of Engineering and Technology (PTLCNCET)**.