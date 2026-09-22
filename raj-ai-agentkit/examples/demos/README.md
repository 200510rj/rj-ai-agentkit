# raj-ai-agentkit — Demonstration Suite

Welcome to the beginner-friendly demonstration suite for **raj-ai-agentkit v0.1**!

This suite proves how the four core AI agent reliability primitives (**Guard**, **Router**, **RAG Gate**, and **Judge**) operate together in practice.

> [!NOTE]
> All demos in this folder run **100% locally and offline**. They require **NO API keys, NO internet connection, NO external vector databases, and NO heavy ML frameworks** (such as PyTorch, Laya, or OpenAI). They use `RuleEngine` for fast, zero-dependency decision making.

---

## 📂 Demo Inventory

| File | Component | What It Demonstrates |
|---|---|---|
| `01_guard_demo.py` | **Guard** | Safety verification: allowing safe inputs and blocking prompt injection. |
| `02_router_demo.py` | **Router** | Intent classification: routing requests to `general`, `technical`, or `billing`. |
| `03_rag_gate_demo.py` | **RAG Gate** | Knowledge gating: deciding whether external/private retrieval is needed. |
| `04_judge_demo.py` | **Judge** | Output quality evaluation: scoring answers (`pass`/`fail`/`borderline`) and feedback. |
| `05_full_pipeline_demo.py` | **Full Pipeline** | Complete `Guard → Router → RAG Gate → Execution → Judge` pipeline flow. |
| `06_ollama_demo.py` | **Ollama Local LLM** | Real local LLM decision primitives using Ollama (`OpenAIProvider` + `LLMEngine`). |
| `07_ollama_full_pipeline.py` | **Ollama Full Pipeline** | Real local LLM pipeline (`Guard → Router → RAG Gate → Execution → Judge`). |
| `08 — Real RAG Demo` | **Real RAG Pipeline** | Real RAG architecture (`Guard → Router → RAG Gate → Application Retriever → LLM → Judge`). Located at `examples/rag_demo/real_rag_pipeline.py`. |
| `09 — Embedding RAG` | **Embedding RAG Pipeline** | Semantic embedding RAG pipeline using local dense vector cosine similarity. Located at `examples/rag_demo/embedding_rag_pipeline.py`. |
| `10 — Retriever Comparison` | **Retriever Comparison** | Comparative benchmark of keyword (`SimpleRetriever`) vs semantic (`EmbeddingRetriever`). Located at `examples/rag_demo/compare_retrievers.py`. |
| `11 — Real Tool Execution` | **Real Tool Execution** | `agentkit` decision layer surrounding real Python tool execution (`Guard → Router → RAGGate → Tool Execution → Judge`). Located at `examples/tool_demo/tool_pipeline.py`. |

---

## 📚 Real RAG Pipeline Demo (`08 — Real RAG Demo`)

Located at `examples/rag_demo/real_rag_pipeline.py`.

### Architecture Boundary
```text
User Query
    │
    ▼
  Guard
    │
    ▼
  Router
    │
    ▼
 RAGGate  ───── skip ───────► Direct LLM Generation
    │ retrieve
    ▼
Application Retriever (SimpleRetriever)
    │
    ▼
Retrieved Context (synthetic docs)
    │
    ▼
LLM Generation (Ollama / RuleEngine)
    │
    ▼
  Judge
    │
    ▼
Final Answer
```

* **`RAGGate`**: Evaluates `should_retrieve` (`True`/`False`) decision ONLY. It does **NOT** import the retriever, query a vector database, or load files.
* **Application (`SimpleRetriever`)**: Independent standard-library retriever that loads files from `examples/rag_demo/knowledge/` and matches query terms.
* **LLM**: Grounded generation using retrieved context.
* **`Judge`**: Evaluates final answer quality.

### Execution Command

```bash
python examples/rag_demo/real_rag_pipeline.py
```

---

## 🦙 Real Local Ollama Integration Demos

To run the local LLM integration demos (`06` and `07`):

### Prerequisites
1. Install Ollama from [ollama.com](https://ollama.com).
2. Start the local server: `ollama serve`.
3. Pull your local model: `ollama pull qwen3.5:4b` (or `qwen2.5-coder:7b`, `llama3.2:3b`).

### Execution Commands

```powershell
# Set optional model override (defaults to llama3.2:3b if not set)
$env:AGENTKIT_OLLAMA_MODEL="qwen3.5:4b"

# Run Ollama decision layer demo
python examples\demos\06_ollama_demo.py

# Run Ollama full pipeline demo
python examples\demos\07_ollama_full_pipeline.py
```

---

## 🚀 Setup & Execution

### Prerequisites

Install `raj-ai-agentkit` in editable mode from the project root:

```bash
pip install -e .
```

### Running the Demos

Run any demo directly using Python:

```bash
# Demo 1 — Guard
python examples/demos/01_guard_demo.py

# Demo 2 — Router
python examples/demos/02_router_demo.py

# Demo 3 — RAG Gate
python examples/demos/03_rag_gate_demo.py

# Demo 4 — Judge
python examples/demos/04_judge_demo.py

# Demo 5 — Full Pipeline
python examples/demos/05_full_pipeline_demo.py
```

---

## 🛠️ Real Tool Execution Demo (`11 — Real Tool Execution`)

Located at `examples/tool_demo/tool_pipeline.py`.

### Architecture Boundary

```text
                         User Request
                              │
                              ▼
                           Guard
                              │
                              ▼
                           Router
                              │
                              ▼
                          RAGGate
                              │
                 ┌────────────┴────────────┐
                 │                         │
               SKIP                     RETRIEVE
                 │                         │
                 │                    Knowledge Retriever
                 │                         │
                 │                         ▼
                 │                    Context
                 │                         │
                 └────────────┬────────────┘
                              ▼
                        Tool Selection
                              │
                              ▼
                       Registered Tool
                              │
                              ▼
                          Tool Result
                              │
                              ▼
                            Judge
                              │
                              ▼
                         Final Answer
```

* **`agentkit`**: Decision layer ONLY (`Guard`, `Router`, `RAGGate`, `Judge`).
* **Application**: Execution layer (`tools.py`, `tool_registry.py`, execution loop).

### Execution Command

```bash
python examples/tool_demo/tool_pipeline.py
```

---

## 🛠️ Component Overview

### 1. Guard (`Guard`)
* **Role**: Safety & policy enforcement layer for inputs and outputs.
* **Function**: Checks content against policy instructions (e.g. prompt injection, PII, toxicity).
* **Output**: `GuardResult` with verdict (`"allow"` or `"block"`), confidence score, and a list of specific violations.

### 2. Router (`Router`)
* **Role**: Intent classification and handler selection.
* **Function**: Evaluates user queries against route descriptions and selects the best destination.
* **Output**: `RouteResult` with verdict (`route_name`), confidence score, per-route scores, and optional handler execution via `route_and_execute()`.

### 3. RAG Gate (`RAGGate`)
* **Role**: Decision layer for Retrieval-Augmented Generation.
* **Function**: Determines whether a query requires external/private knowledge retrieval.
* **Output**: `GateResult` with `should_retrieve` (`True`/`False`) and verdict (`"retrieve"` or `"skip"`).
* **Important**: `RAGGate` ONLY makes the decision; it does NOT own or query a vector database itself.

### 4. Judge (`Judge`)
* **Role**: Quality evaluation layer for agent responses.
* **Function**: Scores generated responses against quality criteria (`relevance`, `accuracy`, `clarity`, `completeness`).
* **Output**: `JudgeResult` with score (`0.0–1.0`), verdict (`"pass"`, `"fail"`, or `"borderline"`), and actionable feedback string for retry loops.

---

## 🔄 How the Full Pipeline Works

```
User Query
    │
    ▼
[1] GUARD ────────── block ───────────► [Pipeline Stopped / Rejected]
    │ allow
    ▼
[2] ROUTER ───────────────────────────► Selected Route (e.g. "general", "billing")
    │
    ▼
[3] RAG GATE ──────── skip ───────────► Direct Generation
    │ retrieve
    ▼
[4] EXECUTION ────────────────────────► Retrieve Context + Generate Response (Caller Code)
    │
    ▼
[5] JUDGE ─────────── fail ───────────► [Retry / Regeneration Loop]
    │ pass
    ▼
Final Response Returned to User
```

---

## 🔍 What is Mocked vs. Real

| Aspect | In Demos | In Production |
|---|---|---|
| **DecisionEngine** | `RuleEngine` (keyword & regex matching) | `LLMEngine` (OpenAI, Ollama, Anthropic) or `LayaEngine` (~33ms model) |
| **Retrieval** | Simulated string context | Vector database lookup (ChromaDB, Pinecone, FAISS) |
| **Agent Execution** | Local string formatting | LLM call or tool execution |
| **Guard/Router/Gate/Judge APIs** | **REAL** (`agentkit` public API) | **REAL** (`agentkit` public API) |
