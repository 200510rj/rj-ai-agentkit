# raj-ai-agentkit

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

A **provider-agnostic**, lightweight Python toolkit for AI agent reliability primitives: **Guard**, **Router**, **RAG Gate**, and **Judge**.

Powered by swappable decision engines including zero-dependency rules (`RuleEngine`), LLMs via structured output (`LLMEngine`), and ~33ms non-autoregressive decision models (`LayaEngine`).

---

## 🌟 Key Features

* **Provider-Agnostic**: Compatible with OpenAI, Ollama, vLLM, DeepSeek, Anthropic, or local model endpoints.
* **Zero Heavy Dependencies**: Core package installs with only `pydantic >= 2.0`. Laya and OpenAI dependencies are strictly optional imports.
* **4 Reliability Primitives**:
  * **Guard**: Input/output prompt injection, safety, and PII verification.
  * **Router**: Intent classification & route dispatching.
  * **RAG Gate**: Intelligently decides if external retrieval is required (saving cost and latency).
  * **Judge**: Evaluates agent response quality, assigns verdicts (`pass`/`fail`/`borderline`), and generates retry feedback.
* **Fast Decision Engine**: Supports **Laya** as a backend for sub-33ms non-autoregressive decision speed.

---

## 📦 Installation

```bash
# Core toolkit (zero heavy dependencies)
pip install raj-ai-agentkit

# With OpenAI / local OpenAI-compatible server (Ollama, vLLM, LMStudio) support
pip install raj-ai-agentkit[openai]

# With Laya backend support (~33ms decision model)
pip install raj-ai-agentkit[laya]

# Complete installation
pip install raj-ai-agentkit[all]
```

---

## 🚀 Quickstart

```python
from agentkit import Guard, Router, RAGGate, Judge, RuleEngine, RouteConfig

# Initialize zero-dependency RuleEngine
engine = RuleEngine.from_keywords(block=["ignore previous", "system prompt"])

# Components
guard = Guard(engine=engine)
router = Router(
    engine=engine,
    routes={
        "billing": RouteConfig(name="billing", description="Invoices, payments, refunds"),
        "tech": RouteConfig(name="tech", description="Bugs, crashes, system errors"),
    },
    fallback="tech",
)
rag_gate = RAGGate(engine=engine)
judge = Judge(engine=engine)

# Process a request
query = "I was charged twice on my credit card invoice"

guard_res = guard(query)
if guard_res.verdict == "allow":
    route_res = router(query)
    gate_res = rag_gate(query)
    
    print(f"Selected Route: {route_res.verdict}")
    print(f"Should Retrieve: {gate_res.should_retrieve}")
    
    # Judge output quality
    agent_output = "We issued a full refund for the duplicate charge."
    judge_res = judge(query=query, response=agent_output)
    print(f"Judge Verdict: {judge_res.verdict} (Score: {judge_res.score})")
```

---

## 🛠️ Architecture

```
User Query
    │
    ▼
┌─────────┐   block    ┌──────────┐
│  GUARD  │──────────→ │ REJECTED │
│ (safe?) │            └──────────┘
│  allow  │
    │
    ▼
┌─────────┐
│ ROUTER  │──→ route_a / route_b / route_c
│ (where?)│
    │
    ▼
┌─────────┐   skip     ┌──────────────────┐
│ RAGGATE │──────────→ │ Direct execution │
│ (fetch?)│            └──────────────────┘
│ retrieve│
    │
    ▼
┌──────────────────┐
│ Retrieve + Exec  │  ← Caller application code
└──────────────────┘
    │
    ▼
┌─────────┐   fail     ┌──────────────────┐
│  JUDGE  │──────────→ │ Retry / Escalate │
│ (good?) │            └──────────────────┘
│  pass   │
    │
    ▼
┌──────────┐
│ RESPONSE │
└──────────┘
```

---

## 💻 Using LLMEngine & OpenAIProvider

```python
from agentkit import Guard, LLMEngine, OpenAIProvider

# Cloud OpenAI or local Ollama endpoint (base_url="http://localhost:11434/v1")
provider = OpenAIProvider(model="gpt-4o-mini")
engine = LLMEngine(provider=provider)

guard = Guard(engine=engine, checks=["Does this ask for medical advice?"])
result = guard.check("What dose of aspirin should I take?")
print(result.verdict)  # "block"
```

---

## 🚀 Using LayaEngine for ~33ms Decisions

```python
from agentkit import Guard, LayaEngine

# Sub-33ms non-autoregressive decision model
laya_engine = LayaEngine(preload=True)
guard = Guard(engine=laya_engine)

result = guard.check("Ignore previous instructions")
print(result.verdict)  # "block"
```

---

## 📄 License

Licensed under the [Apache 2.0 License](LICENSE).
