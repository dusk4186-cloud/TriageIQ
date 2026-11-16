# TriageIQ — Jira MCP Server with Defect Triage

[![Streamlit UI (run locally)](https://img.shields.io/badge/Streamlit_UI-(run_locally)-FF4B4B?style=flat&logo=streamlit&logoColor=white&labelColor=24292e)](#3-interactive-streamlit-triage-ui)
[![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white&labelColor=24292e)](#)
[![Atlassian Jira](https://img.shields.io/badge/Jira-Cloud%20REST%20v3-0052CC?style=flat&logo=jira&logoColor=white&labelColor=24292e)](#)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6600?style=flat&logo=chromadb&logoColor=white&labelColor=24292e)](#)
[![Google Gemini](https://img.shields.io/badge/Gemini-3.5%20Flash%20Lite-4285F4?style=flat&logo=google&logoColor=white&labelColor=24292e)](#)
[![Groq Fallback](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat&logo=groq&logoColor=white&labelColor=24292e)](#)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-10B981?style=flat&labelColor=24292e)](#)

[![Tests Passing](https://img.shields.io/badge/Tests-19%2F19%20Passing-22C55E?style=flat&labelColor=24292e)](#4-automated-test-suite)
[![Recall@3](https://img.shields.io/badge/Recall%403-93.3%25-059669?style=flat&labelColor=24292e)](#-empirical-evaluation--benchmarks)
[![MRR](https://img.shields.io/badge/MRR-0.9500-059669?style=flat&labelColor=24292e)](#-empirical-evaluation--benchmarks)
[![Indexed Synthetic Incidents](https://img.shields.io/badge/Indexed-64%20Synthetic%20Incidents-8B5CF6?style=flat&labelColor=24292e)](#-empirical-evaluation--benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-EAB308?style=flat&labelColor=24292e)](#-license--acknowledgments)

> 📹 **Video Demo**: [Watch the 30-Second Walkthrough Video]

https://github.com/user-attachments/assets/8ebdfea1-009b-40df-8dab-fa28bc415175

A Model Context Protocol (MCP) server that connects an LLM to Jira Cloud and a ChromaDB index of past (synthetic) defects. It offers 9 tools: 7 for Jira tickets and 2 for finding similar past defects and suggesting likely root causes.

> ℹ️ **Data Notice**: The 64 indexed incidents are synthetic test data, not real production incidents. Triage output is a suggestion for a developer to check, not a verified diagnosis.

---

## 📌 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [Key Capabilities & Workflow](#-key-capabilities--workflow)
- [Sample Triage Output (Mock Mode)](#-sample-triage-output-mock-mode)
- [The Unified 9-Tool Suite](#-the-unified-9-tool-suite)
- [Empirical Evaluation & Benchmarks](#-empirical-evaluation--benchmarks)
- [Getting Started](#-getting-started)
  - [Quickstart (Try it without a Jira account)](#quickstart-try-it-without-a-jira-account)
  - [Prerequisites](#prerequisites)
  - [Configuration (.env)](#configuration-env)
- [Execution & Usage Modes](#-execution--usage-modes)
  - [1. Natural Language AI Orchestrator](#1-natural-language-ai-orchestrator)
  - [2. MCP Server via Stdio (Claude Desktop / Cursor)](#2-mcp-server-via-stdio-claude-desktop--cursor)
  - [3. Interactive Streamlit Triage UI](#3-interactive-streamlit-triage-ui)
  - [4. Automated Test Suite](#4-automated-test-suite)
- [Limitations & Known Constraints](#-limitations--known-constraints)
- [Project Structure](#-project-structure)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🏗️ Architectural Overview

```
                                ┌───────────────────────────┐
                                │      User (Terminal)      │
                                └─────────────┬─────────────┘
                                              │ Natural Language
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ AI ORCHESTRATION LAYER (orchestrator.py)                                                │
│                                                                                         │
│  • Model Cascade: gemini-3.5-flash-lite -> gemini-flash-lite-latest -> gemini-3.1-flash-lite -> gemini-3.5-flash │
│  • Backup Key: Auto-rotates GEMINI_BACKUP_API_KEY for resilience upon quota exhaustion │
│  • Cross-Provider Fallback: Groq (Llama 3.3 70B Versatile) on 503 / 429 demand spikes   │
│  • Context Preservation: Preserves conversation history across model switches           │
└─────────────────────────────────────────────┬───────────────────────────────────────────┘
                                              │ MCP Protocol (JSON-RPC 2024-11-05 via stdio)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ TRIAGEIQ MCP SERVER (server.py)                                                         │
│                                                                                         │
│   ┌─────────────────────────────────────────┐ ┌───────────────────────────────────────┐ │
│   │ Jira Lifecycle Tools (1 - 7)            │ │ Defect Intelligence Tools (8 - 9)     │ │
│   │ • List Projects, Search Issues (JQL)    │ │ • jira_find_similar_defects           │ │
│   │ • Get / Create / Update / Comment Issue │ │ • jira_triage_defect                  │ │
│   │ • State Transitions & Workflow Engine   │ │ • Semantic Similarity & Suggestions   │ │
│   └────────────────────┬────────────────────┘ └───────────────────┬───────────────────┘ │
└────────────────────────┼──────────────────────────────────────────┼─────────────────────┘
                         │                                          │
                         ▼                                          ▼
      ┌─────────────────────────────────────┐    ┌──────────────────────────────────────┐
      │ Atlassian Jira Cloud REST API v3/v2 │    │ ChromaDB Local Vector Store          │
      │ • /rest/api/3/search/jql            │    │ • 64 Indexed Synthetic Incidents     │
      │ • ADF (Atlassian Document Format)   │    │ • SentenceTransformer all-MiniLM-L6  │
      │ • Basic & Bearer Authentication     │    │ • Metadata: Logs & Known Root Causes │
      │ • Code-Level Read-Only Safety Block │    └──────────────────────────────────────┘
      └─────────────────────────────────────┘
```

---

## ✨ Key Capabilities & Workflow

> **Core Workflow**: *"Gemini turns the top 3 matches into a likely root cause, evidence, fix steps and a Low / Medium / High confidence level; falls back to Groq (Llama 3.3 70B) on simulated 429/503 errors."*

1. **Local Credential Isolation**:
   - Jira credentials stay strictly in the local MCP server process memory and are never sent to the model.
2. **Context Window Optimization**:
   - Jira Cloud API endpoints return large JSON payloads with nested ADF (Atlassian Document Format) trees. TriageIQ extracts clean Markdown strings to prevent context-window bloat.
3. **Defect Memory Bank (ChromaDB Vector Store)**:
   - Indexes 64 synthetic defects with simulated incident logs and known root causes written into the dataset. Embeddings are generated locally using `all-MiniLM-L6-v2`.
4. **4-Part Root Cause Analysis (RCA)**:
   - The `jira_triage_defect` tool pulls the target issue, queries the vector database for the top 3 semantic matches, and synthesizes a structured triage report with:
     - **`### 1. Likely Root Cause`**: Failure hypothesis synthesized across historical defect patterns.
     - **`### 2. Evidence from Similar Historical Defects`**: Referenced ticket keys with contextual summaries of similar failures and past solutions.
     - **`### 3. Recommended Investigation & Fix Steps`**: Suggested debugging steps for engineers to verify.
     - **`### 4. Confidence Level (Low / Medium / High) and Justification`**: Heuristic confidence score generated by the model based on retrieved evidence (uncalibrated).
5. **Multi-Model Resilience & Groq Fallback**:
   - **Model Cascade**: `gemini-3.5-flash-lite` $\rightarrow$ `gemini-flash-lite-latest` $\rightarrow$ `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.5-flash`.
   - **Resilience**: Supports an optional secondary Gemini API key for failover. *(Check your provider’s terms before using a second key)*.
   - **Cross-Provider Groq Fallback**: On 503 service outages or persistent 429 limits, execution shifts to **Groq (`llama-3.3-70b-versatile`)**, preserving conversation history and tool-calling capabilities.
6. **Code-Level Read-Only Protection (`JIRA_READ_ONLY=true`)**:
   - Mutating operations (`create_issue`, `update_issue`, `add_comment`, `transition_issue`) are blocked directly at the client level (`jira_client.py`) when `JIRA_READ_ONLY=true` (enabled by default in code and `.env.example`).
   - Triage results are presented in chat by default (`post_comment_to_jira=False`). Even if comment posting is requested, the client-level guard blocks write execution unless explicitly unlocked.

---

## 🔍 Sample Triage Output (Mock Mode)

Below is an authentic CLI session log running `orchestrator.py` in mock mode *(trimmed excerpt)*, demonstrating natural language user intent, automated tool calling against the local stdio MCP server (`jira_triage_defect`), ChromaDB semantic retrieval, structured 4-part RCA generation, and read-only safety guard enforcement:

```text
You > Triage defect SCRUM-1

[*] [MCP Execution] (gemini-3.5-flash-lite) Calling: jira_triage_defect
    Parameters: {'issue_key': 'SCRUM-1', 'post_comment_to_jira': False}
    [Tool result] Triage completed for SCRUM-1

[AI Assistant (gemini-3.5-flash-lite)]:
Here is the successful AI triage recommendation for SCRUM-1:

### 1. Likely Root Cause
The likely root cause is a misconfiguration or hardcoded constraint within the token generation or validation logic (similar to how SCRUM-16 involved JWT configuration issues and SCRUM-58 involved incorrect JWT expiration settings). Specifically, the token's Time-To-Live (TTL) is likely hardcoded to 15 minutes in the backend token issuance service or configuration file, overriding the intended expiration configuration.

### 2. Evidence from Similar Historical Defects
* SCRUM-58: Highlights issues related to backend JWT token expiration handling and frontend/backend synchronization mismatches.
* SCRUM-45: Demonstrates how session and token management logic can improperly manage session lifecycles across clients.
* SCRUM-16: Illustrates defects originating from JWT signing, configuration oversights, or environment-specific defaults affecting token validity.

### 3. Recommended Investigation & Fix Steps
1. Inspect Token Generation Config: Search the codebase for token generation logic and environment variables where token TTL or expiration is defined (checking for a hardcoded 15m or 900 seconds value).
2. Review Backend Authentication Service: Check configuration files (e.g., application.yml, .env) to verify why the configured duration is not being respected by the mobile client's token issuer.
3. Analyze Token Payload: Decode a newly issued token on the mobile client using JWT.io to verify the exp (expiration) claim timestamp against the issuance time.
4. Implement and Test Fix: Update the token expiration configuration and verify with automated tests covering mobile client token lifecycle scenarios.

### 4. Confidence Level
Medium — While historical defects provide strong context regarding JWT and session management failures, none explicitly match a mobile-specific short expiration bug. However, the symptoms point directly to a backend token generation configuration or TTL mismatch.

---

Would you like me to post this triage recommendation as a comment directly to Jira ticket SCRUM-1?

--------------------------------------------------------------------------

You > Yes, please post this triage recommendation as a comment to SCRUM-1

[*] [MCP Execution] (gemini-3.5-flash-lite) Calling: jira_triage_defect
    Parameters: {'issue_key': 'SCRUM-1', 'post_comment_to_jira': True}
    [Tool result] Write operation blocked: read-only mode

[AI Assistant (gemini-3.5-flash-lite)]:
Write operation blocked: The server is operating in read-only mode (JIRA_READ_ONLY=true).
To permit adding comments or mutating Jira tickets, set JIRA_READ_ONLY=false in your .env file.
```

---

## 🛠️ The Unified 9-Tool Suite

| # | Tool Name | Scope | Description | Key Parameters |
| :-: | :--- | :---: | :--- | :--- |
| **1** | `jira_list_projects` | Jira CRUD | Discovers accessible Jira projects and keys | *None* |
| **2** | `jira_search_issues` | Jira CRUD | Searches tickets via JQL or free-text keywords | `query` *(str)*, `max_results` *(int, default: 10)* |
| **3** | `jira_get_issue` | Jira CRUD | Fetches full issue details, description & recent comments | `issue_key` *(str)*, `include_comments` *(bool, default: True)* |
| **4** | `jira_create_issue` | Jira CRUD | Creates a new Task, Bug, or Story in project (default: `SCRUM`) | `summary` *(str)*, `project_key` *(str)*, `issue_type` *(str)*, `description` *(str)*, `priority` *(str)* |
| **5** | `jira_update_issue` | Jira CRUD | Updates summary, description, priority, assignee, or labels | `issue_key` *(str)*, `summary`, `description`, `priority`, `assignee`, `labels` |
| **6** | `jira_add_comment` | Jira CRUD | Appends a comment or update note to an existing ticket | `issue_key` *(str)*, `comment` *(str)* |
| **7** | `jira_transition_issue` | Jira CRUD | Moves an issue across workflow stages or inspects valid next states | `issue_key` *(str)*, `transition_name` *(str)*, `list_transitions` *(bool)* |
| **8** | `jira_find_similar_defects` | Defect RAG | **Semantic vector search across past (synthetic) defects**; returns similarity scores, incident logs, and known root causes | `query` *(str)*, `top_k` *(int, default: 3)* |
| **9** | `jira_triage_defect` | Defect RAG | **Defect Triaging**: pulls the issue, finds historical evidence, generates RCA suggestions via Gemini, and optionally posts findings to Jira | `issue_key` *(str)*, `post_comment_to_jira` *(bool, default: False)* |

---

## 📊 Empirical Evaluation & Benchmarks

To evaluate the semantic retrieval accuracy of the ChromaDB index (`all-MiniLM-L6-v2`) against varied developer vocabulary, we evaluated 15 test queries using partly rephrased symptom descriptions:

> ⚠️ **Evaluation Context**: These 15 queries were written by the same author who generated the synthetic dataset on a small corpus of 64 tickets, so this score represents an upper bound. Real-world performance on messy production issue trackers with typos, incomplete descriptions, or domain jargon will be lower.

### Benchmark Summary ([benchmark_results.json](benchmark_results.json))
| Metric | Result | Benchmark Description |
| :--- | :---: | :--- |
| **Indexed Synthetic Count** | **64 issues** | 60 synthetic defects with simulated logs + 4 baseline tasks |
| **Top-1 Retrieval (Recall@1)** | **93.3%** (14/15) | The target ticket was the #1 retrieved result |
| **Top-3 Retrieval (Recall@3)** | **93.3%** (14/15) | The target ticket was ranked in the Top-3 |
| **Top-5 Retrieval (Recall@5)** | **100.0%** (15/15) | All 15 target tickets retrieved within the Top-5 |
| **Mean Reciprocal Rank (MRR)** | **0.9500** | Reciprocal rank score across all test cases |
| **Test Suite Pass Rate** | **100%** (19/19) | Unit, read-only security, and stdio MCP protocol tests passing |

### 15-Query Evaluation Breakdown

| # | Ground Truth Defect | Partly Rephrased Query | Retrieved Rank | Status |
| :-: | :--- | :--- | :--- | :-: |
| **1** | `SCRUM-6`: Inventory oversell during flash sale | *"Negative stock numbers on top items; multiple concurrent buyers bought zero-inventory SKUs."* | Rank 4 | Top-5 |
| **2** | `SCRUM-7`: Payment webhook silently failing | *"Stripe shows succeeded, but orders remain in pending state due to webhook verification warnings."* | **Rank 1** | **Top-1** |
| **3** | `SCRUM-8`: Tax calculation API timeout | *"Checkout stalls for 15 seconds then fails with socket timeout calling external tax calculator."* | **Rank 1** | **Top-1** |
| **4** | `SCRUM-9`: Loyalty points not deducting | *"Buyers redeeming reward points at checkout, but point balances are not debited post-order."* | **Rank 1** | **Top-1** |
| **5** | `SCRUM-12`: DB deadlock on order fulfillment | *"Warehouse staff cannot mark orders shipped. Postgres logs show deadlock and ShareLock conflicts."* | **Rank 1** | **Top-1** |
| **6** | `SCRUM-13`: Incorrect currency conversion | *"European shoppers undercharged. EUR/USD exchange rate is 48 hours stale in Redis cache."* | **Rank 1** | **Top-1** |
| **7** | `SCRUM-14`: PDF invoice generation memory leak | *"Node.js process crashes out of memory every 4 hours under peak traffic when rendering invoices."* | **Rank 1** | **Top-1** |
| **8** | `SCRUM-16`: Session hijacking via predictable JWT | *"Pen-test reveals weak JWT signing algorithm, causing concurrent sessions from different IPs."* | **Rank 1** | **Top-1** |
| **9** | `SCRUM-18`: Apple Pay token decryption failure | *"iOS transactions fail at final authorization step with invalid PKCS7 padding error on certificate."* | **Rank 1** | **Top-1** |
| **10** | `SCRUM-19`: Promo code stacking logic error | *"Customers combining mutually exclusive coupons, resulting in negative cart balances."* | **Rank 1** | **Top-1** |
| **11** | `SCRUM-20`: ElasticSearch catalog out of sync | *"Inactive products still appear in search results due to float parsing error during bulk reindex."* | **Rank 1** | **Top-1** |
| **12** | `SCRUM-23`: Duplicate refund processing | *"Double-clicking the refund button in CRM triggers two identical refund transactions."* | **Rank 1** | **Top-1** |
| **13** | `SCRUM-25`: Redis cache stampede | *"DB CPU spikes to 100% on the hour because all featured product cache keys share the same TTL."* | **Rank 1** | **Top-1** |
| **14** | `SCRUM-30`: Gift card double spend race | *"User applied $50 gift card across two simultaneous checkout tabs without balance deduction."* | **Rank 1** | **Top-1** |
| **15** | `SCRUM-48`: Refunds to expired cards fail | *"Refunds to expired credit cards fail silently on Stripe and remain stuck in processing."* | **Rank 1** | **Top-1** |

To reproduce the benchmark:
```powershell
python run_benchmark.py
```

---

## 🚀 Getting Started

### Quickstart (Try it without a Jira account)

You can clone and test the entire defect triage engine without a Jira account (requires Internet access for Gemini LLM responses):

```bash
# 1. Clone the repository
git clone https://github.com/dusk4186-cloud/TriageIQ.git
cd TriageIQ

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build or verify the local ChromaDB index
python "Defect Triage/jira-defect-embedding.py"

# 5. Configure .env (ships with JIRA_MOCK_MODE=true and JIRA_READ_ONLY=true)
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env

# 6. Add your GEMINI_API_KEY to .env and launch
python orchestrator.py
```

### Prerequisites

- **Python 3.11+** installed
- A **Google Gemini API Key** (and optional Groq API Key for fallback)
- *(Optional)* An active **Atlassian Jira Cloud** account (only if testing against live tickets)

### Configuration (`.env`)

Create a `.env` file in the project root:

```env
# ==============================================================================
# Atlassian Jira Configuration
# ==============================================================================
# Set to 'true' to run without a Jira account using in-memory simulated tickets
JIRA_MOCK_MODE=true

# Safety Switch: Set to 'true' (default) to block mutating tools (create, update, comment, transition)
# Set to 'false' only when you want to permit write operations to Jira
JIRA_READ_ONLY=true

JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your_atlassian_api_token
JIRA_PROJECT_KEY=SCRUM

# ==============================================================================
# Google Gemini & Multi-Model Cascade
# ==============================================================================
GEMINI_API_KEY=your_primary_gemini_api_key
# Optional secondary Gemini API key for failover (check provider terms)
# GEMINI_BACKUP_API_KEY=your_backup_gemini_api_key
GEMINI_MODELS=gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-3.1-flash-lite,gemini-3.5-flash

# ==============================================================================
# Groq Cross-Provider Fallback (Optional)
# ==============================================================================
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## 💻 Execution & Usage Modes

### 1. Natural Language AI Orchestrator

Run the multi-model agent that connects to the MCP server and talks to you in natural language:

```powershell
python orchestrator.py
```

**Example queries you can type:**
- *"Node.js process crashes out of memory every 4 hours under peak traffic when rendering invoices"*
- *"Triage defect SCRUM-1 and suggest investigation steps."*
- *"Search for open bugs in project SCRUM."*
- *"Create a high priority bug: Inventory overselling during flash sale."*
- *"Move SCRUM-1 to Done."*

> ⚠️ **Note**: Commands that create, update, or transition tickets require `JIRA_READ_ONLY=false` in `.env`. Setting this is safe in mock mode.

### 2. MCP Server via Stdio (Claude Desktop / Cursor)

To plug TriageIQ into **Claude Desktop**, **Cursor IDE**, or any MCP-compatible host, add the following to your configuration file (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "triageiq": {
      "command": "python",
      "args": [
        "path/to/server.py"
      ],
      "env": {
        "JIRA_URL": "https://your-domain.atlassian.net",
        "JIRA_EMAIL": "your-email@example.com",
        "JIRA_API_TOKEN": "your_atlassian_api_token",
        "JIRA_PROJECT_KEY": "SCRUM",
        "GEMINI_API_KEY": "your_gemini_api_key",
        "JIRA_READ_ONLY": "true"
      }
    }
  }
}
```

Or run standalone:
```powershell
.\run_server.bat
```

### 3. Interactive Streamlit Triage UI

A web UI is included for developers who prefer an interactive visual dashboard:

```powershell
streamlit run "Defect Triage/Defect_Triaging.py"
```

To explore the ChromaDB defect embeddings collection visually:
```powershell
streamlit run chroma_dashboard.py
```

### 4. Automated Test Suite

Run the full end-to-end unit, read-only security, and MCP protocol test suite:

```powershell
python test_jira_server.py
```
*(Runs 19 comprehensive tests: unit functions, ADF parsing, read-only security enforcement, ChromaDB semantic queries, Gemini triage generation, and stdio JSON-RPC handshake).*

---

## ⚠️ Limitations & Known Constraints

The current version includes the following architectural constraints and security considerations:

1. **Indirect Prompt Injection Risk**:
   - The model reads ticket summaries and descriptions submitted by external users. While `JIRA_READ_ONLY=true` blocks unauthorized write operations at the code level, an adversarial ticket description can still manipulate the LLM's reasoning, causing it to return deceptive or misleading root-cause explanations and fix suggestions.
2. **Small Synthetic Benchmark**:
   - The 93.3% Recall@3 evaluation was measured across 15 partly rephrased queries evaluated against 64 synthetic defects generated by the same author.
   - Real-world production data with domain abbreviations, typos, and fragmented descriptions will achieve lower recall and will require continuous re-indexing and domain tuning.
3. **English-Only Embeddings**:
   - The underlying embedding model (`all-MiniLM-L6-v2`) is evaluated and optimized for English text. Multilingual tickets (e.g., Spanish, German, Japanese) require multilingual sentence models (e.g., `paraphrase-multilingual-MiniLM-L12-v2`).
4. **Prompt-Guided Confirmation (Chat Layer)**:
   - The confirmation step for posting comments or creating tickets relies on the LLM adhering to system instructions and the default parameter `post_comment_to_jira=False`.
   - In live mode without `JIRA_READ_ONLY=true`, there is no interactive blocking terminal prompt (`input("Confirm? Y/N")`) in the orchestrator pipeline.
5. **No Automated Severity / Priority ML Classifier**:
   - The engine performs Root Cause Analysis (RCA), evidence retrieval, and investigation recommendations.
   - It does **not** classify ticket severity (Critical / Major / Minor) or dynamically assign priority based on an SLA matrix; ticket priority defaults to `"Medium"` or follows the user's explicit parameter.
6. **Single-Project Default Scope**:
   - The server is optimized with a default project key (`SCRUM`). While multi-project operations are supported by specifying `project_key` in individual tool calls, workspace discovery is optimized for the primary project.
7. **Local Vector Store Persistence**:
   - ChromaDB runs as an embedded SQLite database (`Defect Triage/chroma_db`). In a distributed multi-developer environment, vector store updates require centralized syncing or migration to a hosted cloud vector database (e.g. Pinecone, Qdrant, or PGVector).
8. **Jira Cloud ADF (Atlassian Document Format) Constraints**:
   - Jira Cloud v3 enforces Atlassian Document Format (ADF) JSON trees for rich text. TriageIQ handles bidirectional markdown $\leftrightarrow$ ADF conversion, but complex custom Jira tables or proprietary layout macros are flattened to standard markdown strings.
9. **SentenceTransformer Cold-Start Latency**:
   - On initial launch, loading the `all-MiniLM-L6-v2` embedding weights into local memory takes ~1.5 seconds. Once loaded, similarity query latency drops to ~150ms (measured locally on an Intel Core i5 10th Gen, Windows 10 machine with CPU inference).

---

## 📁 Project Structure

```
TriageIQ/
├── assets/
│   └── TriageIQ_demo.mp4       # 30-second end-to-end video walkthrough
├── server.py                   # Unified Smart Jira MCP Server (9 tools via stdio)
├── jira_client.py              # Atlassian REST API client with code-level read-only guard
├── defect_triage_engine.py     # ChromaDB vector retrieval + Gemini RCA triage generator
├── orchestrator.py             # Multi-model cascading AI orchestrator (Gemini + Groq)
├── mcp_client.py               # Standalone MCP Client CLI & automated demo runner
├── test_jira_server.py         # Automated test suite (19/19 unit, security & stdio tests)
├── run_benchmark.py            # Reproducible Recall@3 retrieval evaluation script
├── benchmark_results.json      # Benchmark results (93.3% Recall@3 across 15 rephrased queries)
├── chroma_dashboard.py         # Streamlit visual dashboard for ChromaDB collection
├── push_defects_to_jira.py     # Batch defect seeding utility for Jira Cloud
├── requirements.txt            # System dependencies (mcp, chromadb, google-genai, etc.)
├── run_server.bat              # Windows batch launcher
├── run_server.ps1              # PowerShell launcher
├── .env.example                # Sanitized configuration template with safe defaults
├── README.md                   # Complete system documentation
└── Defect Triage/
    ├── Defect_Triaging.py      # Streamlit Defect Triage web application
    ├── jira-defect-embedding.py# Embedding pipeline for synthetic Jira defects
    ├── jira_defects.json       # Canonical dataset of 64 synthetic defects
    └── chroma_db/              # Persistent ChromaDB vector index
```

---

## 📄 License & Acknowledgments

- This project is licensed under the **MIT License** — feel free to use, modify, and distribute for personal or commercial projects.
- *Built with AI assistance.*
