# SpatialData + Self-Reflective RAG for Gemma-4-E2B

## Overview

Three interconnected changes to `c:\Pascal's Folders\QIMR\STAgent\src`:

1. **SpatialData as a RAG knowledge base** — clone the `spatialdata` GitHub repo, index it into a **Chroma vector database** (mirroring how `squidpy_rag.py` already does this for Squidpy), and expose it as a combined or parallel retriever so the self-reflective RAG graph can answer questions about both Squidpy *and* SpatialData.
2. **Self-reflective RAG graph** — redesign `squidpy_rag.py` to follow the diagram: `Retrieve → Grade → [Generate → Answer | Re-write query → Retrieve]`.
3. **Gemma-4-E2B via LM Studio** — use `langchain-openai`'s `ChatOpenAI` pointed at the LM Studio local endpoint (no new packages).

### RAG Graph (from diagram)

```
Question → Retrieve → Grade → ◇ Any doc relevant?
                │                  ├─ Yes → Generate → Answer
                └──────────────────┘
                    Re-write query ←─ No
```

---

## User Review Required

> [!IMPORTANT]
> **LLM used (all nodes)**: `gemma4-e2b` via LM Studio:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gemma4-e2b",            # works; google/gemma4-e2b also accepted
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",           # any non-empty string
    temperature=0,
)
```

> [!NOTE]
> **Fully local pipeline** — both the LLM (`gemma4-e2b`) and the embeddings are served by LM Studio. `OPENAI_API_KEY` is **not required**. An embedding model (e.g. `nomic-embed-text`) must also be loaded in LM Studio alongside the chat model. Both use the same `http://localhost:1234/v1` endpoint.

---

## Open Questions

> ✅ All decisions confirmed:
> - **Chat model**: `gemma4-e2b` via LM Studio (OpenAI-compatible endpoint)
> - **Embedding model**: local embedding model via LM Studio (`nomic-embed-text-v1.5`) — no `OPENAI_API_KEY` needed
> - **Chroma database**: **One merged collection** at `./db/chroma_combined_db`
> - **Pipeline**: fully local, no cloud API calls required

---

## Proposed Changes

### Component 1: Environment & Dependencies

#### [MODIFY] [environment.yml](file:///c:/Pascal's%20Folders/QIMR/STAgent/environment.yml)

Remove the `langchain-ollama` line that was added in a prior draft. Add only what is new:
```yaml
# No new packages needed for LLM (langchain-openai already present)
# No spatialdata runtime package needed (RAG indexes source code, not data files)
```

The existing `environment.yml` already has everything required. The only `.env` change needed is adding the LM Studio settings.

#### [MODIFY] [.env.example](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/.env.example)

Add — **all LM Studio vars, no cloud keys needed**:
```dotenv
# LM Studio local endpoint
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=gemma4-e2b
LM_STUDIO_EMBEDDING_MODEL=nomic-embed-text-v1.5
```

> `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` are **not required** for the RAG pipeline.

---

### Component 2: SpatialData Vector Database

This mirrors the existing Squidpy indexing pattern in `squidpy_rag.py` exactly.

#### [MODIFY] [squidpy_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/squidpy_rag.py)

**New constants** (replacing the old Squidpy-only constants):
```python
SQUIDPY_REPO_PATH      = "./packages_available/squidpy"
SPATIALDATA_REPO_PATH  = "./packages_available/spatialdata"
COMBINED_PERSIST_DIR   = "./db/chroma_combined_db"   # single merged collection
```

The old `PERSIST_DIRECTORY` constant (Squidpy-only) is **removed**.

**New method `setup_combined_index()`** replaces `setup_squidpy_index()`:
1. Clones `https://github.com/scverse/squidpy` if not present (existing logic).
2. Clones `https://github.com/scverse/spatialdata` if not present (new).
3. Loads Python files from **both repos** using the same `GenericLoader` + `RecursiveCharacterTextSplitter` (chunk 2000, overlap 200).
4. Adds a `source_repo` metadata field to each chunk (`"squidpy"` or `"spatialdata"`).
5. Initialises embeddings using **LM Studio's local embedding endpoint** — no `OpenAIEmbeddings`, no API key:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model=os.getenv("LM_STUDIO_EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
    base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
    check_embedding_ctx_length=False,  # required for local models
)
```

6. Creates / loads a **single Chroma collection** at `COMBINED_PERSIST_DIR`.
7. Returns a single MMR retriever (k=8):

```python
return combined_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})
```

> [!IMPORTANT]
> **Two models must be loaded in LM Studio simultaneously** when building the index:
> - Chat model: `gemma4-e2b` (for grading, rewriting, generating)
> - Embedding model: `nomic-embed-text-v1.5` (for vectorising code chunks)
>
> LM Studio supports loading both at once via the "Multi-Model" feature (which you already have enabled).

---

### Component 3: Self-Reflective RAG Graph

#### [MODIFY] [squidpy_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/squidpy_rag.py)

**Updated `SquidpyRAGState`**:
```python
class SquidpyRAGState(TypedDict):
    query: str                           # active query (may be rewritten)
    context: List[Document]              # all retrieved docs
    filtered_context: List[Document]     # docs that passed grade filter
    answer: str                          # generated answer
    chat_history: List[AnyMessage]       # conversation history
    rewrite_attempts: int                # loop guard (max 3)
```

**4 nodes matching the diagram**:

| Node | Diagram colour | Role |
|------|---------------|------|
| `retrieve` | 🔵 Blue | Calls `combined_retriever.invoke(state["query"])` |
| `grade_documents` | 🟢 Green | LLM scores each doc `"yes"/"no"`; keeps yes-docs in `filtered_context` |
| `generate` | 🟣 Purple | Generates Squidpy/SpatialData answer from `filtered_context` |
| `rewrite_query` | 🟢 Green (bottom) | LLM rewrites `query` for better retrieval; increments `rewrite_attempts` |

**Conditional router** after `grade_documents`:
```python
def decide_relevance(state: SquidpyRAGState) -> Literal["generate", "rewrite_query"]:
    if state["filtered_context"] or state["rewrite_attempts"] >= 3:
        return "generate"   # Yes branch (or fallback after max rewrites)
    return "rewrite_query"  # No branch → loop
```

**Graph wiring**:
```python
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite_query", rewrite_query)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    decide_relevance,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
graph_builder.add_edge("rewrite_query", "retrieve")   # loop back
graph_builder.add_edge("generate", END)
```

**Document grader** (structured output via Pydantic):
```python
from pydantic import BaseModel, Field
from typing import Literal

class GradeDoc(BaseModel):
    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the query, 'no' otherwise."
    )

grader_llm = llm.with_structured_output(GradeDoc)
```

**Query rewriter** (simple prompt):
```python
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query rewriter for a Squidpy and SpatialData code RAG system. "
     "Rewrite the user's query to be more specific and retrievable from a Python "
     "codebase index. Output only the rewritten query, nothing else."),
    ("user", "Original query: {query}\n\nRewritten query:"),
])
rewriter = rewrite_prompt | llm
```

---

### Component 4: LM Studio Wiring + Config

#### [MODIFY] [squidpy_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/squidpy_rag.py)

Updated `__init__`:
```python
def __init__(
    self,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
):
    # Fall back to .env values if not explicitly provided
    self.model    = model    or os.getenv("LM_STUDIO_MODEL",    "gemma4-e2b")
    self.base_url = base_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    self.api_key  = api_key  or os.getenv("LM_STUDIO_API_KEY",  "lm-studio")

    self.retriever   = self.setup_combined_index()       # single merged Chroma DB
    self.rag_pipeline = self.create_squidpy_rag_pipeline()
```

#### [MODIFY] [config.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/config.py)

Add to `AppConfig.models`:
```python
"lm_studio": {
    "gemma4-e2b": {
        "temperature": 0,
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    }
}
```

Update `AppConfig` data paths (replace old `db_path`):
```python
combined_db_path: str = str(DB_DIR / "chroma_combined_db")  # replaces chroma_squidpy_db
```

#### [MODIFY] [graph_unified.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/graph_unified.py)

The existing `squidpy_rag_agent` tool import and wiring are **unchanged** — it already exists in the tools list. No new tool is added; the `squidpy_rag_agent` now internally queries the combined knowledge base.

---

## Verification Plan

### Automated Tests
- `pytest` from project root (confirm no regressions).
- Add `tests/test_spatialdata_rag.py` — mock the Chroma stores and verify:
  - `grade_documents` node correctly populates `filtered_context`.
  - `decide_relevance` routes to `rewrite_query` when `filtered_context` is empty.
  - Loop guard fires at `rewrite_attempts >= 3`.

### Manual Verification
1. **Indexing**: Run `SquidpyRAGTool()` init; confirm `./packages_available/spatialdata` is cloned and `./db/chroma_combined_db` is created.
2. **Relevant query**: Ask "How do I compute spatial neighbors in Squidpy?" → verify path is `retrieve → grade (relevant) → generate → END`.
3. **Irrelevant/vague query**: Ask something unrelated → verify `grade → rewrite_query → retrieve` loop triggers, then falls through after 3 attempts.
4. **SpatialData query**: Ask "How do I read a Xenium dataset with SpatialData?" → verify the combined retriever returns SpatialData codebase results.
5. **LM Studio**: Confirm `gemma4-e2b` is loaded and responding at `http://localhost:1234/v1/chat/completions`.

---

## Anaconda Testing Guide

All commands are for **Anaconda Prompt (Windows)**. Run from the project root:
`c:\Pascal's Folders\QIMR\STAgent`

### Step 1 — Create & activate the environment

```bat
:: Create from the existing environment file (first time only — takes ~5-10 min)
conda env create -f environment.yml

:: Activate it
conda activate STAgent
```

> If the env already exists and you just want to update it:
> ```bat
> conda env update -f environment.yml --prune
> ```

---

### Step 2 — Configure the .env file

Copy the example and fill in your keys:
```bat
copy src\.env.example src\.env
```

Then open `src\.env` in any text editor and set:
```dotenv
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=gemma4-e2b
LM_STUDIO_EMBEDDING_MODEL=nomic-embed-text-v1.5
```

> No cloud API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) are required for the RAG pipeline.

---

### Step 3 — Verify LM Studio is reachable

With LM Studio running, **both** `gemma4-e2b` (chat) and an embedding model (`nomic-embed-text-v1.5`) must be loaded. Check both are visible:

```bat
python -c "import urllib.request, json; r=urllib.request.urlopen('http://localhost:1234/v1/models'); data=json.loads(r.read()); [print(m['id']) for m in data['data']]"
```

Expected — you should see **two model IDs**:
```
gemma4-e2b
nomic-embed-text-v1.5
```

> If only one appears, open LM Studio → go to **My Models** → load the embedding model alongside the chat model ("Load in background" option).

---

### Step 4 — Build the combined vector index

This clones both repos and creates `./db/chroma_combined_db`. **Only needed once** (or when you want to refresh the index):

```bat
cd "c:\Pascal's Folders\QIMR\STAgent"
python -c "
import sys; sys.path.insert(0, 'src')
from squidpy_rag import SquidpyRAGTool
print('Building combined Squidpy + SpatialData index...')
tool = SquidpyRAGTool()
print('Done. Index is at ./db/chroma_combined_db')
"
```

> First run takes **10-20 minutes** (cloning + embedding ~3000 chunks).
> Subsequent runs load from disk in seconds.

---

### Step 5 — Test the self-reflective RAG pipeline

#### Quick smoke test (single query):
```bat
python -c "
import sys; sys.path.insert(0, 'src')
from squidpy_rag import SquidpyRAGTool

rag = SquidpyRAGTool()

# Test 1: Squidpy question — should hit grade→relevant→generate
print('=== Test 1: Squidpy query ===')
ans = rag.run('How do I compute spatial neighbors in squidpy?')
print(ans[:500])

# Test 2: SpatialData question — combined DB should return spatialdata docs
print('\n=== Test 2: SpatialData query ===')
ans = rag.run('How do I read a Xenium dataset using spatialdata?')
print(ans[:500])
"
```

#### Test the rewrite loop (vague query that forces a rewrite):
```bat
python -c "
import sys; sys.path.insert(0, 'src')
from squidpy_rag import SquidpyRAGTool

rag = SquidpyRAGTool()
print('=== Test 3: Vague query (should trigger rewrite loop) ===')
ans = rag.run('do the thing with the cells')
print(ans[:300])
"
```

#### Full pipeline test via the existing `squidpy_rag_agent` tool:
```bat
python -c "
import sys; sys.path.insert(0, 'src')
from squidpy_rag import squidpy_rag_agent

# Simulate how graph_unified.py calls the tool
result = squidpy_rag_agent.invoke({
    'state': {'messages': []},
    'query': 'Show me how to run Moran I spatial statistics in squidpy'
})
print(result)
"
```

---

### Step 6 — Run the Streamlit UI (full integration)

```bat
cd "c:\Pascal's Folders\QIMR\STAgent\src"
streamlit run unified_app.py
```

Then in the UI, select any model and type a Squidpy or SpatialData question. The agent will internally call `squidpy_rag_agent`, which now routes through the self-reflective graph.

---

### Step 7 — Run automated tests

```bat
cd "c:\Pascal's Folders\QIMR\STAgent"
pytest -v
```

To run only the new RAG tests (once written):
```bat
pytest tests/test_spatialdata_rag.py -v
```

---

### Troubleshooting Quick Reference

| Problem | Fix |
|---------|-----|
| `ConnectionRefusedError` on port 1234 | LM Studio is not running or the server is not started — click "Start Server" in LM Studio |
| `RuntimeError: OPENAI_API_KEY is not set` | Add `OPENAI_API_KEY` to `src/.env` (needed for embeddings even with local LLM) |
| `ModuleNotFoundError: squidpy_rag` | Run from the project root, not from inside `src/` |
| Index build fails midway | Delete `./db/chroma_combined_db` and `./packages_available/spatialdata` then re-run Step 4 |
| Gemma4 returns empty content | In LM Studio, ensure the model is fully loaded (green dot) before running queries |
